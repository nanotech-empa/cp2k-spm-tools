
import os
import numpy as np
import scipy
import scipy.io
import time
import copy
import sys

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

### ---------------------------------------------------------------------------
### General CP2K routines
### ---------------------------------------------------------------------------

# Reads from the cp2k input file:
# * Basis set names for all elements
# * Cell size (bohr radii)
def read_cp2k_input(cp2k_input_file):
    elem_basis_name = {}
    cell = np.zeros(3)
    with open(cp2k_input_file) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            parts = lines[i].split()
            if len(parts) == 0:
                continue
            # Have we found the basis set info?
            if parts[0] == "&KIND":
                elem = parts[1]
                for j in range(10):
                    parts = lines[i+j].split()
                    if parts[0] == "BASIS_SET":
                        basis = parts[1]
                        elem_basis_name[elem] = basis
                        break
            # Have we found the CELL info?
            if parts[0] == "ABC":
                if parts[1] == "[angstrom]":
                    cell[0] = float(parts[2])
                    cell[1] = float(parts[3])
                    cell[2] = float(parts[4])
                else:
                    cell[0] = float(parts[1])
                    cell[1] = float(parts[2])
                    cell[2] = float(parts[3])

    cell *= ang_2_bohr

    return elem_basis_name, cell

# Read atomic positions from .xyz file (in Bohr radiuses)
def read_atoms(file_xyz):
    # Read atomic positions (in angstrom)
    data = np.genfromtxt(file_xyz, dtype=None, skip_header=2)
    data = np.atleast_1d(data)
    elems_nrs = []
    positions = []
    elem_nr = {'H': 1,
               'C': 6,
               'N': 7,
               'O': 8,
               'Na': 11,
               'Br': 35,
               'Au': 79}
    for line in data:
        elem = line[0].decode("utf-8")
        if len(line) > 4:
            nr = line[4]
        elif elem in elem_nr:
            nr = elem_nr[elem]
        else:
            print("Warning: element not recognized!")
            nr = 0
        elems_nrs.append([elem, nr])
        positions.append(np.array([line[1], line[2], line[3]]) * ang_2_bohr)
    return np.array(positions), elems_nrs

# Modify atomic positions such that atoms are centered to the cell
def center_atoms_to_cell(at_positions, cell):
    minx = np.min(at_positions[:, 0])
    maxx = np.max(at_positions[:, 0])
    miny = np.min(at_positions[:, 1])
    maxy = np.max(at_positions[:, 1])
    minz = np.min(at_positions[:, 2])
    maxz = np.max(at_positions[:, 2])

    at_positions[:, 0] += cell[0]/2-(maxx+minx)/2
    at_positions[:, 1] += cell[1]/2-(maxy+miny)/2
    at_positions[:, 2] += cell[2]/2-(maxz+minz)/2

# NB: The fermi energy will be in eV !
def read_fermi_from_cp2k_out(cp2k_out_file):
    fermi = None
    with open(cp2k_out_file) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i].strip()
            parts = line.split()
            if len(parts) == 0:
                continue
            # Have we found the Fermi energy?
            if line.startswith("Fermi Energy [eV]"):
                fermi = float(parts[-1])
            elif line.startswith("Fermi energy:"):
                fermi = float(parts[-1])*hart_2_ev
    if fermi == None:
        print("Warning: Fermi level was not found in the cp2k output.")
    return fermi

### ---------------------------------------------------------------------------
### Basis set routines
### ---------------------------------------------------------------------------

def magic_basis_normalization(basis_sets_):
    basis_sets = copy.deepcopy(basis_sets_)
    for elem, bs in basis_sets.items():
        for shell in bs:
            l = shell[0]
            exps = shell[1]
            coefs = shell[2]
            nexps = len(exps)

            norm_factor = 0
            for i in range(nexps-1):
                for j in range(i+1, nexps):
                    norm_factor += 2*coefs[i]*coefs[j]*(2*np.sqrt(exps[i]*exps[j])/(exps[i]+exps[j]))**((2*l+3)/2)

            for i in range(nexps):
                norm_factor += coefs[i]**2

            for i in range(nexps):
                coefs[i] = coefs[i]*exps[i]**((2*l+3)/4)/np.sqrt(norm_factor)

    return basis_sets

def read_basis_functions(basis_set_file, elem_basis_name):
    basis_sets = {}
    with open(basis_set_file) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            parts = lines[i].split()
            if parts[0] in elem_basis_name:
                elem = parts[0]
                if parts[1] == elem_basis_name[elem] or parts[2] == elem_basis_name[elem]:
                    # We have found the correct basis set
                    basis_functions = []
                    nsets = int(lines[i+1])
                    cursor = 2
                    for j in range(nsets):
                        comp = [int(x) for x in lines[i+cursor].split()]
                        n_princ, l_min, l_max, n_exp = comp[:4]
                        l_arr = np.arange(l_min, l_max+1, 1)
                        n_basisf_for_l = comp[4:]
                        assert len(l_arr) == len(n_basisf_for_l)

                        exps = []
                        coeffs = []

                        for k in range(n_exp):
                            exp_c = [float(x) for x in lines[i+cursor+k+1].split()]
                            exps.append(exp_c[0])
                            coeffs.append(exp_c[1:])

                        exps = np.array(exps)
                        coeffs = np.array(coeffs)

                        indx = 0
                        for l, nl in zip(l_arr, n_basisf_for_l):
                            for i in range(nl):
                                basis_functions.append([l, exps, coeffs[:, indx]])
                                indx += 1
                        cursor += n_exp + 1

                    basis_sets[elem] = basis_functions

    return magic_basis_normalization(basis_sets)

### ---------------------------------------------------------------------------
### RESTART file loading and processing
### ---------------------------------------------------------------------------

# Coefficients for molecular orbitals that match the energy range
# NB: If Fermi energy is passed as "None", it will be set to HOMO energy
# RETURNS: morb_composition[imo][iatom][iset][ishell][iorb (m)]
# AND corresponding eigenvalues (eV) and occupancies for the morbitals
# Energies in ev
def load_restart_wfn_file(restart_file, emin, emax, fermi):

    inpf = scipy.io.FortranFile(restart_file, 'r')

    natom, nspin, nao, nset_max, nshell_max = inpf.read_ints()
    # natom - number of atoms
    # nspin - number of spins
    # nao - number of atomic orbitals
    # nset_max - maximum number of sets in the basis set
    #           (e.g. if one atom's basis set contains 3 sets and every other
    #           atom's contains 1, then this value will still be 3)
    # nshell_max - maximum number of shells in each set

    if nspin > 1:
        print("Spin-polarized input is not supported.")
        return

    # number of sets in the basis set for each atom
    nset_info = inpf.read_ints()

    # number of shells in each of the sets
    nshell_info = inpf.read_ints()

    # number of orbitals in each shell
    nso_info = inpf.read_ints()

    morb_composition = []

    for ispin in range(nspin):
        nmo, homo, lfomo, nelectron = inpf.read_ints()
        # nmo - number of molecular orbitals
        # homo - index of the HOMO
        # lfomo - ???
        # nelectron - number of electrons

        # list containing all eigenvalues and occupancies of the molecular orbitals
        evals_occs = inpf.read_reals()

        evals = evals_occs[:int(len(evals_occs)/2)]
        occs = evals_occs[int(len(evals_occs)/2):]

        # convert evals from hartree to eV
        evals *= hart_2_ev

        if fermi == None:
            fermi = evals[homo-1]

        evals -= fermi

        first_imo = -1

        for imo in range(nmo):
            coefs = inpf.read_reals()
            if evals[imo] < emin:
                continue
            if evals[imo] > emax:
                break
            if first_imo == -1:
                print("First molecular index in energy range: ", imo)
                first_imo = imo
            current_morb_comp = []

            shell_offset = 0
            norb_offset = 0
            orb_offset = 0

            for iatom in range(natom):
                nset = nset_info[iatom]
                current_morb_comp.append([]) # atom index
                for iset in range(nset):
                    nshell = nshell_info[shell_offset]
                    shell_offset += 1
                    current_morb_comp[-1].append([]) # set index
                    ishell = 0
                    while ishell < nshell:
                        norb = nso_info[norb_offset]
                        norb_offset += 1
                        if norb == 0:
                            continue
                        ishell += 1
                        current_morb_comp[-1][-1].append([]) # shell index (l)
                        for iorb in range(norb):
                            current_morb_comp[-1][-1][-1].append(coefs[orb_offset]) # orb index (m)
                            orb_offset += 1
            morb_composition.append(current_morb_comp)

    inpf.close()

    coef_arr = np.empty(len(morb_composition))
    morb_composition_rev = []

    for iatom in range(len(morb_composition[0])):
        morb_composition_rev.append([])
        for iset in range(len(morb_composition[0][iatom])):
            morb_composition_rev[-1].append([])
            for ishell in range(len(morb_composition[0][iatom][iset])):
                morb_composition_rev[-1][-1].append([])
                for iorb in range(len(morb_composition[0][iatom][iset][ishell])):
                    for imo in range(len(morb_composition)):
                        coef_arr[imo] = morb_composition[imo][iatom][iset][ishell][iorb]
                    morb_composition_rev[-1][-1][-1].append(np.copy(coef_arr))


    ref_energy = fermi
    n_sel_morbs = len(morb_composition)

    morb_energies = evals[first_imo:first_imo+n_sel_morbs]
    morb_occs = occs[first_imo:first_imo+n_sel_morbs]

    i_homo = 0
    for i, en in enumerate(morb_energies):
        if en > 0.0:
            i_homo = i - 1
            break
        if np.abs(en) < 1e-6:
            i_homo = i

    return morb_composition_rev, morb_energies, morb_occs, ref_energy, i_homo

### ---------------------------------------------------------------------------
### MOLOG FILE loading and processing
### ---------------------------------------------------------------------------

# Inspiration from
# https://github.com/ondrejkrejci/PPSTM/blob/master/pyPPSTM/ReadSTM.py#L520
def read_cp2k_MO_file(molog_file):
    print("Reading CP2K MOs from:"+molog_file)

    lines = []

    with open(molog_file) as f:
        skip = True
        for l in f:
            if "MO EIGENVALUES" in l and "SCF STEP" not in l:
                skip = False
                continue
            if not skip:
                l = l.strip()
                if(len(l)==0): continue
                lines.append(l)

    if (lines[-1].startswith("HOMO-LUMO gap:") and
        lines[-2].startswith("Fermi energy:")):
        print("Band gap system")
        fermi_energy = hart_2_ev * float(lines[-2].split()[-1])
        del lines[-1]
        del lines[-1]
    elif lines[-1].startswith("Fermi energy:"):
        print("Metallic system")
        fermi_energy = hart_2_ev * float(lines[-1].split()[-1])
        del lines[-1]
    else:
        print("End is not formatted correctly, exiting.")
        return None

    # detect dimensions
    parts = lines[-1].split()
    nbasis = int(parts[0])
    natoms = int(parts[1])

    first_nmo = int(lines[0].split()[0])
    last_nmo = int(lines[-nbasis-3].split()[-1])
    nmos = last_nmo-first_nmo + 1

    nlines_per_block = nbasis + 3
    nblocks = int(len(lines)/nlines_per_block)

    print("Found %d MOs spanned by %d basis functions centered on %d atoms."%(nmos, nbasis, natoms))

    # unfold table
    idx = []
    evals = []
    occs = []
    evecs = [list() for i in range(nbasis)]
    labels = [l.split()[:4] for l in lines[3:nbasis+3]]

    for iblock in range(nblocks):
        #a = iblock*nlines_per_block
        a = 0
        idx.extend([int(x) for x in lines[a].split()])
        evals.extend([float(x) for x in lines[a+1].split()])
        occs.extend([float(x) for x in lines[a+2].split()])
        for j in range(nbasis):
            parts = lines[a+3+j].split()
            assert parts[:4] == labels[j]
            evecs[j].extend([float(x) for x in parts[4:]])
        # Release memory from the processed block
        del lines[:nlines_per_block]

    # convert to numpy arrays
    idx = np.array(idx)
    evals = np.array(evals)
    occs = np.array(occs)
    evecs = np.array(evecs)

    assert evals.shape == (nmos,)
    assert occs.shape == (nmos,)
    assert evecs.shape == (nbasis, nmos)

    # convert hartree to eV
    evals = hart_2_ev * evals
    # NB: evecs[basis_idx, morbital_idx] = overlap/projection/scalar_product

    #### --------------------------------------------------------------------
    #### Further processing into format
    #### molog_data[atom_nr] = ['H', ['3s', '3px', ...], [[evec for 3s], [evec for 3px], ...]
    molog_data = [[] for i in range(natoms)]

    for label, evec in zip(labels, evecs):
        atom_nr = int(label[1]) - 1
        elem = label[2]
        if len(molog_data[atom_nr]) == 0:
            molog_data[atom_nr].extend([elem, [], []])
        molog_data[atom_nr][1].append(label[3])
        molog_data[atom_nr][2].append(evec)

    return molog_data, evals, occs, fermi_energy

# Assuming alphabetical order: x, y, z
# returns in order of increasing m
def cart_coef_to_spherical(l, coefs):
    if l == 0:
        assert len(coefs) == 1
        return np.array(coefs)
    elif l == 1:
        assert len(coefs) == 3
        return np.array([coefs[1], coefs[2], coefs[0]])
    elif l == 2:
        assert len(coefs) == 6
        conv_mat = np.array([[ 0.0, 1.0, 0.0,  0.0, 0.0, 0.0],
                             [ 0.0, 0.0, 0.0,  0.0, 1.0, 0.0],
                             [-0.5, 0.0, 0.0, -0.5, 0.0, 1.0],
                             [ 0.0, 0.0, 1.0,  0.0, 0.0, 0.0],
                             [ 0.5*np.sqrt(3), 0, 0, -0.5*np.sqrt(3), 0, 0]])
        return np.dot(conv_mat, coefs)
    else:
        print("Not implemented.")
        return 0.0

# morb_composition[morb_nr, atom_nr] = [coefs_for_2s, coefs_for_3s, coefs_for_3p, coefs_for_3d, ...]
# coefs_for_3p = [coef_for_m=-1, coef_for_m=0, coef_for_m=1]
def read_and_process_molog(molog_file):

    molog_data, evals, occs, fermi_energy = read_cp2k_MO_file(molog_file)

    natoms = len(molog_data)
    nmos = len(evals)

    morb_composition = [[[] for j in range(natoms)] for i in range(nmos)]

    for i_at in range(natoms):
        elem = molog_data[i_at][0]
        orb_labels = molog_data[i_at][1]
        eig_vecs = molog_data[i_at][2]

        i_orb = 0
        while i_orb < len(orb_labels):

            n_orb = int(orb_labels[i_orb][0])
            cart_orb = orb_labels[i_orb][1:]

            if cart_orb == 's':
                eig_vec = eig_vecs[i_orb]
                for i_mo in range(nmos):
                    morb_composition[i_mo][i_at].append(cart_coef_to_spherical(0, [eig_vec[i_mo]]))
                i_orb += 1
                continue

            elif cart_orb == 'px':
                eig_px = eig_vecs[i_orb]
                eig_py = eig_vecs[i_orb+1]
                eig_pz = eig_vecs[i_orb+2]
                for i_mo in range(nmos):
                    spherical_coefs = cart_coef_to_spherical(1, [eig_px[i_mo], eig_py[i_mo], eig_pz[i_mo]])
                    morb_composition[i_mo][i_at].append(spherical_coefs)
                i_orb += 3
                continue

            elif cart_orb == 'dx2':
                eig_dx2 = eig_vecs[i_orb]
                eig_dxy = eig_vecs[i_orb+1]
                eig_dxz = eig_vecs[i_orb+2]
                eig_dy2 = eig_vecs[i_orb+3]
                eig_dyz = eig_vecs[i_orb+4]
                eig_dz2 = eig_vecs[i_orb+5]
                for i_mo in range(nmos):
                    spherical_coefs = cart_coef_to_spherical(2, [eig_dx2[i_mo], eig_dxy[i_mo], eig_dxz[i_mo],
                                                                 eig_dy2[i_mo], eig_dyz[i_mo], eig_dz2[i_mo]])
                    morb_composition[i_mo][i_at].append(spherical_coefs)
                i_orb += 6
                continue

            else:
                print('Error: found unsupported orbital label')
                break

    morb_energies = evals - fermi_energy

    return morb_composition, morb_energies

### ---------------------------------------------------------------------------
### Methods more directly related to putting stuff on grids
### ---------------------------------------------------------------------------


# Evaluates the spherical harmonics (times r^l) with some unknown normalization
# (source: Carlo's Fortran code)
def spherical_harmonic_grid(l, m, x_grid, y_grid, z_grid):
    c = (2.0/np.pi)**(3.0/4.0)

    # s orbitals
    if (l, m) == (0, 0):
        return c

    # p orbitals
    elif (l, m) == (1, -1):
        return c*2.0*y_grid
    elif (l, m) == (1, 0):
        return c*2.0*z_grid
    elif (l, m) == (1, 1):
        return c*2.0*x_grid

    # d orbitals
    elif (l, m) == (2, -2):
        return c*4.0*x_grid*y_grid
    elif (l, m) == (2, -1):
        return c*4.0*y_grid*z_grid
    elif (l, m) == (2, 0):
        return c*2.0/np.sqrt(3)*(2*z_grid**2-x_grid**2-y_grid**2)
    elif (l, m) == (2, 1):
        return c*4.0*z_grid*x_grid
    elif (l, m) == (2, 2):
        return c*2.0*(x_grid**2-y_grid**2)

    # f orbitals
    elif (l, m) == (3, -3):
        return c*np.sqrt(8/3)*y_grid*(3*x_grid**2-y_grid**2)
    elif (l, m) == (3, -2):
        return c*8.0*x_grid*y_grid*z_grid
    elif (l, m) == (3, -1):
        return c*np.sqrt(8/5)*y_grid*(4*z_grid**2-x_grid**2-y_grid**2)
    elif (l, m) == (3, 0):
        return c*4.0/np.sqrt(15.0)*z_grid*(2.0*z_grid**2-3.0*x_grid**2-3.0*y_grid**2)
    elif (l, m) == (3, 1):
        return c*np.sqrt(8/5)*x_grid*(4*z_grid**2-x_grid**2-y_grid**2)
    elif (l, m) == (3, 2):
        return c*4.0*z_grid*(x_grid**2-y_grid**2)
    elif (l, m) == (3, 3):
        return c*np.sqrt(8/3)*x_grid*(x_grid**2-3.0*y_grid**2)

    print("No spherical harmonic found for l=%d, m=%d" % (l, m))
    return 0

# Adds local 3D box to a global grid by wrapping the boundaries in X and Y
# But not in Z
def add_local_to_global_box(loc_grid, glob_grid, origin_diff):
    loc_n = np.shape(loc_grid)[0:2]
    glob_n = np.shape(glob_grid)[0:2]
    od = origin_diff

    # Move the origin_diff vector to the main global cell (not an image)
    od = od % glob_n

    inds = []
    l_inds = []

    for i in range(len(glob_n)):
        ixs = [[od[i], od[i] + loc_n[i]]]
        l_ixs = [0]
        while ixs[-1][1] > glob_n[i]:
            overshoot = ixs[-1][1]-glob_n[i]
            ixs[-1][1] = glob_n[i]
            l_ixs.append(l_ixs[-1]+glob_n[i]-ixs[-1][0])
            ixs.append([0, overshoot])
        l_ixs.append(loc_n[i])

        inds.append(ixs)
        l_inds.append(l_ixs)

    l_ixs = l_inds[0]
    l_iys = l_inds[1]
    for i, ix in enumerate(inds[0]):
        for j, iy in enumerate(inds[1]):
            glob_grid[ix[0]:ix[1], iy[0]:iy[1], :] += loc_grid[l_ixs[i]:l_ixs[i+1], l_iys[j]:l_iys[j+1], :]

# Adds local 2D or 3D grid to a global grid by wrapping the extending boundaries
def add_local_to_global_grid(loc_grid, glob_grid, origin_diff):
    loc_n = np.shape(loc_grid)
    glob_n = np.shape(glob_grid)
    od = origin_diff

    # Move the origin_diff vector to the main global cell (not an image)
    od = od % glob_n

    inds = []
    l_inds = []

    for i in range(len(glob_n)):
        ixs = [[od[i], od[i] + loc_n[i]]]
        l_ixs = [0]
        while ixs[-1][1] > glob_n[i]:
            overshoot = ixs[-1][1]-glob_n[i]
            ixs[-1][1] = glob_n[i]
            l_ixs.append(l_ixs[-1]+glob_n[i]-ixs[-1][0])
            ixs.append([0, overshoot])
        l_ixs.append(loc_n[i])

        inds.append(ixs)
        l_inds.append(l_ixs)

    if len(inds) == 2:
        l_ixs = l_inds[0]
        l_iys = l_inds[1]
        for i, ix in enumerate(inds[0]):
            for j, iy in enumerate(inds[1]):
                glob_grid[ix[0]:ix[1], iy[0]:iy[1]] += loc_grid[l_ixs[i]:l_ixs[i+1], l_iys[j]:l_iys[j+1]]
    elif len(inds) == 3:
        l_ixs = l_inds[0]
        l_iys = l_inds[1]
        l_izs = l_inds[2]
        for i, ix in enumerate(inds[0]):
            for j, iy in enumerate(inds[1]):
                for k, iz in enumerate(inds[2]):
                    glob_grid[ix[0]:ix[1], iy[0]:iy[1], iz[0]:iz[1]] += \
                        loc_grid[l_ixs[i]:l_ixs[i+1], l_iys[j]:l_iys[j+1], l_izs[k]:l_izs[k+1]]


# Puts the molecular orbitals onto a plane.
# All inputs are needed to be in [a.u.], except for pbc_box (angstrom)
def calc_morb_planes(plane_size, plane_size_n, plane_z,
                     at_positions, at_elems,
                     basis_sets, morb_composition, pbc_box = 12.0):

    time1 = time.time()

    dv = plane_size/plane_size_n
    x_arr = np.arange(0, plane_size[0], dv[0])
    y_arr = np.arange(0, plane_size[1], dv[1])
    x_grid, y_grid = np.meshgrid(x_arr, y_arr, indexing='ij')

    # Define small grid for orbital evaluation
    # and convenient PBC implementation
    loc_cell = np.array([pbc_box,  pbc_box])*ang_2_bohr
    x_arr_loc = np.arange(0, loc_cell[0], dv[0])
    y_arr_loc = np.arange(0, loc_cell[1], dv[1])
    loc_cell_n = np.array([len(x_arr_loc), len(y_arr_loc)])
    # Define it such that the origin is somewhere
    # in the middle but exactly on a grid point
    mid_ixs = (loc_cell_n/2).astype(int)
    x_arr_loc -= x_arr_loc[mid_ixs[0]]
    y_arr_loc -= y_arr_loc[mid_ixs[1]]
    x_grid_loc, y_grid_loc = np.meshgrid(x_arr_loc, y_arr_loc, indexing='ij')

    # Some info
    print("Main plane:   ", plane_size, plane_size_n)
    print("Local plane: ", loc_cell, loc_cell_n)

    num_morbs = len(morb_composition[0][0][0][0])

    morb_planes = [np.zeros(plane_size_n) for _ in range(num_morbs)]

    morb_planes_local = np.zeros((num_morbs, loc_cell_n[0], loc_cell_n[1]))

    print("---- Setup: %.4f" % (time.time() - time1))

    time_radial_calc = 0.0
    time_spherical = 0.0
    time_loc_glob_add = 0.0
    time_loc_lmorb_add = 0.0


    for i_at in range(len(at_positions)):
        elem = at_elems[i_at][0]
        pos = at_positions[i_at]

        # how does the position match with the grid?
        int_shift = (pos[0:2]/dv).astype(int)
        frac_shift = pos[0:2]/dv - int_shift
        origin_diff = int_shift - mid_ixs

        # Shift the local grid such that origin is on the atom
        x_grid_rel_loc = x_grid_loc - frac_shift[0]*dv[0]
        y_grid_rel_loc = y_grid_loc - frac_shift[1]*dv[1]

        z_rel = plane_z - pos[2]

        r_vec_2 = x_grid_rel_loc**2 + y_grid_rel_loc**2 + z_rel**2

        morb_planes_local.fill(0.0)

        for i_shell, shell in enumerate(basis_sets[elem]):
            l = shell[0]
            es = shell[1]
            cs = shell[2]

            # Calculate the radial part of the atomic orbital
            time2 = time.time()
            radial_part = np.zeros(loc_cell_n)
            for e, c in zip(es, cs):
                radial_part += c*np.exp(-1.0*e*r_vec_2)
            time_radial_calc += time.time() - time2

            for i, m in enumerate(range(-l, l+1, 1)):
                time2 = time.time()
                atomic_orb = radial_part*spherical_harmonic_grid(l, m,
                                                                 x_grid_rel_loc,
                                                                 y_grid_rel_loc,
                                                                 z_rel)
                time_spherical += time.time() - time2

                i_set = 0 # SHOULD START SUPPORTING MULTIPLE SET BASES AT SOME POINT
                coef_arr = morb_composition[i_at][i_set][i_shell][i]

                time2 = time.time()

                #morb_planes_local += np.einsum('i,jk', coef_arr, atomic_orb)

                morb_planes_local += np.outer(coef_arr, atomic_orb).reshape(
                                 num_morbs, loc_cell_n[0], loc_cell_n[1])

                time_loc_lmorb_add += time.time() - time2

        time2 = time.time()
        for i_mo in range(num_morbs):
            add_local_to_global_grid(morb_planes_local[i_mo], morb_planes[i_mo], origin_diff)
        time_loc_glob_add += time.time() - time2

    print("---- Radial calc time : %4f" % time_radial_calc)
    print("---- Spherical calc time : %4f" % time_spherical)
    print("---- Loc -> loc_morb time : %4f" % time_loc_lmorb_add)
    print("---- loc_morb -> glob time : %4f" % time_loc_glob_add)
    print("---- Total time: %.4f"%(time.time() - time1))

    return morb_planes


# Puts the molecular orbitals onto a specified region. For a plane, just specify eval_reg_size_n[2] = 1
# All inputs are needed to be in [a.u.], except for pbc_box (angstrom)
def calc_morbs_in_region(eval_cell, eval_cell_n, eval_reg_z,
                         at_positions, at_elems,
                         basis_sets, morb_composition,
                         pbc_box_size = 16.0,
                         print_info=True):

    time1 = time.time()

    dv = eval_cell/eval_cell_n

    pbc_box_size *= ang_2_bohr

    # Define small grid for orbital evaluation
    # and convenient PBC implementation
    loc_cell = np.array([pbc_box_size,  pbc_box_size, eval_cell[2]])
    x_arr_loc = np.arange(0, loc_cell[0], dv[0])
    y_arr_loc = np.arange(0, loc_cell[1], dv[1])
    z_arr_loc = np.arange(0, loc_cell[2], dv[2])
    z_arr_loc += eval_reg_z

    loc_cell_n = np.array([len(x_arr_loc), len(y_arr_loc), len(z_arr_loc)])
    # Define it such that the origin is somewhere
    # in the middle but exactly on a grid point
    mid_ixs = (loc_cell_n/2).astype(int)
    x_arr_loc -= x_arr_loc[mid_ixs[0]]
    y_arr_loc -= y_arr_loc[mid_ixs[1]]

    x_grid_loc, y_grid_loc, z_grid_loc = np.meshgrid(x_arr_loc, y_arr_loc, z_arr_loc, indexing='ij')

    num_morbs = len(morb_composition[0][0][0][0])
    morb_grids = 0 # release memory from previous run (needed in some rare cases)
    #morb_grids = [np.zeros(eval_cell_n) for _ in range(num_morbs)]
    morb_grids = np.zeros((num_morbs, eval_cell_n[0], eval_cell_n[1], eval_cell_n[2]))
    morb_grids_local = np.zeros((num_morbs, loc_cell_n[0], loc_cell_n[1], loc_cell_n[2]))

    # Some info
    if print_info:
        print("Eval cell:   ", eval_cell, eval_cell_n)
        print("Local cell: ", loc_cell, loc_cell_n)
        print("---- Setup: %.4f" % (time.time() - time1))

    time_radial_calc = 0.0
    time_spherical = 0.0
    time_loc_glob_add = 0.0
    time_loc_lmorb_add = 0.0

    for i_at in range(len(at_positions)):
        elem = at_elems[i_at][0]
        pos = at_positions[i_at]

        # how does the position match with the grid?
        int_shift = (pos[0:2]/dv[0:2]).astype(int)
        frac_shift = pos[0:2]/dv[0:2] - int_shift
        origin_diff = int_shift - mid_ixs[0:2]

        # Shift the local grid such that origin is on the atom
        x_grid_rel_loc = x_grid_loc - frac_shift[0]*dv[0]
        y_grid_rel_loc = y_grid_loc - frac_shift[1]*dv[1]

        z_grid_rel_loc = z_grid_loc - pos[2]

        r_vec_2 = x_grid_rel_loc**2 + y_grid_rel_loc**2 + z_grid_rel_loc**2

        morb_grids_local.fill(0.0)

        for i_shell, shell in enumerate(basis_sets[elem]):
            l = shell[0]
            es = shell[1]
            cs = shell[2]

            # Calculate the radial part of the atomic orbital
            time2 = time.time()
            radial_part = np.zeros(loc_cell_n)
            for e, c in zip(es, cs):
                radial_part += c*np.exp(-1.0*e*r_vec_2)
            time_radial_calc += time.time() - time2

            for i, m in enumerate(range(-l, l+1, 1)):
                time2 = time.time()
                atomic_orb = radial_part*spherical_harmonic_grid(l, m,
                                                                 x_grid_rel_loc,
                                                                 y_grid_rel_loc,
                                                                 z_grid_rel_loc)
                time_spherical += time.time() - time2

                i_set = 0 # SHOULD START SUPPORTING MULTIPLE SET BASES AT SOME POINT
                coef_arr = morb_composition[i_at][i_set][i_shell][i]

                time2 = time.time()
                for i_mo in range(num_morbs):
                    morb_grids_local[i_mo] += coef_arr[i_mo]*atomic_orb

                # slow:
                #morb_grids_local += np.outer(coef_arr, atomic_orb).reshape(
                #                 num_morbs, loc_cell_n[0], loc_cell_n[1], loc_cell_n[2])
                time_loc_lmorb_add += time.time() - time2

        time2 = time.time()
        for i_mo in range(num_morbs):
            add_local_to_global_box(morb_grids_local[i_mo], morb_grids[i_mo], origin_diff)
        time_loc_glob_add += time.time() - time2

    if print_info:
        print("---- Radial calc time : %4f" % time_radial_calc)
        print("---- Spherical calc time : %4f" % time_spherical)
        print("---- Loc -> loc_morb time : %4f" % time_loc_lmorb_add)
        print("---- loc_morb -> glob time : %4f" % time_loc_glob_add)
        print("---- Total time: %.4f"%(time.time() - time1))

    return morb_grids

### ---------------------------------------------------------------------------
### MISC
### ---------------------------------------------------------------------------

def write_cube_file(filename, file_xyz, cell, cell_n, data):

    # Read atomic positions (a.u.)
    positions, elems_nrs = read_atoms(file_xyz)

    natoms = len(positions)
    origin = np.array([0.0, 0.0, 0.0])
    origin *= ang_2_bohr

    f = open(filename, 'w')

    f.write('title\n')
    f.write('comment\n')

    dv_br = cell/cell_n
    dv_br = dv_br * np.diag([1, 1, 1])

    f.write("%5d %12.6f %12.6f %12.6f\n"%(natoms, origin[0], origin[1], origin[2]))

    for i in range(3):
        f.write("%5d %12.6f %12.6f %12.6f\n"%(cell_n[i], dv_br[i][0], dv_br[i][1], dv_br[i][2]))

    for i in range(natoms):
        at_x = positions[i][0]
        at_y = positions[i][1]
        at_z = positions[i][2]
        f.write("%5d %12.6f %12.6f %12.6f %12.6f\n"%(elems_nrs[i][1], 0.0, at_x, at_y, at_z))

    data.tofile(f, sep='\n', format='%12.6e')

    f.close()


def read_cube_file(cube_file):

    f = open(cube_file, 'r')
    title = f.readline()
    comment = f.readline()

    axes = [0, 1, 2]
    line = f.readline().split()
    natoms = int(line[0])

    origin = np.array(line[1:], dtype=float) / ang_2_bohr

    shape = np.empty(3,dtype=int)
    cell = np.empty((3, 3))
    for i in range(3):
        n, x, y, z = [float(s) for s in f.readline().split()]
        shape[i] = int(n)
        cell[i] = n * np.array([x, y, z])

    cell = cell / ang_2_bohr

    numbers = np.empty(natoms, int)
    positions = np.empty((natoms, 3))
    for i in range(natoms):
        line = f.readline().split()
        numbers[i] = int(line[0])
        positions[i] = [float(s) for s in line[2:]]

    positions /= ang_2_bohr
    data = np.array(f.read().split(), dtype=float)
    data = data.reshape(shape)
    f.close()

    return title, comment, natoms, origin, shape, cell, numbers, positions, data


def resize_2d_arr_with_interpolation(array, new_shape):
    x_arr = np.linspace(0, 1, array.shape[0])
    y_arr = np.linspace(0, 1, array.shape[1])
    rgi = scipy.interpolate.RegularGridInterpolator(points=[x_arr, y_arr], values=array)

    x_arr_new = np.linspace(0, 1, new_shape[0])
    y_arr_new = np.linspace(0, 1, new_shape[1])
    x_coords = np.repeat(x_arr_new, len(y_arr_new))
    y_coords = np.tile(y_arr_new, len(x_arr_new))

    return rgi(np.array([x_coords, y_coords]).T).reshape(new_shape)

# Extrapolate molecular orbitals from a specified plane to a box or another plane
# in case of "single_plane = True", the orbitals will be only extrapolated on
# a plane "extent" distance away
# Extent in bohr !!!
def extrapolate_morbs(morb_grids, morb_energies, dv, plane_ind,
                      extent, hart_plane, single_plane, use_weighted_avg=True):
    # NB: everything in hartree units!
    time1 = time.time()

    num_morbs = np.shape(morb_grids)[0]
    eval_reg_size_n = np.shape(morb_grids[0])

    if single_plane:
        extrap_morbs = np.zeros((num_morbs, eval_reg_size_n[0], eval_reg_size_n[1]))
    else:
        extrap_morbs = np.zeros((num_morbs, eval_reg_size_n[0], eval_reg_size_n[1],
                                 int(extent/dv[2])))

    for morb_index in range(num_morbs):

        morb_plane = morb_grids[morb_index][:, :, plane_ind]

        if use_weighted_avg:
            # weigh the hartree potential by the molecular orbital
            density_plane = morb_plane**2
            density_plane /= np.sum(density_plane)
            weighted_hartree = density_plane * resize_2d_arr_with_interpolation(hart_plane, density_plane.shape)
            hartree_avg = np.sum(weighted_hartree)
        else:
            hartree_avg = np.mean(hart_plane)

        energy = morb_energies[morb_index]/hart_2_ev
        if energy > hartree_avg:
            print("Warning: unbound state, can't extrapolate! index: %d. Exiting." % morb_index)
            break

        fourier = np.fft.rfft2(morb_plane)
        # NB: rfft2 takes REAL fourier transform over last (y) axis and COMPLEX over other (x) axes
        # dv in BOHR, so k is in 1/bohr
        kx_arr = 2*np.pi*np.fft.fftfreq(morb_plane.shape[0], dv[0])
        ky_arr = 2*np.pi*np.fft.rfftfreq(morb_plane.shape[1], dv[1])

        kx_grid, ky_grid = np.meshgrid(kx_arr, ky_arr,  indexing='ij')

        if single_plane:
            prefactors = np.exp(-np.sqrt(kx_grid**2 + ky_grid**2 - 2*(energy - hartree_avg))*extent)
            extrap_morbs[morb_index, :, :] = np.fft.irfft2(fourier*prefactors, morb_plane.shape)
        else:
            prefactors = np.exp(-np.sqrt(kx_grid**2 + ky_grid**2 - 2*(energy - hartree_avg))*dv[2])
            for iz in range(np.shape(extrap_morbs)[3]):
                fourier *= prefactors
                extrap_morbs[morb_index, :, :, iz] = np.fft.irfft2(fourier, morb_plane.shape)

    print("Extrapolation time: %.3f s"%(time.time()-time1))
    return extrap_morbs
