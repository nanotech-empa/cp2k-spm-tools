
import os
import numpy as np
import time
import copy
import sys

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

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
                cell[0] = float(parts[1])
                cell[1] = float(parts[2])
                cell[2] = float(parts[3])

    cell *= ang_2_bohr

    return elem_basis_name, cell

# Read atomic positions (in Bohr radiuses)
def read_atoms(file_xyz):
    # Read atomic positions (in angstrom)
    data = np.genfromtxt(file_xyz, dtype=None, skip_header=2)
    data = np.atleast_1d(data)
    elems_nrs = []
    positions = []
    elem_nr = {'H': 1,
               'C': 6}
    for line in data:
        elem = line[0].decode("utf-8")
        if len(line) > 4:
            nr = line[4]
        elif elem in elem_nr:
            nr = elem_nr[elem]
        else:
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

### ----------------------------------------------------------------------
### BASIS SET setup
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

### ----------------------------------------------------------------------
### MOLOG FILE loading and processing

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

    # calculate energies wrt middle of the HOMO-LUMO gap
    #lumo_i = np.argmin(occs)
    #homo_i = lumo_i - 1
    #if occs[lumo_i] != 0:
    #    print("Warning: Didn't find LUMO, energies are wrong")
    #gap_middle = (evals[lumo_i] + evals[homo_i])/2
    #morb_energies = evals - gap_middle

    morb_energies = evals - fermi_energy

    return morb_composition, morb_energies

# ------------------------------------------------------------------------------
# Methods more directly related to putting stuff on grids

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

    print("No spherical harmonic found for l=%d, m=%d" % (l, m))
    return 0


def add_local_to_global_grid(loc_grid, glob_grid, origin_diff):
    loc_nx, loc_ny = np.shape(loc_grid)
    glob_nx, glob_ny = np.shape(glob_grid)
    od = origin_diff

    # Move the origin_diff vector to the main global cell (not an image)
    od[0] = od[0] % glob_nx
    od[1] = od[1] % glob_ny

    ixs = [[od[0], od[0] + loc_nx]]
    l_ixs = [0]
    while ixs[-1][1] > glob_nx:
        overshoot = ixs[-1][1]-glob_nx
        ixs[-1][1] = glob_nx
        l_ixs.append(l_ixs[-1]+glob_nx-ixs[-1][0])
        ixs.append([0, overshoot])
    l_ixs.append(loc_nx)

    iys = [[od[1], od[1] + loc_ny]]
    l_iys = [0]
    while iys[-1][1] > glob_ny:
        overshoot = iys[-1][1]-glob_ny
        iys[-1][1] = glob_ny
        l_iys.append(l_iys[-1]+glob_ny-iys[-1][0])
        iys.append([0, overshoot])
    l_iys.append(loc_ny)

    for i, ix in enumerate(ixs):
        for j, iy in enumerate(iys):
            glob_grid[ix[0]:ix[1], iy[0]:iy[1]] += loc_grid[l_ixs[i]:l_ixs[i+1], l_iys[j]:l_iys[j+1]]
