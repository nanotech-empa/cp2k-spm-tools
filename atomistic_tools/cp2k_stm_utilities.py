""" Tools to perform STM/STS analysis on CP2K calculations

- Kristjan Eimre
""" 

import os
import numpy as np
import scipy
import scipy.io
import time
import copy
import sys

import re
import io
import ase
import ase.io

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

            if parts[0] == "A" or parts[0] == "B" or parts[0] == "C":
                prim_vec = np.array([float(x) for x in parts[1:]])
                if np.sum(prim_vec > 0.0) > 1:
                    raise ValueError("Cell is not rectangular")
                ind = np.argmax(prim_vec > 0.0)
                cell[ind] = prim_vec[ind]

    cell *= ang_2_bohr

    if any(cell < 1e-3):
        raise ValueError("Cell " + str(cell) + " is invalid")
    return elem_basis_name, cell

# Read atomic positions from .xyz file (in Bohr radiuses)
def read_xyz(file_xyz):
    with open(file_xyz) as f:
        fxyz_contents = f.read()
    # Replace custom elements (e.g. for spin-pol calcs)
    fxyz_contents = re.sub("([a-zA-Z]+)[0-9]+", r"\1", fxyz_contents)
    atoms = ase.io.read(io.StringIO(fxyz_contents), format="xyz")
    return atoms

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
    for elem, bsets in basis_sets.items():
        for bset in bsets:
            for shell in bset:
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
    """ Reads the basis sets from basis_set_file specified in elem_basis_name

    returns:
    basis_sets["Element"] = 
    """
    basis_sets = {}
    with open(basis_set_file) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            parts = lines[i].split()
            if len(parts) == 0:
                continue
            if parts[0] in elem_basis_name:
                elem = parts[0]
                if parts[1] == elem_basis_name[elem] or (len(parts) > 2 and parts[2] == elem_basis_name[elem]):
                    # We have found the correct basis set
                    basis_functions = []
                    nsets = int(lines[i+1])
                    cursor = 2
                    for j in range(nsets):
                        
                        basis_functions.append([])

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
                            for il in range(nl):
                                basis_functions[-1].append([l, exps, coeffs[:, indx]])
                                indx += 1
                        cursor += n_exp + 1

                    basis_sets[elem] = basis_functions

    return magic_basis_normalization(basis_sets)

### ---------------------------------------------------------------------------
### RESTART file loading and processing
### ---------------------------------------------------------------------------

def load_restart_wfn_file(restart_file, emin, emax):
    """ Reads the molecular orbitals from cp2k restart wavefunction file in specified energy range
    Note that the energy range is in eV and with respect to HOMO energy.
    
    Return:
    morb_composition[ispin][iatom][iset][ishell][iorb] = coefs[i_mo]
    morb_energies[ispin] = energies[i_mo] in eV with respect to HOMO
    morb_occs[ispin] = occupancies[i_mo]
    homo_inds[ispin] = homo_index_for_ispin
    """

    inpf = scipy.io.FortranFile(restart_file, 'r')

    natom, nspin, nao, nset_max, nshell_max = inpf.read_ints()
    #print(natom, nspin, nao, nset_max, nshell_max)
    # natom - number of atoms
    # nspin - number of spins
    # nao - number of atomic orbitals
    # nset_max - maximum number of sets in the basis set
    #           (e.g. if one atom's basis set contains 3 sets and every other
    #           atom's contains 1, then this value will still be 3)
    # nshell_max - maximum number of shells in each set

    # number of sets in the basis set for each atom
    nset_info = inpf.read_ints()
    #print(nset_info)

    # number of shells in each of the sets
    nshell_info = inpf.read_ints()
    #print(nshell_info)

    # number of orbitals in each shell
    nso_info = inpf.read_ints()
    #print(nso_info)

    morb_composition = []
    morb_energies = []
    morb_occs = []

    homo_ens = []

    # different HOMO indexes (for debugging and matching direct cube output)
    loc_homo_inds = []  # indexes wrt to selected morbitals
    glob_homo_inds = [] # global indexes, corresponds to WFN nr (counting start from 1)
    cp2k_homo_inds = [] # cp2k homo indexes, takes also smearing into account (counting start from 1)

    for ispin in range(nspin):
        nmo, homo, lfomo, nelectron = inpf.read_ints()
        #print("nmo, homo, lfomo, nelectron", nmo, homo, lfomo, nelectron)
        # nmo - number of molecular orbitals
        # homo - index of the HOMO
        # lfomo - ???
        # nelectron - number of electrons
        
        # Note that "homo" is affected by smearing. to have the correct, T=0K homo:
        if nspin == 1:
            i_homo = int(nelectron/2) - 1
        else:
            i_homo = nelectron - 1

        # list containing all eigenvalues and occupancies of the molecular orbitals
        evals_occs = inpf.read_reals()
        #print(evals_occs)

        evals = evals_occs[:int(len(evals_occs)/2)]
        occs = evals_occs[int(len(evals_occs)/2):]
        
        evals *= hart_2_ev

        print("S%d nmo: %d, [eV] H-1 %.8f Homo %.8f H+1 %.8f" % (ispin, nmo,
                            evals[i_homo-1], evals[i_homo], evals[i_homo+1]))
        homo_ens.append(evals[i_homo])
        
        ### ---------------------------------------------------------------------
        ### Build up the structure of python lists to hold the morb_composition
        
        morb_composition.append([]) # 1: spin index
        shell_offset = 0
        norb_offset = 0
        orb_offset = 0
        for iatom in range(natom):
            nset = nset_info[iatom]
            morb_composition[-1].append([]) # 2: atom index
            for iset in range(nset):
                nshell = nshell_info[shell_offset]
                shell_offset += 1
                morb_composition[-1][-1].append([]) # 3: set index
                ishell = 0
                while ishell < nshell:
                    norb = nso_info[norb_offset]
                    norb_offset += 1
                    if norb == 0:
                        continue
                    ishell += 1
                    morb_composition[-1][-1][-1].append([]) # 4: shell index (l)
                    for iorb in range(norb):
                        morb_composition[-1][-1][-1][-1].append([]) # 5: orb index (m)
                        # And this will contain the array of coeffs corresponding to each MO
                        orb_offset += 1
        ### ---------------------------------------------------------------------
        
        ### ---------------------------------------------------------------------
        ### Read the coefficients from file and put to the morb_composition list
        
        morb_energies.append([])
        morb_occs.append([])

        first_imo = -1

        for imo in range(nmo):
            coefs = inpf.read_reals()
            if evals[imo] - evals[i_homo] < emin - 1.0:
                continue
            if evals[imo] - evals[i_homo] > emax + 1.0:
                if ispin == nspin - 1:
                    break
                else:
                    continue
            
            if first_imo == -1:
                first_imo = imo

            orb_offset = 0

            morb_energies[ispin].append(evals[imo])
            morb_occs[ispin].append(occs[imo])
            
            for iatom in range(len(morb_composition[ispin])):
                for iset in range(len(morb_composition[ispin][iatom])):
                    for ishell in range(len(morb_composition[ispin][iatom][iset])):
                        for iorb in range(len(morb_composition[ispin][iatom][iset][ishell])):
                            morb_composition[ispin][iatom][iset][ishell][iorb].append(coefs[orb_offset])
                            orb_offset += 1
        ### ---------------------------------------------------------------------
        
        ### ---------------------------------------------------------------------
        # Convert i_mo layer to numpy array
        for iatom in range(len(morb_composition[ispin])):
            for iset in range(len(morb_composition[ispin][iatom])):
                for ishell in range(len(morb_composition[ispin][iatom][iset])):
                    for iorb in range(len(morb_composition[ispin][iatom][iset][ishell])):
                        morb_composition[ispin][iatom][iset][ishell][iorb] = np.array(
                            morb_composition[ispin][iatom][iset][ishell][iorb]
                        )
        ### ---------------------------------------------------------------------

        loc_homo_inds.append(i_homo - first_imo)
        glob_homo_inds.append(i_homo + 1)
        cp2k_homo_inds.append(homo)

    ### ---------------------------------------------------------------------
    # reference energy for RKS is just HOMO, but for UKS will be average of both HOMOs

    if nspin == 1:
        ref_energy = homo_ens[0]
    else:
        ref_energy = (homo_ens[0] + homo_ens[1]) / 2
    
    ### ---------------------------------------------------------------------
    ### Select orbitals and energy and occupation values in specified range
    
    for ispin in range(nspin):
        morb_energies[ispin] -= ref_energy
        first_imo = np.searchsorted(morb_energies[ispin], emin)
        last_imo = np.searchsorted(morb_energies[ispin], emax) - 1
        if last_imo < first_imo:
            print("Warning: No orbitals found in specified energy range!")
            continue
        morb_energies[ispin] = morb_energies[ispin][first_imo:last_imo+1]
        morb_occs[ispin] = morb_occs[ispin][first_imo:last_imo+1]

        for iatom in range(len(morb_composition[ispin])):
            for iset in range(len(morb_composition[ispin][iatom])):
                for ishell in range(len(morb_composition[ispin][iatom][iset])):
                    for iorb in range(len(morb_composition[ispin][iatom][iset][ishell])):
                        morb_composition[ispin][iatom][iset][ishell][iorb] = \
                            morb_composition[ispin][iatom][iset][ishell][iorb][first_imo:last_imo+1]

        loc_homo_inds[ispin] -= first_imo
    ### ---------------------------------------------------------------------
        
    inpf.close()
    homo_inds = [loc_homo_inds, glob_homo_inds, cp2k_homo_inds]
    return morb_composition, morb_energies, morb_occs, homo_inds, ref_energy

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


def add_local_to_global_grid(loc_grid, glob_grid, origin_diff, wrap=(True, True, True)):
    """
    Method to add a grid to another one
    Arguments:
    loc_grid -- grid that will be added to the glob_grid
    glob_grid -- defines "wrapping" boundaries
    origin_diff -- difference of origins between the grids; ignored for directions without wrapping
    wrap -- specifies in which directions to wrap and take PBC into account
    """
    loc_n = np.shape(loc_grid)
    glob_n = np.shape(glob_grid)
    od = origin_diff

    inds = []
    l_inds = []

    for i in range(len(glob_n)):
        
        if wrap[i]:
            # Move the origin_diff vector to the main global cell if wrapping is enabled
            od[i] = od[i] % glob_n[i]

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
        else:
            inds.append([-1])
            l_inds.append([-1])

    l_ixs = l_inds[0]
    l_iys = l_inds[1]
    l_izs = l_inds[2]
    for i, ix in enumerate(inds[0]):
        for j, iy in enumerate(inds[1]):
            for k, iz in enumerate(inds[2]):
                if wrap[0]:
                    i_gl_x = slice(ix[0], ix[1])
                    i_lc_x = slice(l_ixs[i], l_ixs[i+1])
                else:
                    i_gl_x = slice(None)
                    i_lc_x = slice(None)
                if wrap[1]:
                    i_gl_y = slice(iy[0], iy[1])
                    i_lc_y = slice(l_iys[j], l_iys[j+1])
                else:
                    i_gl_y = slice(None)
                    i_lc_y = slice(None)
                if wrap[2]:
                    i_gl_z = slice(iz[0], iz[1])
                    i_lc_z = slice(l_izs[k], l_izs[k+1])
                else:
                    i_gl_z = slice(None)
                    i_lc_z = slice(None)
                
                glob_grid[i_gl_x, i_gl_y, i_gl_z] += loc_grid[i_lc_x, i_lc_y, i_lc_z]



def calc_morbs_in_region(global_cell, global_cell_n,
                         ase_atoms,
                         basis_sets, morb_composition,
                         x_eval_region = None,
                         y_eval_region = None,
                         z_eval_region = None,
                         eval_cutoff = 16.0,
                         print_info = True):
    """ 
    Puts the molecular orbitals onto a specified grid
    Arguments:
    global_cell -- global cell size (x, y, z) in [au]
    global_cell_n -- global cell discretization (x, y, z)
    at_positions -- atomic positions in [au]
    at_elems -- elements of atoms
    x_eval_region -- x evaluation (min, max) in [au]. If min == max, then evaluation only works on a plane.
                     If set, no PBC applied in direction and also no eval_cutoff.
    eval_cutoff -- cutoff for orbital evaluation if eval_region is None
    """

    time1 = time.time()

    dv = global_cell/global_cell_n
    eval_cutoff *= ang_2_bohr

    # Define local grid for orbital evaluation
    # and convenient PBC implementation
    eval_regions = [x_eval_region, y_eval_region, z_eval_region]
    loc_cell_arrays = []
    mid_ixs = np.zeros(3, dtype=int)
    loc_cell_n = np.zeros(3, dtype=int)
    eval_cell_n = np.zeros(3, dtype=int)
    for i in range(3):
        if eval_regions[i] is None:
            # Define range in i direction with 0.0 at index mid_ixs[i]
            loc_arr = np.arange(0, eval_cutoff, dv[i])
            mid_ixs[i] = int(len(loc_arr)/2)
            loc_arr -= loc_arr[mid_ixs[i]]
            loc_cell_arrays.append(loc_arr)
            eval_cell_n[i] = global_cell_n[i]
        else:
            # Define the specified range in direction i
            v_min, v_max = eval_regions[i]
            loc_cell_arrays.append(np.linspace(v_min, v_max, int(np.round((v_max-v_min)/dv[i]))+1))
            mid_ixs[i] = -1
            eval_cell_n[i] = len(loc_cell_arrays[i])
        loc_cell_n[i] = len(loc_cell_arrays[i])

    loc_cell_grids = np.meshgrid(loc_cell_arrays[0], loc_cell_arrays[1], loc_cell_arrays[2], indexing='ij')

    # Some info
    if print_info:
        print("Global cell: ", global_cell_n)
        print("Eval cell: ", eval_cell_n)
        print("local cell: ", loc_cell_n)
        print("---- Setup: %.4f" % (time.time() - time1))

    time_radial_calc = 0.0
    time_spherical = 0.0
    time_loc_glob_add = 0.0
    time_loc_lmorb_add = 0.0

    nspin = len(morb_composition)

    num_morbs = []
    morb_grids = []
    morb_grids_local = []

    for ispin in range(nspin):
        num_morbs.append(len(morb_composition[ispin][0][0][0][0]))
        morb_grids.append(np.zeros((num_morbs[ispin], eval_cell_n[0], eval_cell_n[1], eval_cell_n[2])))
        morb_grids_local.append(np.zeros((num_morbs[ispin], loc_cell_n[0], loc_cell_n[1], loc_cell_n[2])))


    for i_at in range(len(ase_atoms)):
        elem = ase_atoms[i_at].symbol
        pos = ase_atoms[i_at].position * ang_2_bohr

        # how does the position match with the grid?
        int_shift = (pos/dv).astype(int)
        frac_shift = pos/dv - int_shift
        origin_diff = int_shift - mid_ixs

        # Shift the local grid such that origin is on the atom
        rel_loc_cell_grids = []
        for i, loc_grid in enumerate(loc_cell_grids):
            if eval_regions[i] is None:
                rel_loc_cell_grids.append(loc_grid - frac_shift[i]*dv[i])
            else:
                rel_loc_cell_grids.append(loc_grid - pos[i])

        r_vec_2 = rel_loc_cell_grids[0]**2 + \
                  rel_loc_cell_grids[1]**2 + \
                  rel_loc_cell_grids[2]**2

        for ispin in range(nspin):
            morb_grids_local[ispin].fill(0.0)

        for i_set, bset in enumerate(basis_sets[elem]):
            for i_shell, shell in enumerate(bset):
                l = shell[0]
                es = shell[1]
                cs = shell[2]

                # Calculate the radial part of the atomic orbital
                time2 = time.time()
                radial_part = np.zeros(loc_cell_n)
                for e, c in zip(es, cs):
                    radial_part += c*np.exp(-1.0*e*r_vec_2)
                time_radial_calc += time.time() - time2

                for i_orb, m in enumerate(range(-l, l+1, 1)):
                    time2 = time.time()
                    atomic_orb = radial_part*spherical_harmonic_grid(l, m,
                                                                    rel_loc_cell_grids[0],
                                                                    rel_loc_cell_grids[1],
                                                                    rel_loc_cell_grids[2])
                    time_spherical += time.time() - time2
                    time2 = time.time()

                    for i_spin in range(nspin):
                        coef_arr = morb_composition[i_spin][i_at][i_set][i_shell][i_orb]

                        for i_mo in range(num_morbs[i_spin]):
                            morb_grids_local[i_spin][i_mo] += coef_arr[i_mo]*atomic_orb

                        # slow:
                        #morb_grids_local += np.outer(coef_arr, atomic_orb).reshape(
                        #                 num_morbs, loc_cell_n[0], loc_cell_n[1], loc_cell_n[2])
                    time_loc_lmorb_add += time.time() - time2

        time2 = time.time()
        for i_spin in range(nspin):
            for i_mo in range(num_morbs[i_spin]):
                add_local_to_global_grid(morb_grids_local[i_spin][i_mo], morb_grids[i_spin][i_mo], origin_diff, wrap=(mid_ixs != -1))
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

def write_cube_file(filename, ase_atoms, cell, cell_n, data, origin = np.array([0.0, 0.0, 0.0])):

    # Read atomic positions (a.u.)
    positions = ase_atoms.positions * ang_2_bohr
    numbers = ase_atoms.get_atomic_numbers()

    natoms = len(ase_atoms)

    f = open(filename, 'w')

    f.write('title\n')
    f.write('comment\n')

    dv_br = cell/cell_n
    dv_br = dv_br * np.diag([1, 1, 1])

    f.write("%5d %12.6f %12.6f %12.6f\n"%(natoms, origin[0], origin[1], origin[2]))

    for i in range(3):
        f.write("%5d %12.6f %12.6f %12.6f\n"%(cell_n[i], dv_br[i][0], dv_br[i][1], dv_br[i][2]))

    for i in range(natoms):
        at_x, at_y, at_z = positions[i]
        f.write("%5d %12.6f %12.6f %12.6f %12.6f\n"%(numbers[i], 0.0, at_x, at_y, at_z))

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

    # Option 1: less memory usage but might be slower
    data = np.empty(shape[0]*shape[1]*shape[2], dtype=float)
    cursor = 0
    for i, line in enumerate(f):
        ls = line.split()
        data[cursor:cursor+len(ls)] = ls
        cursor += len(ls)

    # Option 2: Takes much more memory (but may be faster)
    #data = np.array(f.read().split(), dtype=float)

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


def extrapolate_morbs(morb_planes, morb_energies, dv,
                      extent, single_plane,
                      work_function=None, hart_plane=None, use_weighted_avg=True):
    """
    Extrapolate molecular orbitals from a specified plane to a box or another plane
    in case of "single_plane = True", the orbitals will be only extrapolated on
    a plane "extent" distance away
    Extent in bohr !!!

    Either the work function or the hartree plane is needed!
    Both are assumed to be in hartree units wrt to Fermi/Homo.

    NB: everything in hartree units!
    """
    time1 = time.time()

    if work_function is None and hart_plane is None:
        print("You must specify either the WF or the hartree plane.")
        return None

    num_morbs = np.shape(morb_planes)[0]
    eval_reg_size_n = np.shape(morb_planes[0])

    if single_plane:
        extrap_morbs = np.zeros((num_morbs, eval_reg_size_n[0], eval_reg_size_n[1]))
    else:
        extrap_morbs = np.zeros((num_morbs, eval_reg_size_n[0], eval_reg_size_n[1],
                                 int(extent/dv[2])))

    for morb_index in range(num_morbs):

        morb_plane = morb_planes[morb_index]

        if work_function != None:
            hartree_avg = work_function
        else:
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

def get_hartree_plane_above_top_atom(hart_cube_data, height):
    """ Returns the hartree plane above topmost atom in z direction

    arguments:
        height - angstrom
        
    returns:
        hartree potential on the plane in hartree units and without any energy shift
    """

    hart_cube = hart_cube_data[-1]
    hart_cell = hart_cube_data[5]
    hart_atomic_pos = hart_cube_data[-2]

    topmost_atom_z = np.max(hart_atomic_pos[:, 2]) # Angstrom
    hart_plane_z = height + topmost_atom_z

    #print("height", height)
    #print("topmost_atom_z", topmost_atom_z)
    #print("hart_cell[2, 2]", hart_cell[2, 2])

    hart_plane_index = int(np.round(hart_plane_z/hart_cell[2, 2]*np.shape(hart_cube)[2]))

    return hart_cube[:, :, hart_plane_index]
