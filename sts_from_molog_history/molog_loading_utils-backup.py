
import os
import numpy as np
import time
import copy
import sys

ang_2_bohr = 1.0/0.52917721067

# Reads from the cp2k input file:
# * Basis set names for all elements
# * Cell size (bohr radii) and estimates the number of volume elements
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
    # Cp2k tries to create grids, where spacing is close to 0.08 angstroms
    step = 0.08
    step *= ang_2_bohr
    cell_n = (np.round(cell/step)).astype(int)

    return elem_basis_name, cell, cell_n

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
            if not skip:
                l = l.strip()
                if(len(l)==0): continue
                lines.append(l)

    # detect dimensions
    parts = lines[-3].split()
    nbasis = int(parts[0])
    natoms = int(parts[1])

    first_nmo = int(lines[1].split()[0])
    last_nmo = int(lines[-nbasis-5].split()[-1])
    nmos = last_nmo-first_nmo + 1

    nlines_per_block = nbasis + 3
    nblocks = int((len(lines)-3)/nlines_per_block)

    print("Found %d MOs spanned by %d basis functions centered on %d atoms."%(nmos, nbasis, natoms))

    assert lines[-1].startswith("HOMO-LUMO gap:")
    assert lines[-2].startswith("Fermi energy:")
    fermi_energy = 27.211385 * float(lines[-2].split()[2])

    # unfold table
    idx = []
    evals = []
    occs = []
    evecs = [list() for i in range(nbasis)]
    labels = [l.split()[:4] for l in lines[4:nbasis+4]]

    first_line = 1

    for iblock in range(nblocks):
        a = first_line + iblock*nlines_per_block
        evals.extend(lines[a+1].split())
        occs.extend(lines[a+2].split())
        for j in range(nbasis):
            parts = lines[a+3+j].split()
            assert parts[:4] == labels[j]
            evecs[j].extend(parts[4:])

    # convert to numpy arrays
    evals = np.array(evals, float)
    occs = np.array(occs, float)
    evecs = np.array(evecs, float)

    assert evals.shape == (nmos,)
    assert occs.shape == (nmos,)
    assert evecs.shape == (nbasis, nmos)

    # convert hartree to eV
    evals = 27.211385 * evals
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

# Same as above, but tries to process the file one line at a time
# (and is slower)
def read_cp2k_MO_file_enh(molog_file):
    print("Reading CP2K MOs from:"+molog_file)

    morb_indexes = []
    evals = []
    occs = []

    # data[i_morb][i_atom][i_aorb] = coeff
    data = []

    def next_line(f):
        #Skips empty lines
        l = ""
        while len(l) == 0:
            l = next(f).strip()
        return l

    fermi_level = 0.0

    with open(molog_file) as f:

        # first line number of columns
        fl_ncols = -1

        for l in f:
            if "MO EIGENVALUES" in l and "SCF STEP" not in l:
                break

        prev_iatom = -1

        for l in f:

            l = l.strip()
            if(len(l)==0):
                continue

            parts = l.split()

            if l.startswith('Fermi'):
                fermi_energy = hart_2_ev * float(parts[-1])
                break

            if fl_ncols == -1:
                # First line = first morbital header
                fl_ncols = len(parts)

            if len(parts) <= fl_ncols:
                # NEW BLOCK
                morb_indexes.extend([float(i) for i in parts])
                # Eigenvalue line:
                parts = next_line(f).split()
                evals.extend([float(i) for i in parts])
                # Occupancy line:
                parts = next_line(f).split()
                occs.extend([float(i) for i in parts])

                data.extend([[] for _ in range(len(parts))])
                continue

            # In a main block
            assert len(parts) > 4

            ibasis = float(parts[0])
            iatom = float(parts[1])
            elem = parts[2]
            orb_label = parts[3]

            coefs = [float(i) for i in parts[4:]]

            ncols = len(coefs)

            if iatom != prev_iatom:
                # Reached new atom: add new list for every morb
                for i in range(ncols):
                    i_morb = i-ncols
                    data[i_morb].append([])
                prev_iatom = iatom

            for i in range(ncols):
                i_morb = i - ncols
                data[i_morb][-1].append(coefs[i])

    evals = np.array(evals) * hart_2_ev
    return data, evals, occs, morb_indexes, fermi_energy

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
    lumo_i = np.argmin(occs)
    homo_i = lumo_i - 1
    if occs[lumo_i] != 0:
        print("Warning: Didn't find LUMO, energies are wrong")
    gap_middle = (evals[lumo_i] + evals[homo_i])/2
    morb_energies = evals - gap_middle
    return morb_composition, morb_energies
