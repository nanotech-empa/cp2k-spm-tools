#!/usr/bin/env python
import os
import numpy as np
import time
import copy
import sys
import re

import argparse

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

import atomistic_tools.cp2k_stm_utilities as csu

from mpi4py import MPI

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

parser = argparse.ArgumentParser(
    description='Calculates the molecular orbitals on a specified grid.')

parser.add_argument(
    '--cp2k_input',
    metavar='FILENAME',
    required=True,
    help='CP2K input of the SCF calculation.')
parser.add_argument(
    '--basis_file',
    metavar='FILENAME',
    required=True,
    help='File containing the used basis sets.')
parser.add_argument(
    '--xyz_file',
    metavar='FILENAME',
    required=True,
    help='.xyz file containing the geometry.')
parser.add_argument(
    '--wfn_file',
    metavar='FILENAME',
    required=True,
    help='Restart file containing the final wavefunction.')
parser.add_argument(
    '--output_file',
    metavar='FILENAME',
    required=True,
    help='File, where to save the output containing the \
          molecular orbitals and supporting info.')

parser.add_argument(
    '--emin',
    type=float,
    metavar='E',
    required=True,
    help='Lowest energy value for selecting orbitals (eV).')
parser.add_argument(
    '--emax',
    type=float,
    metavar='E',
    required=True,
    help='Highest energy value for selecting orbitals (eV).')
parser.add_argument(
    '--eval_region',
    type=str,
    nargs=6,
    metavar='X',
    required=True,
    help=("Specify evaluation region limits [xmin xmax ymin ymax zmin zmax] as a string: "
          "'G' corresponds to global cell limit (also enables PBC if both of pair are 'G') "
          "number and t/b (e.g. '2.5t') specifies distance [ang] from furthest-extending atom "
          "in top (t) or bottom (b) direction. "
          "Number with _element ('2.5t_C') correspondingly from furthest atom of elem. "
          "If xmin=xmax (within 1e-4), then only a plane is evaluated ")
)

parser.add_argument(
    '--dx',
    type=float,
    metavar='DX',
    required=True,
    help='Spatial step for the grid (angstroms).')
parser.add_argument(
    '--eval_cutoff',
    type=float,
    metavar='D',
    default=18.0,
    help=("Size of the region around the atom where each"
          " orbital is evaluated (only used for ref '0').")
)


# Define all variables that must be later broadcasted
args = None
cell = None
ase_atoms = None
basis_sets = None

print("Starting rank %d/%d"%(mpi_rank, mpi_size))

### -----------------------------------------
### SETUP (rank 0)
### -----------------------------------------
setup_success = False
try:
    if mpi_rank == 0:
        args = parser.parse_args()

        ### -----------------------------------------
        ### Read input files
        ### -----------------------------------------

        time0 = time.time()
        elem_basis_names, cell = csu.read_cp2k_input(args.cp2k_input)
        print("Read cp2k input: %.3f" % (time.time()-time0))

        time1 = time.time()
        ase_atoms = csu.read_xyz(args.xyz_file)
        csu.center_atoms_to_cell(ase_atoms.positions, cell/ang_2_bohr)
        print("Read xyz: %.3f" % (time.time()-time1))

        time1 = time.time()
        basis_sets = csu.read_basis_functions(args.basis_file, elem_basis_names)
        print("Read basis sets: %.3f" % (time.time()-time1))

        setup_success = True
finally:
    setup_success = comm.bcast(setup_success, root=0)

if not setup_success:
    print(mpi_rank, "exiting")
    exit(0)

time1 = time.time()

args = comm.bcast(args, root=0)
cell = comm.bcast(cell, root=0)
ase_atoms = comm.bcast(ase_atoms, root=0)
basis_sets = comm.bcast(basis_sets, root=0)

### -----------------------------------------
### Define morb evaluation region
### -----------------------------------------

eval_regions_inp = [args.eval_region[0:2], args.eval_region[2:4], args.eval_region[4:6]]
eval_regions = [[0, 0], [0, 0], [0, 0]]

### Parse evaluation regions specified in input.................

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

for i in range(3):
    if eval_regions_inp[i] == ['G', 'G']:
        eval_regions[i] = None
        continue
    for j in range(2):
        reg_str = eval_regions_inp[i][j]

        if '_' in reg_str:
            elem = reg_str.split('_')[1]
            reg_str = reg_str.split('_')[0]
            sel_positions = ase_atoms.positions[np.array(ase_atoms.get_chemical_symbols()) == elem]
            if len(sel_positions) == 0:
                print("Error: No element %s found. Exiting."%elem)
                exit(1)
        else:
            sel_positions = ase_atoms.positions


        if reg_str == 'G':
            eval_regions[i][j] = 0.0 if j == 0 else cell[i]
        else:
            ref_at_pos = reg_str[-1]
            ref_shift_str = reg_str[:-1]

            if ref_at_pos != 't' and ref_at_pos != 'b':
                print("Error:", reg_str, "needs to end with a 't' or 'b'")
                exit(1)
            if not is_number(ref_shift_str):
                print("Error:", ref_shift_str, "needs to be a number")
                exit(1)
            ref_shift_val = float(ref_shift_str)

            eval_regions[i][j] = (np.min(sel_positions[:, i]) + ref_shift_val
                                    if ref_at_pos == 'b' else
                                  np.max(sel_positions[:, i]) + ref_shift_val)
            eval_regions[i][j] *= ang_2_bohr

    if np.abs(eval_regions[i][0] - eval_regions[i][1]) < 1e-3:
        eval_regions[i][0] = eval_regions[i][1]


def minmax(arr):
    return np.array([np.min(arr), np.max(arr)])

if mpi_rank == 0:
    print("Evaluation regions (Bohr):")
    print("     x:", eval_regions[0])
    print("     y:", eval_regions[1])
    print("     z:", eval_regions[2])
    print("Bounding box of atoms (Bohr):")
    print("     x:", minmax(ase_atoms.positions[:, 0])*ang_2_bohr)
    print("     y:", minmax(ase_atoms.positions[:, 1])*ang_2_bohr)
    print("     z:", minmax(ase_atoms.positions[:, 2])*ang_2_bohr)

# Define real space grid
# Cp2k chooses close to 0.08 angstroms (?)
step = args.dx
step *= ang_2_bohr

global_size_n = (np.round(cell/step)).astype(int)
dv = cell/global_size_n


### -----------------------------------------
### Divide the energy range between the processors
### -----------------------------------------

emin_loc = args.emin + mpi_rank*(args.emax-args.emin)/mpi_size
emax_loc = args.emin + (mpi_rank+1)*(args.emax-args.emin)/mpi_size - 1e-14

morb_composition, morb_energies, morb_occs, homo_inds, ref_energy = \
    csu.load_restart_wfn_file(args.wfn_file, emin_loc, emax_loc)

nspin = len(morb_composition)

for ispin in range(nspin):
    num_orbs = len(morb_composition[ispin][0][0][0][0])
    assert num_orbs == len(morb_energies[ispin])
    print("S%d Rank %d energy range %.2f:%.2f; num orbs: %d" %(ispin, mpi_rank, emin_loc, emax_loc, num_orbs))

### -----------------------------------------
### Calculate the molecular orbitals in the specified region
### -----------------------------------------

morb_grids = csu.calc_morbs_in_region(cell, global_size_n,
                ase_atoms,
                basis_sets, morb_composition,
                x_eval_region = eval_regions[0],
                y_eval_region = eval_regions[1],
                z_eval_region = eval_regions[2],
                eval_cutoff = args.eval_cutoff,
                print_info = (mpi_rank == 0))

grid_shape = morb_grids[0][0].shape

morb_grids_collected = []
morb_energies_collected= []

for ispin in range(nspin):

    morb_grids_rav = morb_grids[ispin].ravel()
    # Collect local array sizes using the high-level mpi4py gather
    sendcounts = np.array(comm.gather(len(morb_grids_rav), 0))

    # Collect the numbers of orbitals
    num_morbs = np.array(comm.gather(len(morb_grids[ispin]), 0))

    # Collect energies
    morb_en_gather = comm.gather(morb_energies[ispin], root=0)
    if mpi_rank == 0:
        morb_energies_collected.append(np.hstack(morb_en_gather))

    if mpi_rank == 0:
        print("spin {} sendcounts: {}, total: {}".format(ispin, sendcounts, sum(sendcounts)))
        recvbuf = np.empty(sum(sendcounts), dtype=float)
    else:
        recvbuf = None

    comm.Gatherv(sendbuf=morb_grids_rav, recvbuf=(recvbuf, sendcounts), root=0)

    if mpi_rank == 0:
        total_num_morbs = np.sum(num_morbs)
        morb_grids_collected.append(recvbuf.reshape(total_num_morbs,
                grid_shape[0], grid_shape[1], grid_shape[2]))
        

if mpi_rank == 0:

    coord_arrays = []
    for i in range(3):
        if eval_regions[i] is None:
            coord_arrays.append(np.linspace(0.0, cell[i]-dv[i], grid_shape[i]))
        else:
            coord_arrays.append(np.linspace(eval_regions[i][0], eval_regions[i][1], grid_shape[i]))

    geom_base = os.path.basename(args.xyz_file)
    geom_label = os.path.splitext(geom_base)[0]

    time1 = time.time()
    elim = np.array([args.emin, args.emax])

    mol_bbox = np.array([np.min(ase_atoms.positions[:, 0]),
                         np.max(ase_atoms.positions[:, 0]),
                         np.min(ase_atoms.positions[:, 1]),
                         np.max(ase_atoms.positions[:, 1]),
                         np.min(ase_atoms.positions[:, 2]),
                         np.max(ase_atoms.positions[:, 2])]) * ang_2_bohr
    
    if nspin == 1:
        np.savez(args.output_file,
            morb_grids_s1=morb_grids_collected[0],
            morb_energies_s1=morb_energies_collected[0],
            homo_s1=homo_inds[0][0],
            x_arr=coord_arrays[0], # Bohr
            y_arr=coord_arrays[1], # Bohr
            z_arr=coord_arrays[2], # Bohr
            mol_bbox=mol_bbox, # Bohr
            elim=elim,
            ref_energy=ref_energy,
            geom_label=geom_label)
    else:
        np.savez(args.output_file,
            morb_grids_s1=morb_grids_collected[0],
            morb_energies_s1=morb_energies_collected[0],
            homo_s1=homo_inds[0][0],
            morb_grids_s2=morb_grids_collected[1],
            morb_energies_s2=morb_energies_collected[1],
            homo_s2=homo_inds[0][1],
            x_arr=coord_arrays[0], # Bohr
            y_arr=coord_arrays[1], # Bohr
            z_arr=coord_arrays[2], # Bohr
            mol_bbox=mol_bbox, # Bohr
            elim=elim,
            ref_energy=ref_energy,
            geom_label=geom_label)

    print("Saved the orbitals to file: %.2fs" % (time.time() - time1))
    print("Total time taken for the whole run: %.2fs" % (time.time() - time0))
