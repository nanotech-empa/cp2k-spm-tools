import os
import numpy as np
import time
import copy
import sys

import argparse

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

import cp2k_utilities as cu

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser(
    description='Calculates the molecular orbitals on a specified grid.')

parser.add_argument(
    '--cp2k_input',
    metavar='FILENAME',
    required=True,
    help='CP2K input of the SCF calculation.')
parser.add_argument(
    '--cp2k_output',
    metavar='FILENAME',
    required=True,
    help='CP2K output of the SCF calculation.')
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
    '--restart_file',
    metavar='FILENAME',
    required=True,
    help='Restart file containing the final wavefunction.')
parser.add_argument(
    '--output_file',
    metavar='FILENAME',
    required=True,
    help='Output containing the molecular orbitals and supporting info.')

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
    '--z_top',
    type=float,
    metavar='H',
    required=True,
    help='Distance of the top plane of the evaluation region from \
          topmost atom (angstroms).')
parser.add_argument(
    '--dx',
    type=float,
    metavar='DX',
    required=True,
    help='Spatial step for the grid (angstroms).')
parser.add_argument(
    '--local_eval_box_size',
    type=float,
    metavar='D',
    default=18.0,
    help='Spatial step for the grid (angstroms).')
parser.add_argument(
    '--single_plane',
    type=bool,
    metavar='SP',
    default=False,
    help='Calculate only single plane of molecular orbitals at z_top')


# Define all varialbles that must be later broadcasted
args = None
cell = None
at_positions = None
at_elems = None
basis_sets = None
morb_composition = None

### -----------------------------------------
### SETUP (rank 0)
### -----------------------------------------
setup_success = False
try:
    if rank == 0:
        args = parser.parse_args()

        ### -----------------------------------------
        ### Read input files
        ### -----------------------------------------

        time0 = time.time()
        elem_basis_names, cell = cu.read_cp2k_input(args.cp2k_input)
        print("Read cp2k input: %.3f" % (time.time()-time0))

        time1 = time.time()
        fermi = cu.read_fermi_from_cp2k_out(args.cp2k_output)
        print("Fermi energy: %.6f" % fermi)
        print("Read cp2k out: %.3f" % (time.time()-time1))

        time1 = time.time()
        at_positions, at_elems = cu.read_atoms(args.xyz_file)
        print("Read xyz: %.3f" % (time.time()-time1))

        time1 = time.time()
        basis_sets = cu.read_basis_functions(args.basis_file, elem_basis_names)
        print("Read basis sets: %.3f" % (time.time()-time1))

        time1 = time.time()
        morb_composition, morb_energies, morb_occs, ref_energy, i_homo = \
            cu.load_restart_wfn_file(args.restart_file, args.emin, args.emax, fermi)
        print("Found %d orbitals" % len(morb_energies))
        print("Read restart: %.3f" % (time.time()-time1))

        setup_success = True
finally:
    setup_success = comm.bcast(setup_success, root=0)

if not setup_success:
    print(rank, "exiting")
    exit(0)

time1 = time.time()

args = comm.bcast(args, root=0)
cell = comm.bcast(cell, root=0)
at_positions = comm.bcast(at_positions, root=0)
at_elems = comm.bcast(at_elems, root=0)
basis_sets = comm.bcast(basis_sets, root=0)
morb_composition = comm.bcast(morb_composition, root=0)

num_morbs = len(morb_composition[0][0][0][0])

morbs_per_rank = num_morbs//size

# Select correct molecular orbitals for each rank
ind_start = rank*morbs_per_rank
ind_end = (rank+1)*morbs_per_rank
if rank == size-1:
    ind_end = num_morbs
print("Rank %d works with orbitals %d:%d" %(rank, ind_start, ind_end))
for iatom in range(len(morb_composition)):
    for iset in range(len(morb_composition[iatom])):
        for ishell in range(len(morb_composition[iatom][iset])):
            for iorb in range(len(morb_composition[iatom][iset][ishell])):
                morb_composition[iatom][iset][ishell][iorb] = \
                    morb_composition[iatom][iset][ishell][iorb][ind_start:ind_end]

if rank == 0:
    print("Initial broadcast time %.4f s" % (time.time() - time1))

### -----------------------------------------
### Define morb evaluation region
### -----------------------------------------

height_above_atoms = args.z_top # angstroms
height_below_atoms = 1.0

top_atom_z = np.max(at_positions[:, 2]) # Bohr
z_top = top_atom_z + height_above_atoms*ang_2_bohr
carb_positions = at_positions[np.array(at_elems)[:, 0] == 'C']
z_bottom = np.min(carb_positions[:, 2]) - height_below_atoms*ang_2_bohr# Bohr

eval_reg_size = np.array([cell[0], cell[1], z_top-z_bottom])

# Define real space grid
# Cp2k chooses close to 0.08 angstroms (?)
step = args.dx
step *= ang_2_bohr

eval_reg_size_n = (np.round(eval_reg_size/step)).astype(int)
dv = eval_reg_size/eval_reg_size_n

# increase the z size such that top plane exactly matches z_top
eval_reg_size[2] += dv[2]
eval_reg_size_n[2] += 1

# z array in bohr and wrt topmost atom
z_arr = np.arange(0.0, eval_reg_size[2], dv[2]) + z_bottom - top_atom_z

if args.single_plane:
    eval_reg_size[2] = dv[2]
    eval_reg_size_n[2] = 1
    z_bottom = z_top
    z_arr = np.array([height_above_atoms*ang_2_bohr])

### -----------------------------------------
### Calculate the molecular orbitals in the specified region
### -----------------------------------------

morb_grids = cu.calc_morbs_in_region(eval_reg_size, eval_reg_size_n, z_bottom,
                at_positions, at_elems,
                basis_sets, morb_composition,
                pbc_box_size = args.local_eval_box_size,
                print_info = (rank == 0))

morb_grids = morb_grids.ravel()
# Collect local array sizes using the high-level mpi4py gather
sendcounts = np.array(comm.gather(len(morb_grids), 0))

if rank == 0:
    print("sendcounts: {}, total: {}".format(sendcounts, sum(sendcounts)))
    recvbuf = np.empty(sum(sendcounts), dtype=float)
else:
    recvbuf = None

comm.Gatherv(sendbuf=morb_grids, recvbuf=(recvbuf, sendcounts), root=0)

if rank == 0:
    time0 = time.time()
    elim = np.array([args.emin, args.emax])
    all_morb_grids = recvbuf.reshape(num_morbs,
            eval_reg_size_n[0], eval_reg_size_n[1], eval_reg_size_n[2])
    np.savez(args.output_file,
        morb_grids=all_morb_grids,
        morb_energies=morb_energies,
        dv=dv,
        z_arr=z_arr,
        elim=elim,
        ref_energy=ref_energy)
    print("Saved the orbitals to file: %.2fs" % (time.time() - time0))
