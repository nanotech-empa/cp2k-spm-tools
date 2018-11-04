#!/usr/bin/env python
import os
import numpy as np
import time
import copy
import sys

import argparse

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

import atomistic_tools.cp2k_grid_orbitals as cgo
import atomistic_tools.cp2k_stm_sts as css
from atomistic_tools import common
from atomistic_tools.cube import Cube

from mpi4py import MPI

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

parser = argparse.ArgumentParser(
    description='Puts the CP2K orbitals on grid and calculates STM.')
# ----------------------------------
# First system: molecule on slab
parser.add_argument(
    '--cp2k_input_file1',
    metavar='FILENAME',
    required=True,
    help='CP2K input of the SCF calculation.')
parser.add_argument(
    '--basis_set_file1',
    metavar='FILENAME',
    required=True,
    help='File containing the used basis sets.')
parser.add_argument(
    '--xyz_file1',
    metavar='FILENAME',
    required=True,
    help='.xyz file containing the geometry.')
parser.add_argument(
    '--wfn_file1',
    metavar='FILENAME',
    required=True,
    help='Restart file containing the final wavefunction.')
parser.add_argument(
    '--emin1',
    type=float,
    metavar='E',
    required=True,
    help='Lowest energy value for selecting orbitals (eV).')
parser.add_argument(
    '--emax1',
    type=float,
    metavar='E',
    required=True,
    help='Highest energy value for selecting orbitals (eV).')

# ----------------------------------
# Second system: only molecule
parser.add_argument(
    '--cp2k_input_file2',
    metavar='FILENAME',
    required=True,
    help='CP2K input of the SCF calculation.')
parser.add_argument(
    '--basis_set_file2',
    metavar='FILENAME',
    required=True,
    help='File containing the used basis sets.')
parser.add_argument(
    '--xyz_file2',
    metavar='FILENAME',
    required=True,
    help='.xyz file containing the geometry.')
parser.add_argument(
    '--wfn_file2',
    metavar='FILENAME',
    required=True,
    help='Restart file containing the final wavefunction.')
parser.add_argument(
    '--nhomo2',
    type=float,
    metavar='E',
    required=True,
    help='Number of homo orbitals.')
parser.add_argument(
    '--nlumo2',
    type=float,
    metavar='E',
    required=True,
    help='Number of lumo orbitals.')
# ----------------------------------

parser.add_argument(
    '--output_file',
    metavar='FILENAME',
    required=True,
    help='File, where to save the output')
parser.add_argument(
    '--eval_region',
    type=str,
    nargs=6,
    metavar='X',
    required=True,
    help=common.eval_region_description
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
    default=14.0,
    help=("Size of the region around the atom where each"
          " orbital is evaluated (only used for 'G' region).")
)

time0 = time.time()

### ------------------------------------------------------
### Parse args for only one rank to suppress duplicate stdio
### ------------------------------------------------------

args = None
args_success = False
try:
    if mpi_rank == 0:
        args = parser.parse_args()
        args_success = True
finally:
    args_success = comm.bcast(args_success, root=0)

if not args_success:
    print(mpi_rank, "exiting")
    exit(0)

args = comm.bcast(args, root=0)

### ------------------------------------------------------
### Evaluate the same molecule orbitals on all mpi ranks
### ------------------------------------------------------

mol_grid_orb = cgo.Cp2kGridOrbitals(0, 1, single_precision=False)
mol_grid_orb.read_cp2k_input(args.cp2k_input_file2)
mol_grid_orb.read_xyz(args.xyz_file2)
mol_grid_orb.read_basis_functions(args.basis_set_file2)
mol_grid_orb.load_restart_wfn_file(args.wfn_file2, n_homo=args.nhomo2, n_lumo=args.nlumo2)

print("R%d/%d: loaded G2, %.2fs"%(mpi_rank, mpi_size, (time.time() - time0)))
sys.stdout.flush()
time1 = time.time()

eval_reg = common.parse_eval_region_input(args.eval_region, mol_grid_orb.ase_atoms, mol_grid_orb.cell)

mol_grid_orb.calc_morbs_in_region(args.dx,
                                x_eval_region = eval_reg[0],
                                y_eval_region = eval_reg[1],
                                z_eval_region = eval_reg[2],
                                reserve_extrap = 0.0,
                                eval_cutoff = args.eval_cutoff)

print("R%d/%d: evaluated G2, %.2fs"%(mpi_rank, mpi_size, (time.time() - time1)))
sys.stdout.flush()
time1 = time.time()

### ------------------------------------------------------
### Evaluate slab system orbitals
### ------------------------------------------------------

slab_grid_orb = cgo.Cp2kGridOrbitals(mpi_rank, mpi_size, mpi_comm=comm, single_precision=False)
slab_grid_orb.read_cp2k_input(args.cp2k_input_file1)
slab_grid_orb.read_xyz(args.xyz_file1)
slab_grid_orb.read_basis_functions(args.basis_set_file1)
slab_grid_orb.load_restart_wfn_file(args.wfn_file1, emin=args.emin1, emax=args.emax1)

print("R%d/%d: loaded G1, %.2fs"%(mpi_rank, mpi_size, (time.time() - time1)))
sys.stdout.flush()
time1 = time.time()

slab_grid_orb.calc_morbs_in_region(args.dx,
                                x_eval_region = eval_reg[0],
                                y_eval_region = eval_reg[1],
                                z_eval_region = eval_reg[2],
                                reserve_extrap = 0.0,
                                eval_cutoff = args.eval_cutoff)

print("R%d/%d: evaluated G1, %.2fs"%(mpi_rank, mpi_size, (time.time() - time1)))
sys.stdout.flush()
time1 = time.time()

### ------------------------------------------------------
### calculate overlap
### ------------------------------------------------------

ve = np.prod(slab_grid_orb.dv)

overlap_matrix =  (np.einsum('iklm, jklm',
    slab_grid_orb.morb_grids[0],
    mol_grid_orb.morb_grids[0])*ve)**2

print("R%d/%d: overlap finished, %.2fs"%(mpi_rank, mpi_size, (time.time()-time1) ))
sys.stdout.flush()

overlap_matrix_rav = overlap_matrix.ravel()
sendcounts = np.array(comm.gather(len(overlap_matrix_rav), 0))

if mpi_rank == 0:
    print("sendcounts: {}, total: {}".format(sendcounts, sum(sendcounts)))
    recvbuf = np.empty(sum(sendcounts), dtype=float)
else:
    recvbuf = None

comm.Gatherv(sendbuf=overlap_matrix_rav, recvbuf=[recvbuf, sendcounts], root=0)

slab_grid_orb.gather_global_energies()

if mpi_rank == 0:
    energies_g1 = slab_grid_orb.global_morb_energies[0]
    energies_g2 = mol_grid_orb.morb_energies[0]
    homo_g2 = mol_grid_orb.homo_inds[0][0]

    overlap_matrix_collected = recvbuf.reshape(
        len(energies_g1),
        len(energies_g2))
        
    np.savez(args.output_file,
        overlap_matrix=overlap_matrix_collected,
        en_grp1=energies_g1,
        en_grp2=energies_g2,
        homo_grp2=homo_g2)
    print("Finish! Total time: %.2fs" % (time.time() - time0))

