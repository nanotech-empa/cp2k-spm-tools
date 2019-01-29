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

parser.add_argument(
    '--cp2k_input_file',
    metavar='FILENAME',
    required=True,
    help='CP2K input of the SCF calculation.')
parser.add_argument(
    '--basis_set_file',
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
    '--hartree_file',
    metavar='FILENAME',
    required=True,
    help='Cube file containing the hartree potential.')

parser.add_argument(
    '--output_file',
    metavar='FILENAME',
    required=True,
    help='File, where to save the output')

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
parser.add_argument(
    '--extrap_extent',
    type=float,
    metavar='H',
    default=5.0,
    required=True,
    help="The extent of the extrapolation region. (angstrom)")

parser.add_argument(
    '--heights',
    nargs='*',
    type=float,
    metavar='H',
    help="List of heights for constant height STM pictures (wrt topmost atom).")
parser.add_argument(
    '--isovalues',
    nargs='*',
    type=float,
    metavar='C',
    help="List of charge density isovalues for constant current STM pictures.")
parser.add_argument(
    '--de',
    type=float,
    default=0.05,
    help="Energy discretization for STS. (eV)")
parser.add_argument(
    '--fwhm',
    type=float,
    default=0.1,
    help="Full width at half maximum for STS gaussian broadening. (eV)")

parser.add_argument(
    '--export_n_orbitals',
    type=int,
    metavar='N',
    default=0,
    help="Number of HOMO and LUMO orbitals to export at '--heights'.")

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
### Evaluate orbitals on the real-space grid
### ------------------------------------------------------

cp2k_grid_orb = cgo.Cp2kGridOrbitals(mpi_rank, mpi_size, mpi_comm=comm, single_precision=True)
cp2k_grid_orb.read_cp2k_input(args.cp2k_input_file)
cp2k_grid_orb.read_xyz(args.xyz_file)
cp2k_grid_orb.center_atoms_to_cell()
cp2k_grid_orb.read_basis_functions(args.basis_set_file)
cp2k_grid_orb.load_restart_wfn_file(args.wfn_file, emin=args.emin-2.0*args.fwhm, emax=args.emax+2.0*args.fwhm)

print("R%d/%d: loaded wfn, %.2fs"%(mpi_rank, mpi_size, (time.time() - time0)))
sys.stdout.flush()
time1 = time.time()

eval_reg = common.parse_eval_region_input(args.eval_region, cp2k_grid_orb.ase_atoms, cp2k_grid_orb.cell)

cp2k_grid_orb.calc_morbs_in_region(args.dx,
                                x_eval_region = eval_reg[0],
                                y_eval_region = eval_reg[1],
                                z_eval_region = eval_reg[2],
                                reserve_extrap = args.extrap_extent,
                                eval_cutoff = args.eval_cutoff)

print("R%d/%d: evaluated wfn, %.2fs"%(mpi_rank, mpi_size, (time.time() - time1)))
sys.stdout.flush()
time1 = time.time()

### ------------------------------------------------------
### Extrapolate orbitals
### ------------------------------------------------------

hart_cube = Cube()
hart_cube.read_cube_file(args.hartree_file)
extrap_plane_z = eval_reg[2][1] / ang_2_bohr - np.max(cp2k_grid_orb.ase_atoms.positions[:, 2])
hart_plane = hart_cube.get_plane_above_topmost_atom(extrap_plane_z) - cp2k_grid_orb.ref_energy/hart_2_ev

cp2k_grid_orb.extrapolate_morbs(hart_plane=hart_plane)

print("R%d/%d: extrapolated wfn, %.2fs"%(mpi_rank, mpi_size, (time.time() - time1)))
sys.stdout.flush()
time1 = time.time()

### ------------------------------------------------------
### Export orbitals
### ------------------------------------------------------

if args.export_n_orbitals > 0:
    orbital_list = list(range(-args.export_n_orbitals + 1, args.export_n_orbitals + 1))
    cp2k_grid_orb.collect_and_save_ch_orbitals(orbital_list, args.heights)

### ------------------------------------------------------
### Run STM-STS analysis
### ------------------------------------------------------

stm = css.STM(mpi_comm = comm, cp2k_grid_orb = cp2k_grid_orb)

stm.gather_global_energies()
stm.divide_by_space()

stm.calculate_maps(args.isovalues, args.heights, args.emin, args.emax, args.de, args.fwhm)

stm.collect_and_save_maps(path=args.output_file)

print("R%d/%d: finished, total time: %.2fs"%(mpi_rank, mpi_size, (time.time() - time0)))
