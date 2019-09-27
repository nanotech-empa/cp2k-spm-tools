#!/usr/bin/env python

# @author  Hillebrand, Fabian
# @date    2019
# @version 2.1.0

import argparse # Terminal arguments
import os
import sys      # System access
import time     # Timing
import resource # Memory usage

# Python3 at least
assert sys.version_info >= (3, 0)

import numpy as np
import sysconfig as sc

ang2bohr   = 1.88972612546
ev2hartree = 0.03674930814

# Own includes
import atomistic_tools.cp2k_stm_sts as css
from atomistic_tools import common
from atomistic_tools.cube import Cube

import hrstm_tools.tip_coeffs as tc
import hrstm_tools.cp2k_grid_matrix as cgm
import hrstm_tools.hrstm_utils as hu
import hrstm_tools.hrstm as hs

import atomistic_tools.cp2k_grid_orbitals as cgo
#import hrstm_tools.ppstm_grid_orbitals as pgo

from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()


parser = argparse.ArgumentParser(description="HR-STM for CP2K based on Chen's"
    + " Derivative Rule and the Probe Particle Model.")

### ----------------------------------------------------------------------------
### Output file
parser.add_argument('--output',
    type=str,
    metavar="OUTPUT",
    default="hrstm",
    required=False,
    help="Name for output file. File extension added automatically.")

### ----------------------------------------------------------------------------
### Input files for sample
parser.add_argument('--cp2k_input_file',
    metavar='FILE',
    required=True,
    help="CP2K input file used for the sample calculation.")

parser.add_argument('--basis_set_file',
    metavar='FILE',
    required=True,
    help="File containing all basis sets used for sample calculation.")

parser.add_argument('--xyz_file',
    metavar='FILE',
    required=True,
    help="File containing the atom positions for the sample.")

parser.add_argument('--wfn_file',
    metavar='FILE',
    required=True,
    help="Restart file containing the basis coefficients for the sample.")

parser.add_argument('--hartree_file',
    metavar='FILE',
    required=True,
    help="Cube file with Hartree potential of sample.")

parser.add_argument('--fwhm_sam',
    type=float,
    metavar='eV',
    default=0.05,
    required=False,
    help="Full width at half maximum for Gaussian broadening of DOS for sample.")

parser.add_argument('--wn',
    type=float,
    metavar='eV',
    default=5.0,
    required=False,
    help="Work function for the sample. Used for the decay constant.")

### ----------------------------------------------------------------------------
### Energy range for sample
parser.add_argument('--emin',
    type=float,
    metavar='eV',
    default=-2.0,
    required=False,
    help="Lower energy used for cut-off with respect to Fermi energy for sample.")

parser.add_argument('--emax',
    type=float,
    metavar='eV',
    default=2.0,
    required=False,
    help="Upper energy used for cut-off with respect to Fermi energy for sample.")

### ----------------------------------------------------------------------------
### Parameters for putting sample orbitals on grid
parser.add_argument('--dx_wfn',
    type=float,
    metavar='Ang',
    default=0.2,
    required=False,
    help="Spacing for wavefunction grid used by interpolation.")

parser.add_argument('--rcut',
    type=float,
    metavar='Ang',
    default=15.0,
    required=False,
    help="Cut-off radius used when computing sample wavefunction.")

parser.add_argument('--extrap_dist',
    type=float,
    metavar='Ang',
    default=0.5,
    required=False,
    help="Starting distance from highest atom where extrapolation is used from.")

### ----------------------------------------------------------------------------
### Parameters for tip
parser.add_argument('--tip_pos_files',
    metavar='FILE',
    nargs='+',
    required=False,
    help="File paths to positions of probe particles (without '_<x,y,z>.npy').")

parser.add_argument('--tip_shift',
    type=float,
    metavar='Ang',
    required=False,
    help="z-distance shift for metal tip with respect to apex probe particle."
    + " Only used with relaxed positions.")

parser.add_argument('--eval_region',
    type=float,
    metavar='Ang',
    nargs='+',
    required=False,
    help="Tip positions in case of non-relaxed positions.")

parser.add_argument('--dx_tip',
    type=float,
    metavar='Ang',
    required=False,
    help="Spacing for a uniform grid for non-relaxed positions.")

parser.add_argument('--pdos_list',
    metavar='FILE',
    nargs='+',
    required=True,
    help="List of PDOS files for the different tip apexes used as tip"
    + " coefficients. Or, alternatively, four numbers corresponding to"
    + " [s py pz px] for uniform PDOS values (constant DOS).")

parser.add_argument('--orbs_tip',
    type=int,
    metavar="l",
    default=1,
    required=False,
    help="Integer indicating which orbitals of the tip are used following the"
    + " angular momentum quantum number.")

parser.add_argument('--fwhm_tip',
    type=float,
    metavar='eV',
    default=0.0,
    required=False,
    help="Full width at half maximum for Gaussian broadening of DOS for tip.")

parser.add_argument('--rotate',
    action="store_true",
    default=False,
    required=False,
    help="If set, tip coefficients will be rotated.")

### ----------------------------------------------------------------------------
### Parameters for HR-STM images
parser.add_argument('--voltages',
    type=float,
    metavar='eV',
    nargs='+',
    required=True,
    help="Voltages used for STM.")

### ----------------------------------------------------------------------------
### Parse args one one rank and broadcast it
### ----------------------------------------------------------------------------

args = None
args_success = False
try:
    if mpi_rank == 0:
        args = parser.parse_args()
        args_success = True
finally:
    args_success = mpi_comm.bcast(args_success, root=0)
if not args_success:
  raise ImportError("Failed to read input arguments.")

args = mpi_comm.bcast(args, root=0)


### ----------------------------------------------------------------------------
### Read tip positions on ranks
### ----------------------------------------------------------------------------

start = time.time()
if args.tip_pos_files is not None and args.tip_shift is not None:
    pos_local, dim_pos, eval_region_wfn, lVec = hu.read_tip_positions(
        args.tip_pos_files, args.tip_shift, args.dx_wfn, mpi_rank, mpi_size, 
        mpi_comm)
elif args.eval_region is not None and args.dx_tip is not None \
    and not args.rotate:
    pos_local, dim_pos, eval_region_wfn, lVec = hu.create_tip_positions(
        args.eval_region, args.dx_tip, mpi_rank, mpi_size, mpi_comm)
else:
    raise ImportError("Missing tip positions.") 
end = time.time()
print("Reading tip positions in {} seconds for rank {}.".format(end-start,
    mpi_rank))

### ----------------------------------------------------------------------------
### Setup tip coefficients
### ----------------------------------------------------------------------------

start = time.time()
tip_coeffs = tc.TipCoefficients(mpi_rank, mpi_size, mpi_comm)
tip_coeffs.read_coefficients(args.orbs_tip, args.pdos_list, 
    min(-max(args.voltages),0)-4.0*args.fwhm_tip,
    max(-min(args.voltages),0)+4.0*args.fwhm_tip)
tip_coeffs.initialize(pos_local, args.rotate)
end = time.time()
print("Reading tip coefficients in {} seconds for rank {}.".format(end-start,
    mpi_rank))

### ----------------------------------------------------------------------------
###  Evaluate sample orbitals
### ----------------------------------------------------------------------------

start = time.time()
wfn_grid_orb = cgo.Cp2kGridOrbitals(mpi_rank, mpi_size, mpi_comm,
    single_precision=False)
#wfn_grid_orb = pgo.PPSTMGridOrbitals(mpi_rank, mpi_size, mpi_comm,
#    single_precision=False)
wfn_grid_orb.read_cp2k_input(args.cp2k_input_file)
wfn_grid_orb.read_xyz(args.xyz_file)
wfn_grid_orb.read_basis_functions(args.basis_set_file)
wfn_grid_orb.load_restart_wfn_file(args.wfn_file,
    emin=args.emin-4.0*args.fwhm_sam, emax=args.emax+4.0*args.fwhm_sam)
wfn_grid_orb.calc_morbs_in_region(args.dx_wfn,
    z_eval_region=eval_region_wfn[2]*ang2bohr,
# Hack for PPSTM: actually used for workfunction
#    reserve_extrap = args.wn,
    reserve_extrap = args.extrap_dist,
    eval_cutoff = args.rcut*ang2bohr)
end = time.time()
print("Building CP2K wave function matrix in {} seconds for rank {}.".format(
  end-start, mpi_rank))

### ----------------------------------------------------------------------------
### Extrapolate orbitals
### ----------------------------------------------------------------------------

start = time.time()
hart_cube = Cube()
hart_cube.read_cube_file(args.hartree_file)
extrap_plane_z = eval_region_wfn[2][1] \
    - np.max(wfn_grid_orb.ase_atoms.positions[:,2])
hart_plane = hart_cube.get_plane_above_topmost_atom(extrap_plane_z) \
  - wfn_grid_orb.ref_energy*ev2hartree
del hart_cube, extrap_plane_z
wfn_grid_orb.extrapolate_morbs(hart_plane=hart_plane)
end = time.time()
print("Extrapolating CP2K wave function matrix in {} seconds for rank {}."
  .format(end-start, mpi_rank))

### ----------------------------------------------------------------------------
### Divide grid orbitals along space and put in wrapper
### ----------------------------------------------------------------------------

start = time.time()
wfn_grid_matrix = cgm.Cp2kGridMatrix(wfn_grid_orb, eval_region_wfn, 
    pos_local[1:], args.orbs_tip, args.wn, mpi_rank, mpi_size, mpi_comm)
del wfn_grid_orb, pos_local
wfn_grid_matrix.divide()
end = time.time()
print("Setting up wave function object in {} seconds for rank {}."
    .format(end-start, mpi_rank))

### ----------------------------------------------------------------------------
### Evaluate HRSTM and write output
### ----------------------------------------------------------------------------

# Meta information
if mpi_rank == 0:
    meta = {'dimGrid' : dim_pos,
            'lVec' : lVec,
            'voltages' : args.voltages,
            'input' : args}
    if os.path.dirname(args.output) != '':
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output+"_meta.npy", meta)
start = time.time()
hrstm = hs.Hrstm(tip_coeffs, dim_pos, wfn_grid_matrix, args.fwhm_sam,
    args.fwhm_tip, mpi_rank, mpi_size, mpi_comm)
hrstm.run(args.voltages)
hrstm.write_compressed(args.output)
end = time.time()
print("Evaluating HRSTM-run method in {} seconds for rank {}.".format(end-start, 
    mpi_rank))
