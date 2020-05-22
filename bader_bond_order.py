#!/usr/bin/env python
import os
import numpy as np
import time
import copy
import sys

import argparse

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

import cp2k_spm_tools.cp2k_grid_orbitals as cgo
from cp2k_spm_tools import common, cube

from mpi4py import MPI

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

parser = argparse.ArgumentParser(
    description='Runs bond order analysis based on Bader basins.')

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
    help='cp2k restart file containing the wavefunction.')
### -----------------------------------------------------------
parser.add_argument(
    '--output_file',
    metavar='FILENAME',
    required=True,
    help='Output file containing the bond orders.')
parser.add_argument(
    '--bader_basins_dir',
    metavar='DIR',
    required=True,
    help='directory containing the Bader basin .cube files.')
### -----------------------------------------------------------
parser.add_argument(
    '--dx',
    type=float,
    metavar='DX',
    default=0.2,
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
    '--eval_region',
    type=str,
    nargs=6,
    metavar='X',
    required=False,
    default = ['G', 'G', 'G', 'G', 'G', 'G'],
    help=common.eval_region_description
)
### -----------------------------------------------------------


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
### Load the Bader basins
### ------------------------------------------------------

bader_atoms = []
bader_masks = []

for f in sorted(os.listdir(args.bader_basins_dir)):
    if f.startswith("BvAt"):
        num = int(f.split(".")[0][4:]) - 1
        bader_atoms.append(num)
        c = cube.Cube()
        c.read_cube_file(args.bader_basins_dir+"/"+f)
        if np.abs(c.dv[0, 0] - args.dx) > 1e-3:
            print("ERROR: Basin cube dx doesn't match specified dx!")
            print(c.dv[0, 0], args.dx)
            exit(0)
        bader_masks.append(c.data > 1e-10)


print("R%d/%d: loaded Bader basins, time: %.2fs"%(mpi_rank, mpi_size, (time.time() - time0)))
sys.stdout.flush()
time1 = time.time()

### ------------------------------------------------------
### Evaluate orbitals on the real-space grid
### ------------------------------------------------------

mol_grid_orb = cgo.Cp2kGridOrbitals(mpi_rank, mpi_size, comm, single_precision=False)
mol_grid_orb.read_cp2k_input(args.cp2k_input_file)
mol_grid_orb.read_xyz(args.xyz_file)
mol_grid_orb.center_atoms_to_cell()
mol_grid_orb.read_basis_functions(args.basis_set_file)
mol_grid_orb.load_restart_wfn_file(args.wfn_file, n_occ=None, n_virt=0)

print("R%d/%d: loaded eval files, time: %.2fs"%(mpi_rank, mpi_size, (time.time() - time1)))
sys.stdout.flush()
time1 = time.time()

eval_reg = common.parse_eval_region_input(args.eval_region, mol_grid_orb.ase_atoms, mol_grid_orb.cell)

mol_grid_orb.calc_morbs_in_region(args.dx,
                                x_eval_region = eval_reg[0],
                                y_eval_region = eval_reg[1],
                                z_eval_region = eval_reg[2],
                                reserve_extrap = 0.0,
                                eval_cutoff = args.eval_cutoff)

print("R%d/%d: evaluated grids, time: %.2fs"%(mpi_rank, mpi_size, (time.time() - time1)))
sys.stdout.flush()
time1 = time.time()

### ------------------------------------------------------
### Calculate Bond orders
### ------------------------------------------------------

bond_order_matrix = np.zeros((len(bader_atoms), len(bader_atoms)))

n_orb_per_rank = []
for i_spin in range(mol_grid_orb.nspin):
    n_orb_per_rank.append(comm.allgather(len(mol_grid_orb.morb_energies[i_spin])))

cell_n = mol_grid_orb.eval_cell_n
vol_elem = np.prod(mol_grid_orb.dv)

if any(cell_n != bader_masks[0].shape):
    print("Error: Basin and evaluation size mismatch.")
    exit(0)

for i_rank in range(mpi_size):

    if mpi_rank == i_rank:
        print("R%d/%d: distributing grids and evaluating products..."%(mpi_rank, mpi_size))
        sys.stdout.flush()
        time1 = time.time()


    for i_spin in range(mol_grid_orb.nspin):

        bcast_buffer = np.empty(np.prod(cell_n)*n_orb_per_rank[i_spin][i_rank])

        if mpi_rank == i_rank:
            bcast_buffer = mol_grid_orb.morb_grids[i_spin].flatten()

        # Broadcast the current rank grids to all
        comm.Bcast([bcast_buffer, MPI.DOUBLE], root=i_rank)

        received_grids = np.reshape(bcast_buffer,
            (n_orb_per_rank[i_spin][i_rank], cell_n[0], cell_n[1], cell_n[2])
        )
        
#        for i_mo in range(received_grids.shape[0]):
#            
#            i_grid = received_grids[i_mo]
#
#            for j_mo in range(mol_grid_orb.morb_grids[i_spin].shape[0]):
#                
#                j_grid = mol_grid_orb.morb_grids[i_spin][j_mo]
#
#                for at_a in range(len(bader_atoms)):
#                    for at_b in range(at_a):
#
#                        i_grid_a = i_grid*bader_masks[at_a]
#                        i_grid_b = i_grid*bader_masks[at_b]
#
#                        scalar_a = np.dot(i_grid_a.flatten(), j_grid.flatten())*vol_elem
#                        scalar_b = np.dot(i_grid_b.flatten(), j_grid.flatten())*vol_elem
#
#                        bond_order_matrix[at_a, at_b] += 4*scalar_a*scalar_b
#                        bond_order_matrix[at_b, at_a] += 4*scalar_a*scalar_b

        n_i = received_grids.shape[0]
        n_j = mol_grid_orb.morb_grids[i_spin].shape[0]

        for at_a in range(len(bader_atoms)):
            for at_b in range(at_a):
                
                i_grid_a = received_grids[:, bader_masks[at_a]].reshape(n_i, -1)
                i_grid_b = received_grids[:, bader_masks[at_b]].reshape(n_i, -1)

                j_grid_a = mol_grid_orb.morb_grids[i_spin][:, bader_masks[at_a]].reshape(n_j, -1)
                j_grid_b = mol_grid_orb.morb_grids[i_spin][:, bader_masks[at_b]].reshape(n_j, -1)

                bo = np.sum(np.einsum("ij,kj", i_grid_a, j_grid_a) * np.einsum("ij,kj", i_grid_b, j_grid_b))

                bond_order_matrix[at_a, at_b] += 4 * bo * vol_elem**2
                bond_order_matrix[at_b, at_a] += 4 * bo * vol_elem**2

    
    if mpi_rank == i_rank:
        print("R%d/%d: ... time: %.2fs"%((mpi_rank, mpi_size, time.time()-time1)))
        sys.stdout.flush()

# collect all contributions
final_bond_order_mat = np.zeros((len(bader_atoms), len(bader_atoms)))
comm.Reduce(bond_order_matrix, final_bond_order_mat, op=MPI.SUM)

if mpi_rank == 0:
    header = ""
    for b_at in bader_atoms:
        header += "%10d" % b_at
    header = header[3:]
    np.savetxt(args.output_file, final_bond_order_mat, fmt="%9.6f", header=header)

print("R%d/%d finished, total time: %.2fs"%(mpi_rank, mpi_size, (time.time() - time0)))
