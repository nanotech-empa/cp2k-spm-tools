#!/usr/bin/env python
import argparse
import sys
import time

import numpy as np
from mpi4py import MPI

import cp2k_spm_tools.cp2k_grid_orbitals as cgo
from cp2k_spm_tools import common

ang_2_bohr = 1.0 / 0.52917721067
hart_2_ev = 27.21138602

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

parser = argparse.ArgumentParser(description="Puts the CP2K orbitals on grid and calculates scalar products.")
# ----------------------------------
# First system: molecule on slab
parser.add_argument("--cp2k_input_file1", metavar="FILENAME", required=True, help="CP2K input of the SCF calculation.")
parser.add_argument("--basis_set_file1", metavar="FILENAME", required=True, help="File containing the used basis sets.")
parser.add_argument("--xyz_file1", metavar="FILENAME", required=True, help=".xyz file containing the geometry.")
parser.add_argument(
    "--wfn_file1", metavar="FILENAME", required=True, help="Restart file containing the final wavefunction."
)
parser.add_argument(
    "--emin1", type=float, metavar="E", required=True, help="Lowest energy value for selecting orbitals (eV)."
)
parser.add_argument(
    "--emax1", type=float, metavar="E", required=True, help="Highest energy value for selecting orbitals (eV)."
)

# ----------------------------------
# Second system: only molecule
parser.add_argument("--cp2k_input_file2", metavar="FILENAME", required=True, help="CP2K input of the SCF calculation.")
parser.add_argument("--basis_set_file2", metavar="FILENAME", required=True, help="File containing the used basis sets.")
parser.add_argument("--xyz_file2", metavar="FILENAME", required=True, help=".xyz file containing the geometry.")
parser.add_argument(
    "--wfn_file2", metavar="FILENAME", required=True, help="Restart file containing the final wavefunction."
)
parser.add_argument("--nhomo2", type=int, metavar="N", required=True, help="Number of homo orbitals.")
parser.add_argument("--nlumo2", type=int, metavar="N", required=True, help="Number of lumo orbitals.")
# ----------------------------------

parser.add_argument("--output_file", metavar="FILENAME", required=True, help="File, where to save the output")
parser.add_argument("--eval_region", type=str, nargs=6, metavar="X", required=True, help=common.eval_region_description)
parser.add_argument("--dx", type=float, metavar="DX", required=True, help="Spatial step for the grid (angstroms).")
parser.add_argument(
    "--eval_cutoff",
    type=float,
    metavar="D",
    default=14.0,
    help=("Size of the region around the atom where each orbital is evaluated (only used for 'G' region)."),
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
mol_grid_orb.load_restart_wfn_file(args.wfn_file2, n_occ=args.nhomo2, n_virt=args.nlumo2)

print("R%d/%d: loaded G2, %.2fs" % (mpi_rank, mpi_size, (time.time() - time0)))
sys.stdout.flush()
time1 = time.time()

eval_reg = common.parse_eval_region_input(args.eval_region, mol_grid_orb.ase_atoms, mol_grid_orb.cell)

mol_grid_orb.calc_morbs_in_region(
    args.dx,
    x_eval_region=eval_reg[0],
    y_eval_region=eval_reg[1],
    z_eval_region=eval_reg[2],
    reserve_extrap=0.0,
    eval_cutoff=args.eval_cutoff,
)

print("R%d/%d: evaluated G2, %.2fs" % (mpi_rank, mpi_size, (time.time() - time1)))
sys.stdout.flush()
time1 = time.time()

### ------------------------------------------------------
### Evaluate slab system orbitals
### ------------------------------------------------------

slab_grid_orb = cgo.Cp2kGridOrbitals(mpi_rank, mpi_size, mpi_comm=comm, single_precision=False)
slab_grid_orb.read_cp2k_input(args.cp2k_input_file1)
slab_grid_orb.read_xyz(args.xyz_file1)
slab_grid_orb.read_basis_functions(args.basis_set_file1)
slab_grid_orb.load_restart_wfn_file(args.wfn_file1, emin=args.emin1 - 0.05, emax=args.emax1 + 0.05)

print("R%d/%d: loaded G1, %.2fs" % (mpi_rank, mpi_size, (time.time() - time1)))
sys.stdout.flush()
time1 = time.time()

slab_grid_orb.calc_morbs_in_region(
    args.dx,
    x_eval_region=eval_reg[0],
    y_eval_region=eval_reg[1],
    z_eval_region=eval_reg[2],
    reserve_extrap=0.0,
    eval_cutoff=args.eval_cutoff,
)

print("R%d/%d: evaluated G1, %.2fs" % (mpi_rank, mpi_size, (time.time() - time1)))
sys.stdout.flush()
time1 = time.time()

### ------------------------------------------------------
### calculate overlap
### ------------------------------------------------------

ve = np.prod(slab_grid_orb.dv)

output_dict = {}

for i_spin_slab in range(slab_grid_orb.nspin):
    for i_spin_mol in range(mol_grid_orb.nspin):
        # The gas phase orbitals can be expressed in the basis of slab orbitals
        # |phi_i> = \sum_j <psi_j|phi_i> |psi_j>
        # And the modulus is
        # <phi_i|phi_i> = \sum_j |<psi_j|phi_i>|^2 = 1
        # Therefore, the matrix of
        # |<phi_i|psi_j>|^2
        # is a good description of the amount of gas phase orbitals in slab orbitals
        # (positive; integral between j1 to j2 gives the amount of |phi_i> in that region)
        overlap_matrix = (
            np.einsum("iklm, jklm", slab_grid_orb.morb_grids[i_spin_slab], mol_grid_orb.morb_grids[i_spin_mol]) * ve
        ) ** 2

        print("R%d/%d: overlap finished, %.2fs" % (mpi_rank, mpi_size, (time.time() - time1)))
        sys.stdout.flush()

        overlap_matrix_rav = overlap_matrix.ravel()
        sendcounts = np.array(comm.gather(len(overlap_matrix_rav), 0))

        if mpi_rank == 0:
            print("sendcounts: {}, total: {}".format(sendcounts, sum(sendcounts)))
            recvbuf = np.empty(sum(sendcounts), dtype=float)
        else:
            recvbuf = None

        comm.Gatherv(sendbuf=overlap_matrix_rav, recvbuf=[recvbuf, sendcounts], root=0)

        if mpi_rank == 0:
            overlap_matrix_collected = recvbuf.reshape(
                (
                    len(slab_grid_orb.global_morb_energies[i_spin_slab]),
                    len(mol_grid_orb.global_morb_energies[i_spin_mol]),
                )
            )
            output_dict["overlap_matrix_s{}s{}".format(i_spin_slab, i_spin_mol)] = overlap_matrix_collected

if mpi_rank == 0:
    output_dict["metadata"] = [
        {
            "nspin_g1": slab_grid_orb.nspin,
            "nspin_g2": mol_grid_orb.nspin,
            "homo_i_g2": mol_grid_orb.i_homo_loc,
        }
    ]

    for i_spin_slab in range(slab_grid_orb.nspin):
        output_dict["energies_g1_s{}".format(i_spin_slab)] = slab_grid_orb.global_morb_energies[i_spin_slab]
    for i_spin_mol in range(mol_grid_orb.nspin):
        output_dict["energies_g2_s{}".format(i_spin_mol)] = mol_grid_orb.global_morb_energies[i_spin_mol]
        # NB: Count starts from 1!
        output_dict["orb_indexes_g2_s{}".format(i_spin_mol)] = mol_grid_orb.cwf.global_morb_indexes[i_spin_mol]

    np.savez(args.output_file, **output_dict)
    print("Finish! Total time: %.2fs" % (time.time() - time0))
