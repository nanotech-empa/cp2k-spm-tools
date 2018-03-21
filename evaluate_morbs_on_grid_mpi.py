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
    '--x_extra',
    type=float,
    metavar='L',
    required=True,
    help='extra distance in x for evaluation region in addition to atom extent (angstroms).')
parser.add_argument(
    '--y_extra',
    type=float,
    metavar='L',
    required=True,
    help='extra distance in y for evaluation region in addition to atom extent (angstroms).')
parser.add_argument(
    '--z_extra',
    type=float,
    metavar='L',
    required=True,
    help='extra distance in z for evaluation region in addition to atom extent (angstroms).')

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
    help='Size of the region (in x and y) around the atom where each orbital is located.')


# Define all variables that must be later broadcasted
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
        if fermi != None:
            print("Fermi energy: %.6f" % fermi)
        print("Read cp2k out: %.3f" % (time.time()-time1))

        time1 = time.time()
        at_positions, at_elems = cu.read_atoms(args.xyz_file)
        cu.center_atoms_to_cell(at_positions, cell)
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


### -----------------------------------------
### Divide the molecular orbitals fairly between processors
### -----------------------------------------

if rank == 0:

    num_morbs = len(morb_composition[0][0][0][0])
    morbs_per_rank = num_morbs//size
    extra_morbs = num_morbs % size

    for i_rank in range(size):
        ind_start = i_rank*morbs_per_rank
        if i_rank < extra_morbs:
            ind_start += i_rank
            ind_end = ind_start + morbs_per_rank + 1
        else:
            ind_start += extra_morbs
            ind_end = ind_start + morbs_per_rank
        print("Rank %d works with orbitals %d:%d" %(i_rank, ind_start, ind_end))

        morb_comp_send = copy.deepcopy(morb_composition)

        for iatom in range(len(morb_comp_send)):
            for iset in range(len(morb_comp_send[iatom])):
                for ishell in range(len(morb_comp_send[iatom][iset])):
                    for iorb in range(len(morb_comp_send[iatom][iset][ishell])):
                        morb_comp_send[iatom][iset][ishell][iorb] = \
                            morb_comp_send[iatom][iset][ishell][iorb][ind_start:ind_end]
        if i_rank != 0:
            comm.send(morb_comp_send, dest=i_rank)
        else:
            morb_comp_scattered = copy.deepcopy(morb_comp_send)
    # Release memory
    morb_composition = 0
    morb_comp_send = 0
else:
    morb_comp_scattered = comm.recv(source=0)


if rank == 0:
    print("Initial broadcast time %.4f s" % (time.time() - time1))

### -----------------------------------------
### Define morb evaluation region
### -----------------------------------------

x_min = np.min(at_positions[:, 0]) - args.x_extra*ang_2_bohr # Bohr
x_max = np.max(at_positions[:, 0]) + args.x_extra*ang_2_bohr # Bohr

y_min = np.min(at_positions[:, 1]) - args.y_extra*ang_2_bohr # Bohr
y_max = np.max(at_positions[:, 1]) + args.y_extra*ang_2_bohr # Bohr

z_min = np.min(at_positions[:, 2]) - args.z_extra*ang_2_bohr # Bohr
z_max = np.max(at_positions[:, 2]) + args.z_extra*ang_2_bohr # Bohr

eval_reg_size = np.array([x_max-x_min, y_max-y_min, z_max-z_min])

eval_cell_origin = np.array([x_min, y_min, z_min])

# Define real space grid
# Cp2k chooses close to 0.08 angstroms (?)
step = args.dx
step *= ang_2_bohr

eval_reg_size_n = (np.round(eval_reg_size/step)).astype(int)
dv = eval_reg_size/eval_reg_size_n

### -----------------------------------------
### Calculate the molecular orbitals in the specified region
### -----------------------------------------

morb_grids = cu.calc_morbs_in_3D_region(eval_reg_size, eval_reg_size_n, eval_cell_origin,
                at_positions, at_elems,
                basis_sets, morb_comp_scattered,
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

    geom_base = os.path.basename(args.xyz_file)
    geom_label = os.path.splitext(geom_base)[0]

    time0 = time.time()
    elim = np.array([args.emin, args.emax])
    all_morb_grids = recvbuf.reshape(num_morbs,
            eval_reg_size_n[0], eval_reg_size_n[1], eval_reg_size_n[2])
    np.savez(args.output_file,
        morb_grids=all_morb_grids,
        morb_energies=morb_energies,
        dv=dv, # Bohr
        eval_cell_origin=eval_cell_origin, # Bohr
        elim=elim,
        ref_energy=ref_energy,
        geom_label=geom_label)
    print("Saved the orbitals to file: %.2fs" % (time.time() - time0))
