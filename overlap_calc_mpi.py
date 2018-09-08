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


from mpi4py import MPI

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

parser = argparse.ArgumentParser(
    description='Calculates the overlap between all pairs of electronic \
                 orbitals in two groups')

parser.add_argument(
    '--npz_file1',
    metavar='FILENAME',
    required=True,
    help='.npz file containing the orbitals and info of 1st group.')

parser.add_argument(
    '--n_homo',
    type=int,
    metavar='N',
    help="Number of HOMO orbitals to select in group 1.")
parser.add_argument(
    '--n_lumo',
    type=int,
    metavar='N',
    help="Number of LUMO orbitals to select in group 1.")

parser.add_argument(
    '--npz_file2',
    metavar='FILENAME',
    required=True,
    help='.npz file containing the orbitals and info of 2nd group.')

args = parser.parse_args()

time0 = time.time()

def load_npz(npz_file):

    npz_file_data = np.load(npz_file)

    x_arr = npz_file_data['x_arr']
    y_arr = npz_file_data['y_arr']
    z_arr = npz_file_data['z_arr']
    mol_bbox = npz_file_data['mol_bbox']
    elim = npz_file_data['elim']
    ref_energy = npz_file_data['ref_energy']
    geom_label = npz_file_data['geom_label']

    morb_grids = npz_file_data['morb_grids_s1']
    morb_energies = npz_file_data['morb_energies_s1']
    homo_ind = npz_file_data['homo_s1']

    vol_elem = (x_arr[1]-x_arr[0])*(y_arr[1]-y_arr[0])*(z_arr[1]-z_arr[0])
    
    return morb_grids, morb_energies, homo_ind, vol_elem

morb_grids1, morb_energies1, homo_ind1, ve1 = load_npz(args.npz_file1)

print("R%d/%d: loaded G1, %.2fs"%(mpi_rank, mpi_size, (time.time() - time0)))
sys.stdout.flush()

time1 = time.time()
morb_grids2, morb_energies2, homo_ind2, ve2 = load_npz(args.npz_file2)

print("R%d/%d: loaded G2, %.2fs"%(mpi_rank, mpi_size, (time.time()-time1)))
sys.stdout.flush()

if morb_grids1[0].shape != morb_grids2[0].shape:
    print("Orbitals are not compatible, exiting!")
    sys.stdout.flush()
    exit()

if np.abs(ve1 -ve2) > 1e-3:
    print("Volume elements are not compatible, exiting!")
    print(ve1, ve2)
    sys.stdout.flush()
    exit()

g1_start = homo_ind1 - args.n_homo
g1_end = homo_ind1 + 1 + args.n_lumo

if g1_start < 0 or g1_end > len(morb_energies1):
    print("Not enough HOMO and/or LUMO orbitals evaluated!")
    print("Found %d HOMO and %d LUMO orbitals." % (
        homo_ind1+1,
        len(morb_energies1) - homo_ind1 - 1))
    sys.stdout.flush()
    exit()

n_orb_g2 = len(morb_energies2)

# Select orbitals for the current mpi rank
base_orb_per_rank = int(np.floor(n_orb_g2/mpi_size))
extra_orbs =  n_orb_g2 - base_orb_per_rank*mpi_size
if mpi_rank < extra_orbs:
    g2_start = mpi_rank*(base_orb_per_rank + 1)
    g2_end = (mpi_rank+1)*(base_orb_per_rank + 1)
else:
    g2_start = mpi_rank*(base_orb_per_rank) + extra_orbs
    g2_end = (mpi_rank+1)*(base_orb_per_rank) + extra_orbs

print("R%d/%d: G2 orbs %d:%d / %d"%(mpi_rank, mpi_size, g2_start, g2_end, n_orb_g2))
sys.stdout.flush()

time1 = time.time()

overlap_matrix =  (np.einsum('iklm, jklm',
    morb_grids1[g1_start:g1_end],
    morb_grids2[g2_start:g2_end])*ve1)**2

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

if mpi_rank == 0:
    overlap_matrix_collected = recvbuf.reshape(
        len(morb_energies1[g1_start:g1_end]),
        len(morb_energies2))
        
    np.savez("./overlap",
        overlap_matrix=overlap_matrix_collected,
        en_grp1=morb_energies1[g1_start:g1_end],
        en_grp2=morb_energies2,
        homo_grp1=args.n_homo)
    print("Finish! Total time: %.2fs" % (time.time() - time0))
