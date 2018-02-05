
import os
import numpy as np
import time
import copy
import sys
import argparse

import pickle

import cp2k_utilities as cu

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

parser = argparse.ArgumentParser(
    description='Select specific morbs and convert to pickled binary.')
parser.add_argument(
    '--cp2k_output',
    metavar='FILENAME',
    required=True,
    help='CP2K output of the SCF calculation.')
parser.add_argument(
    '--restart_file',
    metavar='FILENAME',
    required=True,
    help='Restart file containing the final wavefunction.')
parser.add_argument(
    '--output_file',
    metavar='FILENAME',
    required=True,
    help='Output binary containing the pickled python array.')
parser.add_argument(
    '--emin',
    type=float,
    metavar='E',
    required=True,
    help='Lowest energy value for selecting orbitals.')
parser.add_argument(
    '--emax',
    type=float,
    metavar='E',
    required=True,
    help='Highest energy value for selecting orbitals.')

args = parser.parse_args()

time0 = time.time()
fermi = cu.read_fermi_from_cp2k_out(args.cp2k_output)
print("Read cp2k out: %.3f" % (time.time()-time0))

time1 = time.time()
morb_composition, morb_energies, morb_occs = cu.load_restart_wfn_file(
        args.restart_file, args.emin, args.emax, fermi)
print("Read restart: %.3f" % (time.time()-time1))

with open(args.output_file, 'wb') as handle:
    pickle.dump([morb_composition, morb_energies, morb_occs], handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Total time: %.1f s" % (time.time() - time0))
