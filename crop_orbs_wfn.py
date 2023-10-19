#!/usr/bin/env python
import os
import numpy as np
import time
import copy
import sys

import argparse

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

import cp2k_spm_tools.cp2k_wfn_file as cwf

parser = argparse.ArgumentParser(
    description='Crops CP2K RESTART.wfn file.')

parser.add_argument(
    '--wfn_file',
    metavar='FILENAME',
    required=True,
    help='cp2k restart file containing the wavefunction.')

parser.add_argument(
    '--output_file',
    metavar='FILENAME',
    required=True,
    help='File where to save the output')

parser.add_argument(
    '--emin',
    type=float,
    metavar='E',
    default=0.0,
    help='Lowest energy value for selecting orbitals (eV).')
parser.add_argument(
    '--emax',
    type=float,
    metavar='E',
    default=0.0,
    help='Highest energy value for selecting orbitals (eV).')

parser.add_argument(
    '--n_homo',
    type=int,
    metavar='N',
    default=0,
    help="Number of HOMO orbitals to export.")
parser.add_argument(
    '--n_lumo',
    type=int,
    metavar='N',
    default=0,
    help="Number of LUMO orbitals to export.")

time0 = time.time()

args = parser.parse_args()

cp2k_wfn_f = cwf.Cp2kWfnFile()

if args.n_homo > 0 or args.n_lumo > 0:
    print("Number of orbitals specified, energy limits ignored.")
    cp2k_wfn_f.load_restart_wfn_file(args.wfn_file, n_homo=args.n_homo, n_lumo=args.n_lumo)
else:
    cp2k_wfn_f.load_restart_wfn_file(args.wfn_file, emin=args.emin, emax=args.emax)

print("Loaded wfn, %.2fs"%(time.time() - time0))

cp2k_wfn_f.write_fortran(args.output_file)



