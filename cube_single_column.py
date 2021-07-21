#!/usr/bin/env python
import os
import numpy as np
import time
import copy
import sys

import ase

import argparse

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

from cp2k_spm_tools import common, cube, cube_utils, bader_wrapper


parser = argparse.ArgumentParser(
    description='Reformats cube in case it contains uneven columns (not supported by some software).')

parser.add_argument(
    'cube',
    metavar='FILENAME',
    help='Input cube file.')

parser.add_argument(
    '--output_cube',
    metavar='FILENAME',
    default='single_col.cube',
    help='Output cube file.')
### -----------------------------------------------------------

time0 = time.time()

### ------------------------------------------------------
### Parse args for only one rank to suppress duplicate stdio
### ------------------------------------------------------

args = parser.parse_args()

### ------------------------------------------------------
### Load the cube meta-data
### ------------------------------------------------------

inp_cube = cube.Cube()
inp_cube.read_cube_file(args.cube, read_data=False)
n_atoms = len(inp_cube.ase_atoms)

n_metadata_lines = 6 + n_atoms

column_digit_width = None
num_decimals = 0

with open(args.cube, 'r') as in_f:
    with open(args.output_cube, 'w') as out_f:
        i_line = 0
        for line in in_f:
            if i_line < n_metadata_lines:
                out_f.write(line)
            else:
                vals = line.split()
                for val in vals:

                    val_fl = float(val)
                    if column_digit_width is None:
                        column_digit_width = len(val)
                        if val_fl >= 0.0:
                            column_digit_width += 1
                        # determine num decimals
                        after_p = val.split('.')[1]
                        for c in after_p:
                            if c.isdigit():
                                num_decimals += 1
                            else:
                                break
                    
                    out_f.write("{:{:d}.{:d}e}\n".format(val_fl, column_digit_width, num_decimals))
            i_line += 1

print("Finished, total time: %.2fs" % (time.time() - time0))
