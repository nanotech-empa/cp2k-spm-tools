#!/usr/bin/env python
import os
import numpy as np
import time
import copy
import sys

import argparse

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

from cp2k_spm_tools import common, cube, cube_utils


parser = argparse.ArgumentParser(
    description='Operations on gaussian cube files.')

parser.add_argument(
    'cubes',
    metavar='FILENAME',
    nargs='+',
    help='Gaussian cube files.')

parser.add_argument(
    'operations',
    metavar='OP',
    type=str,
    help='Operations to apply to each cube. Enclose in quotation marks.')

parser.add_argument(
    '--proj_1d',
    metavar='IDs',
    type=str,
    default='no',
    help=("Projects to 'x', 'y' or 'z' dim, possibly averaging (e.g. 'z avg').")
)
parser.add_argument(
    '--skip_result_cube',
    action='store_true',
    help=("Don't write the result cube.")
)
parser.add_argument(
    '--add_artif_core',
    action='store_true',
    help=("Adds artifical core charge to result cube (mainly for Bader analysis).")
)

time0 = time.time()

args = parser.parse_args()


result = None

operations = args.operations.split()

if len(operations) != len(args.cubes):
    print("Error: didn't find match between cubes and operations.")
    print("Did you forget to enclose operations in quotation marks?")
    exit(1)


for i_c, cube_file in enumerate(args.cubes):
    time1 = time.time()
    c = cube.Cube()
    print("Reading %s..." % cube_file)
    c.read_cube_file(cube_file)
    
    if result is None:
        result = copy.deepcopy(c)
        result.data.fill(0)
        
    if np.any(np.abs(c.cell - result.cell) > 1e-4):
        print("Error: cube cell doesn't match: ", cube_file)
        exit(1)
    if np.any(c.data.shape != result.data.shape):
        print("Error: cube shape doesn't match: ", cube_file)
        exit(1)
    
    op = operations[i_c]

    if op == "+":
        result.data += c.data
    elif op == "-":
        result.data -= c.data
    elif op == "*":
        result.data *= c.data
    elif op == "/":
        result.data /= c.data

    print("%s done, time: %.2fs"%(cube_file, (time.time() - time1)))

if args.add_artif_core:
    cube_utils.add_artif_core_charge(result)

if not args.skip_result_cube:
    print("Writing result...")
    result.write_cube_file("./result.cube")

proj_1d_ids = args.proj_1d.split()

if not "no" in proj_1d_ids:
    avg = 'avg' in proj_1d_ids
    proj_dims = []
    if 'x' in proj_1d_ids:
        proj_dims.append(0)
    if 'y' in proj_1d_ids:
        proj_dims.append(1)
    if 'z' in proj_1d_ids:
        proj_dims.append(2)
    
    for pd in proj_dims:
        if avg:
            data_1d = np.mean(result.data, axis=tuple({0, 1, 2} - {pd}))
        else:
            data_1d = np.sum(result.data, axis=tuple({0, 1, 2} - {pd}))
        x_arr = result.origin[pd] + np.linspace(0.0, result.cell[pd, pd], result.data.shape[pd])
        x_arr /= ang_2_bohr
        save_arr = np.column_stack((x_arr, data_1d))
        avg_str = "_avg" if avg else ""
        fname = "./proj_1d_%d" % pd + avg_str + ".txt"
        np.savetxt(fname, save_arr)

print("Finished, total time: %.2fs"%(time.time() - time0))