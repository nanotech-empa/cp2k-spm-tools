#!/usr/bin/env python
import os
import numpy as np
import time
import copy
import sys

import argparse

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

from atomistic_tools import common, cube


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


print("Writing result...")
result.write_cube_file("./result.cube")

print("Finished, total time: %.2fs"%(time.time() - time0))