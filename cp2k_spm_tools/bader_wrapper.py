"""
Routines to call the Henkelmann Bader program
"""

import os
import subprocess

ang_2_bohr = 1.0 / 0.52917721067
hart_2_ev = 27.21138602


def call_bader(folder, cube_file, method="neargrid", basin_atoms=[], ref_cube=None):
    cur_dir = os.getcwd()

    command = "bader -b %s" % method
    if len(basin_atoms) > 0:
        command += " -p sel_atom " + " ".join([str(e + 1) for e in basin_atoms])

    if ref_cube is not None:
        command += " -ref %s" % ref_cube

    command += " %s > bader.log" % cube_file

    print(command)

    try:
        os.chdir(folder)
        subprocess.call(command, shell=True)
    except:
        print("Warning: Couldn't run Bader.")

    os.chdir(cur_dir)
