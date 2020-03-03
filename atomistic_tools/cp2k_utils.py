"""
CP2K utilities
""" 

import os
import numpy as np
import scipy
import scipy.io

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def read_cp2k_output(file_path):
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    eigenvalues = []
    num_el = []
    i_line = 0
    while i_line < len(lines):
        line = lines[i_line]
        # ----
        # Occupied eigenvalues
        if "Eigenvalues of the occupied" in line:
            spin = int(line.split()[-1]) - 1 
            eigenvalues.append([])
            i_line += 2
            while True:
                vals = lines[i_line].split()
                if len(vals) == 0 or not is_float(vals[0]):
                    break
                else:
                    eigenvalues[spin] += [float(v) for v in vals]
                i_line += 1
        # Unoccupied eigenvalues
        if "Lowest Eigenvalues of the unoccupied" in line:
            spin = int(line.split()[-1]) - 1 
            i_line += 3
            while True:
                vals = lines[i_line].split()
                if len(vals) == 0 or not is_float(vals[0]):
                    break
                else:
                    eigenvalues[spin] += [float(v) for v in vals]
                i_line += 1
        # num electrons
        if "Number of electrons" in line:
            n = int(line.split()[-1])
            spin_line = lines[i_line-2]
            if "Spin " in spin_line:
                spin = int(spin_line.split()[-1]) - 1
                if len(num_el) < (spin + 1):
                    num_el.append(n)
            else:
                if len(num_el) == 0:
                    num_el.append(n)
        # ----
        i_line += 1

    nspin = len(eigenvalues)

    for i in range(nspin):
        eigenvalues[i] = np.array(eigenvalues[i]) * hart_2_ev
    
    num_el = np.array(num_el)
    if nspin == 1:
        homo = (num_el/2).astype(int) - 1
    else:
        homo = num_el - 1
        
    return nspin, homo, eigenvalues
