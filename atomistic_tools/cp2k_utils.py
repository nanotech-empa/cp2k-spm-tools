"""
CP2K utilities
""" 

import os
import numpy as np
import scipy
import scipy.io
import re

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def parse_cp2k_output(file_path):
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    results = {}

    def add_res_list(name):
        if name not in results:
            results[name] = []

    i_line = 0
    while i_line < len(lines):
        line = lines[i_line]

        # ----------------------------------------------------------------
        # num electrons
        if "Number of electrons" in line:
            add_res_list('num_el')
            n = int(line.split()[-1])
            spin_line = lines[i_line-2]
            if "Spin " in spin_line:
                results['nspin'] = 2
                spin = int(spin_line.split()[-1]) - 1
            else:
                spin = 0
                results['nspin'] = 1

            if len(results['num_el']) == spin:
                results['num_el'].append(n)
            else:
                print("Warning: something is wrong with num. el. parsing.")
        # ----------------------------------------------------------------
        # Energy (overwrite so that we only have the last (converged) one)
        if "ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.):" in line:
            results['energy'] = float(line.split()[-1]) * hart_2_ev        
        # ----------------------------------------------------------------
        # Occupied eigenvalues (normal SCF)
        if "Eigenvalues of the occupied" in line:
            add_res_list('evals')
            spin = int(line.split()[-1]) - 1 
            results['evals'].append([])
            i_line += 2
            while True:
                vals = lines[i_line].split()
                if len(vals) == 0 or not is_float(vals[0]):
                    break
                else:
                    results['evals'][spin] += [float(v) * hart_2_ev for v in vals]
                i_line += 1
        # ----------------------------------------------------------------
        # Unoccupied eigenvalues (normal SCF)
        if "Lowest Eigenvalues of the unoccupied" in line:
            spin = int(line.split()[-1]) - 1 
            i_line += 3
            while True:
                vals = lines[i_line].split()
                if len(vals) == 0 or not is_float(vals[0]):
                    break
                else:
                    results['evals'][spin] += [float(v) * hart_2_ev for v in vals]
                i_line += 1
        # ----------------------------------------------------------------
        # GW output
        if "GW quasiparticle energies" in line:
            add_res_list('gw_mo')
            add_res_list('gw_occ')
            add_res_list('gw_e_scf')
            add_res_list('gw_eval')

            spin = 1 if 'beta' in line else 0

            if len(results['gw_mo']) > spin:
                # we already have a set, overwrite with later iteration
                results['gw_mo'][spin] = []
                results['gw_occ'][spin] = []
                results['gw_e_scf'][spin] = []
                results['gw_eval'][spin] = []
            else:
                results['gw_mo'].append([])
                results['gw_occ'].append([])
                results['gw_e_scf'].append([])
                results['gw_eval'].append([])

            i_line += 10

            while True:
                line_loc = lines[i_line]
                if "GW HOMO-LUMO gap" in line_loc:
                    break
                vals = line_loc.split()
                # header & example line:
                #             MO      E_SCF       Sigc   Sigc_fit   Sigx-vxc          Z       E_GW
                #     75 ( occ )    -10.030      1.723      0.000     -4.297      1.000    -10.031
                if len(vals) == 10 and is_float(vals[0]):
                    results['gw_mo'][spin].append(int(vals[0]) - 1) # start orb count from 0
                    results['gw_occ'][spin].append(1 if vals[2] == 'occ' else 0)
                    results['gw_e_scf'][spin].append(float(vals[4]) * hart_2_ev)
                    results['gw_eval'][spin].append(float(vals[9]) * hart_2_ev)
                i_line += 1
        # ----------------------------------------------------------------
        # IC output
        if "Single-electron energies" in line and " with image charge (ic) correction" in line:
            add_res_list('ic_mo')
            add_res_list('ic_occ')
            add_res_list('ic_en')
            add_res_list('ic_delta')

            spin = 1 if 'beta' in line else 0

            if len(results['ic_mo']) > spin:
                # we already have a set, overwrite with later iteration
                results['ic_mo'][spin] = []
                results['ic_occ'][spin] = []
                results['ic_en'][spin] = []
                results['ic_delta'][spin] = []
            else:
                results['ic_mo'].append([])
                results['ic_occ'].append([])
                results['ic_en'].append([])
                results['ic_delta'].append([])

            i_line += 7

            while True:
                line_loc = lines[i_line]
                if "IC HOMO-LUMO gap" in line_loc:
                    break
                vals = line_loc.split()
                # header & example line:
                #             MO     E_n before ic corr           Delta E_ic    E_n after ic corr
                #     80 ( occ )                 -6.952                1.186               -5.766
                if len(vals) == 7 and is_float(vals[0]):
                    results['ic_mo'][spin].append(int(vals[0]) - 1) # start orb count from 0
                    results['ic_occ'][spin].append(1 if vals[2] == 'occ' else 0)
                    results['ic_en'][spin].append(float(vals[4]) * hart_2_ev)
                    results['ic_delta'][spin].append(float(vals[5]) * hart_2_ev)
                i_line += 1
        # ----------------------------------------------------------------
        i_line += 1

    # ----------------------------------------------------------------
    # Determine HOMO indexes w.r.t. outputted eigenvalues
    results['homo'] = []
    for i_spin in range(results['nspin']):
        # In case of GW and IC, the MO count doesn't start from 0
        # so use the occupations
        if 'gw_occ' in results:
            results['homo'].append(results['gw_occ'][i_spin].index(0) - 1)
        elif 'ic_occ' in results:
            results['homo'].append(results['ic_occ'][i_spin].index(0) - 1)
        else:
            # In case of normal SCF, use the electron numbers
            if results['nspin'] == 1:
                results['homo'].append(int(results['num_el'][i_spin]/2) - 1)
            else:
                results['homo'].append(results['num_el'][i_spin] - 1)

    # ----------------------------------------------------------------
    # convert "lowest level" to numpy arrays
    for key in results:
        if isinstance(results[key], list):
            for i in range(len(results[key])):
                if isinstance(results[key][i], list):
                    results[key][i] = np.array(results[key][i])

    return results

def read_cp2k_pdos_file(file_path):
    header = open(file_path).readline()
    fermi = float(re.search("Fermi.* ([+-]?[0-9]*[.]?[0-9]+)", header).group(1))
    try:
        kind = re.search("atomic kind.(\S+)", header).group(1)
    except:
        kind = None
    data = np.loadtxt(file_path)
    out_data = np.zeros((data.shape[0], 2))
    out_data[:, 0] = (data[:, 1] - fermi) * hart_2_ev # energy

    out_data[:, 1] = np.sum(data[:, 3:], axis=1) # "contracted pdos"
    return out_data, kind



