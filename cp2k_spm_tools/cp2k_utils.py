"""
CP2K utilities
"""

import re

import numpy as np

ang_2_bohr = 1.0 / 0.52917721067
hart_2_ev = 27.21138602


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_cp2k_output(file_path):
    with open(file_path, "r") as f:
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
            add_res_list("num_el")
            n = int(line.split()[-1])
            spin_line = lines[i_line - 2]
            if "Spin " in spin_line:
                results["nspin"] = 2
                spin = int(spin_line.split()[-1]) - 1
            else:
                spin = 0
                results["nspin"] = 1

            if len(results["num_el"]) == spin:
                results["num_el"].append(n)
            else:
                print("Warning: something is wrong with num. el. parsing.")
        # ----------------------------------------------------------------
        # Energy (overwrite so that we only have the last (converged) one)
        if "ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.):" in line:
            results["energy"] = float(line.split()[-1]) * hart_2_ev
        # ----------------------------------------------------------------
        # Occupied eigenvalues (normal SCF)
        if "Eigenvalues of the occupied" in line:
            add_res_list("evals")
            spin = int(line.split()[-1]) - 1
            results["evals"].append([])
            i_line += 2
            while True:
                vals = lines[i_line].split()
                if len(vals) == 0 or not is_float(vals[0]):
                    break
                else:
                    results["evals"][spin] += [float(v) * hart_2_ev for v in vals]
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
                    results["evals"][spin] += [float(v) * hart_2_ev for v in vals]
                i_line += 1
        # ----------------------------------------------------------------
        # GW output
        if "Sigx-vxc (eV)" in line and "E_GW (eV)" in line:
            add_res_list("mo")
            add_res_list("occ")
            add_res_list("gw_eval")
            add_res_list("g0w0_eval")
            add_res_list("g0w0_e_scf")

            i_line += 1

            gw_mo = []
            gw_occ = []
            gw_e_scf = []
            gw_eval = []

            while True:
                line_loc = lines[i_line]
                if "GW HOMO-LUMO gap" in line_loc:
                    spin = 1 if "Beta" in line_loc else 0

                    if len(results["mo"]) > spin:
                        # we already have a set, overwrite with later iteration
                        results["mo"][spin] = gw_mo
                        results["occ"][spin] = gw_occ
                        results["gw_eval"][spin] = gw_eval
                    else:
                        results["mo"].append(gw_mo)
                        results["occ"].append(gw_occ)
                        results["gw_eval"].append(gw_eval)
                        results["g0w0_eval"].append(gw_eval)
                        results["g0w0_e_scf"].append(gw_e_scf)

                    break

                vals = line_loc.split()
                # header & example line:
                #     Molecular orbital   E_SCF (eV)       Sigc (eV)   Sigx-vxc (eV)       E_GW (eV)
                #        1 ( occ )           -26.079           6.728         -10.116         -26.068
                if len(vals) == 8 and is_float(vals[0]):
                    gw_mo.append(int(vals[0]) - 1)  # start orb count from 0
                    gw_occ.append(1 if vals[2] == "occ" else 0)
                    gw_e_scf.append(float(vals[4]))
                    gw_eval.append(float(vals[7]))
                i_line += 1
        # ----------------------------------------------------------------
        # IC output
        if "E_n before ic corr" in line and "Delta E_ic" in line:
            add_res_list("mo")
            add_res_list("occ")
            add_res_list("ic_en")
            add_res_list("ic_delta")

            i_line += 1

            ic_mo = []
            ic_occ = []
            ic_en = []
            ic_delta = []

            while True:
                line_loc = lines[i_line]
                if "IC HOMO-LUMO gap" in line_loc:
                    spin = 1 if "Beta" in line_loc else 0

                    if len(results["mo"]) > spin:
                        # we already have a set, overwrite with later iteration
                        results["mo"][spin] = ic_mo
                        results["occ"][spin] = ic_occ
                        results["ic_en"][spin] = ic_en
                        results["ic_delta"][spin] = ic_delta
                    else:
                        results["mo"].append(ic_mo)
                        results["occ"].append(ic_occ)
                        results["ic_en"].append(ic_en)
                        results["ic_delta"].append(ic_delta)

                    break

                vals = line_loc.split()
                # header & example line:
                #           MO     E_n before ic corr           Delta E_ic    E_n after ic corr
                #   70 ( occ )                -11.735                1.031              -10.705
                if len(vals) == 7 and is_float(vals[0]):
                    ic_mo.append(int(vals[0]) - 1)  # start orb count from 0
                    ic_occ.append(1 if vals[2] == "occ" else 0)
                    ic_en.append(float(vals[4]))
                    ic_delta.append(float(vals[5]))
                i_line += 1

        # ----------------------------------------------------------------
        i_line += 1

    # ----------------------------------------------------------------
    # Determine HOMO indexes w.r.t. outputted eigenvalues
    results["homo"] = []

    if "occ" in results:
        # In case of GW and IC, the MO count doesn't start from 0
        # so use the occupations
        for i_spin in range(results["nspin"]):
            results["homo"].append(results["occ"][i_spin].index(0) - 1)
    else:
        # In case of normal SCF, use the electron numbers
        for i_spin in range(results["nspin"]):
            if results["nspin"] == 1:
                results["homo"].append(int(results["num_el"][i_spin] / 2) - 1)
            else:
                results["homo"].append(results["num_el"][i_spin] - 1)
            # Also create 'mo' and 'occ' arrays
            add_res_list("occ")
            add_res_list("mo")
            occ = np.ones(len(results["evals"][i_spin]))
            occ[results["homo"][i_spin] + 1 :] = 0
            mo = np.arange(len(results["evals"][i_spin]))
            results["occ"].append(occ)
            results["mo"].append(mo)

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
        kind = re.search(r"atomic kind.(\S+)", header).group(1)
    except:
        kind = None
    data = np.loadtxt(file_path)
    out_data = np.zeros((data.shape[0], 2))
    out_data[:, 0] = (data[:, 1] - fermi) * hart_2_ev  # energy

    out_data[:, 1] = np.sum(data[:, 3:], axis=1)  # "contracted pdos"
    return out_data, kind
