"""
Useful tools for various situations
"""

import numpy as np
import scipy
import sys

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

eval_region_description = (
    "Specify evaluation region limits [xmin xmax ymin ymax zmin zmax] (ang) as a string: "
    "'G' corresponds to global cell limit (also enables PBC if both of pair are 'G'); "
    "a number specifies absolute position wrt cell zero; t/b and number (e.g. 't2.5') "
    "specifies distance [ang] from furthest-extending atom in positive (p) or negative (n) "
    "direction. Number with _element ('p2.5_C') correspondingly from furthest atom of "
    "elem. If xmin=xmax (within 1e-4), then only a plane is assumed.")

def parse_eval_region_input(eval_reg_inp, ase_atoms, cell):

    eval_regions_inp = [eval_reg_inp[0:2], eval_reg_inp[2:4], eval_reg_inp[4:6]]
    eval_regions = [[0, 0], [0, 0], [0, 0]]

    for i in range(3):
        if eval_regions_inp[i] == ['G', 'G']:
            eval_regions[i] = None
            continue
        for j in range(2):
            reg_str = eval_regions_inp[i][j]

            has_chem_el = False

            if '_' in reg_str:
                elem = reg_str.split('_')[1]
                reg_str = reg_str.split('_')[0]
                sel_positions = ase_atoms.positions[np.array(ase_atoms.get_chemical_symbols()) == elem]
                if len(sel_positions) == 0:
                    print("Error: No element %s found. Exiting."%elem)
                    sys.exit(1)
                has_chem_el = True
            else:
                sel_positions = ase_atoms.positions


            if reg_str == 'G':
                eval_regions[i][j] = 0.0 if j == 0 else cell[i]
            elif is_number(reg_str):
                if has_chem_el:
                    print("Unrecognized option ", eval_regions_inp[i][j])
                    sys.exit(1)
                eval_regions[i][j] = float(reg_str)
                eval_regions[i][j] *= ang_2_bohr
                
            else:
                ref_at_pos = reg_str[0]
                ref_shift_str = reg_str[1:]

                if ref_at_pos != 'p' and ref_at_pos != 'n':
                    print("Error:", reg_str, "needs to start with a 'p' or 'n'")
                    sys.exit(1)
                if not is_number(ref_shift_str):
                    print("Error:", ref_shift_str, "needs to be a number")
                    sys.exit(1)
                ref_shift_val = float(ref_shift_str)

                eval_regions[i][j] = (np.min(sel_positions[:, i]) + ref_shift_val
                                        if ref_at_pos == 'n' else
                                    np.max(sel_positions[:, i]) + ref_shift_val)
                eval_regions[i][j] *= ang_2_bohr

        if np.abs(eval_regions[i][0] - eval_regions[i][1]) < 1e-3:
            eval_regions[i][0] = eval_regions[i][1]
    return eval_regions



def resize_2d_arr_with_interpolation(array, new_shape):
    x_arr = np.linspace(0, 1, array.shape[0])
    y_arr = np.linspace(0, 1, array.shape[1])
    rgi = scipy.interpolate.RegularGridInterpolator(points=[x_arr, y_arr], values=array)

    x_arr_new = np.linspace(0, 1, new_shape[0])
    y_arr_new = np.linspace(0, 1, new_shape[1])
    x_coords = np.repeat(x_arr_new, len(y_arr_new))
    y_coords = np.tile(y_arr_new, len(x_arr_new))

    return rgi(np.array([x_coords, y_coords]).T).reshape(new_shape)