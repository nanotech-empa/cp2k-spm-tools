
import os
import numpy as np
import time
import copy
import sys
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cp2k_utilities as cu

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

parser = argparse.ArgumentParser(
    description='Produces LDOS data from supplied molecular orbitals \
                 along x direction on a plane normal to z direction.')
parser.add_argument(
    '--npz_file',
    metavar='FILENAME',
    required=True,
    help='.npz file containing the molecular orbitals and supporting info.')
parser.add_argument(
    '--output_dir',
    metavar='FILENAME',
    required=True,
    help="Directory containing the output data.")
parser.add_argument(
    '--sts_plane_height',
    type=float,
    metavar='H',
    required=True,
    help="The height of the sts plane from the topmost atom in angstroms. "\
         "Must lie inside the box/plane or top of it (triggers extrapolation).")
parser.add_argument(
    '--sts_de',
    type=float,
    default=0.05,
    help="Energy discretization for STS. (eV)")
parser.add_argument(
    '--sts_fwhm',
    nargs='+',
    type=float,
    metavar='FWHM',
    help='Full width half maximums for the broadening of each orbital.')

parser.add_argument(
    '--hartree_file',
    metavar='FILENAME',
    default=None,
    help="Cube file containing the hartree potential." \
         "Only needed if sts_plane_height is out of the morb eval region.")


args = parser.parse_args()

time0 = time.time()

output_dir = args.output_dir
if output_dir[-1] != '/':
    output_dir += '/'

npz_file_data = np.load(args.npz_file)

morb_grids = npz_file_data['morb_grids']
morb_energies = npz_file_data['morb_energies']
dv = npz_file_data['dv']
z_arr = npz_file_data['z_arr']
emin, emax = npz_file_data['elim']
ref_energy = npz_file_data['ref_energy']
geom_name = npz_file_data['geom_label']

z_bottom = z_arr[0]
z_top = z_arr[-1]
plane_index = int(np.round((args.sts_plane_height*ang_2_bohr - z_bottom)/dv[2]))

num_morbs = np.shape(morb_grids)[0]
eval_reg_size_n = np.shape(morb_grids[0])
eval_reg_size = dv*eval_reg_size_n

if plane_index > len(z_arr) - 1:
    # Extrapolation is needed! Load the Hartree potential
    if args.hartree_file == None:
        print("Hartree potential must be supplied if STS plane is out of region")
        exit()

    time1 = time.time()
    hart_cube_data = cu.read_cube_file(args.hartree_file)
    hart_cube = hart_cube_data[-1]
    hart_cell = hart_cube_data[5]
    hart_atomic_pos = hart_cube_data[-2]
    print("Read hartree: %.3f" % (time.time()-time1))

    topmost_atom_z = np.max(hart_atomic_pos[:, 2]) # Angstrom
    hart_plane_z = args.extrap_plane + topmost_atom_z
    hart_plane_index = int(np.round(hart_plane_z/hart_cell[2, 2]*np.shape(hart_cube)[2]))

    hart_plane = hart_cube[:, :, hart_plane_index] - ref_energy/hart_2_ev

    print("Hartree on extrapolation plane: min: %.4f; max: %.4f; avg: %.4f (eV)" % (
                                                    np.min(hart_plane)*hart_2_ev,
                                                    np.max(hart_plane)*hart_2_ev,
                                                    np.mean(hart_plane)*hart_2_ev))
    # ------------------------------
    # Do the extrapolation !!!
    # ------------------------------

    morb_planes = cu.extrapolate_morbs(morb_grids, morb_energies, dv, -1,
                                         args.sts_plane_height - z_top,
                                         hart_plane, True, use_weighted_avg=True)

else:
    morb_planes = np.zeros((num_morbs, eval_reg_size_n[0], eval_reg_size_n[1]))
    for i_mo in range(num_morbs):
        morb_planes[i_mo, :, :] =  morb_grids[i_mo, :, :, plane_index]

### ----------------------------------------------------------------
### Plot some orbitals for troubleshooting
### ----------------------------------------------------------------
i_homo = 0
for i, en in enumerate(morb_energies):
    if en > 0.0:
        i_homo = i - 1
        break
    if np.abs(en) < 1e-6:
        i_homo = i

select = [i_homo - 1, i_homo, i_homo + 1, i_homo + 2]

sel_morbs = np.zeros((eval_reg_size_n[0], 4*eval_reg_size_n[1]))

for i, i_mo in enumerate(select):
    sel_morbs[:, i*eval_reg_size_n[1]:(i+1)*eval_reg_size_n[1]] = morb_planes[i_mo]

x_arr = np.arange(0, eval_reg_size[0], dv[0])
y_arr_inc = np.arange(0, 4*eval_reg_size[1], dv[1])
x_grid_inc, y_grid_inc = np.meshgrid(x_arr, y_arr_inc, indexing='ij')

max_val = np.max(sel_morbs)

plt.figure(figsize=(12, int(eval_reg_size_n[1]/eval_reg_size_n[0]*12*4)))
plt.pcolormesh(x_grid_inc, y_grid_inc, sel_morbs, vmax=max_val, vmin=-max_val, cmap='seismic') # seismic bwr
plt.savefig(output_dir+"orbs.png", dpi=300, bbox_inches='tight')
plt.close()


### ----------------------------------------------------------------
### Calculate the LDOS based on the orbitals
### ----------------------------------------------------------------

de = args.sts_de

e_arr = np.arange(emin, emax+de, de)

x_arr_ang = np.arange(0.0, eval_reg_size_n[0]*dv[0], dv[0])/ang_2_bohr

x_e_grid, e_grid = np.meshgrid(x_arr_ang, e_arr, indexing='ij')

def calculate_ldos(de, fwhm, broad_type):

    def lorentzian(x):
        gamma = 0.5*fwhm
        return gamma/(np.pi*(x**2+gamma**2))

    def gaussian(x):
        sigma = fwhm/2.3548
        return np.exp(-x**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

    pldos = np.zeros((eval_reg_size_n[0], len(e_arr)))

    for i_mo, morb_plane in enumerate(morb_planes):
        en = morb_energies[i_mo]
        avg_morb = np.mean(morb_plane**2, axis=1)

        if broad_type == 'l':
            morb_ldos_broad = np.outer(avg_morb, lorentzian(e_arr - en))
        else:
            morb_ldos_broad = np.outer(avg_morb, gaussian(e_arr - en))

        pldos += morb_ldos_broad

    return pldos


fwhm_arr = args.sts_fwhm

height = args.sts_plane_height

for fwhm in fwhm_arr:
    for broad_type in ['g']:

        pldos = calculate_ldos(de, fwhm, broad_type)

        ofname = output_dir + "ldos_%s_h%.1f_fwhm%.2f%s.txt" % (geom_name, height, fwhm, broad_type)
        header = "geom: %s; height(ang): %.1f; fwhm(eV): %.4f; broad: %s; " % (geom_name, height, fwhm, broad_type) + \
                 "xmin(ang): %.5f; xmax(ang): %.5f; nx: %d; " % (np.min(x_arr_ang), np.max(x_arr_ang), len(x_arr_ang)) + \
                 "emin(eV): %.5f; emax(eV): %.5f; ne: %d" % (np.min(e_arr), np.max(e_arr), len(e_arr))
        np.savetxt(ofname, pldos, header=header, fmt="%.4e")

print("Completed in %.1f s" % (time.time()-time0))
