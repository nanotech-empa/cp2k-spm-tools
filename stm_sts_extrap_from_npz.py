import os
import numpy as np
import time
import copy
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import skimage.transform

import argparse

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

import cp2k_utilities as cu

parser = argparse.ArgumentParser(
    description="Extrapolates supplied molecular orbitals " \
                 "and calculates STM, STS.")

parser.add_argument(
    '--npz_file',
    metavar='FILENAME',
    required=True,
    help='.npz file containing the molecular orbitals and supporting info.')

parser.add_argument(
    '--hartree_file',
    metavar='FILENAME',
    required=True,
    help='Cube file containing the hartree potential.')

parser.add_argument(
    '--extrap_plane',
    type=float,
    metavar='H',
    required=True,
    help="The height of the extrapolation plane from the topmost atom. "\
         "Must lie inside the box, where the orbitals were calculated. (angstrom)")

parser.add_argument(
    '--extrap_extent',
    type=float,
    metavar='H',
    required=True,
    help="The extent of the extrapolation region. (angstrom)")

parser.add_argument(
    '--output_dir',
    metavar='FILENAME',
    required=True,
    help="Directory containing the output figures.")

parser.add_argument(
    '--bias_voltages',
    nargs='+',
    type=float,
    metavar='U',
    required=True,
    help="List of bias voltages. (Volts)")

parser.add_argument(
    '--stm_plane_heights',
    nargs='*',
    type=float,
    metavar='H',
    help="List of heights for constant height STM pictures (wrt topmost atom).")

parser.add_argument(
    '--stm_isovalues',
    nargs='*',
    type=float,
    metavar='C',
    help="List of charge density isovalues for constant current STM pictures.")

parser.add_argument(
    '--sts_plane_heights',
    nargs='*',
    type=float,
    metavar='H',
    help="List of heights for STS. (angstroms)")
parser.add_argument(
    '--sts_de',
    type=float,
    default=0.05,
    help="Energy discretization for STS. (eV)")
parser.add_argument(
    '--sts_fwhm',
    type=float,
    default=0.1,
    help="Full width at half maximum for STS gaussian broadening. (eV)")


args = parser.parse_args()

npz_file_data = np.load(args.npz_file)

morb_grids = npz_file_data['morb_grids']
morb_energies = npz_file_data['morb_energies']
dv = npz_file_data['dv']
z_arr = npz_file_data['z_arr']
elim = npz_file_data['elim']
ref_energy = npz_file_data['ref_energy']

z_top = z_arr[-1]
eval_reg_size_n = np.shape(morb_grids[0])
eval_reg_size = dv * eval_reg_size_n

x_arr = np.arange(0.0, eval_reg_size_n[0]*dv[0], dv[0])/ang_2_bohr
y_arr = np.arange(0.0, eval_reg_size_n[1]*dv[1], dv[1])/ang_2_bohr
x_grid, y_grid = np.meshgrid(x_arr, y_arr, indexing='ij')

i_homo = 0
for i, en in enumerate(morb_energies):
    if en > 0.0:
        i_homo = i - 1
        break
    if np.abs(en) < 1e-6:
        i_homo = i

def get_plane_index(z, z_arr, dz):
    return int(np.round((z-z_arr[0])/dz))

### -----------------------------------------
### EXTRAPOLATION
### -----------------------------------------

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
fig_size = 6
plt.figure(figsize=(fig_size*hart_plane.shape[0]/hart_plane.shape[1]+1.0, fig_size))
plt.pcolormesh(hart_plane.T*hart_2_ev, cmap='seismic')
plt.colorbar()
plt.axis('scaled')
plt.savefig(args.output_dir+"/hartree.png", dpi=300)
plt.close()

def extrapolate_morbs(morb_grids, morb_energies, dv, plane_ind, extent, hart_plane, use_weighted_avg=True):
    # NB: everything in hartree units!
    extrap_planes = []
    time1 = time.time()

    num_morbs = np.shape(morb_grids)[0]
    extrap_morbs = np.zeros((num_morbs, eval_reg_size_n[0], eval_reg_size_n[1], int(extent*ang_2_bohr/dv[2])))

    for morb_index in range(num_morbs):

        morb_plane = morb_grids[morb_index][:, :, plane_ind]

        if use_weighted_avg:
            # weigh the hartree potential by the molecular orbital
            density_plane = morb_plane**2
            density_plane /= np.sum(density_plane)
            weighted_hartree = density_plane * skimage.transform.resize(hart_plane, density_plane.shape, mode='reflect')
            hartree_avg = np.sum(weighted_hartree)
        else:
            hartree_avg = np.mean(hart_plane)

        energy = morb_energies[morb_index]/hart_2_ev
        if energy > hartree_avg:
            print("Warning: unbound state, can't extrapolate! index: %d. Exiting." % morb_index)
            break

        fourier = np.fft.rfft2(morb_plane)
        # NB: rfft2 takes REAL fourier transform over last (y) axis and COMPLEX over other (x) axes
        # dv in BOHR, so k is in 1/bohr
        kx_arr = 2*np.pi*np.fft.fftfreq(morb_plane.shape[0], dv[0])
        ky_arr = 2*np.pi*np.fft.rfftfreq(morb_plane.shape[1], dv[1])

        kx_grid, ky_grid = np.meshgrid(kx_arr, ky_arr,  indexing='ij')

        prefactors = np.exp(-np.sqrt(kx_grid**2 + ky_grid**2 - 2*(energy - hartree_avg))*dv[2])

        for iz in range(np.shape(extrap_morbs)[3]):
            fourier *= prefactors
            extrap_morbs[morb_index, :, :, iz] = np.fft.irfft2(fourier, morb_plane.shape)

    print("Extrapolation time: %.3f s"%(time.time()-time1))
    return extrap_morbs

extrap_plane_index = get_plane_index(args.extrap_plane*ang_2_bohr, z_arr, dv[2])
if extrap_plane_index >= np.shape(morb_grids[0])[2]:
    print(z_arr[-1])
    print("Error: the extrapolation plane can't be outside the initial box (z_max = %.2f)"
           % (z_arr[-1]/ang_2_bohr))
    exit(1)
extrap_morbs = extrapolate_morbs(morb_grids, morb_energies, dv, extrap_plane_index,
                                 args.extrap_extent, hart_plane, use_weighted_avg=True)

total_morb_grids = np.concatenate((morb_grids, extrap_morbs), axis=3)

# In bohr and wrt topmost atom
total_z_arr = np.arange(0.0, np.shape(total_morb_grids)[3]*dv[2], dv[2]) + z_arr[0]

### -----------------------------------------
### Summing charge densities according to bias voltages
### -----------------------------------------

charge_dens_arr = np.zeros((len(args.bias_voltages),
                            total_morb_grids[0].shape[0],
                            total_morb_grids[0].shape[1],
                            total_morb_grids[0].shape[2]))

for i_bias, bias in enumerate(args.bias_voltages):
    for imo, morb_grid in enumerate(total_morb_grids):
        if morb_energies[imo] > np.max([0.0, bias]):
            break
        if morb_energies[imo] >= np.min([0.0, bias]):
            charge_dens_arr[i_bias] += morb_grid**2

### -----------------------------------------
### Constant height STM
### -----------------------------------------

for plane_height in args.stm_plane_heights:
    for i_bias, bias in enumerate(args.bias_voltages):
        plane_index = get_plane_index(plane_height*ang_2_bohr, total_z_arr, dv[2])

        plt.figure(figsize=(7, 6))
        plot_data = charge_dens_arr[i_bias][:, :, plane_index]
        max_val = np.max(plot_data)
        plt.pcolormesh(x_grid, y_grid, plot_data, vmax=max_val, vmin=0, cmap='gist_heat')
        plt.xlabel("x (angstrom)")
        plt.ylabel("y (angstrom)")
        plt.colorbar()
        plt.axis('scaled')
        plt.savefig(args.output_dir+"/stm_ch_v%.2f_h%.1f.png"%(bias, plane_height), dpi=300)
        plt.close()


### -----------------------------------------
### Constant current STM
### -----------------------------------------

def get_isosurf(data, value, z_vals, interp=True):
    rev_data = data[:, :, ::-1]
    rev_z_vals = z_vals[::-1]

    indexes = np.argmax(rev_data > value, axis=2)

    # IF indexes are 0, then it probably didn't find the correct value
    # And set it as the bottom surface
    indexes[indexes == 0] = len(z_vals) - 1

    if interp:
        z_val_plane = np.ones(np.shape(rev_data)[0:2])*z_vals[0]
        for ix in range(np.shape(rev_data)[0]):
            for iy in range(np.shape(rev_data)[1]):
                ind = indexes[ix, iy]
                if ind == len(z_vals) - 1:
                    continue
                val_g = rev_data[ix, iy, ind]
                z_val_g = rev_z_vals[ind]
                val_s = rev_data[ix, iy, ind - 1]
                z_val_s = rev_z_vals[ind - 1]
                z_val_plane[ix, iy] = (value - val_s)/(val_g - val_s)*(z_val_g - z_val_s) + z_val_s
        return z_val_plane

    return rev_z_vals[indexes]

for isovalue in args.stm_isovalues:
    for i_bias, bias in enumerate(args.bias_voltages):
        const_cur_imag = get_isosurf(charge_dens_arr[i_bias], isovalue, total_z_arr, True)

        plt.figure(figsize=(7, 6))
        plot_data = const_cur_imag/ang_2_bohr
        max_val = np.max(plot_data)

        plt.pcolormesh(x_grid, y_grid, plot_data, vmax=max_val, cmap='gist_heat')
        plt.xlabel("x (angstrom)")
        plt.ylabel("y (angstrom)")
        plt.colorbar()
        plt.axis('scaled')
        plt.savefig(args.output_dir+"/stm_cc_v%.2f_i%.1e.png"%(bias, isovalue), dpi=300)
        plt.close()


### -----------------------------------------
### STS
### -----------------------------------------
e_arr = np.arange(elim[0], elim[1]+args.sts_de, args.sts_de)

def calculate_ldos(de, fwhm, plane_index, e_arr, broad_type='g'):
    def lorentzian(x):
        gamma = 0.5*fwhm
        return gamma/(np.pi*(x**2+gamma**2))
    def gaussian(x):
        sigma = fwhm/2.3548
        return np.exp(-x**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    pldos = np.zeros((eval_reg_size_n[0], eval_reg_size_n[1], len(e_arr)))
    for i_mo, morb_grid in enumerate(total_morb_grids):
        en = morb_energies[i_mo]
        morb_plane = morb_grid[:, :, plane_index]**2
        if broad_type == 'l':
            morb_ldos_broad = np.einsum('ij,k', morb_plane, lorentzian(e_arr - en))
        else:
            morb_ldos_broad = np.einsum('ij,k', morb_plane, gaussian(e_arr - en))
        pldos += morb_ldos_broad
    return pldos

for plane_height in args.sts_plane_heights:
    plane_index = get_plane_index(plane_height*ang_2_bohr, total_z_arr, dv[2])

    pldos = calculate_ldos(args.sts_de, args.sts_fwhm, plane_index, e_arr)

    for i, energy in enumerate(e_arr):
        plt.figure(figsize=(7, 6))
        plot_data = pldos[:, :, i]

        max_val = np.max(pldos)
        min_val = np.min(pldos)

        plt.pcolormesh(x_grid, y_grid, plot_data, vmax=max_val, vmin=min_val, cmap='bwr')
        plt.xlabel("x (angstrom)")
        plt.ylabel("y (angstrom)")
        plt.colorbar()
        plt.title("h = %.2f; U = %.3f" % (plane_height, energy))
        plt.axis('scaled')
        plt.savefig(args.output_dir+"/sts_h%.2f_nr%d.png"%(plane_height, i), dpi=300)
        plt.close()
