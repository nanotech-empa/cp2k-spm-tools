
import os
import numpy as np
import time
import copy
import sys
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import cp2k_utilities as cu

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

parser = argparse.ArgumentParser(
    description='Processes 1d LDOS data by FT-STS.')

parser.add_argument(
    '--ldos_file',
    metavar='FILENAME',
    required=True,
    help='.txt file containing the ldos data.')
parser.add_argument(
    '--output_dir',
    metavar='FILENAME',
    required=True,
    help="Directory containing the output data.")

parser.add_argument(
    '--crop_dist_l',
    type=float,
    metavar='D',
    required=True,
    help="Crop distance from left side (either from the end or from a defect).")
parser.add_argument(
    '--crop_dist_r',
    type=float,
    metavar='D',
    required=True,
    help="Crop distance from right side (either from the end or from a defect).")
parser.add_argument(
    '--crop_defect_l',
    type=int,
    default=1,
    help="1 - crop distance wrt a defect; 0 - from the end; left side")
parser.add_argument(
    '--crop_defect_r',
    type=int,
    default=1,
    help="1 - crop distance wrt a defect; 0 - from the end; right side")


parser.add_argument(
    '--padding_x',
    type=float,
    default=300.0,
    help="Zero-padding added to the LDOS before Fourier transform.")
parser.add_argument(
    '--emin',
    type=float,
    default=-0.7,
    help="Min value of energy.")
parser.add_argument(
    '--emax',
    type=float,
    default=0.7,
    help="Max value of energy.")


time0 = time.time()
args = parser.parse_args()

def read_header(ldos_file_path):
    with open(ldos_file_path, 'r') as f:
        header = f.readline()
    vals = []
    for hpart in header.split(';'):
        vals.append(hpart.split(': ')[1].split('(')[0])

    geom_name = vals[0]
    height = float(vals[1])
    fwhm = float(vals[2])
    broad_type = vals[3]
    xmin = float(vals[4])
    xmax = float(vals[5])
    nx = int(vals[6])
    emin = float(vals[7])
    emax = float(vals[8])
    ne = int(vals[9])

    return geom_name, height, fwhm, broad_type, xmin, xmax, nx, emin, emax, ne

ldos_raw = np.loadtxt(args.ldos_file)
geom_name, height, fwhm, broad_type, xmin, xmax, nx, emin, emax, ne = read_header(args.ldos_file)

print("xrange:", xmin, xmax, nx)
print("erange:", emin, emax, ne)

x_arr_whole = np.linspace(xmin, xmax, nx)
e_arr_whole = np.linspace(emin, emax, ne)

dx = x_arr_whole[1]-x_arr_whole[0]

### -------------------------------------------------------
### Crop the LDOS in space
### -------------------------------------------------------

e_averaged = np.mean(ldos_raw, axis=1)
first_half = e_averaged[:len(e_averaged)//2]
second_half = e_averaged[len(e_averaged)//2:]

crop_x_l = args.crop_dist_l
if args.crop_defect_l != 0:
    index_l = np.argmax(first_half)
    crop_x_l = x_arr_whole[index_l] + args.crop_dist_l

crop_x_r = xmax - args.crop_dist_l
if args.crop_defect_r != 0:
    index_r = np.argmax(second_half) + len(first_half)
    crop_x_r = x_arr_whole[index_r] - args.crop_dist_l

# align cropping, such that remaining area is a multiple of lattice parameter (minus dx!)
lattice_param = 3*1.42
crop_len = crop_x_r - crop_x_l
crop_len_goal = np.round(crop_len/lattice_param)*lattice_param - dx
extra_shift = (crop_len_goal - crop_len)/2

crop_x_l -= extra_shift
crop_x_r += extra_shift

crop_l = int(np.round(crop_x_l/dx))
# shift the other end due to putting on grid error
crop_x_r += np.round(crop_x_l/dx)*dx - crop_x_l
crop_r = int(np.round(crop_x_r/dx))

x_arr = np.copy(x_arr_whole[crop_l:crop_r+1])
ldos = np.copy(ldos_raw[crop_l:crop_r+1])

### -------------------------------------------------------
### Crop the LDOS in energy
### -------------------------------------------------------

e_arr = np.copy(e_arr_whole)
if e_arr[0] < args.emin:
    index = np.argmax(e_arr>args.emin)-1
    e_arr = e_arr[index:]
    ldos = ldos[:, index:]
if e_arr[-1] > args.emax:
    index = np.argmax(e_arr>args.emax)+1
    e_arr = e_arr[:index]
    ldos = ldos[:, :index]

print("dx", dx)
align_check = (x_arr[-1]-x_arr[0])%lattice_param
print("alignment check:", align_check, align_check - lattice_param)

### -------------------------------------------------------
### Remove row avg and add padding
### -------------------------------------------------------

def remove_row_average(ldos):
    ldos_no_avg = np.copy(ldos)
    for i in range(np.shape(ldos)[1]):
        ldos_no_avg[:, i] -= np.mean(ldos[:, i])
    return ldos_no_avg

def add_padding(x_arr, ldos, padding_x, lattice_param):
    if padding_x <= 0.0:
        return x_arr, ldos
    dx = x_arr[1]-x_arr[0]
    init_len = x_arr[-1]-x_arr[0]

    # align resulting x-length to lattice param
    pad_len = init_len+2*padding_x
    pad_len_goal = np.round(pad_len/lattice_param)*lattice_param - dx
    padding_x = (pad_len_goal - init_len)/2

    pad_n_l = int(np.round(padding_x/dx))
    pad_x_l = pad_n_l*dx
    grid_shift = padding_x - pad_x_l
    pad_n_r = int(np.round((padding_x+grid_shift)/dx))
    pad_x_r = pad_n_r*dx

    padded_x_arr = np.arange(x_arr[0]-pad_x_l, x_arr[-1]+pad_x_r+1e-6, dx)
    padded_ldos = np.zeros((np.shape(ldos)[0]+pad_n_l+pad_n_r, np.shape(ldos)[1]))
    padded_ldos[pad_n_l:-pad_n_r] = ldos

    return padded_x_arr, padded_ldos

ldos = remove_row_average(ldos)

x_arr, ldos = add_padding(x_arr, ldos, args.padding_x, lattice_param)

align_check = (x_arr[-1]-x_arr[0])%lattice_param
print("alignment check:", align_check, align_check - lattice_param)

### -------------------------------------------------------
### Take the Fourier Transform
### -------------------------------------------------------

def fourier_transform(ldos, dx, lattice_param):

    ft = np.fft.rfft(ldos, axis=0)
    aft = np.abs(ft)

    # Corresponding k points
    k_arr = 2*np.pi*np.fft.rfftfreq(len(ldos[:, 0]), dx)
    # Note: Since we took the FT of the charge density, the wave vectors are
    #       twice the ones of the underlying wave function.
    k_arr = k_arr / 2

    # Lattice spacing for the ribbon = 3x c-c distance
    # Brillouin zone boundary [1/angstroms]
    bzboundary = np.pi / lattice_param

    dk = k_arr[1]
    bzb_index = int(np.round(bzboundary/dk))+1

    return k_arr, aft, dk, bzboundary, bzb_index

k_arr, aft, dk, bzboundary, bzb_index = fourier_transform(ldos, dx, lattice_param)

### -------------------------------------------------------
### Save FT data to npz
### -------------------------------------------------------

file_base = os.path.basename(args.ldos_file)
file_name, ext = os.path.splitext(file_base)
figname = file_name + "_crop%.0f%s_%.0f%s_e%.1f_%.1f_pad%.0f" % (
    args.crop_dist_l, 'f' if args.crop_defect_l == 0 else 't',
    args.crop_dist_r, 'f' if args.crop_defect_r == 0 else 't',
    args.emin, args.emax, args.padding_x)

np.savez(args.output_dir+figname,x_arr=k_arr, y_arr=e_arr, values=aft,
        x_label="k (1/angstrom)", y_label="E (eV)")

### -------------------------------------------------------
### Make plots
### -------------------------------------------------------


x_grid_whole, e_grid_whole = np.meshgrid(x_arr_whole, e_arr_whole, indexing='ij')
k_grid, e_k_grid = np.meshgrid(k_arr, e_arr, indexing='ij')

for gamma in [1.0, 0.5, 0.2]:

    f, (ax1, ax2) = plt.subplots(2, figsize=(18.0, 12.0))

    ax1.pcolormesh(x_grid_whole, e_grid_whole, ldos_raw,
                    norm=colors.PowerNorm(gamma=gamma),
                    vmax=np.max(ldos_raw))
    ax1.axvline(crop_x_l, color='r')
    ax1.axvline(crop_x_r, color='r')
    ax1.axhline(e_arr[0], color='r')
    ax1.axhline(e_arr[-1], color='r')
    ax1.set_xlabel("x (angstrom)")
    ax1.set_ylabel("E (eV)")

    ax2.pcolormesh(k_grid, e_k_grid, aft,
                    norm=colors.PowerNorm(gamma=gamma),
                    vmax=np.max(aft))
    ax2.set_ylim([np.min(e_arr), np.max(e_arr)])
    ax2.set_xlim([0.0, 2.0])
    ax2.set_xlabel("k (1/angstrom)")
    ax2.set_ylabel("E (eV)")

    plt.savefig(args.output_dir+figname+"_g%.1f.png"%gamma, dpi=300, bbox_inches='tight')
    plt.close()
