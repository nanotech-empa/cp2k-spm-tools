import os
import numpy as np
import time
import copy
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

import argparse

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

import atomistic_tools.cp2k_stm_utilities as csu

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
parser.add_argument(
    '--sts_elim',
    nargs='*',
    type=float,
    help=("Energy limits (emin, emax) for STS (eV)."
          "Default is to take the whole calculated energy range.")
)
parser.add_argument(
    '--orb_plane_heights',
    nargs='*',
    type=float,
    metavar='H',
    help="List of heights for orbital evaluation. (angstroms)")
parser.add_argument(
    '--n_homo',
    type=int,
    metavar='N',
    help="Number of HOMO orbitals to evaluate.")
parser.add_argument(
    '--n_lumo',
    type=int,
    metavar='N',
    help="Number of LUMO orbitals to evaluate.")


parser.add_argument(
    '--skip_data_output',
    action='store_true',
    help="Switch to skip outputting the STM picture data to files.")
parser.set_defaults(skip_data_output=False)
parser.add_argument(
    '--skip_figs',
    action='store_true',
    help="Switch to skip outputting figures.")
parser.set_defaults(skip_figs=False)


args = parser.parse_args()

time0 = time.time()

npz_file_data = np.load(args.npz_file)

x_arr = npz_file_data['x_arr']
y_arr = npz_file_data['y_arr']
z_arr = npz_file_data['z_arr']
mol_bbox = npz_file_data['mol_bbox']
elim = npz_file_data['elim']
ref_energy = npz_file_data['ref_energy']
geom_label = npz_file_data['geom_label']

morb_grids = [npz_file_data['morb_grids_s1']]
morb_energies = [npz_file_data['morb_energies_s1']]
homo_inds = [npz_file_data['homo_s1']]

if 'morb_grids_s2' in npz_file_data:
    morb_grids.append(npz_file_data['morb_grids_s2'])
    morb_energies.append(npz_file_data['morb_energies_s2'])
    homo_inds.append(npz_file_data['homo_s2'])

eval_reg_size = np.array([x_arr[-1] - x_arr[0], y_arr[-1] - y_arr[0], z_arr[-1] - z_arr[0]])
eval_reg_size_n = morb_grids[0][0].shape

if eval_reg_size_n[2] == 1:
    print("Note: only a single plane was evaluated, const-current output will be skipped.")
    dv = np.array([x_arr[1] - x_arr[0], y_arr[1] - y_arr[0], x_arr[1] - x_arr[0]])
else:
    dv = np.array([x_arr[1] - x_arr[0], y_arr[1] - y_arr[0], z_arr[1] - z_arr[0]])

nspin = len(morb_grids)

# z_arr with respect to topmost atom
z_arr -= mol_bbox[5]

x_grid, y_grid = np.meshgrid(x_arr, y_arr, indexing='ij')
x_grid /= ang_2_bohr
y_grid /= ang_2_bohr

def get_plane_index(z, z_arr, dz):
    return int(np.round((z-z_arr[0])/dz))

# Size of all the figures:
figure_size = 4
figure_size_xy = (figure_size*eval_reg_size[0]/eval_reg_size[1]+1.0, figure_size)

### -----------------------------------------
### EXTRAPOLATION
### -----------------------------------------

time1 = time.time()
hart_cube_data = csu.read_cube_file(args.hartree_file)
print("Read hartree: %.3f" % (time.time()-time1))

# Get Hartree plane, convert to eV and shift by ref (so it matches morb_energies) 
hart_plane = csu.get_hartree_plane_above_top_atom(hart_cube_data, args.extrap_plane)*hart_2_ev - ref_energy

print("Hartree on extrapolation plane: min: %.4f; max: %.4f; avg: %.4f (eV)" % (
                                                np.min(hart_plane),
                                                np.max(hart_plane),
                                                np.mean(hart_plane)))

if not args.skip_figs:
    plt.figure(figsize=figure_size_xy)
    plt.pcolormesh(hart_plane.T, cmap='seismic')
    plt.colorbar()
    plt.axis('scaled')
    plt.savefig(args.output_dir+"/hartree.png", dpi=300, bbox_inches='tight')
    plt.close()


extrap_plane_index = get_plane_index(args.extrap_plane*ang_2_bohr, z_arr, dv[2])
if extrap_plane_index >= eval_reg_size_n[2]:
    print("Error: the extrapolation plane can't be outside the initial box (z_max = %.2f)"
           % (z_arr[-1]/ang_2_bohr))
    exit(1)

total_morb_grids = []
for ispin in range(nspin):
    extrap_morbs = csu.extrapolate_morbs(morb_grids[ispin][:, :, :, extrap_plane_index],
                                    morb_energies[ispin], dv,
                                    args.extrap_extent*ang_2_bohr, False,
                                    hart_plane=hart_plane/hart_2_ev,
                                    use_weighted_avg=True)

    total_morb_grid = np.concatenate((morb_grids[ispin], extrap_morbs), axis=3)
    total_morb_grids.append(total_morb_grid)

extended_region_n = np.shape(total_morb_grids[0])

# In bohr and wrt topmost atom
total_z_arr = np.arange(0.0, extended_region_n[3]*dv[2], dv[2]) + z_arr[0]

### -----------------------------------------
### Plotting methods
### -----------------------------------------

class FormatScalarFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, fformat="%1.1f", offset=True, mathText=True):
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,
                                                        useMathText=mathText)
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)


def make_plot(data, fpath, title=None, center0=False, vmin=None, vmax=None, cmap='gist_heat'):
    if not isinstance(data, (list,)):
        data = [data]
    if not isinstance(center0, (list,)):
        center0 = [center0]
    if not isinstance(title, (list,)):
        title = [title]

    plt.figure(figsize=(figure_size_xy[0], len(data)*figure_size_xy[1]))
    for i, data_e in enumerate(data):
        plt.subplot(len(data), 1, i+1)
        if center0[i]:
            data_amax = np.max(np.abs(data_e))
            plt.pcolormesh(x_grid, y_grid, data_e, vmin=-data_amax, vmax=data_amax, cmap=cmap)
        else:
            plt.pcolormesh(x_grid, y_grid, data_e, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.xlabel("x (angstrom)")
        plt.ylabel("y (angstrom)")
        if 1e-3 < np.max(data) < 1e3:
            cb = plt.colorbar()
        else:
            cb = plt.colorbar(format=FormatScalarFormatter("%.1f"))
        cb.formatter.set_powerlimits((-2, 2))
        cb.update_ticks()
        if i < len(title):
            plt.title(title[i], loc='left')
    plt.axis('scaled')
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close()

def make_series_plot(data, fpath):
    plt.figure(figsize=(figure_size_xy[1]*len(args.bias_voltages), figure_size_xy[0]))
    for i_bias, bias in enumerate(args.bias_voltages):
        plt.subplot(1, len(args.bias_voltages), i_bias+1)
        plt.pcolormesh(x_grid, y_grid, data[:, :, i_bias], cmap='gist_heat')
        plt.axis('scaled')
        plt.title("V=%.2f"%bias)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close()

### -----------------------------------------
### Saving HOMO and LUMO
### -----------------------------------------

for i_height, plane_height in enumerate(args.orb_plane_heights):
    plane_index = get_plane_index(plane_height*ang_2_bohr, total_z_arr, dv[2])
    if plane_index < 0:
        print("Height %.1f is outside evaluation range, skipping." % plane_height)
        continue
    if not args.skip_figs:
        for ispin in range(nspin):
            for i_h in range(-args.n_homo+1, args.n_lumo+1):
                ind = homo_inds[ispin] + i_h
                if ind > len(morb_energies[ispin]) - 1:
                    print("Homo %d is out of energy range, ignoring" % i_h)
                    break
                fpath = args.output_dir+"/orb_h%.1f_s%d_%02dhomo%d.png"%(plane_height, ispin, i_h+args.n_homo-1, i_h)
                title = "homo %d, E=%.6f" % (i_h, morb_energies[ispin][ind])
                plot_data = total_morb_grids[ispin][ind][:, :, plane_index]
                make_plot([plot_data, plot_data**2], fpath, title=[title, "square"], center0=[True, False], cmap='seismic')

### -----------------------------------------
### Summing charge densities according to bias voltages
### -----------------------------------------

# Sum up both spins

charge_dens_arr = np.zeros((len(args.bias_voltages),
                            extended_region_n[1],
                            extended_region_n[2],
                            extended_region_n[3]))

# NB Homo is only added to negative bias !
for i_bias, bias in enumerate(args.bias_voltages):
    for ispin in range(nspin):
        for imo, morb_grid in enumerate(total_morb_grids[ispin]):
            if morb_energies[ispin][imo] > np.max([0.0, bias]):
                break
            if morb_energies[ispin][imo] > np.min([0.0, bias]):
                charge_dens_arr[i_bias, :, :, :] += morb_grid**2

### -----------------------------------------
### Constant height STM
### -----------------------------------------

time1 = time.time()

const_height_data = np.zeros((
    len(args.stm_plane_heights),
    x_grid.shape[0], x_grid.shape[1],
    len(args.bias_voltages)))

for i_height, plane_height in enumerate(args.stm_plane_heights):
    for i_bias, bias in enumerate(args.bias_voltages):
        plane_index = get_plane_index(plane_height*ang_2_bohr, total_z_arr, dv[2])
        const_height_data[i_height, :, :, i_bias] = charge_dens_arr[i_bias][:, :, plane_index]

    if not args.skip_figs:
        for i_bias, bias in enumerate(args.bias_voltages):
            fpath = args.output_dir+"/stm_ch_v%.2f_h%.1f.png"%(bias, plane_height)
            title = "h = %.1f; V = %.2f" % (plane_height, bias)
            make_plot(const_height_data[i_height, :, :, i_bias], fpath, title=title)
        
        fpath = args.output_dir+"/stm_ch_h%.1f.png"%(plane_height)
        make_series_plot(const_height_data[i_height], fpath)

if not args.skip_data_output:
    fpath = args.output_dir+"/stm_ch.npz"
    np.savez(fpath, data=const_height_data, heights=args.stm_plane_heights, bias=args.bias_voltages,
        x=x_arr/ang_2_bohr, y=y_arr/ang_2_bohr)

print("Time taken to create CH images: %.1f" % (time.time() - time1))

### -----------------------------------------
### Constant current STM
### -----------------------------------------

def get_isosurf(data, value, z_vals, interp=True):
    rev_data = data[:, :, ::-1]
    rev_z_vals = z_vals[::-1]

    # Add a zero-layer at start to make sure we surpass it
    zero_layer = np.zeros((data.shape[0], data.shape[1], 1))
    rev_data = np.concatenate((zero_layer, rev_data), axis=2)
    rev_z_vals = np.concatenate(([10.0], rev_z_vals))

    # Find first index that surpasses the isovalue
    indexes = np.argmax(rev_data > value, axis=2)
    # If an index is 0, no values in array are bigger than the specified
    num_surpasses = (indexes == 0).sum()
    if num_surpasses != 0:
        print("Warning: The isovalue %.3e was not reached for %d pixels" % (value, num_surpasses))
    # Set surpasses as the bottom surface
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

if eval_reg_size_n[2] != 1:

    time1 = time.time()

    const_curr_data = np.zeros((
        len(args.stm_isovalues),
        x_grid.shape[0], x_grid.shape[1],
        len(args.bias_voltages)))

    for i_iso, isovalue in enumerate(args.stm_isovalues):
        for i_bias, bias in enumerate(args.bias_voltages):
            const_cur_imag = get_isosurf(charge_dens_arr[i_bias], isovalue, total_z_arr, True)
            const_curr_data[i_iso, :, :, i_bias] = const_cur_imag/ang_2_bohr

        if not args.skip_figs:
            for i_bias, bias in enumerate(args.bias_voltages):
                fpath = args.output_dir+"/stm_cc_v%.2f_i%.1e.png"%(bias, isovalue)
                title = "isov = %.1e; V = %.2f" % (isovalue, bias)
                make_plot(const_curr_data[i_iso, :, :, i_bias], fpath, title=title)
            
            fpath = args.output_dir+"/stm_cc_i%.1e.png"%(isovalue)
            make_series_plot(const_curr_data[i_iso], fpath)

    if not args.skip_data_output:
        fpath = args.output_dir+"/stm_cc.npz"
        np.savez(fpath, data=const_curr_data, isovals=args.stm_isovalues, bias=args.bias_voltages,
            x=x_arr/ang_2_bohr, y=y_arr/ang_2_bohr)

    print("Time taken to create CC images: %.1f" % (time.time() - time1))

### -----------------------------------------
### STS
### -----------------------------------------

time1 = time.time()

if len(args.sts_elim) != 2:
    e_arr = np.arange(elim[0], elim[1]+args.sts_de, args.sts_de)
else:
    e_arr = np.arange(args.sts_elim[0], args.sts_elim[1]+args.sts_de, args.sts_de)

def calculate_ldos(de, fwhm, plane_index, e_arr, broad_type='g'):
    def lorentzian(x):
        gamma = 0.5*fwhm
        return gamma/(np.pi*(x**2+gamma**2))
    def gaussian(x):
        sigma = fwhm/2.3548
        return np.exp(-x**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    pldos = np.zeros((eval_reg_size_n[0], eval_reg_size_n[1], len(e_arr)))
    for ispin in range(nspin):
        for i_mo, morb_grid in enumerate(total_morb_grids[ispin]):
            en = morb_energies[ispin][i_mo]
            morb_plane = morb_grid[:, :, plane_index]**2
            if broad_type == 'l':
                morb_ldos_broad = np.einsum('ij,k', morb_plane, lorentzian(e_arr - en))
            else:
                morb_ldos_broad = np.einsum('ij,k', morb_plane, gaussian(e_arr - en))
            pldos += morb_ldos_broad
    return pldos

sts_data = np.zeros((
    len(args.sts_plane_heights),
    x_grid.shape[0], x_grid.shape[1],
    len(e_arr)))

for i_height, plane_height in enumerate(args.sts_plane_heights):
    plane_index = get_plane_index(plane_height*ang_2_bohr, total_z_arr, dv[2])

    pldos = calculate_ldos(args.sts_de, args.sts_fwhm, plane_index, e_arr)
    sts_data[i_height, :, :, :] = pldos

    if not args.skip_figs:
        max_val = np.max(pldos)
        min_val = np.min(pldos)

        for i, energy in enumerate(e_arr):
            fpath = args.output_dir+"/sts_h%.2f_nr%d.png"%(plane_height, i)
            title = "h = %.2f; U = %.3f" % (plane_height, energy)
            make_plot(pldos[:, :, i], fpath, title=title, vmin=min_val, vmax=max_val,  cmap='seismic')

if not args.skip_data_output:
    fpath = args.output_dir+"/sts.npz"
    np.savez(fpath, data=sts_data, heights=args.sts_plane_heights, e=e_arr,
        x=x_arr/ang_2_bohr, y=y_arr/ang_2_bohr)

print("Time taken to create STS images: %.1f" % (time.time() - time1))

print("Total time taken %.1f s" % (time.time() - time0))
