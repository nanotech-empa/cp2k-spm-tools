
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
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch


import cp2k_stm_utilities as csu

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

parser = argparse.ArgumentParser(
    description='Analyzes orbitals.')

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
    '--emin',
    type=float,
    default=-1.0,
    help="Min value of energy.")
parser.add_argument(
    '--emax',
    type=float,
    default=2.0,
    help="Max value of energy.")
parser.add_argument(
    '--crop_x_l',
    type=float,
    metavar='D',
    default=254.82,
    help="Crop coordinate from left before FT")
parser.add_argument(
    '--crop_x_r',
    type=float,
    metavar='D',
    default=365.36,
    help="Crop coordinate from right before FT")
parser.add_argument(
    '--work_function',
    type=float,
    metavar='WF',
    default=None,
    help="Work function in eV of the system (fermi and vacuum level difference)." \
         "For extrapolation either this or the hartree file is needed.")
parser.add_argument(
    '--hartree_file',
    metavar='FILENAME',
    default=None,
    help="Cube file containing the hartree potential." \
         "Only needed if sts_plane_height is out of the morb eval region.")
parser.add_argument(
    '--lat_param',
    type=float,
    metavar='A',
    required=True,
    help="Lattice parameter of the system (needed for plotting).")
parser.add_argument(
    '--ftsts_de',
    type=float,
    metavar='DE',
    default=0.005,
    help="Energy step for FTLDOS.")
parser.add_argument(
    '--ftsts_fwhm',
    type=float,
    metavar='FWHM',
    default=0.03,
    help="Energy broadening for the FTLDOS.")


args = parser.parse_args()

time0 = time.time()

output_dir = args.output_dir
if output_dir[-1] != '/':
    output_dir += '/'



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


plane_index = get_plane_index(args.sts_plane_height*ang_2_bohr, z_arr, dv[2])

morb_planes = []
num_morbs = []

for ispin in range(nspin):
    num_morbs_s = np.shape(morb_grids[ispin])[0]
    num_morbs.append(num_morbs_s)

    if plane_index > len(z_arr) - 1:
        # Extrapolation is needed!

        extrap_plane_index = len(z_arr) - 1

        if args.work_function != None:
            extrap_planes = csu.extrapolate_morbs(morb_grids[ispin][:, :, :, extrap_plane_index],
                                                morb_energies[ispin], dv,
                                                args.sts_plane_height*ang_2_bohr, True,
                                                work_function=args.work_function/hart_2_ev)
        elif args.hartree_file != None:

            time1 = time.time()
            hart_cube_data = csu.read_cube_file(args.hartree_file)
            print("Read hartree: %.3f" % (time.time()-time1))

            # Get Hartree plane, convert to eV and shift by ref (so it matches morb_energies) 
            hart_plane = csu.get_hartree_plane_above_top_atom(hart_cube_data, z_arr[-1])*hart_2_ev - ref_energy

            print("Hartree on extrapolation plane: min: %.4f; max: %.4f; avg: %.4f (eV)" % (
                                                            np.min(hart_plane),
                                                            np.max(hart_plane),
                                                            np.mean(hart_plane)))
            
            extrap_planes = csu.extrapolate_morbs(morb_grids[ispin][:, :, :, extrap_plane_index],
                                                morb_energies[ispin], dv,
                                                args.sts_plane_height*ang_2_bohr, True,
                                                hart_plane=hart_plane/hart_2_ev, use_weighted_avg=True)
        else:
            print("Work function or Hartree potential must be supplied if STS plane is out of region")
            exit()
        
        morb_planes.append(extrap_planes)

    else:
        morb_planes_s = np.zeros((num_morbs[ispin], eval_reg_size_n[0], eval_reg_size_n[1]))
        for i_mo in range(num_morbs[ispin]):
            morb_planes_s[i_mo, :, :] =  morb_grids[ispin][i_mo, :, :, plane_index]

        morb_planes.append(morb_planes_s)




for ispin in range(nspin):

    ### ----------------------------------------------------------------
    ### Select orbitals
    ### ----------------------------------------------------------------
    time1 = time.time()

    i_homo = homo_inds[ispin]

    i_first = np.argmax(morb_energies[ispin] > args.emin)
    i_last = np.argmax(morb_energies[ispin] > args.emax) - 1
    if i_last == -1:
        i_last = len(morb_energies[ispin]) -1
    i_selected = np.arange(i_first, i_last+1, 1)
    n_selected = len(i_selected)
    n_homo = i_homo - i_first + 1
    n_lumo = i_last - i_homo


    sel_morbs = np.zeros((eval_reg_size_n[0], n_selected*eval_reg_size_n[1]))

    for i, i_mo in enumerate(i_selected):
        sel_morbs[:, i*eval_reg_size_n[1]:(i+1)*eval_reg_size_n[1]] = morb_planes[ispin][i_mo]

    y_arr_inc = np.linspace(0, n_selected*eval_reg_size[1], n_selected*eval_reg_size_n[1])
    x_grid_inc, y_grid_inc = np.meshgrid(x_arr, y_arr_inc, indexing='ij')


    print("Selected orbitals: %.3f" % (time.time()-time1))

    ### ----------------------------------------------------------------
    ### Average orbitals
    ### ----------------------------------------------------------------

    time1 = time.time()

    sel_morbs_avg = np.zeros((eval_reg_size_n[0], n_selected))

    for i, i_mo in enumerate(i_selected):
        sel_morbs_avg[:, i] = np.mean(morb_planes[ispin][i_mo]**2, axis=1) 

    print("Averaged orbitals: %.3f" % (time.time()-time1))


    ### ----------------------------------------------------------------
    ### FT functions
    ### ----------------------------------------------------------------


    def remove_row_average(ldos):
        ldos_no_avg = np.copy(ldos)
        for i in range(np.shape(ldos)[1]):
            ldos_no_avg[:, i] -= np.mean(ldos[:, i])
        return ldos_no_avg

    def add_padding(ldos, dx, amount):
        pad_n = int(amount//dx)
        padded_ldos = np.zeros((np.shape(ldos)[0]+2*pad_n, np.shape(ldos)[1]))
        padded_ldos[pad_n:-pad_n] = ldos
        return padded_ldos

    def fourier_transform(ldos, dx, lattice_param):

        ft = np.fft.rfft(ldos, axis=0)
        aft = np.abs(ft)

        # Corresponding k points
        k_arr = 2*np.pi*np.fft.rfftfreq(len(ldos[:, 0]), dx)
        # Note: Since we took the FT of the charge density, the wave vectors are
        #       twice the ones of the underlying wave function.
        #k_arr = k_arr / 2

        # Lattice spacing for the ribbon = 3x c-c distance
        # Brillouin zone boundary [1/angstroms]
        bzboundary = np.pi / lattice_param

        dk = k_arr[1]
        bzb_index = int(np.round(bzboundary/dk))+1

        return k_arr, aft, dk, bzboundary, bzb_index


    ### ----------------------------------------------------------------
    ### Crop and take FT
    ### ----------------------------------------------------------------
    time1 = time.time()

    dx_ang = (x_arr[1]-x_arr[0])/ang_2_bohr

    crop_i_l = int(args.crop_x_l//dx_ang)
    if args.crop_x_r < 0.0:
        crop_i_r = -1
    else:
        crop_i_r = int(args.crop_x_r//dx_ang)

    print("aaaa", crop_i_l, crop_i_r)
    print("SMA shape", sel_morbs_avg.shape)

    modif_sel_morbs_avg = np.copy(sel_morbs_avg[crop_i_l:crop_i_r, :])
    modif_sel_morbs_avg = remove_row_average(modif_sel_morbs_avg)
    modif_sel_morbs_avg = add_padding(modif_sel_morbs_avg, dx_ang, 400.0)
    k_arr, aft, dk, bzboundary, bzb_index = fourier_transform(modif_sel_morbs_avg, dx_ang, args.lat_param)

    k_grid_inc, y_k_grid_inc = np.meshgrid(k_arr, y_arr_inc, indexing='ij')

    print("Took Fourier transform: %.3f" % (time.time()-time1))

    ### ----------------------------------------------------------------
    ### Energy broadening
    ### ----------------------------------------------------------------

    time1 = time.time()

    def gaussian(x, fwhm):
        sigma = fwhm/2.3548
        return np.exp(-x**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

    y_arr = np.linspace(0, eval_reg_size[1], eval_reg_size_n[1])

    broadened_sel_morbs_avg = np.zeros((eval_reg_size_n[0], n_selected*eval_reg_size_n[1]))
    broadened_sel_morbs_ft = np.zeros((aft.shape[0], n_selected*eval_reg_size_n[1]))

    de = args.ftsts_de
    fwhm = args.ftsts_fwhm
    e_arr = np.arange(args.emin, args.emax+de, de)

    k_grid, e_grid = np.meshgrid(k_arr, e_arr, indexing='ij')

    ftldos = np.zeros((len(k_arr), len(e_arr)))

    for i, i_mo in enumerate(i_selected):

        # Broaden the isolated morbs artificially
        broadened_sel_morbs_avg[:, i*eval_reg_size_n[1]:(i+1)*eval_reg_size_n[1]] = np.outer(
            sel_morbs_avg[:, i], gaussian(y_arr - eval_reg_size[1]/2, eval_reg_size[1]/6))
        broadened_sel_morbs_ft[:, i*eval_reg_size_n[1]:(i+1)*eval_reg_size_n[1]] = np.outer(
            aft[:, i], gaussian(y_arr - eval_reg_size[1]/2, eval_reg_size[1]/6))

        # Produce LDOS
        en = morb_energies[ispin][i_mo]
        morb_ftldos_broad = np.outer(aft[:, i], gaussian(e_arr - en, fwhm))
        ftldos += morb_ftldos_broad

    print("Energy broadening: %.3f" % (time.time()-time1))

    ### ----------------------------------------------------------------
    ### Plot all
    ### ----------------------------------------------------------------

    time1 = time.time()

    fig_x_size = 5*12
    fig_y_size = int(eval_reg_size_n[1]/eval_reg_size_n[0]*12*len(i_selected))
    max_size = (2**16-1)/200
    if fig_y_size > max_size:
        fig_x_size = int(fig_x_size*max_size/fig_y_size)
        fig_y_size = max_size

    fig = plt.figure(figsize=(fig_x_size, fig_y_size))

    gs1 = gridspec.GridSpec(1, 3, wspace=0.0, hspace=0.0)
    gs1.update(left=0.0, right=0.6, wspace=0.0)

    ax1 = fig.add_subplot(gs1[0, 0])
    ax2 = fig.add_subplot(gs1[0, 1])
    ax3 = fig.add_subplot(gs1[0, 2])

    gs2 = gridspec.GridSpec(1, 1, wspace=0.0, hspace=0.0)
    gs2.update(left=0.61, right=1.0, wspace=0.0)

    ax4 = fig.add_subplot(gs2[0, 0])

    max_val = np.max(np.abs(sel_morbs))
    ax1.pcolormesh(x_grid_inc, y_grid_inc, sel_morbs, vmax=max_val, vmin=-max_val, cmap='seismic') # seismic bwr
    ax1.xaxis.set_visible(False)
    ax1.axhline(n_homo*eval_reg_size[1], color='lightgray')

    max_val = np.max(broadened_sel_morbs_avg)
    ax2.pcolormesh(x_grid_inc, y_grid_inc, broadened_sel_morbs_avg, vmax=max_val, cmap='gist_ncar', norm=colors.PowerNorm(gamma=0.5))
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.axvline(args.crop_x_l*ang_2_bohr, color='lightgray')
    ax2.axvline(args.crop_x_r*ang_2_bohr, color='lightgray')
    ax2.axhline(n_homo*eval_reg_size[1], color='lightgray')

    max_val = np.max(broadened_sel_morbs_ft)
    ax3.pcolormesh(k_grid_inc, y_k_grid_inc, broadened_sel_morbs_ft, vmax=max_val, cmap='gist_ncar')
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax3.axhline(n_homo*eval_reg_size[1], color='lightgray')
    ax3.set_xlim([0.0, 5*bzboundary])

    max_val = np.max(ftldos)
    ax4.pcolormesh(k_grid, e_grid, ftldos, vmax=max_val, cmap='gist_ncar')
    ax4.axhline(0.0, color='lightgray')
    ax4.set_xlim([0.0, 5*bzboundary])
    ax4.set_yticks(np.arange(np.round(args.emin, 1), np.round(args.emax, 1), 0.05))

    # Add index and energy on left side
    ytick_pos = np.linspace(0.5*eval_reg_size[1], (len(i_selected)-0.5)*eval_reg_size[1], len(i_selected))
    ytick_labels = ["HOMO%+d e=%.4f" % (ind-i_homo, morb_energies[ispin][ind]) for ind in i_selected]
    ax1.set_yticks(ytick_pos)
    ax1.set_yticklabels(ytick_labels)

    # connect individual orbitals to the FT-LDOS

    for i, i_mo in enumerate(i_selected):
        xy1 = (4.95*bzboundary, ytick_pos[i])
        xy2 = (0.0, morb_energies[ispin][i_mo])
        con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="data",
                        axesA=ax3, axesB=ax4, color="red")
        ax3.add_artist(con)
        ax3.text(4.95*bzboundary, ytick_pos[i], "%.3f"%morb_energies[ispin][i_mo], color='red', ha='right', va='center')

        ax4.axhline(morb_energies[ispin][i_mo], color='red', alpha=0.3)

    plt.savefig(output_dir+"orb_analysis_%s_h%.1f_s%d.png"%(geom_label, args.sts_plane_height, ispin), dpi=200, bbox_inches='tight')
    plt.close()

    print("Final plot: %.3f" % (time.time()-time1))
