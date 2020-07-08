#!/usr/bin/env python
import os
import numpy as np
import time
import copy
import sys

import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cp2k_spm_tools import igor

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

fig_y = 4.0

title_font_size = 18

parser = argparse.ArgumentParser(
    description='Makes images from the STM .npz files.')

### ----------------------------------------------------------------------
### Input and output files
parser.add_argument(
    '--stm_npz',
    metavar='FILENAME',
    default=None,
    help='File containing STM data.')
parser.add_argument(
    '--orb_npz',
    metavar='FILENAME',
    default=None,
    help='File containing ORB data.')
parser.add_argument(
    '--output_dir',
    metavar='DIR',
    default="./",
    help='Output directory.')
### ----------------------------------------------------------------------

args = parser.parse_args()

def make_plot(fig, ax, data, extent, title=None, title_size=None, center0=False, vmin=None, vmax=None, cmap='gist_heat', noadd=False):
    if center0:
        data_amax = np.max(np.abs(data))
        im = ax.imshow(data.T, origin='lower', cmap=cmap, interpolation='bicubic', extent=extent, vmin=-data_amax, vmax=data_amax)
    else:
        im = ax.imshow(data.T, origin='lower', cmap=cmap, interpolation='bicubic', extent=extent, vmin=vmin, vmax=vmax)
    
    if noadd:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel(r"x ($\AA$)")
        ax.set_ylabel(r"y ($\AA$)")
        cb = fig.colorbar(im, ax=ax)
        cb.formatter.set_powerlimits((-2, 2))
        cb.update_ticks()
    ax.set_title(title, loc='left')
    if title_size:
        ax.title.set_fontsize(title_size)
    ax.axis('scaled')

def make_series_label(info, i_spin=None):
    if info['type'] == 'const-height sts':
        label = 'ch-sts fwhm=%.2f h=%.1f' % (info['fwhm'], info['height'])
    elif info['type'] == 'const-height stm':
        label = 'ch-stm fwhm=%.2f h=%.1f' % (info['fwhm'], info['height'])
    elif info['type'] == 'const-isovalue sts':
        label = 'cc-sts fwhm=%.2f isov=%.0e' % (info['fwhm'], info['isovalue'])
    elif info['type'] == 'const-isovalue stm':
        label = 'cc-stm fwhm=%.2f isov=%.0e' % (info['fwhm'], info['isovalue'])
    
    elif info['type'] == 'const-height orbital':
        label = 'ch-orb h=%.1f, s%d' % (info['height'], i_spin)
    elif info['type'] == 'const-height orbital^2':
        label = 'ch-orb^2 h=%.1f, s%d' % (info['height'], i_spin)
    elif info['type'] == 'const-isovalue orbital^2':
        label = 'cc-orb^2 isov=%.0e, s%d' % (info['isovalue'], i_spin)
    else:
        print("No support for: " + str(info))
    
    return label

def make_orb_label(index, homo_index):

    i_rel_homo = index - homo_index
    
    if i_rel_homo < 0:
        hl_label = "HOMO%+d" % i_rel_homo
    elif i_rel_homo == 0:
        hl_label = "HOMO"
    elif i_rel_homo == 1:
        hl_label = "LUMO"
    else:
        hl_label = "LUMO%+d" % (i_rel_homo-1)

    return "MO %d, " % index + hl_label 


def plot_series_and_export_igor(general_info, info, data, make_plot_args, plot_dir, itx_dir):

    e_arr = general_info['energies']
    x_arr = general_info['x_arr'] * 0.529177
    y_arr = general_info['y_arr'] * 0.529177

    orb_indexes = general_info.get('orb_indexes')
    homo = general_info.get('homo')
    spin = general_info.get('spin')

    extent = [np.min(x_arr), np.max(x_arr), np.min(y_arr), np.max(y_arr)]

    figure_xy_ratio = (np.max(x_arr)-np.min(x_arr)) / (np.max(y_arr)-np.min(y_arr))

    series_label = make_series_label(info, spin)

    for i_e, energy in enumerate(e_arr):

        # ---------------------------------------------------
        # Build labels, title and file name
        mo_label = None
        if orb_indexes is not None:
            mo_label =  make_orb_label(orb_indexes[i_e], homo)

        title = '%s\n' % series_label
        if mo_label is not None:
            title += mo_label + ' '
        title += 'E=%.2f eV' % energy

        plot_name = series_label.lower().replace(" ", '_').replace("=", '').replace('^', '').replace(',', '')
        if mo_label is not None:
            plot_name += "_mo%03d_e%.2f" % (orb_indexes[i_e], energy)
        else:
            plot_name += "_%03d_e%.2f" % (i_e, energy)

        # ---------------------------------------------------
        # Make the plot
        fig = plt.figure(figsize=(fig_y*figure_xy_ratio, fig_y))

        ax = plt.gca()
        make_plot(fig, ax, data[i_e, :, :], extent, title=title, title_size=title_font_size, noadd=False, **make_plot_args)

        plt.savefig(plot_dir + "/" + plot_name + ".png", dpi=200, bbox_inches='tight')
        plt.close()

        # ---------------------------------------------------
        # export IGOR format
        igorwave = igor.Wave2d(
                data=data[i_e, :, :],
                xmin=extent[0],
                xmax=extent[1],
                xlabel='x [Angstroms]',
                ymin=extent[2],
                ymax=extent[3],
                ylabel='y [Angstroms]',
        )
        igorwave.write(itx_dir + "/" + plot_name + ".itx")
        # ---------------------------------------------------


def plot_all_series(general_info, series_info, series_data, plot_dir, itx_dir):

    for info, data in zip(series_info, series_data):
        
        make_plot_args = {
            'cmap': 'gist_heat',
            'center0': False
        }

        if 'const-height sts' in info['type']:
            make_plot_args['cmap'] = 'seismic'
        elif 'const-height orb' in info['type']:

            make_plot_args['cmap'] = 'seismic'

            # ORB^2 export
            orb2_info = copy.deepcopy(info)
            orb2_info['type'] = 'const-height orbital^2'
            print("Plotting series: " + str(orb2_info))
            plot_series_and_export_igor(general_info, orb2_info, data**2, make_plot_args, plot_dir, itx_dir)

            make_plot_args['center0'] = True

        print("Plotting series: " + str(info))
        plot_series_and_export_igor(general_info, info, data, make_plot_args, plot_dir, itx_dir)

### ----------------------------------------------------------------------
### STM.NPZ
### ----------------------------------------------------------------------

if args.stm_npz is not None:

    stm_dir = args.output_dir + "./stm"
    if not os.path.exists(stm_dir):
        os.makedirs(stm_dir)
    
    stm_itx_dir = args.output_dir + "./stm_itx"
    if not os.path.exists(stm_itx_dir):
        os.makedirs(stm_itx_dir)

    loaded_data = np.load(args.stm_npz, allow_pickle=True)
 
    stm_general_info = loaded_data['stm_general_info'][()]
    stm_series_info = loaded_data['stm_series_info']
    stm_series_data = loaded_data['stm_series_data']

    plot_all_series(stm_general_info, stm_series_info, stm_series_data, stm_dir, stm_itx_dir)

### ----------------------------------------------------------------------
### ORB.NPZ
### ----------------------------------------------------------------------

if args.orb_npz is not None:

    orb_dir = args.output_dir + "./orb"
    if not os.path.exists(orb_dir):
        os.makedirs(orb_dir)

    orb_itx_dir = args.output_dir + "./orb_itx"
    if not os.path.exists(orb_itx_dir):
        os.makedirs(orb_itx_dir)

    loaded_data = np.load(args.orb_npz, allow_pickle=True)

    s0_orb_general_info = loaded_data['s0_orb_general_info'][()]
    s0_orb_series_info = loaded_data['s0_orb_series_info']
    s0_orb_series_data = loaded_data['s0_orb_series_data']

    plot_all_series(s0_orb_general_info, s0_orb_series_info, s0_orb_series_data, orb_dir, orb_itx_dir)

    if 's1_orb_general_info' in loaded_data.files:
        s1_orb_general_info = loaded_data['s1_orb_general_info'][()]
        s1_orb_series_info = loaded_data['s1_orb_series_info']
        s1_orb_series_data = loaded_data['s1_orb_series_data']

        plot_all_series(s1_orb_general_info, s1_orb_series_info, s1_orb_series_data, orb_dir, orb_itx_dir)

