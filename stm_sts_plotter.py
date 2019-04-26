#!/usr/bin/env python
import os
import numpy as np
import time
import copy
import sys

import argparse

import matplotlib.pyplot as plt
import matplotlib

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

fig_y = 4.0

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

class FormatScalarFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, fformat="%1.1f", offset=True, mathText=True):
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,
                                                        useMathText=mathText)
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)

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
        if 1e-3 < np.max(data) < 1e3:
            cb = fig.colorbar(im, ax=ax)
        else:
            cb = fig.colorbar(im, ax=ax, format=FormatScalarFormatter("%.1f"))
        cb.formatter.set_powerlimits((-2, 2))
        cb.update_ticks()
    ax.set_title(title, loc='left')
    if title_size:
        ax.title.set_fontsize(title_size)
    ax.axis('scaled')

### ----------------------------------------------------------------------
### STM.NPZ
### ----------------------------------------------------------------------

if args.stm_npz is not None:

    stm_dir = args.output_dir + "./stm"
    if not os.path.exists(stm_dir):
        os.makedirs(stm_dir)

    loaded_data = np.load(args.stm_npz)
 
    isovalues = loaded_data['isovalues']
    heights = loaded_data['heights']
    e_arr = loaded_data['e_arr']
    x_arr = loaded_data['x_arr'] * 0.529177
    y_arr = loaded_data['y_arr'] * 0.529177

    sts_cc = loaded_data['cc_sts']
    stm_cc = loaded_data['cc_stm'].astype(np.float32)
    sts_ch = loaded_data['ch_sts']
    stm_ch = loaded_data['ch_stm']

    ### ----------------------------------------------------
    ### Create series
    stm_series = {}
    for i_iv, iv in enumerate(isovalues):
        stm_series["cc-stm, isov=%.0e" % iv] = stm_cc[i_iv, :, :, :]
        stm_series["cc-sts, isov=%.0e" % iv] = sts_cc[i_iv, :, :, :]
    for i_h, h in enumerate(heights):
        stm_series["ch-stm, h=%.1f" % h] = stm_ch[i_h, :, :, :]
        stm_series["ch-sts, h=%.1f" % h] = sts_ch[i_h, :, :, :]
    ### ----------------------------------------------------

    extent = [np.min(x_arr), np.max(x_arr), np.min(y_arr), np.max(y_arr)]
    de = (np.max(e_arr) - np.min(e_arr)) / (len(e_arr)-1)

    figure_xy_ratio = (np.max(x_arr)-np.min(x_arr)) / (np.max(y_arr)-np.min(y_arr))

    for series_label, series in stm_series.items():
        for i_e, energy in enumerate(e_arr):
            fig = plt.figure(figsize=(fig_y*figure_xy_ratio, fig_y))

            cmap = 'gist_heat'
            if 'ch-sts' in series_label:
                cmap = 'seismic'

            title = '%s, E=%.2f eV'%(series_label, energy)
            data = stm_series[series_label]
            ax = plt.gca()
            make_plot(fig, ax, data[:, :, i_e], extent, title=title, cmap=cmap, noadd=False)
            
            series_name = series_label.lower().replace(" ", '_').replace("=", '').replace(",", '')
            plot_name = "/%s_%03de%.2f.png" % (series_name, i_e, energy)
            plt.savefig(stm_dir + plot_name, dpi=200, bbox_inches='tight')
            plt.close()


### ----------------------------------------------------------------------
### ORB.NPZ
### ----------------------------------------------------------------------

if args.orb_npz is not None:

    orb_dir = args.output_dir + "./orb"
    if not os.path.exists(orb_dir):
        os.makedirs(orb_dir)

    loaded_data = np.load(args.orb_npz)
 
    orbital_data = loaded_data['orbitals']
    heights = loaded_data['heights']
    orb_indexes = loaded_data['orb_list']
    energies = loaded_data['energies']
    x_arr = loaded_data['x_arr'] * 0.529177
    y_arr = loaded_data['y_arr'] * 0.529177
    
    nspin = len(orbital_data)

    ### ----------------------------------------------------
    ### Create series
    orbital_series = {}
    
    ### Labels for each image in the series
    labels = {}
    
    for i_spin in range(nspin):
        for i_h, h in enumerate(heights):
            series_1 = "orb h=%.1f, s%d" % (h, i_spin)
            series_2 = "orb^2 h=%.1f, s%d" % (h, i_spin)
            orbital_series[series_1] = orbital_data[i_spin, i_h, :, :, :]
            orbital_series[series_2] = orbital_data[i_spin, i_h, :, :, :]**2
            
            labels[series_1] = []
            labels[series_2] = []
            for i_loc, i_orb in enumerate(orb_indexes):
                if i_orb <= 0:
                    label = "HOMO%+d E=%.4f eV" % (i_orb, energies[i_spin][i_loc])
                else:
                    label = "LUMO%+d E=%.4f eV" % (i_orb-1, energies[i_spin][i_loc])
                labels[series_1].append(label)
                labels[series_2].append(label)
    
    ### ----------------------------------------------------
    extent = [np.min(x_arr), np.max(x_arr), np.min(y_arr), np.max(y_arr)]

    figure_xy_ratio = (np.max(x_arr)-np.min(x_arr)) / (np.max(y_arr)-np.min(y_arr))

    for series_label, series in orbital_series.items():
        for i_orb_loc, i_orb in enumerate(orb_indexes):
            fig = plt.figure(figsize=(fig_y*figure_xy_ratio, fig_y))

            cmap = 'gist_heat'

            title = '%s\n%s'%(series_label, labels[series_label][i_orb_loc])
            data = orbital_series[series_label]
            ax = plt.gca()

            if "orb " in series_label:
                cmap = 'seismic'
                center0 = True
            else:
                cmap = 'gist_heat'
                center0 = False

            make_plot(fig, ax, data[i_orb_loc, :, :], extent, title=title, center0=center0, cmap=cmap, noadd=False)

            series_name = series_label + "_%02d" % i_orb_loc + labels[series_label][i_orb_loc]
            plot_name = "/" + series_name.lower().replace(" ", '_').replace("=", '').replace('^', '').replace(',', '') + ".png"
            plt.savefig(orb_dir + plot_name, dpi=200, bbox_inches='tight')
            plt.close()
