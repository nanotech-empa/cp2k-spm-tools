"""
Tools to perform FT-STS analysis on orbitals evaluated on grid
""" 

import os
import numpy as np
import scipy
import scipy.io
import scipy.special
import time
import copy
import sys

import re
import io
import ase
import ase.io

from .cp2k_grid_orbitals import Cp2kGridOrbitals

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

class FTSTS:
    """
    Class to perform FT-STS analysis on gridded orbitals
    """

    def __init__(self, cp2k_grid_orb):
        """
        Convert all lengths from [au] to [ang]
        """

        self.cp2k_grid_orb = cp2k_grid_orb
        self.nspin = cp2k_grid_orb.nspin
        self.mpi_rank = cp2k_grid_orb.mpi_rank
        self.mpi_size = cp2k_grid_orb.mpi_size
        self.cell_n = cp2k_grid_orb.eval_cell_n
        self.dv = cp2k_grid_orb.dv / ang_2_bohr
        self.origin = cp2k_grid_orb.origin / ang_2_bohr

        self.morbs_1d = None
        self.morb_fts = None
        self.k_arr = None
        self.dk = None

        self.ldos = None
        self.ftldos = None
        self.e_arr = None

        self.ldos_extent = None
        self.ftldos_extent = None

    def remove_row_average(self, ldos):
        ldos_no_avg = np.copy(ldos)
        for i in range(np.shape(ldos)[1]):
            ldos_no_avg[:, i] -= np.mean(ldos[:, i])
        return ldos_no_avg

    def add_padding(self, ldos, amount_factor):
        pad_n = int(amount_factor*ldos.shape[0])
        padded_ldos = np.zeros((np.shape(ldos)[0]+2*pad_n, np.shape(ldos)[1]))
        padded_ldos[pad_n:-pad_n] = ldos
        return padded_ldos

    def fourier_transform(self, ldos):

        ft = np.fft.rfft(ldos, axis=0)
        aft = np.abs(ft)

        # Corresponding k points
        k_arr = 2*np.pi*np.fft.rfftfreq(len(ldos[:, 0]), self.dv[0])
        # Note: Since we took the FT of the charge density, the wave vectors are
        #       twice the ones of the underlying wave function.
        #k_arr = k_arr / 2

        # Brillouin zone boundary [1/angstroms]
        #bzboundary = np.pi / lattice_param
        #bzb_index = int(np.round(bzboundary/dk))+1

        dk = k_arr[1]

        return k_arr, aft, dk
    
    def gaussian(self, x, fwhm):
        sigma = fwhm/2.3548
        return np.exp(-x**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

    def project_orbitals_1d(self, axis=0, gauss_pos=None, gauss_fwhm=5.0):

        self.morbs_1d = []

        cell_middle_point = self.origin + (self.dv*self.cell_n)/2.0

        avg_axis = [0, 1]
        avg_axis.remove(axis)

        for ispin in range(self.nspin):
            self.morbs_1d.append(np.zeros((self.cell_n[0], len(self.cp2k_grid_orb.morb_grids[ispin]))))
            for i_mo, morb_grid in enumerate(self.cp2k_grid_orb.morb_grids[ispin]):
                en = self.cp2k_grid_orb.morb_energies[ispin][i_mo]
                morb_plane = morb_grid[:, :, 0]
                if gauss_pos is None:
                    avg_morb = np.mean(morb_plane**2, axis=1)
                else:
                    raise NotImplementedError

                self.morbs_1d[ispin][:, i_mo] = avg_morb

    def take_fts(self, padding=1.0, remove_row_avg=True):

        self.morb_fts = []
        for ispin in range(self.nspin):
            if remove_row_avg:
                tmp_morbs = self.remove_row_average(self.morbs_1d[ispin])
            else:
                tmp_morbs = self.morbs_1d[ispin]
            tmp_morbs = self.add_padding(tmp_morbs, padding)
            self.k_arr, m_fts, self.dk = self.fourier_transform(tmp_morbs)
            self.morb_fts.append(m_fts)

    def make_ftldos(self, emin, emax, de, fwhm):
        
        self.e_arr = np.arange(emin, emax+de/2, de)

        self.ldos = np.zeros((self.cell_n[0], len(self.e_arr)))
        self.ftldos = np.zeros((len(self.k_arr), len(self.e_arr)))

        self.ldos_extent = [0.0, self.cell_n[0] * self.dv[0], emin, emax]
        self.ftldos_extent = [0.0, self.k_arr[-1], emin, emax]

        for ispin in range(self.nspin):
            for i_mo, en_mo in enumerate(self.cp2k_grid_orb.morb_energies[ispin]):
                # Produce LDOS
                self.ldos += np.outer(self.morbs_1d[ispin][:, i_mo], self.gaussian(self.e_arr - en_mo, fwhm))
                # Produce FTLDOS
                self.ftldos += np.outer(self.morb_fts[ispin][:, i_mo], self.gaussian(self.e_arr - en_mo, fwhm))

    def get_ftldos_bz(self, nbz, lattice_param):
        """
        Return part of previously calculated FTLDOS, which corresponds
        to the selected number of BZs (nbz) for specified lattice parameter (ang).
        """
        # Brillouin zone boundary [1/angstroms]
        bzboundary = np.pi / lattice_param
        nbzb_index = int(np.round(nbz*bzboundary/self.dk))+1

        return self.ftldos[:nbzb_index, :], [0.0, nbz*bzboundary, np.min(self.e_arr), np.max(self.e_arr)]







