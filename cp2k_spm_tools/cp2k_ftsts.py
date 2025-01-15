"""
Tools to perform FT-STS analysis on orbitals evaluated on grid
"""

import numpy as np

ang_2_bohr = 1.0 / 0.52917721067
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
        for i in range(ldos.shape[1]):
            ldos_no_avg[:, i] -= np.mean(ldos[:, i])
        return ldos_no_avg

    def add_padding(self, ldos, amount_factor):
        # assumes that first index is space
        pad_n = int(amount_factor * ldos.shape[0])
        if pad_n == 0:
            return ldos
        padded_ldos = np.zeros((np.shape(ldos)[0] + 2 * pad_n, np.shape(ldos)[1]))
        padded_ldos[pad_n:-pad_n] = ldos
        return padded_ldos

    def crop_padding(self, ldos, tol=1e-6):
        # assumes that first index is space
        max_for_every_x = np.max(ldos, axis=1)
        i_crop_1 = np.argmax(max_for_every_x > tol)
        i_crop_2 = len(max_for_every_x) - np.argmax(max_for_every_x[::-1] > tol)

        return ldos[i_crop_1:i_crop_2], i_crop_1, i_crop_2

    def crop_edges(self, ldos, dist=10.0):
        crop_index = int(np.round(dist / self.dv[0]))
        return ldos[crop_index:-crop_index]

    def fourier_transform(self, ldos):
        ft = np.fft.rfft(ldos, axis=0)
        aft = np.abs(ft)

        # Corresponding k points
        k_arr = 2 * np.pi * np.fft.rfftfreq(len(ldos[:, 0]), self.dv[0])
        # Note: Since we took the FT of the charge density, the wave vectors are
        #       twice the ones of the underlying wave function.
        # k_arr = k_arr / 2

        # Brillouin zone boundary [1/angstroms]
        # bzboundary = np.pi / lattice_param
        # bzb_index = int(np.round(bzboundary/dk))+1

        dk = k_arr[1]

        return k_arr, aft, dk

    def gaussian(self, x, fwhm):
        sigma = fwhm / 2.3548
        return np.exp(-(x**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

    def project_orbitals_1d(self, axis=0, gauss_pos=None, gauss_fwhm=2.0):
        self.morbs_1d = []

        if axis != 0:
            dv = np.swapaxes(self.dv, axis, 0)
        else:
            dv = self.dv

        for ispin in range(self.nspin):
            self.morbs_1d.append(np.zeros((self.cell_n[0], len(self.cp2k_grid_orb.morb_grids[ispin]))))
            for i_mo, morb_grid in enumerate(self.cp2k_grid_orb.morb_grids[ispin]):
                if axis != 0:
                    morb_grid = np.swapaxes(morb_grid, axis, 0)
                if gauss_pos is None:
                    morb_1d = np.mean(morb_grid**2, axis=(1, 2))
                else:
                    ny = morb_grid.shape[1]
                    y_arr = np.linspace(-dv[1] * ny / 2.0, dv[1] * ny / 2.0, ny)
                    y_gaussian = self.gaussian(y_arr - gauss_pos, gauss_fwhm)
                    morb_plane = np.mean(morb_grid**2, axis=2)
                    morb_1d = np.dot(morb_plane, y_gaussian)

                self.morbs_1d[ispin][:, i_mo] = morb_1d

    def take_fts(self, crop_padding=True, crop_edges=0.0, remove_row_avg=True, padding=1.0):
        self.morb_fts = []
        for ispin in range(self.nspin):
            tmp_morbs = self.morbs_1d[ispin]
            if crop_padding:
                tmp_morbs, i_crop_1, i_crop_2 = self.crop_padding(tmp_morbs)
            if crop_edges > 0.0:
                tmp_morbs = self.crop_edges(tmp_morbs, dist=crop_edges)
            if remove_row_avg:
                tmp_morbs = self.remove_row_average(tmp_morbs)
            if padding > 0.0:
                tmp_morbs = self.add_padding(tmp_morbs, padding)
            self.k_arr, m_fts, self.dk = self.fourier_transform(tmp_morbs)
            self.morb_fts.append(m_fts)
        borders = i_crop_1 * self.dv[0] + crop_edges, i_crop_2 * self.dv[0] - crop_edges
        return borders

    def make_ftldos(self, emin, emax, de, fwhm):
        self.e_arr = np.arange(emin, emax + de / 2, de)

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
        nbzb_index = int(np.round(nbz * bzboundary / self.dk)) + 1

        return self.ftldos[:nbzb_index, :], [0.0, nbz * bzboundary, self.ftldos_extent[2], self.ftldos_extent[3]]
