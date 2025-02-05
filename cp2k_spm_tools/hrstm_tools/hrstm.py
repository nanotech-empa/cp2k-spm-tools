# @author Hillebrand, Fabian
# @date   2019

import time

import numpy as np
import scipy as sp


class Hrstm:
    """
    Provides a relatively generic HR-STM simulator.

    Needs to be given an object for the wave function and the tip coefficients
    that provide certain information.
    The tip DOS can be energy-independent (i.e. "constant") or energy-dependent
    in which case it can be either broadened using Gaussians or left as Dirac
    functions depending on what the full-width at half maximum is set as.

    This class supports parallelism. However, the grids should be divided along
    x-axis only.
    """

    def __init__(self, tip_coeffs, dim_pos, wfn_grid_matrix, sam_fwhm, tip_fwhm, mpi_rank=0, mpi_size=1, mpi_comm=None):
        self.mpi_rank = mpi_rank
        self.mpi_size = mpi_size
        self.mpi_comm = mpi_comm
        self._tc = tip_coeffs
        self._dim_pos = dim_pos
        self._gm = wfn_grid_matrix
        self._sigma = sam_fwhm / 2.35482
        self._variance = self._sigma
        if self._tc.type != "constant":
            self._tau = tip_fwhm / 2.35482
            # Dirac tip:
            if np.isclose(tip_fwhm, 0.0):
                self._check = self._check_dirac
                self._factor = self._dirac
            # Gaussian broadened tip:
            else:
                self._variance = self._sigma * self._tau / (self._sigma**2 + self._tau**2) ** 0.5
                self._check = self._check_gaussian
                self._factor = self._gaussian
        # Constant tip
        else:
            self._tau = None
            self._check = self._check_constant
            self._factor = self._constant

    ### ------------------------------------------------------------------------

    def _check_constant(self, ene_sam, enes_tip, voltages):
        try:  # Test if enes_tip is a container
            skip = np.array([True for ene in enes_tip])
        except TypeError:
            skip = True
        try:  # Test if voltages is a container
            for voltage in voltages:
                skip &= ~(
                    -4.0 * self._sigma < ene_sam <= voltage + 4.0 * self._sigma
                    or voltage - 4.0 * self._sigma < ene_sam <= 4.0 * self._sigma
                )
        except TypeError:
            skip &= ~(
                -4.0 * self._sigma < ene_sam <= voltages + 4.0 * self._sigma
                or voltages - 4.0 * self._sigma < ene_sam <= 4.0 * self._sigma
            )
        return ~skip

    def _constant(self, ene_sam, ene_tip, voltage):
        """Constant tip density and Gaussian density for sample."""
        return 0.5 * (
            sp.special.erf((voltage - ene_sam) / (2.0**0.5 * self._sigma))
            - sp.special.erf((0.0 - ene_sam) / (2.0**0.5 * self._sigma))
        )

    ### ------------------------------------------------------------------------

    def _check_gaussian(self, ene_sam, enes_tip, voltages):
        vals = (
            (enes_tip * ene_sam > 0.0) | ((ene_sam <= 0.0) & (enes_tip == 0.0)) | ((ene_sam == 0.0) & (enes_tip <= 0.0))
        )
        skip = True
        try:
            for voltage in voltages:
                skip &= np.abs(voltage - ene_sam + enes_tip) >= 4.0 * (self._sigma + self._tau)
        except TypeError:
            skip &= np.abs(voltages - ene_sam + enes_tip) >= 4.0 * (self._sigma + self._tau)
        return ~(skip | vals)

    def _gaussian(self, ene_sam, ene_tip, voltage):
        """Gaussian density for tip and sample."""
        # Product of two Gaussian is a Gaussian but don't forget pre-factor
        mean = (self._sigma**2 * (ene_tip + voltage) + self._tau**2 * ene_sam) / (self._sigma**2 + self._tau**2)
        sigma = self._variance
        correction = (
            1.0
            / (2.0 * np.pi * (self._sigma**2 + self._tau**2)) ** 0.5
            * np.exp(-0.5 * (ene_sam - ene_tip - voltage) ** 2 / (self._sigma**2 + self._tau**2))
        )
        return (
            0.5
            * correction
            * (
                sp.special.erf((voltage - mean) / (2.0**0.5 * sigma))
                - sp.special.erf((0.0 - mean) / (2.0**0.5 * sigma))
            )
        )

    ### ------------------------------------------------------------------------

    def _check_dirac(self, ene_sam, enes_tip, voltages):
        vals = (
            (enes_tip * ene_sam > 0.0) | ((ene_sam <= 0.0) & (enes_tip == 0.0)) | ((ene_sam == 0.0) & (enes_tip <= 0.0))
        )
        skip = True
        try:
            for voltage in voltages:
                skip &= np.abs(voltage - ene_sam + enes_tip) >= 4.0 * self._sigma
        except TypeError:
            skip &= np.abs(voltages - ene_sam + enes_tip) >= 4.0 * self._sigma
        return ~(skip | vals)

    def _dirac(self, ene_sam, ene_tip, voltage):
        """
        Gaussian density of states (integration with a Dirac function).

        Note: This is also the limit of self._gaussian as self._tau -> 0
        """
        # Minus sign since voltage is added to tip energy:
        # Relevant range is then (0,-voltage] or (-voltage,0]
        if 0 < ene_tip <= -voltage or -voltage < ene_tip <= 0:
            return np.exp(-0.5 * ((ene_sam - ene_tip - voltage) / self._sigma) ** 2) / (
                self._sigma * (2 * np.pi) ** 0.5
            )
        return 0.0

    ### ------------------------------------------------------------------------
    ### Store and collect
    ### ------------------------------------------------------------------------

    def gather(self):
        """Gathers the current and returns it on rank 0."""
        if self.mpi_comm is None:
            return self.local_current
        if self.mpi_rank == 0:
            current = np.empty(self._dim_pos + (len(self._voltages),))
        else:
            current = None
        outputs = self.mpi_comm.allgather(len(self.local_current.ravel()))
        self.mpi_comm.Gatherv(self.local_current, [current, outputs], root=0)
        return current

    def write(self, filename):
        """
        Writes the current to a file (*.npy).

        The file is written as a 1-dimensional array. The reconstruction has
        thus be done by hand. It can be reshaped into a 4-dimensional array in
        the form [zIdx,yIdx,xIdx,vIdx].
        """
        pass

    def write_compressed(self, filename, tol=1e-3):
        """
        Writes the current compressed to a file (*.npz).

        The file is written as a 1-dimensional array similar to write().
        Furthermore, in order to load the current use np.load()['arr_0'].

        Pay attention: This method invokes a gather!
        """
        current = self.gather()
        if self.mpi_rank == 0:
            for iheight in range(self._dim_pos[-1]):
                for ivol in range(len(self._voltages)):
                    max_val = np.max(np.abs(current[:, :, iheight, ivol]))
                    current[:, :, iheight, ivol][np.abs(current[:, :, iheight, ivol]) < max_val * tol] = 0.0
            np.savez_compressed(filename, current.ravel())

    ### ------------------------------------------------------------------------
    ### Running HR-STM
    ### ------------------------------------------------------------------------

    def run(self, voltages, info=True):
        """Performs the HR-STM simulation."""
        self._voltages = np.array(voltages)
        self.local_current = np.zeros((len(self._voltages),) + self._tc.grid_dim)
        totTM = 0.0
        totVL = 0.0
        # Over each separate tunnel process (e.g. to O- or C-atom)
        for itunnel in range(self._tc.ntunnels):
            for ispin_sam in range(self._gm.nspin):
                for iene_sam, ene_sam in enumerate(self._gm.ene[ispin_sam]):
                    for ispin_tip, enes_tip in enumerate(self._tc.ene):
                        ienes_tip = np.arange(len(enes_tip))
                        for iene_tip in [
                            iene_tip for iene_tip in ienes_tip[self._check(ene_sam, enes_tip, self._voltages)]
                        ]:
                            # Current tip energy
                            ene_tip = self._tc.ene[ispin_tip][iene_tip]
                            start = time.time()
                            tunnel_matrix_squared = (
                                np.einsum(
                                    "i...,i...->...",
                                    self._tc[itunnel, ispin_tip, iene_tip],
                                    self._gm[itunnel, ispin_sam, iene_sam],
                                )
                            ) ** 2
                            end = time.time()
                            totTM += end - start
                            start = time.time()
                            for ivol, voltage in enumerate(self._voltages):
                                if self._check(ene_sam, ene_tip, voltage):
                                    self.local_current[ivol] += (
                                        self._factor(ene_sam, ene_tip, voltage) * tunnel_matrix_squared
                                    )
                            end = time.time()
                            totVL += end - start
        # Copy to assure C-contiguous array
        self.local_current = self.local_current.transpose((1, 2, 3, 0)).copy()
        if info:
            print("Total time for tunneling matrix was {:} seconds.".format(totTM))
            print("Total time for voltage loop was {:} seconds.".format(totVL))
