# @author Hillebrand, Fabian
# @date   2019

import tarfile
import time

import numpy as np
import scipy as sp

hartreeToEV = 27.21138602


def const_coeffs(s=0.0, py=0.0, pz=0.0, px=0.0):
    """
    Creates coefficients for seperated tunnelling to each orbital.
    The energies are set to zero.
    """
    cc = np.array([s, py, pz, px]) != 0.0
    coeffs = np.empty(
        (sum(cc), 4),
    )
    ene = np.zeros(sum(cc))
    if s != 0.0:
        coeffs[sum(cc[:1]) - 1] = [s**0.5, 0.0, 0.0, 0.0]
    if py != 0.0:
        coeffs[sum(cc[:2]) - 1] = [0.0, py**0.5, 0.0, 0.0]
    if pz != 0.0:
        coeffs[sum(cc[:3]) - 1] = [0.0, 0.0, pz**0.5, 0.0]
    if px != 0.0:
        coeffs[sum(cc[:4]) - 1] = [0.0, 0.0, 0.0, px**0.5]
    return [coeffs], [ene]


def read_PDOS(filename, eMin=0.0, eMax=0.0):
    """
    Reads coefficients from *.pdos file and uses these to construct tip
    coefficients. The eigenvalues are shifted such that the Fermi energy
    is at 0 and scaled such that the units are in eV.

    Note: This function does currently not support spin (for the tip).

    pdos A list containing matrices. Rows correspond to eigenvalues
         while columns to orbitals.
    eigs A list containing arrays for eigenvalues per spin.
    """
    # Open file
    if type(filename) is not tarfile.ExFileObject:
        f = open(filename)
    else:  # Tar file already open
        f = filename

    lines = list(line for line in (ln.strip() for ln in f) if line)
    # TODO spins
    noSpins = 1
    noEigsTotal = len(lines) - 2
    noDer = len(lines[1].split()[5:])

    homoEnergies = [float(lines[0].split()[-2]) * hartreeToEV]
    pdos = [np.empty((noEigsTotal, noDer))]
    eigs = [np.empty((noEigsTotal))]

    # Read all coefficients, cut later
    for lineIdx, line in enumerate(lines[2:]):
        parts = line.split()
        eigs[0][lineIdx] = float(parts[1]) * hartreeToEV
        pdos[0][lineIdx, :] = [float(val) for val in parts[3:]]

    # Cut coefficients to energy range
    startIdx = [None] * noSpins
    for spinIdx in range(noSpins):
        try:
            startIdx[spinIdx] = np.where(eigs[spinIdx] >= eMin + homoEnergies[spinIdx])[0][0]
        except Exception as _e:
            startIdx[spinIdx] = 0
    endIdx = [None] * noSpins
    for spinIdx in range(noSpins):
        try:
            endIdx[spinIdx] = np.where(eigs[spinIdx] > eMax + homoEnergies[spinIdx])[0][0]
        except Exception as _e:
            endIdx[spinIdx] = len(eigs[spinIdx])
    if endIdx <= startIdx:
        raise ValueError("Restricted energy-range too restrictive: endIdx <= startIdx")

    eigs = [eigs[spinIdx][startIdx[spinIdx] : endIdx[spinIdx]] - homoEnergies[spinIdx] for spinIdx in range(noSpins)]
    pdos = [pdos[spinIdx][startIdx[spinIdx] : endIdx[spinIdx], :] for spinIdx in range(noSpins)]

    return pdos, eigs


class TipCoefficients:
    """
    Structure that provides access to tip coefficients for each point in a list
    of grids.

    The coefficients can be rotated for the grid. The rotation is given through
    two different grids corresponding to the points of rotation and the rotated
    points. The axis of rotation is fixed in the x-y-plane.

    This structure provides access to the coefficients via bracket operators.
    The following structure is provided: [itunnel,ispin,iene][iorb,x,y,z].
    Note that this evaluation may be performed lazily.
    """

    def __init__(self, mpi_rank=0, mpi_size=1, mpi_comm=None):
        self.mpi_rank = mpi_rank
        self.mpi_size = mpi_size
        self.mpi_comm = mpi_comm

    ### ------------------------------------------------------------------------
    ### Read function for coefficients
    ### ------------------------------------------------------------------------

    def read_coefficients(self, norbs, pdos_list, emin, emax):
        """
        Reads coefficients from files or via command line if given in the
        shape (s, py, pz, px).
        Coefficients are broadcasted to all MPI processes.
        """
        self._norbs = norbs
        self._singles = None
        self._ene = None
        self._type = None
        self._grid_dim = None
        self._ntunnels = None
        self._coeffs = None
        # Read untransformed tip coefficients and energies
        if self.mpi_rank == 0:
            self._singles = []
            idx = 0  # Index of input argument
            while idx < len(pdos_list):
                # tar.gz file instead
                if pdos_list[idx].endswith("tar.gz"):
                    tar = tarfile.open(pdos_list[idx], "r:gz")
                    for member in tar.getmembers():
                        single, self._ene = read_PDOS(tar.extractfile(member), emin, emax)
                        for ispin in range(len(single)):
                            single[ispin] = single[ispin][:, : (self.norbs + 1) ** 2] ** 0.5
                        self._singles.append(single)
                    assert self.type != "constant", "Tried to mix tip types!"
                    self._type = "gaussian"
                    idx += 1
                # Normal input
                else:
                    try:
                        single, self._ene = const_coeffs(
                            s=float(pdos_list[idx]),
                            py=float(pdos_list[idx + 1]),
                            pz=float(pdos_list[idx + 2]),
                            px=float(pdos_list[idx + 3]),
                        )
                        assert self.type != "gaussian", "Tried to mix tip types!"
                        self._type = "constant"
                        idx += 4
                    except ValueError:
                        single, self._ene = read_PDOS(pdos_list[idx], emin, emax)
                        # Take square root to obtain proper coefficients
                        for ispin in range(len(single)):
                            single[ispin] = single[ispin][:, : (self.norbs + 1) ** 2] ** 0.5
                        idx += 1
                        assert self.type != "constant", "Tried to mix tip types!"
                        self._type = "gaussian"
                    self._singles.append(single)
        # Broadcast untransformed coefficients and energies
        if self.mpi_comm is not None:
            self._singles = self.mpi_comm.bcast(self._singles, root=0)
            self._ene = self.mpi_comm.bcast(self._ene, root=0)
            self._type = self.mpi_comm.bcast(self._type, root=0)

    ### ------------------------------------------------------------------------
    ### Initialize coefficients
    ### ------------------------------------------------------------------------

    def initialize(self, pos, rotate=False):
        """
        Computes rotational matrix if necessary.
        This method does not communicate. Positions should be split up
        before hand to avoid unnecessary calculations.
        """
        self._grid_dim = np.shape(pos[0])[1:]
        self._ntunnels = len(pos) - 1
        # s-orbtial on tip only
        if self.norbs == 0:
            return

        start = time.time()
        # p-orbital on tip
        self._rot_matrix = [None] * self.ntunnels
        if not rotate:
            for itunnel in range(self.ntunnels):
                self._rot_matrix[itunnel] = 1
        else:
            npoints = np.prod(self.grid_dim)
            shifted_pos = []
            rm_data = []
            for i in range(self.ntunnels):
                shifted_pos.append(
                    np.array([pos[i + 1][0] - pos[i][0], pos[i + 1][1] - pos[i][1], pos[i + 1][2] - pos[i][2]]).reshape(
                        3, npoints
                    )
                )
                rm_data.append(np.empty(9 * npoints))
            # Sparse matrix storage as COO
            ihelper = np.arange(3 * npoints, dtype=int)
            # rows = [0,1,2,...,0,1,2,...,0,1,2,...]
            rm_rows = np.tile(ihelper, 3)
            # cols = [0,0,0,1,1,1,2,2,2,...]
            rm_cols = np.repeat(ihelper, 3)
            del ihelper
            # Matrix data
            for itunnel in range(self.ntunnels):
                v = np.array([0.0, 0.0, -1.0])
                # Rotated vector
                w = shifted_pos[itunnel]
                w /= np.linalg.norm(w, axis=0)
                # Rotation axis (no rotation around x-y)
                n = np.cross(v, w, axisb=0).transpose()
                # Trigonometric values
                cosa = np.dot(v, w)
                sina = (1 - cosa**2) ** 0.5

                #  R = [[n[0]**2*(1-cosa)+cosa, n[0]*n[1]*(1-cosa)-n[2]*sina, n[0]*n[2]*(1-cosa)+n[1]*sina],
                #       [n[1]*n[0]*(1-cosa)+n[2]*sina, n[1]**2*(1-cosa)+cosa, n[1]*n[2]*(1-cosa)-n[0]*sina],
                #       [n[2]*n[1]*(1-cosa)-n[1]*sina, n[2]*n[1]*(1-cosa)+n[0]*sina, n[2]**2*(1-cosa)+cosa]]
                # Permutation matrix for betas: (y,z,x) -> (x,y,z)
                #  P = [[0,0,1],
                #       [1,0,0],
                #       [0,1,0]]
                # Note: The rotational matrix R is with respect to the sample which is R^T for the tip
                #       --> R to R^T for going from tip to sample, R^T to R for gradient
                rm_data[itunnel][:npoints] = n[1] ** 2 * (1 - cosa) + cosa
                rm_data[itunnel][npoints : 2 * npoints] = n[2] * n[1] * (1 - cosa) - n[0] * sina
                rm_data[itunnel][2 * npoints : 3 * npoints] = n[0] * n[1] * (1 - cosa) + n[2] * sina
                rm_data[itunnel][3 * npoints : 4 * npoints] = n[1] * n[2] * (1 - cosa) + n[0] * sina
                rm_data[itunnel][4 * npoints : 5 * npoints] = n[2] ** 2 * (1 - cosa) + cosa
                rm_data[itunnel][5 * npoints : 6 * npoints] = n[0] * n[2] * (1 - cosa) - n[1] * sina
                rm_data[itunnel][6 * npoints : 7 * npoints] = n[1] * n[0] * (1 - cosa) - n[2] * sina
                rm_data[itunnel][7 * npoints : 8 * npoints] = n[2] * n[0] * (1 - cosa) + n[1] * sina
                rm_data[itunnel][8 * npoints : 9 * npoints] = n[0] ** 2 * (1 - cosa) + cosa

            # Build large matrices
            for itunnel in range(self.ntunnels):
                self._rot_matrix[itunnel] = sp.sparse.csr_matrix(
                    sp.sparse.coo_matrix((rm_data[itunnel], (rm_rows, rm_cols)), shape=(3 * npoints, 3 * npoints))
                )

        end = time.time()
        print("Rotational matrices took {} seconds".format(end - start))

    ### ------------------------------------------------------------------------
    ### Access operators
    ### ------------------------------------------------------------------------

    @property
    def ene(self):
        """List of energies per spin."""
        return self._ene

    @property
    def singles(self):
        """Untransformed coefficients."""
        return self._singles

    @property
    def type(self):
        """Either gaussian or constant."""
        return self._type

    @property
    def grid_dim(self):
        """Dimension of grid for tip positions."""
        return self._grid_dim

    @property
    def norbs(self):
        """Number of tip orbitals."""
        return self._norbs

    @property
    def nspin(self):
        """Number of spins."""
        return len(self.ene)

    @property
    def ntunnels(self):
        """Number of tunnellings."""
        return self._ntunnels

    def __getitem__(self, ituple):
        """Takes an index tuple (itunnel,ispin,iene)."""
        # Unpack indices
        itunnel, ispin, iene = ituple
        if self._coeffs is not None and self._cene == iene and self._cspin == ispin and self._ctunnel == itunnel:
            return self._coeffs
        # Storage container
        self._coeffs = np.empty(((self.norbs + 1) ** 2,) + self.grid_dim)

        # s-orbitals: Are never rotated
        self._coeffs[0].fill(self._singles[itunnel][ispin][iene, 0])
        if self.norbs > 0:
            # p-orbitals: Are rotated like vectors
            self._coeffs[1].fill(self.singles[itunnel][ispin][iene, 1])
            self._coeffs[2].fill(self.singles[itunnel][ispin][iene, 2])
            self._coeffs[3].fill(self.singles[itunnel][ispin][iene, 3])
            # Flat view of coeffs[1:4]
            flat_coeffs = self._coeffs[1:4].ravel()
            # Provoke write into flat view instead of overwriting variable with [:]
            flat_coeffs[:] = self._rot_matrix[itunnel] * self._coeffs[1:4].flatten()
        # Save some information
        self._ctunnel, self._cspin, self._cene = ituple
        return self._coeffs
