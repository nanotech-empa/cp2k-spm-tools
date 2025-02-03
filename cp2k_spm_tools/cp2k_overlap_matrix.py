import numpy as np
import scipy
import scipy.io

ang_2_bohr = 1.0 / 0.52917721067
hart_2_ev = 27.21138602


class Cp2kOverlapMatrix:
    """
    Class to deal with the CP2K overlap matrix
    """

    def __init__(self):
        self.sparse_mat = None

    def read_ascii_csr(self, file_name, n_basis_f):
        # might make more sense to store in dense format...

        csr_txt = np.loadtxt(file_name)

        sparse_mat = scipy.sparse.csr_matrix(
            (csr_txt[:, 2], (csr_txt[:, 0] - 1, csr_txt[:, 1] - 1)), shape=(n_basis_f, n_basis_f)
        )

        # add also the lower triangular part
        sparse_mat += sparse_mat.T

        # diagonal got added by both triangular sides
        sparse_mat.setdiag(sparse_mat.diagonal() / 2)
