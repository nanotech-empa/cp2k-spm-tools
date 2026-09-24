from __future__ import annotations

import numpy as np

from cp2k_spm_tools.cp2k_unfolding.geometry import snap_primitive_vectors_to_supercell
from cp2k_spm_tools.cp2k_unfolding.kpath import folded_kpoints_from_supercell_matrix


def test_folded_kpoints_from_supercell_matrix():
    k_frac = folded_kpoints_from_supercell_matrix(np.diag([2, 2]))
    assert k_frac.shape == (4, 2)
    assert np.allclose(
        np.array(sorted(map(tuple, k_frac))),
        [[0.0, 0.0], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]],
    )


def test_snap_primitive_vectors_to_supercell():
    supercell = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    approx = np.array([[1.01, 0.0, 0.0], [0.0, 1.49, 0.0]])

    primitive, matrix, matrix_float, correction_norm = snap_primitive_vectors_to_supercell(approx, supercell)

    assert np.allclose(primitive, [[1.0, 0.0, 0.0], [0.0, 1.5, 0.0]])
    assert np.array_equal(matrix, [[2, 0], [0, 2]])
    assert matrix_float.shape == (2, 2)
    assert correction_norm > 0.0
