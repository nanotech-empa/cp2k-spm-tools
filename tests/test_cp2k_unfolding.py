from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from cp2k_spm_tools.cli.unfold_wfn_sparse import parse_atom_indices, parse_path_labels, parse_vectors
from cp2k_spm_tools.cp2k_unfolding.geometry import (
    build_modulo_lattice_ao_mapping,
    snap_primitive_vectors_to_supercell,
)
from cp2k_spm_tools.cp2k_unfolding.kpath import folded_kpoints_from_supercell_matrix, kfrac_to_cart
from cp2k_spm_tools.cp2k_unfolding.unfolding import unfold_band_weights_full


def test_cli_input_parsers():
    vectors = parse_vectors("1 0 0; 0, 2, 0")
    assert np.allclose(vectors, [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

    assert parse_atom_indices("1 3-5 7..6") == [0, 2, 3, 4, 6, 5]
    assert parse_path_labels("G M K G") == ["G", "M", "K", "G"]
    assert parse_path_labels("GMKG") == ["G", "M", "K", "G"]


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


def test_sparse_unfolding_two_replica_chain():
    primitive_vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    supercell_vectors = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mapping = build_modulo_lattice_ao_mapping(
        symbols=["H", "H"],
        coords_cart=coords,
        primitive_vectors=primitive_vectors,
        supercell_vectors=supercell_vectors,
        aos_per_symbol={"H": 1},
        verbose=False,
    )

    k_frac = folded_kpoints_from_supercell_matrix(np.array([[2, 0], [0, 1]]))
    k_cart = kfrac_to_cart(k_frac, primitive_vectors)
    coeffs = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=float) / np.sqrt(2.0)

    weights = unfold_band_weights_full(coeffs, k_cart, sp.eye(2, format="csr"), mapping, verbose=False)

    assert np.allclose(mapping.ao_to_replica, [0, 1])
    assert np.allclose(mapping.ao_to_local, [0, 0])
    assert np.allclose(weights, [[1.0, 0.0], [0.0, 1.0]], atol=1.0e-12)
