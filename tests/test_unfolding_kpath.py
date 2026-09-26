from itertools import product

import numpy as np
import pytest

from cp2k_spm_tools.cp2k_unfolding.kpath import (
    folded_kpoints_from_supercell_matrix,
    guess_2d_lattice_type,
    project_kpoints_to_kpath,
    standard_kpath,
)

CELLS = [
    ("square", [[1, 0, 0], [0, 1, 0]]),
    ("rectangular", [[1, 0, 0], [0, 1.7, 0]]),
    ("hexagonal", [[1, 0, 0], [0.5, np.sqrt(3) / 2, 0]]),
    ("hexagonal", [[1, 0, 0], [-0.5, np.sqrt(3) / 2, 0]]),
    ("centered_rectangular", [[1, 0, 0], [0.25, np.sqrt(15) / 4, 0]]),
    ("centered_rectangular", [[1, 0, 0], [-0.25, np.sqrt(15) / 4, 0]]),
    ("oblique", [[1, 0, 0], [0.4, 1.3, 0]]),
    ("oblique", [[1, 0, 0], [-0.4, 1.3, 0]]),
]


@pytest.mark.parametrize("family,rows", CELLS)
@pytest.mark.parametrize("basis", [np.eye(2), [[0, 1], [1, 0]], [[1, 4], [0, 1]]])
def test_special_points_against_brillouin_zone(family, rows, basis):
    primitive = np.asarray(basis) @ np.asarray(rows, dtype=float)
    assert guess_2d_lattice_type(primitive) == family
    points, path = standard_kpath(2, primitive_vectors=primitive)
    assert path[0] == path[-1] == "G"
    assert set(path) <= points.keys()

    # Independent Bragg-plane construction: |k| <= |k-G| for every
    # reciprocal-lattice vector G. This also checks transformed input bases.
    reciprocal = 2 * np.pi * np.linalg.inv(primitive[:, :2]).T
    lattice_indices = np.array([n for n in product(range(-12, 13), repeat=2) if n != (0, 0)])
    reciprocal_sites = lattice_indices @ reciprocal
    for label, frac in points.items():
        cart = frac @ reciprocal
        excess = reciprocal_sites @ cart - np.sum(reciprocal_sites**2, axis=1) / 2
        assert np.max(excess) < 1e-8, (family, label, frac)
        if label != "G":
            assert np.count_nonzero(np.abs(excess) < 1e-8) >= 1
        if label in {"K", "H", "H1", "A1", "S", "M"}:
            # Hexagonal M is an edge midpoint; square M is a vertex.
            if label == "M" and family == "hexagonal":
                continue
            assert np.count_nonzero(np.abs(excess) < 1e-8) >= 2


def test_hexagonal_basis_conventions():
    for sign, expected_k in [(1, [2 / 3, 1 / 3]), (-1, [1 / 3, 1 / 3])]:
        primitive = np.array([[1, 0, 0], [sign / 2, np.sqrt(3) / 2, 0]])
        points, path = standard_kpath(2, primitive_vectors=primitive)
        np.testing.assert_allclose(points["K"], expected_k)
        np.testing.assert_allclose(points["M"], [0.5, 0])
        assert path == ["G", "K", "M", "G"]


@pytest.mark.parametrize("family,rows", CELLS)
def test_projection_uses_available_folded_points_and_keeps_closing_gamma(family, rows):
    primitive = np.asarray(rows, dtype=float)
    points, path = standard_kpath(2, primitive_vectors=primitive)
    folded = folded_kpoints_from_supercell_matrix(6 * np.eye(2, dtype=int))
    indices, x, segments, t, equivalent, ticks = project_kpoints_to_kpath(folded, points, path, primitive)
    assert len(folded) == 36
    np.testing.assert_allclose((equivalent - folded[indices]), np.rint(equivalent - folded[indices]), atol=1e-12)
    assert np.all(np.diff(x) >= -1e-12)
    gamma = np.where(np.all(np.isclose(folded, 0), axis=1))[0][0]
    np.testing.assert_allclose(x[indices == gamma], [0, ticks[-1]], atol=1e-12)
    for q, segment, parameter in zip(equivalent, segments, t):
        expected = points[path[segment]] + parameter * (points[path[segment + 1]] - points[path[segment]])
        np.testing.assert_allclose(q, expected, atol=1e-12)


def test_projection_for_non_reduced_basis():
    primitive = np.array([[1, 4, 0], [0, 1, 0]], dtype=float)
    points, path = standard_kpath(2, primitive_vectors=primitive)
    folded = folded_kpoints_from_supercell_matrix(6 * np.eye(2, dtype=int))
    _, _, _, _, equivalent, _ = project_kpoints_to_kpath(folded, points, path, primitive)
    # Every named endpoint is sampled by this commensurate supercell,
    # including ones requiring translations outside the old +/-1 search.
    for point in points.values():
        assert np.any(np.all(np.isclose(equivalent, point), axis=1))
    assert np.max(np.abs(equivalent)) > 1


def test_one_dimensional_closed_path():
    primitive = np.array([[1, 0, 0]])
    points, path = standard_kpath(1, primitive_vectors=primitive)
    assert path == ["G", "X"]
    folded = folded_kpoints_from_supercell_matrix(np.array([[6]]))
    indices, x, _, _, _, ticks = project_kpoints_to_kpath(folded, points, path + ["G"], primitive)
    np.testing.assert_allclose(x[indices == 0], [0, ticks[-1]])
    assert len(indices) == 7


def test_reject_mismatched_family():
    with pytest.raises(ValueError, match="does not match"):
        standard_kpath(2, "hexagonal", np.array([[1, 0, 0], [0, 1, 0]]))


def test_reject_out_of_plane_vectors():
    with pytest.raises(ValueError, match="xy plane"):
        standard_kpath(2, primitive_vectors=np.array([[1, 0, 1], [0, 1, 0]]))
