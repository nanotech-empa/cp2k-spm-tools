"""Physical cell conventions at the CP2K input/unfolding boundary."""

import numpy as np
import pytest

from cp2k_spm_tools.cp2k_unfolding.io import parse_cp2k_cell_vectors


def write_cell(tmp_path, text):
    path = tmp_path / "cp2k.inp"
    path.write_text(f"&FORCE_EVAL\n &SUBSYS\n  &CELL\n{text}\n  &END CELL\n &END SUBSYS\n&END FORCE_EVAL\n")
    return path


@pytest.mark.parametrize(
    "unit,scale",
    [
        ("", 1),
        ("angstrom", 1),
        ("ang", 1),
        ("bohr", 0.529177210903),
        ("au", 0.529177210903),
        ("a.u.", 0.529177210903),
        ("nm", 10),
        ("pm", 0.01),
    ],
)
@pytest.mark.parametrize("form", ["ABC", "vectors"])
def test_length_units(tmp_path, unit, scale, form):
    token = f"[{unit}]" if unit else ""
    text = f"ABC {token} 2 4 6" if form == "ABC" else f"A {token} 2 0 0\nB {token} 0 4 0\nC {token} 0 0 6"
    cell = parse_cp2k_cell_vectors(write_cell(tmp_path, text))
    np.testing.assert_allclose(cell, np.diag([2, 4, 6]) * scale, rtol=1e-9, atol=1e-12)


def test_explicit_vectors_preserve_orientation_and_mixed_units(tmp_path):
    cell = parse_cp2k_cell_vectors(write_cell(tmp_path, "A [nm] 0 0.2 0\nB [angstrom] -3 1 0\nC [pm] 0 0 600"))
    np.testing.assert_array_equal(cell, [[0, 2, 0], [-3, 1, 0], [0, 0, 6]])


@pytest.mark.parametrize("gamma", [60, 120])
@pytest.mark.parametrize("keyword,unit", [("ALPHA_BETA_GAMMA", "deg"), ("ANGLES", "rad")])
def test_hexagonal_lengths_and_angles(tmp_path, gamma, keyword, unit):
    angles = np.array([90.0, 90.0, gamma])
    values = angles if unit == "deg" else np.deg2rad(angles)
    text = f"ABC 2 2 6\n{keyword} [{unit}] " + " ".join(map(str, values))
    cell = parse_cp2k_cell_vectors(write_cell(tmp_path, text))
    expected = [[2, 0, 0], [1 if gamma == 60 else -1, np.sqrt(3), 0], [0, 0, 6]]
    np.testing.assert_allclose(cell, expected, atol=1e-12)
    np.testing.assert_allclose(parse_cp2k_cell_vectors(write_cell(tmp_path, text), dim=2), expected[:2], atol=1e-12)


def test_triclinic_cell_has_declared_metric_and_cp2k_orientation(tmp_path):
    lengths = np.array([4.0, 5.0, 6.0])
    cell = parse_cp2k_cell_vectors(write_cell(tmp_path, "ABC 4 5 6\nALPHA_BETA_GAMMA 70 80 75"))
    metric = cell @ cell.T
    np.testing.assert_allclose(np.diag(metric), lengths**2)
    expected_dots = [5 * 6 * np.cos(np.deg2rad(70)), 4 * 6 * np.cos(np.deg2rad(80)), 4 * 5 * np.cos(np.deg2rad(75))]
    np.testing.assert_allclose([metric[1, 2], metric[0, 2], metric[0, 1]], expected_dots)
    np.testing.assert_allclose(cell[0, 1:], 0, atol=1e-12)
    assert abs(cell[1, 2]) < 1e-12 and np.linalg.det(cell) > 0


def test_abc_takes_precedence_and_nested_reference_cell_is_ignored(tmp_path):
    text = "A 7 0 0\nB 0 8 0\nC 0 0 9\nABC 2 4 6\n&CELL_REF\nABC 100 100 100\n&END CELL_REF"
    np.testing.assert_array_equal(parse_cp2k_cell_vectors(write_cell(tmp_path, text)), np.diag([2, 4, 6]))


@pytest.mark.parametrize(
    "text,match",
    [
        ("ABC [feet] 2 4 6", "Unsupported CP2K ABC unit"),
        ("A [feet] 2 0 0\nB 0 4 0\nC 0 0 6", "Unsupported CP2K A unit"),
        ("ABC 2 4 6\nANGLES [turn] .25 .25 .25", "Unsupported CP2K ALPHA_BETA_GAMMA unit"),
        ("ABC 2 4", "exactly three"),
        ("ABC 2 4 nan", "finite"),
        ("ABC 2 4 -6", "positive"),
        ("ABC 2 4 6\nANGLES 90 90 180", "between 0 and 180"),
        ("ABC 2 4 6\nANGLES 10 10 170", "nondegenerate"),
        ("A 2 0 0\nB 4 0 0\nC 0 0 6", "nondegenerate"),
        ("ABC 2 4 6\nCELL_FILE_NAME cell.cif", "resolved explicit"),
        ("ABC 2 4 6\nSYMMETRY HEXAGONAL", "resolved explicit"),
        ("ABC 2 4 6\nMULTIPLE_UNIT_CELL 2 1 1", "resolved explicit"),
    ],
)
def test_invalid_or_unresolved_cells_are_rejected(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        parse_cp2k_cell_vectors(write_cell(tmp_path, text))


@pytest.mark.parametrize(
    "cell_text",
    [
        "A 2 0 0\nB -1 1.7320508075688772 0\nC 0 0 6",
        "ABC 2 2 6\nANGLES 90 90 120",
        "ABC [nm] .2 .2 .6\nANGLES 90 90 120",
        "ABC [bohr] 3.77945224926 3.77945224926 11.33835674778\nANGLES 90 90 120",
    ],
)
def test_equivalent_cell_definitions_produce_known_unfolded_weights(tmp_path, cell_text):
    import scipy.sparse as sp

    from cp2k_spm_tools.cp2k_unfolding.geometry import build_modulo_lattice_ao_mapping
    from cp2k_spm_tools.cp2k_unfolding.kpath import folded_kpoints_from_supercell_matrix, kfrac_to_cart
    from cp2k_spm_tools.cp2k_unfolding.unfolding import unfold_band_weights_full

    supercell = parse_cp2k_cell_vectors(write_cell(tmp_path, cell_text), dim=2)
    primitive = np.array([[1.0, 0, 0], [-1, np.sqrt(3), 0]])
    mapping = build_modulo_lattice_ao_mapping(
        symbols=["H", "H"],
        coords_cart=np.array([[0.0, 0, 0], [1.0, 0, 0]]),
        primitive_vectors=primitive,
        supercell_vectors=supercell,
        aos_per_symbol={"H": 1},
        verbose=False,
    )
    k_cart = kfrac_to_cart(folded_kpoints_from_supercell_matrix(np.diag([2, 1])), primitive)
    coeffs = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
    weights = unfold_band_weights_full(coeffs, k_cart, sp.eye(2, format="csr"), mapping, verbose=False)
    np.testing.assert_allclose(weights, np.eye(2), atol=1e-12)
