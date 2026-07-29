from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cp2k_spm_tools.cli.overlap_to_sparse_npz import main as overlap_cli
from cp2k_spm_tools.cp2k_overlap_matrix import (
    parse_cp2k_overlap_matrix_log_data,
    read_sparse_overlap_npz,
    write_sparse_overlap_npz,
)

OVERLAP_LOG = Path(__file__).parent / "data" / "cp2k_ch4_overlap_matrix.log"


def assert_overlap_metadata_equal(left, right):
    for name in ("basis_index", "atom_index", "element", "orbital"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))


def test_parse_cp2k_overlap_matrix_log_data():
    parsed = parse_cp2k_overlap_matrix_log_data(OVERLAP_LOG)

    assert parsed.matrix.shape == (58, 58)
    assert parsed.matrix.nnz == 2728
    assert (parsed.matrix - parsed.matrix.T).nnz == 0
    assert parsed.matrix[0, 0] == pytest.approx(1.00000033963887)
    assert parsed.matrix[0, 1] == pytest.approx(0.4164459773709)
    assert parsed.matrix[22, 31] == pytest.approx(0.0996847193513)
    assert parsed.matrix[57, 57] == pytest.approx(0.96944472587675)

    np.testing.assert_array_equal(parsed.basis_index, np.arange(1, 59))
    np.testing.assert_array_equal(
        np.bincount(parsed.atom_index, minlength=6)[1:],
        np.array([22, 9, 9, 9, 9]),
    )
    np.testing.assert_array_equal(parsed.element, np.array(["C"] * 22 + ["H"] * 36))
    np.testing.assert_array_equal(
        parsed.orbital[:5],
        np.array(["2s", "3s", "4s", "3py", "3pz"]),
    )
    np.testing.assert_array_equal(
        parsed.orbital[-5:],
        np.array(["3pz", "3px", "4py", "4pz", "4px"]),
    )


def test_parse_threshold_preserves_shape_and_metadata():
    full = parse_cp2k_overlap_matrix_log_data(OVERLAP_LOG)
    parsed = parse_cp2k_overlap_matrix_log_data(OVERLAP_LOG, threshold=0.03)

    assert parsed.matrix.shape == (58, 58)
    assert parsed.matrix.nnz == 1700
    assert full.matrix[14, 0] != 0.0
    assert parsed.matrix[14, 0] == 0.0
    assert_overlap_metadata_equal(parsed, full)


def test_sparse_overlap_npz_roundtrip(tmp_path):
    output_path = tmp_path / "overlap.npz"
    expected = parse_cp2k_overlap_matrix_log_data(OVERLAP_LOG, threshold=0.01)

    write_sparse_overlap_npz(OVERLAP_LOG, output_path, threshold=0.01)
    parsed = read_sparse_overlap_npz(output_path)

    assert output_path.exists()
    assert parsed.matrix.shape == (58, 58)
    assert parsed.matrix.nnz == 2178
    np.testing.assert_allclose(parsed.matrix.toarray(), expected.matrix.toarray())
    assert_overlap_metadata_equal(parsed, expected)
    with np.load(output_path) as data:
        assert set(data.files) == {
            "row",
            "col",
            "data",
            "shape",
            "basis_index",
            "atom_index",
            "element",
            "orbital",
            "threshold",
        }
        assert data["threshold"] == pytest.approx(0.01)


def test_cli_writes_sparse_overlap_npz(tmp_path):
    output_path = tmp_path / "overlap-cli.npz"

    overlap_cli(
        [
            str(OVERLAP_LOG),
            str(output_path),
            "--threshold",
            "0.03",
        ]
    )
    parsed = read_sparse_overlap_npz(output_path)

    assert parsed.matrix.shape == (58, 58)
    assert parsed.matrix.nnz == 1700
    np.testing.assert_array_equal(
        np.bincount(parsed.atom_index, minlength=6)[1:],
        np.array([22, 9, 9, 9, 9]),
    )
