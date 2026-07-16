from __future__ import annotations

import numpy as np
import pytest

from cp2k_spm_tools.cli.overlap_to_sparse_npz import main as overlap_cli
from cp2k_spm_tools.cp2k_overlap_matrix import (
    parse_cp2k_overlap_matrix_log_data,
    read_sparse_overlap_npz,
    write_sparse_overlap_npz,
)

OVERLAP_LOG = """
 Some CP2K output before the matrix
 OVERLAP MATRIX
              1              2
      1       1 C  s       1.000000D+00   1.250000D-01
      2       1 C  px      1.250000D-01   1.000000D+00
              3
      1       1 C  s       0.000000D+00
      2       1 C  px     -2.500000D-02
      3       2 H  s       5.000000D-03
"""


def write_overlap_log(tmp_path):
    path = tmp_path / "cp2k-overlap.log"
    path.write_text(OVERLAP_LOG)
    return path


def test_parse_cp2k_overlap_matrix_log_data(tmp_path):
    parsed = parse_cp2k_overlap_matrix_log_data(write_overlap_log(tmp_path))

    assert parsed.matrix.shape == (3, 3)
    np.testing.assert_allclose(
        parsed.matrix.toarray(),
        np.array(
            [
                [1.0, 0.125, 0.0],
                [0.125, 1.0, -0.025],
                [0.0, 0.0, 0.005],
            ]
        ),
    )
    np.testing.assert_array_equal(parsed.basis_index, np.array([1, 2, 3]))
    np.testing.assert_array_equal(parsed.atom_index, np.array([1, 1, 2]))
    np.testing.assert_array_equal(parsed.element, np.array(["C", "C", "H"]))
    np.testing.assert_array_equal(parsed.orbital, np.array(["s", "px", "s"]))


def test_parse_applies_threshold_and_nao(tmp_path):
    parsed = parse_cp2k_overlap_matrix_log_data(
        write_overlap_log(tmp_path), nao=4, threshold=0.03
    )

    assert parsed.matrix.shape == (4, 4)
    np.testing.assert_allclose(
        parsed.matrix.toarray(),
        np.array(
            [
                [1.0, 0.125, 0.0, 0.0],
                [0.125, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    np.testing.assert_array_equal(parsed.basis_index, np.array([1, 2, 3, 4]))
    np.testing.assert_array_equal(parsed.atom_index, np.array([1, 1, 2, 0]))


def test_sparse_overlap_npz_roundtrip(tmp_path):
    input_path = write_overlap_log(tmp_path)
    output_path = tmp_path / "overlap.npz"

    write_sparse_overlap_npz(input_path, output_path, threshold=0.01)
    parsed = read_sparse_overlap_npz(output_path)

    assert output_path.exists()
    np.testing.assert_allclose(
        parsed.matrix.toarray(),
        np.array(
            [
                [1.0, 0.125, 0.0],
                [0.125, 1.0, -0.025],
                [0.0, 0.0, 0.0],
            ]
        ),
    )
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
    input_path = write_overlap_log(tmp_path)
    output_path = tmp_path / "overlap-cli.npz"

    overlap_cli([str(input_path), str(output_path), "--threshold", "0.03", "--nao", "4"])
    parsed = read_sparse_overlap_npz(output_path)

    assert parsed.matrix.shape == (4, 4)
    assert parsed.matrix.nnz == 4
