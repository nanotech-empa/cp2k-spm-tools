from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from cp2k_spm_tools.cli.overlap_to_sparse_npz import main as overlap_cli
from cp2k_spm_tools.cp2k_overlap_matrix import Cp2kOverlapMatrix

OVERLAP_LOG = Path(__file__).parent / "data" / "cp2k_ch4_overlap_matrix.log"


def assert_overlap_metadata_equal(left, right):
    for name in ("basis_index", "atom_index", "element", "orbital"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))


def assert_parses_like_reference_log(path):
    """The parse of ``path`` must match the pristine reference log exactly."""

    expected = Cp2kOverlapMatrix.from_cp2k_output(OVERLAP_LOG)
    parsed = Cp2kOverlapMatrix.from_cp2k_output(path)

    assert parsed.matrix.shape == expected.matrix.shape
    np.testing.assert_allclose(parsed.matrix.toarray(), expected.matrix.toarray())
    assert_overlap_metadata_equal(parsed, expected)
    return parsed


def write_log(tmp_path, text):
    path = tmp_path / "cp2k-overlap.log"
    path.write_text(text)
    return path


def test_parse_cp2k_overlap_matrix_log_data():
    parsed = Cp2kOverlapMatrix.from_cp2k_output(OVERLAP_LOG)

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


def test_parse_stops_after_first_block(tmp_path):
    """A second matrix must not be summed into the first via duplicate COO coordinates.

    CP2K prints "OVERLAP MATRIX" once per matrix -- the reference log holds one
    title above 29 column-header blocks -- so a repeated title starts a new
    matrix and must terminate the parse.
    """

    path = write_log(tmp_path, OVERLAP_LOG.read_text() * 2)

    assert_parses_like_reference_log(path)


def test_parse_ignores_trailing_table(tmp_path):
    """A later table shaped like overlap data must not be absorbed into the matrix."""

    trailing = """
 MULLIKEN POPULATION ANALYSIS

                                1                    2
     1    1  C     2s        9.99000000000000     8.88000000000000
     2    1  C     3s        7.77000000000000     6.66000000000000
"""
    path = write_log(tmp_path, OVERLAP_LOG.read_text() + trailing)

    assert_parses_like_reference_log(path)


def test_parse_warns_on_unexpected_preamble_lines(tmp_path):
    """Lines skipped inside the block are reported, so a partial parse is never silent."""

    text = OVERLAP_LOG.read_text().replace(
        " OVERLAP MATRIX\n",
        " OVERLAP MATRIX\n Unexpected preamble line\n",
        1,
    )
    path = write_log(tmp_path, text)

    with pytest.warns(UserWarning, match="Skipped 1 unexpected line"):
        assert_parses_like_reference_log(path)


def test_parse_warns_on_missing_ao_rows(tmp_path):
    log = write_log(
        tmp_path,
        " OVERLAP MATRIX\n"
        "              1              2              3\n"
        "      1       1 C  s       1.000000D+00   0.000000D+00   0.000000D+00\n"
        "      3       2 H  s       0.000000D+00   0.000000D+00   1.000000D+00\n",
    )

    with pytest.warns(UserWarning, match="No overlap data row for AO index 2"):
        parsed = Cp2kOverlapMatrix.from_cp2k_output(log)

    assert parsed.matrix.shape == (3, 3)
    assert parsed.atom_index[1] == 0
    assert parsed.element[1] == ""


def test_parse_warns_on_multiple_missing_ao_rows(tmp_path):
    log = write_log(
        tmp_path,
        " OVERLAP MATRIX\n"
        "              1              2              3              4\n"
        "      1       1 C  s       1.000000D+00   0.000000D+00   0.000000D+00   0.000000D+00\n"
        "      4       2 H  s       0.000000D+00   0.000000D+00   0.000000D+00   1.000000D+00\n",
    )

    with pytest.warns(UserWarning, match=r"No overlap data row for AO index 2, 3\b"):
        parsed = Cp2kOverlapMatrix.from_cp2k_output(log)

    assert parsed.matrix.shape == (4, 4)
    assert parsed.atom_index[1] == 0 and parsed.atom_index[2] == 0


def test_parse_warns_on_many_missing_ao_rows_truncates_message(tmp_path):
    """The warning lists at most 10 missing indices, then a total count."""

    n_cols = 15
    header = " ".join(str(i) for i in range(1, n_cols + 1))
    values = " ".join("1.000000D+00" if i == 0 else "0.000000D+00" for i in range(n_cols))
    log = write_log(
        tmp_path,
        f" OVERLAP MATRIX\n{header}\n      1       1 C  s       {values}\n",
    )

    with pytest.warns(UserWarning, match=r"\.\.\. \(14 total\)"):
        parsed = Cp2kOverlapMatrix.from_cp2k_output(log)

    assert parsed.matrix.shape == (n_cols, n_cols)


def test_parse_does_not_warn_on_complete_log():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Cp2kOverlapMatrix.from_cp2k_output(OVERLAP_LOG)

    assert caught == []


def test_parse_threshold_preserves_shape_and_metadata():
    full = Cp2kOverlapMatrix.from_cp2k_output(OVERLAP_LOG)
    parsed = Cp2kOverlapMatrix.from_cp2k_output(OVERLAP_LOG, threshold=0.03)

    assert parsed.matrix.shape == (58, 58)
    assert parsed.matrix.nnz == 1700
    assert full.matrix[14, 0] != 0.0
    assert parsed.matrix[14, 0] == 0.0
    assert_overlap_metadata_equal(parsed, full)


def test_sparse_overlap_npz_roundtrip(tmp_path):
    output_path = tmp_path / "overlap.npz"
    expected = Cp2kOverlapMatrix.from_cp2k_output(OVERLAP_LOG, threshold=0.01)

    Cp2kOverlapMatrix.from_cp2k_output(OVERLAP_LOG, threshold=0.01).to_npz(output_path)
    parsed = Cp2kOverlapMatrix.from_npz(output_path)

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


def test_sparse_overlap_npz_roundtrip_preserves_threshold(tmp_path):
    output_path = tmp_path / "overlap.npz"

    Cp2kOverlapMatrix.from_cp2k_output(OVERLAP_LOG, threshold=0.02).to_npz(output_path)
    parsed = Cp2kOverlapMatrix.from_npz(output_path)

    assert parsed.threshold == pytest.approx(0.02)


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
    parsed = Cp2kOverlapMatrix.from_npz(output_path)

    assert parsed.matrix.shape == (58, 58)
    assert parsed.matrix.nnz == 1700
    np.testing.assert_array_equal(
        np.bincount(parsed.atom_index, minlength=6)[1:],
        np.array([22, 9, 9, 9, 9]),
    )
