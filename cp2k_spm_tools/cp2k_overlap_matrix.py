from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp

ang_2_bohr = 1.0 / 0.52917721067
hart_2_ev = 27.21138602


class Cp2kOverlapMatrix:
    """Class to deal with the CP2K overlap matrix"""

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
        self.sparse_mat = sparse_mat


@dataclass
class Cp2kOverlapMatrixLog:
    """Sparse CP2K AO overlap matrix together with per-row AO metadata."""

    matrix: sp.csr_matrix
    basis_index: np.ndarray
    atom_index: np.ndarray
    element: np.ndarray
    orbital: np.ndarray


def parse_cp2k_overlap_matrix_log_data(
    path: Union[str, Path],
    nao: Optional[int] = None,
    *,
    threshold: Optional[float] = None,
) -> Cp2kOverlapMatrixLog:
    """Parse CP2K human-readable ``OVERLAP MATRIX`` blocks.

    CP2K prints the matrix in repeated column blocks. Each data row starts with
    the 1-based AO index, atom index, element, and orbital label, followed by
    one matrix value for every active column header. The returned sparse matrix
    keeps only entries whose absolute value is larger than ``threshold``.
    """

    path = Path(path)
    float_re = re.compile(r"^[+-]?(?:[0-9]+(?:[.][0-9]*)?|[.][0-9]+)(?:[EeDd][+-]?[0-9]+)?$")

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    basis: Dict[int, Tuple[int, str, str]] = {}
    current_cols: Optional[List[int]] = None
    inside_overlap_matrix = False
    max_index = 0
    threshold_value = 0.0 if threshold is None else float(threshold)

    def is_int_token(token: str) -> bool:
        return token.isdigit()

    def is_float_token(token: str) -> bool:
        return bool(float_re.match(token))

    with path.open("r", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if "OVERLAP MATRIX" in line:
                inside_overlap_matrix = True
                current_cols = None
                continue

            if not inside_overlap_matrix:
                continue

            parts = line.split()

            if parts and all(is_int_token(token) for token in parts):
                current_cols = [int(token) - 1 for token in parts]
                max_index = max(max_index, max(current_cols) + 1)
                continue

            if current_cols is None or len(parts) < 5:
                continue
            if not (is_int_token(parts[0]) and is_int_token(parts[1])):
                continue

            value_tokens = parts[4:]
            if len(value_tokens) != len(current_cols):
                continue
            if not all(is_float_token(token) for token in value_tokens):
                continue

            irow = int(parts[0]) - 1
            atom = int(parts[1])
            element = parts[2]
            orbital = parts[3]
            basis[irow] = (atom, element, orbital)
            max_index = max(max_index, irow + 1)

            for jcol, token in zip(current_cols, value_tokens):
                value = float(token.replace("D", "E").replace("d", "e"))
                if abs(value) > threshold_value:
                    rows.append(irow)
                    cols.append(jcol)
                    vals.append(value)

    if not basis:
        raise ValueError(f"No overlap-matrix entries found in {path}")

    n_basis = int(nao) if nao is not None else max_index
    matrix = sp.coo_matrix((vals, (rows, cols)), shape=(n_basis, n_basis)).tocsr()
    basis_index = np.arange(1, n_basis + 1, dtype=np.int64)
    atom_index = np.zeros(n_basis, dtype=np.int64)
    elements = np.full(n_basis, "", dtype="U8")
    orbitals = np.full(n_basis, "", dtype="U16")

    for irow, (atom, element, orbital) in basis.items():
        if irow < n_basis:
            atom_index[irow] = atom
            elements[irow] = element
            orbitals[irow] = orbital

    return Cp2kOverlapMatrixLog(
        matrix=matrix,
        basis_index=basis_index,
        atom_index=atom_index,
        element=elements,
        orbital=orbitals,
    )


def parse_cp2k_overlap_matrix_log(path: Union[str, Path], nao: Optional[int] = None) -> sp.csr_matrix:
    """Parse CP2K human-readable ``OVERLAP MATRIX`` blocks as a sparse CSR matrix."""

    return parse_cp2k_overlap_matrix_log_data(path, nao=nao).matrix


def read_sparse_overlap_npz(path_or_file) -> Cp2kOverlapMatrixLog:
    """Read sparse CP2K overlap data written by :func:`write_sparse_overlap_npz`."""

    with np.load(path_or_file) as data:
        arrays = {key: data[key] for key in data.files}

    matrix = sp.coo_matrix(
        (arrays["data"], (arrays["row"], arrays["col"])),
        shape=tuple(arrays["shape"]),
    ).tocsr()
    return Cp2kOverlapMatrixLog(
        matrix=matrix,
        basis_index=arrays["basis_index"],
        atom_index=arrays["atom_index"],
        element=arrays["element"],
        orbital=arrays["orbital"],
    )


def write_sparse_overlap_npz(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    threshold: float = 0.0,
    nao: Optional[int] = None,
) -> None:
    """Convert a CP2K overlap matrix log to a compressed sparse NPZ file.

    The NPZ stores COO arrays (``row``, ``col``, ``data``, ``shape``) and AO
    metadata arrays (``basis_index``, ``atom_index``, ``element``, ``orbital``).
    This compact schema is suitable for downstream sparse unfolding workflows.
    """

    parsed = parse_cp2k_overlap_matrix_log_data(input_path, nao=nao, threshold=threshold)
    matrix = parsed.matrix.tocoo()
    np.savez_compressed(
        output_path,
        row=matrix.row.astype(np.int64),
        col=matrix.col.astype(np.int64),
        data=matrix.data.astype(np.float64),
        shape=np.asarray(matrix.shape, dtype=np.int64),
        basis_index=parsed.basis_index,
        atom_index=parsed.atom_index,
        element=parsed.element,
        orbital=parsed.orbital,
        threshold=np.asarray(threshold, dtype=np.float64),
    )
