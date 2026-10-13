from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

ang_2_bohr = 1.0 / 0.52917721067
hart_2_ev = 27.21138602


@dataclass
class Cp2kOverlapMatrix:
    """Sparse CP2K AO overlap matrix together with per-row AO metadata."""

    matrix: sp.csr_matrix
    basis_index: np.ndarray
    atom_index: np.ndarray
    element: np.ndarray
    orbital: np.ndarray
    threshold: float = 0.0

    @classmethod
    def from_cp2k_output(cls, path: Union[str, Path], *, threshold: float = 0.0) -> "Cp2kOverlapMatrix":
        """Parse CP2K human-readable ``OVERLAP MATRIX`` blocks.

        CP2K prints the matrix under a single ``OVERLAP MATRIX`` title, repeating
        only the column headers beneath it. Each data row starts with the 1-based
        AO index, atom index, element, and orbital label, followed by one matrix
        value for every active column header. The returned sparse matrix keeps only
        entries whose absolute value is larger than ``threshold``. Its shape and AO
        metadata are inferred from the parsed indices and do not depend on that
        threshold.

        Only the **first** complete matrix is parsed. Parsing stops at the first
        line that is neither a column header nor a data row, and at any second
        ``OVERLAP MATRIX`` title, so unrelated tables later in a full CP2K output
        are never folded into the result.

        Args:
            path: CP2K output or log file containing an ``OVERLAP MATRIX`` block.
            threshold: Keep only matrix entries with an absolute value larger than
                this value.

        Returns:
            The sparse CSR overlap matrix and its per-orbital metadata.

        Raises:
            ValueError: If the file contains no overlap-matrix entries.

        Warns:
            UserWarning: If unexpected lines were skipped between the title and the
                first data row, which may indicate an unrecognised output format, or
                if the parsed AO indices have gaps, leaving empty metadata rows.
        """

        path = Path(path)
        float_re = re.compile(r"^[+-]?(?:[0-9]+(?:[.][0-9]*)?|[.][0-9]+)(?:[EeDd][+-]?[0-9]+)?$")

        # COO entries accumulated across CP2K's repeated matrix-column blocks.
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []

        # Per-AO metadata, keyed by the zero-based matrix row index.
        basis: Dict[int, Tuple[int, str, str]] = {}

        # Parsing state for the active column block and the inferred matrix size.
        current_cols: Optional[List[int]] = None
        inside_overlap_matrix = False
        max_index = 0

        # Whether any data row has been parsed yet, and the unexpected lines seen
        # between the title and the first data row.
        data_started = False
        skipped: List[str] = []

        def is_float_token(token: str) -> bool:
            return bool(float_re.match(token))

        with path.open("r", errors="replace") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue

                if "OVERLAP MATRIX" in line:
                    # CP2K prints the title once per matrix and repeats only the
                    # column headers beneath it, so a second title starts a new
                    # matrix and therefore ends this one.
                    if data_started:
                        break
                    inside_overlap_matrix = True
                    current_cols = None
                    continue

                if not inside_overlap_matrix:
                    continue

                parts = line.split()

                if all(token.isdigit() for token in parts):
                    current_cols = [int(token) - 1 for token in parts]
                    max_index = max(max_index, max(current_cols) + 1)
                    continue

                is_data = (
                    current_cols is not None
                    and len(parts) >= 5
                    and parts[0].isdigit()
                    and parts[1].isdigit()
                    and len(parts[4:]) == len(current_cols)
                    and all(is_float_token(token) for token in parts[4:])
                )

                if not is_data:
                    # Once data has been seen, the first line that is neither a
                    # column header nor a data row ends the matrix. Anything after
                    # it belongs to an unrelated part of the CP2K output and must
                    # not be summed into the overlap entries.
                    if data_started:
                        break
                    skipped.append(line)
                    continue

                data_started = True
                value_tokens = parts[4:]
                irow = int(parts[0]) - 1
                atom = int(parts[1])
                element = parts[2]
                orbital = parts[3]
                basis[irow] = (atom, element, orbital)
                max_index = max(max_index, irow + 1)

                for jcol, token in zip(current_cols, value_tokens):
                    value = float(token.replace("D", "E").replace("d", "e"))
                    if abs(value) > threshold:
                        rows.append(irow)
                        cols.append(jcol)
                        vals.append(value)

        if skipped:
            warnings.warn(
                f"Skipped {len(skipped)} unexpected line(s) before overlap data in {path}",
                stacklevel=2,
            )

        if not basis:
            raise ValueError(f"No overlap-matrix entries found in {path}")

        # Rows are filled in only where the log actually had a data row, so a gap
        # leaves atom_index 0 and empty element/orbital strings for that AO.
        missing = sorted(set(range(max_index)) - basis.keys())
        if missing:
            shown = ", ".join(str(index + 1) for index in missing[:10])
            if len(missing) > 10:
                shown += f", ... ({len(missing)} total)"
            warnings.warn(
                f"No overlap data row for AO index {shown} in {path}; "
                f"their atom, element and orbital metadata will be empty",
                stacklevel=2,
            )

        matrix = sp.coo_matrix((vals, (rows, cols)), shape=(max_index, max_index)).tocsr()
        basis_index = np.arange(1, max_index + 1, dtype=np.int64)
        atom_index = np.zeros(max_index, dtype=np.int64)
        elements = np.full(max_index, "", dtype="U8")
        orbitals = np.full(max_index, "", dtype="U16")

        for irow, (atom, element, orbital) in basis.items():
            atom_index[irow] = atom
            elements[irow] = element
            orbitals[irow] = orbital

        return cls(
            matrix=matrix,
            basis_index=basis_index,
            atom_index=atom_index,
            element=elements,
            orbital=orbitals,
            threshold=threshold,
        )

    @classmethod
    def from_npz(cls, path_or_file) -> "Cp2kOverlapMatrix":
        """Read the sparse archive written by :meth:`to_npz`."""

        with np.load(path_or_file) as data:
            arrays = {key: data[key] for key in data.files}

        matrix = sp.coo_matrix(
            (arrays["data"], (arrays["row"], arrays["col"])),
            shape=tuple(arrays["shape"]),
        ).tocsr()
        return cls(
            matrix=matrix,
            basis_index=arrays["basis_index"],
            atom_index=arrays["atom_index"],
            element=arrays["element"],
            orbital=arrays["orbital"],
            threshold=float(arrays["threshold"]),
        )

    def to_npz(self, path) -> None:
        """Write this matrix and its AO metadata as a compressed sparse archive.

        The NPZ stores COO arrays (``row``, ``col``, ``data``, ``shape``) and AO
        metadata arrays (``basis_index``, ``atom_index``, ``element``, ``orbital``,
        ``threshold``). This compact schema is suitable for downstream sparse
        unfolding workflows.
        """

        matrix = self.matrix.tocoo()
        np.savez_compressed(
            path,
            row=matrix.row.astype(np.int64),
            col=matrix.col.astype(np.int64),
            data=matrix.data.astype(np.float64),
            shape=np.asarray(matrix.shape, dtype=np.int64),
            basis_index=self.basis_index,
            atom_index=self.atom_index,
            element=self.element,
            orbital=self.orbital,
            threshold=np.asarray(self.threshold, dtype=np.float64),
        )
