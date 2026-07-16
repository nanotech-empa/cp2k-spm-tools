from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np

HARTREE_TO_EV = 27.211386245988

_LIST_RE = re.compile(r"list(\d+)")


def _read_header(path: Path) -> tuple[list[str], float | None]:
    with path.open() as handle:
        first = handle.readline()
        second = handle.readline()

    fermi_au = None
    match = re.search(
        r"E\(Fermi\)\s*=\s*([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)",
        first,
    )
    if match:
        fermi_au = float(match.group(1))

    tokens = second.strip().lstrip("#").split()
    try:
        start = tokens.index("Occupation") + 1
    except ValueError as exc:
        raise ValueError(f"Cannot find PDOS channel header in {path}") from exc
    return tokens[start:], fermi_au


def _atom_index_from_filename(path: Path) -> int:
    match = _LIST_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot infer atom list number from {path.name}")
    return int(match.group(1)) - 1


def _spin_from_filename(path: Path) -> int:
    return 1 if "BETA" in path.name.upper() else 0


def parse_cp2k_atom_pdos_files(
    pattern: str | Path,
    *,
    threshold: float = 1.0e-4,
) -> dict[str, np.ndarray]:
    """Parse CP2K atom LDOS/PDOS files into sparse projection arrays.

    The sparse rows store non-negligible projection values as
    spin, atom_index, mo_index, channel_index, value. Atom and MO indices
    are zero-based. Per-spin eigenvalues and occupations are stored once and can
    be joined through mo_index.
    """

    files = sorted(Path(p) for p in glob.glob(str(pattern)))
    if not files:
        raise FileNotFoundError(f"No CP2K PDOS files matched {pattern!r}")

    channel_to_index: dict[str, int] = {}
    rows_spin: list[int] = []
    rows_atom: list[int] = []
    rows_mo: list[int] = []
    rows_channel: list[int] = []
    rows_value: list[float] = []
    evals_by_spin: dict[int, np.ndarray] = {}
    occs_by_spin: dict[int, np.ndarray] = {}
    fermi_by_spin: dict[int, float] = {}

    for path in files:
        atom_index = _atom_index_from_filename(path)
        spin = _spin_from_filename(path)
        channels, fermi_au = _read_header(path)
        if fermi_au is not None:
            fermi_by_spin.setdefault(spin, fermi_au)

        data = np.loadtxt(path)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 3 + len(channels):
            raise ValueError(
                f"PDOS file {path} has {data.shape[1]} columns but header lists "
                f"{len(channels)} channels"
            )

        mo_index = data[:, 0].astype(np.int64) - 1
        evals_by_spin.setdefault(spin, data[:, 1].astype(np.float64))
        occs_by_spin.setdefault(spin, data[:, 2].astype(np.float64))

        projections = data[:, 3 : 3 + len(channels)]
        mo_rows, channel_cols = np.nonzero(np.abs(projections) >= threshold)
        for row, col in zip(mo_rows, channel_cols):
            channel = channels[col]
            channel_index = channel_to_index.setdefault(channel, len(channel_to_index))
            rows_spin.append(spin)
            rows_atom.append(atom_index)
            rows_mo.append(int(mo_index[row]))
            rows_channel.append(channel_index)
            rows_value.append(float(projections[row, col]))

    channels_ordered = np.empty(len(channel_to_index), dtype="U16")
    for channel, index in channel_to_index.items():
        channels_ordered[index] = channel

    arrays: dict[str, np.ndarray] = {
        "format_version": np.asarray(1, dtype=np.int64),
        "threshold": np.asarray(threshold, dtype=np.float64),
        "channels": channels_ordered,
        "spin": np.asarray(rows_spin, dtype=np.int16),
        "atom_index": np.asarray(rows_atom, dtype=np.int64),
        "mo_index": np.asarray(rows_mo, dtype=np.int64),
        "channel_index": np.asarray(rows_channel, dtype=np.int16),
        "projection": np.asarray(rows_value, dtype=np.float32),
    }
    for spin in sorted(evals_by_spin):
        arrays[f"evals_au_spin_{spin}"] = evals_by_spin[spin]
        arrays[f"evals_ev_spin_{spin}"] = evals_by_spin[spin] * HARTREE_TO_EV
        arrays[f"occs_spin_{spin}"] = occs_by_spin[spin]
        if spin in fermi_by_spin:
            arrays[f"fermi_au_spin_{spin}"] = np.asarray(
                fermi_by_spin[spin], dtype=np.float64
            )
            arrays[f"fermi_ev_spin_{spin}"] = np.asarray(
                fermi_by_spin[spin] * HARTREE_TO_EV, dtype=np.float64
            )

    return arrays


def write_sparse_atom_pdos_npz(
    pattern: str | Path,
    output_path: str | Path,
    *,
    threshold: float = 1.0e-4,
) -> None:
    arrays = parse_cp2k_atom_pdos_files(pattern, threshold=threshold)
    np.savez_compressed(output_path, **arrays)
