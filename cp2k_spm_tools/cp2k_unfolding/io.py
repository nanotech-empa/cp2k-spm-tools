from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import constants

from cp2k_spm_tools.cp2k_overlap_matrix import (
    Cp2kOverlapMatrixLog,
    parse_cp2k_overlap_matrix_log,
    parse_cp2k_overlap_matrix_log_data,
    read_sparse_overlap_npz,
    write_sparse_overlap_npz,
)

__all__ = [
    "Cp2kOverlapMatrixLog",
    "SupercellWavefunctions",
    "parse_cp2k_cell_vectors",
    "parse_cp2k_overlap_matrix_log",
    "parse_cp2k_overlap_matrix_log_data",
    "print_eigenvalue_summary",
    "read_cp2k_wfn",
    "read_sparse_overlap_npz",
    "read_xyz_coordinates",
    "write_sparse_overlap_npz",
]

hartree_to_ev = constants.physical_constants["Hartree energy in eV"][0]


@dataclass
class SupercellWavefunctions:
    evals_ev: list[np.ndarray]
    occs: list[np.ndarray]
    coeffs: list[np.ndarray]  # coeffs[ispin][imo, iao]
    ref_energy_ev: float


def read_cp2k_wfn(
    wfn_path: str | Path,
    *,
    emin: float | None = None,
    emax: float | None = None,
    n_occ: int | None = None,
    n_virt: int | None = None,
) -> SupercellWavefunctions:
    try:
        from cp2k_spm_tools.cp2k_wfn_file import Cp2kWfnFile
    except ImportError as exc:
        raise ImportError(
            "cp2k-spm-tools is required. Install it or add the repository to PYTHONPATH."
        ) from exc

    cwf = Cp2kWfnFile(mpi_rank=0, mpi_size=1, mpi_comm=None)
    try:
        cwf.load_restart_wfn_file(
            str(wfn_path),
            emin=emin,
            emax=emax,
            n_occ=n_occ,
            n_virt=n_virt,
        )
    except IndexError as exc:
        raise RuntimeError(
            "Failed while reading the CP2K WFN. Most likely the .wfn contains "
            "occupied orbitals only, so cp2k-spm-tools cannot access the LUMO "
            "to define the HOMO-LUMO reference energy. Re-run CP2K with ADDED_MOS > 0."
        ) from exc

    return SupercellWavefunctions(
        evals_ev=[np.asarray(x, dtype=float) for x in cwf.evals_sel],
        occs=[np.asarray(x, dtype=float) for x in cwf.occs_sel],
        coeffs=[np.asarray(x, dtype=float) for x in cwf.coef_array],
        ref_energy_ev=float(cwf.ref_energy),
    )


def print_eigenvalue_summary(wfn: SupercellWavefunctions, spin: int = 0, n: int = 10) -> None:
    ev = wfn.evals_ev[spin]
    eh = ev / hartree_to_ev

    print(f"Hartree to eV: {hartree_to_ev:.12f}")
    print(f"Number of selected eigenvalues: {len(ev)}")
    print(f"Reference energy: {wfn.ref_energy_ev:.10f} eV")
    print(f"Reference energy: {wfn.ref_energy_ev / hartree_to_ev:.10f} Ha")
    print()

    print(f"First {min(n, len(ev))} selected eigenvalues:")
    for i, (e_ev, e_ha) in enumerate(zip(ev[:n], eh[:n])):
        print(f"{i:4d}  {e_ev:16.10f} eV   {e_ha:16.10f} Ha")

    print()

    print(f"Last {min(n, len(ev))} selected eigenvalues:")
    offset = max(0, len(ev) - n)
    for i, (e_ev, e_ha) in enumerate(zip(ev[-n:], eh[-n:]), start=offset):
        print(f"{i:4d}  {e_ev:16.10f} eV   {e_ha:16.10f} Ha")


def read_xyz_coordinates(path: str | Path) -> tuple[list[str], np.ndarray]:
    path = Path(path)
    lines = path.read_text().splitlines()
    natom = int(lines[0].split()[0])

    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in lines[2 : 2 + natom]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return symbols, np.asarray(coords, dtype=float)


def parse_cp2k_cell_vectors(cp2k_input_file: str | Path, dim: int | None = None) -> np.ndarray:
    """Parse CP2K &CELL vectors A/B/C or ABC from an input file."""
    path = Path(cp2k_input_file)
    if not path.exists():
        raise FileNotFoundError(f"CP2K input file not found: {path}")

    lines = path.read_text(errors="replace").splitlines()
    in_cell = False
    vectors: dict[str, np.ndarray] = {}
    abc = None

    def strip_unit(tokens: list[str]) -> tuple[str | None, list[str]]:
        if tokens and tokens[0].startswith("[") and tokens[0].endswith("]"):
            return tokens[0].strip("[]").lower(), tokens[1:]
        return None, tokens

    for raw in lines:
        line = raw.split("!", 1)[0].split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        key = parts[0].upper()

        if key == "&CELL":
            in_cell = True
            continue
        if in_cell and key.startswith("&END"):
            in_cell = False
            continue
        if not in_cell:
            continue

        if key in {"A", "B", "C"}:
            _, values = strip_unit(parts[1:])
            if len(values) >= 3:
                vectors[key] = np.array([float(x) for x in values[:3]], dtype=float)
        elif key == "ABC":
            _, values = strip_unit(parts[1:])
            if len(values) >= 3:
                abc = np.array([float(x) for x in values[:3]], dtype=float)

    if all(k in vectors for k in ("A", "B", "C")):
        cell = np.vstack([vectors["A"], vectors["B"], vectors["C"]])
    elif abc is not None:
        cell = np.diag(abc)
    else:
        raise ValueError("Could not parse CP2K cell vectors from input file")

    if dim is None:
        dim = 3
    return cell[:dim]
