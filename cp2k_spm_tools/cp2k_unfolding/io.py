from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase.geometry import cellpar_to_cell
from scipy import constants

from cp2k_spm_tools.cp2k_overlap_matrix import Cp2kOverlapMatrix

__all__ = [
    "Cp2kOverlapMatrix",
    "SupercellWavefunctions",
    "parse_cp2k_cell_vectors",
    "print_eigenvalue_summary",
    "read_cp2k_wfn",
    "read_xyz_coordinates",
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
        raise ImportError("cp2k-spm-tools is required. Install it or add the repository to PYTHONPATH.") from exc

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
    """Read one numeric CP2K &CELL as row vectors in Angstrom.

    Support A/B/C or ABC with ALPHA_BETA_GAMMA (alias ANGLES), with
    CP2K's default Angstrom/degree units. ABC takes precedence over A/B/C.
    External cells, imposed symmetry and cell replication must first be
    resolved to explicit vectors; they cannot be silently ignored here.
    """
    path = Path(cp2k_input_file)
    if not path.exists():
        raise FileNotFoundError(f"CP2K input file not found: {path}")

    lines = path.read_text(errors="replace").splitlines()
    depth = 0
    sections: list[dict[str, list[str]]] = []
    keywords = {"A", "B", "C", "ABC", "ALPHA_BETA_GAMMA", "CELL_FILE_NAME", "SYMMETRY", "MULTIPLE_UNIT_CELL"}
    bohr_to_angstrom = constants.physical_constants["Bohr radius"][0] * 1e10
    length_units = {
        "angstrom": 1.0,
        "ang": 1.0,
        "bohr": bohr_to_angstrom,
        "au": bohr_to_angstrom,
        "a.u.": bohr_to_angstrom,
        "nm": 10.0,
        "pm": 0.01,
    }
    angle_units = {"deg": 1.0, "degree": 1.0, "rad": 180.0 / np.pi, "radian": 180.0 / np.pi}

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
            sections.append({})
            depth = 1
            continue
        if depth and key == "&END":
            depth -= 1
            continue
        if depth and key.startswith("&"):
            depth += 1
            continue
        if depth != 1:
            continue
        key = "ALPHA_BETA_GAMMA" if key == "ANGLES" else key
        if key in keywords:
            if key in sections[-1]:
                raise ValueError(f"Repeated CP2K CELL keyword: {key}")
            sections[-1][key] = parts[1:]

    if depth or len(sections) != 1:
        raise ValueError("Expected exactly one complete CP2K &CELL section")
    entries = sections[0]
    if "CELL_FILE_NAME" in entries or [x.upper() for x in entries.get("SYMMETRY", ["NONE"])] != ["NONE"]:
        raise ValueError("External or symmetry-imposed CP2K cells must be supplied as resolved explicit A/B/C vectors")
    if [int(x) for x in entries.get("MULTIPLE_UNIT_CELL", ["1", "1", "1"])] != [1, 1, 1]:
        raise ValueError("Replicated CP2K cells must be supplied as resolved explicit A/B/C vectors")

    def read_triplet(key: str, units: dict[str, float], default_unit: str) -> np.ndarray:
        unit, values = strip_unit(entries[key])
        unit = unit or default_unit
        if unit not in units:
            raise ValueError(f"Unsupported CP2K {key} unit: {unit}")
        if len(values) != 3:
            raise ValueError(f"CP2K {key} requires exactly three numeric values")
        result = np.array([float(x.replace("D", "E").replace("d", "e")) for x in values]) * units[unit]
        if not np.all(np.isfinite(result)):
            raise ValueError(f"CP2K {key} requires finite numeric values")
        return result

    if "ABC" in entries:
        abc = read_triplet("ABC", length_units, "angstrom")
        angles = (
            read_triplet("ALPHA_BETA_GAMMA", angle_units, "deg") if "ALPHA_BETA_GAMMA" in entries else np.full(3, 90.0)
        )
        if np.any(abc <= 0) or np.any(angles <= 0) or np.any(angles >= 180):
            raise ValueError("CP2K ABC lengths must be positive and cell angles must lie between 0 and 180 degrees")
        try:
            cell = cellpar_to_cell(np.concatenate([abc, angles]))
        except (AssertionError, ValueError) as exc:
            raise ValueError("CP2K ABC angles do not define a nondegenerate cell") from exc
    elif all(key in entries for key in ("A", "B", "C")):
        cell = np.vstack([read_triplet(key, length_units, "angstrom") for key in ("A", "B", "C")])
    else:
        raise ValueError("Could not parse CP2K cell vectors from input file")

    if np.linalg.matrix_rank(cell) != 3:
        raise ValueError("CP2K cell vectors must define a nondegenerate three-dimensional cell")

    if dim is None:
        dim = 3
    if dim not in (1, 2, 3):
        raise ValueError("Cell dimension must be 1, 2 or 3")
    return cell[:dim]
