from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cp2k_spm_tools.cp2k_unfolding.geometry import (
    build_modulo_lattice_ao_mapping,
    infer_aos_per_symbol_from_overlap_metadata,
    infer_aos_per_symbol_from_wfn,
    snap_primitive_vectors_to_supercell,
)
from cp2k_spm_tools.cp2k_unfolding.io import (
    parse_cp2k_cell_vectors,
    parse_cp2k_overlap_matrix_log_data,
    read_cp2k_wfn,
    read_sparse_overlap_npz,
    read_xyz_coordinates,
)
from cp2k_spm_tools.cp2k_unfolding.kpath import (
    folded_kpoints_from_supercell_matrix,
    guess_2d_lattice_type,
    kfrac_to_cart,
    project_kpoints_to_kpath,
    standard_kpath,
)
from cp2k_spm_tools.cp2k_unfolding.pdos import write_sparse_atom_pdos_npz
from cp2k_spm_tools.cp2k_unfolding.unfolding import mo_norms_sparse, unfold_band_weights_full


def parse_vectors(text: str) -> np.ndarray:
    rows = []
    for row in text.replace(";", "\n").splitlines():
        row = row.strip()
        if not row:
            continue
        values = [float(x) for x in row.replace(",", " ").split()]
        if len(values) != 3:
            raise ValueError("Each primitive vector must contain three numbers")
        rows.append(values)
    if not rows:
        raise ValueError("At least one primitive vector is required")
    return np.asarray(rows, dtype=float)




def parse_atom_indices(text: str | None) -> list[int] | None:
    """Parse 1-based atom indices/ranges into zero-based indices."""
    if text is None or not str(text).strip():
        return None
    items: list[int] = []
    clean = str(text).replace(",", " ")
    for token in clean.split():
        if ".." in token:
            left, right = token.split("..", 1)
            start = int(left)
            stop = int(right)
            step = 1 if stop >= start else -1
            items.extend(range(start, stop + step, step))
        elif "-" in token and token.count("-") == 1 and not token.startswith("-"):
            left, right = token.split("-", 1)
            start = int(left)
            stop = int(right)
            step = 1 if stop >= start else -1
            items.extend(range(start, stop + step, step))
        else:
            items.append(int(token))
    if any(index < 1 for index in items):
        raise ValueError("Atom indices are 1-based and must be positive")
    return [index - 1 for index in items]

def parse_path_labels(text: str) -> list[str]:
    clean = text.strip().replace("Γ", "G")
    if not clean:
        return []
    if "-" in clean or "," in clean or " " in clean:
        return [x for x in clean.replace(",", "-").replace(" ", "-").split("-") if x]
    return list(clean)


def write_unfolding_npz(
    *,
    wfn_path: str | Path,
    overlap_path: str | Path,
    xyz_path: str | Path,
    cp2k_input_path: str | Path,
    output_path: str | Path,
    primitive_vectors_approx: np.ndarray,
    lattice_type: str = "auto",
    path_labels: list[str] | None = None,
    emin: float | None = None,
    emax: float | None = None,
    tol: float = 1.0e-5,
    basis_cluster_tol: float = 5.0e-2,
    overlap_format: str = "auto",
    overlap_threshold: float = 1.0e-10,
    pdos_pattern: str | Path | None = None,
    pdos_output_path: str | Path | None = None,
    pdos_threshold: float = 1.0e-4,
    primitive_basis_atom_indices: list[int] | None = None,
) -> None:
    dim = int(primitive_vectors_approx.shape[0])
    supercell_vectors = parse_cp2k_cell_vectors(cp2k_input_path, dim=dim)
    primitive_vectors, supercell_matrix, matrix_float, correction_norm = snap_primitive_vectors_to_supercell(
        primitive_vectors_approx,
        supercell_vectors,
    )

    if lattice_type == "auto":
        lattice_type = guess_2d_lattice_type(primitive_vectors) if dim == 2 else "1d"
    hs_points, default_path = standard_kpath(dim, lattice_type=lattice_type, primitive_vectors=primitive_vectors)
    if path_labels is None or not path_labels:
        path_labels = default_path
    missing = [label for label in path_labels if label not in hs_points]
    if missing:
        raise ValueError(f"Path labels not available for {lattice_type}: {missing}")

    symbols, coords = read_xyz_coordinates(xyz_path)
    if overlap_format == "auto":
        overlap_format = "sparse" if str(overlap_path).endswith(".npz") else "log"
    if overlap_format == "sparse":
        overlap = read_sparse_overlap_npz(overlap_path)
    elif overlap_format == "log":
        overlap = parse_cp2k_overlap_matrix_log_data(
            overlap_path,
            threshold=overlap_threshold,
        )
    else:
        raise ValueError(f"Unknown overlap format: {overlap_format!r}")

    wfn = read_cp2k_wfn(wfn_path, emin=emin, emax=emax)

    k_frac_folded = folded_kpoints_from_supercell_matrix(supercell_matrix)
    k_cart_folded = kfrac_to_cart(k_frac_folded, primitive_vectors)
    path_k_indices, path_x, path_segments, path_t, path_q_equiv, x_ticks = project_kpoints_to_kpath(
        k_frac_folded,
        hs_points,
        path_labels,
        primitive_vectors,
        tol_cart=1.0e-6,
    )

    arrays = {
        "format_version": np.asarray(1, dtype=np.int64),
        "dim": np.asarray(dim, dtype=np.int64),
        "primitive_vectors_approx": primitive_vectors_approx,
        "primitive_vectors": primitive_vectors,
        "supercell_vectors": supercell_vectors,
        "supercell_matrix": supercell_matrix.astype(np.int64),
        "supercell_matrix_float": matrix_float,
        "correction_norm": np.asarray(correction_norm, dtype=np.float64),
        "k_frac_folded": k_frac_folded,
        "path_k_indices": path_k_indices.astype(np.int64),
        "path_x": path_x,
        "path_segments": path_segments.astype(np.int64),
        "path_t": path_t,
        "path_q_equiv": path_q_equiv,
        "x_ticks": np.asarray(x_ticks, dtype=np.float64),
        "x_tick_labels": np.asarray(path_labels, dtype="U16"),
        "path_labels": np.asarray(path_labels, dtype="U16"),
        "lattice_type": np.asarray(lattice_type, dtype="U32"),
        "ref_energy_ev": np.asarray(wfn.ref_energy_ev, dtype=np.float64),
    }

    for label, point in hs_points.items():
        arrays[f"hs_point_{label}"] = np.asarray(point, dtype=np.float64)

    if pdos_output_path is not None:
        arrays["pdos_projection_filename"] = np.asarray(
            Path(pdos_output_path).name, dtype="U128"
        )
        arrays["pdos_projection_threshold"] = np.asarray(
            pdos_threshold, dtype=np.float64
        )

    for spin, coeffs in enumerate(wfn.coeffs):
        if overlap.matrix.shape != (coeffs.shape[1], coeffs.shape[1]):
            raise ValueError(
                f"Overlap shape {overlap.matrix.shape} does not match WFN AO count {coeffs.shape[1]}"
            )
        if getattr(overlap, "atom_index", None) is not None and len(overlap.atom_index):
            aos_per_symbol = infer_aos_per_symbol_from_overlap_metadata(
                symbols, overlap.atom_index
            )
        else:
            aos_per_symbol = infer_aos_per_symbol_from_wfn(symbols, coeffs.shape[1])
        mapping = build_modulo_lattice_ao_mapping(
            symbols=symbols,
            coords_cart=coords,
            primitive_vectors=primitive_vectors,
            supercell_vectors=supercell_vectors,
            aos_per_symbol=aos_per_symbol,
            tol=basis_cluster_tol,
            primitive_basis_atom_indices=primitive_basis_atom_indices,
        )
        arrays[f"atom_mapping_displacements_spin_{spin}"] = mapping.atom_displacements_cart
        arrays[f"atom_to_basis_spin_{spin}"] = mapping.atom_to_basis.astype(np.int64)
        arrays[f"atom_to_replica_spin_{spin}"] = mapping.atom_to_replica.astype(np.int64)
        arrays[f"basis_frac_coords_spin_{spin}"] = mapping.basis_frac_coords
        if primitive_basis_atom_indices is not None:
            arrays["primitive_basis_atom_indices"] = np.asarray(
                primitive_basis_atom_indices, dtype=np.int64
            ) + 1
        weights = unfold_band_weights_full(
            coeffs,
            k_cart_folded,
            overlap.matrix,
            mapping,
            verbose=True,
        )
        arrays[f"evals_ev_spin_{spin}"] = wfn.evals_ev[spin]
        arrays[f"occs_spin_{spin}"] = wfn.occs[spin]
        arrays[f"weights_spin_{spin}"] = weights
        arrays[f"mo_norms_spin_{spin}"] = mo_norms_sparse(coeffs, overlap.matrix)

    np.savez_compressed(output_path, **arrays)

    if pdos_pattern is not None and pdos_output_path is not None:
        write_sparse_atom_pdos_npz(
            pdos_pattern,
            pdos_output_path,
            threshold=pdos_threshold,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute CP2K localized-basis unfolding weights.")
    parser.add_argument("wfn")
    parser.add_argument("overlap")
    parser.add_argument("output")
    parser.add_argument("--xyz", required=True)
    parser.add_argument("--cp2k-input", required=True)
    parser.add_argument("--primitive-vectors", required=True)
    parser.add_argument("--path", default="G-K-M-G")
    parser.add_argument("--lattice-type", default="auto")
    parser.add_argument("--emin", type=float, default=None)
    parser.add_argument("--emax", type=float, default=None)
    parser.add_argument("--tol", type=float, default=1.0e-5)
    parser.add_argument("--basis-cluster-tol", type=float, default=5.0e-2)
    parser.add_argument("--primitive-basis-atoms", default=None)
    parser.add_argument("--overlap-format", choices=["auto", "sparse", "log"], default="auto")
    parser.add_argument("--overlap-threshold", type=float, default=1.0e-10)
    parser.add_argument("--pdos-glob", default=None)
    parser.add_argument("--pdos-output", default=None)
    parser.add_argument("--pdos-threshold", type=float, default=1.0e-4)
    args = parser.parse_args(argv)

    write_unfolding_npz(
        wfn_path=args.wfn,
        overlap_path=args.overlap,
        xyz_path=args.xyz,
        cp2k_input_path=args.cp2k_input,
        output_path=args.output,
        primitive_vectors_approx=parse_vectors(args.primitive_vectors),
        lattice_type=args.lattice_type,
        path_labels=parse_path_labels(args.path),
        emin=args.emin,
        emax=args.emax,
        tol=args.tol,
        basis_cluster_tol=args.basis_cluster_tol,
        overlap_format=args.overlap_format,
        overlap_threshold=args.overlap_threshold,
        pdos_pattern=args.pdos_glob,
        pdos_output_path=args.pdos_output,
        pdos_threshold=args.pdos_threshold,
        primitive_basis_atom_indices=parse_atom_indices(args.primitive_basis_atoms),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
