from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from cp2k_spm_tools.cli.unfold_wfn_sparse import (
    parse_atom_indices,
    parse_path_labels,
    parse_vectors,
    write_unfolding_npz,
)
from cp2k_spm_tools.cp2k_unfolding.geometry import (
    build_modulo_lattice_ao_mapping,
    infer_aos_per_symbol_from_overlap_metadata,
    infer_aos_per_symbol_from_wfn,
    snap_primitive_vectors_to_supercell,
)
from cp2k_spm_tools.cp2k_unfolding.io import (
    Cp2kOverlapMatrixLog,
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


def _pack_overlap(overlap: Cp2kOverlapMatrixLog) -> dict[str, np.ndarray | tuple[int, int]]:
    matrix = overlap.matrix.tocsr()
    return {
        "data": matrix.data,
        "indices": matrix.indices,
        "indptr": matrix.indptr,
        "shape": matrix.shape,
        "basis_index": overlap.basis_index,
        "atom_index": overlap.atom_index,
        "element": overlap.element,
        "orbital": overlap.orbital,
    }


def _unpack_overlap(payload: dict[str, np.ndarray | tuple[int, int]]) -> Cp2kOverlapMatrixLog:
    matrix = sp.csr_matrix(
        (payload["data"], payload["indices"], payload["indptr"]),
        shape=payload["shape"],
    )
    return Cp2kOverlapMatrixLog(
        matrix=matrix,
        basis_index=payload["basis_index"],
        atom_index=payload["atom_index"],
        element=payload["element"],
        orbital=payload["orbital"],
    )


def write_unfolding_npz_mpi(
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
    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise ImportError("mpi4py is required for the MPI unfolding CLI.") from exc

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size == 1:
        write_unfolding_npz(
            wfn_path=wfn_path,
            overlap_path=overlap_path,
            xyz_path=xyz_path,
            cp2k_input_path=cp2k_input_path,
            output_path=output_path,
            primitive_vectors_approx=primitive_vectors_approx,
            lattice_type=lattice_type,
            path_labels=path_labels,
            emin=emin,
            emax=emax,
            tol=tol,
            basis_cluster_tol=basis_cluster_tol,
            overlap_format=overlap_format,
            overlap_threshold=overlap_threshold,
            pdos_pattern=pdos_pattern,
            pdos_output_path=pdos_output_path,
            pdos_threshold=pdos_threshold,
            primitive_basis_atom_indices=primitive_basis_atom_indices,
        )
        return

    dim = int(primitive_vectors_approx.shape[0])
    if rank == 0:
        supercell_vectors = parse_cp2k_cell_vectors(cp2k_input_path, dim=dim)
        primitive_vectors, supercell_matrix, matrix_float, correction_norm = snap_primitive_vectors_to_supercell(
            primitive_vectors_approx,
            supercell_vectors,
        )
        resolved_lattice_type = lattice_type
        if resolved_lattice_type == "auto":
            resolved_lattice_type = guess_2d_lattice_type(primitive_vectors) if dim == 2 else "1d"
        hs_points, default_path = standard_kpath(
            dim,
            lattice_type=resolved_lattice_type,
            primitive_vectors=primitive_vectors,
        )
        resolved_path_labels = path_labels if path_labels else default_path
        missing = [label for label in resolved_path_labels if label not in hs_points]
        if missing:
            raise ValueError(f"Path labels not available for {resolved_lattice_type}: {missing}")

        symbols, coords = read_xyz_coordinates(xyz_path)
        resolved_overlap_format = overlap_format
        if resolved_overlap_format == "auto":
            resolved_overlap_format = "sparse" if str(overlap_path).endswith(".npz") else "log"
        if resolved_overlap_format == "sparse":
            overlap = read_sparse_overlap_npz(overlap_path)
        elif resolved_overlap_format == "log":
            overlap = parse_cp2k_overlap_matrix_log_data(
                overlap_path,
                threshold=overlap_threshold,
            )
        else:
            raise ValueError(f"Unknown overlap format: {resolved_overlap_format!r}")

        k_frac_folded = folded_kpoints_from_supercell_matrix(supercell_matrix)
        k_cart_folded = kfrac_to_cart(k_frac_folded, primitive_vectors)
        path_k_indices, path_x, path_segments, path_t, path_q_equiv, x_ticks = project_kpoints_to_kpath(
            k_frac_folded,
            hs_points,
            resolved_path_labels,
            primitive_vectors,
            tol_cart=1.0e-6,
        )

        setup = {
            "supercell_vectors": supercell_vectors,
            "primitive_vectors": primitive_vectors,
            "supercell_matrix": supercell_matrix,
            "matrix_float": matrix_float,
            "correction_norm": correction_norm,
            "lattice_type": resolved_lattice_type,
            "path_labels": resolved_path_labels,
            "symbols": symbols,
            "coords": coords,
            "overlap": _pack_overlap(overlap),
            "k_frac_folded": k_frac_folded,
            "k_cart_folded": k_cart_folded,
            "path_k_indices": path_k_indices,
            "path_x": path_x,
            "path_segments": path_segments,
            "path_t": path_t,
            "path_q_equiv": path_q_equiv,
            "x_ticks": x_ticks,
            "hs_points": hs_points,
        }
    else:
        setup = None

    setup = comm.bcast(setup, root=0)
    supercell_vectors = setup["supercell_vectors"]
    primitive_vectors = setup["primitive_vectors"]
    supercell_matrix = setup["supercell_matrix"]
    symbols = setup["symbols"]
    coords = setup["coords"]
    overlap = _unpack_overlap(setup["overlap"])
    k_cart_folded = setup["k_cart_folded"]

    wfn = read_cp2k_wfn(wfn_path, emin=emin, emax=emax)

    if rank == 0:
        arrays = {
            "format_version": np.asarray(1, dtype=np.int64),
            "dim": np.asarray(dim, dtype=np.int64),
            "primitive_vectors_approx": primitive_vectors_approx,
            "primitive_vectors": primitive_vectors,
            "supercell_vectors": supercell_vectors,
            "supercell_matrix": supercell_matrix.astype(np.int64),
            "supercell_matrix_float": setup["matrix_float"],
            "correction_norm": np.asarray(setup["correction_norm"], dtype=np.float64),
            "k_frac_folded": setup["k_frac_folded"],
            "path_k_indices": setup["path_k_indices"].astype(np.int64),
            "path_x": setup["path_x"],
            "path_segments": setup["path_segments"].astype(np.int64),
            "path_t": setup["path_t"],
            "path_q_equiv": setup["path_q_equiv"],
            "x_ticks": np.asarray(setup["x_ticks"], dtype=np.float64),
            "x_tick_labels": np.asarray(setup["path_labels"], dtype="U16"),
            "path_labels": np.asarray(setup["path_labels"], dtype="U16"),
            "lattice_type": np.asarray(setup["lattice_type"], dtype="U32"),
            "ref_energy_ev": np.asarray(wfn.ref_energy_ev, dtype=np.float64),
        }
        for label, point in setup["hs_points"].items():
            arrays[f"hs_point_{label}"] = np.asarray(point, dtype=np.float64)
        if pdos_output_path is not None:
            arrays["pdos_projection_filename"] = np.asarray(Path(pdos_output_path).name, dtype="U128")
            arrays["pdos_projection_threshold"] = np.asarray(pdos_threshold, dtype=np.float64)
    else:
        arrays = None

    for spin, coeffs in enumerate(wfn.coeffs):
        if overlap.matrix.shape != (coeffs.shape[1], coeffs.shape[1]):
            raise ValueError(
                f"Overlap shape {overlap.matrix.shape} does not match WFN AO count {coeffs.shape[1]}"
            )
        if getattr(overlap, "atom_index", None) is not None and len(overlap.atom_index):
            aos_per_symbol = infer_aos_per_symbol_from_overlap_metadata(symbols, overlap.atom_index)
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
            verbose=(rank == 0 and spin == 0),
        )

        if rank == 0:
            arrays[f"atom_mapping_displacements_spin_{spin}"] = mapping.atom_displacements_cart
            arrays[f"atom_to_basis_spin_{spin}"] = mapping.atom_to_basis.astype(np.int64)
            arrays[f"atom_to_replica_spin_{spin}"] = mapping.atom_to_replica.astype(np.int64)
            arrays[f"basis_frac_coords_spin_{spin}"] = mapping.basis_frac_coords
            if primitive_basis_atom_indices is not None:
                arrays["primitive_basis_atom_indices"] = np.asarray(
                    primitive_basis_atom_indices, dtype=np.int64
                ) + 1

        nmo = coeffs.shape[0]
        local_indices = np.array_split(np.arange(nmo, dtype=np.int64), size)[rank]
        if local_indices.size:
            print(f"rank {rank}/{size}: spin {spin}, MOs {local_indices[0]}..{local_indices[-1]}")
            local_coeffs = coeffs[local_indices]
            local_weights = unfold_band_weights_full(
                local_coeffs,
                k_cart_folded,
                overlap.matrix,
                mapping,
                verbose=(rank == 0),
            )
            local_norms = mo_norms_sparse(local_coeffs, overlap.matrix)
        else:
            local_weights = np.empty((len(k_cart_folded), 0), dtype=float)
            local_norms = np.empty(0, dtype=float)

        gathered = comm.gather((local_indices, local_weights, local_norms), root=0)
        if rank == 0:
            weights = np.empty((len(k_cart_folded), nmo), dtype=float)
            norms = np.empty(nmo, dtype=float)
            for indices, block, block_norms in gathered:
                weights[:, indices] = block
                norms[indices] = block_norms
            arrays[f"evals_ev_spin_{spin}"] = wfn.evals_ev[spin]
            arrays[f"occs_spin_{spin}"] = wfn.occs[spin]
            arrays[f"weights_spin_{spin}"] = weights
            arrays[f"mo_norms_spin_{spin}"] = norms

    if rank == 0:
        np.savez_compressed(output_path, **arrays)
        if pdos_pattern is not None and pdos_output_path is not None:
            write_sparse_atom_pdos_npz(
                pdos_pattern,
                pdos_output_path,
                threshold=pdos_threshold,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute CP2K unfolding weights with MPI over MOs.")
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

    write_unfolding_npz_mpi(
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
