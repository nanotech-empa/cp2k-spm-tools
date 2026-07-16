from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .geometry import AOMapping


def mo_norms_sparse(coeffs: np.ndarray, S_super: sp.spmatrix) -> np.ndarray:
    """Return diagonal C S C values without densifying the overlap matrix."""
    SC = S_super @ coeffs.T
    return np.einsum("ni,in->n", coeffs.conjugate(), SC)


def primitive_overlap_metric(S_super: sp.spmatrix, mapping: AOMapping) -> np.ndarray:
    """Extract the replica-0 primitive overlap block for diagnostics."""
    home = np.where(mapping.ao_to_replica == 0)[0]
    order = np.argsort(mapping.ao_to_local[home])
    home = home[order]

    if home.size != mapping.nao_prim:
        raise ValueError("Replica 0 does not contain exactly nao_prim AOs.")

    return S_super[home[:, None], home].toarray()


@dataclass
class SparseUnfoldingCache:
    """Precomputed sparse data for fast non-orthogonal unfolding."""

    S_coo: sp.coo_matrix
    coeffs: np.ndarray
    SC: np.ndarray
    ao_to_replica: np.ndarray
    ao_to_local: np.ndarray
    replica_vectors_cart: np.ndarray
    edge_local_row: np.ndarray
    edge_local_col: np.ndarray
    edge_delta_R: np.ndarray
    nao_prim: int
    nrep: int


def prepare_sparse_unfolding_cache(
    coeffs: np.ndarray,
    S_super: sp.spmatrix,
    mapping: AOMapping,
) -> SparseUnfoldingCache:
    """Precompute all k-independent quantities for sparse non-orthogonal unfolding."""
    S_coo = S_super.tocoo()
    coeffs = np.asarray(coeffs)

    if coeffs.ndim != 2:
        raise ValueError("coeffs must have shape (nmo, nao_super)")
    if coeffs.shape[1] != mapping.nao_super:
        raise ValueError("coeffs AO dimension does not match mapping.nao_super")

    SC = S_super @ coeffs.T

    row = S_coo.row
    col = S_coo.col

    edge_local_row = mapping.ao_to_local[row]
    edge_local_col = mapping.ao_to_local[col]
    edge_delta_R = (
        mapping.replica_vectors_cart[mapping.ao_to_replica[col]]
        - mapping.replica_vectors_cart[mapping.ao_to_replica[row]]
    )

    return SparseUnfoldingCache(
        S_coo=S_coo,
        coeffs=coeffs,
        SC=np.asarray(SC),
        ao_to_replica=mapping.ao_to_replica,
        ao_to_local=mapping.ao_to_local,
        replica_vectors_cart=mapping.replica_vectors_cart,
        edge_local_row=edge_local_row,
        edge_local_col=edge_local_col,
        edge_delta_R=edge_delta_R,
        nao_prim=mapping.nao_prim,
        nrep=mapping.nrep,
    )


def sparse_bloch_overlap_metric_from_cache(
    cache: SparseUnfoldingCache,
    k_cart: np.ndarray,
) -> np.ndarray:
    """Assemble S_k = B_k^H S B_k using only nonzero entries of S."""
    k_cart = np.asarray(k_cart, dtype=float)
    phase = np.exp(1j * (cache.edge_delta_R @ k_cart)) / cache.nrep
    data = cache.S_coo.data * phase

    Sk_sparse = sp.coo_matrix(
        (data, (cache.edge_local_row, cache.edge_local_col)),
        shape=(cache.nao_prim, cache.nao_prim),
        dtype=np.complex128,
    )
    return Sk_sparse.toarray()


def sparse_bloch_rhs_from_cache(
    cache: SparseUnfoldingCache,
    k_cart: np.ndarray,
) -> np.ndarray:
    """Assemble V(k) = B_k^H S C for all selected MOs at once."""
    k_cart = np.asarray(k_cart, dtype=float)
    phase_mu = np.exp(
        -1j * (cache.replica_vectors_cart[cache.ao_to_replica] @ k_cart)
    ) / np.sqrt(cache.nrep)

    weighted_SC = phase_mu[:, None] * cache.SC

    V = np.zeros((cache.nao_prim, cache.coeffs.shape[0]), dtype=np.complex128)
    np.add.at(V, cache.ao_to_local, weighted_SC)
    return V


def unfold_band_weights_sparse_full(
    coeffs: np.ndarray,
    k_path_cart: np.ndarray,
    S_super: sp.spmatrix,
    mapping: AOMapping,
    *,
    rcond: float = 1e-10,
    verbose: bool = True,
) -> np.ndarray:
    """Return weights[ik, imo] with the full sparse non-orthogonal formula."""
    cache = prepare_sparse_unfolding_cache(coeffs, S_super, mapping)
    weights = np.empty((len(k_path_cart), coeffs.shape[0]), dtype=float)

    if verbose:
        print(f"nao_super = {mapping.nao_super}")
        print(f"nao_prim  = {mapping.nao_prim}")
        print(f"nrep      = {mapping.nrep}")
        print(f"nmo       = {coeffs.shape[0]}")
        print(f"nnz(S)    = {cache.S_coo.nnz}")

    for ik, k_cart in enumerate(k_path_cart):
        if verbose and (ik % max(1, len(k_path_cart) // 10) == 0 or ik == len(k_path_cart) - 1):
            print(f"k-point {ik + 1}/{len(k_path_cart)}")

        Sk = sparse_bloch_overlap_metric_from_cache(cache, k_cart)
        V = sparse_bloch_rhs_from_cache(cache, k_cart)

        try:
            X = np.linalg.solve(Sk, V)
        except np.linalg.LinAlgError:
            X = np.linalg.pinv(Sk, rcond=rcond) @ V

        W = np.einsum("an,an->n", V.conjugate(), X)
        weights[ik, :] = np.real_if_close(W).real

    return weights


def spectral_weight_full(
    coeff: np.ndarray,
    k_cart: np.ndarray,
    S_super: sp.spmatrix,
    mapping: AOMapping,
    *,
    rcond: float = 1e-10,
) -> float:
    """Single-orbital wrapper around the sparse full non-orthogonal formula."""
    weights = unfold_band_weights_sparse_full(
        np.asarray(coeff)[None, :],
        np.asarray(k_cart, dtype=float)[None, :],
        S_super,
        mapping,
        rcond=rcond,
        verbose=False,
    )
    return float(weights[0, 0])


def unfold_band_weights_full(
    coeffs: np.ndarray,
    k_path_cart: np.ndarray,
    S_super: sp.spmatrix,
    mapping: AOMapping,
    *,
    verbose: bool = True,
) -> np.ndarray:
    """Backward-compatible name for the sparse full unfolding implementation."""
    return unfold_band_weights_sparse_full(
        coeffs,
        k_path_cart,
        S_super,
        mapping,
        verbose=verbose,
    )


def fourier_project_coefficients(coeff: np.ndarray, k_cart: np.ndarray, mapping: AOMapping) -> np.ndarray:
    """Simple coefficient-only Fourier projection for diagnostics."""
    phase_by_replica = np.exp(-1j * mapping.replica_vectors_cart @ k_cart)
    phases = phase_by_replica[mapping.ao_to_replica]

    d = np.zeros(mapping.nao_prim, dtype=np.complex128)
    np.add.at(d, mapping.ao_to_local, phases * coeff)
    d /= np.sqrt(mapping.nrep)
    return d


def spectral_weight_simple(
    coeff: np.ndarray, k_cart: np.ndarray, S0: np.ndarray, mapping: AOMapping
) -> float:
    """Simplified diagnostic weight, not used for final plots."""
    d = fourier_project_coefficients(coeff, k_cart, mapping)
    w = np.vdot(d, S0 @ d)
    return float(np.real_if_close(w))


def unfold_band_weights(
    coeffs: np.ndarray, k_path_cart: np.ndarray, S_super: sp.spmatrix, mapping: AOMapping
) -> np.ndarray:
    """Backward-compatible alias for the full unfolding formula."""
    return unfold_band_weights_full(coeffs, k_path_cart, S_super, mapping)
