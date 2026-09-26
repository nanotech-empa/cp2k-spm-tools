from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np

from .geometry import lattice_matrix


def reciprocal_vectors(primitive_vectors: np.ndarray) -> np.ndarray:
    A = lattice_matrix(primitive_vectors)
    return 2 * np.pi * np.linalg.inv(A).T


def folded_kpoints_from_supercell_matrix(M: np.ndarray) -> np.ndarray:
    """Fractional primitive reciprocal coordinates of k-points folded to supercell Gamma."""
    dim = M.shape[0]
    det = int(round(abs(np.linalg.det(M))))
    MinvT = np.linalg.inv(M).T

    k_frac: list[np.ndarray] = []
    max_scan = det * 4 + 1
    for indices in np.ndindex(*([max_scan] * dim)):
        q = MinvT @ np.asarray(indices, dtype=float)
        q = q - np.floor(q)

        if not any(np.allclose(q, old, atol=1e-10) for old in k_frac):
            k_frac.append(q)

        if len(k_frac) == det:
            return np.asarray(k_frac)

    raise RuntimeError("Could not find all folded k-points.")


def kfrac_to_cart(k_frac: np.ndarray, primitive_vectors: np.ndarray) -> np.ndarray:
    B = reciprocal_vectors(primitive_vectors)
    dim = B.shape[0]
    k_dim = np.asarray(k_frac) @ B.T
    k_cart = np.zeros((len(k_dim), 3))
    k_cart[:, :dim] = k_dim
    return k_cart


def lattice_lengths_angles_2d(primitive_vectors: np.ndarray) -> tuple[float, float, float]:
    """Return |a|, |b|, and the angle a-b in degrees for a 2D lattice."""
    a = np.asarray(primitive_vectors[0, :2], dtype=float)
    b = np.asarray(primitive_vectors[1, :2], dtype=float)
    la = float(np.linalg.norm(a))
    lb = float(np.linalg.norm(b))
    cosang = float(np.dot(a, b) / (la * lb))
    cosang = max(-1.0, min(1.0, cosang))
    angle = float(np.degrees(np.arccos(cosang)))
    return la, lb, angle


def _reduced_2d_basis(primitive_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-reduce an in-plane row-vector basis, retaining its integer map."""
    vectors = np.asarray(primitive_vectors, dtype=float)
    if vectors.shape != (2, 3) or not np.all(np.isfinite(vectors)):
        raise ValueError("2D primitive vectors must be a finite (2, 3) array")
    if not np.allclose(vectors[:, 2], 0.0, atol=1e-10, rtol=0):
        raise ValueError("2D unfolding requires primitive vectors in the xy plane")
    if np.linalg.matrix_rank(vectors[:, :2]) != 2:
        raise ValueError("2D primitive vectors must be linearly independent")
    transform = np.eye(2, dtype=np.int64)
    reduced = vectors.copy()
    for _ in range(100):
        norms = np.einsum("ij,ij->i", reduced, reduced)
        if norms[1] < norms[0]:
            reduced = reduced[[1, 0]]
            transform = transform[[1, 0]]
            norms = norms[::-1]
        multiple = int(np.rint(np.dot(reduced[0], reduced[1]) / norms[0]))
        if multiple == 0:
            break
        reduced[1] -= multiple * reduced[0]
        transform[1] -= multiple * transform[0]
    else:
        raise ValueError("Could not reduce the 2D primitive basis")
    if np.dot(reduced[0], reduced[1]) < 0:
        reduced[1] *= -1
        transform[1] *= -1
    return reduced, transform


def _classify_reduced_2d(reduced: np.ndarray, rtol: float, angle_tol: float) -> str:
    la, lb, angle = lattice_lengths_angles_2d(reduced)
    equal_lengths = abs(la - lb) <= rtol * max(la, lb)
    if abs(angle - 90.0) <= angle_tol:
        return "square" if equal_lengths else "rectangular"
    if equal_lengths and abs(angle - 60.0) <= angle_tol:
        return "hexagonal"
    # A centered rectangular lattice can reduce either to a rhombus or
    # to a short conventional axis plus one of its equal-length diagonals.
    dot = float(np.dot(reduced[0], reduced[1]))
    if equal_lengths or abs(2 * dot - la**2) <= rtol * la**2:
        return "centered_rectangular"
    return "oblique"


def guess_2d_lattice_type(primitive_vectors: np.ndarray, rtol: float = 1e-3, angle_tol: float = 1e-2) -> str:
    """Identify all five 2D Bravais families, including non-reduced bases."""
    reduced, _ = _reduced_2d_basis(primitive_vectors)
    return _classify_reduced_2d(reduced, rtol, angle_tol)


def _bragg_vertex(metric: np.ndarray, first: Sequence[int], second: Sequence[int]) -> np.ndarray:
    """Intersect two reciprocal Bragg planes in fractional coordinates."""
    normals = np.asarray([first, second], dtype=float)
    rhs = np.einsum("ni,ij,nj->n", normals, metric, normals) / 2
    return np.linalg.solve(normals @ metric, rhs)


def standard_kpath(dim: int, lattice_type: str | None = None, primitive_vectors: np.ndarray | None = None):
    """Return first-Brillouin-zone special points in the input reciprocal basis.

    Reduce the direct lattice, construct vertices from reciprocal Bragg planes,
    then transform fractional coordinates back to the supplied primitive basis.
    No interpolated points are added to the supercell's unfolding grid.
    """
    if lattice_type is not None:
        lattice_type = lattice_type.lower()
    lattice_type = {"hex": "hexagonal", "graphene": "hexagonal"}.get(lattice_type, lattice_type)
    if dim == 1:
        if lattice_type not in {None, "auto", "1d"}:
            raise ValueError(f"lattice_type {lattice_type!r} is incompatible with a 1D cell")
        return {"G": np.array([0.0]), "X": np.array([0.5])}, ["G", "X"]
    if dim != 2:
        raise NotImplementedError("Standard paths are currently implemented only for 1D and 2D.")
    if primitive_vectors is None:
        raise ValueError("primitive_vectors are needed to define a cell-dependent 2D path")

    reduced, transform = _reduced_2d_basis(primitive_vectors)
    detected = _classify_reduced_2d(reduced, 1e-3, 1e-2)
    if lattice_type not in {None, "auto", detected}:
        raise ValueError(f"lattice_type {lattice_type!r} does not match the primitive cell ({detected})")

    if detected == "centered_rectangular" and not np.isclose(
        np.linalg.norm(reduced[0]), np.linalg.norm(reduced[1]), rtol=1e-3, atol=0
    ):
        # Use equal-length primitive diagonals to label the centered cell.
        change = np.array([[0, 1], [-1, 1]])
        reduced = change @ reduced
        transform = change @ transform
    reciprocal = reciprocal_vectors(reduced)
    metric = reciprocal.T @ reciprocal
    points = {"G": np.array([0.0, 0.0])}
    if detected == "square":
        points.update(X=np.array([0.5, 0.0]), M=np.array([0.5, 0.5]))
        path = ["G", "X", "M", "G"]
    elif detected == "rectangular":
        points.update(X=np.array([0.5, 0.0]), S=np.array([0.5, 0.5]), Y=np.array([0.0, 0.5]))
        path = ["G", "X", "S", "Y", "G"]
    elif detected == "hexagonal":
        points.update(K=_bragg_vertex(metric, [1, 0], [1, 1]), M=np.array([0.5, 0.0]))
        path = ["G", "K", "M", "G"]
    elif detected == "centered_rectangular":
        points.update(
            X=_bragg_vertex(metric, [1, 0], [0, -1]),
            A1=_bragg_vertex(metric, [1, 0], [1, 1]),
            Y=np.array([0.5, 0.5]),
        )
        path = ["G", "X", "A1", "Y", "G"]
    else:
        points.update(
            Y=np.array([0.0, 0.5]),
            H=_bragg_vertex(metric, [0, 1], [1, 1]),
            C=np.array([0.5, 0.5]),
            H1=_bragg_vertex(metric, [1, 0], [1, 1]),
            X=np.array([0.5, 0.0]),
        )
        path = ["G", "Y", "H", "C", "H1", "X", "G"]
    points = {label: np.linalg.solve(transform, point) for label, point in points.items()}

    # Preserve the original directions for conventional square/rectangular
    # bases and the original 60-degree hexagonal path.
    la, lb, angle = lattice_lengths_angles_2d(np.asarray(primitive_vectors))
    if detected in {"square", "rectangular"} and abs(angle - 90.0) < 1e-2:
        points = {"G": np.array([0.0, 0.0]), "X": np.array([0.5, 0.0])}
        if detected == "square":
            points["M"] = np.array([0.5, 0.5])
        else:
            points.update(S=np.array([0.5, 0.5]), Y=np.array([0.0, 0.5]))
    elif detected == "hexagonal" and abs(la - lb) <= 1e-3 * max(la, lb):
        if min(abs(angle - 60.0), abs(angle - 120.0)) < 1e-2:
            points = {
                "G": np.array([0.0, 0.0]),
                "K": np.array([2.0 / 3.0 if angle < 90.0 else 1.0 / 3.0, 1.0 / 3.0]),
                "M": np.array([0.5, 0.0]),
            }
    return points, path


def kpath_axis_from_fractional_path(points: dict[str, np.ndarray], path: Sequence[str], primitive_vectors: np.ndarray):
    """Return cumulative x-axis coordinates and tick positions for a high-symmetry path."""
    frac_nodes = [points[label] for label in path]
    cart_nodes = kfrac_to_cart(np.asarray(frac_nodes), primitive_vectors)

    x_nodes = [0.0]
    for i in range(1, len(cart_nodes)):
        x_nodes.append(x_nodes[-1] + float(np.linalg.norm(cart_nodes[i] - cart_nodes[i - 1])))

    return np.asarray(x_nodes), cart_nodes, frac_nodes


def project_kpoints_to_kpath(
    k_frac_points: np.ndarray,
    points: dict[str, np.ndarray],
    path: Sequence[str],
    primitive_vectors: np.ndarray,
    tol_cart: float = 1e-6,
):
    """Project folded k-points onto a high-symmetry path."""
    k_frac_points = np.asarray(k_frac_points, dtype=float)
    dim = k_frac_points.shape[1]
    x_nodes, cart_nodes, frac_nodes = kpath_axis_from_fractional_path(points, path, primitive_vectors)

    # A non-reduced primitive basis can put first-BZ points beyond +/-1
    # in fractional coordinates. Bound translations by each segment itself.
    reciprocal_inverse = np.linalg.inv(reciprocal_vectors(primitive_vectors))
    frac_tolerance = tol_cart * np.linalg.norm(reciprocal_inverse, axis=1) + 1e-10
    projected = []

    for ik, q in enumerate(k_frac_points):
        occurrences = []
        for iseg in range(len(path) - 1):
            start, stop = frac_nodes[iseg : iseg + 2]
            lower = np.ceil(np.minimum(start, stop) - q - frac_tolerance).astype(int)
            upper = np.floor(np.maximum(start, stop) - q + frac_tolerance).astype(int)
            for shift in product(*(range(lo, hi + 1) for lo, hi in zip(lower, upper))):
                q_equiv = q + np.asarray(shift)
                q_cart = kfrac_to_cart(np.asarray([q_equiv]), primitive_vectors)[0]
                a_cart = cart_nodes[iseg]
                v = cart_nodes[iseg + 1] - a_cart
                denom = float(np.dot(v, v))
                if denom == 0.0:
                    continue
                t = float(np.dot(q_cart - a_cart, v) / denom)
                if t < -1e-10 or t > 1.0 + 1e-10:
                    continue
                closest = a_cart + t * v
                dist = float(np.linalg.norm(q_cart - closest))
                if dist <= tol_cart:
                    x = x_nodes[iseg] + t * float(np.linalg.norm(v))
                    candidate = (dist, ik, x, iseg, t, q_equiv)
                    # Adjacent segments share one endpoint; a repeated point
                    # at a different path distance (e.g. closing Gamma) stays.
                    duplicate = next(
                        (i for i, old in enumerate(occurrences) if abs(old[2] - x) <= 1e-10),
                        None,
                    )
                    if duplicate is None:
                        occurrences.append(candidate)
                    elif dist < occurrences[duplicate][0]:
                        occurrences[duplicate] = candidate
        projected.extend(occurrences)

    projected.sort(key=lambda entry: entry[2])
    indices = np.asarray([p[1] for p in projected], dtype=int)
    x = np.asarray([p[2] for p in projected], dtype=float)
    seg = np.asarray([p[3] for p in projected], dtype=int)
    t = np.asarray([p[4] for p in projected], dtype=float)
    q_equiv = np.asarray([p[5] for p in projected], dtype=float) if projected else np.empty((0, dim))

    return indices, x, seg, t, q_equiv, x_nodes
