from __future__ import annotations

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


def guess_2d_lattice_type(
    primitive_vectors: np.ndarray, rtol: float = 1e-3, angle_tol: float = 1e-2
) -> str:
    """Guess a simple 2D Bravais-lattice family."""
    la, lb, angle = lattice_lengths_angles_2d(primitive_vectors)
    equal_lengths = abs(la - lb) <= rtol * max(la, lb)
    right_angle = abs(angle - 90.0) <= angle_tol
    hex_angle = min(abs(angle - 60.0), abs(angle - 120.0)) <= angle_tol

    if equal_lengths and hex_angle:
        return "hexagonal"
    if equal_lengths and right_angle:
        return "square"
    if right_angle:
        return "rectangular"
    return "oblique"


def standard_kpath(
    dim: int, lattice_type: str | None = None, primitive_vectors: np.ndarray | None = None
):
    """Return high-symmetry points and a standard path in fractional reciprocal coordinates."""
    if dim == 1:
        points = {
            "G": np.array([0.0]),
            "X": np.array([0.5]),
        }
        return points, ["G", "X"]

    if dim != 2:
        raise NotImplementedError("Standard paths are currently implemented only for 1D and 2D.")

    if lattice_type is None:
        if primitive_vectors is None:
            raise ValueError("primitive_vectors are needed when lattice_type is not given")
        lattice_type = guess_2d_lattice_type(primitive_vectors)

    lattice_type = lattice_type.lower()

    if lattice_type in {"hex", "hexagonal", "graphene"}:
        points = {
            "G": np.array([0.0, 0.0]),
            "K": np.array([2.0 / 3.0, 1.0 / 3.0]),
            "M": np.array([0.5, 0.0]),
        }
        path = ["G", "K", "M", "G"]
    elif lattice_type == "square":
        points = {
            "G": np.array([0.0, 0.0]),
            "X": np.array([0.5, 0.0]),
            "M": np.array([0.5, 0.5]),
        }
        path = ["G", "X", "M", "G"]
    elif lattice_type in {"rectangular", "oblique"}:
        points = {
            "G": np.array([0.0, 0.0]),
            "X": np.array([0.5, 0.0]),
            "S": np.array([0.5, 0.5]),
            "Y": np.array([0.0, 0.5]),
        }
        path = ["G", "X", "S", "Y", "G"]
    else:
        raise ValueError(f"Unknown 2D lattice_type: {lattice_type!r}")

    return points, path


def kpath_axis_from_fractional_path(
    points: dict[str, np.ndarray], path: Sequence[str], primitive_vectors: np.ndarray
):
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
    x_nodes, cart_nodes, _ = kpath_axis_from_fractional_path(points, path, primitive_vectors)

    integer_shifts = list(np.ndindex(*([3] * dim)))
    integer_shifts = [np.asarray(s, dtype=float) - 1.0 for s in integer_shifts]

    projected = []

    for ik, q in enumerate(k_frac_points):
        best = None

        for shift in integer_shifts:
            q_equiv = q + shift
            q_cart = kfrac_to_cart(np.asarray([q_equiv]), primitive_vectors)[0]

            for iseg in range(len(path) - 1):
                a_cart = cart_nodes[iseg]
                b_cart = cart_nodes[iseg + 1]
                v = b_cart - a_cart
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
                    if best is None or candidate[0] < best[0]:
                        best = candidate

        if best is not None:
            projected.append(best)

    indices = np.asarray([p[1] for p in projected], dtype=int)
    x = np.asarray([p[2] for p in projected], dtype=float)
    seg = np.asarray([p[3] for p in projected], dtype=int)
    t = np.asarray([p[4] for p in projected], dtype=float)
    q_equiv = np.asarray([p[5] for p in projected], dtype=float) if projected else np.empty((0, dim))

    return indices, x, seg, t, q_equiv, x_nodes
