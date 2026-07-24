from __future__ import annotations

from typing import Sequence

import numpy as np


def plot_unfolded_kpath(
    *,
    path_k_indices: np.ndarray,
    path_x: np.ndarray,
    x_ticks: Sequence[float],
    x_tick_labels: Sequence[str],
    energies_ev: np.ndarray,
    weights: np.ndarray,
    marker_scale: float = 220.0,
    min_marker_size: float = 1e-8,
    ax=None,
):
    """Plot unfolded bands with marker size proportional to spectral weight."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    for ik, x in zip(path_k_indices, path_x):
        sizes = marker_scale * np.maximum(weights[ik], 0.0)
        mask = sizes > min_marker_size
        ax.scatter(
            np.full(np.count_nonzero(mask), x),
            energies_ev[mask],
            s=sizes[mask],
            alpha=0.7,
            color="black",
        )

    for xt in x_ticks:
        ax.axvline(xt, linewidth=0.8, alpha=0.4)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xticks(x_ticks, x_tick_labels)
    ax.set_xlim(x_ticks[0], x_ticks[-1])
    ax.set_xlabel("primitive-cell k-path")
    ax.set_ylabel("Energy - reference [eV]")
    ax.set_title("Unfolded weights on standard k-path")
    ax.figure.tight_layout()
    return ax
