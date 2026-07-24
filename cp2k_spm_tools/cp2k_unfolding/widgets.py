from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import (
    guess_dimensionality_from_cell_and_coords,
    guess_primitive_vectors_from_geometry,
    matrix_to_text,
    parse_matrix_text,
)
from .io import parse_cp2k_cell_vectors, read_xyz_coordinates
from .kpath import guess_2d_lattice_type


@dataclass
class PrimitiveCellWidgets:
    primitive_vectors_widget: object
    supercell_vectors_widget: object
    lattice_type_widget: object
    symbols: list[str]
    coords: np.ndarray
    dim: int
    primitive_guess: np.ndarray
    supercell_guess: np.ndarray
    lattice_type_guess: str


def create_primitive_cell_widgets(
    *,
    xyz_file: str | Path,
    cp2k_input_file: str | Path,
    requested_dim: int | None = 2,
    default_lattice_type: str = "auto",
    tol: float = 2e-3,
    min_quality: float = 0.80,
) -> PrimitiveCellWidgets:
    """Read geometry, guess primitive vectors, and create editable widgets."""
    import ipywidgets as widgets

    symbols, coords = read_xyz_coordinates(xyz_file)
    full_cell_vectors = parse_cp2k_cell_vectors(cp2k_input_file, dim=3)

    dim = requested_dim
    if dim is None:
        dim = guess_dimensionality_from_cell_and_coords(full_cell_vectors, coords)

    supercell_guess = full_cell_vectors[:dim]
    primitive_guess = guess_primitive_vectors_from_geometry(
        symbols,
        coords,
        supercell_guess,
        dim=dim,
        tol=tol,
        min_quality=min_quality,
    )

    lattice_type_guess = default_lattice_type
    if lattice_type_guess == "auto" and dim == 2:
        lattice_type_guess = guess_2d_lattice_type(primitive_guess)

    primitive_vectors_widget = widgets.Textarea(
        value=matrix_to_text(primitive_guess),
        description="primitive",
        layout=widgets.Layout(width="720px", height="90px"),
    )
    supercell_vectors_widget = widgets.Textarea(
        value=matrix_to_text(supercell_guess),
        description="supercell",
        layout=widgets.Layout(width="720px", height="90px"),
    )
    lattice_type_widget = widgets.Dropdown(
        options=["auto", "hexagonal", "square", "rectangular", "oblique", "1d"],
        value=(
            lattice_type_guess
            if lattice_type_guess in ["hexagonal", "square", "rectangular", "oblique", "1d"]
            else "auto"
        ),
        description="k-path",
    )

    return PrimitiveCellWidgets(
        primitive_vectors_widget=primitive_vectors_widget,
        supercell_vectors_widget=supercell_vectors_widget,
        lattice_type_widget=lattice_type_widget,
        symbols=symbols,
        coords=coords,
        dim=int(dim),
        primitive_guess=primitive_guess,
        supercell_guess=supercell_guess,
        lattice_type_guess=lattice_type_guess,
    )


def read_primitive_cell_widgets(widget_state: PrimitiveCellWidgets):
    """Return primitive vectors, supercell vectors, and final lattice type."""
    primitive_vectors = parse_matrix_text(
        widget_state.primitive_vectors_widget.value, expected_dim=widget_state.dim
    )
    supercell_vectors = parse_matrix_text(
        widget_state.supercell_vectors_widget.value, expected_dim=widget_state.dim
    )

    lattice_type = widget_state.lattice_type_widget.value
    if lattice_type == "auto":
        lattice_type = guess_2d_lattice_type(primitive_vectors) if widget_state.dim == 2 else "1d"

    return primitive_vectors, supercell_vectors, lattice_type
