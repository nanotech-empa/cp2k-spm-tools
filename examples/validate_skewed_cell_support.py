#!/usr/bin/env python
"""Validate skewed-cell support with bundled benzene CP2K example data.

This is a lightweight regression script for the WFN gridding path. It is meant
for manual validation before running expensive real CP2K/SPM workflows.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cp2k_spm_tools.cp2k_grid_orbitals import Cp2kGridOrbitals  # noqa: E402
from cp2k_spm_tools.cube import Cube  # noqa: E402

DATA_DIR = REPO_ROOT / "examples" / "data"
BENZENE_DIR = DATA_DIR / "benzene_cp2k_scf"
CP2K_INPUT = BENZENE_DIR / "cp2k.inp"
XYZ_FILE = BENZENE_DIR / "geom.xyz"
WFN_FILE = BENZENE_DIR / "PROJ-RESTART.wfn"
BASIS_FILE = DATA_DIR / "BASIS_MOLOPT"

ORTHOGONAL_CELL_BLOCK = """    &CELL
      ABC 14.0 14.0 14.0
    &END CELL
"""
SKEWED_CELL_BLOCK = """    &CELL
      A [angstrom] 14.0 0.0 0.0
      B [angstrom] 2.0 13.856406460551018 0.0
      C [angstrom] 0.0 0.0 14.0
    &END CELL
"""


def make_skewed_input(tmpdir: Path) -> Path:
    text = CP2K_INPUT.read_text()
    if ORTHOGONAL_CELL_BLOCK not in text:
        raise RuntimeError("Could not find the expected benzene CELL block.")
    skewed_input = tmpdir / "benzene_skewed_cell.inp"
    skewed_input.write_text(text.replace(ORTHOGONAL_CELL_BLOCK, SKEWED_CELL_BLOCK, 1))
    return skewed_input


def load_benzene_case(cp2k_input: Path, dx: float, centered_positions: np.ndarray | None = None) -> Cp2kGridOrbitals:
    cgo = Cp2kGridOrbitals(single_precision=False)
    cgo.read_cp2k_input(cp2k_input)
    cgo.read_xyz(XYZ_FILE)
    if centered_positions is None:
        cgo.center_atoms_to_cell()
    else:
        cgo.ase_atoms.positions[:] = centered_positions
    cgo.read_basis_functions(BASIS_FILE)
    cgo.load_restart_wfn_file(WFN_FILE, n_occ=1, n_virt=1)
    cgo.calc_morbs_in_region(dx, pbc=(True, True, True), eval_cutoff=14.0, print_info=False)
    return cgo


def check_orthogonal_cube_generation(tmpdir: Path) -> None:
    cgo = load_benzene_case(CP2K_INPUT, dx=0.8)
    cube_path = tmpdir / "benzene_homo.cube"
    cgo.write_cube(cube_path, orbital_nr=0)

    cube = Cube()
    cube.read_cube_file(cube_path)

    assert cube.data.shape == (17, 17, 17), cube.data.shape
    np.testing.assert_allclose(cube.cell, np.diag(np.diag(cube.cell)))
    assert np.max(np.abs(cube.data)) > 1e-2
    print("orthogonal cube generation: ok")


def check_reciprocal_metric_reduces_to_orthogonal_formula() -> None:
    cgo = Cp2kGridOrbitals()
    cgo.cell_vectors = np.diag([4.0, 6.0, 10.0])
    cgo.cell = np.linalg.norm(cgo.cell_vectors, axis=1)
    cgo.dv = cgo.cell / np.array([8, 12, 10])
    cgo.eval_cell_n = np.array([8, 12, 10])
    cgo._update_grid_vectors_from_lengths()

    g2_grid = cgo._surface_reciprocal_grids((8, 12))
    kx_arr = 2 * np.pi * np.fft.fftfreq(8, cgo.dv[0])
    ky_arr = 2 * np.pi * np.fft.rfftfreq(12, cgo.dv[1])
    kx_grid, ky_grid = np.meshgrid(kx_arr, ky_arr, indexing="ij")
    np.testing.assert_allclose(g2_grid, kx_grid**2 + ky_grid**2)
    print("orthogonal reciprocal metric: ok")


def skewed_grid_cartesian_points(cgo: Cp2kGridOrbitals) -> np.ndarray:
    ix, iy, iz = np.meshgrid(
        np.arange(cgo.eval_cell_n[0]),
        np.arange(cgo.eval_cell_n[1]),
        np.arange(cgo.eval_cell_n[2]),
        indexing="ij",
    )
    points = (
        cgo.origin
        + ix[..., None] * cgo.dv_vectors[0]
        + iy[..., None] * cgo.dv_vectors[1]
        + iz[..., None] * cgo.dv_vectors[2]
    )
    return points.reshape(-1, 3)


def check_skewed_grid_against_orthogonal_reference(skewed_input: Path) -> None:
    orth = load_benzene_case(CP2K_INPUT, dx=0.35)
    skewed = load_benzene_case(skewed_input, dx=0.35, centered_positions=orth.ase_atoms.positions.copy())

    axes = [np.arange(n) * dv + origin for n, dv, origin in zip(orth.eval_cell_n, orth.dv, orth.origin)]
    skewed_points = skewed_grid_cartesian_points(skewed)

    overlap = np.ones(len(skewed_points), dtype=bool)
    for axis, grid_axis in enumerate(axes):
        overlap &= (skewed_points[:, axis] >= grid_axis[0]) & (skewed_points[:, axis] <= grid_axis[-1])

    overlap_fraction = float(overlap.mean())
    assert overlap_fraction > 0.9, overlap_fraction

    for orbital_index in range(skewed.morb_grids[0].shape[0]):
        interp = RegularGridInterpolator(axes, orth.morb_grids[0][orbital_index], bounds_error=False, fill_value=np.nan)
        reference = interp(skewed_points[overlap])
        candidate = skewed.morb_grids[0][orbital_index].reshape(-1)[overlap]
        abs_diff = np.abs(candidate - reference)
        ref_abs = np.abs(reference)
        important = ref_abs > max(1e-5, 1e-3 * np.nanmax(ref_abs))

        p95 = np.nanpercentile(abs_diff, 95)
        p99 = np.nanpercentile(abs_diff, 99)
        rms_important = np.sqrt(np.nanmean(abs_diff[important] ** 2))

        assert p95 < 2e-4, (orbital_index, p95)
        assert p99 < 2e-3, (orbital_index, p99)
        assert rms_important < 3e-3, (orbital_index, rms_important)

        print(
            f"skewed orbital {orbital_index}: ok "
            f"(overlap={overlap_fraction:.3f}, p95={p95:.3e}, p99={p99:.3e}, rms={rms_important:.3e})"
        )


def check_skewed_cube_cell_vectors(tmpdir: Path, skewed_input: Path) -> None:
    cgo = load_benzene_case(skewed_input, dx=0.8)
    cube_path = tmpdir / "benzene_skewed_homo.cube"
    cgo.write_cube(cube_path, orbital_nr=0)

    cube = Cube()
    cube.read_cube_file(cube_path, read_data=False)
    np.testing.assert_allclose(cube.cell, cgo.eval_cell_vectors, atol=5e-5)
    assert abs(cube.cell[1, 0]) > 1e-6
    print("skewed cube cell vectors: ok")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        skewed_input = make_skewed_input(tmpdir)
        check_orthogonal_cube_generation(tmpdir)
        check_reciprocal_metric_reduces_to_orthogonal_formula()
        check_skewed_grid_against_orthogonal_reference(skewed_input)
        check_skewed_cube_cell_vectors(tmpdir, skewed_input)
    print("skewed-cell validation passed")


if __name__ == "__main__":
    main()
