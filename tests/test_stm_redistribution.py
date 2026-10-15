"""Check p-tip equivalence and bounded workspace during spatial redistribution."""

import os
import shutil
import subprocess
import sys
import tracemalloc
from types import SimpleNamespace

import numpy as np
import pytest

from cp2k_spm_tools.cp2k_stm_sts import STM


class SerialComm:
    def allgather(self, values):
        return [values]

    def Gatherv(self, sendbuf, recvbuf, root):
        assert root == 0
        np.copyto(recvbuf[0], sendbuf)


def make_stm(grids, vectors=None, comm=None):
    comm = SerialComm() if comm is None else comm
    cgo = SimpleNamespace(
        nspin=len(grids),
        mpi_rank=comm.Get_rank() if hasattr(comm, "Get_rank") else 0,
        mpi_size=comm.Get_size() if hasattr(comm, "Get_size") else 1,
        eval_cell_n=np.array(grids[0].shape[1:]),
        dv=np.array([0.3, 0.4, 0.5]) if vectors is None else np.linalg.norm(vectors, axis=1),
        origin=np.array([0.1, 0.2, 0.3]),
        global_morb_energies=None,
        morb_energies=[np.arange(len(g), dtype=float) for g in grids],
        morb_grids=grids,
        dtype=grids[0].dtype,
        ase_atoms=SimpleNamespace(positions=np.zeros((1, 3))),
    )
    if vectors is not None:
        cgo.dv_vectors = vectors
    stm = STM(comm, cgo, [0.0, 0.3, 1.0])
    stm.gather_global_energies()
    return stm


def global_p_tip(stm, grids):
    """Reference the original full-grid finite differences and storage dtype."""
    return STM.in_plane_gradient_squared(
        np.gradient(grids, axis=1),
        np.gradient(grids, axis=2),
        stm.dv,
        getattr(stm.cgo, "dv_vectors", None),
    ).astype(stm.cgo.dtype)


VECTORS = [None, np.diag([0.3, 0.4, 0.5]), np.array([[0.3, 0.1, 0], [0.07, 0.4, 0], [0, 0, 0.5]])]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("vectors", VECTORS, ids=["legacy", "orthogonal", "skewed"])
def test_serial_redistribution_preserves_orbitals_and_mixed_tip_signals(dtype, vectors):
    rng = np.random.default_rng(12)
    grids = [rng.normal(size=(n, 11, 7, 5)).astype(dtype) for n in (3, 2)]
    originals = [g.copy() for g in grids]
    stm = make_stm(grids, vectors)
    reference = [global_p_tip(stm, g) for g in grids]
    stm.divide_by_space()
    for spin, (grid, p_tip) in enumerate(zip(grids, reference)):
        np.testing.assert_array_equal(grid, originals[spin])
        np.testing.assert_array_equal(stm.local_orbitals[spin], grid)
        np.testing.assert_array_equal(stm.local_orbital_ptip[spin], p_tip)
        assert stm.local_orbital_ptip[spin].dtype == dtype
        for orbital in range(len(grid)):
            for ratio in (0.0, 0.3, 1.0):
                expected = (1 - ratio) * grid[orbital] ** 2 + ratio * p_tip[orbital]
                np.testing.assert_array_equal(stm.s_p_type_signal(spin, orbital, ratio), expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("vectors", VECTORS, ids=["legacy", "orthogonal", "skewed"])
@pytest.mark.parametrize("start,stop", [(0, 4), (4, 7), (7, 11), (0, 1), (1, 2), (10, 11), (11, 11)])
def test_local_gradients_match_global_reference_at_slab_and_physical_edges(dtype, vectors, start, stop):
    grids = np.random.default_rng(23).normal(size=(4, 11, 7, 5)).astype(dtype)
    stm = make_stm([grids], vectors)
    reference = global_p_tip(stm, grids)[:, start:stop]
    halo_start, halo_stop = (max(0, start - 1), min(11, stop + 1)) if start < stop else (start, stop)
    local = stm._local_p_tip_contribution(grids[:, halo_start:halo_stop], slice(start - halo_start, stop - halo_start))
    np.testing.assert_array_equal(local, reference)


@pytest.mark.parametrize("shape", [(1, 7), (7, 1)])
def test_p_tip_rejects_in_plane_axes_too_short_for_finite_differences(shape):
    stm = make_stm([np.ones((2, *shape, 5))])
    with pytest.raises(ValueError, match="at least two grid points"):
        stm.divide_by_space()
    stm.ptip_enabled = False
    stm.divide_by_space()
    np.testing.assert_array_equal(stm.local_orbitals[0], stm.cgo.morb_grids[0])
    assert stm.local_orbital_ptip == []


def test_p_tip_workspace_does_not_require_full_gradient_stacks():
    # Input grids pre-exist. Allow four stacks for outputs and workspace;
    # the former implementation needs substantially more for its gradients.
    grids = np.random.default_rng(5).normal(size=(24, 48, 40, 16))
    stm = make_stm([grids], VECTORS[-1])
    if tracemalloc.is_tracing():
        pytest.skip("Requires a separate tracemalloc measurement.")
    tracemalloc.start()
    try:
        stm.divide_by_space()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * grids.nbytes, f"Redistribution allocated {peak} bytes for a {grids.nbytes}-byte input"


MPI_WORKER = r"""
import sys
import numpy as np
from mpi4py import MPI
from test_stm_redistribution import make_stm, global_p_tip, VECTORS

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
dtype, nx, cell = np.dtype(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
vectors = VECTORS[-1] if cell == "skewed" else VECTORS[1]
full, owned = [], []
for spin in range(2):
    counts = [(r + spin) % 3 for r in range(size)]
    grid = np.random.default_rng(42 + spin).normal(size=(sum(counts), nx, 5, 4)).astype(dtype)
    first = sum(counts[:rank])
    full.append(grid)
    owned.append(grid[first:first + counts[rank]].copy())
stm = make_stm(owned, vectors, comm)
stm.divide_by_space()
start, stop = stm.x_ind_per_rank(rank)
for spin, grid in enumerate(full):
    p_tip = global_p_tip(stm, grid)[:, start:stop]
    np.testing.assert_array_equal(stm.local_orbitals[spin], grid[:, start:stop])
    np.testing.assert_array_equal(stm.local_orbital_ptip[spin], p_tip)
    for orbital in range(len(grid)):
        for ratio in (0.0, 0.3, 1.0):
            reference = (1 - ratio) * grid[orbital, start:stop] ** 2 + ratio * p_tip[orbital]
            np.testing.assert_array_equal(stm.s_p_type_signal(spin, orbital, ratio), reference)
np.testing.assert_array_equal(stm.local_origin, stm.origin + start * vectors[0])
"""


@pytest.mark.parametrize("ranks,nx", [(2, 7), (4, 3)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("cell", ["orthogonal", "skewed"])
def test_mpi_redistribution_preserves_boundaries_and_empty_partitions(ranks, nx, dtype, cell):
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if launcher is None:
        pytest.skip("MPI launcher unavailable")
    pytest.importorskip("mpi4py")
    environment = dict(os.environ)
    environment.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    environment.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    environment.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    # The worker imports the small test/reference helpers, not an installed app.
    test_dir = os.path.dirname(__file__)
    environment["PYTHONPATH"] = test_dir + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run(
        [launcher, "-n", str(ranks), sys.executable, "-c", MPI_WORKER, dtype, str(nx), cell],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
