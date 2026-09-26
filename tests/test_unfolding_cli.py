from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import scipy.sparse as sp

from cp2k_spm_tools.cli import unfold_wfn_sparse as serial
from cp2k_spm_tools.cli import unfold_wfn_sparse_mpi as mpi
from cp2k_spm_tools.cp2k_overlap_matrix import Cp2kOverlapMatrix

ARGS = [
    "unused.wfn",
    "unused.npz",
    "unused-output.npz",
    "--xyz",
    "unused.xyz",
    "--cp2k-input",
    "unused.inp",
    "--primitive-vectors",
    "1 0 0; 0 1 0",
]


@pytest.mark.parametrize("module,writer", [(serial, "write_unfolding_npz"), (mpi, "write_unfolding_npz_mpi")])
@pytest.mark.parametrize(
    "extra,expected",
    [
        ([], []),
        (["--path", ""], []),
        (["--path", "G-X-M-G"], ["G", "X", "M", "G"]),
        (["--path", "GXA1YG"], ["G", "X", "A1", "Y", "G"]),
        (["--path", "GYHCH1XG"], ["G", "Y", "H", "C", "H1", "X", "G"]),
    ],
)
def test_cli_path_selection(monkeypatch, module, writer, extra, expected):
    capture = Mock()
    monkeypatch.setattr(module, writer, capture)
    assert module.main(ARGS + extra) == 0
    assert capture.call_args.kwargs["path_labels"] == expected


@pytest.mark.parametrize("module,writer", [(serial, "write_unfolding_npz"), (mpi, "write_unfolding_npz_mpi")])
@pytest.mark.parametrize(
    "extra",
    [
        ["--emin", "-2"],
        ["--emax", "2"],
        ["--emin", "2", "--emax", "-2"],
        ["--emin", "2", "--emax", "2"],
    ],
)
def test_cli_rejects_invalid_energy_window_before_io(monkeypatch, capsys, module, writer, extra):
    capture = Mock()
    monkeypatch.setattr(module, writer, capture)
    with pytest.raises(SystemExit) as exc:
        module.main(ARGS + extra)
    assert exc.value.code == 2
    assert "--emin" in capsys.readouterr().err
    capture.assert_not_called()


@pytest.mark.parametrize("writer", [serial.write_unfolding_npz, mpi.write_unfolding_npz_mpi])
def test_writer_rejects_one_sided_window_before_io(writer):
    with pytest.raises(ValueError, match="set together"):
        writer(
            wfn_path="unused",
            overlap_path="unused",
            xyz_path="unused",
            cp2k_input_path="unused",
            output_path="unused",
            primitive_vectors_approx=np.array([[1, 0, 0]]),
            emin=-2,
        )


@pytest.mark.parametrize("module", [serial, mpi])
def test_removed_noop_tolerance_is_rejected(module):
    with pytest.raises(SystemExit):
        module.main(ARGS + ["--tol", "0.1"])


@pytest.mark.parametrize(
    "family,rows,expected_path",
    [
        ("1d", [[1, 0, 0]], ["G", "X"]),
        ("square", [[1, 0, 0], [0, 1, 0]], ["G", "X", "M", "G"]),
        ("rectangular", [[1, 0, 0], [0, 1.7, 0]], ["G", "X", "S", "Y", "G"]),
        ("hexagonal", [[1, 0, 0], [0.5, np.sqrt(3) / 2, 0]], ["G", "K", "M", "G"]),
        ("hexagonal", [[1, 0, 0], [-0.5, np.sqrt(3) / 2, 0]], ["G", "K", "M", "G"]),
        ("centered_rectangular", [[1, 0, 0], [0.25, np.sqrt(15) / 4, 0]], ["G", "X", "A1", "Y", "G"]),
        ("oblique", [[1, 0, 0], [0.4, 1.3, 0]], ["G", "Y", "H", "C", "H1", "X", "G"]),
    ],
)
def test_serial_output_and_bloch_weights_for_all_families(tmp_path, monkeypatch, family, rows, expected_path):
    primitive = np.asarray(rows, dtype=float)
    dim = len(primitive)
    translations = np.asarray(list(product(range(6), repeat=dim)))
    fractional_k = translations / 6
    coordinates = translations @ primitive
    n = len(coordinates)
    # Independent analytical Bloch coefficients: one state at each folded q.
    coefficients = np.exp(2j * np.pi * (fractional_k @ translations.T)) / np.sqrt(n)
    wavefunctions = SimpleNamespace(
        coeffs=[coefficients],
        evals_ev=[np.arange(n, dtype=float)],
        occs=[np.ones(n)],
        ref_energy_ev=0.0,
    )
    monkeypatch.setattr(serial, "read_cp2k_wfn", lambda *a, **kw: wavefunctions)
    xyz = tmp_path / "atoms.xyz"
    xyz.write_text(str(n) + "\n\n" + "\n".join("H " + " ".join(map(str, row)) for row in coordinates) + "\n")
    inp = tmp_path / "cp2k.inp"
    supercell = np.diag([10.0, 10.0, 10.0])
    supercell[:dim] = 6 * primitive
    inp.write_text(
        "&CELL\n"
        + "\n".join(name + " " + " ".join(map(str, row)) for name, row in zip("ABC", supercell))
        + "\n&END CELL\n"
    )
    overlap_path = tmp_path / "overlap.npz"
    Cp2kOverlapMatrix(
        matrix=sp.eye(n, format="csr"),
        basis_index=np.arange(1, n + 1),
        atom_index=np.arange(1, n + 1),
        element=np.full(n, "H"),
        orbital=np.full(n, "1s"),
    ).to_npz(overlap_path)
    output = tmp_path / "bands.npz"
    kwargs = dict(
        wfn_path="mocked.wfn",
        overlap_path=overlap_path,
        xyz_path=xyz,
        cp2k_input_path=inp,
        output_path=output,
        primitive_vectors_approx=primitive,
    )
    serial.write_unfolding_npz(**kwargs)
    with np.load(output) as data:
        assert data["lattice_type"].item() == family
        assert data["path_labels"].tolist() == expected_path
        np.testing.assert_allclose(data["k_frac_folded"], fractional_k, atol=1e-12)
        np.testing.assert_allclose(data["weights_spin_0"], np.eye(n), atol=2e-12)
        np.testing.assert_allclose(data["mo_norms_spin_0"], 1, atol=2e-12)
        default_weights = data["weights_spin_0"].copy()
    # Explicit paths affect path metadata only, never the complete weights.
    serial.write_unfolding_npz(**kwargs, path_labels=["G", expected_path[1]])
    with np.load(output) as data:
        assert data["path_labels"].tolist() == ["G", expected_path[1]]
        np.testing.assert_array_equal(data["weights_spin_0"], default_weights)
