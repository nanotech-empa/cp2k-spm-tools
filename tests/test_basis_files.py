"""Read the same orbital basis from single or distributed parameter files."""

from types import SimpleNamespace

import numpy as np
import pytest

from cp2k_spm_tools.cp2k_grid_orbitals import Cp2kGridOrbitals


def definition(element, name, exponent):
    return f"{element} {name} alias-{name} third-{name}\n1\n1 0 0 1 1\n{exponent} 1.0\n"


@pytest.fixture
def basis_files(tmp_path):
    hydrogen = definition("H", "orb-H", 0.7)
    carbon = definition("C", "orb-C", 1.2)
    files = {"H": hydrogen, "C": carbon, "both": hydrogen + carbon, "aux": definition("H", "admm-H", 2.0)}
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    return tmp_path


def read_basis(files, kinds=None):
    reader = Cp2kGridOrbitals(0, 1)
    reader.kind_elem_basis = kinds or {"H": ("H", "orb-H"), "H1": ("H", "orb-H"), "C": ("C", "orb-C")}
    reader.read_basis_functions(files)
    return reader.basis_sets


def assert_same_basis(left, right):
    assert left.keys() == right.keys()
    for kind in left:
        assert len(left[kind]) == len(right[kind])
        for left_set, right_set in zip(left[kind], right[kind]):
            assert len(left_set) == len(right_set)
            for (left_l, left_exp, left_coef), (right_l, right_exp, right_coef) in zip(left_set, right_set):
                assert left_l == right_l
                np.testing.assert_array_equal(left_exp, right_exp)
                np.testing.assert_array_equal(left_coef, right_coef)


@pytest.mark.parametrize("form", ["string", "path", "list", "tuple"])
def test_single_file_input_forms_preserve_coefficients(basis_files, form):
    path = basis_files / "both"
    value = {"string": str(path), "path": path, "list": [path], "tuple": (path,)}[form]
    actual = read_basis(value)
    assert actual["H"][0][0][0] == 0
    np.testing.assert_array_equal(actual["H"][0][0][1], [0.7])
    np.testing.assert_allclose(actual["H"][0][0][2], [0.7**0.75], rtol=1e-15)
    np.testing.assert_allclose(actual["C"][0][0][2], [1.2**0.75], rtol=1e-15)
    assert actual["H"][0][0][2] is not actual["H1"][0][0][2]


def test_split_and_combined_files_give_identical_coefficients(basis_files):
    combined = read_basis(basis_files / "both")
    split = read_basis([basis_files / "H", basis_files / "C"])
    assert_same_basis(combined, split)


def test_auxiliary_file_order_does_not_replace_orbital_bases(basis_files):
    reference = read_basis(basis_files / "both")
    actual = read_basis([basis_files / "aux", basis_files / "C", basis_files / "H"])
    assert_same_basis(reference, actual)


def test_first_matching_file_has_cp2k_precedence(basis_files):
    alternative = basis_files / "alternative"
    alternative.write_text(definition("H", "orb-H", 3.0))
    first = read_basis([alternative, basis_files / "both"])
    second = read_basis([basis_files / "both", alternative])
    np.testing.assert_array_equal(first["H"][0][0][1], [3.0])
    np.testing.assert_array_equal(second["H"][0][0][1], [0.7])


def test_first_definition_in_a_file_has_cp2k_precedence(basis_files):
    path = basis_files / "both"
    path.write_text(path.read_text() + definition("H", "orb-H", 3.0))
    np.testing.assert_array_equal(read_basis(path)["H"][0][0][1], [0.7])


def test_all_header_aliases_are_supported(basis_files):
    actual = read_basis(basis_files / "both", {"custom": ("H", "third-orb-H")})
    np.testing.assert_array_equal(actual["custom"][0][0][1], [0.7])


def test_missing_orbital_definition_names_the_kind_and_files(basis_files):
    with pytest.raises(ValueError, match="C.*orb-C") as error:
        read_basis([basis_files / "H", basis_files / "aux"])
    assert str(basis_files / "H") in str(error.value)


@pytest.mark.parametrize("first,second", [(["one"], ["two"]), (["a", "b"], ["c", "d"])])
def test_overlap_cli_accepts_one_or_several_files(monkeypatch, first, second):
    from cp2k_spm_tools.cli import overlap_from_wfns

    class StopAfterBasis(Exception):
        pass

    class Comm:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def bcast(self, value, root=0):
            return value

    parsed = {}
    original_parse = overlap_from_wfns.argparse.ArgumentParser.parse_args

    def parse(parser, *args, **kwargs):
        namespace = original_parse(parser, *args, **kwargs)
        parsed.update(vars(namespace))
        return namespace

    class Reader:
        def __init__(self, *args, **kwargs):
            pass

        def read_cp2k_input(self, path):
            pass

        def read_xyz(self, path):
            pass

        def read_basis_functions(self, paths):
            assert paths == second
            raise StopAfterBasis

    argv = [
        "overlap",
        "--basis_set_file1",
        *first,
        "--basis_set_file2",
        *second,
        "--cp2k_input_file1",
        "input1",
        "--cp2k_input_file2",
        "input2",
        "--xyz_file1",
        "xyz1",
        "--xyz_file2",
        "xyz2",
        "--wfn_file1",
        "wfn1",
        "--wfn_file2",
        "wfn2",
        "--emin1",
        "-5",
        "--emax1",
        "2",
        "--nhomo2",
        "1",
        "--nlumo2",
        "1",
        "--output_file",
        "out.npz",
        "--eval_region",
        "0",
        "1",
        "0",
        "1",
        "0",
        "1",
        "--dx",
        "0.2",
    ]
    monkeypatch.setattr(overlap_from_wfns, "MPI", SimpleNamespace(COMM_WORLD=Comm()))
    monkeypatch.setattr(overlap_from_wfns.argparse.ArgumentParser, "parse_args", parse)
    monkeypatch.setattr(overlap_from_wfns.cgo, "Cp2kGridOrbitals", Reader)
    monkeypatch.setattr(overlap_from_wfns.sys, "argv", argv)
    with pytest.raises(StopAfterBasis):
        overlap_from_wfns.main()
    assert parsed["basis_set_file1"] == first
    assert parsed["basis_set_file2"] == second
