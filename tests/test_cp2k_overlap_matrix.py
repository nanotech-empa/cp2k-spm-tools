from cp2k_spm_tools.cp2k_overlap_matrix import Cp2kOverlapMatrix


def test_read_ascii_csr_populates_symmetric_sparse_mat(tmp_path):
    csr_file = tmp_path / "overlap.csr"
    csr_file.write_text("1 1 1.0\n1 2 0.25\n2 2 1.0\n")

    overlap_matrix = Cp2kOverlapMatrix()
    overlap_matrix.read_ascii_csr(str(csr_file), n_basis_f=2)

    assert overlap_matrix.sparse_mat is not None
    assert overlap_matrix.sparse_mat.shape == (2, 2)

    dense = overlap_matrix.sparse_mat.toarray()
    assert dense[0, 1] == dense[1, 0] == 0.25
    assert dense[0, 0] == 1.0
