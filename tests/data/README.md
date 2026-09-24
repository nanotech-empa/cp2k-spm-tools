`cp2k_ch4_overlap_matrix.log` is the complete overlap-matrix output from a
real CP2K calculation of a CH4 molecule. It is retained as parser regression
data so that sparse-matrix shape, AO metadata, symmetry, thresholding, and NPZ
round trips are tested against production-format CP2K output.

`examples/overlap_to_sparse_npz/run.sh` reads this same file, so that the
example demonstrates real CP2K output rather than a hand-written excerpt. Move
or rename it and that example breaks too.
