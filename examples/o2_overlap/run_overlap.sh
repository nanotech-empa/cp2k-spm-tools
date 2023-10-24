#!/bin/bash -l

SLAB_FOLDER=../o2_cp2k_scf
MOL_FOLDER=../o2_cp2k_scf

mkdir out

mpirun -n 2  python3 ../../overlap_from_wfns.py \
  --cp2k_input_file1 "$SLAB_FOLDER"/inp \
  --basis_set_file1 "$SLAB_FOLDER"/BASIS_MOLOPT \
  --xyz_file1 "$SLAB_FOLDER"/aiida.coords.xyz \
  --wfn_file1 "$SLAB_FOLDER"/PROJ-RESTART.wfn \
  --emin1 -18.0 \
  --emax1  8.0 \
  --cp2k_input_file2 "$MOL_FOLDER"/inp \
  --basis_set_file2 "$MOL_FOLDER"/BASIS_MOLOPT \
  --xyz_file2 "$MOL_FOLDER"/aiida.coords.xyz \
  --wfn_file2 "$MOL_FOLDER"/PROJ-RESTART.wfn \
  --nhomo2 2 \
  --nlumo2 2 \
  --output_file "./out/overlap.npz" \
  --eval_region "n-2.0_O" "p2.0_O" "n-2.0_O" "p2.0_O" "n-2.0_O" "p2.0_O" \
  --dx 0.2 \
  --eval_cutoff 14.0 \

