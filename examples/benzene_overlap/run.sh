#!/bin/bash -l

# for demo, take slab and molecule to be the same calculation
SLAB_FOLDER=../data/benzene_cp2k_scf
MOL_FOLDER=../data/benzene_cp2k_scf
BASIS_PATH="../data/BASIS_MOLOPT"

mkdir out

mpirun -n 2  cp2k-overlap-from-wfns \
  --cp2k_input_file1 "$SLAB_FOLDER"/cp2k.inp \
  --basis_set_file1 $BASIS_PATH \
  --xyz_file1 "$SLAB_FOLDER"/geom.xyz \
  --wfn_file1 "$SLAB_FOLDER"/PROJ-RESTART.wfn \
  --emin1 -8.0 \
  --emax1  8.0 \
  --cp2k_input_file2 "$MOL_FOLDER"/cp2k.inp \
  --basis_set_file2 $BASIS_PATH \
  --xyz_file2 "$MOL_FOLDER"/geom.xyz \
  --wfn_file2 "$MOL_FOLDER"/PROJ-RESTART.wfn \
  --nhomo2 2 \
  --nlumo2 2 \
  --output_file "./out/overlap.npz" \
  --eval_region "n-2.0_C" "p2.0_C" "n-2.0_C" "p2.0_C" "n-2.0_C" "p2.0_C" \
  --dx 0.2 \
  --eval_cutoff 14.0 \

