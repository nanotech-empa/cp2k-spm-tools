#!/bin/bash -l

FOLDER=".."

mpirun -n 2 bader_bond_order.py \
  --cp2k_input_file "$FOLDER"/cp2k.inp \
  --basis_set_file "$FOLDER"/BASIS_MOLOPT \
  --xyz_file "$FOLDER"/c2h2-opt.xyz \
  --wfn_file "$FOLDER"/PROJ-RESTART.wfn \
\
  --output_file "./bond_order.txt" \
  --bader_basins_dir "./" \
\
  --dx 0.06 \
  --eval_cutoff 16.0 \
  --eval_region "n-2.0_H" "p2.0_H" "n-3.0_C" "p3.0_C" "n-3.0_C" "p3.0_C" \
#  --eval_region "G" "G" "G" "G" "G" "G"

