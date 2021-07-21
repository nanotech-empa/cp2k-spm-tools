#!/bin/bash -l

FOLDER="."

mpirun -n 2 cube_from_wfn.py \
  --cp2k_input_file "$FOLDER"/cp2k.inp \
  --basis_set_file "$FOLDER"/BASIS_MOLOPT \
  --xyz_file "$FOLDER"/geom.xyz \
  --wfn_file "$FOLDER"/PROJ-RESTART.wfn \
  --output_dir "./out/" \
\
  --dx 0.2 \
  --eval_cutoff 14.0 \
  --eval_region "G" "G" "G" "G" "G" "G" \
\
  --n_homo 5 \
  --n_lumo 5 \
  --orb_square \
\
  --charge_dens \
  --spin_dens \
\
