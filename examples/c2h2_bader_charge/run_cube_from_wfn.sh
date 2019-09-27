#!/bin/bash -l

FOLDER="../c2h2_cp2k_scf"

mkdir out

mpirun -n 2 python3 ../../cube_from_wfn.py \
  --cp2k_input_file "$FOLDER"/cp2k.inp \
  --basis_set_file "$FOLDER"/BASIS_MOLOPT \
  --xyz_file "$FOLDER"/geom.xyz \
  --wfn_file "$FOLDER"/PROJ-RESTART.wfn \
  --output_dir "./out/" \
\
  --dx 0.08 \
  --eval_cutoff 14.0 \
  --eval_region "G" "G" "G" "G" "G" "G" \
\
  --charge_dens \
  --charge_dens_artif_core \
\
