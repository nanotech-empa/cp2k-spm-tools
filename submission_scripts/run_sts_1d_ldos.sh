#!/bin/bash -l

FOLDER=.

python3 ./sts_1d_ldos_from_npz.py \
  --npz_file "$FOLDER"/morbs_h1.npz \
  --output_dir "$FOLDER"/sts_output \
  --sts_plane_height 1.0 \
  --sts_de 0.005 \
  --sts_fwhm 0.01 \
  --hartree_file "$FOLDER"/V_HARTREE-v_hartree-1_0.cube 
