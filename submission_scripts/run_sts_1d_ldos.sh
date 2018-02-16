#!/bin/bash -l

FOLDER=.

python3 ./sts_1d_ldos_from_npz.py \
  --npz_file "$FOLDER"/morbs_h1_dx0.2.npz \
  --output_dir "$FOLDER"/sts_output \
  --sts_plane_height 1.0 \
  --sts_de 0.005 \
  --sts_fwhm 0.01 \
  --work_function 4.36
