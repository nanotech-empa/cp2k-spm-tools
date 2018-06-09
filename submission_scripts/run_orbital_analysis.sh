#!/bin/bash -l

FOLDER=..

cd $FOLDER

mkdir orb_output

python3 ./orbital_analysis_from_npz.py \
  --npz_file "$FOLDER"/morbs_dx0.2.npz \
  --output_dir "$FOLDER"/orb_output \
  --sts_plane_height 1.0 \
  --emin -0.5 \
  --emax 0.5 \
  --lat_param 4.26 \
  --crop_x_l 0.0 \
  --crop_x_r -1.0 \
  --work_function 4.36

