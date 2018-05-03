#!/bin/bash -l

FOLDER=..

mkdir $FOLDER/sts_output

python3 ./ftsts_1d_ldos_from_npz.py \
  --npz_file "$FOLDER"/morbs_dx0.2.npz \
  --output_dir "$FOLDER"/sts_output \
  --sts_plane_height 5.0 \
  --sts_de 0.01 \
  --sts_fwhm 0.02 0.1 \
  --hartree_file "$FOLDER"/PROJ-HART-v_hartree-1_0.cube \
  --crop_dist_l 0 \
  --crop_dist_r -1 \
  --crop_defect_l 0 \
  --crop_defect_r 0 \
  --padding_x 600.0 \
  --emin -0.5 \
  --emax 0.5 \
  --lat_param 12.935 \
  --gammas 1.0 \
  --vmax_coefs 1.0 0.5
