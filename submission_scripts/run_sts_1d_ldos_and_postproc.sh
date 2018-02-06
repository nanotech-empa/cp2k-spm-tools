#!/bin/bash -l

FOLDER=.

cd $FOLDER

mkdir sts_output

python3 ./sts_1d_ldos_from_npz.py \
  --npz_file "$FOLDER"/morbs_h1.npz \
  --output_dir "$FOLDER"/sts_output \
  --sts_plane_height 1.0 \
  --sts_de 0.005 \
  --sts_fwhm 0.01 \
  --hartree_file "$FOLDER"/V_HARTREE-v_hartree-1_0.cube

# the oldest .txt in the sts_output folder 
LDOS_FILE=$(ls -tr sts_output | grep .txt | head -n 1)

python3 ./ldos_postproc.py \
  --ldos_file "$FOLDER"/sts_output/$LDOS_FILE \
  --output_dir "$FOLDER"/sts_output/ \
  --crop_dist_l 5.0 \
  --crop_dist_r 5.0 \
  --crop_defect_l 1 \
  --crop_defect_r 1 \
  --padding_x 300.0 \
  --emin -0.7 \
  --emax 0.7 \
  --gammas 1.0 0.5 \
  --vmax_coefs 1.0 0.5


