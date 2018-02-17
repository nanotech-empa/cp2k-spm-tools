#!/bin/bash -l

FOLDER=.

cd $FOLDER

mkdir sts_output

# --------------------------------------
# Cropping 

CROP_DIST_L=5.0
CROP_DIST_R=5.0
CROP_DEF_L=1
CROP_DEF_R=1

if ls | grep -q L60
then
  CROP_DIST_L=254.82
  CROP_DIST_R=365.36
  CROP_DEF_L=0
  CROP_DEF_R=0
fi

if ls | grep -q L50
then
  CROP_DIST_L=190.00
  CROP_DIST_R=330.40
  CROP_DEF_L=0
  CROP_DEF_R=0
fi

# --------------------------------------
echo "FTSTS h=5, extrapolation from h=3"

python3 ./ftsts_1d_ldos_from_npz.py \
  --npz_file "$FOLDER"/morbs_h3_dx0.2.npz \
  --output_dir "$FOLDER"/sts_output \
  --sts_plane_height 5.0 \
  --sts_de 0.005 \
  --sts_fwhm 0.01 0.05 \
  --work_function 4.36 \
  --crop_dist_l $CROP_DIST_L \
  --crop_dist_r $CROP_DIST_R \
  --crop_defect_l $CROP_DEF_L \
  --crop_defect_r $CROP_DEF_R \
  --padding_x 300.0 \
  --emin -0.7 \
  --emax 0.7 \
  --gammas 1.0 \
  --vmax_coefs 1.0 0.5
