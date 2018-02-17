#!/bin/bash -l

FOLDER=.

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
echo "ORB ANALYSIS h=5, extrapolation from h=3"

python3 ./orbital_analysis_from_npz.py \
  --npz_file "$FOLDER"/morbs_h3_dx0.2.npz \
  --output_dir "$FOLDER"/sts_output \
  --sts_plane_height 5.0 \
  --nhomo 100 \
  --nlumo 1 \
  --crop_x_l $CROP_DIST_L \
  --crop_x_r $CROP_DIST_R \
  --work_function 4.36

