#!/bin/bash -l

FOLDER=.

cd $FOLDER

rm -rf sts_output
mkdir sts_output

# --------------------------------------
# Cropping in SPACE

CROP_DIST_L=5.0
CROP_DIST_R=5.0
CROP_DEF_L=1
CROP_DEF_R=1

if ls | grep -q cnt120-L60
then
  CROP_DIST_L=256.95
  CROP_DIST_R=363.23
  CROP_DEF_L=0
  CROP_DEF_R=0
fi

if ls | grep -q cnt120-L50
then
  CROP_DIST_L=192.13
  CROP_DIST_R=328.27
  CROP_DEF_L=0
  CROP_DEF_R=0
fi

if ls | grep -q cnt120-L100-15
then
  CROP_DIST_L=443.13
  CROP_DIST_R=579.27
  CROP_DEF_L=0
  CROP_DEF_R=0
fi

if ls | grep -q cnt120-L12
then
  CROP_DIST_L=14.26
  CROP_DIST_R=120.54
  CROP_DEF_L=0
  CROP_DEF_R=0
fi

if ls | grep -q cnt120-L2-
then
  CROP_DIST_L=14.80
  CROP_DIST_R=27.40
  CROP_DEF_L=0
  CROP_DEF_R=0
fi

if ls | grep -q cnt120-L15
then
  CROP_DIST_L=18.00
  CROP_DIST_R=154.00
  CROP_DEF_L=0
  CROP_DEF_R=0
fi

if ls | grep -q cnt1212-L50
then
  CROP_DIST_L=192.13
  CROP_DIST_R=328.27
  CROP_DEF_L=0
  CROP_DEF_R=0
fi

if [ $CROP_DEF_L == 1 ]
then
  echo "#### Warning: No custom cropping defined, orb analysis won't work ####"
  echo "#### Folder:" $(pwd)
fi

# --------------------------------------

python3 ./orbital_analysis_from_npz.py \
  --npz_file "$FOLDER"/morbs_h1_dx0.2.npz \
  --output_dir "$FOLDER"/sts_output \
  --sts_plane_height 1.0 \
  --emin -0.7 \
  --emax 0.7 \
  --lat_param 4.26 \
  --crop_x_l $CROP_DIST_L \
  --crop_x_r $CROP_DIST_R \
  --work_function 4.36

