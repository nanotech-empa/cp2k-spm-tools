#!/bin/bash -l

FOLDER=.

cd $FOLDER

mkdir sts_output

python3 ./sts_1d_ldos_from_npz.py \
  --npz_file "$FOLDER"/morbs_h1_dx0.2.npz \
  --output_dir "$FOLDER"/sts_output \
  --sts_plane_height 1.0 \
  --sts_de 0.005 \
  --sts_fwhm 0.01 0.05 \
  --work_function 4.36

# the newest .txt in the sts_output folder 
LDOS_FILE=$(ls -t sts_output | grep .txt | head -n 1)

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


python3 ./ldos_postproc.py \
  --ldos_file "$FOLDER"/sts_output/$LDOS_FILE \
  --output_dir "$FOLDER"/sts_output/ \
  --crop_dist_l $CROP_DIST_L \
  --crop_dist_r $CROP_DIST_R \
  --crop_defect_l $CROP_DEF_L \
  --crop_defect_r $CROP_DEF_R \
  --padding_x 300.0 \
  --emin -0.7 \
  --emax 0.7 \
  --gammas 1.0 0.5 \
  --vmax_coefs 1.0 0.5


