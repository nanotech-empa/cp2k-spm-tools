#!/bin/bash -l

FOLDER=.

mkdir sts_output

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


