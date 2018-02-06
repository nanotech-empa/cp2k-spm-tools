#!/bin/bash -l

FOLDER=.

python3 ./ldos_postproc.py \
  --ldos_file "$FOLDER"/sts_output/ldos_cnt120-ideal-L15_h1.0_fwhm0.01g.txt \
  --output_dir "$FOLDER"/sts_output/ \
  --crop_dist_l 5.0 \
  --crop_dist_r 5.0 \
  --crop_defect_l 1 \
  --crop_defect_r 1 \
  --padding_x 300.0 \
  --emin -0.7 \
  --emax 0.7
