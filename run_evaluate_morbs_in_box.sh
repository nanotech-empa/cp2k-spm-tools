#!/bin/bash -l

FOLDER=/home/kristjan/local_work/precursor_9agnr

python3 ./evaluate_morbs_in_box.py \
  --cp2k_input "$FOLDER"/cp2k.inp \
  --cp2k_output "$FOLDER"/cp2k.out \
  --basis_file "$FOLDER"/BR \
  --xyz_file "$FOLDER"/p.xyz \
  --restart_file "$FOLDER"/PROJ-RESTART.wfn \
  --output_file "$FOLDER"/morb_grid \
  --emin -1.0 \
  --emax  1.0 \
  --z_top 4.0 \
  --dx 0.5 \
  --local_eval_box_size 22.0 \
  | tee eval_morbs_in_box.out
