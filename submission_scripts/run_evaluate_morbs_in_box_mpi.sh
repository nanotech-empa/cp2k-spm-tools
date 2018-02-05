#!/bin/bash -l

FOLDER=/home/kristjan/local_work/precursor_9agnr

mpirun -n 2 python3 ./evaluate_morbs_in_box_mpi.py \
  --cp2k_input "$FOLDER"/cp2k.inp \
  --cp2k_output "$FOLDER"/cp2k.out \
  --basis_file "$FOLDER"/BR \
  --xyz_file "$FOLDER"/p.xyz \
  --restart_file "$FOLDER"/PROJ-RESTART.wfn \
  --output_file "$FOLDER"/morb_grid \
  --emin -0.2 \
  --emax  0.1 \
  --z_top 2.0 \
  --dx 0.5 \
  --local_eval_box_size 10.0 \
  | tee eval_morbs_in_box_mpi.out
