#!/bin/bash -l

FOLDER=..

# Oldest .xyz file in the folder will be set as the geometry
GEOM=$(ls -tr $FOLDER | grep .xyz | head -n 1)

#python3 ./evaluate_morbs_in_box_mpi.py \
#  --cp2k_input "$FOLDER"/cp2k.inp \
#  --basis_file "$FOLDER"/BR \
#  --xyz_file "$FOLDER"/$GEOM \
#  --wfn_file "$FOLDER"/PROJ-RESTART.wfn \
#  --output_file "$FOLDER"/morbs_dx0.2 \
#  --emin -2.0 \
#  --emax  2.0 \
#  --eval_region G G G G 4.0t 4.0t \
#  --dx 0.2 \
#  --eval_cutoff 14.0 \

python3 ./evaluate_morbs_in_box_mpi.py \
  --cp2k_input "$FOLDER"/cp2k.inp \
  --basis_file "$FOLDER"/BR \
  --xyz_file "$FOLDER"/$GEOM \
  --wfn_file "$FOLDER"/PROJ-RESTART.wfn \
  --output_file "$FOLDER"/morbs_dx0.2 \
  --emin -2.0 \
  --emax  3.0 \
  --eval_region G G G G " -2.0b_C" 4.0t \
  --dx 0.2 \
  --eval_cutoff 14.0 \
