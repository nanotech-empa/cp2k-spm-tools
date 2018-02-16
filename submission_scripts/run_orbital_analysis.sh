#!/bin/bash -l

FOLDER=.

python3 ./orbital_analysis_from_npz.py \
  --npz_file "$FOLDER"/morbs_h1_dx0.2.npz \
  --output_dir "$FOLDER"/sts_output \
  --sts_plane_height 1.0 \
  --nhomo 100 \
  --nlumo 1 \
  --hartree_file "$FOLDER"/V_HARTREE-v_hartree-1_0.cube 
