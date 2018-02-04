#!/bin/bash -l

FOLDER=/home/kristjan/local_work/precursor_9agnr

python3 ./stm_sts_extrap_from_npz.py \
  --npz_file "$FOLDER"/morb_grid.npz \
  --hartree_file "$FOLDER"/V_HARTREE-v_hartree-1_0.cube \
  --extrap_plane 2.0 \
  --extrap_extent 8.0 \
  --output_dir "$FOLDER"/output \
  --bias_voltages -0.1 0.1 \
  --stm_plane_heights 4.0 5.0 \
  --stm_isovalues  1e-7 1e-6 \
  --sts_plane_heights 5.0 \
  --sts_de 0.025 \
  --sts_fwhm 0.05
