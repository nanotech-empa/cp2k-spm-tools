#!/bin/bash -l

FOLDER=..

mkdir $FOLDER/stm_output 

python3 ./stm_sts_extrap_from_npz.py \
  --npz_file "$FOLDER"/morbs_dx0.2.npz \
  --hartree_file "$FOLDER"/PROJ-HART-v_hartree-1_0.cube \
  --extrap_plane 3.0 \
  --extrap_extent 8.0 \
  --output_dir "$FOLDER"/stm_output \
\
  --bias_voltages -1.0 2.0 \
  --stm_plane_heights 3.0 5.0 \
  --stm_isovalues  1e-7 1e-6 \
\
  --sts_plane_heights 5.0 \
  --sts_de 0.05 \
  --sts_fwhm 0.10 \
  --sts_elim -2.0 2.0 \
\
  --orb_plane_heights 3.0 5.0 \
  --n_homo 10 \
  --n_lumo 10 \
\
  --skip_data_output \
#  --skip_figs
