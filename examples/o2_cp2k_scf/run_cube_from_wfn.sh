#!/bin/bash -l

DIR="./"

mkdir cubes

/Users/cpi/opt/cp2k-spm-tools/cp2k_spm_tools/cp2k_wfn_file.py \
  --cp2k_input_file $DIR/inp \
  --basis_set_file $DIR/BASIS_MOLOPT \
  --xyz_file $DIR/aiida.coords.xyz \
  --wfn_file $DIR/PROJ-RESTART.wfn \
  --output_dir ./cubes/ \
  --n_homo 2 \
  --n_lumo 2 \
  --dx .079364908 \
  --eval_cutoff 8.0 \
#  --orb_square \
#  --charge_dens \
#  --spin_dens \

