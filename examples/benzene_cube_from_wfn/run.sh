#!/bin/bash -l

DATA_PATH="../data/benzene_cp2k_scf/"
BASIS_PATH="../data/BASIS_MOLOPT"

mkdir cubes

cp2k-cube-from-wfn \
  --cp2k_input_file $DATA_PATH/cp2k.inp \
  --basis_set_file $BASIS_PATH \
  --xyz_file $DATA_PATH/geom.xyz \
  --wfn_file $DATA_PATH/PROJ-RESTART.wfn \
  --output_dir ./cubes/ \
  --n_homo 1 \
  --n_lumo 1 \
  --dx 0.8 \
  --eval_cutoff 14.0 \
#  --orb_square \
#  --charge_dens \
#  --spin_dens \

