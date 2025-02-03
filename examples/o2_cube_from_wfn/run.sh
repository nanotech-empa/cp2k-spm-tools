#!/bin/bash -l

DIR="../data/o2_cp2k_scf"
BASIS_PATH="../data/BASIS_MOLOPT"

mkdir cubes

../../cube_from_wfn.py \
  --cp2k_input_file $DIR/inp \
  --basis_set_file $BASIS_PATH \
  --xyz_file $DIR/aiida.coords.xyz \
  --wfn_file $DIR/PROJ-RESTART.wfn \
  --output_dir ./cubes/ \
  --n_homo 3 \
  --n_lumo 3 \
  --dx .714283 \
  --eval_cutoff 14.0 \
  --do_not_center_atoms
#  --orb_square \
#  --charge_dens \
#  --spin_dens \

