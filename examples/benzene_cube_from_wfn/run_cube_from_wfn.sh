#!/bin/bash -l

DIR="../benzene_cp2k_scf/"

mkdir cubes

../../cube_from_wfn.py \
  --cp2k_input_file $DIR/cp2k.inp \
  --basis_set_file ../BASIS_MOLOPT \
  --xyz_file $DIR/geom.xyz \
  --wfn_file $DIR/PROJ-RESTART.wfn \
  --output_dir ./cubes/ \
  --n_homo 1 \
  --n_lumo 1 \
  --dx 0.2 \
  --eval_cutoff 14.0 \
#  --orb_square \
#  --charge_dens \
#  --spin_dens \

