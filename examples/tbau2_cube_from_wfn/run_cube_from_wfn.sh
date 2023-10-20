#!/bin/bash -l

DIR="../tbau2_cp2k_scf"

mkdir cubes

../../cube_from_wfn.py \
  --cp2k_input_file $DIR/inp \
  --basis_set_file $DIR/BASIS_MOLOPT \
  --xyz_file $DIR/aiida.coords.xyz \
  --wfn_file $DIR/PROJ-RESTART.wfn \
  --output_dir ./cubes/ \
  --n_homo 2 \
  --n_lumo 2 \
  --dx .040404251481 \
  --eval_cutoff 10.0 \
#  --orb_square \
#  --charge_dens \
#  --spin_dens \

