#!/bin/bash -l

FOLDER="../data/c2h2_cp2k_scf"
BASIS_PATH="../data/BASIS_MOLOPT"

mkdir out

mpirun -n 2 python3 ../../cube_from_wfn.py \
  --cp2k_input_file "$FOLDER"/cp2k.inp \
  --basis_set_file $BASIS_PATH \
  --xyz_file "$FOLDER"/geom.xyz \
  --wfn_file "$FOLDER"/PROJ-RESTART.wfn \
  --output_dir "./out/" \
\
  --dx 0.08 \
  --eval_cutoff 14.0 \
  --eval_region "G" "G" "G" "G" "G" "G" \
\
  --charge_dens \
  --charge_dens_artif_core \

cd out

wget http://theory.cm.utexas.edu/henkelman/code/bader/download/bader_lnx_64.tar.gz

tar -xvf bader_lnx_64.tar.gz

./bader -b weight -p all_atom  charge_density.cube -ref charge_density_artif.cube

