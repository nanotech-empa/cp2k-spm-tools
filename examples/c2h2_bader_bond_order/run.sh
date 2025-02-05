#!/bin/bash -l

FOLDER="../data/c2h2_cp2k_scf"
BASIS_PATH="../data/BASIS_MOLOPT"

mkdir out

echo "### 1: calculate charge density cube ###"

mpirun -n 2 cp2k-cube-from-wfn \
  --cp2k_input_file "$FOLDER"/cp2k.inp \
  --basis_set_file $BASIS_PATH \
  --xyz_file "$FOLDER"/geom.xyz \
  --wfn_file "$FOLDER"/PROJ-RESTART.wfn \
  --output_dir "./out/" \
\
  --dx 0.1 \
  --eval_cutoff 14.0 \
  --eval_region "n-2.0_H" "p2.0_H" "n-3.0_C" "p3.0_C" "n-3.0_C" "p3.0_C" \
\
  --n_homo 2 \
  --n_lumo 2 \
  --orb_square \
\
  --charge_dens \
  --charge_dens_artif_core \
  --spin_dens \

echo "### 2: calculate Bader basins ###"

cd out

wget http://theory.cm.utexas.edu/henkelman/code/bader/download/bader_lnx_64.tar.gz

tar -xvf bader_lnx_64.tar.gz

# neargrid
#./bader -p sel_atom 1 2 3 4 charge_density_artif.cube
# weight
./bader -b weight -p sel_atom 1 2 3 4 charge_density_artif.cube

cd ..

echo "### 3: calculate bond order based on the Bader basins ###"

mpirun -n 2 cp2k-bader-bond-order \
  --cp2k_input_file "$FOLDER"/cp2k.inp \
  --basis_set_file $BASIS_PATH \
  --xyz_file "$FOLDER"/geom.xyz \
  --wfn_file "$FOLDER"/PROJ-RESTART.wfn \
\
  --output_file "./out/bond_order.txt" \
  --bader_basins_dir "./out/" \
\
  --dx 0.1 \
  --eval_cutoff 14.0 \
  --eval_region "n-2.0_H" "p2.0_H" "n-3.0_C" "p3.0_C" "n-3.0_C" "p3.0_C" \
#  --eval_region "G" "G" "G" "G" "G" "G"

