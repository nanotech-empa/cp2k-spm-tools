#!/bin/bash -l

FOLDER="../data/benzene_cp2k_scf"
BASIS_PATH="../data/BASIS_MOLOPT"

mkdir out

mpirun -n 2 cp2k-stm-sts-wfn \
  --cp2k_input_file "$FOLDER"/cp2k.inp \
  --basis_set_file $BASIS_PATH \
  --xyz_file "$FOLDER"/geom.xyz \
  --wfn_file "$FOLDER"/PROJ-RESTART.wfn \
  --hartree_file "$FOLDER"/PROJ-v_hartree-1_0.cube \
  --output_file "./out/stm.npz" \
  --orb_output_file "./out/orb.npz" \
\
  --eval_region "G" "G" "G" "G" "n-1.5_C" "p3.5" \
  --dx 0.15 \
  --eval_cutoff 14.0 \
  --extrap_extent 2.0 \
  --p_tip_ratios 0.0 1.0 \
\
  --n_homo 4 \
  --n_lumo 4 \
  --orb_heights 3.0 5.0 \
  --orb_isovalues 1e-7 \
  --orb_fwhms 0.1 \
\
  --energy_range -3.0 3.0 1.0 \
  --heights 4.5 \
  --isovalues 1e-7 \
  --fwhms 0.5 \

cd out 
cp2k-stm-sts-plot --orb_npz orb.npz --stm_npz stm.npz

