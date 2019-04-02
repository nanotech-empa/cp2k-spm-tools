#!/bin/bash -l

FOLDER="."

mpirun -n 3 python3 ~/work/atomistic_tools/stm_sts_from_wfn.py \
  --cp2k_input_file "$FOLDER"/aiida.inp \
  --basis_set_file "$FOLDER"/BASIS_MOLOPT \
  --xyz_file "$FOLDER"/geom.xyz \
  --wfn_file "$FOLDER"/aiida-RESTART.wfn \
  --hartree_file "$FOLDER"/aiida-HART-v_hartree-1_0.cube \
  --output_file "./stm.npz" \
  --orb_output_file "./orb.npz" \
\
  --emin -2.0 \
  --emax 2.0 \
\
  --eval_region "G" "G" "G" "G" "n-1.5_C" "p3.0" \
  --dx 0.2 \
  --eval_cutoff 14.0 \
  --extrap_extent 5.0 \
\
  --orb_heights 3.0 5.0 \
  --n_homo_ch 5 \
  --n_lumo_ch 5 \
\
  --isovalues 1e-8 1e-6 \
  --heights 3.0 5.0 \
  --de 0.05 \
  --fwhm 0.10 \
