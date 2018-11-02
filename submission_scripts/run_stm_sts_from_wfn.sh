#!/bin/bash -l

FOLDER="/home/kristjan/local_work/stm-test/parent_calc/parent_calc"

mpirun -n 3 python3 ./stm_sts_from_wfn.py \
  --cp2k_input_file "$FOLDER"/aiida.inp \
  --xyz_file "$FOLDER"/geom.xyz \
  --basis_set_file "$FOLDER"/BASIS_MOLOPT \
  --wfn_file "$FOLDER"/aiida-RESTART.wfn \
  --hartree_file "$FOLDER"/aiida-HART-v_hartree-1_0.cube \
  --output_file "./stm.npz" \
\
  --emin -2.0 \
  --emax 2.0 \
  --eval_region "n-3.0_C" "p3.0_C" "n-3.0_C" "p3.0_C" "n-1.5_C" "p3.0" \
  --dx 0.2 \
  --eval_cutoff 14.0 \
  --extrap_extent 5.0 \
\
  --stm_isovalues 1e-8 1e-6 \
  --stm_heights 3.0 5.0 \
  --sts_de 0.05 \
  --sts_fwhm 0.10 \

