#!/bin/bash -l
# ================ MARCONI =================
#PBS -u keimre00
#PBS -A Pra14_3518
#PBS -q serial
#
#PBS -N sts
#
#PBS -l walltime=4:00:00

FOLDER=/marconi/home/userexternal/keimre00/scratch/cp2k_cnt_12_0/ideal_DZVP

python3 ./sts.py \
  --cp2k_input "$FOLDER"/cp2k.inp \
  --cp2k_output "$FOLDER"/out.log \
  --basis_file "$FOLDER"/BR \
  --xyz_file "$FOLDER"/CNT12_0.xyz \
  --restart_file "$FOLDER"/PROJ-RESTART.wfn \
  --emin -2.0 \
  --emax  2.0 \
  --height 1.0 \
  --dx 0.0852 \
  --de 0.002 \
  --fwhm 0.01 0.05 \
  | tee sts.out

