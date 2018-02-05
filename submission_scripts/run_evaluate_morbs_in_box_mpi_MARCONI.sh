#!/bin/bash -l
#SBATCH --job-name="na_rib"
#SBATCH --nodes=1 # the number of ranks (total)
#SBATCH --ntasks-per-node=64 # the number of ranks per node
#SBATCH --cpus-per-task=1 # use this for threaded applications
#SBATCH --time=6:00:00 

#SBATCH --account=Pra14_3518
#SBATCH --partition=knl_usr_prod

#SBATCH --mem=86GB

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load env-knl

module load intel/pe-xe-2017--binary
module load intelmpi/2017--binary

#module load gnu/6.1.0
#module load openmpi/1-10.3--gnu--6.1.0

module load mkl/2017--binary
module load python/3.5.2
module load numpy/1.11.2--python--3.5.2
module load scipy/0.18.1--python--3.5.2
module load mpi4py/2.0.0--python--3.5.2

ulimit -s unlimited

FOLDER=.

EXE=/cineca/prod/opt/compilers/python/3.5.2/none/bin/python3

echo $(which python3)

echo $PATH

mpiexec -n $SLURM_NTASKS $EXE ./evaluate_morbs_in_box_mpi.py \
  --cp2k_input "$FOLDER"/cp2k.inp \
  --cp2k_output "$FOLDER"/cp2k.out \
  --basis_file "$FOLDER"/BR \
  --xyz_file "$FOLDER"/p_opt_centered.xyz \
  --restart_file "$FOLDER"/PROJ-RESTART.wfn \
  --output_file "$FOLDER"/morb_grid \
  --emin -1.0 \
  --emax  1.0 \
  --z_top 4.0 \
  --dx 0.2 \
  --local_eval_box_size 22.0 \
  | tee eval_morbs_in_box_mpi.out
