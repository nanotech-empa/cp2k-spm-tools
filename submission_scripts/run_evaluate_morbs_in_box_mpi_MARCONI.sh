#!/bin/bash -l
#SBATCH --job-name="eval_morbs"
#SBATCH --nodes=2 # the number of ranks (total)
#SBATCH --ntasks-per-node=64 # the number of ranks per node
#SBATCH --cpus-per-task=1 # use this for threaded applications
#SBATCH --time=0:30:00 

#SBATCH --account=Pra14_3518
#SBATCH --partition=knl_usr_dbg

#SBATCH --mem=83GB

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

# Oldest .xyz file in the folder will be set as the geometry
GEOM=$(ls -tr $FOLDER | grep .xyz | head -n 1)

mpiexec -n $SLURM_NTASKS $EXE ./evaluate_morbs_in_box_mpi.py \
  --cp2k_input "$FOLDER"/cp2k.inp \
  --cp2k_output "$FOLDER"/cp2k.out \
  --basis_file "$FOLDER"/BR \
  --xyz_file "$FOLDER"/$GEOM \
  --restart_file "$FOLDER"/PROJ-RESTART.wfn \
  --output_file "$FOLDER"/morbs_h1_dx0.2 \
  --emin -2.0 \
  --emax  2.0 \
  --z_top 1.0 \
  --dx 0.2 \
  --local_eval_box_size 12.0 \
  --single_plane True \
  | tee eval_morbs_in_box_mpi.out
