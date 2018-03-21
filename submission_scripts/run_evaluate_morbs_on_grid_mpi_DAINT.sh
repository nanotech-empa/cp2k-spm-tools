#!/bin/bash -l
#SBATCH --job-name="morb_calc"
#SBATCH --nodes=1 # the number of ranks (total)
#SBATCH --ntasks-per-node=12 # the number of ranks per node
#SBATCH --cpus-per-task=1 # use this for threaded applications
#SBATCH --time=0:30:00 
#SBATCH --constraint=gpu
#SBATCH --partition=low

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1
module load daint-gpu

module load cray-python/17.09.1

ulimit -s unlimited

FOLDER=.

# Oldest .xyz file in the folder will be set as the geometry
GEOM=$(ls -tr $FOLDER | grep .xyz | head -n 1)

srun -n $SLURM_NTASKS python3 ./evaluate_morbs_on_grid_mpi.py \
  --cp2k_input "$FOLDER"/cp2k.inp \
  --cp2k_output "$FOLDER"/cp2k.out \
  --basis_file "$FOLDER"/BR \
  --xyz_file "$FOLDER"/$GEOM \
  --restart_file "$FOLDER"/PROJ-RESTART.wfn \
  --output_file "$FOLDER"/morbs_grid_dx0.4 \
  --emin -1.0 \
  --emax  1.0 \
  --x_extra 4.0 \
  --y_extra 4.0 \
  --z_extra 4.0 \
  --dx 0.4 \
  --local_eval_box_size 12.0 \
  | tee eval_morbs_on_grid_mpi.out

