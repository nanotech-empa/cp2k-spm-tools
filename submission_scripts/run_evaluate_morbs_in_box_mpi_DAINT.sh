#!/bin/bash -l
#SBATCH --job-name="morb_calc"
#SBATCH --nodes=1 # the number of ranks (total)
#SBATCH --ntasks-per-node=12 # the number of ranks per node
#SBATCH --cpus-per-task=1 # use this for threaded applications
#SBATCH --time=3:00:00 
#SBATCH --constraint=gpu
#SBATCH --partition=normal

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1
module load daint-gpu

module load cray-python/17.09.1

ulimit -s unlimited

FOLDER=.

# Oldest .xyz file in the folder will be set as the geometry
GEOM=$(ls -tr $FOLDER | grep .xyz | head -n 1)

srun -n $SLURM_NTASKS python3 ./evaluate_morbs_in_box_mpi.py \
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
  | tee eval_morbs_in_box_mpi_h1.out

srun -n $SLURM_NTASKS python3 ./evaluate_morbs_in_box_mpi.py \
  --cp2k_input "$FOLDER"/cp2k.inp \
  --cp2k_output "$FOLDER"/cp2k.out \
  --basis_file "$FOLDER"/BR \
  --xyz_file "$FOLDER"/$GEOM \
  --restart_file "$FOLDER"/PROJ-RESTART.wfn \
  --output_file "$FOLDER"/morbs_h3_dx0.2 \
  --emin -2.0 \
  --emax  2.0 \
  --z_top 3.0 \
  --dx 0.2 \
  --local_eval_box_size 12.0 \
  --single_plane True \
  | tee eval_morbs_in_box_mpi_h3.out

