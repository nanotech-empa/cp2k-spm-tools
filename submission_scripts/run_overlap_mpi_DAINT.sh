#!/bin/bash -l
#SBATCH --job-name="overlap"
#SBATCH --nodes=1 # the number of ranks (total)
#SBATCH --ntasks-per-node=1 # the number of ranks per node
#SBATCH --cpus-per-task=1 # use this for threaded applications
#SBATCH --time=0:20:00 
#SBATCH --constraint=gpu
#SBATCH --partition=normal

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1
module load daint-gpu

module load cray-python/3.6.1.1
export PYTHONPATH=$PYTHONPATH:"/users/keimre/soft/ase"

ulimit -s unlimited

FOLDER=..

# Oldest .xyz file in the folder will be set as the geometry
GEOM=$(ls -tr $FOLDER | grep .xyz | head -n 1)

srun -n $SLURM_NTASKS --mpi=pmi2 python3 ./overlap_calc_mpi.py \
  --npz_file1 ../mol-scf/morbs_dx0.2.npz \
  --npz_file2 ../au-scf/morbs_dx0.2.npz \
  --n_homo 3 \
  --n_lumo 3

