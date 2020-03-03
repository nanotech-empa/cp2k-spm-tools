#!/bin/bash -l
#SBATCH --job-name="cube_split"
#SBATCH --nodes=2 # the number of ranks (total)
#SBATCH --ntasks-per-node=12 # the number of ranks per node
#SBATCH --cpus-per-task=1 # use this for threaded applications
#SBATCH --time=02:00:00 
#SBATCH --constraint=gpu
#SBATCH --partition=normal
#SBATCH --account=s904

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1
module load daint-gpu

module load cray-python/3.6.1.1
export PYTHONPATH=$PYTHONPATH:"/users/keimre/soft/ase"

ulimit -s unlimited

SCRIPT_DIR="/users/keimre/atomistic_tools"
BASIS_PATH="/users/keimre/soft/cp2k_5.1_18265/data/BASIS_MOLOPT"

export PATH="/users/keimre/bin:$PATH"

if ! type -P "bader";
then
    echo "Error: bader executable not in path!"
    exit 1
fi

mkdir -p out

srun -n $SLURM_NTASKS --mpi=pmi2 python3 $SCRIPT_DIR/cube_split.py \
  "../PROJ-RHO-ELECTRON_DENSITY-1_0.cube" \
  --atom_box_size 6.0 \
  --output_dir "./out/" \

