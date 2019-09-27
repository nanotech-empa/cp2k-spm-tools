#!/bin/bash -l
#SBATCH --job-name="overlap"
#SBATCH --nodes=2 # the number of ranks (total)
#SBATCH --ntasks-per-node=12 # the number of ranks per node
#SBATCH --cpus-per-task=1 # use this for threaded applications
#SBATCH --time=4:00:00 
#SBATCH --constraint=gpu
#SBATCH --partition=normal

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1
module load daint-gpu

module load cray-python/3.6.1.1
export PYTHONPATH=$PYTHONPATH:"/users/keimre/soft/ase"

ulimit -s unlimited

SLAB_FOLDER=..
MOL_FOLDER=..

srun -n $SLURM_NTASKS --mpi=pmi2 python3 ./overlap_from_wfns.py \
  --cp2k_input_file1 "$SLAB_FOLDER"/aiida.inp \
  --basis_set_file1 "$SLAB_FOLDER"/BASIS_MOLOPT \
  --xyz_file1 "$SLAB_FOLDER"/geom.xyz \
  --wfn_file1 "$SLAB_FOLDER"/aiida-RESTART.wfn \
  --emin1 -2.0 \
  --emax1  2.0 \
  --cp2k_input_file2 "$MOL_FOLDER"/aiida.inp \
  --basis_set_file2 "$MOL_FOLDER"/BASIS_MOLOPT \
  --xyz_file2 "$MOL_FOLDER"/geom.xyz \
  --wfn_file2 "$MOL_FOLDER"/aiida-RESTART.wfn \
  --nhomo2 4 \
  --nlumo2 4 \
  --output_file "./overlap.npz" \
  --eval_region "n-2.0_C" "p2.0_C" "n-2.0_C" "p2.0_C" "n-2.0_C" "p2.0_C" \
  --dx 0.2 \
  --eval_cutoff 14.0 \

