#!/bin/bash -l
#SBATCH --job-name="morb_calc"
#SBATCH --nodes=1 # the number of ranks (total)
#SBATCH --ntasks-per-node=4 # the number of ranks per node
#SBATCH --cpus-per-task=1 # use this for threaded applications
#SBATCH --time=2:00:00 
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

#srun -n $SLURM_NTASKS --mpi=pmi2 python3 ./evaluate_morbs_in_box_mpi.py \
#  --cp2k_input "$FOLDER"/cp2k.inp \
#  --basis_file "$FOLDER"/BR \
#  --xyz_file "$FOLDER"/$GEOM \
#  --wfn_file "$FOLDER"/PROJ-RESTART.wfn \
#  --output_file "$FOLDER"/morbs_dx0.2 \
#  --emin -2.0 \
#  --emax  3.0 \
#  --eval_region G G G G t4.0 t4.0 \
#  --dx 0.2 \
#  --eval_cutoff 14.0 \
#  | tee eval_morbs_in_box_mpi.out

srun -n $SLURM_NTASKS --mpi=pmi2 python3 ./evaluate_morbs_in_box_mpi.py \
  --cp2k_input "$FOLDER"/cp2k.inp \
  --basis_file "$FOLDER"/BR \
  --xyz_file "$FOLDER"/$GEOM \
  --wfn_file "$FOLDER"/PROJ-RESTART.wfn \
  --output_file "$FOLDER"/morbs_dx0.2 \
  --emin -2.0 \
  --emax  3.0 \
  --eval_region G G G G b-2.0_C t4.0 \
  --dx 0.2 \
  --eval_cutoff 14.0 \
  | tee eval_morbs_in_box_mpi.out

