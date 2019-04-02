#!/bin/bash -l
#SBATCH --job-name="morb_calc"
#SBATCH --nodes=1 # the number of ranks (total)
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

FOLDER=".."

# Oldest .xyz file in the folder will be set as the geometry
GEOM=$(ls -tr $FOLDER | grep .xyz | head -n 1)

srun -n $SLURM_NTASKS --mpi=pmi2 python3 ./stm_sts_from_wfn.py \
  --cp2k_input_file "$FOLDER"/aiida.inp \
  --basis_set_file "$FOLDER"/BASIS_MOLOPT \
  --xyz_file "$FOLDER"/$GEOM \
  --wfn_file "$FOLDER"/aiida-RESTART.wfn \
  --hartree_file "$FOLDER"/aiida-HART-v_hartree-1_0.cube \
  --output_file "./stm.npz" \
  --orb_output_file "./orb.npz" \
\
  --emin -2.0 \
  --emax 2.0 \
\
  --eval_region "G" "G" "G" "G" "n-1.5_C" "p3.0" \
  --dx 0.2 \
  --eval_cutoff 14.0 \
  --extrap_extent 5.0 \
\
  --orb_heights 3.0 5.0 \
  --n_homo_ch 5 \
  --n_lumo_ch 5 \
\
  --isovalues 1e-8 1e-6 \
  --heights 3.0 5.0 \
  --de 0.05 \
  --fwhm 0.10 \
