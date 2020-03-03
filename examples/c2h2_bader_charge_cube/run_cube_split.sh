#!/bin/bash -l

if ! type -P "bader";
then
    echo "Error: bader executable not in path!"
    exit 1
fi

mkdir -p out

mpirun -n 2 cube_split.py \
  "../c2h2_cp2k_scf/charge_density.cube" \
  --atom_box_size 3.7 \
  --output_dir "./out/" \

