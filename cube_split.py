#!/usr/bin/env python
import argparse
import copy
import os
import time

import ase
import numpy as np

ang_2_bohr = 1.0 / 0.52917721067
hart_2_ev = 27.21138602

from mpi4py import MPI

from cp2k_spm_tools import bader_wrapper, cube, cube_utils

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

parser = argparse.ArgumentParser(description="Splits the cube file into smaller cubes centered around atoms.")

parser.add_argument("cube", metavar="FILENAME", help="Input cube file.")
parser.add_argument(
    "--atom_box_size",
    type=float,
    metavar="L",
    required=False,
    default=8.0,
    help="specify the evaluation box (L^3) size around each atom in [ang].",
)
parser.add_argument("--output_dir", metavar="DIR", default=".", help="directory where to output the cubes.")
### -----------------------------------------------------------

time0 = time.time()

### ------------------------------------------------------
### Parse args for only one rank to suppress duplicate stdio
### ------------------------------------------------------

args = None
args_success = False
try:
    if mpi_rank == 0:
        args = parser.parse_args()
        args_success = True
finally:
    args_success = comm.bcast(args_success, root=0)

if not args_success:
    print(mpi_rank, "exiting")
    exit(0)

args = comm.bcast(args, root=0)

output_dir = args.output_dir if args.output_dir[-1] == "/" else args.output_dir + "/"

### ------------------------------------------------------
### Load the cube meta-data
### ------------------------------------------------------

inp_cube = cube.Cube()
inp_cube.read_cube_file(args.cube, read_data=False)
n_atoms = len(inp_cube.ase_atoms)

dv = np.diag(inp_cube.dv)

### ------------------------------------------------------
### Add periodic images of atoms that are close to the border
### ------------------------------------------------------

border_atom_images = ase.Atoms()

d_cell = np.diag(inp_cube.cell) / ang_2_bohr

inc_box = np.array(
    [
        inp_cube.origin / ang_2_bohr - args.atom_box_size / 2,
        inp_cube.origin / ang_2_bohr + d_cell + args.atom_box_size / 2,
    ]
)


def point_in_box(p, box):
    return box[0, 0] < p[0] < box[1, 0] and box[0, 1] < p[1] < box[1, 1] and box[0, 2] < p[2] < box[1, 2]


for i_x in [-1, 0, 1]:
    for i_y in [-1, 0, 1]:
        for i_z in [-1, 0, 1]:
            if i_x == 0 and i_y == 0 and i_z == 0:
                continue
            pbc_vec = np.array([i_x, i_y, i_z]) * d_cell
            for atom in inp_cube.ase_atoms:
                pos = atom.position + pbc_vec
                if point_in_box(pos, inc_box):
                    new_at = copy.deepcopy(atom)
                    new_at.position = pos
                    border_atom_images.append(new_at)


### ------------------------------------------------------
### Analyze file memory layout
### ------------------------------------------------------

fhandle = open(args.cube, "r")
n_metadata_lines = 6 + n_atoms
# where does cube data start?
data_start_offset = 0
for i_l in range(n_metadata_lines):
    fhandle.readline()
data_start_offset = fhandle.tell()

# how is the cube data organized?
# NB: THe following assumes that the number and width of columns
# remains the same for the whole file
### NOT TRUE FOR CP2K OUTPUTS ###
data_line = fhandle.readline()
n_columns = len(data_line.split())
data_line_offset = fhandle.tell() - data_start_offset

print("----------- data org: ", data_line, n_columns, data_line_offset)


def get_nth_value(n):
    row = int(n / n_columns)
    col = n % n_columns
    fhandle.seek(data_start_offset + row * data_line_offset)
    return fhandle.readline().split()[col]


### ------------------------------------------------------
### Divide the atoms between the mpi processes
### ------------------------------------------------------

base_atoms_per_rank = int(np.floor(n_atoms / mpi_size))
extra_atoms = n_atoms - base_atoms_per_rank * mpi_size
if mpi_rank < extra_atoms:
    i_atom_start = mpi_rank * (base_atoms_per_rank + 1)
    i_atom_end = (mpi_rank + 1) * (base_atoms_per_rank + 1)
else:
    i_atom_start = mpi_rank * (base_atoms_per_rank) + extra_atoms
    i_atom_end = (mpi_rank + 1) * (base_atoms_per_rank) + extra_atoms

print("R%d/%d, atom indexes %d:%d " % (mpi_rank, mpi_size, i_atom_start, i_atom_end))

### ------------------------------------------------------
### Loop over atoms and extract local cubes
### ------------------------------------------------------


def parse_cube_data(extract_indexes):
    """
    TOO SLOW TO BE USEFUL...
    Parse the whole cube file value-by-value (slow)
    by only taking the assigned memory...
    slow but memory usage is okay
    Also doesn't assume "fixed" column width in cube file
    """
    cube_data = np.zeros(len(extract_indexes))

    fhandle.seek(data_start_offset)
    cur_index = 0
    cube_i = 0
    for line in fhandle:
        vals = np.array(line.split(), dtype=float)

        for i_val, val in enumerate(vals):
            if cur_index + i_val in extract_indexes:
                cube_data[cube_i] = val
                cube_i += 1
        cur_index += len(vals)

    return cube_data


for i_at in range(i_atom_start, i_atom_end):
    at_pos = inp_cube.ase_atoms[i_at].position

    cube_pos_1 = at_pos - 0.5 * np.array([1.0, 1.0, 1.0]) * args.atom_box_size
    cube_pos_2 = at_pos + 0.5 * np.array([1.0, 1.0, 1.0]) * args.atom_box_size

    cube_pos_1_i = np.round((cube_pos_1 - inp_cube.origin / ang_2_bohr) / dv).astype(int)
    cube_pos_2_i = np.round((cube_pos_2 - inp_cube.origin / ang_2_bohr) / dv).astype(int)

    x_inds = np.arange(cube_pos_1_i[0], cube_pos_2_i[0]) % inp_cube.cell_n[0]
    y_inds = np.arange(cube_pos_1_i[1], cube_pos_2_i[1]) % inp_cube.cell_n[1]
    z_inds = np.arange(cube_pos_1_i[2], cube_pos_2_i[2]) % inp_cube.cell_n[2]

    cube_data = np.zeros(len(x_inds) * len(y_inds) * len(z_inds))

    # fastest index is z
    cube_i = 0
    for ix in x_inds:
        for iy in y_inds:
            for iz in z_inds:
                # extract_indexes.append(iz + iy * inp_cube.cell_n[2] + ix * inp_cube.cell_n[2] * inp_cube.cell_n[1])
                ind = iz + iy * inp_cube.cell_n[2] + ix * inp_cube.cell_n[2] * inp_cube.cell_n[1]
                cube_data[cube_i] = get_nth_value(ind)
                cube_i += 1

    # Add only the atoms that fit in the box
    atoms_in_box = ase.Atoms()
    middle_at_i = 0
    for at in inp_cube.ase_atoms + border_atom_images:
        if point_in_box(at.position, np.array([cube_pos_1, cube_pos_2])):
            if np.allclose(at.position, at_pos):
                middle_at_i = len(atoms_in_box)
            atoms_in_box.append(copy.deepcopy(at))

    # Save the new cube
    new_cube = cube.Cube(
        title="charge dens",
        comment="atom %d" % i_at,
        ase_atoms=atoms_in_box,
        origin=cube_pos_1_i * dv * ang_2_bohr + inp_cube.origin,
        cell=np.diag((cube_pos_2_i - cube_pos_1_i) * dv * ang_2_bohr),
        data=cube_data.reshape((len(x_inds), len(y_inds), len(z_inds))),
    )

    local_dir = output_dir + "atom_%04d/" % i_at
    os.makedirs(local_dir, exist_ok=True)

    cube_name = "at_%04d.cube" % i_at
    new_cube.write_cube_file(local_dir + cube_name)
    cube_utils.add_artif_core_charge(new_cube)
    ref_cube_name = "at_%04d_artif.cube" % i_at
    new_cube.write_cube_file(local_dir + ref_cube_name)

    neargrid_dir = local_dir + "neargrid/"
    weight_dir = local_dir + "weight/"

    os.makedirs(neargrid_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)

    bader_wrapper.call_bader(
        neargrid_dir, "../" + cube_name, ref_cube="../" + ref_cube_name, basin_atoms=[middle_at_i], method="neargrid"
    )
    bader_wrapper.call_bader(
        weight_dir, "../" + cube_name, ref_cube="../" + ref_cube_name, basin_atoms=[middle_at_i], method="weight"
    )

    with open(local_dir + "info.txt", "w") as info_f:
        info_f.write("basin_atom_local_index: %d\n" % middle_at_i)


print("R%d/%d finished, total time: %.2fs" % (mpi_rank, mpi_size, (time.time() - time0)))
