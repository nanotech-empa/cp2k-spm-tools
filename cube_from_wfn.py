#!/usr/bin/env python
import argparse
import time

import numpy as np

ang_2_bohr = 1.0 / 0.52917721067
hart_2_ev = 27.21138602

from mpi4py import MPI

import cp2k_spm_tools.cp2k_grid_orbitals as cgo
from cp2k_spm_tools import common

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

parser = argparse.ArgumentParser(description="Creates Gaussian cube files from cp2k .wfn file.")

parser.add_argument(
    "--cp2k_input_file",
    metavar="FILENAME",
    required=True,
    help="CP2K input of the SCF calculation.",
)
parser.add_argument(
    "--basis_set_file",
    metavar="FILENAME",
    required=True,
    help="File containing the used basis sets.",
)
parser.add_argument(
    "--xyz_file",
    metavar="FILENAME",
    required=True,
    help=".xyz file containing the geometry.",
)
parser.add_argument(
    "--wfn_file",
    metavar="FILENAME",
    required=True,
    help="cp2k restart file containing the wavefunction.",
)

parser.add_argument(
    "--output_dir",
    metavar="DIR",
    required=True,
    help="directory where to output the cubes.",
)
### -----------------------------------------------------------
parser.add_argument(
    "--dx",
    type=float,
    metavar="DX",
    default=0.2,
    help="Spatial step for the grid (angstroms).",
)
parser.add_argument(
    "--eval_cutoff",
    type=float,
    metavar="D",
    default=14.0,
    help=("Size of the region around the atom where each orbital is evaluated (only used for 'G' region)."),
)
parser.add_argument(
    "--eval_region",
    type=str,
    nargs=6,
    metavar="X",
    required=False,
    default=["G", "G", "G", "G", "G", "G"],
    help=common.eval_region_description,
)
parser.add_argument(
    "--pbc",
    type=int,
    nargs=3,
    metavar="X",
    required=False,
    default=[1, 1, 1],
    help="periodic boundary conditions in directions [x,y,z]. (1=on, 0=off)",
)
### -----------------------------------------------------------
parser.add_argument(
    "--n_homo",
    type=int,
    metavar="N",
    default=0,
    help="Number of HOMO orbitals to export.",
)
parser.add_argument(
    "--n_lumo",
    type=int,
    metavar="N",
    default=0,
    help="Number of LUMO orbitals to export.",
)
parser.add_argument(
    "--orb_square",
    action="store_true",
    help=("Additionally generate the square (RHO) for each MO."),
)
### -----------------------------------------------------------
parser.add_argument(
    "--charge_dens",
    action="store_true",
    help=("Calculate charge density (all occupied orbitals are evaluated)."),
)
parser.add_argument(
    "--charge_dens_artif_core",
    action="store_true",
    help=("Calculate charge density with 'fake' artificial core (all occ orbitals are evaluated)."),
)
parser.add_argument(
    "--spin_dens",
    action="store_true",
    help=("Calculate spin density (all occupied orbitals are evaluated)."),
)
### -----------------------------------------------------------
parser.add_argument("--do_not_center_atoms", action="store_true", help=("Center atoms to cell."))
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
### Evaluate orbitals on the real-space grid
### ------------------------------------------------------

n_homo = args.n_homo
n_lumo = args.n_lumo

n_homo_range = n_homo
if args.charge_dens or args.spin_dens:
    n_homo_range = None

mol_grid_orb = cgo.Cp2kGridOrbitals(mpi_rank, mpi_size, comm, single_precision=False)
mol_grid_orb.read_cp2k_input(args.cp2k_input_file)
mol_grid_orb.read_xyz(args.xyz_file)
if not args.do_not_center_atoms:
    mol_grid_orb.center_atoms_to_cell()
mol_grid_orb.read_basis_functions(args.basis_set_file)
mol_grid_orb.load_restart_wfn_file(args.wfn_file, n_occ=n_homo_range, n_virt=n_lumo)

eval_reg = common.parse_eval_region_input(args.eval_region, mol_grid_orb.ase_atoms, mol_grid_orb.cell)

mol_grid_orb.calc_morbs_in_region(
    args.dx,
    x_eval_region=eval_reg[0],
    y_eval_region=eval_reg[1],
    z_eval_region=eval_reg[2],
    pbc=np.array(args.pbc, dtype=bool),
    reserve_extrap=0.0,
    eval_cutoff=args.eval_cutoff,
)


### ------------------------------------------------------
### Export the data
### ------------------------------------------------------

ase_atoms = mol_grid_orb.ase_atoms
origin = mol_grid_orb.origin
cell = mol_grid_orb.eval_cell * np.eye(3)
vol_elem = np.prod(mol_grid_orb.dv)

for imo in np.arange(n_homo + n_lumo):
    # i_rel_homo = imo - n_homo + 1
    for ispin in range(mol_grid_orb.nspin):
        ispin_homo = mol_grid_orb.cwf.i_homo[ispin]
        if imo >= len(mol_grid_orb.cwf.global_morb_indexes[ispin]):
            continue

        global_index = mol_grid_orb.cwf.global_morb_indexes[ispin][imo]
        i_rel_homo = global_index - ispin_homo
        if i_rel_homo < 0:
            hl_label = "HOMO%+d" % i_rel_homo
        elif i_rel_homo == 0:
            hl_label = "HOMO"
        elif i_rel_homo == 1:
            hl_label = "LUMO"
        else:
            hl_label = "LUMO%+d" % (i_rel_homo - 1)

        name = "S%d_%d_%s" % (ispin, global_index + 1, hl_label)
        mol_grid_orb.write_cube(output_dir + name + ".cube", i_rel_homo, spin=ispin)

        if args.orb_square:
            mol_grid_orb.write_cube(output_dir + name + "_sq.cube", i_rel_homo, spin=ispin, square=True)

if args.charge_dens:
    mol_grid_orb.calculate_and_save_charge_density(output_dir + "charge_density.cube")
if args.charge_dens_artif_core:
    mol_grid_orb.calculate_and_save_charge_density(output_dir + "charge_density_artif.cube", artif_core=True)

if args.spin_dens:
    mol_grid_orb.calculate_and_save_spin_density(output_dir + "spin_density.cube")

print("R%d/%d: finished, total time: %.2fs" % (mpi_rank, mpi_size, (time.time() - time0)))
