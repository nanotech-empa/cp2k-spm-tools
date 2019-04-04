#!/usr/bin/env python
import os
import numpy as np
import time
import copy
import sys

import argparse

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

import atomistic_tools.cp2k_grid_orbitals as cgo
from atomistic_tools import common, cube

parser = argparse.ArgumentParser(
    description='Creates Gaussian cube files from cp2k .wfn file.')

parser.add_argument(
    '--cp2k_input_file',
    metavar='FILENAME',
    required=True,
    help='CP2K input of the SCF calculation.')
parser.add_argument(
    '--basis_set_file',
    metavar='FILENAME',
    required=True,
    help='File containing the used basis sets.')
parser.add_argument(
    '--xyz_file',
    metavar='FILENAME',
    required=True,
    help='.xyz file containing the geometry.')
parser.add_argument(
    '--wfn_file',
    metavar='FILENAME',
    required=True,
    help='cp2k restart file containing the wavefunction.')

parser.add_argument(
    '--output_dir',
    metavar='DIR',
    required=True,
    help='directory where to output the cubes.')

parser.add_argument(
    '--n_homo',
    type=int,
    metavar='N',
    default=0,
    help="Number of HOMO orbitals to export.")
parser.add_argument(
    '--n_lumo',
    type=int,
    metavar='N',
    default=0,
    help="Number of LUMO orbitals to export.")

parser.add_argument(
    '--spin_dens_n',
    type=int,
    metavar='N',
    default=0,
    help=("Number of HOMO orbitals to include for spin"
          " density calculation. 0 disables spin density export")
)
parser.add_argument(
    '--dx',
    type=float,
    metavar='DX',
    default=0.2,
    help='Spatial step for the grid (angstroms).')
parser.add_argument(
    '--eval_cutoff',
    type=float,
    metavar='D',
    default=14.0,
    help=("Size of the region around the atom where each"
          " orbital is evaluated (only used for 'G' region).")
)
parser.add_argument(
    '--gen_rho',
    action='store_true',
    help=("Additionally generate the square (RHO) for each MO.")
)

time0 = time.time()

args = parser.parse_args()

n_homo = max(args.n_homo, args.spin_dens_n)
n_lumo = args.n_lumo

output_dir = args.output_dir if args.output_dir[-1] == '/' else args.output_dir+"/"

mol_grid_orb = cgo.Cp2kGridOrbitals(0, 1, single_precision=False)
mol_grid_orb.read_cp2k_input(args.cp2k_input_file)
mol_grid_orb.read_xyz(args.xyz_file)
mol_grid_orb.center_atoms_to_cell()
mol_grid_orb.read_basis_functions(args.basis_set_file)
mol_grid_orb.load_restart_wfn_file(args.wfn_file, n_homo=n_homo, n_lumo=n_lumo)

eval_reg = common.parse_eval_region_input(["G", "G", "G", "G", "G", "G"], mol_grid_orb.ase_atoms, mol_grid_orb.cell)

mol_grid_orb.calc_morbs_in_region(args.dx,
                                x_eval_region = eval_reg[0],
                                y_eval_region = eval_reg[1],
                                z_eval_region = eval_reg[2],
                                reserve_extrap = 0.0,
                                eval_cutoff = args.eval_cutoff)

ase_atoms = mol_grid_orb.ase_atoms
origin = mol_grid_orb.origin
cell = mol_grid_orb.eval_cell*np.eye(3)

vol_elem = np.prod(mol_grid_orb.dv)

spin_dens = np.zeros(mol_grid_orb.morb_grids[0][0].shape)

for imo in np.arange(n_homo+n_lumo):
    i_rel_homo = imo - n_homo + 1
    
    for ispin in range(mol_grid_orb.nspin):
        time1 = time.time()
        energy = mol_grid_orb.morb_energies[ispin][imo]

        name = "HOMO%+d_S%d_E%.3f" % (i_rel_homo, ispin, energy)

        norm_sq = np.sum(mol_grid_orb.morb_grids[ispin][imo]**2)*vol_elem
        comment = "E=%.8e eV (wrt HOMO), norm-1=%.4e" % (energy, norm_sq-1.0)

        c = cube.Cube(title=name, comment=comment, ase_atoms=ase_atoms, origin=origin, cell=cell, data=mol_grid_orb.morb_grids[ispin][imo])
        c.write_cube_file(output_dir + name + ".cube")

        if args.gen_rho:
            c = cube.Cube(title="Squared " + name, comment=comment, ase_atoms=ase_atoms, origin=origin, cell=cell, data=mol_grid_orb.morb_grids[ispin][imo]**2)
            c.write_cube_file(output_dir + name + "_sq.cube")

        print("Wrote %s.cube, norm-1=%.4e, time %.2fs" % (name, norm_sq-1.0, (time.time()-time1)))

    if i_rel_homo <= 0 and mol_grid_orb.nspin == 2 and args.spin_dens_n != 0:
        spin_dens_contrib = np.zeros(mol_grid_orb.morb_grids[0][0].shape)
        for ispin in range(mol_grid_orb.nspin):
            spin_dens_contrib += (ispin*2-1)*mol_grid_orb.morb_grids[ispin][imo]**2
        spin_dens += spin_dens_contrib
        spin_norm = np.sum(np.abs(spin_dens_contrib))*vol_elem
        print("Abs spin density contribution: integral(|rho1-rho2|)=%.4e" % spin_norm)


if mol_grid_orb.nspin == 2 and args.spin_dens_n != 0:
    time1 = time.time()
    spin_norm = np.sum(np.abs(spin_dens))*vol_elem
    comment = "Abs. total spin density: %.4e" % spin_norm
    c = cube.Cube(title="Spin density", comment=comment, ase_atoms=mol_grid_orb.ase_atoms,
                  origin=mol_grid_orb.origin, cell=mol_grid_orb.eval_cell*np.eye(3), data=spin_dens)
    c.write_cube_file(output_dir + "spin_density.cube")
    print(comment)

