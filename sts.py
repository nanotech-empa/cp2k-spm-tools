
import os
import numpy as np
import time
import copy
import sys
import argparse
import matplotlib
import matplotlib.pyplot as plt

import cp2k_utilities as cu

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

parser = argparse.ArgumentParser(
    description='Produces LDOS data from a CP2K SCF calculation \
                 along x direction on a plane normal to z direction.')
parser.add_argument(
    '--cp2k_input',
    metavar='FILENAME',
    required=True,
    help='CP2K input of the SCF calculation.')
parser.add_argument(
    '--cp2k_output',
    metavar='FILENAME',
    required=True,
    help='CP2K output of the SCF calculation.')
parser.add_argument(
    '--basis_file',
    metavar='FILENAME',
    required=True,
    help='File containing the used basis sets.')
parser.add_argument(
    '--xyz_file',
    metavar='FILENAME',
    required=True,
    help='.xyz file containing the geometry.')
parser.add_argument(
    '--restart_file',
    metavar='FILENAME',
    required=True,
    help='Restart file containing the final wavefunction.')

parser.add_argument(
    '--emin',
    type=float,
    metavar='E',
    required=True,
    help='Lowest energy value for selecting orbitals.')
parser.add_argument(
    '--emax',
    type=float,
    metavar='E',
    required=True,
    help='Highest energy value for selecting orbitals.')
parser.add_argument(
    '--height',
    type=float,
    metavar='H',
    required=True,
    help='Distance of the LDOS plane from highest C atom.')
parser.add_argument(
    '--dx',
    type=float,
    metavar='DX',
    required=True,
    help='Spatial step for the grid, strongly recommended to \
          pick a divisor of the lattice parameter.')

parser.add_argument(
    '--de',
    type=float,
    metavar='DE',
    required=True,
    help='Energy grid spacing for the resulting LDOS.')
parser.add_argument(
    '--fwhm',
    nargs='+',
    type=float,
    metavar='FWHM',
    required=True,
    help='Full width half maximums for the broadening of each orbital.')

args = parser.parse_args()

time0 = time.time()
elem_basis_names, cell = cu.read_cp2k_input(args.cp2k_input)
print("Read cp2k input: %.3f" % (time.time()-time0))

time1 = time.time()
fermi = cu.read_fermi_from_cp2k_out(args.cp2k_output)
print("Read cp2k out: %.3f" % (time.time()-time1))

time1 = time.time()
at_positions, at_elems = cu.read_atoms(args.xyz_file)
print("Read xyz: %.3f" % (time.time()-time1))

time1 = time.time()
basis_sets = cu.read_basis_functions(args.basis_file, elem_basis_names)
print("Read basis sets: %.3f" % (time.time()-time1))

time1 = time.time()
morb_composition, morb_energies, morb_occs = cu.load_restart_wfn_file(
        args.restart_file, args.emin, args.emax, fermi)
print("Read restart: %.3f" % (time.time()-time1))

geom_folder, geom_file = os.path.split(args.xyz_file)
geom_folder += "/"
geom_name, ext = os.path.splitext(geom_file)

height = args.height # Plane distance in z direction from topmost atom; angstroms

emin = args.emin
emax = args.emax

# Define real space grid
# Cp2k chooses close to 0.08 angstroms (?)
step = args.dx # Good to choose a factor of the lattice parameter 4.26
step *= ang_2_bohr
cell_n = (np.round(cell/step)).astype(int)

# !!! NB: Redefine cell such that step stays the same (and a factor of lattice param)
cell = cell_n*step

cu.center_atoms_to_cell(at_positions, cell)

### ---------------------------------------------------------------------
### MORB CALCULATION
### ---------------------------------------------------------------------

# Define the plane
time1 = time.time()

carb_positions = at_positions[np.array(at_elems)[:, 0] == 'C']

plane_z = np.max(carb_positions[:, 2]) + height*ang_2_bohr

dv = cell[0:2]/cell_n[0:2]
x_arr = np.arange(0, cell[0], dv[0])
y_arr = np.arange(0, cell[1], dv[1])
x_grid, y_grid = np.meshgrid(x_arr, y_arr, indexing='ij')

# Define small grid for orbital evaluation
# and convenient PBC implementation
loc_cell = np.array([10.0,  10.0])*ang_2_bohr
x_arr_loc = np.arange(0, loc_cell[0], dv[0])
y_arr_loc = np.arange(0, loc_cell[1], dv[1])
loc_cell_n = np.array([len(x_arr_loc), len(y_arr_loc)])
# Define it such that the origin is somewhere
# in the middle but exactly on a grid point
mid_ixs = (loc_cell_n/2).astype(int)
x_arr_loc -= x_arr_loc[mid_ixs[0]]
y_arr_loc -= y_arr_loc[mid_ixs[1]]
x_grid_loc, y_grid_loc = np.meshgrid(x_arr_loc, y_arr_loc, indexing='ij')

# Some info
print("Main cell:   ", cell, cell_n)
print("Local plane: ", loc_cell, loc_cell_n)

morb_planes = 0 # release memory from previous run (needed in some rare cases)
morb_planes = [np.zeros(cell_n[0:2]) for _ in range(len(morb_composition))]

print("---- Setup: %.4f" % (time.time() - time1))

time_radial_calc = 0.0
time_spherical = 0.0
time_loc_glob_add = 0.0

for i_at in range(len(at_positions)):
        elem = at_elems[i_at][0]
        pos = at_positions[i_at]

        # how does the position match with the grid?
        int_shift = (pos[0:2]/dv).astype(int)
        frac_shift = pos[0:2]/dv - int_shift

        # Shift the local grid such that origin is on the atom
        x_grid_rel_loc = x_grid_loc - frac_shift[0]*dv[0]
        y_grid_rel_loc = y_grid_loc - frac_shift[1]*dv[1]

        z_rel = plane_z - pos[2]

        r_vec_2 = x_grid_rel_loc**2 + y_grid_rel_loc**2 + z_rel**2

        for i_shell, shell in enumerate(basis_sets[elem]):
            l = shell[0]
            es = shell[1]
            cs = shell[2]

            # Calculate the radial part of the atomic orbital
            time2 = time.time()
            radial_part = np.zeros(loc_cell_n)
            for e, c in zip(es, cs):
                radial_part += c*np.exp(-1.0*e*r_vec_2)
            time_radial_calc += time.time() - time2

            for i, m in enumerate(range(-l, l+1, 1)):
                time2 = time.time()
                atomic_orb = radial_part*cu.spherical_harmonic_grid(l, m,
                                                                 x_grid_rel_loc,
                                                                 y_grid_rel_loc,
                                                                 z_rel)
                time_spherical += time.time() - time2

                for i_mo in range(len(morb_composition)):
                    i_set = 0 # SHOULD START SUPPORTING MULTIPLE SET BASES AT SOME POINT
                    coef = morb_composition[i_mo][i_at][i_set][i_shell][i]

                    # Add the atomic orbital on the local grid to the global grid
                    origin_diff = int_shift - mid_ixs
                    time2 = time.time()
                    cu.add_local_to_global_grid(coef*atomic_orb, morb_planes[i_mo], origin_diff)
                    time_loc_glob_add += time.time() - time2


print("---- Radial calc time : %4f" % time_radial_calc)
print("---- Spherical calc time : %4f" % time_spherical)
print("---- Loc -> glob time : %4f" % time_loc_glob_add)
print("---- Total time: %.4f"%(time.time() - time1))

### ----------------------------------------------------------------------
### Make a plot of some orbitals
### ----------------------------------------------------------------------

i_homo = 0
i_lumo = 0

for i, en in enumerate(morb_energies):
    if en > 0.0:
        i_lumo = i
        i_homo = i - 1
        break

select = [i_homo - 1, i_homo, i_lumo, i_lumo + 1]

sel_morbs = np.zeros((cell_n[0], 4*cell_n[1]))

for i, i_mo in enumerate(select):
    sel_morbs[:, i*cell_n[1]:(i+1)*cell_n[1]] = morb_planes[i_mo]
    print(i_mo, "energy", morb_energies[i_mo])

y_arr_inc = np.arange(0, 4*cell[1], dv[1])

x_grid_inc, y_grid_inc = np.meshgrid(x_arr, y_arr_inc, indexing='ij')

max_val = np.max(sel_morbs)

plt.figure(figsize=(12, int(cell[1]/cell[0]*12*4)))
plt.pcolormesh(x_grid_inc, y_grid_inc, sel_morbs, vmax=max_val, vmin=-max_val, cmap='seismic') # seismic bwr
plt.savefig(geom_folder+geom_name+'.png', dpi=300)
plt.clf()

### ----------------------------------------------------------------
### Calculate the LDOS based on the orbitals
### ----------------------------------------------------------------

#de = 0.01 # eV
#de = 0.002 # eV
de = args.de

e_arr = np.arange(emin, emax+de, de)

x_arr_ang = x_arr / ang_2_bohr

x_e_grid, e_grid = np.meshgrid(x_arr_ang, e_arr, indexing='ij')

def calculate_ldos(de, fwhm, broad_type):

    def lorentzian(x):
        gamma = 0.5*fwhm
        return gamma/(np.pi*(x**2+gamma**2))

    def gaussian(x):
        sigma = fwhm/2.3548
        return np.exp(-x**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

    pldos = np.zeros((cell_n[0], len(e_arr)))

    for i_mo, morb_plane in enumerate(morb_planes):
        en = morb_energies[i_mo]
        avg_morb = np.mean(morb_plane**2, axis=1)

        if broad_type == 'l':
            morb_ldos_broad = np.outer(avg_morb, lorentzian(e_arr - en))
        else:
            morb_ldos_broad = np.outer(avg_morb, gaussian(e_arr - en))

        pldos += morb_ldos_broad

    return pldos

#geom_name = file_xyz.split('/')[-1].split('.')[0]
#ofolder = "/home/kristjan/local_work/cnt_molog_ldos/step0.0852/"

#fwhm_arr = [0.01, 0.02, 0.05] # eV
fwhm_arr = args.fwhm

for fwhm in fwhm_arr:
    for broad_type in ['g']:

        pldos = calculate_ldos(de, fwhm, broad_type)

        ofname = geom_folder + "ldos_%s_h%.1f_fwhm%.2f%s.txt" % (geom_name, height, fwhm, broad_type)
        header = "geom: %s; height(ang): %.1f; fwhm(eV): %.4f; broad: %s; " % (geom_name, height, fwhm, broad_type) + \
                 "xmin(ang): %.5f; xmax(ang): %.5f; nx: %d; " % (np.min(x_arr_ang), np.max(x_arr_ang), len(x_arr_ang)) + \
                 "emin(eV): %.5f; emax(eV): %.5f; ne: %d" % (np.min(e_arr), np.max(e_arr), len(e_arr))
        np.savetxt(ofname, pldos, header=header, fmt="%.4e")

print("Completed in %.1f s" % (time.time()-time0))
