# @author Hillebrand, Fabian
# @date   2019


import os
import numpy as np

import scipy
import scipy.io
import scipy.interpolate
import scipy.ndimage

import time
import copy
import sys
import os
# Include directory
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../")

import re
import io
import ase
import ase.io

from atomistic_tools.cp2k_grid_orbitals import Cp2kGridOrbitals

from mpi4py import MPI

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602


class PPSTMGridOrbitals(Cp2kGridOrbitals):
    """
    Replaces Gaussian with an exponential depending on the decay constant.
    This exists primarily to debug and should not be used (or with great care).
    A particular nasty thing is that the extrapolation plane is used as the work
    function (in eV).

    @attention This is for debugging and may not run correctly.
    """

    ### ---------------------------------------------------------------------------
    ### Methods directly related to putting stuff on grids
    ### ---------------------------------------------------------------------------

    def _spherical_harmonic_grid(self, l, m, x_grid, y_grid, z_grid, r):
        """
        Evaluates the spherical harmonics (times r^l) with some unknown normalization
        (source: Carlo's Fortran code)
        """
        c = (2.0/np.pi)**(3.0/4.0)

        # s orbitals
        if (l, m) == (0, 0):
            return c

        # p orbitals
        elif (l, m) == (1, -1):
            return c*2.0*y_grid / r
        elif (l, m) == (1, 0):
            return c*2.0*z_grid / r
        elif (l, m) == (1, 1):
            return c*2.0*x_grid / r 

        # d orbitals
        elif (l, m) == (2, -2):
            return c*4.0*x_grid*y_grid / r**2
        elif (l, m) == (2, -1):
            return c*4.0*y_grid*z_grid / r**2
        elif (l, m) == (2, 0):
            return c*2.0/np.sqrt(3)*(2*z_grid**2-x_grid**2-y_grid**2) / r**2
        elif (l, m) == (2, 1):
            return c*4.0*z_grid*x_grid / r**2
        elif (l, m) == (2, 2):
            return c*2.0*(x_grid**2-y_grid**2) / r**2

        # f orbitals
        elif (l, m) == (3, -3):
            return c*np.sqrt(8/3)*y_grid*(3*x_grid**2-y_grid**2) / r**3
        elif (l, m) == (3, -2):
            return c*8.0*x_grid*y_grid*z_grid / r**3
        elif (l, m) == (3, -1):
            return c*np.sqrt(8/5)*y_grid*(4*z_grid**2-x_grid**2-y_grid**2) / r**3
        elif (l, m) == (3, 0):
            return c*4.0/np.sqrt(15.0)*z_grid*(2.0*z_grid**2-3.0*x_grid**2-3.0*y_grid**2) / r**3
        elif (l, m) == (3, 1):
            return c*np.sqrt(8/5)*x_grid*(4*z_grid**2-x_grid**2-y_grid**2) / r**3
        elif (l, m) == (3, 2):
            return c*4.0*z_grid*(x_grid**2-y_grid**2) / r**3
        elif (l, m) == (3, 3):
            return c*np.sqrt(8/3)*x_grid*(x_grid**2-3.0*y_grid**2) / r**3

        print("No spherical harmonic found for l=%d, m=%d" % (l, m))
        return 0

    def calc_morbs_in_region(self, dr_guess,
                            x_eval_region = None,
                            y_eval_region = None,
                            z_eval_region = None,
                            eval_cutoff = 14.0,
                            reserve_extrap = 0.0,
                            print_info = True):
        """ 
        Puts the molecular orbitals onto a specified grid
        Arguments:
        dr_guess -- spatial discretization step [ang], real value will change for every axis due to rounding  
        x_eval_region -- x evaluation (min, max) in [au]. If min == max, then evaluation only works on a plane.
                        If set, no PBC applied in direction and also no eval_cutoff.
                        If left at None, the whole range of the cell is taken and PBCs are applied.
        eval_cutoff -- cutoff in [ang] for orbital evaluation if eval_region is None
        """

        time1 = time.time()

        dr_guess *= ang_2_bohr
        eval_cutoff *= ang_2_bohr
        reserve_extrap *= ang_2_bohr
        # NOTE Watch out with reserve_extrap!
        kappa = np.sqrt(2.0 / hart_2_ev * reserve_extrap)

        global_cell_n = (np.round(self.cell/dr_guess)).astype(int)
        self.dv = self.cell / global_cell_n

        # Define local grid for orbital evaluation
        # and convenient PBC implementation
        eval_regions = [x_eval_region, y_eval_region, z_eval_region]
        loc_cell_arrays = []
        mid_ixs = np.zeros(3, dtype=int)
        loc_cell_n = np.zeros(3, dtype=int)
        eval_cell_n = np.zeros(3, dtype=int)
        self.origin = np.zeros(3)
        for i in range(3):
            if eval_regions[i] is None:
                # Define range in i direction with 0.0 at index mid_ixs[i]
                loc_arr = np.arange(0, eval_cutoff, self.dv[i])
                mid_ixs[i] = int(len(loc_arr)/2)
                loc_arr -= loc_arr[mid_ixs[i]]
                loc_cell_arrays.append(loc_arr)
                eval_cell_n[i] = global_cell_n[i]
                self.origin[i] = 0.0
            else:
                # Define the specified range in direction i
                v_min, v_max = eval_regions[i]
                ### TODO: Probably should use np.arange to have exactly matching dv in the local grid... ###
                loc_cell_arrays.append(np.linspace(v_min, v_max, int(np.round((v_max-v_min)/self.dv[i]))+1))
                mid_ixs[i] = -1
                eval_cell_n[i] = len(loc_cell_arrays[i])
                self.origin[i] = v_min
                
            loc_cell_n[i] = len(loc_cell_arrays[i])

        loc_cell_grids = np.meshgrid(loc_cell_arrays[0], loc_cell_arrays[1], loc_cell_arrays[2], indexing='ij')

        # Some info
        if print_info:
            print("Global cell: ", global_cell_n)
            print("Eval cell: ", eval_cell_n)
            print("local cell: ", loc_cell_n)
            print("---- Setup: %.4f" % (time.time() - time1))

        time_radial_calc = 0.0
        time_spherical = 0.0
        time_loc_glob_add = 0.0
        time_loc_lmorb_add = 0.0

        nspin = len(self.morb_composition)

        num_morbs = []
        morb_grids_local = []
        self.morb_grids = []

        ext_z_n = int(np.round(reserve_extrap/self.dv[2]))

        for ispin in range(nspin):
            num_morbs.append(len(self.morb_composition[ispin][0][0][0][0]))
            self.morb_grids.append(np.zeros((num_morbs[ispin], eval_cell_n[0], eval_cell_n[1], eval_cell_n[2] + ext_z_n), dtype=self.dtype))
            morb_grids_local.append(np.zeros((num_morbs[ispin], loc_cell_n[0], loc_cell_n[1], loc_cell_n[2]), dtype=self.dtype))

        self.eval_cell_n = np.array([eval_cell_n[0], eval_cell_n[1], eval_cell_n[2] + ext_z_n])
        self.eval_cell = self.eval_cell_n * self.dv
        self.last_calc_iz = eval_cell_n[2] - 1

        for i_at in range(len(self.ase_atoms)):
            #elem = self.ase_atoms[i_at].symbol
            kind = self.atom_kinds[i_at]
            pos = self.ase_atoms[i_at].position * ang_2_bohr

            # how does the position match with the grid?
            int_shift = (pos/self.dv).astype(int)
            frac_shift = pos/self.dv - int_shift
            origin_diff = int_shift - mid_ixs

            # Shift the local grid such that origin is on the atom
            rel_loc_cell_grids = []
            for i, loc_grid in enumerate(loc_cell_grids):
                if eval_regions[i] is None:
                    rel_loc_cell_grids.append(loc_grid - frac_shift[i]*self.dv[i])
                else:
                    rel_loc_cell_grids.append(loc_grid - pos[i])

            r_vec_2 = rel_loc_cell_grids[0]**2 + \
                    rel_loc_cell_grids[1]**2 + \
                    rel_loc_cell_grids[2]**2
            r_vec = r_vec_2**0.5

            for ispin in range(nspin):
                morb_grids_local[ispin].fill(0.0)

            for i_set, bset in enumerate(self.basis_sets[kind]):
                for i_shell, shell in enumerate(bset):
                    l = shell[0]
                    if l > 1:
                        continue
                    es = shell[1]
                    cs = shell[2]

                    # Calculate the radial part of the atomic orbital
                    time2 = time.time()
                    radial_part = np.exp(-kappa*r_vec)
                    time_radial_calc += time.time() - time2

                    for i_orb, m in enumerate(range(-l, l+1, 1)):
                        time2 = time.time()
                        atomic_orb = radial_part*self._spherical_harmonic_grid(l, m,
                                                                        rel_loc_cell_grids[0],
                                                                        rel_loc_cell_grids[1],
                                                                        rel_loc_cell_grids[2], r_vec)
                        time_spherical += time.time() - time2
                        time2 = time.time()

                        for i_spin in range(nspin):
                            #print("---------------")
                            #print(i_spin, len(self.morb_composition))
                            #print(i_at, len(self.morb_composition[i_spin]))
                            #print(i_set, len(self.morb_composition[i_spin][i_at]))
                            #print(i_shell, len(self.morb_composition[i_spin][i_at][i_set]))
                            #print(i_orb, len(self.morb_composition[i_spin][i_at][i_set][i_shell]))
                            #print("---------------")

                            coef_arr = self.morb_composition[i_spin][i_at][i_set][i_shell][i_orb]

                            for i_mo in range(num_morbs[i_spin]):
                                morb_grids_local[i_spin][i_mo] += coef_arr[i_mo]*atomic_orb

                            # slow:
                            #morb_grids_local += np.outer(coef_arr, atomic_orb).reshape(
                            #                 num_morbs, loc_cell_n[0], loc_cell_n[1], loc_cell_n[2])
                        time_loc_lmorb_add += time.time() - time2

            time2 = time.time()
            for i_spin in range(nspin):
                for i_mo in range(num_morbs[i_spin]):
                    if ext_z_n == 0:
                        self._add_local_to_global_grid(
                            morb_grids_local[i_spin][i_mo],
                            self.morb_grids[i_spin][i_mo],
                            origin_diff,
                            wrap=(mid_ixs != -1))
                    else:
                        self._add_local_to_global_grid(
                            morb_grids_local[i_spin][i_mo],
                            self.morb_grids[i_spin][i_mo][:, :, :-ext_z_n],
                            origin_diff,
                            wrap=(mid_ixs != -1))
            time_loc_glob_add += time.time() - time2

        if print_info:
            print("---- Radial calc time : %4f" % time_radial_calc)
            print("---- Spherical calc time : %4f" % time_spherical)
            print("---- Loc -> loc_morb time : %4f" % time_loc_lmorb_add)
            print("---- loc_morb -> glob time : %4f" % time_loc_glob_add)
            print("---- Total time: %.4f"%(time.time() - time1))

