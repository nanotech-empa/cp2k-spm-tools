"""
Tools to perform STM/STS analysis on orbitals evaluated on grid
""" 

import os
import numpy as np
import scipy
import scipy.io
import scipy.special
import time
import copy
import sys

import re
import io
import ase
import ase.io

from .cp2k_grid_orbitals import Cp2kGridOrbitals

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

class STM:
    """
    Class to perform STM and STS analysis on gridded orbitals
    """

    def __init__(self, mpi_comm, cp2k_grid_orb):

        self.cp2k_grid_orb = cp2k_grid_orb
        self.nspin = cp2k_grid_orb.nspin
        self.mpi_rank = cp2k_grid_orb.mpi_rank
        self.mpi_size = cp2k_grid_orb.mpi_size
        self.cell_n = cp2k_grid_orb.eval_cell_n
        self.dv = cp2k_grid_orb.dv
        self.origin = cp2k_grid_orb.origin

        self.mpi_comm = mpi_comm

        self.global_morb_energies_by_rank = None
        self.global_morb_energies = None

        self.sts_isovalues = [1e-8, 1e-6]
        self.sts_heights = [3.0, 5.0]

        # TODO: Would be nice to have a datatype containing orbitals and all of their grid info
        # and also to access planes above atoms at different heights...
        self.local_orbitals = None # orbitals defined in local space for this mpi_rank
        self.local_cell_n = None
        self.local_cell = None
        self.local_origin = None

        # output maps
        self.e_arr = None
        self.cc_ldos = None
        self.cc_map = None
        self.ch_ldos = None
        self.ch_map = None

    def x_ind_per_rank(self, rank):
        # which x indexes to allocate to rank
        base_ix_per_rank = int(np.floor(self.cell_n[0] / self.mpi_size))
        extra_ix = self.cell_n[0] - base_ix_per_rank*self.mpi_size

        if rank < extra_ix:
            x_ind_start = rank*(base_ix_per_rank + 1)
            x_ind_end = (rank+1)*(base_ix_per_rank + 1)
        else:
            x_ind_start = rank*(base_ix_per_rank) + extra_ix
            x_ind_end = (rank+1)*(base_ix_per_rank) + extra_ix

        return x_ind_start, x_ind_end

    def divide_by_space(self):

        self.local_orbitals = []

        x_ind_start, x_ind_end = self.x_ind_per_rank(self.mpi_rank)
        self.local_cell_n = np.array([x_ind_end - x_ind_start, self.cell_n[1], self.cell_n[2]])
        num_spatial_points = (x_ind_end - x_ind_start) * self.cell_n[1] * self.cell_n[2]

        self.local_origin = self.origin
        self.local_origin[0] += x_ind_start*self.dv[0]
        self.local_cell = self.local_cell_n*self.dv

        for ispin in range(self.nspin):

            orbitals_per_rank = np.array([len(gme) for gme in self.global_morb_energies_by_rank[ispin]])
            total_orb = sum(orbitals_per_rank)

            for rank in range(self.mpi_size):

                # which indexes to send?
                ix_start, ix_end = self.x_ind_per_rank(rank)
                
                if self.mpi_rank == rank:
                    recvbuf = np.empty(sum(orbitals_per_rank)*num_spatial_points, dtype=self.cp2k_grid_orb.dtype)
                    print("R%d expecting counts: " % (self.mpi_rank) + str(orbitals_per_rank*num_spatial_points))
                    sys.stdout.flush()
                else:
                    recvbuf = None

                sendbuf = self.cp2k_grid_orb.morb_grids[ispin][:, ix_start:ix_end, :, :].ravel()
                print("R%d -> %d sending %d" %(self.mpi_rank, rank, len(sendbuf)))
                sys.stdout.flush()

                # Send the orbitals
                self.mpi_comm.Gatherv(sendbuf=sendbuf,
                    recvbuf=[recvbuf, orbitals_per_rank*num_spatial_points], root=rank)

                if self.mpi_rank == rank:
                    self.local_orbitals.append(recvbuf.reshape(total_orb, self.local_cell_n[0], self.local_cell_n[1], self.local_cell_n[2]))


    def gather_global_energies(self):
        self.global_morb_energies_by_rank = []
        self.global_morb_energies = []
        for ispin in range(self.nspin):
            morb_en_gather = self.mpi_comm.allgather(self.cp2k_grid_orb.morb_energies[ispin])
            self.global_morb_energies_by_rank.append(morb_en_gather)
            self.global_morb_energies.append(np.hstack(morb_en_gather))

    def gather_orbitals_from_mpi(self, to_rank, from_rank):
        self.current_orbitals = []
        for ispin in range(self.nspin):

            if self.mpi_rank == from_rank:
                self.mpi_comm.Send(self.cp2k_grid_orb.morb_grids[ispin].ravel(), to_rank)
            if self.mpi_rank == to_rank:
                num_rcv_orb = len(self.global_morb_energies[ispin][from_rank])
                cell_n = self.cp2k_grid_orb.eval_cell_n
                rcv_buf = np.empty(num_rcv_orb*cell_n[0]*cell_n[1]*cell_n[2], dtype=self.cp2k_grid_orb.dtype)
                self.mpi_comm.Recv(rcv_buf, from_rank)
                self.current_orbitals.append(rcv_buf.reshape(num_rcv_orb, cell_n[0], cell_n[1], cell_n[2]))

    ### -----------------------------------------
    ### Making pictures
    ### -----------------------------------------

    def _get_isosurf_indexes(self, data, value, interp=True):
        rev_data = data[:, :, ::-1]
        
        # Add a zero-layer at start to make sure we surpass it
        zero_layer = np.zeros((data.shape[0], data.shape[1], 1))
        rev_data = np.concatenate((zero_layer, rev_data), axis=2)
        
        nz = rev_data.shape[2]

        # Find first index that surpasses the isovalue
        indexes = np.argmax(rev_data > value, axis=2)
        # If an index is 0, no values in array are bigger than the specified
        num_surpasses = (indexes == 0).sum()
        if num_surpasses != 0:
            print("Warning: The isovalue %.3e was not reached for %d pixels" % (value, num_surpasses))
        # Set surpasses as the bottom surface
        indexes[indexes == 0] = nz - 1
        
        if interp:
            indexes_float = indexes.astype(float)
            for ix in range(np.shape(rev_data)[0]):
                for iy in range(np.shape(rev_data)[1]):
                    ind = indexes[ix, iy]
                    if ind == nz - 1:
                        continue
                    val_g = rev_data[ix, iy, ind]
                    val_s = rev_data[ix, iy, ind - 1]
                    indexes_float[ix, iy] = ind - (val_g-value)/(val_g-val_s)
            return nz - indexes_float - 1
        return nz - indexes.astype(float) - 1

    def _index_with_interpolation(self, index_arr, array):
        i = index_arr.astype(int)
        remain = index_arr-i
        iplus = np.clip(i+1, a_min=None, a_max=len(array)-1)
        return array[iplus]*remain +(1-remain)*array[i]

    def _take_2d_from_3d(self, val_arr,z_indices):
        # Get number of columns and rows in values array
        nx, ny, nz = val_arr.shape
        # Get linear indices 
        idx = z_indices + nz*np.arange(ny) + nz*ny*np.arange(nx)[:,None]
        return val_arr.flatten()[idx]

    def _index_with_interpolation_3d(self, index_arr, array_3d):
        i = index_arr.astype(int)
        remain = index_arr-i
        iplus = np.clip(i+1, a_min=None, a_max=array_3d.shape[2]-1)
        return self._take_2d_from_3d(array_3d, iplus)*remain +(1-remain)*self._take_2d_from_3d(array_3d, i)

    def gaussian(self, x, fwhm):
        sigma = fwhm/2.3548
        return np.exp(-x**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

    def gaussian_area(self, a, b, x0, fwhm):
        sigma = fwhm/2.3548
        integral = 0.5*(scipy.special.erf((b-x0)/(np.sqrt(2)*sigma)) - scipy.special.erf((a-x0)/(np.sqrt(2)*sigma)))
        return np.abs(integral)

    def local_data_plane_above_atoms(self, local_data, height):
        """
        Returns the 2d plane above topmost atom in z direction
        height in [angstrom]
        """
        topmost_atom_z = np.max(self.cp2k_grid_orb.ase_atoms.positions[:, 2]) # Angstrom
        plane_z = (height + topmost_atom_z) * ang_2_bohr
        plane_z_wrt_orig = plane_z - self.local_origin[2]

        plane_index = int(np.round(plane_z_wrt_orig/self.local_cell[2]*self.local_cell_n[2]))
        return local_data[:, :, plane_index]

    def calculate_maps(self, isovalues, heights, emin, emax, de, fwhm):
        
        self.sts_isovalues = isovalues
        self.sts_heights = heights

        if emin * emax >= 0.0:
            self.e_arr = np.arange(emin, emax+de/2, de)
            self.cc_ldos, self.cc_map, self.ch_ldos, self.ch_map = self.create_series(self.e_arr, fwhm)
        else:
            e_arr_neg = np.arange(emin, 0.0-de/2, de)
            e_arr_pos = np.arange(0.0, emax+de/2, de)
            self.e_arr = np.concatenate((e_arr_neg, e_arr_pos))

            cc_ldos_n, cc_map_n, ch_ldos_n, ch_map_n = self.create_series(e_arr_neg, fwhm)
            cc_ldos_p, cc_map_p, ch_ldos_p, ch_map_p = self.create_series(e_arr_pos, fwhm)

            self.cc_ldos = np.concatenate((cc_ldos_n, cc_ldos_p), axis=3)
            self.cc_map = np.concatenate((cc_map_n, cc_map_p), axis=3)
            self.ch_ldos = np.concatenate((ch_ldos_n, ch_ldos_p), axis=3)
            self.ch_map = np.concatenate((ch_map_n, ch_map_p), axis=3)
    
    def apply_zero_threshold(self, data_array, z_thresh):
        # apply it to every energy slice independently
        for i_series in range(data_array.shape[0]):
            for i_e in range(data_array.shape[3]):
                sli = data_array[i_series, :, :, i_e]
                slice_absmax = np.max(np.abs(sli))
                sli[np.abs(sli) < slice_absmax*z_thresh] = 0.0


    def collect_and_save_maps(self, path = "./stm.npz", to_rank = 0):
        nx_per_rank = np.array([ self.x_ind_per_rank(r)[1] - self.x_ind_per_rank(r)[0] for r in range(self.mpi_size) ])
        n_cc = len(self.sts_isovalues)
        n_ch = len(self.sts_heights)
        ny = self.cell_n[1]
        ne = len(self.e_arr)

        arr_list = [self.cc_ldos, self.cc_map, self.ch_ldos, self.ch_map]
        n1_list = [n_cc, n_cc, n_ch, n_ch]
        name_list = ["cc_sts", "cc_stm", "ch_sts", "ch_stm"]
        collected_arrays = []

        for arr, n1 in zip(arr_list, n1_list):

            if self.mpi_rank == to_rank:
                recvbuf = np.empty(sum(nx_per_rank)*ny*ne*n1, dtype=self.cp2k_grid_orb.dtype)
                print("R%d expecting counts: " % (self.mpi_rank) + str(nx_per_rank*ny*ne*n1))
            else:
                recvbuf = None
            # the swap the x axis as first, to be able to concatenate and reshape easily 
            sendbuf = arr.swapaxes(0, 1).ravel()

            self.mpi_comm.Gatherv(sendbuf=sendbuf, recvbuf=[recvbuf, nx_per_rank*ny*ne*n1], root=to_rank)
            if self.mpi_rank == to_rank:
                recvbuf = recvbuf.reshape(self.cell_n[0], n1, self.cell_n[1], ne)
                collected_arrays.append(recvbuf.swapaxes(1, 0))
        
        if self.mpi_rank == to_rank:
            save_data = dict(zip(name_list, collected_arrays))
            ### ----------------
            ### Reduce filesize
            # data type
            save_data['cc_stm'] = save_data['cc_stm'].astype(np.float16) # all values either way ~ between -2 and 8
            save_data['cc_sts'] = save_data['cc_sts'].astype(np.float32)
            save_data['ch_stm'] = save_data['ch_stm'].astype(np.float32)
            save_data['ch_sts'] = save_data['ch_sts'].astype(np.float32)
            # zero threshold
            z_thres = 1e-3
            self.apply_zero_threshold(save_data['cc_sts'], z_thres)
            self.apply_zero_threshold(save_data['ch_stm'], z_thres)
            self.apply_zero_threshold(save_data['ch_sts'], z_thres)
            ### ----------------

            # additonally add info
            save_data['isovalues'] = np.array(self.sts_isovalues) 
            save_data['heights'] = np.array(self.sts_heights) 
            save_data['e_arr'] = self.e_arr
            save_data['x_arr'] = np.arange(0.0, self.cell_n[0]*self.dv[0] + self.dv[0]/2, self.dv[0]) + self.origin[0]
            save_data['y_arr'] = np.arange(0.0, self.cell_n[1]*self.dv[1] + self.dv[1]/2, self.dv[1]) + self.origin[1]
            np.savez_compressed(path, **save_data)


    def create_series(self, e_arr, fwhm):

        rev_output = False
        if np.abs(e_arr[-1]) < np.abs(e_arr[0]):
            e_arr =  e_arr[::-1]
            rev_output = True
        de = e_arr[1] - e_arr[0]

        cc_ldos = np.zeros((len(self.sts_isovalues), self.local_cell_n[0], self.local_cell_n[1], len(e_arr)), dtype=self.cp2k_grid_orb.dtype)
        cc_map = np.zeros((len(self.sts_isovalues), self.local_cell_n[0], self.local_cell_n[1], len(e_arr)), dtype=self.cp2k_grid_orb.dtype)
        ch_ldos = np.zeros((len(self.sts_heights), self.local_cell_n[0], self.local_cell_n[1], len(e_arr)), dtype=self.cp2k_grid_orb.dtype)
        ch_map = np.zeros((len(self.sts_heights), self.local_cell_n[0], self.local_cell_n[1], len(e_arr)), dtype=self.cp2k_grid_orb.dtype)

        def index_energy(inp):
            if not inp.any():
                return None
            return np.argmax(inp)
        
        cur_charge_dens = np.zeros(self.local_cell_n)

        z_arr = np.arange(0.0, self.local_cell_n[2]*self.dv[2], self.dv[2]) + self.origin[2]
        # to angstrom and WRT to topmost atom
        z_arr /= ang_2_bohr
        z_arr -= np.max(self.cp2k_grid_orb.ase_atoms.positions[:, 2])

        for i_e, e in enumerate(e_arr):
            
            # ---------------------
            # Update charge density
            for ispin in range(self.nspin):
                # orbitals that are 
                # within +- 2.0 fwhm of e
                close_i1 = index_energy(self.global_morb_energies[ispin] > e - 2.0*fwhm)
                close_i2 = index_energy(self.global_morb_energies[ispin] > e + 2.0*fwhm)
                if close_i2 is not None:
                    close_i2 += 1
                close_energies = self.global_morb_energies[ispin][close_i1:close_i2]
                close_grids = self.local_orbitals[ispin][close_i1:close_i2]
                
                for i_m, morb_en in enumerate(close_energies):
                    
                    if e == e_arr[0]:
                        broad_factor = self.gaussian_area(0.0, e, morb_en, fwhm)
                    else:
                        broad_factor = self.gaussian_area(e-de, e, morb_en, fwhm)
                    cur_charge_dens += broad_factor*close_grids[i_m]**2

            # ---------------------
            # find surfaces corresponding to isovalues
            for i_iso, isoval in enumerate(self.sts_isovalues):
                
                i_isosurf = self._get_isosurf_indexes(cur_charge_dens, isoval, True)
                cc_map[i_iso, :, :, i_e] = self._index_with_interpolation(i_isosurf, z_arr)
                
                for ispin in range(self.nspin):
                    # orbitals that are 
                    # within +- 2.0 fwhm of e
                    close_i1 = index_energy(self.global_morb_energies[ispin] > e - 2.0*fwhm)
                    close_i2 = index_energy(self.global_morb_energies[ispin] > e + 2.0*fwhm)
                    if close_i2 is not None:
                        close_i2 += 1
                    close_energies = self.global_morb_energies[ispin][close_i1:close_i2]
                    close_grids = self.local_orbitals[ispin][close_i1:close_i2]
                    
                    for i_m, morb_en in enumerate(close_energies):
                        morb_on_surf = self._index_with_interpolation_3d(i_isosurf, close_grids[i_m]**2)
                        cc_ldos[i_iso, :, :, i_e] += self.gaussian(e - morb_en, fwhm) * morb_on_surf
            # ---------------------
            # find constant height images
            for i_h, height in enumerate(self.sts_heights):
                
                ch_map[i_h, :, :, i_e] = self.local_data_plane_above_atoms(cur_charge_dens, height)

                for ispin in range(self.nspin):
                    # orbitals that are 
                    # within +- 2.0 fwhm of e
                    close_i1 = index_energy(self.global_morb_energies[ispin] > e - 2.0*fwhm)
                    close_i2 = index_energy(self.global_morb_energies[ispin] > e + 2.0*fwhm)
                    if close_i2 is not None:
                        close_i2 += 1
                    close_energies = self.global_morb_energies[ispin][close_i1:close_i2]
                    close_grids = self.local_orbitals[ispin][close_i1:close_i2]
                    
                    for i_m, morb_en in enumerate(close_energies):
                        morb_on_surf = self.local_data_plane_above_atoms(close_grids[i_m]**2, height)
                        ch_ldos[i_h, :, :, i_e] += self.gaussian(e - morb_en, fwhm) * morb_on_surf

        if rev_output:
            return cc_ldos[:, :, :, ::-1], cc_map[:, :, :, ::-1], ch_ldos[:, :, :, ::-1], ch_map[:, :, :, ::-1]
        else:
            return cc_ldos, cc_map, ch_ldos, ch_map

        

