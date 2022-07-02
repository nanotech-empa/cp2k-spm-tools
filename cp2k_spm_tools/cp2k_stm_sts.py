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

    def __init__(self, mpi_comm, cp2k_grid_orb, p_tip_ratios):

        self.cgo = cp2k_grid_orb
        self.nspin = self.cgo.nspin
        self.mpi_rank = self.cgo.mpi_rank
        self.mpi_size = self.cgo.mpi_size
        self.cell_n = self.cgo.eval_cell_n
        self.dv = self.cgo.dv
        self.origin = self.cgo.origin
        self.global_morb_energies = self.cgo.global_morb_energies

        self.mpi_comm = mpi_comm

        self.p_tip_ratios = p_tip_ratios
        # add a check if ptip needs to be calculated, as it can use a lot of memory
        self.ptip_enabled = True
        if all(r == 0.0 for r in p_tip_ratios):
            self.ptip_enabled = False

        self.global_morb_energies_by_rank = None

        self.z_arr = np.arange(0.0, self.cell_n[2]*self.dv[2], self.dv[2]) + self.origin[2]
        # to angstrom and WRT to topmost atom
        self.z_arr /= ang_2_bohr
        self.z_arr -= np.max(self.cgo.ase_atoms.positions[:, 2])

        # TODO: Would be nice to have a datatype containing orbitals and all of their grid info
        # and also to access planes above atoms at different heights...
        self.local_orbitals = None # orbitals defined in local space for this mpi_rank
        self.local_cell_n = None
        self.local_cell = None
        self.local_origin = None

        # p-tip contribution of the orbitals defined in local space for this mpi_rank
        self.local_orbital_ptip = None
        
        # Dictionary containing all the STM/STS/ORB output
        self.series_output = {}
        """
        self.series_output = {
            's0 orb': {
                'general_info': {
                    'energies': [-0.4, -0.3, -0.1, ...],
                    'orb_indexes': [255, 256, 257, ...],
                    'HOMO': 257,
                },
                'series_info': [
                    {'type': 'ch-orb', 'height': 3.0},
                    {'type': 'ch-sts', 'height': 3.0, 'fwhm': 0.1},
                    ...,
                ],
                'series_data': [
                    [n_orb, nx, ny],
                    [n_orb, nx, ny],
                    ...,
                ]
            }
            's1 orb': {
                ...
            }
        }
        """


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
        self.local_orbital_ptip = []

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
                    recvbuf = np.empty(sum(orbitals_per_rank)*num_spatial_points, dtype=self.cgo.dtype)
                    #print("R%d expecting counts: " % (self.mpi_rank) + str(orbitals_per_rank*num_spatial_points))
                    #sys.stdout.flush()
                else:
                    recvbuf = None

                sendbuf = self.cgo.morb_grids[ispin][:, ix_start:ix_end, :, :].ravel()
                #print("R%d -> %d sending %d" %(self.mpi_rank, rank, len(sendbuf)))
                #sys.stdout.flush()

                # Send the orbitals
                self.mpi_comm.Gatherv(sendbuf=sendbuf,
                    recvbuf=[recvbuf, orbitals_per_rank*num_spatial_points], root=rank)

                if self.mpi_rank == rank:
                    self.local_orbitals.append(recvbuf.reshape(total_orb, self.local_cell_n[0], self.local_cell_n[1], self.local_cell_n[2]))
            
            if self.ptip_enabled:
                ### Calculate and divide also the p-tip contribution,
                ### as derivatives are hard to account for after dividing the orbitals in space
                p_tip_contrib  = (np.gradient(self.cgo.morb_grids[ispin], axis=1)/self.dv[0])**2
                p_tip_contrib += (np.gradient(self.cgo.morb_grids[ispin], axis=2)/self.dv[1])**2

                for rank in range(self.mpi_size):
                    ix_start, ix_end = self.x_ind_per_rank(rank)
                    if self.mpi_rank == rank:
                        recvbuf = np.empty(sum(orbitals_per_rank)*num_spatial_points, dtype=self.cgo.dtype)
                    else:
                        recvbuf = None
                    sendbuf = p_tip_contrib[:, ix_start:ix_end, :, :].ravel()
                    self.mpi_comm.Gatherv(sendbuf=sendbuf,
                        recvbuf=[recvbuf, orbitals_per_rank*num_spatial_points], root=rank)
                    if self.mpi_rank == rank:
                        self.local_orbital_ptip.append(recvbuf.reshape(total_orb, self.local_cell_n[0], self.local_cell_n[1], self.local_cell_n[2]))


    def gather_global_energies(self):
        self.global_morb_energies_by_rank = []
        self.global_morb_energies = []
        for ispin in range(self.nspin):
            morb_en_gather = self.mpi_comm.allgather(self.cgo.morb_energies[ispin])
            self.global_morb_energies_by_rank.append(morb_en_gather)
            self.global_morb_energies.append(np.hstack(morb_en_gather))

    def gather_orbitals_from_mpi(self, to_rank, from_rank):
        self.current_orbitals = []
        for ispin in range(self.nspin):

            if self.mpi_rank == from_rank:
                self.mpi_comm.Send(self.cgo.morb_grids[ispin].ravel(), to_rank)
            if self.mpi_rank == to_rank:
                num_rcv_orb = len(self.global_morb_energies[ispin][from_rank])
                cell_n = self.cgo.eval_cell_n
                rcv_buf = np.empty(num_rcv_orb*cell_n[0]*cell_n[1]*cell_n[2], dtype=self.cgo.dtype)
                self.mpi_comm.Recv(rcv_buf, from_rank)
                self.current_orbitals.append(rcv_buf.reshape(num_rcv_orb, cell_n[0], cell_n[1], cell_n[2]))

    ### -----------------------------------------
    ### Making data series
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
        topmost_atom_z = np.max(self.cgo.ase_atoms.positions[:, 2]) # Angstrom
        plane_z = (height + topmost_atom_z) * ang_2_bohr
        plane_z_wrt_orig = plane_z - self.local_origin[2]

        plane_index = int(np.round(plane_z_wrt_orig/self.local_cell[2]*self.local_cell_n[2]))
        return local_data[:, :, plane_index]

    def s_p_type_signal(self, i_spin, i_mo, p_tip_ratio):
        if p_tip_ratio == 0.0:
            return self.local_orbitals[i_spin][i_mo]**2
        else:
            return (1.0 - p_tip_ratio) * self.local_orbitals[i_spin][i_mo]**2 + p_tip_ratio * self.local_orbital_ptip[i_spin][i_mo]

    def build_stm_series(self, e_arr, fwhms, heights, isovalues, p_tip_ratio=0.0):

        #print("Create series: " + str(e_arr))

        rev_output = False
        if np.abs(e_arr[-1]) < np.abs(e_arr[0]):
            e_arr =  e_arr[::-1]
            rev_output = True

        cc_ldos = np.zeros((len(fwhms), len(isovalues), self.local_cell_n[0], self.local_cell_n[1], len(e_arr)), dtype=self.cgo.dtype)
        cc_map = np.zeros((len(fwhms), len(isovalues), self.local_cell_n[0], self.local_cell_n[1], len(e_arr)), dtype=self.cgo.dtype)
        ch_ldos = np.zeros((len(fwhms), len(heights), self.local_cell_n[0], self.local_cell_n[1], len(e_arr)), dtype=self.cgo.dtype)
        ch_map = np.zeros((len(fwhms), len(heights), self.local_cell_n[0], self.local_cell_n[1], len(e_arr)), dtype=self.cgo.dtype)

        def index_energy(inp):
            if not inp.any():
                return None
            return np.argmax(inp)

        for i_fwhm, fwhm in enumerate(fwhms):
        
            cur_charge_dens = np.zeros(self.local_cell_n)

            last_e = 0.0

            for i_e, e in enumerate(e_arr):
                # ---------------------
                # Contributing orbitals in the energy range since last energy value
                close_indexes = []
                for ispin in range(self.nspin):
                    e1 = np.min([last_e, e])
                    e2 = np.max([last_e, e])
                    close_i1 = index_energy(self.global_morb_energies[ispin] > e1 - 2.0*fwhm)
                    close_i2 = index_energy(self.global_morb_energies[ispin] > e2 + 2.0*fwhm)
                    if close_i1 is None:
                        close_i1 = 0
                    if close_i2 is None:
                        close_i2 = len(self.global_morb_energies[ispin])
                    close_indexes.append(np.arange(close_i1, close_i2))

                # ---------------------
                # Update charge density
                for ispin in range(self.nspin):
                    for i_mo in close_indexes[ispin]:
                        morb_en = self.global_morb_energies[ispin][i_mo]
                        broad_factor = self.gaussian_area(last_e, e, morb_en, fwhm)
                        cur_charge_dens += broad_factor*self.s_p_type_signal(ispin, i_mo, p_tip_ratio)

                # ---------------------
                # find surfaces corresponding to isovalues
                for i_iso, isoval in enumerate(isovalues):
                    
                    i_isosurf = self._get_isosurf_indexes(cur_charge_dens, isoval, True)
                    cc_map[i_fwhm, i_iso, :, :, i_e] = self._index_with_interpolation(i_isosurf, self.z_arr)
                    
                    for ispin in range(self.nspin):
                        for i_mo in close_indexes[ispin]:
                            morb_en = self.global_morb_energies[ispin][i_mo]
                            morb_on_surf = self._index_with_interpolation_3d(
                                i_isosurf,
                                self.s_p_type_signal(ispin, i_mo, p_tip_ratio)
                            )
                            cc_ldos[i_fwhm, i_iso, :, :, i_e] += self.gaussian(e - morb_en, fwhm) * morb_on_surf
                
                # ---------------------
                # find constant height images
                for i_h, height in enumerate(heights):
                    
                    ch_map[i_fwhm, i_h, :, :, i_e] = self.local_data_plane_above_atoms(cur_charge_dens, height)

                    for ispin in range(self.nspin):
                        for i_mo in close_indexes[ispin]:
                            morb_en = self.global_morb_energies[ispin][i_mo]
                            morb_on_plane = self.local_data_plane_above_atoms(
                                self.s_p_type_signal(ispin, i_mo, p_tip_ratio),
                                height
                            )
                            ch_ldos[i_fwhm, i_h, :, :, i_e] += self.gaussian(e - morb_en, fwhm) * morb_on_plane
                last_e = e

        if rev_output:
            return cc_ldos[:, :, :, :, ::-1], cc_map[:, :, :, :, ::-1], ch_ldos[:, :, :, :, ::-1], ch_map[:, :, :, :, ::-1]
        else:
            return cc_ldos, cc_map, ch_ldos, ch_map

    
    def calculate_stm_maps(self, fwhms, isovalues, heights, energies, series_name='stm', i_series_offset=0):
        """
        STM maps for specified energies, isovalues, heights, fwhms.
        """

        e_arr = np.sort(energies)
        emin = e_arr[0]
        emax = e_arr[-1]

        if series_name not in self.series_output:
            number_of_series = (len(heights) + len(isovalues)) * len(fwhms) * 2 * len(self.p_tip_ratios)
            self.series_output[series_name] = {
                'general_info': {'energies': e_arr},
                'series_info': [],
                'series_data': np.zeros((number_of_series, len(e_arr), self.local_cell_n[0], self.local_cell_n[1]), dtype=self.cgo.dtype)
            }
        
        i_series_counter = i_series_offset

        for p_tip_ratio in self.p_tip_ratios:

            if emin * emax >= 0.0:
                cc_sts, cc_stm, ch_sts, ch_stm = self.build_stm_series(e_arr, fwhms, heights, isovalues, p_tip_ratio)
            else:
                e_arr_neg = e_arr[e_arr <= 0.0]
                e_arr_pos = e_arr[e_arr > 0.0]

                cc_sts_n, cc_stm_n, ch_sts_n, ch_stm_n = self.build_stm_series(e_arr_neg, fwhms, heights, isovalues, p_tip_ratio)
                cc_sts_p, cc_stm_p, ch_sts_p, ch_stm_p = self.build_stm_series(e_arr_pos, fwhms, heights, isovalues, p_tip_ratio)

                cc_sts = np.concatenate((cc_sts_n, cc_sts_p), axis=4)
                cc_stm = np.concatenate((cc_stm_n, cc_stm_p), axis=4)
                ch_sts = np.concatenate((ch_sts_n, ch_sts_p), axis=4)
                ch_stm = np.concatenate((ch_stm_n, ch_stm_p), axis=4)

            # Move energy axis to position 2 (Other axes remain in their original order)
            cc_sts = np.moveaxis(cc_sts, 4, 2)
            cc_stm = np.moveaxis(cc_stm, 4, 2)
            ch_sts = np.moveaxis(ch_sts, 4, 2)
            ch_stm = np.moveaxis(ch_stm, 4, 2)

            ### ------------------------------------
            ### Save the data to self.series_output


            for i_fwhm, fwhm in enumerate(fwhms):

                for i_h, h in enumerate(heights):
                    self.series_output[series_name]['series_info'].append({
                        'type': 'const-height sts',
                        'height': h,
                        'fwhm': fwhm,
                        'p_tip_ratio': p_tip_ratio,
                    })
                    self.series_output[series_name]['series_data'][i_series_counter, :, :, :] = ch_sts[i_fwhm, i_h, :, :, :]
                    i_series_counter += 1

                for i_isov, isov in enumerate(isovalues):
                    self.series_output[series_name]['series_info'].append({
                        'type': 'const-isovalue sts',
                        'isovalue': isov,
                        'fwhm': fwhm,
                        'p_tip_ratio': p_tip_ratio,
                    })
                    self.series_output[series_name]['series_data'][i_series_counter, :, :, :] = cc_sts[i_fwhm, i_isov, :, :, :]
                    i_series_counter += 1

                for i_h, h in enumerate(heights):
                    self.series_output[series_name]['series_info'].append({
                        'type': 'const-height stm',
                        'height': h,
                        'fwhm': fwhm,
                        'p_tip_ratio': p_tip_ratio,
                    })
                    self.series_output[series_name]['series_data'][i_series_counter, :, :, :] = ch_stm[i_fwhm, i_h, :, :, :]
                    i_series_counter += 1

                for i_isov, isov in enumerate(isovalues):
                    self.series_output[series_name]['series_info'].append({
                        'type': 'const-isovalue stm',
                        'isovalue': isov,
                        'fwhm': fwhm,
                        'p_tip_ratio': p_tip_ratio,
                    })
                    self.series_output[series_name]['series_data'][i_series_counter, :, :, :] = cc_stm[i_fwhm, i_isov, :, :, :]
                    i_series_counter += 1
            

    def collect_local_grid(self, local_arr, global_shape, to_rank = 0):
        """
        local_arr needs to have x as first axis
        """

        size_except_x = np.prod(global_shape[1:])

        nx_per_rank = np.array([ self.x_ind_per_rank(r)[1] - self.x_ind_per_rank(r)[0] for r in range(self.mpi_size) ])

        if self.mpi_rank == to_rank:
            recvbuf = np.empty(sum(nx_per_rank)*size_except_x, dtype=self.cgo.dtype)
            #print("R%d expecting counts: " % (self.mpi_rank) + str(nx_per_rank*size_except_x))
        else:
            recvbuf = None
            
        sendbuf = local_arr.ravel()

        self.mpi_comm.Gatherv(sendbuf=sendbuf, recvbuf=[recvbuf, nx_per_rank*size_except_x], root=to_rank)
        if self.mpi_rank == to_rank:
            recvbuf = recvbuf.reshape(global_shape)
        return recvbuf

    def collect_series_maps(self):

        nx = self.cell_n[0]
        ny = self.cell_n[1]

        for label, ser in self.series_output.items():

            ne = len(ser['general_info']['energies'])
            n_ser = len(ser['series_info'])

            ser['series_data'] = self.collect_local_grid(ser['series_data'].swapaxes(0, 2), np.array([nx, ne, n_ser, ny]))

            if self.mpi_rank == 0:
                ser['series_data'] = ser['series_data'].swapaxes(2, 0)

    def apply_zero_threshold(self, data_array, zero_thresh):
        # apply it to every energy slice independently
        for i_series in range(data_array.shape[0]):
            for i_e in range(data_array.shape[1]):
                sli = data_array[i_series, i_e, :, :]
                slice_absmax = np.max(np.abs(sli))
                sli[np.abs(sli) < slice_absmax*zero_thresh] = 0.0

    def collect_and_save_stm_maps(self, path = "./stm.npz"):

        self.collect_series_maps()
        
        if self.mpi_rank == 0:

            save_data = {
                'stm_general_info': self.series_output['stm']['general_info'],
                'stm_series_info': self.series_output['stm']['series_info'],
                'stm_series_data': self.series_output['stm']['series_data'].astype(np.float32),
            }

            ### ----------------
            ### Reduce filesize further by zero threshold
            zero_thresh = 1e-3
            self.apply_zero_threshold(save_data['stm_series_data'], zero_thresh)
            ### ----------------

            # additionally add info
            save_data['stm_general_info']['x_arr'] = np.arange(0.0, self.cell_n[0]*self.dv[0] + self.dv[0]/2, self.dv[0]) + self.origin[0]
            save_data['stm_general_info']['y_arr'] = np.arange(0.0, self.cell_n[1]*self.dv[1] + self.dv[1]/2, self.dv[1]) + self.origin[1]
            np.savez_compressed(path, **save_data)

        # Reset, otherwise can cause problems (more versatile to NOT reset, though)
        self.series_output = {}

    ### -----------------------------------------
    ### Orbital analysis and export
    ### -----------------------------------------

    def create_orb_series(self, orb_indexes, height_list=[], isoval_list=[], fwhm_list=[]):
        """
        orb_indexes - orbital indexes w.r.t. to "ref_index_glob" for both spin channels
        """
        orb_indexes_wrt_data_start = []
        n_orb = len(orb_indexes)

        ens_list = []

        # Setup the series' dictionaries
        for i_spin in range(self.nspin):
            label = 's%d_orb' % i_spin
            self.series_output[label] = {}

            orb_indexes_wrt_data_start.append(list(np.array(orb_indexes) + self.cgo.cwf.ref_index_glob))

        # Orbital / energy info
        for i_spin in range(self.nspin):
            label = 's%d_orb' % i_spin

            ens_list.append(self.global_morb_energies[i_spin][orb_indexes_wrt_data_start[i_spin]])
            physical_index_list = self.cgo.cwf.global_morb_indexes[i_spin][orb_indexes_wrt_data_start[i_spin]]

            self.series_output[label]['general_info'] = {
                'energies': ens_list[-1],
                'orb_indexes': physical_index_list,
                'homo': physical_index_list[self.cgo.i_homo_glob[i_spin]],
                'spin': i_spin,
            }
            self.series_output[label]['series_info'] = []
        
        number_of_series = (
            len(height_list) # just s-type WFN LDOS
            + len(height_list) * len(self.p_tip_ratios) # p-type WFN ch-signals
            + len(isoval_list) * len(self.p_tip_ratios) # p-type WFN cc-signals
            + len(height_list) * 2*len(fwhm_list) * len(self.p_tip_ratios) # p-type ch sts & stm signals
            + len(isoval_list) * 2*len(fwhm_list) * len(self.p_tip_ratios) # p-type cc sts & stm signals
        )

        # Orbital series
        for i_spin in range(self.nspin):
            label = 's%d_orb' % i_spin

            self.series_output[label]['series_data'] = np.zeros(
                (number_of_series, n_orb, self.local_cell_n[0], self.local_cell_n[1]),
            dtype=self.cgo.dtype)

            i_series_counter = 0

            ### constant-height orbital series

            for i_h, h in enumerate(height_list):

                ## orbital wavefunction

                # series info
                self.series_output[label]['series_info'].append({
                    'type': 'const-height orbital',
                    'height': h,
                })

                # series data
                i_orb_count = 0
                for i_mo in orb_indexes_wrt_data_start[i_spin]:
                    self.series_output[label]['series_data'][i_series_counter, i_orb_count, :, :] = (
                        self.local_data_plane_above_atoms(self.local_orbitals[i_spin][i_mo], h)
                    )
                    i_orb_count += 1
                i_series_counter += 1

                ## orbital ch-signal with the different tips
                for i_p, p_tip_ratio in enumerate(self.p_tip_ratios):

                    # series info
                    self.series_output[label]['series_info'].append({
                        'type': 'const-height orbital sts',
                        'height': h,
                        'p_tip_ratio':  p_tip_ratio,
                    })

                    # series data
                    i_orb_count = 0
                    for i_mo in orb_indexes_wrt_data_start[i_spin]:
                        s_p_data = self.s_p_type_signal(i_spin, i_mo, p_tip_ratio)
                        self.series_output[label]['series_data'][i_series_counter, i_orb_count, :, :] = (
                            self.local_data_plane_above_atoms(s_p_data, h)
                        )
                        i_orb_count += 1
                    i_series_counter += 1

            ### constant-isovalue orbital series

            for i_isov, isov in enumerate(isoval_list):
                for i_p, p_tip_ratio in enumerate(self.p_tip_ratios):

                    # series info
                    self.series_output[label]['series_info'].append({
                        'type': 'const-isovalue orbital sts',
                        'isovalue': isov,
                        'p_tip_ratio':  p_tip_ratio,
                    })

                    # series data
                    i_orb_count = 0
                    for i_mo in orb_indexes_wrt_data_start[i_spin]:
                        i_isosurf = self._get_isosurf_indexes(
                            self.s_p_type_signal(i_spin, i_mo, p_tip_ratio),
                            isov, True
                        )
                        self.series_output[label]['series_data'][i_series_counter, i_orb_count, :, :] = (
                            self._index_with_interpolation(i_isosurf, self.z_arr)
                        )
                        i_orb_count += 1
                    i_series_counter += 1

            self.calculate_stm_maps(
                fwhm_list, isoval_list, height_list, ens_list[i_spin], series_name='s%d_orb'%i_spin, i_series_offset=i_series_counter
            )


    def collect_and_save_orb_maps(self, path = "./orb.npz"):

        self.collect_series_maps()
        
        if self.mpi_rank == 0:

            save_data = {
                's0_orb_general_info': self.series_output['s0_orb']['general_info'],
                's0_orb_series_info': self.series_output['s0_orb']['series_info'],
                's0_orb_series_data': self.series_output['s0_orb']['series_data'].astype(np.float32),
            }

            ### ----------------
            ### Reduce filesize further by zero threshold
            zero_thresh = 1e-3
            self.apply_zero_threshold(save_data['s0_orb_series_data'], zero_thresh)
            ### ----------------

            # additionally add info
            save_data['s0_orb_general_info']['x_arr'] = np.arange(0.0, self.cell_n[0]*self.dv[0] + self.dv[0]/2, self.dv[0]) + self.origin[0]
            save_data['s0_orb_general_info']['y_arr'] = np.arange(0.0, self.cell_n[1]*self.dv[1] + self.dv[1]/2, self.dv[1]) + self.origin[1]

            if "s1_orb" in self.series_output:
                save_data['s1_orb_general_info'] = self.series_output['s1_orb']['general_info']
                save_data['s1_orb_series_info'] = self.series_output['s1_orb']['series_info']
                save_data['s1_orb_series_data'] = self.series_output['s1_orb']['series_data'].astype(np.float32)

                self.apply_zero_threshold(save_data['s1_orb_series_data'], zero_thresh)
                save_data['s1_orb_general_info']['x_arr'] = np.arange(0.0, self.cell_n[0]*self.dv[0] + self.dv[0]/2, self.dv[0]) + self.origin[0]
                save_data['s1_orb_general_info']['y_arr'] = np.arange(0.0, self.cell_n[1]*self.dv[1] + self.dv[1]/2, self.dv[1]) + self.origin[1]

            np.savez_compressed(path, **save_data)

        # Reset, otherwise can cause problems (more versatile to NOT reset, though)
        self.series_output = {}
