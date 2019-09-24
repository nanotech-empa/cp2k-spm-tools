# @author Hillebrand, Fabian
# @date   2019

import numpy as np

from .interpolator import Interpolator

ang2bohr   = 1.88972612546
ev2hartree = 0.03674930814


class Cp2kGridMatrix:
    """
    Class that provides a wrapper for Cp2kGridOrbtials such that they can be
    evaluated on an arbitrary grid using interpolation. Scaled derivatives are 
    also accessible.

    This structure provides access to the wave functions via bracket operators.
    The following structure is provided: [itunnel,ispin,iene][derivative,x,y,z].
    Note that this evaluation may be performed lazily.
    """

    def __init__(self, cp2k_grid_orb, eval_region, tip_pos, norbs_tip, wn,
        mpi_rank=0, mpi_size=1, mpi_comm=None):
        self.mpi_rank = mpi_rank
        self.mpi_size = mpi_size
        self.mpi_comm = mpi_comm
        self._norbs_tip = norbs_tip
        self._decay = (2*wn*ev2hartree)**0.5
        self._grids = tip_pos
        # Complete energy and wave function matrix when using no MPI
        self._ene = cp2k_grid_orb.morb_energies
        self._wfn_matrix = cp2k_grid_orb.morb_grids
        self._wfn_dim = np.shape(self.wfn_matrix[0])[1:]
        self._eval_region = eval_region
        self._eval_region_local = self.eval_region
        self._reg_grid = None
        self._ase_atoms = cp2k_grid_orb.ase_atoms
        self._dv = cp2k_grid_orb.dv / ang2bohr
        self._nspin = cp2k_grid_orb.nspin
        # Storage for computed wave function on evaluation grid
        self._wfn = None
        self._divide_flag = False

    def _get_slice(cls, wm, ids, axis):
        """
        Retrieves a slice specified by a tuple (imin, imax) (inclusive)
        from an assumed-periodic array. The slice is along the specified
        axis.
        """
        dim = np.shape(wm)[axis]
        slice_lower = [slice(None)]*wm.ndim
        slice_upper = [slice(None)]*wm.ndim
        if ids[0] < 0:
            slice_lower[axis] = slice(ids[0],None)
            if ids[1] >= dim:
                slice_upper[axis] = slice(None,ids[1]-dim+1)
                # Case: Spill-over on both sides.
                return np.concatenate([wm[tuple(slice_lower)],wm,
                    wm[tuple(slice_upper)]], axis=axis)
            else:
                slice_upper[axis] = slice(None,ids[1]+1)
                # Case: Spill-over only on lower side.
                return np.concatenate([wm[tuple(slice_lower)],
                    wm[tuple(slice_upper)]], axis=axis)
        elif ids[1] >= dim:
            slice_lower[axis] = slice(ids[0],None)
            slice_upper[axis] = slice(None, ids[1]-dim+1)
            # Case: Spill-over only on upper side.
            return np.concatenate([wm[tuple(slice_lower)],
                wm[tuple(slice_upper)]], axis=axis)
        else:
            slice_lower[axis] = slice(ids[0],ids[1]+1)
            # Case: No spill-over
            return wm[tuple(slice_lower)]

    def divide(self):
        """
        Divides the grid obritals to the different MPI ranks along the
        x-direction rather than the energies.

        Note that this function overwrites the stored wave function matrix
        and must be called.
        """
        if self._divide_flag:
            raise AssertionError("Tried to call Cp2kGridMatrix.divide() twice!")
        self._divide_flag = True
        # Index range needed by this rank (inclusive)
        isx = np.array([\
              np.floor(min([np.min(pos[0]-self.eval_region[0][0])
                            for pos in self.grids]) / self._dv[0]),
              np.ceil( max([np.max(pos[0]-self.eval_region[0][0])
                            for pos in self.grids]) / self._dv[0]),
              ], dtype=int)
        isy = np.array([\
              np.floor(min([np.min(pos[1]-self.eval_region[1][0])
                            for pos in self.grids]) / self._dv[1]),
              np.ceil( max([np.max(pos[1]-self.eval_region[1][0])
                            for pos in self.grids]) / self._dv[1]),
              ], dtype=int)
        if self.mpi_comm is None:
            wfn_matrix = [self._get_slice(self.wfn_matrix[ispin],isx,1) for ispin in \
                range(self.nspin)]
        else:
            ene = []
            wfn_matrix = []
            # Distribute and gather energies and wave functions on MPI ranks
            for ispin in range(self.nspin):
                # Gather energies
                ene_separated = self.mpi_comm.allgather(self.ene[ispin])
                nene_by_rank = np.array([len(val) for val in ene_separated])
                ene.append(np.hstack(ene_separated))
                # Indices needed for the tip position on MPI rank
                isx_all = self.mpi_comm.allgather(isx)
                # Dimension of local grid for wave function matrix
                wfn_dim_local = (isx[1]-isx[0]+1,)+self.wfn_dim[1:]
                npoints = np.product(wfn_dim_local)
                # Gather the necessary stuff
                for rank in range(self.mpi_size):
                    if self.mpi_rank == rank:
                        recvbuf = np.empty(len(ene[ispin])*npoints)
                    else:
                        recvbuf = None
                    sendbuf = np.array(self._get_slice(self.wfn_matrix[ispin],
                        isx_all[rank],1),order='C').ravel()
                    self.mpi_comm.Gatherv(sendbuf=sendbuf, recvbuf=[recvbuf,
                        nene_by_rank*npoints], root=rank)
                    if self.mpi_rank == rank:
                        wfn_matrix.append(recvbuf.reshape(
                            (len(ene[ispin]),) + wfn_dim_local))
            self._ene = ene
        self._wfn_matrix = []
        for ispin in range(self.nspin):
            self._wfn_matrix.append(self._get_slice(wfn_matrix[ispin],isy,2))
        # Set evaluation region for this MPI rank
        self._eval_region_local[0] = self.eval_region[0][0]+isx*self._dv[0]
        self._eval_region_local[1] = self.eval_region[1][0]+isy*self._dv[1]
        self._wfn_dim = np.shape(self.wfn_matrix[0])[1:]
        self._reg_grid = ( \
            np.linspace(self.eval_region_local[0][0],
                        self.eval_region_local[0][1],self.wfn_dim[0]),
            np.linspace(self.eval_region_local[1][0],
                        self.eval_region_local[1][1],self.wfn_dim[1]),
            np.linspace(self.eval_region_local[2][0],
                        self.eval_region_local[2][1],self.wfn_dim[2]))

    ### ------------------------------------------------------------------------
    ### Access operators
    ### ------------------------------------------------------------------------

    @property
    def ene(self):
        """ List of energies per spin in eV. """
        return self._ene
    @property
    def wfn_matrix(self):
        """ Local underlying matrix defined on regular grid in atomic units. """
        return self._wfn_matrix
    @property
    def eval_region_local(self):
        """ Limits of evaluation grid for local non-relaxed tip scan in
            Angstrom. 
        """
        return self._eval_region_local
    @property
    def eval_region(self):
        """ Limits of evaluation grid for complete non-relaxed tip scan in 
            Angstrom. 
        """
        return self._eval_region
    @property
    def reg_grid(self):
        """ Regular grid where local wave function matrix is defined on in 
            Angstrom. 
        """
        return self._reg_grid
    @property
    def grids(self):
        """ Local evaluation grids for tip positions in Angstrom. """
        return self._grids
    @property
    def ase_atoms(self):
        """ ASE atom object. """
        return self._ase_atoms
    @property
    def wfn_dim(self):
        """ Dimension of local grid for wave function matrix. """
        return self._wfn_dim
    @property
    def grid_dim(self):
        """ Dimension of local evaluation grids for tip positions. """
        return np.shape(self._grids[0])[1:]
    @property
    def norbs_tip(self):
        """ Number of tip orbitals (0 for s, 1 for p). """
        return self._norbs_tip
    @property
    def decay(self):
        """ Decay constant in atomic units. """
        return self._decay
    @property
    def nspin(self):
        """ Number of spins for sample. """
        return self._nspin

    def __getitem__(self, itupel):
        igrid, ispin, iene = itupel
        # Check if already evaluated
        if self._wfn is not None \
            and self._cgrid == igrid \
            and self._cspin == ispin \
            and self._cene == iene:
            return self._wfn
        # Storage container
        self._wfn = np.empty(((self.norbs_tip+1)**2,)+self.grid_dim)
        # Create interpolator
        interp = Interpolator(self.reg_grid, self.wfn_matrix[ispin][iene])
        self._wfn[0] = interp(*self.grids[igrid])
        if self.norbs_tip:
            self._wfn[1] = interp.gradient(*self.grids[igrid],2) / ang2bohr / self.decay
            self._wfn[2] = interp.gradient(*self.grids[igrid],3) / ang2bohr / self.decay
            self._wfn[3] = interp.gradient(*self.grids[igrid],1) / ang2bohr / self.decay
        self._cgrid, self._cspin, self._cene = itupel
        return self._wfn
