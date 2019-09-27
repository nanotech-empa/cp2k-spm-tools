# @author Hillebrand, Fabian
# @date   2019

import numpy as np
from mpi4py import MPI


def read_PPPos(filename):
    """
    Loads the positions obtained from the probe particle model and returns
    them as a [3,X,Y,Z]-array together with the lVec (which defines the 
    non-relaxed grid scan in terms of the lowest probe particle (oxygen)).
    The units are in Angstrom.
    """
    disposX = np.transpose(np.load(filename+"_x.npy")).copy()
    disposY = np.transpose(np.load(filename+"_y.npy")).copy()
    disposZ = np.transpose(np.load(filename+"_z.npy")).copy()
    lvec = (np.load(filename+"_vec.npy"))
    # Stack arrays to form 4-dimensional array of size [3, noX, noY, noZ]
    dispos = (disposX,disposY,disposZ)
    return dispos, lvec

def apply_bounds(grid, lVec):
    """
    Assumes periodicity and restricts grid positions to a box in x- and
    y-direction.
    """
    dx = lVec[1,0]-lVec[0,0]
    grid[0][grid[0] >= lVec[1,0]] -= dx
    grid[0][grid[0] < lVec[0,0]]  += dx
    dy = lVec[2,1]-lVec[0,1]
    grid[1][grid[1] >= lVec[2,1]] -= dy
    grid[1][grid[1] < lVec[0,1]]  += dy
    return grid

def read_tip_positions(files, shift, dx, mpi_rank=0, mpi_size=1, mpi_comm=None):
    """
    Reads the tip positions and determines the necessary grid orbital evaluation
    region for the sample via the tip positions.

    pos_local       List with the tip positions for this rank.
    dim_pos         Number of all tip positions along each axis.
    eval_region     Limits of evaluation grid encompassing all tip positions.
    lVec            4x3 matrix defining non-relaxed tip positions 
                    (with respect to the oxygen).
    """
    # Only reading on one rank, could be optimized but not the bottleneck
    if mpi_rank == 0:
        pos_all = []
        for filename in files:
            positions, lVec = read_PPPos(filename)
            pos_all.append(positions)
        dim_pos = np.shape(pos_all[0])[1:]
        # Metal tip (needed only for rotation, no tunnelling considered)
        pos_all.insert(0, np.mgrid[ \
            lVec[0,0]:lVec[0,0]+lVec[1,0]:dim_pos[0]*1j, \
            lVec[0,1]:lVec[0,1]+lVec[2,1]:dim_pos[1]*1j, \
            lVec[0,2]+shift:lVec[0,2]+lVec[3,2]+shift:dim_pos[2]*1j])
        # Evaluation region for sample (x,y periodic)
        xmin = lVec[0,0]
        xmax = lVec[0,0]+lVec[1,0]
        ymin = lVec[0,1]
        ymax = lVec[0,1]+lVec[2,1]
        zmin = min([np.min(pos[2]) for pos in pos_all[1:]])-dx/2
        zmax = max([np.max(pos[2]) for pos in pos_all[1:]])+dx/2
        eval_region_wfn = np.array([[xmin,xmax], [ymin,ymax], [zmin,zmax]])
        # No MPI
        if mpi_comm is None:
            return pos_all, dim_pos, eval_region_wfn, lVec
    else:
        pos_all = [[None]*3]*(len(files)+1)
        lVec = None
        dim_pos = None
        eval_region_wfn = None
    # Broadcast small things
    lVec = mpi_comm.bcast(lVec, root=0)
    dim_pos = mpi_comm.bcast(dim_pos, root=0)
    eval_region_wfn = mpi_comm.bcast(eval_region_wfn, root=0)
    # Divide up tip positions along x-axis
    all_x_ids = np.array_split(np.arange(dim_pos[0]), mpi_size)
    lengths = [len(all_x_ids[rank])*np.product(dim_pos[1:]) 
        for rank in range(mpi_size)]
    offsets = [all_x_ids[rank][0]*np.product(dim_pos[1:]) 
        for rank in range(mpi_size)]
    # Prepare storage and then scatter grids
    pos_local = [np.empty((3,len(all_x_ids[mpi_rank]),)+dim_pos[1:])
        for i in range(len(files)+1)]
    for gridIdx in range(len(files)+1):
        for axis in range(3):
            mpi_comm.Scatterv([pos_all[gridIdx][axis], lengths, offsets, 
                MPI.DOUBLE], pos_local[gridIdx][axis], root=0)
    return pos_local, dim_pos, eval_region_wfn, lVec

def create_tip_positions(eval_region, dx, mpi_rank=0, mpi_size=1, mpi_comm=None):
    """
    Creates uniform grids for tip positions. Due to the structure of the code,
    this returns a tuple with twice the same grid. Rotations are not supported.

    pos_local       List with the tip positions for this rank.
    dim_pos         Number of all tip positions along each axis.
    eval_region     Limits of evaluation grid encompassing all tip positions.
    lVec            4x3 matrix defining non-relaxed tip positions 
                    (with respect to the oxygen).
    """
    eval_region = np.reshape(eval_region,(3,2))
    lVec = np.zeros((4,3))
    lVec[0,0] = eval_region[0,0]
    lVec[1,0] = eval_region[0,1]-eval_region[0,0]
    lVec[0,1] = eval_region[1,0]
    lVec[2,1] = eval_region[1,1]-eval_region[1,0]
    lVec[0,2] = eval_region[2,0]
    lVec[3,2] = eval_region[2,1]-eval_region[2,0] 
    dim_pos = (int(lVec[1,0]/dx+1), int(lVec[2,1]/dx+1), int(lVec[3,2]/dx+1))
    # True spacing
    dxyz = [lVec[i+1,i] / (dim_pos[i]-1) for i in range(3)]
    # Divide tip positions before building grids
    all_x_ids = np.array_split(np.arange(dim_pos[0]), mpi_size)
    start = lVec[0,0]+dxyz[0]*all_x_ids[mpi_rank][0]
    end = lVec[0,0]+dxyz[0]*all_x_ids[mpi_rank][-1]
    grid = np.mgrid[ \
        start:end:len(all_x_ids[mpi_rank])*1j,
        lVec[0,1]:lVec[0,1]+lVec[2,1]:dim_pos[1]*1j,
        lVec[0,2]:lVec[0,2]+lVec[3,2]:dim_pos[2]*1j]
    # Increase eval_region to account for non-periodic axis
    eval_region[2,0] -= dxyz[-1] / 2
    eval_region[2,1] += dxyz[-1] / 2
    # Positions emulating apex atom + metal tip
    pos_local = [grid, grid]
    return pos_local, dim_pos, eval_region, lVec
