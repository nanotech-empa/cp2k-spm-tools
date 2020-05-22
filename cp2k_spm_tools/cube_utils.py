"""
Routines regarding gaussian cube files
"""

import numpy as np

import ase

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

def calc_ioniz_potential(hartree_cube, homo_en_hrt):
    """
    In case of a slab, for accurate result, the dipole correction should be enabled.
    This case, however, is not handled.
    Only free molecules will have accurate result.

    Hartree cube in [Hrt]
    homo_en_hrt in [Hrt] with the same reference
    """

    # average the hartree pot in x and y directions and keep z
    hartree_1d = np.mean(hartree_cube.data, axis=(0,1))
    # take the maximum value of the averaged hartree potential
    # above the molecule (3 ang) until the end of the box
    z_ind_start = hartree_cube.get_z_index(np.max(hartree_cube.ase_atoms.positions[:, 2]) + 3.0)
    vacuum_level = np.max(hartree_1d[z_ind_start:])

    return (vacuum_level - homo_en_hrt) * hart_2_ev



def add_artif_core_charge(charge_dens_cube):
    """
    This function adds an artificial large core charge such that
    the Bader analysis is more robust. The total charge does not remain accurate.
    """

    cell = np.diag(charge_dens_cube.cell)
    cell_n = charge_dens_cube.cell_n
    origin = charge_dens_cube.origin
    dv_au = cell/cell_n

    x = np.linspace(0.0, cell[0], cell_n[0]) + origin[0]
    y = np.linspace(0.0, cell[1], cell_n[1]) + origin[1]
    z = np.linspace(0.0, cell[2], cell_n[2]) + origin[2]

    for at in charge_dens_cube.ase_atoms:
        if at.number == 1:
            # No core density for H
            continue
        p = at.position * ang_2_bohr

        if (p[0] < np.min(x) - 0.5 or p[0] > np.max(x) + 0.5 or
                p[1] < np.min(y) - 0.5 or p[1] > np.max(y) + 0.5 or
                p[2] < np.min(z) - 0.5 or p[2] > np.max(z) + 0.5):
            continue
            
        # Distance of the **Center** of each voxel to the atom 
        x_grid, y_grid, z_grid = np.meshgrid(x - p[0] - dv_au[0]/2, y - p[1] - dv_au[1]/2, z - p[2] - dv_au[2]/2, indexing='ij')
        r_grid = np.sqrt(x_grid**2 + y_grid**2 + z_grid**2)
        x_grid = None
        y_grid = None
        z_grid = None


        core_charge = at.number - at.number % 8 # not exact...

        r_hat = 0.8
        h_hat = 20.0
        hat_func = (h_hat-h_hat*r_grid/r_hat)
        hat_func[r_grid > r_hat] = 0.0
        charge_dens_cube.data = np.maximum(hat_func, charge_dens_cube.data)
