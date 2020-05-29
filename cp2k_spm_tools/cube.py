"""
Routines regarding gaussian cube files
"""

import numpy as np

import ase

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602


class Cube:
    """
    Gaussian cube
    """

    def __init__(self, title=None, comment=None, ase_atoms=None,
                 origin=np.array([0.0, 0.0, 0.0]), cell=None, cell_n=None, data=None):
        """
        cell in [au] and (3x3)
        """
        self.title = title
        self.comment = comment
        self.ase_atoms = ase_atoms
        self.origin = origin
        self.cell = cell
        self.data = data
        if data is not None:
            self.cell_n = data.shape
        else:
            self.cell_n = None


    def write_cube_file(self, filename):

        positions = self.ase_atoms.positions * ang_2_bohr
        numbers = self.ase_atoms.get_atomic_numbers()

        natoms = len(self.ase_atoms)

        f = open(filename, 'w')

        if self.title is None:
            f.write(filename+'\n')
        else:
            f.write(self.title+'\n')

        if self.comment is None:
            f.write('cube\n')
        else:
            f.write(self.comment+'\n')

        dv_br = self.cell/self.data.shape

        f.write("%5d %12.6f %12.6f %12.6f\n"%(natoms, self.origin[0], self.origin[1], self.origin[2]))

        for i in range(3):
            f.write("%5d %12.6f %12.6f %12.6f\n"%(self.data.shape[i], dv_br[i][0], dv_br[i][1], dv_br[i][2]))

        for i in range(natoms):
            at_x, at_y, at_z = positions[i]
            f.write("%5d %12.6f %12.6f %12.6f %12.6f\n"%(numbers[i], 0.0, at_x, at_y, at_z))

        self.data.tofile(f, sep='\n', format='%12.6e')

        f.close()

    def read_cube_file(self, filename, read_data=True):

        f = open(filename, 'r')
        self.title = f.readline().rstrip()
        self.comment = f.readline().rstrip()

        line = f.readline().split()
        natoms = int(line[0])

        if natoms < 0:
            print("Warning: the cube %s has negative number of atoms")
            print("         meaning that there could be multiple data sections")
            print("         and each of those will have a header")
            natoms = -natoms

        self.origin = np.array(line[1:], dtype=float)

        self.cell_n = np.empty(3,dtype=int)
        self.cell = np.empty((3, 3))
        for i in range(3):
            n, x, y, z = [float(s) for s in f.readline().split()]
            self.cell_n[i] = int(n)
            self.cell[i] = n * np.array([x, y, z])

        numbers = np.empty(natoms, int)
        positions = np.empty((natoms, 3))
        for i in range(natoms):
            line = f.readline().split()
            numbers[i] = int(line[0])
            positions[i] = [float(s) for s in line[2:]]

        positions /= ang_2_bohr # convert from bohr to ang

        self.ase_atoms = ase.Atoms(numbers=numbers, positions=positions)

        if read_data:
            # Option 1: less memory usage but might be slower
            self.data = np.empty(self.cell_n[0]*self.cell_n[1]*self.cell_n[2], dtype=float)
            cursor = 0
            for i, line in enumerate(f):
                ls = line.split()
                self.data[cursor:cursor+len(ls)] = ls
                cursor += len(ls)

            # Option 2: Takes much more memory (but may be faster)
            #data = np.array(f.read().split(), dtype=float)

            self.data = self.data.reshape(self.cell_n)

        f.close()

    def get_plane_above_topmost_atom(self, height):
        """
        Returns the 2d plane above topmost atom in z direction
        height in [angstrom]
        """
        topmost_atom_z = np.max(self.ase_atoms.positions[:, 2]) # Angstrom
        plane_z = (height + topmost_atom_z) * ang_2_bohr

        plane_index = int(np.round(plane_z/self.cell[2, 2]*np.shape(self.data)[2]))
        return self.data[:, :, plane_index]

    def get_z_index(self, z_ang):
        # returns the index value for a given z coordinate in angstrom
        return int(np.round(z_ang * ang_2_bohr/self.cell[2, 2]*np.shape(self.data)[2]))

    @property
    def dv(self):
        """ in [ang] """
        return self.cell/self.cell_n/ang_2_bohr
    
    @property
    def dv_ang(self):
        """ in [ang] """
        return self.cell/self.cell_n/ang_2_bohr

    @property
    def dv_au(self):
        """ in [au] """
        return self.cell/self.cell_n
    
    @property
    def x_arr_au(self):
        """ in [au] """
        return np.arange(self.origin[0], self.origin[0] + (self.cell_n[0]-0.5)*self.dv_au[0,0], self.dv_au[0,0])
    @property
    def y_arr_au(self):
        """ in [au] """
        return np.arange(self.origin[1], self.origin[1] + (self.cell_n[1]-0.5)*self.dv_au[1,1], self.dv_au[1,1])
    @property
    def z_arr_au(self):
        """ in [au] """
        return np.arange(self.origin[2], self.origin[2] + (self.cell_n[2]-0.5)*self.dv_au[2,2], self.dv_au[2,2])

    @property
    def x_arr_ang(self):
        """ in [ang] """
        return self.x_arr_au/ang_2_bohr
    @property
    def y_arr_ang(self):
        """ in [ang] """
        return self.y_arr_au/ang_2_bohr
    @property
    def z_arr_ang(self):
        """ in [ang] """
        return self.z_arr_au/ang_2_bohr
