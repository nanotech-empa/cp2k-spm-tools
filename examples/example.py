import numpy as np
from cp2k_spm_tools.cp2k_grid_orbitals import Cp2kGridOrbitals

ang_2_bohr = 1.88972612463

### Create the gridding object and load the cp2k data ###
cgo = Cp2kGridOrbitals()
cgo.read_cp2k_input("./cp2k.inp")
cgo.read_xyz("./geom.xyz")
cgo.read_basis_functions("./BASIS_MOLOPT")
# load "homo-1" to "lumo+1" from the .wfn file
cgo.load_restart_wfn_file("./PROJ-RESTART.wfn", n_occ=2, n_virt=2) 

### Define the region where to evaluate the orbitals ###
x_eval_reg = None # take whole cell in x direction
y_eval_reg = [0.0, cgo.cell[1]/2] # half cell in y direction

# in z, take +- 2 angstrom from the furthest-extending carbon atoms
c_z = [c.position[2] for c in cgo.ase_atoms if c.symbol=='C']
z_eval_reg = [
    (np.min(c_z) - 2.0) * ang_2_bohr,
    (np.max(c_z) + 2.0) * ang_2_bohr
]

### Evaluate the orbitals in the defined region ###
cgo.calc_morbs_in_region(
    dr_guess = 0.15,
    x_eval_region = x_eval_reg,
    y_eval_region = y_eval_reg,
    z_eval_region = z_eval_reg)

cgo.write_cube("./homo-1.cube", orbital_nr=-1)
cgo.write_cube("./homo.cube", orbital_nr=0)
cgo.write_cube("./lumo.cube", orbital_nr=1)
cgo.write_cube("./lumo+1.cube", orbital_nr=2)
