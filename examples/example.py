from cp2k_spm_tools.cp2k_grid_orbitals import Cp2kGridOrbitals

### Create the gridding object and load the cp2k data ###
cgo = Cp2kGridOrbitals()
cgo.read_cp2k_input("./cp2k.inp")
cgo.read_xyz("./geom.xyz")
cgo.read_basis_functions("./BASIS_MOLOPT")
cgo.load_restart_wfn_file("./PROJ-RESTART.wfn", n_occ=2, n_virt=2) 

### Evaluate the orbitals in the specific region ###
cgo.calc_morbs_in_region(
    dr_guess = 0.15,                        # grid spacing
    x_eval_region = None,                   # take whole cell in x
    y_eval_region = [0.0, cgo.cell[1]/2],   # half cell in y
    z_eval_region = [36.0, 44.0],           # around the molecule in z
)

cgo.write_cube("./homo-1.cube", orbital_nr=-1)
cgo.write_cube("./homo.cube", orbital_nr=0)
