[![DOI](https://zenodo.org/badge/133041124.svg)](https://zenodo.org/badge/latestdoi/133041124)

## CP2K Scanning Probe Microscopy tools

Library and scripts to perform scanning probe microscopy simulations based on a [CP2K](https://www.cp2k.org/) calculation.

Features include:
* Processing the various output files of [CP2K](https://www.cp2k.org/), including the `.wfn` file
* Scanning Tunnelling Microscopy and Spectroscopy (STM/STS) analysis
* Fourier-Transformed STS analysis for finite cutouts of periodic systems
* Orbital hybridization analysis for adsorbed systems
* High-resolution STM (HRSTM) simulations

Most of the functionality of this library is built on top of the possibility to evaluate the Kohn-Sham orbitals encoded in the `.wfn` file on an arbitrarily defined grid. This is illustrated by the following script applied for a nanographene adsorbed on a Au(111) slab (total of 1252 atoms and 10512 electrons):

```python
import numpy as np
from cp2k_spm_tools.cp2k_grid_orbitals import Cp2kGridOrbitals

ang_2_bohr = 1.88972612463

### Create the gridding object and load the cp2k data ###
cgo = Cp2kGridOrbitals()
cgo.read_cp2k_input("./cp2k.inp")
cgo.read_xyz("./geom.xyz")
cgo.read_basis_functions("./BASIS_MOLOPT")
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

cgo.write_cube("./homo.cube", orbital_nr=0)
cgo.write_cube("./lumo.cube", orbital_nr=1)
```

The evaluated HOMO orbital in the defined region:


<img src="examples/example.png" width="600">
