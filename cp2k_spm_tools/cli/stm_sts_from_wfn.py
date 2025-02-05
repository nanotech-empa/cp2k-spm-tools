#!/usr/bin/env python
import argparse
import sys
import time

import numpy as np
from mpi4py import MPI

import cp2k_spm_tools.cp2k_grid_orbitals as cgo
import cp2k_spm_tools.cp2k_stm_sts as css
from cp2k_spm_tools import common, cube_utils
from cp2k_spm_tools.cube import Cube


def main():
    ang_2_bohr = 1.0 / 0.52917721067
    hart_2_ev = 27.21138602

    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    parser = argparse.ArgumentParser(description="Puts the CP2K orbitals on grid and calculates STM.")

    ### ----------------------------------------------------------------------
    ### Input and output files
    parser.add_argument(
        "--cp2k_input_file", metavar="FILENAME", required=True, help="CP2K input of the SCF calculation."
    )
    parser.add_argument(
        "--basis_set_file", metavar="FILENAME", required=True, help="File containing the used basis sets."
    )
    parser.add_argument("--xyz_file", metavar="FILENAME", required=True, help=".xyz file containing the geometry.")
    parser.add_argument(
        "--wfn_file", metavar="FILENAME", required=True, help="Restart file containing the final wavefunction."
    )
    parser.add_argument(
        "--hartree_file", metavar="FILENAME", required=True, help="Cube file containing the hartree potential."
    )
    parser.add_argument(
        "--output_file", metavar="FILENAME", default="./stm.npz", help="File, where to save the STM/STS output"
    )
    parser.add_argument(
        "--orb_output_file", metavar="FILENAME", default="./orb.npz", help="File, where to save the orbital output"
    )
    ### ----------------------------------------------------------------------
    ### Parameters for putting orbitals on grid
    parser.add_argument(
        "--eval_region", type=str, nargs=6, metavar="X", required=True, help=common.eval_region_description
    )
    parser.add_argument("--dx", type=float, metavar="DX", required=True, help="Spatial step for the grid (angstroms).")
    parser.add_argument(
        "--eval_cutoff",
        type=float,
        metavar="D",
        default=16.0,
        help=("Size of the region around the atom where each orbital is evaluated (only used for 'G' region)."),
    )
    parser.add_argument(
        "--extrap_extent",
        type=float,
        metavar="H",
        default=4.0,
        required=True,
        help="The extent of the extrapolation region. (angstrom)",
    )
    ### ----------------------------------------------------------------------
    ### Gas phase analysis parameters - image at orbital energies
    parser.add_argument("--n_homo", type=int, metavar="N", default=0, help="Number of HOMO orbitals to analyse.")
    parser.add_argument("--n_lumo", type=int, metavar="N", default=0, help="Number of LUMO orbitals to analyse.")
    parser.add_argument(
        "--orb_heights",
        nargs="*",
        type=float,
        metavar="H",
        help="List of heights for constant height orbital pictures (wrt topmost atom).",
    )
    parser.add_argument(
        "--orb_isovalues",
        nargs="*",
        type=float,
        metavar="C",
        help="List of charge density isovalues for constant current orbital pictures",
    )
    parser.add_argument(
        "--orb_fwhms",
        nargs="*",
        type=float,
        default=[0.02],
        help="Full width at half maximum for orbital STS gaussian broadening. (eV)",
    )
    ### ----------------------------------------------------------------------
    ### Slab system analysis parameters - images at specified energies
    ###
    ### Option 1: continuous selection
    parser.add_argument(
        "--energy_range",
        nargs=3,
        type=float,
        metavar="E",
        help="Selection of STM/STS energy values based on a range: min, max and differential.",
    )
    ###
    ### Option 2: discrete selection
    parser.add_argument(
        "--energies", nargs="*", type=float, metavar="E", help="Discrete energies where to run the STM/STS."
    )
    ### ----------------------------------------------------------------------
    ### Parameters for STM/STS series
    parser.add_argument(
        "--heights",
        nargs="*",
        type=float,
        metavar="H",
        help="List of heights for constant height STM pictures (wrt topmost atom).",
    )
    parser.add_argument(
        "--isovalues",
        nargs="*",
        type=float,
        metavar="C",
        help="List of charge density isovalues for constant current STM pictures.",
    )
    parser.add_argument(
        "--fwhms",
        nargs="*",
        type=float,
        default=[0.1],
        help="Full width at half maximum for STS gaussian broadening. (eV)",
    )
    ### ----------------------------------------------------------------------
    ### P - tip ratio list
    parser.add_argument(
        "--p_tip_ratios",
        nargs="+",
        type=float,
        metavar="P",
        default=[0.0],
        help=("List of p character of the STM tip: 0.0 correspondsto fully s-type and 1.0 to fully p-type tip"),
    )
    ### ----------------------------------------------------------------------

    time0 = time.time()

    ### ------------------------------------------------------
    ### Parse args for only one rank to suppress duplicate stdio
    ### ------------------------------------------------------

    args = None
    args_success = False
    try:
        if mpi_rank == 0:
            args = parser.parse_args()
            args_success = True
    finally:
        args_success = comm.bcast(args_success, root=0)

    if not args_success:
        print(mpi_rank, "exiting")
        exit(0)

    args = comm.bcast(args, root=0)

    ### ------------------------------------------------------
    ### Energy values for STM/STS
    ### ------------------------------------------------------

    if args.energies is not None:
        e_arr = np.array(args.energies)
    elif args.energy_range is not None:
        emin, emax, de = args.energy_range
        e_arr = np.arange(emin, emax + de / 2, de)
    else:
        e_arr = None

    max_fwhm = np.max(args.fwhms)
    if e_arr is not None:
        sel_emin = np.min(e_arr) - 2.0 * max_fwhm
        sel_emax = np.max(e_arr) + 2.0 * max_fwhm
    else:
        sel_emin = None
        sel_emax = None

    ### ------------------------------------------------------
    ### Evaluate orbitals on the real-space grid
    ### ------------------------------------------------------

    cp2k_grid_orb = cgo.Cp2kGridOrbitals(mpi_rank, mpi_size, mpi_comm=comm, single_precision=True)
    cp2k_grid_orb.read_cp2k_input(args.cp2k_input_file)
    cp2k_grid_orb.read_xyz(args.xyz_file)
    cp2k_grid_orb.center_atoms_to_cell()
    cp2k_grid_orb.read_basis_functions(args.basis_set_file)
    cp2k_grid_orb.load_restart_wfn_file(
        args.wfn_file, emin=sel_emin, emax=sel_emax, n_occ=args.n_homo, n_virt=args.n_lumo
    )

    print("R%d/%d: loaded wfn, %.2fs" % (mpi_rank, mpi_size, (time.time() - time0)))
    sys.stdout.flush()
    time1 = time.time()

    eval_reg = common.parse_eval_region_input(args.eval_region, cp2k_grid_orb.ase_atoms, cp2k_grid_orb.cell)

    # --------
    # Make sure extrap extent is compatible with heights
    atoms_max_z = np.max(cp2k_grid_orb.ase_atoms.positions[:, 2])
    eval_z_above_atoms = eval_reg[2][1] - atoms_max_z
    extrap_extent = args.extrap_extent
    for hs in [args.orb_heights, args.heights]:
        if hs is not None:
            if np.max(hs) - eval_z_above_atoms > extrap_extent:
                print("Increasing extrap. extent to be compatible with heights.")
                extrap_extent = np.max(hs) - eval_z_above_atoms
    # --------

    cp2k_grid_orb.calc_morbs_in_region(
        args.dx,
        x_eval_region=eval_reg[0],
        y_eval_region=eval_reg[1],
        z_eval_region=eval_reg[2],
        pbc=(True, True, False),
        reserve_extrap=extrap_extent,
        eval_cutoff=args.eval_cutoff,
    )

    print("R%d/%d: evaluated wfn, %.2fs" % (mpi_rank, mpi_size, (time.time() - time1)))
    sys.stdout.flush()
    time1 = time.time()

    ### ------------------------------------------------------
    ### Extrapolate orbitals
    ### ------------------------------------------------------

    hart_cube = Cube()
    hart_cube.read_cube_file(args.hartree_file)
    extrap_plane_z = eval_reg[2][1] / ang_2_bohr - np.max(cp2k_grid_orb.ase_atoms.positions[:, 2])
    hart_plane = hart_cube.get_plane_above_topmost_atom(extrap_plane_z) - cp2k_grid_orb.ref_energy / hart_2_ev

    cp2k_grid_orb.extrapolate_morbs(hart_plane=hart_plane)

    print("R%d/%d: extrapolated wfn, %.2fs" % (mpi_rank, mpi_size, (time.time() - time1)))
    sys.stdout.flush()
    time1 = time.time()

    ### ------------------------------------------------------
    ### Calculate the ionization potential (just for output)
    ### ------------------------------------------------------

    if mpi_rank == 0:
        # NB: currently only accurate for isolated molecules
        if cp2k_grid_orb.nspin == 1:
            homo_en = cp2k_grid_orb.global_morb_energies[0][cp2k_grid_orb.i_homo_glob[0]]
        else:
            homo_en = np.max(
                [
                    cp2k_grid_orb.global_morb_energies[0][cp2k_grid_orb.i_homo_glob[0]],
                    cp2k_grid_orb.global_morb_energies[1][cp2k_grid_orb.i_homo_glob[1]],
                ]
            )
        ion_pot = cube_utils.find_vacuum_level_naive(hart_cube) - (homo_en + cp2k_grid_orb.ref_energy)
        print("IONIZATION POTENIAL (eV): %.6f (accurate only for isolated molecules)" % ion_pot)
        sys.stdout.flush()

    ### ------------------------------------------------------
    ### Set up STM object
    ### ------------------------------------------------------

    stm = css.STM(mpi_comm=comm, cp2k_grid_orb=cp2k_grid_orb, p_tip_ratios=args.p_tip_ratios)
    stm.gather_global_energies()
    stm.divide_by_space()

    ### ------------------------------------------------------
    ### Run STM-STS analysis for orbitals
    ### ------------------------------------------------------

    orb_heights = args.orb_heights if args.orb_heights is not None else []
    orb_isovalues = args.orb_isovalues if args.orb_isovalues is not None else []
    orb_fwhms = args.orb_fwhms if args.orb_fwhms is not None else []

    if len(orb_fwhms) != 0 and (len(orb_heights) != 0 or len(orb_isovalues) != 0):
        orbital_list = list(range(-args.n_homo + 1, args.n_lumo + 1))

        stm.create_orb_series(orbital_list, orb_heights, orb_isovalues, orb_fwhms)

        stm.collect_and_save_orb_maps(path=args.orb_output_file)

    ### ------------------------------------------------------
    ### Run STM-STS analysis for general energies
    ### ------------------------------------------------------

    heights = args.heights if args.heights is not None else []
    isovalues = args.isovalues if args.isovalues is not None else []
    fwhms = args.fwhms if args.fwhms is not None else []

    if e_arr is not None and len(fwhms) != 0 and (len(heights) != 0 or len(isovalues) != 0):
        stm.calculate_stm_maps(fwhms, isovalues, heights, e_arr)

        stm.collect_and_save_stm_maps(path=args.output_file)

    print("R%d/%d: finished, total time: %.2fs" % (mpi_rank, mpi_size, (time.time() - time0)))


if __name__ == "__main__":
    main()
