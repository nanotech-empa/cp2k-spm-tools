#!/usr/bin/env python
import argparse
import time

from mpi4py import MPI

from cp2k_spm_tools import common


def main():
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    parser = argparse.ArgumentParser(description="Runs bond order analysis based on Bader basins.")

    parser.add_argument(
        "--cp2k_input_file", metavar="FILENAME", required=True, help="CP2K input of the SCF calculation."
    )
    parser.add_argument(
        "--basis_set_file", metavar="FILENAME", required=True, help="File containing the used basis sets."
    )
    parser.add_argument("--xyz_file", metavar="FILENAME", required=True, help=".xyz file containing the geometry.")
    parser.add_argument(
        "--wfn_file", metavar="FILENAME", required=True, help="cp2k restart file containing the wavefunction."
    )
    parser.add_argument(
        "--output_file", metavar="FILENAME", required=True, help="Output file containing the bond orders."
    )
    parser.add_argument(
        "--bader_basins_dir", metavar="DIR", required=True, help="directory containing the Bader basin .cube files."
    )
    parser.add_argument("--dx", type=float, metavar="DX", default=0.2, help="Spatial step for the grid (angstroms).")
    parser.add_argument(
        "--eval_cutoff",
        type=float,
        metavar="D",
        default=14.0,
        help=("Size of the region around the atom where each orbital is evaluated (only used for 'G' region)."),
    )
    parser.add_argument(
        "--eval_region",
        type=str,
        nargs=6,
        metavar="X",
        required=False,
        default=["G", "G", "G", "G", "G", "G"],
        help=common.eval_region_description,
    )

    # Rest of your existing code, unchanged
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

    # ... rest of your existing code ...

    print("R%d/%d finished, total time: %.2fs" % (mpi_rank, mpi_size, (time.time() - time0)))


if __name__ == "__main__":
    main()
