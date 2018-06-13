import numpy as np
import scipy.io
import time

import argparse

import cp2k_utilities as cu

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

parser = argparse.ArgumentParser(
    description='Select only morbs that are in the energy range.')
parser.add_argument(
    '--cp2k_output',
    metavar='FILENAME',
    required=True,
    help='CP2K output of the SCF calculation.')
parser.add_argument(
    '--restart_file',
    metavar='FILENAME',
    required=True,
    help='Restart file containing the final wavefunction.')
parser.add_argument(
    '--output_file',
    metavar='FILENAME',
    required=True,
    help='Output binary containing the cropped morbs.')
parser.add_argument(
    '--emin',
    type=float,
    metavar='E',
    required=True,
    help='Lowest energy value for selecting orbitals.')
parser.add_argument(
    '--emax',
    type=float,
    metavar='E',
    required=True,
    help='Highest energy value for selecting orbitals.')

args = parser.parse_args()

emin = args.emin
emax = args.emax

time0 = time.time()
fermi = cu.read_fermi_from_cp2k_out(args.cp2k_output)
print("Read cp2k out: %.3f" % (time.time()-time0))

inpf = scipy.io.FortranFile(args.restart_file, 'r')
outf = scipy.io.FortranFile(args.output_file, 'w')

line1 = inpf.read_ints()
outf.write_record(line1)

natom, nspin, nao, nset_max, nshell_max = line1

nset_info = inpf.read_ints()
outf.write_record(nset_info)

nshell_info = inpf.read_ints()
outf.write_record(nshell_info)

nso_info = inpf.read_ints()
outf.write_record(nso_info)

for ispin in range(nspin):
    nmo, homo, lfomo, nelectron = inpf.read_ints()
    evals_occs = inpf.read_reals()

    evals = evals_occs[:int(len(evals_occs)/2)]
    occs = evals_occs[int(len(evals_occs)/2):]

    proc_evals = evals*hart_2_ev
    if fermi == None:
        fermi = proc_evals[homo-1]
    proc_evals -= fermi

    num_smaller = len(np.where(proc_evals < emin)[0])

    crop_indexes = np.where((proc_evals >= emin) & (proc_evals <= emax))
    crop_evals = evals[crop_indexes]
    crop_occs = occs[crop_indexes]

    homo -= num_smaller
    lfomo -= num_smaller
    outf.write_record(np.array([len(crop_evals), homo, lfomo, nelectron], dtype=np.int32))
    outf.write_record(np.concatenate([crop_evals, crop_occs]))

    for imo in range(nmo):
        coefs = inpf.read_reals()
        if proc_evals[imo] < emin:
            continue
        if proc_evals[imo] > emax:
            break
        outf.write_record(coefs)

inpf.close()
outf.close()

print("Completed in %.2f s" % (time.time()-time0))
