import numpy as np
import scipy.io
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('restartfile', metavar='FILENAME', help='Restart-WFN file')
parser.add_argument('asciifile', metavar='FILENAME', help='Output ascii file')
args = parser.parse_args()

start_time = time.time()

inp_file = args.restartfile
out_file = args.asciifile

inpf = scipy.io.FortranFile(inp_file, 'r')
outf = open(out_file, 'w')

natom, nspin, nao, nset_max, nshell_max = inpf.read_ints()
outf.write("%d %d %d %d %d\n" % (natom, nspin, nao, nset_max, nshell_max))

nset_info = inpf.read_ints()
outf.write(' '.join(map(str, nset_info))+"\n")

nshell_info = inpf.read_ints()
outf.write(' '.join(map(str, nshell_info))+"\n")

nso_info = inpf.read_ints()
outf.write(' '.join(map(str, nso_info))+"\n")

for ispin in range(nspin):
    nmo, homo, lfomo, nelectron = inpf.read_ints()
    outf.write("%d %d %d %d\n" % (nmo, homo, lfomo, nelectron))

    evals_occs = inpf.read_reals()
    outf.write(' '.join(map(str, evals_occs))+"\n")

    for imo in range(nmo):
        coefs = inpf.read_reals()
        outf.write(' '.join(map(str, coefs))+"\n")

inpf.close()
outf.close()

print("Completed in %.2f s" % (time.time()-start_time))
