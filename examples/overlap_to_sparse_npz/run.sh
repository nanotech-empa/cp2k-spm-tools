#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# Real CP2K output: the full AO overlap matrix of a CH4 molecule.
INPUT_FILE="$REPO_ROOT/tests/data/cp2k_ch4_overlap_matrix.log"

mkdir -p "$SCRIPT_DIR/out"
OUTPUT_FILE="$SCRIPT_DIR/out/overlap_sparse.npz"

cp2k-overlap-to-sparse-npz \
  "$INPUT_FILE" \
  "$OUTPUT_FILE" \
  --threshold 1e-2

OUTPUT_FILE="$OUTPUT_FILE" python - <<'PY'
import os

import numpy as np

from cp2k_spm_tools.cp2k_overlap_matrix import Cp2kOverlapMatrix

parsed = Cp2kOverlapMatrix.from_npz(os.environ["OUTPUT_FILE"])
n_rows, n_cols = parsed.matrix.shape

print(f"shape:          {n_rows} x {n_cols}")
print(f"stored entries: {parsed.matrix.nnz} of {n_rows * n_cols} ({parsed.matrix.nnz / (n_rows * n_cols):.1%})")
print(f"symmetric:      {(parsed.matrix - parsed.matrix.T).nnz == 0}")

# One line per atom instead of one entry per atomic orbital: a real basis set
# has far too many AOs to print individually.
print("atomic orbitals per atom:")
for atom in np.unique(parsed.atom_index):
    mask = parsed.atom_index == atom
    labels = ", ".join(parsed.orbital[mask][:4])
    n_ao = int(mask.sum())
    print(f"  atom {atom} ({parsed.element[mask][0]}): {n_ao:2d} AOs -- {labels}{', ...' if n_ao > 4 else ''}")
PY
