#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$SCRIPT_DIR/out"
OUTPUT_FILE="$SCRIPT_DIR/out/overlap_sparse.npz"

cp2k-overlap-to-sparse-npz \
  "$SCRIPT_DIR/cp2k_overlap_matrix.log" \
  "$OUTPUT_FILE" \
  --threshold 1e-2

OUTPUT_FILE="$OUTPUT_FILE" python - <<'PY'
import os

from cp2k_spm_tools.cp2k_overlap_matrix import read_sparse_overlap_npz

parsed = read_sparse_overlap_npz(os.environ["OUTPUT_FILE"])
print("shape:", parsed.matrix.shape)
print("stored entries:", parsed.matrix.nnz)
print("atoms:", parsed.atom_index.tolist())
print("elements:", parsed.element.tolist())
print("orbitals:", parsed.orbital.tolist())
PY
