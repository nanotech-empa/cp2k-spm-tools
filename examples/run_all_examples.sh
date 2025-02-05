#!/bin/bash

# Loop through all subdirectories
for dir in */ ; do
  if [ -f "$dir/run.sh" ]; then
    echo ""
    echo "# ----"
    echo "# Running example in $dir"
    echo "# ----"
    (cd "$dir" && ./run.sh)
  fi
done