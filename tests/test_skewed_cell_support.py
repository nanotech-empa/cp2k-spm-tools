"""Run the existing benzene WFN and independent metric reference checks in CI."""

import os
import subprocess
import sys
from pathlib import Path


def test_skewed_cell_reference_checks(tmp_path):
    script = Path(__file__).resolve().parents[1] / "examples" / "validate_skewed_cell_support.py"
    environment = dict(os.environ, OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "skewed-cell validation passed" in result.stdout
