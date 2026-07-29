from __future__ import annotations

import argparse

from cp2k_spm_tools.cp2k_overlap_matrix import Cp2kOverlapMatrix


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for overlap-log conversion."""

    parser = argparse.ArgumentParser(description="Convert CP2K AO overlap matrix logs to sparse NPZ files.")
    parser.add_argument("input", help="CP2K output/log file containing an OVERLAP MATRIX block.")
    parser.add_argument("output", help="Output compressed sparse NPZ file.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Keep entries with absolute value larger than this threshold.",
    )
    return parser


def main(argv=None) -> None:
    """Convert a human-readable CP2K overlap matrix to sparse NPZ format."""

    args = build_parser().parse_args(argv)
    Cp2kOverlapMatrix.from_cp2k_output(args.input, threshold=args.threshold).to_npz(args.output)


if __name__ == "__main__":
    main()
