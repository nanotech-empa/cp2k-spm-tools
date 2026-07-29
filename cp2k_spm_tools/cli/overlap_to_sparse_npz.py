from __future__ import annotations

import argparse

from cp2k_spm_tools.cp2k_overlap_matrix import write_sparse_overlap_npz


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
    write_sparse_overlap_npz(
        args.input,
        args.output,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
