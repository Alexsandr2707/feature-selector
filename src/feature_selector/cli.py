"""Command-line interface for running experiments from YAML configurations."""

import argparse
import json
import sys
from collections.abc import Sequence

from .artifacts import serializable
from .experiment import run_experiment


def main(argv: Sequence[str] | None = None) -> int:
    """Run a configured experiment and print its artifact path and metrics as JSON.

    Parameters:
        argv (Sequence[str] | None): Command-line arguments excluding the program name;
            None reads the process arguments.

    Returns:
        int: Zero on success or one for handled experiment errors, printed to stderr.
            Argument parsing errors exit through argparse.
    """
    parser = argparse.ArgumentParser(prog="feature-selector")

    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="Train and evaluate on train/validation")
    run_parser.add_argument("config", help="Path to a YAML config")

    args = parser.parse_args(argv)

    try:
        result = run_experiment(args.config)
    except (ValueError, TypeError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    report = {"artifact_dir": result.artifact_dir, "metrics": result.metrics}
    print(json.dumps(serializable(report), ensure_ascii=False))
    return 0
