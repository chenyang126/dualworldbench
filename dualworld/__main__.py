"""Dispatch public commands to the canonical suite scripts."""

import argparse
import subprocess
import sys

from ._paths import ROOT

COMMANDS = {
    "validate": "scripts/validate_data.py",
    "simworld": "scripts/run_simworld.py",
    "textworld": "scripts/run_textworld.py",
    "dualworld-vqa": "qa_gt/run_vqa_api.py",
    "score-simworld": "scripts/score_simworld.py",
    "score-vqa": "scripts/score_vqa.py",
    "build-nc-references": "scripts/build_nc_references.py",
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m dualworld",
        description="DualWorldBench: SimWorld, TextWorld and DualWorld-VQA. "
                    "Use COMMAND --help for suite-specific arguments.")
    parser.add_argument("command", choices=COMMANDS)
    # Delegate all remaining arguments, including --help, unchanged.
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    args = parser.parse_args(argv[:1])
    return subprocess.call([sys.executable, str(ROOT / COMMANDS[args.command]), *argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
