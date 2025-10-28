#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compatibility wrapper for running the standalone long-run with legacy CLI,
with extended logging defaults and SelfEval toggle.

- Maps `--turn` -> `--turns`
- Passes through standard args and unknown args
- Auto-names summary CSV to logu/<ts>_s{inner}_t{turns}_{ckpt}.csv when not provided
- Allows toggling SelfEval via `--self_eval {on,off}` (propagates as env UZR_SELF_EVAL)
"""

import sys
import os
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--turn", type=int)
    parser.add_argument("--turns", type=int)
    parser.add_argument("--device")
    parser.add_argument("--inner_steps", type=int)
    parser.add_argument("--ckpt")
    # Extended logging controls
    parser.add_argument("--summary_csv")
    parser.add_argument("--summary_json")
    parser.add_argument("--summary_dir")
    # Self-eval toggle
    parser.add_argument("--self_eval", choices=["on", "off"])
    known, unknown = parser.parse_known_args()

    # Ensure we can import as package: add parent of this file's directory
    here = Path(__file__).resolve().parent
    parent = here
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

    # Build argv for underlying runner
    new_argv = ["infer_longrun_standalone.py"]
    turns = known.turns if known.turns is not None else known.turn
    if turns is not None:
        new_argv += ["--turns", str(turns)]
    if known.device is not None:
        new_argv += ["--device", known.device]
    if known.inner_steps is not None:
        new_argv += ["--inner_steps", str(known.inner_steps)]
    if known.ckpt is not None:
        new_argv += ["--ckpt", known.ckpt]

    # Prepare summary paths (ensure directories exist)
    summary_csv = known.summary_csv
    summary_json = known.summary_json
    if known.summary_dir and not (summary_csv or summary_json):
        # If only a directory is provided, use defaults inside it
        d = Path(known.summary_dir)
        d.mkdir(parents=True, exist_ok=True)
        summary_csv = str(d / "infer_summary.csv")
        summary_json = str(d / "infer_summary.json")
    # Auto-name CSV under logu/<timestamp>_s{inner}_t{turns}_{model}.csv when not set
    if not summary_csv:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_base = Path(known.ckpt).stem if known.ckpt else "model"
        inner = known.inner_steps if known.inner_steps is not None else 0
        tval = turns if turns is not None else 0
        out_dir = Path("logu"); out_dir.mkdir(parents=True, exist_ok=True)
        summary_csv = str(out_dir / f"{ts}_s{inner}_t{tval}_{model_base}.csv")
    # Ensure parent dirs exist
    for pth in [summary_csv, summary_json]:
        if pth:
            Path(pth).parent.mkdir(parents=True, exist_ok=True)
    if summary_csv:
        new_argv += ["--summary_csv", summary_csv]
    if summary_json:
        new_argv += ["--summary_json", summary_json]

    # Extended logging hint
    os.environ.setdefault("UZR_LOG_EXTENDED", "1")
    # Self-eval toggle propagated via env so constructors can read it
    if known.self_eval is not None:
        os.environ["UZR_SELF_EVAL"] = "1" if known.self_eval == "on" else "0"

    # Delegate to module main()
    import importlib
    mod = importlib.import_module("uzr.infer_longrun_standalone")

    # Temporarily swap sys.argv so argparse in callee sees mapped args
    old_argv = sys.argv
    try:
        sys.argv = [new_argv[0]] + new_argv[1:] + unknown
        mod.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()

