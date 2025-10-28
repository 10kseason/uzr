#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Logged-use wrapper (refresh) that delegates to uzr.infer_longrun_standalone
with extended logging defaults and SelfEval toggle support.

- Maps `--turn` -> `--turns`
- Passes through unknown args
- Auto-names summary CSV to logu/<ts>_s{inner}_t{turns}_{ckpt}.csv when not provided
- Supports `--self_eval {on,off}` via env UZR_SELF_EVAL
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--turn", type=int)
    ap.add_argument("--turns", type=int)
    ap.add_argument("--device")
    ap.add_argument("--inner_steps", type=int)
    ap.add_argument("--ckpt")
    # Extended logging controls
    ap.add_argument("--summary_csv")
    ap.add_argument("--summary_json")
    ap.add_argument("--summary_dir")
    # Identity passthrough (default kept upstream)
    ap.add_argument("--identity")
    # Self-eval toggle
    ap.add_argument("--self_eval", choices=["on", "off"])
    known, unknown = ap.parse_known_args()

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
    if known.identity is not None:
        new_argv += ["--identity", known.identity]

    # Prepare summary paths
    summary_csv = known.summary_csv
    summary_json = known.summary_json
    if known.summary_dir and not (summary_csv or summary_json):
        d = Path(known.summary_dir)
        d.mkdir(parents=True, exist_ok=True)
        summary_csv = str(d / "infer_summary.csv")
        summary_json = str(d / "infer_summary.json")
    if not summary_csv:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_base = Path(known.ckpt).stem if known.ckpt else "model"
        inner = known.inner_steps if known.inner_steps is not None else 0
        tval = turns if turns is not None else 0
        out_dir = Path("logu"); out_dir.mkdir(parents=True, exist_ok=True)
        summary_csv = str(out_dir / f"{ts}_s{inner}_t{tval}_{model_base}.csv")
    for pth in [summary_csv, summary_json]:
        if pth:
            Path(pth).parent.mkdir(parents=True, exist_ok=True)
    if summary_csv:
        new_argv += ["--summary_csv", summary_csv]
    if summary_json:
        new_argv += ["--summary_json", summary_json]

    # Extended logging hint and SelfEval toggle
    os.environ.setdefault("UZR_LOG_EXTENDED", "1")
    if known.self_eval is not None:
        os.environ["UZR_SELF_EVAL"] = "1" if known.self_eval == "on" else "0"

    # Delegate to module main()
    import importlib
    mod = importlib.import_module("uzr.infer_longrun_standalone")
    old_argv = sys.argv
    try:
        sys.argv = [new_argv[0]] + new_argv[1:] + unknown
        mod.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()

