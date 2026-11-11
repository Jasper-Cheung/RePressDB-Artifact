#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RePressDB B3-C v1.0.0
Balanced-order concurrency confirmation experiment.

Purpose
-------
Remove the remaining experimental confound in B3:
the previous B3 experiment always executed

    foreground_only -> background_migration

This version keeps all storage/query parameters unchanged and only changes
the execution schedule:

    AB order:
        foreground -> background

    BA order:
        background -> foreground

Each pair uses:
- identical logical scale
- identical thread count
- identical query seed
- identical migration seed
- identical truth verification

The goal is NOT to improve performance.
The goal is to obtain an unbiased estimate of migration interference.

Run:
------
Place this file beside the B3 v1.1.1 program and run directly in PyCharm.

Required existing directory:
    results_b3_scale_concurrency_v110

Output:
-------
    results_b3_c_balanced_concurrency_v100
"""

from pathlib import Path
import importlib.util
import json
import hashlib
import sys
import time
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


VERSION = "B3-C-v1.0.0"
SOURCE_RESULT = "results_b3_scale_concurrency_v110"
OUTPUT = "results_b3_c_balanced_concurrency_v100"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=True)


def main():

    root = Path(__file__).resolve().parent
    old = root / SOURCE_RESULT
    out = root / OUTPUT
    out.mkdir(exist_ok=True)

    print("=" * 100)
    print("RePressDB B3-C v1.0.0 balanced-order confirmation")
    print("AB/BA execution only. No storage/controller/query changes.")
    print("=" * 100)

    if not old.exists():
        raise FileNotFoundError(
            "Missing previous B3 result directory: " + str(old)
        )

    records = {
        "version": VERSION,
        "parent_result": SOURCE_RESULT,
        "parent_hash": sha256_file(
            root / "RUN_THIS_ONLY_REPRESSDB_B3_SCALE_CONCURRENCY_RECOVER_V1_1_1.py"
        )
        if (
            root /
            "RUN_THIS_ONLY_REPRESSDB_B3_SCALE_CONCURRENCY_RECOVER_V1_1_1.py"
        ).exists()
        else None,
        "protocol": {
            "orders": ["AB", "BA"],
            "AB": [
                "foreground_only",
                "background_migration",
            ],
            "BA": [
                "background_migration",
                "foreground_only",
            ],
            "unchanged": [
                "logical scales",
                "threads",
                "queries",
                "truth checking",
                "migration amount",
                "random seeds",
            ],
        },
        "status": "prepared",
        "created_time": time.time(),
    }

    write_json(
        out / "b3c_protocol.json",
        records
    )

    print()
    print("Protocol generated.")
    print()
    print("This script is a controlled confirmation wrapper.")
    print("It deliberately does not reuse biased AB-only conclusions.")
    print("Continue by connecting it with the B3 execution engine.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
