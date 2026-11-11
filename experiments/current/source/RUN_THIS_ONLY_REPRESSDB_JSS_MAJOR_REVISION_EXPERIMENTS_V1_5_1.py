#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
RePressDB JSS Major-Revision Experiment Suite v1.5.1
====================================================

PURPOSE
-------
This is the only new experiment entry point required for the Journal of
Systems and Software revision. It does not rewrite the manuscript. It produces
the additional evidence needed before the manuscript is revised.

Run order
---------
J1. End-to-end continuous replay on the single-active store
    * 8 pre-registered BDG2 buildings
    * three frozen R1 trace families
    * chunk sizes 8192 and 16384
    * initial-best, frozen Shape-EWMA-BE, frozen RePressDB
    * three repeats with a balanced cyclic Latin-square policy order
    * fresh process and fresh copied store for every policy run
    * every query checked against the independent float32 truth
    * query, controller, migration, catalog/dispatch, and migration-stage time
    * fresh-process prefix and steady-state results
    * one controlled competing-memory-pressure confirmation subset

J2. Multi-seed continuous replay
    * all 32 frozen R1 buildings
    * three pre-registered independent trace seeds
    * three trace families and two chunk sizes
    * initial-best, best-static, AF-HC-D, Shape-EWMA-BE, RePressDB,
      Page-Hinkley-BE, and migration-aware hindsight oracle
    * trace realization is the primary statistical unit
    * paired confidence intervals across trace seeds

J2A. Unified RePressDB component ablation
    * full RePressDB
    * short-window-only
    * no confirmation
    * no break-even
    * no cooldown
    * no write budget
    * same multi-seed plans and same measured cost tables

J3. Process-termination and corruption fuzzing
    * abrupt os._exit at random Python line events during migration
    * active payload truncation after catalog commit
    * active metadata corruption after catalog commit
    * orphan staging/final files
    * recovery, full checksum, exact-one-active-payload, no-temp-file, and
      query truth checks after every trial
    * 500 randomized/corruption trials by default
    * this is not a physical power-loss test

J4. Calibration overhead and robustness
    * fresh full calibration over the 32 paired BDG2 buildings
    * wall time, CPU time, process I/O bytes, final disk footprint
    * 10%, 25%, and 50% sampled-profile studies
    * bounded cost-table errors of +/-5%, +/-10%, and +/-20%
    * decisions use sampled/perturbed tables; evaluation uses the true table

J5. Consolidated acceptance and upload package
    * result consistency checks
    * protocol and experiment freeze hashes
    * one ZIP containing CSV, JSON, NPZ, and logs for manuscript revision

SCIENTIFIC FREEZE
-----------------
The script must read the completed S1 and R1 result directories.

Shape-EWMA-BE remains the configuration frozen in PEMS04 S1.
RePressDB remains:
    long window             300
    short window            100
    minimum observations    200
    decision interval       100
    confirmation rounds     3
    cooldown                300
    prediction horizon      1200
    minimum relative gain   8%
    rewrite safety factor   1.25

No S1, R1, PEMS08, BDG2, multi-seed, end-to-end, recovery, calibration,
or sensitivity result may be used to retune either frozen policy.

DIRECT RUN
----------
Put this file in the RePressDB project root and run it directly in PyCharm.

Required scripts in the same directory:
    RUN_THIS_ONLY_REPRESSDB_S1_SHAPE_EWMA_BE_V1_3_1.py
    RUN_THIS_ONLY_REPRESSDB_R1_CONTINUOUS_NONSTATIONARY_V1_4_1.py
    RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_DEV_PEMS04_V0_7_0.py
    RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_DEADBAND_RECOVER_V0_7_3.py
    RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_V0_7_4.py
    RUN_THIS_ONLY_REPRESSDB_A2_PEMS08_BDG2_HOLDOUT_V0_8_0.py
    RUN_THIS_ONLY_REPRESSDB_E5_BDG2_FROZEN_DIRECT_V0_6_0.py
    RUN_THIS_ONLY_REPRESSDB_B1_SINGLE_ACTIVE_STORE_V0_9_0.py
    RUN_THIS_ONLY_REPRESSDB_E0C_V0_2_0.py
    RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT.py
    RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT_V0_2_0.py

Required completed result directories:
    results_s1_shape_ewma_be_v130/
    results_r1_continuous_nonstationary_v140/
    results_a0_a1_hotcold_pems04_v070/
    results_a0_a1_hotcold_deadband_pems04_v073/
    results_a0_a1_hotcold_deadband_pems04_v074/
    results_a2_hotcold_holdouts_v080/
    results_b1_single_active_store_v090/
    results_b2_full_bdg2_scale_v100/

Output:
    results_jss_major_revision_v151/

RESUME
------
All long stages are checkpointed. Rerun this same file after interruption.
Do not delete the output directory unless the full experiment is intentionally
being restarted.

INTERNAL WORKER MODES
---------------------
The script starts isolated subprocesses for end-to-end cells and crash trials.
The user does not need to supply command-line arguments.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import random
import shutil
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import psutil
except Exception:
    psutil = None


VERSION = "1.5.1"
TARGET_JOURNAL = "Journal of Systems and Software"
MASTER_SEED = 20260730

OUTPUT_DIR_NAME = "results_jss_major_revision_v151"

S1_SCRIPT_NAME = "RUN_THIS_ONLY_REPRESSDB_S1_SHAPE_EWMA_BE_V1_3_1.py"
R1_SCRIPT_NAME = "RUN_THIS_ONLY_REPRESSDB_R1_CONTINUOUS_NONSTATIONARY_V1_4_1.py"
V070_SCRIPT_NAME = "RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_DEV_PEMS04_V0_7_0.py"
V073_SCRIPT_NAME = "RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_DEADBAND_RECOVER_V0_7_3.py"
V074_SCRIPT_NAME = "RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_V0_7_4.py"
A2_SCRIPT_NAME = "RUN_THIS_ONLY_REPRESSDB_A2_PEMS08_BDG2_HOLDOUT_V0_8_0.py"
E5_SCRIPT_NAME = "RUN_THIS_ONLY_REPRESSDB_E5_BDG2_FROZEN_DIRECT_V0_6_0.py"
B1_SCRIPT_NAME = "RUN_THIS_ONLY_REPRESSDB_B1_SINGLE_ACTIVE_STORE_V0_9_0.py"

S1_RESULT_DIR_NAME = "results_s1_shape_ewma_be_v130"
R1_RESULT_DIR_NAME = "results_r1_continuous_nonstationary_v140"
A0_RESULT_DIR_NAME = "results_a0_a1_hotcold_pems04_v070"
V073_RESULT_DIR_NAME = "results_a0_a1_hotcold_deadband_pems04_v073"
V074_RESULT_DIR_NAME = "results_a0_a1_hotcold_deadband_pems04_v074"
A2_RESULT_DIR_NAME = "results_a2_hotcold_holdouts_v080"
B1_RESULT_DIR_NAME = "results_b1_single_active_store_v090"
B2_RESULT_DIR_NAME = "results_b2_full_bdg2_scale_v100"

CHUNK_SIZES = (8192, 16384)
TRACE_FAMILIES = (
    "R1_A_gradual_periodic",
    "R1_B_mean_reverting",
    "R1_C_noisy_bursts",
)
CORE_POLICIES = (
    "initial_best",
    "shape_ewma_be_frozen",
    "repressdb_frozen",
)

# J1: End-to-end actual execution.
R2_BUILDING_COUNT = 8
R2_REPEATS = 3
R2_POLICY_PERMUTATIONS = (
    CORE_POLICIES,
    (
        CORE_POLICIES[1],
        CORE_POLICIES[2],
        CORE_POLICIES[0],
    ),
    (
        CORE_POLICIES[2],
        CORE_POLICIES[0],
        CORE_POLICIES[1],
    ),
)
R2_FRESH_PREFIX_QUERIES = 5000
R2_BOOTSTRAPS = 20_000
R2_MEMORY_PRESSURE_BYTES = 1024 * 1024 * 1024
R2_MEMORY_PRESSURE_TRACE = "R1_C_noisy_bursts"
R2_MEMORY_PRESSURE_SIZE = 16384

# J2: Multiple independently frozen trace realizations.
MULTISEED_TRACE_SEEDS = (
    20260801,
    20260802,
    20260803,
)
MULTISEED_EVENTS_PER_BUILDING = 10_000
MULTISEED_BOOTSTRAPS = 20_000

# PH-BE development grid. It is selected only with PEMS04 development traces.
PH_DELTAS = (0.005,)
PH_THRESHOLDS = (10.0,)
PH_WINDOW_CAPS = (300,)
PH_DEADBANDS = (0.05,)
PH_HORIZONS = (600,)
PH_TIE_TOLERANCE = 0.005
PH_SAFETY_LIMIT = 1.02

# J3: process crash/corruption fuzz.
CRASH_TOTAL_TRIALS = 500
CRASH_RANDOM_KILL_TRIALS = 300
CRASH_PAYLOAD_TRUNCATION_TRIALS = 75
CRASH_METADATA_CORRUPTION_TRIALS = 50
CRASH_ORPHAN_TRIALS = 75
CRASH_TRUTH_QUERIES_PER_TRIAL = 20

# J4: calibration sampling and bounded error.
CALIBRATION_SAMPLE_FRACTIONS = (0.10, 0.25, 0.50)
CALIBRATION_SAMPLE_SEEDS = (
    20260811,
    20260812,
    20260813,
)
COST_ERROR_LEVELS = (0.00, 0.05, 0.10, 0.20)
CALIBRATION_ROBUSTNESS_TRACES = (
    "R1_C_noisy_bursts",
)
COST_ERROR_SEEDS = (
    20260821,
    20260822,
    20260823,
)

# Frozen controller constants. They are checked against the imported module.
DECISION_INTERVAL = 100
MIN_OBSERVATIONS = 200
BUDGET_INTERVAL_GLOBAL_QUERIES = 1000

CATEGORY_CLASS_WEIGHTS = {
    "point": {"point": 1.0},
    "scan": {
        "short_scan": 0.20,
        "medium_scan": 0.30,
        "long_scan": 0.50,
    },
    "aggregate": {
        "short_agg": 0.20,
        "medium_agg": 0.25,
        "long_agg": 0.35,
        "threshold": 0.20,
    },
}


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(
            json_safe(value),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
            sort_keys=False,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def append_jsonl_fsync(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                json_safe(dict(row)),
                ensure_ascii=False,
                allow_nan=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        json_safe(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_seed(*parts: Any) -> int:
    token = "|".join(str(part) for part in parts)
    return int.from_bytes(
        hashlib.sha256(token.encode("utf-8")).digest()[:8],
        "little",
        signed=False,
    )


def stable_hash(*parts: Any) -> str:
    token = "|".join(str(part) for part in parts)
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except FileNotFoundError:
                pass
    return int(total)


def load_module(path: Path, module_name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    specification = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def geometric_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array) & (array > 0)]
    if array.size == 0:
        return math.nan
    return float(np.exp(np.mean(np.log(array))))


def percentile_interval(
    values: Sequence[float],
    seed_parts: Sequence[Any],
    replicates: int,
    statistic: str = "median",
) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(
        stable_seed(MASTER_SEED, "bootstrap", *seed_parts)
    )
    index = rng.integers(
        0,
        array.size,
        size=(replicates, array.size),
    )
    sampled = array[index]
    if statistic == "median":
        estimates = np.median(sampled, axis=1)
    elif statistic == "mean":
        estimates = np.mean(sampled, axis=1)
    elif statistic == "geometric_mean":
        estimates = np.exp(np.mean(np.log(sampled), axis=1))
    else:
        raise ValueError(statistic)
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_empty_or_resume_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def checkpoint_complete(path: Path, expected_hash: Optional[str] = None) -> bool:
    if not path.exists():
        return False
    try:
        marker = read_json(path)
    except Exception:
        return False
    if expected_hash is not None:
        return marker.get("freeze_sha256") == expected_hash
    return bool(marker.get("complete", False))


def run_subprocess_checked(
    arguments: Sequence[str],
    log_path: Path,
    environment: Optional[Mapping[str, str]] = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if environment:
        env.update({str(k): str(v) for k, v in environment.items()})
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(arguments),
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parent),
            env=env,
        )
        return int(process.wait())


def sqlite_checkpoint_database(database_path: Path) -> None:
    if not database_path.exists():
        return
    connection = sqlite3.connect(str(database_path))
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.commit()
    finally:
        connection.close()


def touch_memory(buffer: bytearray, stride: int = 4096) -> int:
    checksum = 0
    for position in range(0, len(buffer), stride):
        buffer[position] = (buffer[position] + 1) % 251
        checksum += buffer[position]
    return checksum


@dataclasses.dataclass
class LoadedModules:
    s1: Any
    r1: Any
    v073: Any
    v074: Any
    a2: Any
    e5: Any
    b1: Any
    core: Any
    e1v2: Any
    e1: Any
    base: Any


@dataclasses.dataclass
class ProjectPaths:
    project: Path
    output: Path
    scripts: Dict[str, Path]
    results: Dict[str, Path]


def resolve_project_paths(project: Path) -> ProjectPaths:
    scripts = {
        "s1": project / S1_SCRIPT_NAME,
        "r1": project / R1_SCRIPT_NAME,
        "v070": project / V070_SCRIPT_NAME,
        "v073": project / V073_SCRIPT_NAME,
        "v074": project / V074_SCRIPT_NAME,
        "a2": project / A2_SCRIPT_NAME,
        "e5": project / E5_SCRIPT_NAME,
        "b1": project / B1_SCRIPT_NAME,
    }
    results = {
        "s1": project / S1_RESULT_DIR_NAME,
        "r1": project / R1_RESULT_DIR_NAME,
        "a0": project / A0_RESULT_DIR_NAME,
        "v073": project / V073_RESULT_DIR_NAME,
        "v074": project / V074_RESULT_DIR_NAME,
        "a2": project / A2_RESULT_DIR_NAME,
        "b1": project / B1_RESULT_DIR_NAME,
        "b2": project / B2_RESULT_DIR_NAME,
    }
    for path in [*scripts.values(), *results.values()]:
        if not path.exists():
            raise FileNotFoundError(path)
    output = project / OUTPUT_DIR_NAME
    output.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(
        project=project,
        output=output,
        scripts=scripts,
        results=results,
    )


def load_all_modules(paths: ProjectPaths, tag: str) -> LoadedModules:
    s1 = load_module(paths.scripts["s1"], f"repressdb_s1_{tag}")
    r1 = load_module(paths.scripts["r1"], f"repressdb_r1_{tag}")
    v073 = load_module(paths.scripts["v073"], f"repressdb_v073_{tag}")
    v074 = load_module(paths.scripts["v074"], f"repressdb_v074_{tag}")
    a2 = load_module(paths.scripts["a2"], f"repressdb_a2_{tag}")
    e5 = load_module(paths.scripts["e5"], f"repressdb_e5_{tag}")
    b1 = load_module(paths.scripts["b1"], f"repressdb_b1_{tag}")

    patch_dir = paths.output / "_patched_core"
    patch_dir.mkdir(parents=True, exist_ok=True)
    core = v073.build_corrected_core(
        paths.scripts["v070"],
        patch_dir,
    )
    core.__file__ = str(paths.scripts["v070"].resolve())
    e1v2, e1, base = core.load_repressdb_modules()

    return LoadedModules(
        s1=s1,
        r1=r1,
        v073=v073,
        v074=v074,
        a2=a2,
        e5=e5,
        b1=b1,
        core=core,
        e1v2=e1v2,
        e1=e1,
        base=base,
    )


def collect_environment_report(
    paths: ProjectPaths,
) -> Dict[str, Any]:
    package_versions: Dict[str, Any] = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sqlite": sqlite3.sqlite_version,
        "psutil": (
            getattr(psutil, "__version__", "unknown")
            if psutil is not None
            else None
        ),
    }
    for module_name in ("lz4", "zstandard"):
        try:
            module = __import__(module_name)
            package_versions[module_name] = getattr(
                module,
                "__version__",
                "unknown",
            )
        except Exception as error:
            package_versions[module_name] = f"unavailable: {error}"

    memory: Dict[str, Any] = {}
    cpu: Dict[str, Any] = {}
    disks: List[Dict[str, Any]] = []
    if psutil is not None:
        virtual = psutil.virtual_memory()
        memory = {
            "total_bytes": int(virtual.total),
            "available_bytes_at_start": int(virtual.available),
        }
        cpu = {
            "logical_cpu_count": int(psutil.cpu_count(logical=True) or 0),
            "physical_cpu_count": int(psutil.cpu_count(logical=False) or 0),
            "frequency_mhz": (
                float(psutil.cpu_freq().current)
                if psutil.cpu_freq() is not None
                else None
            ),
        }
        with contextlib.suppress(Exception):
            for partition in psutil.disk_partitions(all=False):
                disks.append(
                    {
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "filesystem": partition.fstype,
                        "options": partition.opts,
                    }
                )

    script_rows = []
    for name, path in paths.scripts.items():
        script_rows.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "line_count": len(path.read_text(encoding="utf-8").splitlines()),
            }
        )
    report = {
        "captured_at_ns": time.time_ns(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu": cpu,
        "memory": memory,
        "disks": disks,
        "package_versions": package_versions,
        "current_working_directory": str(Path.cwd()),
        "project_directory": str(paths.project.resolve()),
        "output_directory": str(paths.output.resolve()),
        "sqlite_pragmas_used": {
            "journal_mode": "WAL",
            "synchronous": "FULL",
        },
        "timer": "time.perf_counter_ns",
        "data_type": "little-endian float32 through numpy float32",
        "scripts": script_rows,
    }
    write_json_atomic(
        paths.output / "environment_report.json",
        report,
    )
    return report


def verify_parent_freezes(
    paths: ProjectPaths,
    modules: LoadedModules,
) -> Dict[str, str]:
    s1_assessment = read_json(
        paths.results["s1"] / "s1_assessment.json"
    )
    r1_assessment = read_json(
        paths.results["r1"] / "r1_assessment.json"
    )
    b1_assessment = read_json(
        paths.results["b1"] / "b1_assessment.json"
    )
    if not bool(
        s1_assessment.get("acceptance_checks", {}).get("all_ten", False)
    ):
        raise RuntimeError("S1 is not an accepted frozen parent.")
    if not bool(
        r1_assessment.get("acceptance_checks", {}).get("all_ten", False)
    ):
        raise RuntimeError("R1 is not an accepted frozen parent.")
    b1_checks = b1_assessment.get("acceptance_checks", {})
    if not b1_checks or not all(bool(v) for v in b1_checks.values()):
        raise RuntimeError("B1 is not an integrity-passing frozen parent.")

    shape_path = (
        paths.results["s1"]
        / "shape_ewma_be_frozen_config.json"
    )
    shape_config = read_json(shape_path)
    s1_freeze = (
        paths.results["s1"]
        / "s1_experiment_freeze_sha256.txt"
    ).read_text(encoding="utf-8").strip()
    if shape_config.get("s1_experiment_freeze_sha256") != s1_freeze:
        raise AssertionError("Shape-EWMA-BE config is not bound to S1 freeze.")

    r1_freeze = (
        paths.results["r1"]
        / "r1_experiment_freeze_sha256.txt"
    ).read_text(encoding="utf-8").strip()
    b1_freeze = (
        paths.results["b1"]
        / "b1_experiment_freeze_sha256.txt"
    ).read_text(encoding="utf-8").strip()
    a2_freeze = (
        paths.results["a2"]
        / "a2_experiment_freeze_sha256.txt"
    ).read_text(encoding="utf-8").strip()
    v074_freeze = (
        paths.results["v074"]
        / "experiment_freeze_sha256.txt"
    ).read_text(encoding="utf-8").strip()

    required_constants = {
        "LONG_WINDOW": 300,
        "SHORT_WINDOW": 100,
        "MIN_OBSERVATIONS": 200,
        "DECISION_INTERVAL": 100,
        "CONFIRMATION_ROUNDS": 3,
        "COOLDOWN_QUERIES": 300,
        "PREDICTION_HORIZON": 1200,
        "MIN_RELATIVE_IMPROVEMENT": 0.08,
        "REWRITE_SAFETY_FACTOR": 1.25,
    }
    observed = {
        name: getattr(modules.core, name)
        for name in required_constants
    }
    if observed != required_constants:
        raise AssertionError(
            f"Frozen RePressDB constants changed: {observed}"
        )

    return {
        "s1_freeze_sha256": s1_freeze,
        "r1_freeze_sha256": r1_freeze,
        "b1_freeze_sha256": b1_freeze,
        "a2_freeze_sha256": a2_freeze,
        "afhc_freeze_sha256": v074_freeze,
        "shape_config_sha256": sha256_file(shape_path),
    }


# ---------------------------------------------------------------------------
# R1 context and compact worker assets
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class R1Context:
    chunks: List[Any]
    metadata: Dict[str, Dict[str, Any]]
    query_pools: Dict[str, Dict[str, List[Any]]]
    profiles: Dict[str, Dict[str, Dict[str, float]]]
    encode: Dict[str, Dict[str, int]]
    decode: Dict[str, Dict[str, int]]
    migration_costs: Dict[str, Dict[Tuple[str, str], int]]
    migration_bytes: Dict[str, Dict[Tuple[str, str], int]]
    codec_frame: pd.DataFrame
    calibration_frame: pd.DataFrame
    migration_frame: pd.DataFrame
    selected_buildings: pd.DataFrame
    chunk_manifest: pd.DataFrame
    stored_by_chunk: Dict[str, Dict[str, Any]]



def rebuild_r1_chunks_from_frozen_calibration(
    paths: ProjectPaths,
    modules: LoadedModules,
) -> Tuple[List[Any], pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Reconstruct the 64 frozen R1 chunks from the already calibrated lossless
    candidate files.

    This path is scientifically equivalent to reading the original BDG2 CSV
    slices because every decoded float32 byte sequence is checked against the
    raw_sha256 stored in the frozen R1 chunk manifest. It also removes a
    fragile dependency on the absolute source-file path captured when R1 was
    first run.
    """
    manifest = pd.read_csv(
        paths.results["r1"] / "r1_chunk_manifest.csv"
    ).sort_values(
        ["chunk_size", "building_rank"]
    ).reset_index(drop=True)

    calibration_dir = paths.results["r1"] / "calibration"
    codec_frame = pd.read_csv(
        calibration_dir / "codec_records.csv"
    )
    storage_dir = calibration_dir / "_candidate_storage"
    stored_by_chunk: Dict[str, Dict[str, Any]] = {}
    manifest_n = {
        str(row.chunk_id): int(row.chunk_size)
        for row in manifest.itertuples(index=False)
    }
    for chunk_id, group in codec_frame.groupby("chunk_id"):
        chunk_id = str(chunk_id)
        local: Dict[str, Any] = {}
        for codec_row in group.itertuples(index=False):
            codec = str(codec_row.codec)
            stem = storage_dir / chunk_id / codec
            data_path = stem.with_suffix(".bin")
            metadata_candidate = stem.with_suffix(".meta.bin")
            if not data_path.exists():
                raise FileNotFoundError(data_path)
            if metadata_candidate.exists():
                metadata_path: Optional[Path] = metadata_candidate
                parsed_metadata = modules.base.parse_block_metadata(
                    metadata_path.read_bytes()
                )
            else:
                metadata_path = None
                parsed_metadata = (
                    {"level": 3}
                    if codec == "whole_zstd"
                    else {}
                )
            local[codec] = modules.base.StoredChunk(
                codec=codec,
                n=manifest_n[chunk_id],
                data_path=data_path,
                metadata_path=metadata_path,
                payload_size_bytes=int(codec_row.payload_size_bytes),
                metadata_size_bytes=int(codec_row.metadata_size_bytes),
                metadata=parsed_metadata,
            )
        stored_by_chunk[chunk_id] = local

    representations = modules.base.representation_registry(3)
    chunks: List[Any] = []
    metadata: Dict[str, Dict[str, Any]] = {}

    for row in manifest.itertuples(index=False):
        chunk_id = str(row.chunk_id)
        if chunk_id not in stored_by_chunk:
            raise FileNotFoundError(
                f"Frozen candidate representation missing for {chunk_id}"
            )
        local_map = stored_by_chunk[chunk_id]
        preferred = (
            "whole_lz4"
            if "whole_lz4" in local_map
            else sorted(local_map)[0]
        )
        stored = local_map[preferred]
        stats = modules.base.IOStats()
        with open(stored.data_path, "rb") as payload_handle:
            values = representations[preferred].decode_all(
                stored,
                stored.metadata,
                payload_handle,
                stats,
            )
        values = np.ascontiguousarray(values, dtype=np.float32)
        if int(values.size) != int(row.chunk_size):
            raise AssertionError(
                f"Decoded length mismatch for {chunk_id}: "
                f"{values.size} != {row.chunk_size}"
            )
        observed_sha256 = hashlib.sha256(values.tobytes()).hexdigest()
        if observed_sha256 != str(row.raw_sha256):
            raise AssertionError(
                f"Frozen raw SHA-256 mismatch for {chunk_id}: "
                f"{observed_sha256} != {row.raw_sha256}"
            )

        chunks.append(
            modules.base.ChunkUnit(
                chunk_id=chunk_id,
                series_index=int(row.building_rank),
                chunk_start=int(row.chunk_start),
                chunk_size=int(row.chunk_size),
                values=values,
            )
        )
        metadata[chunk_id] = {
            "unit_type": "building",
            "unit_id": str(row.building_id),
            "building_id": str(row.building_id),
            "meter_type": str(row.meter_type),
            "site_id": str(row.site_id),
            "source_file": str(row.source_file),
            "source_column": str(row.source_column),
            "reconstruction_source": "frozen_r1_candidate_storage",
        }

    return chunks, manifest, metadata

def load_r1_context(
    paths: ProjectPaths,
    modules: LoadedModules,
) -> R1Context:
    selected = pd.read_csv(
        paths.results["r1"] / "r1_selected_buildings.csv"
    )
    try:
        chunks, rebuilt_manifest, metadata = modules.r1.load_chunks(
            selected,
            modules.e5,
            modules.base,
        )
        chunk_reconstruction_source = "original_bdg2_csv"
    except FileNotFoundError as error:
        print(
            "[JSS v1.5.1] Original BDG2 source path is unavailable. "
            "Reconstructing all frozen chunks from the lossless R1 "
            "candidate storage instead."
        )
        print(f"[JSS v1.5.1] Missing path: {error}")
        chunks, rebuilt_manifest, metadata = (
            rebuild_r1_chunks_from_frozen_calibration(
                paths,
                modules,
            )
        )
        chunk_reconstruction_source = "frozen_r1_candidate_storage"
    frozen_manifest = pd.read_csv(
        paths.results["r1"] / "r1_chunk_manifest.csv"
    )
    compare_columns = [
        "chunk_id",
        "building_rank",
        "building_id",
        "meter_type",
        "site_id",
        "source_file",
        "source_column",
        "chunk_size",
        "chunk_start",
        "chunk_end_exclusive",
        "raw_bytes",
        "raw_sha256",
    ]
    left = frozen_manifest[compare_columns].sort_values(
        ["chunk_size", "building_rank"]
    ).reset_index(drop=True)
    right = rebuilt_manifest[compare_columns].sort_values(
        ["chunk_size", "building_rank"]
    ).reset_index(drop=True)
    if not left.equals(right):
        raise AssertionError("Rebuilt R1 chunk manifest differs from frozen R1.")
    print(
        f"[JSS v1.5.1] R1 chunk source: {chunk_reconstruction_source}; "
        f"all {len(rebuilt_manifest)} raw SHA-256 values verified."
    )

    calibration_dir = paths.results["r1"] / "calibration"
    calibration_frame = pd.read_csv(
        calibration_dir / "calibration_records.csv"
    )
    codec_frame = pd.read_csv(
        calibration_dir / "codec_records.csv"
    )
    migration_frame = pd.read_csv(
        calibration_dir / "measured_migration_costs.csv"
    )
    profiles = modules.s1.load_profiles(calibration_frame)
    encode, decode = modules.s1.load_encode_decode_maps(codec_frame)
    migration_costs, migration_bytes = modules.s1.parse_migration_maps(
        migration_frame
    )
    query_pools = modules.a2.build_query_pools(
        modules.base,
        modules.e1,
        modules.core,
        chunks,
    )
    stored_by_chunk = modules.r1.reconstruct_stored(
        modules.base,
        chunks,
        codec_frame,
        calibration_dir / "_candidate_storage",
    )
    return R1Context(
        chunks=chunks,
        metadata=metadata,
        query_pools=query_pools,
        profiles=profiles,
        encode=encode,
        decode=decode,
        migration_costs=migration_costs,
        migration_bytes=migration_bytes,
        codec_frame=codec_frame,
        calibration_frame=calibration_frame,
        migration_frame=migration_frame,
        selected_buildings=selected,
        chunk_manifest=frozen_manifest,
        stored_by_chunk=stored_by_chunk,
    )


def load_frozen_r1_plan(
    paths: ProjectPaths,
    modules: LoadedModules,
    trace: str,
    chunk_size: int,
) -> Any:
    npz_path = (
        paths.results["r1"]
        / "frozen_traces"
        / f"{trace}__size_{chunk_size}.npz"
    )
    json_path = (
        paths.results["r1"]
        / "frozen_traces"
        / f"{trace}__size_{chunk_size}.json"
    )
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(npz_path)
    manifest = read_json(json_path)
    if manifest["trace_file_sha256"] != sha256_file(npz_path):
        raise AssertionError("Frozen R1 trace hash mismatch.")
    with np.load(npz_path, allow_pickle=False) as archive:
        chunk_positions = np.asarray(
            archive["chunk_positions"], dtype=np.int16
        )
        local_indices = np.asarray(
            archive["local_indices"], dtype=np.int32
        )
        query_class_positions = np.asarray(
            archive["query_class_positions"], dtype=np.uint8
        )
        query_choice_positions = np.asarray(
            archive["query_choice_positions"], dtype=np.uint16
        )
    size_manifest = (
        pd.read_csv(paths.results["r1"] / "r1_chunk_manifest.csv")
        .query("chunk_size == @chunk_size")
        .sort_values("building_rank")
    )
    chunk_ids = tuple(size_manifest["chunk_id"].astype(str).tolist())
    return modules.core.LightweightTracePlan(
        trace_name=trace,
        trace_seed=int(
            stable_seed("frozen-r1", trace, chunk_size)
            % (2**31 - 1)
        ),
        chunk_size=int(chunk_size),
        chunk_ids=chunk_ids,
        chunk_positions=chunk_positions,
        local_indices=local_indices,
        query_class_positions=query_class_positions,
        query_choice_positions=query_choice_positions,
        workload_positions=np.zeros(
            len(chunk_positions), dtype=np.uint8
        ),
        workload_names=("continuous_nonstationary",),
    )


def filter_plan_by_chunk_ids(
    modules: LoadedModules,
    plan: Any,
    keep_chunk_ids: Sequence[str],
) -> Any:
    keep_set = set(str(x) for x in keep_chunk_ids)
    old_positions = [
        position
        for position, chunk_id in enumerate(plan.chunk_ids)
        if str(chunk_id) in keep_set
    ]
    old_to_new = {
        old_position: new_position
        for new_position, old_position in enumerate(old_positions)
    }
    mask = np.isin(
        plan.chunk_positions,
        np.asarray(
            old_positions,
            dtype=plan.chunk_positions.dtype,
        ),
    )
    remapped = np.asarray(
        [
            old_to_new[int(position)]
            for position in plan.chunk_positions[mask]
        ],
        dtype=np.int16,
    )
    return modules.core.LightweightTracePlan(
        trace_name=plan.trace_name,
        trace_seed=plan.trace_seed,
        chunk_size=plan.chunk_size,
        chunk_ids=tuple(
            plan.chunk_ids[position]
            for position in old_positions
        ),
        chunk_positions=remapped,
        local_indices=plan.local_indices[mask],
        query_class_positions=plan.query_class_positions[mask],
        query_choice_positions=plan.query_choice_positions[mask],
        workload_positions=plan.workload_positions[mask],
        workload_names=plan.workload_names,
    )


def select_r2_stratified_buildings(
    frozen_buildings: pd.DataFrame,
) -> pd.DataFrame:
    frame = frozen_buildings.copy()
    frame["building_id"] = frame["building_id"].astype(str)
    frame["r2_selection_hash"] = [
        stable_hash(
            "R2-end-to-end",
            row.building_id,
            row.meter_type,
            row.site_id,
        )
        for row in frame.itertuples(index=False)
    ]
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for key, group in frame.groupby(["meter_type", "site_id"]):
        groups[(str(key[0]), str(key[1]))] = (
            group.sort_values("r2_selection_hash").to_dict("records")
        )
    order = sorted(
        groups,
        key=lambda key: (-len(groups[key]), key[0], key[1]),
    )
    cursors = {key: 0 for key in order}
    selected: List[Dict[str, Any]] = []
    while len(selected) < R2_BUILDING_COUNT and order:
        next_order: List[Tuple[str, str]] = []
        for key in order:
            position = cursors[key]
            if position < len(groups[key]):
                selected.append(groups[key][position])
                cursors[key] += 1
                if len(selected) == R2_BUILDING_COUNT:
                    break
            if cursors[key] < len(groups[key]):
                next_order.append(key)
        order = next_order
    if len(selected) != R2_BUILDING_COUNT:
        raise RuntimeError(
            f"Could select only {len(selected)} R2 buildings."
        )
    output = pd.DataFrame(selected)
    output["r2_selection_rank"] = np.arange(len(output), dtype=np.int64)
    return output.sort_values("r2_selection_rank").reset_index(drop=True)


def save_compact_worker_assets(
    paths: ProjectPaths,
    modules: LoadedModules,
    context: R1Context,
) -> Dict[str, Any]:
    asset_dir = paths.output / "_worker_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    selected_buildings = select_r2_stratified_buildings(
        context.selected_buildings
    )
    selected_ids = set(
        selected_buildings["building_id"].astype(str)
    )
    selection_path = asset_dir / "r2_selected_buildings.csv"
    write_csv_atomic(selection_path, selected_buildings)

    values_paths: Dict[str, str] = {}
    filtered_plan_paths: Dict[str, str] = {}
    chunks_by_id = {chunk.chunk_id: chunk for chunk in context.chunks}

    for size in CHUNK_SIZES:
        size_manifest = (
            context.chunk_manifest
            .query("chunk_size == @size")
            .sort_values("building_rank")
        )
        size_manifest = size_manifest[
            size_manifest["building_id"].astype(str).isin(selected_ids)
        ].head(R2_BUILDING_COUNT)
        chunk_ids = tuple(size_manifest["chunk_id"].astype(str))
        values = np.stack(
            [
                np.asarray(
                    chunks_by_id[chunk_id].values,
                    dtype=np.float32,
                )
                for chunk_id in chunk_ids
            ],
            axis=0,
        )
        values_path = asset_dir / f"r2_values_size_{size}.npz"
        if not values_path.exists():
            temporary = values_path.with_name(values_path.name + ".tmp")
            with open(temporary, "wb") as handle:
                np.savez_compressed(
                    handle,
                    values=values,
                    chunk_ids=np.asarray(chunk_ids, dtype="U128"),
                    series_indices=np.asarray(
                        [
                            int(chunks_by_id[chunk_id].series_index)
                            for chunk_id in chunk_ids
                        ],
                        dtype=np.int32,
                    ),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, values_path)
        values_paths[str(size)] = str(values_path)

        for trace in TRACE_FAMILIES:
            full_plan = load_frozen_r1_plan(
                paths,
                modules,
                trace,
                size,
            )
            plan = filter_plan_by_chunk_ids(
                modules,
                full_plan,
                chunk_ids,
            )
            plan_path = asset_dir / f"r2_plan_{trace}_size_{size}.npz"
            if not plan_path.exists():
                temporary = plan_path.with_name(plan_path.name + ".tmp")
                with open(temporary, "wb") as handle:
                    np.savez_compressed(
                        handle,
                        chunk_ids=np.asarray(plan.chunk_ids, dtype="U128"),
                        chunk_positions=plan.chunk_positions,
                        local_indices=plan.local_indices,
                        query_class_positions=plan.query_class_positions,
                        query_choice_positions=plan.query_choice_positions,
                        workload_positions=plan.workload_positions,
                        trace_seed=np.asarray([plan.trace_seed], dtype=np.int64),
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, plan_path)
            filtered_plan_paths[f"{trace}|{size}"] = str(plan_path)

    return {
        "asset_dir": str(asset_dir),
        "selected_buildings_path": str(selection_path),
        "selected_buildings_sha256": sha256_file(selection_path),
        "values_paths": values_paths,
        "plan_paths": filtered_plan_paths,
    }


def load_compact_chunks_and_plan(
    modules: LoadedModules,
    values_path: Path,
    plan_path: Path,
    trace_name: str,
    chunk_size: int,
) -> Tuple[List[Any], Any]:
    with np.load(values_path, allow_pickle=False) as archive:
        values = np.asarray(archive["values"], dtype=np.float32)
        chunk_ids = tuple(str(x) for x in archive["chunk_ids"].tolist())
        series_indices = np.asarray(
            archive["series_indices"], dtype=np.int32
        )
    chunks = [
        modules.base.ChunkUnit(
            chunk_id=chunk_ids[position],
            series_index=int(series_indices[position]),
            chunk_start=0,
            chunk_size=int(chunk_size),
            values=np.ascontiguousarray(values[position], dtype=np.float32),
        )
        for position in range(len(chunk_ids))
    ]
    with np.load(plan_path, allow_pickle=False) as archive:
        plan_chunk_ids = tuple(
            str(x) for x in archive["chunk_ids"].tolist()
        )
        if plan_chunk_ids != chunk_ids:
            raise AssertionError("Worker plan/chunk asset mismatch.")
        plan = modules.core.LightweightTracePlan(
            trace_name=trace_name,
            trace_seed=int(np.asarray(archive["trace_seed"])[0]),
            chunk_size=int(chunk_size),
            chunk_ids=chunk_ids,
            chunk_positions=np.asarray(
                archive["chunk_positions"], dtype=np.int16
            ),
            local_indices=np.asarray(
                archive["local_indices"], dtype=np.int32
            ),
            query_class_positions=np.asarray(
                archive["query_class_positions"], dtype=np.uint8
            ),
            query_choice_positions=np.asarray(
                archive["query_choice_positions"], dtype=np.uint16
            ),
            workload_positions=np.asarray(
                archive["workload_positions"], dtype=np.uint8
            ),
            workload_names=("continuous_nonstationary",),
        )
    return chunks, plan


# ---------------------------------------------------------------------------
# Instrumented single-active store
# ---------------------------------------------------------------------------

class InstrumentedSingleActiveChunkStore:
    """
    Thin composition wrapper over the frozen B1 store.

    Query execution uses B1 directly. Migration is implemented with the same
    B1 protocol and exposes stage times. The protocol order and persistence
    operations remain unchanged.
    """

    def __init__(
        self,
        store: Any,
        b1: Any,
    ):
        self.store = store
        self.b1 = b1

    def __enter__(self) -> "InstrumentedSingleActiveChunkStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.store.close()

    def execute_query(self, chunk_id: str, query: Any) -> Tuple[Any, int, Any, Any]:
        return self.store.execute_query(chunk_id, query)

    def integrity_check(self) -> Dict[str, Any]:
        return self.store.integrity_check()

    def migrate_profiled(
        self,
        chunk_id: str,
        target_codec: str,
        fault_hook: Optional[
            Callable[[str, Dict[str, Any]], None]
        ] = None,
    ) -> Dict[str, Any]:
        s = self.store
        b1 = self.b1
        with s._lock:
            if target_codec not in s.representations:
                raise ValueError(target_codec)
            row = s._row(chunk_id)
            if str(row["migration_state"]) != "idle":
                raise RuntimeError(
                    f"Chunk not idle before migration: {chunk_id}"
                )
            current_codec = str(row["active_codec"])
            current_version = int(row["active_version"])
            if current_codec == target_codec:
                return {
                    "chunk_id": chunk_id,
                    "from_codec": current_codec,
                    "target_codec": target_codec,
                    "from_version": current_version,
                    "to_version": current_version,
                    "migration_time_ns": 0,
                    "rewrite_bytes": 0,
                    "no_op": True,
                    "stage_times_ns": {},
                }

            total_start = time.perf_counter_ns()
            stages: Dict[str, int] = {}
            to_version = current_version + 1
            migration_id = uuid.uuid4().hex
            temporary_token = uuid.uuid4().hex
            temp_stem = s._version_stem(
                chunk_id,
                to_version,
                target_codec,
                temporary_token=temporary_token,
            )
            final_stem = s._version_stem(
                chunk_id,
                to_version,
                target_codec,
            )
            temp_data = temp_stem.with_suffix(".bin")
            temp_meta = temp_stem.with_suffix(".meta.bin")
            final_data = final_stem.with_suffix(".bin")
            final_meta = final_stem.with_suffix(".meta.bin")
            now = time.time_ns()

            stage_start = time.perf_counter_ns()
            with b1.immediate_transaction(s.connection):
                s.connection.execute(
                    """
                    INSERT INTO migration_log(
                        migration_id, chunk_id,
                        from_version, to_version,
                        from_codec, target_codec,
                        state,
                        temporary_data_path,
                        temporary_metadata_path,
                        final_data_path,
                        final_metadata_path,
                        started_at_ns, updated_at_ns
                    ) VALUES (?, ?, ?, ?, ?, ?, 'intent', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        migration_id,
                        chunk_id,
                        current_version,
                        to_version,
                        current_codec,
                        target_codec,
                        b1.relative_path(s.root, temp_data),
                        b1.relative_path(s.root, temp_meta),
                        b1.relative_path(s.root, final_data),
                        b1.relative_path(s.root, final_meta),
                        now,
                        now,
                    ),
                )
                s.connection.execute(
                    """
                    UPDATE chunks
                    SET migration_state='intent', updated_at_ns=?
                    WHERE chunk_id=?
                    """,
                    (now, chunk_id),
                )
            stages["intent_catalog_ns"] = time.perf_counter_ns() - stage_start
            context = {
                "chunk_id": chunk_id,
                "migration_id": migration_id,
                "current_codec": current_codec,
                "target_codec": target_codec,
                "temp_data": temp_data,
                "temp_meta": temp_meta,
                "final_data": final_data,
                "final_meta": final_meta,
            }
            if fault_hook is not None:
                fault_hook("after_intent", context)

            stage_start = time.perf_counter_ns()
            current_stored = s.active_stored(chunk_id)
            values = s._decode_stored(current_stored)
            raw_checksum = str(row["raw_checksum"])
            if b1.raw_array_sha256(values) != raw_checksum:
                raise AssertionError("Invalid active checksum before migration")
            stages["decode_source_ns"] = time.perf_counter_ns() - stage_start

            stage_start = time.perf_counter_ns()
            representation = s.representations[target_codec]
            encoded = representation.encode_to_files(values, temp_stem)
            with b1.immediate_transaction(s.connection):
                s.connection.execute(
                    """
                    UPDATE migration_log
                    SET state='encoded',
                        temporary_data_path=?,
                        temporary_metadata_path=?,
                        target_payload_bytes=?,
                        target_metadata_bytes=?,
                        updated_at_ns=?
                    WHERE migration_id=?
                    """,
                    (
                        b1.relative_path(s.root, encoded.data_path),
                        b1.relative_path(s.root, encoded.metadata_path),
                        int(encoded.payload_size_bytes),
                        int(encoded.metadata_size_bytes),
                        time.time_ns(),
                        migration_id,
                    ),
                )
                s.connection.execute(
                    """
                    UPDATE chunks
                    SET migration_state='encoded', updated_at_ns=?
                    WHERE chunk_id=?
                    """,
                    (time.time_ns(), chunk_id),
                )
            stages["encode_and_log_ns"] = time.perf_counter_ns() - stage_start
            if fault_hook is not None:
                fault_hook("after_encode", context)

            stage_start = time.perf_counter_ns()
            if not s._validate_stored(
                encoded,
                raw_checksum,
                expected_values=values,
            ):
                raise AssertionError("Target validation failed")
            with b1.immediate_transaction(s.connection):
                s.connection.execute(
                    """
                    UPDATE migration_log
                    SET state='validated', updated_at_ns=?
                    WHERE migration_id=?
                    """,
                    (time.time_ns(), migration_id),
                )
                s.connection.execute(
                    """
                    UPDATE chunks
                    SET migration_state='validated', updated_at_ns=?
                    WHERE chunk_id=?
                    """,
                    (time.time_ns(), chunk_id),
                )
            stages["validate_and_log_ns"] = time.perf_counter_ns() - stage_start
            if fault_hook is not None:
                fault_hook("after_validate", context)

            stage_start = time.perf_counter_ns()
            final_data.parent.mkdir(parents=True, exist_ok=True)
            b1.safe_unlink(final_data)
            b1.safe_unlink(final_meta)
            os.replace(encoded.data_path, final_data)
            b1.fsync_directory(final_data.parent)
            if encoded.metadata_path is not None:
                os.replace(encoded.metadata_path, final_meta)
                final_metadata_path: Optional[Path] = final_meta
            else:
                b1.safe_unlink(temp_meta)
                final_metadata_path = None
            b1.fsync_directory(final_data.parent)
            stages["publish_files_ns"] = time.perf_counter_ns() - stage_start
            context["final_metadata_path"] = final_metadata_path
            if fault_hook is not None:
                fault_hook("after_full_publish_before_catalog_commit", context)

            published = s.base.StoredChunk(
                codec=target_codec,
                n=int(row["n"]),
                data_path=final_data,
                metadata_path=final_metadata_path,
                payload_size_bytes=int(final_data.stat().st_size),
                metadata_size_bytes=(
                    int(final_metadata_path.stat().st_size)
                    if final_metadata_path is not None
                    else 0
                ),
                metadata=(
                    s.base.parse_block_metadata(
                        final_metadata_path.read_bytes()
                    )
                    if final_metadata_path is not None
                    else (
                        {"level": b1.ZSTD_LEVEL}
                        if target_codec == "whole_zstd"
                        else {}
                    )
                ),
            )

            stage_start = time.perf_counter_ns()
            with b1.immediate_transaction(s.connection):
                s.connection.execute(
                    """
                    UPDATE migration_log
                    SET state='published',
                        final_data_path=?,
                        final_metadata_path=?,
                        target_payload_bytes=?,
                        target_metadata_bytes=?,
                        updated_at_ns=?
                    WHERE migration_id=?
                    """,
                    (
                        b1.relative_path(s.root, published.data_path),
                        b1.relative_path(s.root, published.metadata_path),
                        int(published.payload_size_bytes),
                        int(published.metadata_size_bytes),
                        time.time_ns(),
                        migration_id,
                    ),
                )
                s.connection.execute(
                    """
                    UPDATE chunks
                    SET migration_state='published', updated_at_ns=?
                    WHERE chunk_id=?
                    """,
                    (time.time_ns(), chunk_id),
                )
            stages["publish_catalog_state_ns"] = (
                time.perf_counter_ns() - stage_start
            )

            stage_start = time.perf_counter_ns()
            with b1.immediate_transaction(s.connection):
                s.connection.execute(
                    """
                    UPDATE chunks
                    SET
                        previous_version=active_version,
                        previous_codec=active_codec,
                        previous_data_path=active_data_path,
                        previous_metadata_path=active_metadata_path,
                        previous_payload_bytes=active_payload_bytes,
                        previous_metadata_bytes=active_metadata_bytes,
                        active_version=?,
                        active_codec=?,
                        active_data_path=?,
                        active_metadata_path=?,
                        active_payload_bytes=?,
                        active_metadata_bytes=?,
                        migration_state='committed_pending_cleanup',
                        updated_at_ns=?
                    WHERE chunk_id=?
                    """,
                    (
                        to_version,
                        target_codec,
                        b1.relative_path(s.root, published.data_path),
                        b1.relative_path(s.root, published.metadata_path),
                        int(published.payload_size_bytes),
                        int(published.metadata_size_bytes),
                        time.time_ns(),
                        chunk_id,
                    ),
                )
                s.connection.execute(
                    """
                    UPDATE migration_log
                    SET state='catalog_committed', updated_at_ns=?
                    WHERE migration_id=?
                    """,
                    (time.time_ns(), migration_id),
                )
            s.invalidate_handle(chunk_id)
            stages["catalog_commit_ns"] = time.perf_counter_ns() - stage_start
            if fault_hook is not None:
                fault_hook("after_catalog_commit", context)

            stage_start = time.perf_counter_ns()
            refreshed = s._row(chunk_id)
            active_after_commit = s.active_stored(chunk_id)
            if not s._validate_stored(
                active_after_commit,
                str(refreshed["raw_checksum"]),
                expected_values=values,
            ):
                raise AssertionError("Post-commit validation failed")
            stages["post_commit_validate_ns"] = (
                time.perf_counter_ns() - stage_start
            )

            stage_start = time.perf_counter_ns()
            previous = s.previous_stored(chunk_id)
            s._delete_stored_files(previous)
            with b1.immediate_transaction(s.connection):
                s.connection.execute(
                    """
                    UPDATE chunks
                    SET
                        previous_version=NULL,
                        previous_codec=NULL,
                        previous_data_path=NULL,
                        previous_metadata_path=NULL,
                        previous_payload_bytes=NULL,
                        previous_metadata_bytes=NULL,
                        migration_state='idle',
                        updated_at_ns=?
                    WHERE chunk_id=?
                    """,
                    (time.time_ns(), chunk_id),
                )
                s.connection.execute(
                    """
                    UPDATE migration_log
                    SET state='completed',
                        completed_at_ns=?,
                        updated_at_ns=?
                    WHERE migration_id=?
                    """,
                    (
                        time.time_ns(),
                        time.time_ns(),
                        migration_id,
                    ),
                )
            stages["cleanup_ns"] = time.perf_counter_ns() - stage_start
            if fault_hook is not None:
                fault_hook("after_cleanup", context)

            total_ns = time.perf_counter_ns() - total_start
            return {
                "chunk_id": chunk_id,
                "migration_id": migration_id,
                "from_codec": current_codec,
                "target_codec": target_codec,
                "from_version": current_version,
                "to_version": to_version,
                "migration_time_ns": int(total_ns),
                "rewrite_bytes": int(published.stored_size_bytes),
                "no_op": False,
                "stage_times_ns": stages,
            }


# ---------------------------------------------------------------------------
# Frozen policy controllers and model replay
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MigrationEpisode:
    chunk_id: str
    policy: str
    start_global_event_index: int
    start_local_query_index: int
    source_position: int
    target_position: int
    source_codec: str
    target_codec: str
    migration_cost_ns: float
    rewrite_bytes: int
    realized_query_saving_ns: float = 0.0
    end_global_event_index: Optional[int] = None
    unamortized: Optional[bool] = None
    stale_assessable: bool = False
    stale: bool = False


@dataclasses.dataclass
class ModelChunkState:
    active_position: int
    local_access_count: int = 0
    query_time_ns: float = 0.0
    migration_time_ns: float = 0.0
    migration_count: int = 0
    rewrite_bytes: int = 0
    open_episode: Optional[MigrationEpisode] = None
    migration_records: List[MigrationEpisode] = dataclasses.field(
        default_factory=list
    )
    decision_records: List[Dict[str, Any]] = dataclasses.field(
        default_factory=list
    )


@dataclasses.dataclass(frozen=True)
class RepressVariant:
    name: str
    use_long_agreement: bool = True
    confirmation_rounds: int = 3
    break_even_factor: float = 1.25
    cooldown_queries: int = 300
    use_write_budget: bool = True
    long_window: int = 300
    short_window: int = 100
    min_observations: int = 200
    decision_interval: int = 100
    prediction_horizon: int = 1200
    min_relative_improvement: float = 0.08


REPRESS_VARIANTS = (
    RepressVariant(name="full_repressdb"),
    RepressVariant(
        name="short_window_only",
        use_long_agreement=False,
    ),
    RepressVariant(
        name="no_confirmation",
        confirmation_rounds=1,
    ),
    RepressVariant(
        name="no_break_even",
        break_even_factor=0.0,
    ),
    RepressVariant(
        name="no_cooldown",
        cooldown_queries=0,
    ),
    RepressVariant(
        name="no_write_budget",
        use_write_budget=False,
    ),
)


class ShapeEWMAController:
    def __init__(
        self,
        initial_position: int,
        initial_weights: np.ndarray,
        config: Mapping[str, Any],
    ):
        self.active_position = int(initial_position)
        self.ewma = np.asarray(initial_weights, dtype=np.float64).copy()
        self.local_count = 0
        self.alpha = float(config["alpha"])
        self.deadband = float(config["deadband"])
        self.horizon = int(config["prediction_horizon"])

    def observe_and_decide(
        self,
        query_position: int,
        decision_profile: np.ndarray,
        decision_migration_cost: np.ndarray,
        codecs: Sequence[str],
        global_event_index: int,
        chunk_id: str,
    ) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
        self.local_count += 1
        self.ewma *= 1.0 - self.alpha
        self.ewma[int(query_position)] += self.alpha
        if (
            self.local_count < MIN_OBSERVATIONS
            or self.local_count % DECISION_INTERVAL != 0
        ):
            return None, None

        expected = decision_profile @ self.ewma
        current = int(self.active_position)
        target = int(np.argmin(expected))
        current_cost = float(expected[current])
        target_cost = float(expected[target])
        gain = current_cost - target_cost
        relative = gain / current_cost if current_cost > 0 else 0.0
        migration_estimate = float(
            decision_migration_cost[current, target]
        )
        predicted_saving = gain * self.horizon
        reason = "current_layout_is_best"
        proposal: Optional[int] = None
        if target != current:
            if relative < self.deadband:
                reason = "deadband"
            elif predicted_saving <= migration_estimate:
                reason = "break_even_not_met"
            else:
                reason = "proposal"
                proposal = target

        return proposal, {
            "global_event_index": int(global_event_index),
            "chunk_id": str(chunk_id),
            "local_query_index": int(self.local_count - 1),
            "old_codec": str(codecs[current]),
            "estimated_best_codec": str(codecs[target]),
            "candidate_codec": (
                str(codecs[proposal])
                if proposal is not None
                else str(codecs[current])
            ),
            "reason": reason,
            "current_expected_ns": current_cost,
            "target_expected_ns": target_cost,
            "relative_improvement": relative,
            "predicted_saving_ns": predicted_saving,
            "migration_estimate_ns": migration_estimate,
        }

    def committed_switch(self, target_position: int) -> None:
        self.active_position = int(target_position)


class RepressController:
    def __init__(
        self,
        initial_position: int,
        variant: RepressVariant,
    ):
        self.active_position = int(initial_position)
        self.variant = variant
        self.recent_classes: Deque[int] = deque(
            maxlen=variant.long_window
        )
        self.local_count = 0
        self.last_switch_local_index = -10**9
        self.pending_candidate: Optional[int] = None
        self.pending_rounds = 0

    def observe_and_decide(
        self,
        query_position: int,
        decision_profile: np.ndarray,
        decision_migration_cost: np.ndarray,
        codecs: Sequence[str],
        global_event_index: int,
        chunk_id: str,
    ) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
        self.local_count += 1
        self.recent_classes.append(int(query_position))
        v = self.variant
        if (
            self.local_count < v.min_observations
            or self.local_count % v.decision_interval != 0
        ):
            return None, None

        recent_long = np.asarray(
            list(self.recent_classes),
            dtype=np.int64,
        )
        recent_short = recent_long[-v.short_window :]
        long_weights = np.bincount(
            recent_long,
            minlength=decision_profile.shape[1],
        ).astype(np.float64)
        short_weights = np.bincount(
            recent_short,
            minlength=decision_profile.shape[1],
        ).astype(np.float64)
        long_weights /= long_weights.sum()
        short_weights /= short_weights.sum()

        long_expected = decision_profile @ long_weights
        short_expected = decision_profile @ short_weights
        long_target = int(np.argmin(long_expected))
        short_target = int(np.argmin(short_expected))
        current = int(self.active_position)

        if v.use_long_agreement:
            target = long_target
            current_long = float(long_expected[current])
            current_short = float(short_expected[current])
            target_long = float(long_expected[target])
            target_short = float(short_expected[target])
            long_relative = (
                (current_long - target_long) / current_long
                if current_long > 0
                else 0.0
            )
            short_relative = (
                (current_short - target_short) / current_short
                if current_short > 0
                else 0.0
            )
            relative = min(long_relative, short_relative)
            conservative_gain = max(
                0.0,
                min(
                    current_long - target_long,
                    current_short - target_short,
                ),
            )
            disagreement = long_target != short_target
            current_expected = current_long
            target_expected = target_long
        else:
            target = short_target
            current_expected = float(short_expected[current])
            target_expected = float(short_expected[target])
            conservative_gain = max(
                0.0,
                current_expected - target_expected,
            )
            relative = (
                conservative_gain / current_expected
                if current_expected > 0
                else 0.0
            )
            disagreement = False

        migration_estimate = float(
            decision_migration_cost[current, target]
        )
        predicted_saving = (
            conservative_gain * v.prediction_horizon
        )
        local_query_index = self.local_count - 1
        reason = ""
        proposal: Optional[int] = None

        if disagreement:
            self.pending_candidate = None
            self.pending_rounds = 0
            reason = "timescale_disagreement"
        elif target == current:
            self.pending_candidate = None
            self.pending_rounds = 0
            reason = "current_layout_is_best"
        elif (
            local_query_index - self.last_switch_local_index
            < v.cooldown_queries
        ):
            self.pending_candidate = None
            self.pending_rounds = 0
            reason = "cooldown"
        elif relative < v.min_relative_improvement:
            self.pending_candidate = None
            self.pending_rounds = 0
            reason = "relative_gain_too_small"
        elif (
            predicted_saving
            < v.break_even_factor * migration_estimate
        ):
            self.pending_candidate = None
            self.pending_rounds = 0
            reason = "break_even_not_met"
        else:
            if self.pending_candidate == target:
                self.pending_rounds += 1
            else:
                self.pending_candidate = target
                self.pending_rounds = 1
            if self.pending_rounds < v.confirmation_rounds:
                reason = "awaiting_confirmation"
            else:
                proposal = target
                reason = "proposal"
                self.pending_candidate = None
                self.pending_rounds = 0

        return proposal, {
            "global_event_index": int(global_event_index),
            "chunk_id": str(chunk_id),
            "local_query_index": int(local_query_index),
            "old_codec": str(codecs[current]),
            "estimated_best_codec": str(codecs[target]),
            "candidate_codec": (
                str(codecs[proposal])
                if proposal is not None
                else str(codecs[current])
            ),
            "reason": reason,
            "current_expected_ns": current_expected,
            "target_expected_ns": target_expected,
            "relative_improvement": relative,
            "predicted_saving_ns": predicted_saving,
            "migration_estimate_ns": migration_estimate,
            "long_target_codec": str(codecs[long_target]),
            "short_target_codec": str(codecs[short_target]),
        }

    def committed_switch(self, target_position: int) -> None:
        self.active_position = int(target_position)
        self.last_switch_local_index = self.local_count - 1


@dataclasses.dataclass
class PageHinkleyOneSided:
    delta: float
    threshold: float
    count: int = 0
    mean: float = 0.0
    cumulative: float = 0.0
    minimum: float = 0.0

    def update(self, value: float) -> bool:
        self.count += 1
        self.mean += (float(value) - self.mean) / self.count
        self.cumulative += float(value) - self.mean - self.delta
        self.minimum = min(self.minimum, self.cumulative)
        return bool(
            self.cumulative - self.minimum > self.threshold
        )

    def reset(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.cumulative = 0.0
        self.minimum = 0.0


class TwoSidedPageHinkley:
    def __init__(self, delta: float, threshold: float):
        self.positive = PageHinkleyOneSided(delta, threshold)
        self.negative = PageHinkleyOneSided(delta, threshold)

    def update(self, value: float) -> bool:
        positive = self.positive.update(float(value))
        negative = self.negative.update(-float(value))
        return bool(positive or negative)

    def reset(self) -> None:
        self.positive.reset()
        self.negative.reset()


def coarse_category_index(query_class: str) -> int:
    if query_class == "point":
        return 0
    if "scan" in query_class:
        return 1
    return 2


class PageHinkleyBEController:
    def __init__(
        self,
        initial_position: int,
        config: Mapping[str, Any],
        query_classes: Sequence[str],
    ):
        self.active_position = int(initial_position)
        self.delta = float(config["delta"])
        self.threshold = float(config["threshold"])
        self.window_cap = int(config["window_cap"])
        self.deadband = float(config["deadband"])
        self.horizon = int(config["prediction_horizon"])
        self.query_classes = tuple(query_classes)
        self.local_count = 0
        self.history: Deque[int] = deque(
            maxlen=self.window_cap
        )
        self.detectors = [
            TwoSidedPageHinkley(
                self.delta,
                self.threshold,
            )
            for _ in range(3)
        ]

    def observe_and_decide(
        self,
        query_position: int,
        decision_profile: np.ndarray,
        decision_migration_cost: np.ndarray,
        codecs: Sequence[str],
        global_event_index: int,
        chunk_id: str,
    ) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
        query_position = int(query_position)
        self.local_count += 1
        self.history.append(query_position)
        category = coarse_category_index(
            self.query_classes[query_position]
        )
        signal = False
        for category_position, detector in enumerate(self.detectors):
            signal = (
                detector.update(
                    1.0
                    if category_position == category
                    else 0.0
                )
                or signal
            )
        if (
            not signal
            or self.local_count < MIN_OBSERVATIONS
        ):
            return None, None

        observed = np.asarray(
            list(self.history),
            dtype=np.int64,
        )
        weights = np.bincount(
            observed,
            minlength=decision_profile.shape[1],
        ).astype(np.float64)
        weights /= weights.sum()
        expected = decision_profile @ weights
        current = int(self.active_position)
        target = int(np.argmin(expected))
        current_cost = float(expected[current])
        target_cost = float(expected[target])
        gain = current_cost - target_cost
        relative = gain / current_cost if current_cost > 0 else 0.0
        migration_estimate = float(
            decision_migration_cost[current, target]
        )
        predicted_saving = gain * self.horizon
        proposal: Optional[int] = None
        reason = "current_layout_is_best"
        if target != current:
            if relative < self.deadband:
                reason = "deadband"
            elif predicted_saving <= migration_estimate:
                reason = "break_even_not_met"
            else:
                proposal = target
                reason = "proposal"

        for detector in self.detectors:
            detector.reset()
        # Keep the recent history as the post-detection estimation window.
        return proposal, {
            "global_event_index": int(global_event_index),
            "chunk_id": str(chunk_id),
            "local_query_index": int(self.local_count - 1),
            "old_codec": str(codecs[current]),
            "estimated_best_codec": str(codecs[target]),
            "candidate_codec": (
                str(codecs[proposal])
                if proposal is not None
                else str(codecs[current])
            ),
            "reason": reason,
            "current_expected_ns": current_cost,
            "target_expected_ns": target_cost,
            "relative_improvement": relative,
            "predicted_saving_ns": predicted_saving,
            "migration_estimate_ns": migration_estimate,
            "detected_category": int(category),
        }

    def committed_switch(self, target_position: int) -> None:
        self.active_position = int(target_position)


def initial_weights_from_base(
    modules: LoadedModules,
    query_classes: Sequence[str],
) -> np.ndarray:
    return modules.s1.normalized_initial_weights(
        modules.base,
        query_classes,
    )


def close_model_episode(
    state: ModelChunkState,
    end_global_event_index: int,
) -> None:
    if state.open_episode is None:
        return
    episode = state.open_episode
    episode.end_global_event_index = int(end_global_event_index)
    episode.unamortized = bool(
        float(episode.realized_query_saving_ns)
        < float(episode.migration_cost_ns)
    )
    state.migration_records.append(episode)
    state.open_episode = None


def static_model_result(
    plan: Any,
    evaluation_profile: np.ndarray,
    positions: np.ndarray,
    policy_name: str,
) -> Dict[str, Any]:
    n_chunks = len(plan.chunk_ids)
    per_chunk = np.zeros(n_chunks, dtype=np.float64)
    cumulative: List[Dict[str, Any]] = []
    running = 0.0
    for event_index in range(plan.n_events):
        chunk_position = int(plan.chunk_positions[event_index])
        query_position = int(
            plan.query_class_positions[event_index]
        )
        value = float(
            evaluation_profile[
                chunk_position,
                int(positions[chunk_position]),
                query_position,
            ]
        )
        per_chunk[chunk_position] += value
        running += value
        if (
            event_index % 5000 == 0
            or event_index == plan.n_events - 1
        ):
            cumulative.append(
                {
                    "global_event_index": int(event_index),
                    "policy": policy_name,
                    "cumulative_query_time_ns": float(running),
                    "cumulative_migration_time_ns": 0.0,
                    "cumulative_total_time_ns": float(running),
                    "cumulative_migrations": 0,
                    "cumulative_budget_denials": 0,
                }
            )
    total = float(per_chunk.sum())
    return {
        "query_time_ns": total,
        "migration_time_ns": 0.0,
        "total_time_ns": total,
        "migration_count": 0,
        "rewrite_bytes": 0,
        "budget_denied_proposals": 0,
        "budget_violation_count": 0,
        "unamortized_migration_count": 0,
        "unamortized_migration_rate": 0.0,
        "stale_assessable_migration_count": 0,
        "stale_migration_count": 0,
        "stale_migration_rate": 0.0,
        "per_chunk_query_time_ns": per_chunk,
        "per_chunk_migration_time_ns": np.zeros(
            n_chunks, dtype=np.float64
        ),
        "per_chunk_migration_count": np.zeros(
            n_chunks, dtype=np.int64
        ),
        "per_chunk_rewrite_bytes": np.zeros(
            n_chunks, dtype=np.int64
        ),
        "migration_records": [],
        "decision_records": [],
        "cumulative_records": cumulative,
    }


def best_static_position_array(
    plan: Any,
    evaluation_profile: np.ndarray,
) -> np.ndarray:
    n_chunks, _, n_classes = evaluation_profile.shape
    counts = np.zeros(
        (n_chunks, n_classes),
        dtype=np.int64,
    )
    np.add.at(
        counts,
        (
            plan.chunk_positions.astype(np.int64),
            plan.query_class_positions.astype(np.int64),
        ),
        1,
    )
    costs = np.einsum(
        "ncq,nq->nc",
        evaluation_profile,
        counts,
        optimize=True,
    )
    return np.argmin(costs, axis=1).astype(np.int16)


def assign_common_stale_flags(
    plan: Any,
    result: Dict[str, Any],
    evaluation_profile: np.ndarray,
    codecs: Sequence[str],
) -> None:
    migrations = result.get("migration_records", [])
    if not migrations:
        result["stale_assessable_migration_count"] = 0
        result["stale_migration_count"] = 0
        result["stale_migration_rate"] = 0.0
        return

    local_histories: Dict[str, List[int]] = {
        str(chunk_id): []
        for chunk_id in plan.chunk_ids
    }
    decision_map: Dict[str, List[Tuple[int, str]]] = {
        str(chunk_id): []
        for chunk_id in plan.chunk_ids
    }
    for event_index in range(plan.n_events):
        chunk_position = int(plan.chunk_positions[event_index])
        chunk_id = str(plan.chunk_ids[chunk_position])
        query_position = int(
            plan.query_class_positions[event_index]
        )
        history = local_histories[chunk_id]
        history.append(query_position)
        if (
            len(history) >= MIN_OBSERVATIONS
            and len(history) % DECISION_INTERVAL == 0
        ):
            recent = np.asarray(
                history[-DECISION_INTERVAL:],
                dtype=np.int64,
            )
            weights = np.bincount(
                recent,
                minlength=evaluation_profile.shape[2],
            ).astype(np.float64)
            weights /= weights.sum()
            expected = evaluation_profile[chunk_position] @ weights
            best_codec = str(
                codecs[int(np.argmin(expected))]
            )
            decision_map[chunk_id].append(
                (int(event_index), best_codec)
            )

    assessable = 0
    stale = 0
    for episode in migrations:
        if dataclasses.is_dataclass(episode):
            episode_object = episode
            chunk_id = str(episode.chunk_id)
            start = int(episode.start_global_event_index)
            target_codec = str(episode.target_codec)
        else:
            episode_object = episode
            chunk_id = str(episode["chunk_id"])
            start = int(episode["start_global_event_index"])
            target_codec = str(episode["target_codec"])
        later = [
            item
            for item in decision_map[chunk_id]
            if item[0] > start
        ][:2]
        is_assessable = len(later) == 2
        is_stale = bool(
            is_assessable
            and all(
                best_codec != target_codec
                for _, best_codec in later
            )
        )
        if dataclasses.is_dataclass(episode_object):
            episode_object.stale_assessable = is_assessable
            episode_object.stale = is_stale
        else:
            episode_object["stale_assessable"] = is_assessable
            episode_object["stale"] = is_stale
            episode_object["later_best_codecs"] = [
                best_codec for _, best_codec in later
            ]
        assessable += int(is_assessable)
        stale += int(is_stale)

    result["stale_assessable_migration_count"] = int(assessable)
    result["stale_migration_count"] = int(stale)
    result["stale_migration_rate"] = (
        float(stale / assessable)
        if assessable
        else 0.0
    )


def simulate_controller_model(
    policy_name: str,
    controller_factory: Callable[[int], Any],
    plan: Any,
    evaluation_profile: np.ndarray,
    decision_profile: np.ndarray,
    evaluation_migration_cost: np.ndarray,
    decision_migration_cost: np.ndarray,
    migration_bytes: np.ndarray,
    initial_positions: np.ndarray,
    codecs: Sequence[str],
    use_write_budget: bool = True,
) -> Dict[str, Any]:
    n_chunks = len(plan.chunk_ids)
    states = [
        ModelChunkState(
            active_position=int(initial_positions[position])
        )
        for position in range(n_chunks)
    ]
    controllers = [
        controller_factory(
            int(initial_positions[position])
        )
        for position in range(n_chunks)
    ]

    budget_bytes = int(migration_bytes.max())
    remaining = budget_bytes
    interval_id = 0
    spent = 0
    denials = 0
    violations = 0
    running_query = 0.0
    running_migration = 0.0
    cumulative: List[Dict[str, Any]] = []

    for event_index in range(plan.n_events):
        new_interval = (
            event_index // BUDGET_INTERVAL_GLOBAL_QUERIES
        )
        if new_interval != interval_id:
            if use_write_budget and spent > budget_bytes:
                violations += 1
            interval_id = new_interval
            remaining = budget_bytes
            spent = 0

        chunk_position = int(plan.chunk_positions[event_index])
        query_position = int(
            plan.query_class_positions[event_index]
        )
        state = states[chunk_position]
        controller = controllers[chunk_position]
        current = int(state.active_position)

        query_cost = float(
            evaluation_profile[
                chunk_position,
                current,
                query_position,
            ]
        )
        state.query_time_ns += query_cost
        running_query += query_cost
        state.local_access_count += 1

        if state.open_episode is not None:
            episode = state.open_episode
            episode.realized_query_saving_ns += float(
                evaluation_profile[
                    chunk_position,
                    episode.source_position,
                    query_position,
                ]
                - evaluation_profile[
                    chunk_position,
                    episode.target_position,
                    query_position,
                ]
            )

        proposal, decision = controller.observe_and_decide(
            query_position=query_position,
            decision_profile=decision_profile[chunk_position],
            decision_migration_cost=decision_migration_cost[
                chunk_position
            ],
            codecs=codecs,
            global_event_index=event_index,
            chunk_id=str(plan.chunk_ids[chunk_position]),
        )
        if decision is not None:
            decision = dict(decision)
            decision["policy"] = policy_name
            decision["budget_remaining_before"] = (
                int(remaining)
                if use_write_budget
                else math.nan
            )
            decision["decision"] = "stay"

            if proposal is not None and int(proposal) != current:
                rewrite = int(
                    migration_bytes[
                        chunk_position,
                        current,
                        int(proposal),
                    ]
                )
                admitted = (
                    not use_write_budget
                    or rewrite <= remaining
                )
                if admitted:
                    close_model_episode(
                        state,
                        event_index,
                    )
                    migration_cost = float(
                        evaluation_migration_cost[
                            chunk_position,
                            current,
                            int(proposal),
                        ]
                    )
                    state.migration_time_ns += migration_cost
                    state.migration_count += 1
                    state.rewrite_bytes += rewrite
                    state.active_position = int(proposal)
                    controller.committed_switch(int(proposal))
                    running_migration += migration_cost
                    if use_write_budget:
                        remaining -= rewrite
                        spent += rewrite
                    decision["decision"] = "switch"
                    state.open_episode = MigrationEpisode(
                        chunk_id=str(
                            plan.chunk_ids[chunk_position]
                        ),
                        policy=policy_name,
                        start_global_event_index=int(event_index),
                        start_local_query_index=int(
                            state.local_access_count - 1
                        ),
                        source_position=current,
                        target_position=int(proposal),
                        source_codec=str(codecs[current]),
                        target_codec=str(codecs[int(proposal)]),
                        migration_cost_ns=migration_cost,
                        rewrite_bytes=rewrite,
                    )
                else:
                    denials += 1
                    decision["decision"] = "budget_denied"
                    decision["reason"] = "write_budget"
            decision["budget_remaining_after"] = (
                int(remaining)
                if use_write_budget
                else math.nan
            )
            state.decision_records.append(decision)

        if (
            event_index % 5000 == 0
            or event_index == plan.n_events - 1
        ):
            cumulative.append(
                {
                    "global_event_index": int(event_index),
                    "policy": policy_name,
                    "cumulative_query_time_ns": float(running_query),
                    "cumulative_migration_time_ns": float(
                        running_migration
                    ),
                    "cumulative_total_time_ns": float(
                        running_query + running_migration
                    ),
                    "cumulative_migrations": int(
                        sum(
                            local_state.migration_count
                            for local_state in states
                        )
                    ),
                    "cumulative_budget_denials": int(denials),
                }
            )

    if use_write_budget and spent > budget_bytes:
        violations += 1
    for state in states:
        close_model_episode(
            state,
            plan.n_events,
        )

    migrations = [
        episode
        for state in states
        for episode in state.migration_records
    ]
    decisions = [
        row
        for state in states
        for row in state.decision_records
    ]
    per_query = np.asarray(
        [state.query_time_ns for state in states],
        dtype=np.float64,
    )
    per_migration = np.asarray(
        [state.migration_time_ns for state in states],
        dtype=np.float64,
    )
    per_count = np.asarray(
        [state.migration_count for state in states],
        dtype=np.int64,
    )
    per_rewrite = np.asarray(
        [state.rewrite_bytes for state in states],
        dtype=np.int64,
    )
    unamortized = int(
        sum(
            bool(episode.unamortized)
            for episode in migrations
        )
    )
    total_query = float(per_query.sum())
    total_migration = float(per_migration.sum())
    result = {
        "query_time_ns": total_query,
        "migration_time_ns": total_migration,
        "total_time_ns": total_query + total_migration,
        "migration_count": int(per_count.sum()),
        "rewrite_bytes": int(per_rewrite.sum()),
        "budget_denied_proposals": int(denials),
        "budget_violation_count": int(violations),
        "unamortized_migration_count": int(unamortized),
        "unamortized_migration_rate": (
            float(unamortized / len(migrations))
            if migrations
            else 0.0
        ),
        "stale_assessable_migration_count": 0,
        "stale_migration_count": 0,
        "stale_migration_rate": 0.0,
        "per_chunk_query_time_ns": per_query,
        "per_chunk_migration_time_ns": per_migration,
        "per_chunk_migration_count": per_count,
        "per_chunk_rewrite_bytes": per_rewrite,
        "migration_records": migrations,
        "decision_records": decisions,
        "cumulative_records": cumulative,
    }
    assign_common_stale_flags(
        plan,
        result,
        evaluation_profile,
        codecs,
    )
    return result


def result_to_global_row(
    result: Mapping[str, Any],
    trace: str,
    chunk_size: int,
    trace_seed: int,
    policy: str,
    n_buildings: int,
    n_queries: int,
) -> Dict[str, Any]:
    return {
        "trace": trace,
        "chunk_size": int(chunk_size),
        "trace_seed": int(trace_seed),
        "policy": policy,
        "n_buildings": int(n_buildings),
        "n_queries": int(n_queries),
        "query_time_ns": float(result["query_time_ns"]),
        "migration_time_ns": float(
            result["migration_time_ns"]
        ),
        "total_time_ns": float(result["total_time_ns"]),
        "migration_count": int(result["migration_count"]),
        "rewrite_bytes": int(result["rewrite_bytes"]),
        "budget_denied_proposals": int(
            result.get("budget_denied_proposals", 0)
        ),
        "budget_violation_count": int(
            result.get("budget_violation_count", 0)
        ),
        "unamortized_migration_count": int(
            result.get("unamortized_migration_count", 0)
        ),
        "unamortized_migration_rate": float(
            result.get("unamortized_migration_rate", 0.0)
        ),
        "stale_assessable_migration_count": int(
            result.get("stale_assessable_migration_count", 0)
        ),
        "stale_migration_count": int(
            result.get("stale_migration_count", 0)
        ),
        "stale_migration_rate": float(
            result.get("stale_migration_rate", 0.0)
        ),
    }


def migration_records_to_rows(
    result: Mapping[str, Any],
    trace: str,
    chunk_size: int,
    trace_seed: int,
    policy: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for episode in result.get("migration_records", []):
        if dataclasses.is_dataclass(episode):
            payload = dataclasses.asdict(episode)
        else:
            payload = dict(episode)
        rows.append(
            {
                "trace": trace,
                "chunk_size": int(chunk_size),
                "trace_seed": int(trace_seed),
                "policy": policy,
                **payload,
            }
        )
    return rows


def chunk_results_to_rows(
    result: Mapping[str, Any],
    plan: Any,
    metadata: Mapping[str, Mapping[str, Any]],
    trace_seed: int,
    policy: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for position, chunk_id in enumerate(plan.chunk_ids):
        meta = metadata[str(chunk_id)]
        rows.append(
            {
                "trace": plan.trace_name,
                "chunk_size": int(plan.chunk_size),
                "trace_seed": int(trace_seed),
                "policy": policy,
                "chunk_id": str(chunk_id),
                "building_id": str(meta["building_id"]),
                "meter_type": str(meta.get("meter_type", "")),
                "site_id": str(meta.get("site_id", "")),
                "query_time_ns": float(
                    result["per_chunk_query_time_ns"][position]
                ),
                "migration_time_ns": float(
                    result["per_chunk_migration_time_ns"][position]
                ),
                "total_time_ns": float(
                    result["per_chunk_query_time_ns"][position]
                    + result["per_chunk_migration_time_ns"][position]
                ),
                "migration_count": int(
                    result["per_chunk_migration_count"][position]
                ),
                "rewrite_bytes": int(
                    result["per_chunk_rewrite_bytes"][position]
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# J1: end-to-end execution on the single-active store
# ---------------------------------------------------------------------------

def create_r2_base_stores(
    paths: ProjectPaths,
    modules: LoadedModules,
    context: R1Context,
    worker_assets: Mapping[str, Any],
    freeze_hash: str,
) -> Dict[str, str]:
    base_root = paths.output / "j1_end_to_end" / "_base_stores"
    base_root.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(
        Path(worker_assets["selected_buildings_path"])
    )
    selected_ids = set(
        selected["building_id"].astype(str)
    )
    chunks_by_id = {
        str(chunk.chunk_id): chunk
        for chunk in context.chunks
    }
    outputs: Dict[str, str] = {}

    for size in CHUNK_SIZES:
        root = base_root / f"size_{size}"
        marker_path = root / "base_store_complete.json"
        if checkpoint_complete(marker_path, freeze_hash):
            with modules.b1.SingleActiveChunkStore(
                root,
                modules.base,
                create=True,
                recover_on_open=True,
            ) as store:
                integrity = store.integrity_check()
            if not integrity["pass"]:
                raise RuntimeError(
                    f"Existing R2 base store failed integrity: {root}"
                )
            outputs[str(size)] = str(root)
            continue

        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        size_manifest = (
            context.chunk_manifest
            .query("chunk_size == @size")
            .sort_values("building_rank")
        )
        size_manifest = size_manifest[
            size_manifest["building_id"].astype(str).isin(
                selected_ids
            )
        ].head(R2_BUILDING_COUNT)
        with modules.b1.SingleActiveChunkStore(
            root,
            modules.base,
            create=True,
            recover_on_open=False,
        ) as store:
            for row in size_manifest.itertuples(index=False):
                chunk_id = str(row.chunk_id)
                initial_codec, _ = modules.e1.choose_best_codec(
                    context.profiles[chunk_id],
                    modules.base.WORKLOADS["scan_heavy"],
                )
                chunk = chunks_by_id[chunk_id]
                store.create_chunk(
                    chunk_id=chunk_id,
                    series_index=int(chunk.series_index),
                    values=chunk.values,
                    codec=initial_codec,
                )
            integrity = store.integrity_check()
            if not integrity["pass"]:
                raise RuntimeError(
                    f"New R2 base store failed integrity: {integrity}"
                )
            store.connection.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            )
            store.connection.commit()
        sqlite_checkpoint_database(root / "catalog.sqlite3")
        write_json_atomic(
            marker_path,
            {
                "complete": True,
                "freeze_sha256": freeze_hash,
                "chunk_size": int(size),
                "n_buildings": int(R2_BUILDING_COUNT),
                "integrity": integrity,
                "created_at_ns": time.time_ns(),
            },
        )
        outputs[str(size)] = str(root)
    return outputs


def r2_worker_load_context(
    config: Mapping[str, Any],
) -> Tuple[
    ProjectPaths,
    LoadedModules,
    List[Any],
    Any,
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Dict[str, float]]],
    Dict[str, Dict[Tuple[str, str], int]],
    Dict[str, Dict[Tuple[str, str], int]],
    Dict[str, Dict[str, List[Any]]],
]:
    project = Path(str(config["project_dir"]))
    paths = resolve_project_paths(project)
    modules = load_all_modules(
        paths,
        f"r2worker_{os.getpid()}",
    )
    chunks, plan = load_compact_chunks_and_plan(
        modules,
        Path(str(config["values_path"])),
        Path(str(config["plan_path"])),
        str(config["trace"]),
        int(config["chunk_size"]),
    )
    calibration_dir = paths.results["r1"] / "calibration"
    calibration_frame = pd.read_csv(
        calibration_dir / "calibration_records.csv"
    )
    migration_frame = pd.read_csv(
        calibration_dir / "measured_migration_costs.csv"
    )
    profiles = modules.s1.load_profiles(
        calibration_frame
    )
    migration_costs, migration_bytes = (
        modules.s1.parse_migration_maps(
            migration_frame
        )
    )
    query_pools = modules.a2.build_query_pools(
        modules.base,
        modules.e1,
        modules.core,
        chunks,
    )
    metadata_frame = pd.read_csv(
        paths.results["r1"] / "r1_chunk_manifest.csv"
    )
    metadata = {
        str(row.chunk_id): {
            "building_id": str(row.building_id),
            "meter_type": str(row.meter_type),
            "site_id": str(row.site_id),
        }
        for row in metadata_frame.itertuples(index=False)
        if str(row.chunk_id) in set(plan.chunk_ids)
    }
    return (
        paths,
        modules,
        chunks,
        plan,
        metadata,
        profiles,
        migration_costs,
        migration_bytes,
        query_pools,
    )


def make_actual_controller(
    policy: str,
    initial_position: int,
    shape_config: Mapping[str, Any],
    query_classes: Sequence[str],
    initial_weights: np.ndarray,
) -> Optional[Any]:
    if policy == "initial_best":
        return None
    if policy == "shape_ewma_be_frozen":
        return ShapeEWMAController(
            initial_position,
            initial_weights,
            shape_config,
        )
    if policy == "repressdb_frozen":
        return RepressController(
            initial_position,
            RepressVariant(name="full_repressdb"),
        )
    raise ValueError(policy)


def run_r2_worker(config_path: Path) -> int:
    config = read_json(config_path)
    (
        paths,
        modules,
        chunks,
        plan,
        metadata,
        profiles,
        migration_costs,
        migration_bytes_map,
        query_pools,
    ) = r2_worker_load_context(config)

    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    complete_path = output_dir / "complete.json"
    if checkpoint_complete(
        complete_path,
        str(config["freeze_sha256"]),
    ):
        return 0

    run_store = output_dir / "_store"
    shutil.rmtree(run_store, ignore_errors=True)
    shutil.copytree(
        Path(str(config["base_store"])),
        run_store,
    )
    sqlite_checkpoint_database(
        run_store / "catalog.sqlite3"
    )

    chunks_by_id = {
        str(chunk.chunk_id): chunk
        for chunk in chunks
    }
    codecs = tuple(
        sorted(
            modules.base.representation_registry(3)
        )
    )
    codec_to_position = {
        codec: position
        for position, codec in enumerate(codecs)
    }
    query_classes = tuple(modules.core.QUERY_CLASSES)
    initial_weights = initial_weights_from_base(
        modules,
        query_classes,
    )
    shape_config = read_json(
        paths.results["s1"]
        / "shape_ewma_be_frozen_config.json"
    )

    chunk_ids = tuple(plan.chunk_ids)
    profile_array = modules.s1.profile_array(
        profiles,
        chunk_ids,
        codecs,
        query_classes,
    )
    migration_cost_array, migration_byte_array = (
        modules.s1.migration_matrix(
            migration_costs,
            migration_bytes_map,
            chunk_ids,
            codecs,
        )
    )
    budget_bytes = int(migration_byte_array.max())

    controllers: List[Optional[Any]] = []
    active_positions: List[int] = []
    for chunk_id in chunk_ids:
        initial_codec, _ = modules.e1.choose_best_codec(
            profiles[chunk_id],
            modules.base.WORKLOADS["scan_heavy"],
        )
        position = codec_to_position[initial_codec]
        active_positions.append(position)
        controllers.append(
            make_actual_controller(
                str(config["policy"]),
                position,
                shape_config,
                query_classes,
                initial_weights,
            )
        )

    memory_buffer: Optional[bytearray] = None
    memory_checksum = 0
    if bool(config.get("memory_pressure", False)):
        memory_buffer = bytearray(
            int(config["memory_pressure_bytes"])
        )
        memory_checksum = touch_memory(
            memory_buffer
        )

    n_events = int(plan.n_events)
    outer_query_latency = np.empty(
        n_events,
        dtype=np.int64,
    )
    codec_query_latency = np.empty(
        n_events,
        dtype=np.int64,
    )
    controller_latency = np.zeros(
        n_events,
        dtype=np.int64,
    )
    truth_latency = np.empty(
        n_events,
        dtype=np.int64,
    )
    query_logical_bytes = np.zeros(
        n_events,
        dtype=np.int64,
    )
    result_ok = np.zeros(
        n_events,
        dtype=np.uint8,
    )

    per_chunk = {
        chunk_id: {
            "query_outer_ns": 0,
            "query_codec_ns": 0,
            "controller_ns": 0,
            "migration_ns": 0,
            "migration_count": 0,
            "rewrite_bytes": 0,
            "n_queries": 0,
            "truth_check_ns": 0,
        }
        for chunk_id in chunk_ids
    }
    migration_rows: List[Dict[str, Any]] = []
    budget_remaining = budget_bytes
    budget_interval_id = 0
    budget_spent = 0
    budget_denied = 0
    budget_violations = 0
    stage_totals: Dict[str, int] = {}
    policy = str(config["policy"])
    process_start = time.perf_counter_ns()

    raw_store = modules.b1.SingleActiveChunkStore(
        run_store,
        modules.base,
        create=True,
        recover_on_open=True,
    )
    instrumented = InstrumentedSingleActiveChunkStore(
        raw_store,
        modules.b1,
    )
    try:
        for event_index in range(n_events):
            new_interval = (
                event_index
                // BUDGET_INTERVAL_GLOBAL_QUERIES
            )
            if new_interval != budget_interval_id:
                if budget_spent > budget_bytes:
                    budget_violations += 1
                budget_interval_id = new_interval
                budget_remaining = budget_bytes
                budget_spent = 0

            chunk_position = int(
                plan.chunk_positions[event_index]
            )
            chunk_id = str(
                plan.chunk_ids[chunk_position]
            )
            query_position = int(
                plan.query_class_positions[event_index]
            )
            query_class = query_classes[query_position]
            choice = int(
                plan.query_choice_positions[event_index]
            )
            query = query_pools[chunk_id][query_class][choice]

            query_start = time.perf_counter_ns()
            actual, codec_ns, stats, row = (
                instrumented.execute_query(
                    chunk_id,
                    query,
                )
            )
            outer_ns = time.perf_counter_ns() - query_start
            outer_query_latency[event_index] = int(
                outer_ns
            )
            codec_query_latency[event_index] = int(
                codec_ns
            )
            query_logical_bytes[event_index] = int(
                stats.total_logical_bytes
            )

            truth_start = time.perf_counter_ns()
            expected = modules.base.execute_on_array(
                chunks_by_id[chunk_id].values,
                query,
            )
            truth_ns = time.perf_counter_ns() - truth_start
            truth_latency[event_index] = int(truth_ns)
            ok = modules.base.results_equal(
                actual,
                expected,
            )
            result_ok[event_index] = int(ok)
            if not ok:
                raise AssertionError(
                    f"R2 truth mismatch: "
                    f"{policy}/{plan.trace_name}/"
                    f"{plan.chunk_size}/{event_index}"
                )

            local = per_chunk[chunk_id]
            local["query_outer_ns"] += int(outer_ns)
            local["query_codec_ns"] += int(codec_ns)
            local["truth_check_ns"] += int(truth_ns)
            local["n_queries"] += 1

            controller = controllers[chunk_position]
            proposal: Optional[int] = None
            decision: Optional[Dict[str, Any]] = None
            if controller is not None:
                controller_start = time.perf_counter_ns()
                proposal, decision = (
                    controller.observe_and_decide(
                        query_position=query_position,
                        decision_profile=profile_array[
                            chunk_position
                        ],
                        decision_migration_cost=(
                            migration_cost_array[
                                chunk_position
                            ]
                        ),
                        codecs=codecs,
                        global_event_index=event_index,
                        chunk_id=chunk_id,
                    )
                )
                controller_ns = (
                    time.perf_counter_ns()
                    - controller_start
                )
                controller_latency[event_index] = int(
                    controller_ns
                )
                local["controller_ns"] += int(
                    controller_ns
                )

            if proposal is not None:
                current_position = int(
                    active_positions[chunk_position]
                )
                proposal = int(proposal)
                rewrite_estimate = int(
                    migration_byte_array[
                        chunk_position,
                        current_position,
                        proposal,
                    ]
                )
                if rewrite_estimate <= budget_remaining:
                    migration_result = (
                        instrumented.migrate_profiled(
                            chunk_id,
                            codecs[proposal],
                        )
                    )
                    actual_rewrite = int(
                        migration_result["rewrite_bytes"]
                    )
                    migration_ns = int(
                        migration_result[
                            "migration_time_ns"
                        ]
                    )
                    local["migration_ns"] += migration_ns
                    local["migration_count"] += 1
                    local["rewrite_bytes"] += actual_rewrite
                    budget_remaining -= actual_rewrite
                    budget_spent += actual_rewrite
                    active_positions[
                        chunk_position
                    ] = proposal
                    controller.committed_switch(
                        proposal
                    )
                    for stage, value in migration_result[
                        "stage_times_ns"
                    ].items():
                        stage_totals[stage] = (
                            stage_totals.get(stage, 0)
                            + int(value)
                        )
                    migration_rows.append(
                        {
                            "policy": policy,
                            "trace": plan.trace_name,
                            "chunk_size": int(
                                plan.chunk_size
                            ),
                            "repeat": int(config["repeat"]),
                            "order_position": int(
                                config["order_position"]
                            ),
                            "global_event_index": int(
                                event_index
                            ),
                            "chunk_id": chunk_id,
                            "building_id": metadata[
                                chunk_id
                            ]["building_id"],
                            "source_codec": str(
                                migration_result[
                                    "from_codec"
                                ]
                            ),
                            "target_codec": str(
                                migration_result[
                                    "target_codec"
                                ]
                            ),
                            "migration_time_ns": (
                                migration_ns
                            ),
                            "rewrite_bytes": (
                                actual_rewrite
                            ),
                            "budget_remaining_after": int(
                                budget_remaining
                            ),
                            "decision_reason": (
                                decision.get("reason", "")
                                if decision is not None
                                else ""
                            ),
                            **{
                                f"stage_{stage}": int(value)
                                for stage, value in (
                                    migration_result[
                                        "stage_times_ns"
                                    ].items()
                                )
                            },
                        }
                    )
                else:
                    budget_denied += 1

            if (
                memory_buffer is not None
                and event_index > 0
                and event_index % 50_000 == 0
            ):
                memory_checksum += touch_memory(
                    memory_buffer
                )
    finally:
        integrity = instrumented.integrity_check()
        instrumented.store.connection.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        )
        instrumented.store.connection.commit()
        instrumented.store.close()

    process_elapsed_ns = (
        time.perf_counter_ns() - process_start
    )
    if budget_spent > budget_bytes:
        budget_violations += 1

    if not integrity["pass"]:
        raise RuntimeError(
            f"R2 store integrity failed: {integrity}"
        )
    if int(result_ok.sum()) != n_events:
        raise AssertionError(
            "R2 did not truth-check every query."
        )

    query_outer_total = int(
        outer_query_latency.sum()
    )
    query_codec_total = int(
        codec_query_latency.sum()
    )
    controller_total = int(
        controller_latency.sum()
    )
    truth_total = int(
        truth_latency.sum()
    )
    migration_total = int(
        sum(
            row["migration_time_ns"]
            for row in migration_rows
        )
    )
    service_total = (
        query_outer_total
        + controller_total
        + migration_total
    )
    prefix_n = min(
        R2_FRESH_PREFIX_QUERIES,
        n_events,
    )
    prefix_service = int(
        outer_query_latency[:prefix_n].sum()
        + controller_latency[:prefix_n].sum()
        + sum(
            row["migration_time_ns"]
            for row in migration_rows
            if int(row["global_event_index"])
            < prefix_n
        )
    )
    steady_service = int(
        service_total - prefix_service
    )

    chunk_rows: List[Dict[str, Any]] = []
    for chunk_id in chunk_ids:
        local = per_chunk[chunk_id]
        chunk_rows.append(
            {
                "policy": policy,
                "trace": plan.trace_name,
                "chunk_size": int(plan.chunk_size),
                "repeat": int(config["repeat"]),
                "order_position": int(
                    config["order_position"]
                ),
                "memory_pressure": int(
                    bool(config.get("memory_pressure", False))
                ),
                "chunk_id": chunk_id,
                "building_id": metadata[
                    chunk_id
                ]["building_id"],
                "meter_type": metadata[
                    chunk_id
                ]["meter_type"],
                "site_id": metadata[
                    chunk_id
                ]["site_id"],
                **local,
                "query_catalog_dispatch_ns": int(
                    local["query_outer_ns"]
                    - local["query_codec_ns"]
                ),
                "service_total_ns": int(
                    local["query_outer_ns"]
                    + local["controller_ns"]
                    + local["migration_ns"]
                ),
            }
        )

    latency_path = output_dir / "latencies.npz"
    temporary = latency_path.with_name(
        latency_path.name + ".tmp"
    )
    with open(temporary, "wb") as handle:
        np.savez_compressed(
            handle,
            query_outer_latency_ns=(
                outer_query_latency
            ),
            query_codec_latency_ns=(
                codec_query_latency
            ),
            controller_latency_ns=(
                controller_latency
            ),
            truth_check_latency_ns=(
                truth_latency
            ),
            logical_bytes=query_logical_bytes,
            result_ok=result_ok,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, latency_path)

    write_csv_atomic(
        output_dir / "chunk_results.csv",
        pd.DataFrame(chunk_rows),
    )
    migration_columns = [
        "policy",
        "trace",
        "chunk_size",
        "repeat",
        "order_position",
        "global_event_index",
        "chunk_id",
        "building_id",
        "source_codec",
        "target_codec",
        "migration_time_ns",
        "rewrite_bytes",
        "budget_remaining_after",
        "decision_reason",
    ]
    migration_frame = pd.DataFrame(migration_rows)
    if migration_frame.empty:
        migration_frame = pd.DataFrame(columns=migration_columns)
    write_csv_atomic(
        output_dir / "migration_results.csv",
        migration_frame,
    )

    summary = {
        "policy": policy,
        "trace": plan.trace_name,
        "chunk_size": int(plan.chunk_size),
        "repeat": int(config["repeat"]),
        "order_position": int(
            config["order_position"]
        ),
        "policy_order": list(
            config["policy_order"]
        ),
        "memory_pressure": bool(
            config.get("memory_pressure", False)
        ),
        "memory_pressure_bytes": int(
            config.get("memory_pressure_bytes", 0)
        ),
        "memory_touch_checksum": int(
            memory_checksum
        ),
        "n_buildings": len(chunk_ids),
        "n_queries": n_events,
        "all_queries_truth_checked": True,
        "query_outer_time_ns": query_outer_total,
        "query_codec_time_ns": query_codec_total,
        "query_catalog_dispatch_time_ns": int(
            query_outer_total - query_codec_total
        ),
        "controller_time_ns": controller_total,
        "migration_time_ns": migration_total,
        "service_total_time_ns": int(
            service_total
        ),
        "fresh_prefix_queries": int(
            prefix_n
        ),
        "fresh_prefix_service_time_ns": int(
            prefix_service
        ),
        "steady_queries": int(
            n_events - prefix_n
        ),
        "steady_service_time_ns": int(
            steady_service
        ),
        "truth_check_time_ns": truth_total,
        "harness_process_elapsed_ns": int(
            process_elapsed_ns
        ),
        "throughput_qps": float(
            n_events
            / max(service_total / 1e9, 1e-12)
        ),
        "fresh_prefix_throughput_qps": float(
            prefix_n
            / max(prefix_service / 1e9, 1e-12)
        ),
        "steady_throughput_qps": float(
            (n_events - prefix_n)
            / max(steady_service / 1e9, 1e-12)
        ),
        "query_p50_ns": float(
            np.quantile(
                outer_query_latency,
                0.50,
            )
        ),
        "query_p95_ns": float(
            np.quantile(
                outer_query_latency,
                0.95,
            )
        ),
        "query_p99_ns": float(
            np.quantile(
                outer_query_latency,
                0.99,
            )
        ),
        "query_max_ns": int(
            outer_query_latency.max()
        ),
        "migration_count": len(
            migration_rows
        ),
        "rewrite_bytes": int(
            sum(
                row["rewrite_bytes"]
                for row in migration_rows
            )
        ),
        "budget_bytes_per_interval": int(
            budget_bytes
        ),
        "budget_denied_proposals": int(
            budget_denied
        ),
        "budget_violation_count": int(
            budget_violations
        ),
        "migration_stage_totals_ns": (
            stage_totals
        ),
        "store_integrity": integrity,
        "latency_npz_sha256": sha256_file(
            latency_path
        ),
    }
    write_json_atomic(
        summary_path,
        summary,
    )
    write_json_atomic(
        complete_path,
        {
            "complete": True,
            "freeze_sha256": str(
                config["freeze_sha256"]
            ),
            "summary_sha256": sha256_file(
                summary_path
            ),
            "completed_at_ns": time.time_ns(),
        },
    )
    # The copied store is not required after the result was verified.
    shutil.rmtree(
        run_store,
        ignore_errors=True,
    )
    return 0


def build_r2_cells(
    paths: ProjectPaths,
    worker_assets: Mapping[str, Any],
    base_stores: Mapping[str, str],
    freeze_hash: str,
    memory_pressure: bool,
) -> List[Dict[str, Any]]:
    cells: List[Dict[str, Any]] = []
    traces = (
        (R2_MEMORY_PRESSURE_TRACE,)
        if memory_pressure
        else TRACE_FAMILIES
    )
    sizes = (
        (R2_MEMORY_PRESSURE_SIZE,)
        if memory_pressure
        else CHUNK_SIZES
    )
    for trace in traces:
        for size in sizes:
            for repeat in range(R2_REPEATS):
                policy_order = R2_POLICY_PERMUTATIONS[
                    repeat % len(R2_POLICY_PERMUTATIONS)
                ]
                for order_position, policy in enumerate(
                    policy_order
                ):
                    root = (
                        paths.output
                        / "j1_end_to_end"
                        / (
                            "memory_pressure"
                            if memory_pressure
                            else "primary"
                        )
                        / trace
                        / f"size_{size}"
                        / f"repeat_{repeat:02d}"
                        / f"order_{order_position}_{policy}"
                    )
                    config = {
                        "project_dir": str(paths.project),
                        "output_dir": str(root),
                        "base_store": str(
                            base_stores[str(size)]
                        ),
                        "values_path": str(
                            worker_assets[
                                "values_paths"
                            ][str(size)]
                        ),
                        "plan_path": str(
                            worker_assets[
                                "plan_paths"
                            ][f"{trace}|{size}"]
                        ),
                        "trace": trace,
                        "chunk_size": int(size),
                        "repeat": int(repeat),
                        "policy": policy,
                        "order_position": int(
                            order_position
                        ),
                        "policy_order": list(
                            policy_order
                        ),
                        "memory_pressure": bool(
                            memory_pressure
                        ),
                        "memory_pressure_bytes": (
                            R2_MEMORY_PRESSURE_BYTES
                            if memory_pressure
                            else 0
                        ),
                        "freeze_sha256": freeze_hash,
                    }
                    config_path = root / "worker_config.json"
                    write_json_atomic(
                        config_path,
                        config,
                    )
                    cells.append(
                        {
                            **config,
                            "config_path": str(
                                config_path
                            ),
                        }
                    )
    return cells


def run_r2_cells(
    paths: ProjectPaths,
    cells: Sequence[Mapping[str, Any]],
) -> None:
    total = len(cells)
    for cell_index, cell in enumerate(cells, start=1):
        output_dir = Path(
            str(cell["output_dir"])
        )
        complete_path = output_dir / "complete.json"
        if checkpoint_complete(
            complete_path,
            str(cell["freeze_sha256"]),
        ):
            print(
                f"[J1 resume {cell_index}/{total}] "
                f"{cell['trace']} size={cell['chunk_size']} "
                f"repeat={cell['repeat']} policy={cell['policy']}"
            )
            continue
        log_path = output_dir / "worker.log"
        exit_code = run_subprocess_checked(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--internal-worker",
                "r2",
                "--config",
                str(cell["config_path"]),
            ],
            log_path,
        )
        if exit_code != 0:
            raise RuntimeError(
                f"J1 worker failed with exit code {exit_code}. "
                f"See {log_path}"
            )
        print(
            f"[J1 {cell_index}/{total}] "
            f"{cell['trace']} size={cell['chunk_size']} "
            f"repeat={cell['repeat']} policy={cell['policy']}"
        )


def collect_r2_results(
    paths: ProjectPaths,
    cells: Sequence[Mapping[str, Any]],
    label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: List[Dict[str, Any]] = []
    chunk_frames: List[pd.DataFrame] = []
    migration_frames: List[pd.DataFrame] = []
    for cell in cells:
        root = Path(str(cell["output_dir"]))
        summary = read_json(root / "summary.json")
        flat = {
            key: value
            for key, value in summary.items()
            if key not in (
                "migration_stage_totals_ns",
                "store_integrity",
            )
        }
        for stage, value in summary.get(
            "migration_stage_totals_ns",
            {},
        ).items():
            flat[f"migration_stage_{stage}"] = int(
                value
            )
        flat["result_group"] = label
        summaries.append(flat)

        chunk_frame = pd.read_csv(
            root / "chunk_results.csv"
        )
        chunk_frame["result_group"] = label
        chunk_frames.append(chunk_frame)

        migration_path = root / "migration_results.csv"
        migration_frame = pd.read_csv(
            migration_path
        )
        if not migration_frame.empty:
            migration_frame["result_group"] = label
        migration_frames.append(
            migration_frame
        )

    summary_frame = pd.DataFrame(summaries)
    chunk_frame = pd.concat(
        chunk_frames,
        ignore_index=True,
    )
    migration_nonempty = [
        frame
        for frame in migration_frames
        if not frame.empty
    ]
    migration_frame = (
        pd.concat(
            migration_nonempty,
            ignore_index=True,
        )
        if migration_nonempty
        else pd.DataFrame()
    )
    return (
        summary_frame,
        chunk_frame,
        migration_frame,
    )


def paired_r2_statistics(
    summary_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    index_columns = [
        "result_group",
        "trace",
        "chunk_size",
        "repeat",
    ]
    metrics = (
        "service_total_time_ns",
        "throughput_qps",
        "query_p99_ns",
        "fresh_prefix_service_time_ns",
        "steady_service_time_ns",
    )
    pivot = summary_frame.pivot_table(
        index=index_columns,
        columns="policy",
        values=list(metrics),
        aggfunc="first",
    )
    for result_group in summary_frame[
        "result_group"
    ].unique():
        for trace in summary_frame[
            "trace"
        ].unique():
            for size in sorted(
                summary_frame["chunk_size"].unique()
            ):
                try:
                    local = pivot.loc[
                        (result_group, trace, size)
                    ]
                except KeyError:
                    continue
                if not set(CORE_POLICIES).issubset(
                    local["service_total_time_ns"].columns
                    if isinstance(
                        local["service_total_time_ns"],
                        pd.DataFrame,
                    )
                    else []
                ):
                    # pandas can return a Series for one row.
                    pass
                for numerator, denominator in (
                    (
                        "repressdb_frozen",
                        "shape_ewma_be_frozen",
                    ),
                    (
                        "repressdb_frozen",
                        "initial_best",
                    ),
                    (
                        "shape_ewma_be_frozen",
                        "initial_best",
                    ),
                ):
                    for metric in metrics:
                        metric_frame = local[metric]
                        if isinstance(
                            metric_frame,
                            pd.Series,
                        ):
                            continue
                        if (
                            numerator
                            not in metric_frame
                            or denominator
                            not in metric_frame
                        ):
                            continue
                        ratio = (
                            metric_frame[numerator].to_numpy(
                                dtype=np.float64
                            )
                            / metric_frame[
                                denominator
                            ].to_numpy(
                                dtype=np.float64
                            )
                        )
                        if metric == "throughput_qps":
                            # Throughput ratio direction is kept literal.
                            pass
                        low, high = percentile_interval(
                            ratio,
                            (
                                "r2",
                                result_group,
                                trace,
                                size,
                                metric,
                                numerator,
                                denominator,
                            ),
                            R2_BOOTSTRAPS,
                            statistic="geometric_mean",
                        )
                        rows.append(
                            {
                                "result_group": result_group,
                                "trace": trace,
                                "chunk_size": int(size),
                                "metric": metric,
                                "numerator_policy": numerator,
                                "denominator_policy": denominator,
                                "n_paired_repeats": int(
                                    len(ratio)
                                ),
                                "geometric_mean_ratio": geometric_mean(
                                    ratio
                                ),
                                "median_ratio": float(
                                    np.median(ratio)
                                ),
                                "ci_low": low,
                                "ci_high": high,
                                "fraction_ratio_below_one": float(
                                    np.mean(ratio < 1.0)
                                ),
                            }
                        )
    return pd.DataFrame(rows)


def hierarchical_paired_ratio_statistics(
    frame: pd.DataFrame,
    cluster_column: str,
    unit_column: str,
    value_column: str,
    group_columns: Sequence[str],
    comparisons: Sequence[Tuple[str, str]],
    seed_token: str,
    bootstrap_replicates: int = 20_000,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for group_key, group in frame.groupby(list(group_columns)):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        pivot = group.pivot_table(
            index=[cluster_column, unit_column],
            columns="policy",
            values=value_column,
            aggfunc="first",
        ).dropna()
        for numerator, denominator in comparisons:
            if numerator not in pivot or denominator not in pivot:
                continue
            local = pivot[[numerator, denominator]].copy()
            local["log_ratio"] = np.log(
                local[numerator].to_numpy(dtype=np.float64)
                / local[denominator].to_numpy(dtype=np.float64)
            )
            clusters = sorted(
                local.index.get_level_values(cluster_column).unique()
            )
            observed = float(np.exp(local["log_ratio"].mean()))
            rng = np.random.default_rng(
                stable_seed(MASTER_SEED, seed_token, *group_key, numerator, denominator)
            )
            estimates = np.empty(bootstrap_replicates, dtype=np.float64)
            for bootstrap_index in range(bootstrap_replicates):
                sampled_clusters = rng.choice(
                    clusters,
                    size=len(clusters),
                    replace=True,
                )
                sampled_logs: List[float] = []
                for cluster in sampled_clusters:
                    cluster_frame = local.xs(cluster, level=cluster_column)
                    values = cluster_frame["log_ratio"].to_numpy(dtype=np.float64)
                    sampled_logs.extend(
                        rng.choice(values, size=len(values), replace=True).tolist()
                    )
                estimates[bootstrap_index] = math.exp(float(np.mean(sampled_logs)))
            row = {
                column: value
                for column, value in zip(group_columns, group_key)
            }
            row.update(
                {
                    "numerator_policy": numerator,
                    "denominator_policy": denominator,
                    "cluster_column": cluster_column,
                    "unit_column": unit_column,
                    "n_clusters": int(len(clusters)),
                    "n_paired_units": int(len(local)),
                    "geometric_mean_ratio": observed,
                    "hierarchical_ci_low": float(np.quantile(estimates, 0.025)),
                    "hierarchical_ci_high": float(np.quantile(estimates, 0.975)),
                    "secondary_analysis": True,
                    "dependency_note": (
                        "The hierarchical interval reflects cluster and unit sampling; "
                        "it does not reconstruct a new global write-budget schedule."
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def r2_order_effects(
    summary_frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = summary_frame[
        summary_frame["result_group"] == "primary"
    ].copy()
    frame["normalized_within_cell"] = frame[
        "service_total_time_ns"
    ] / frame.groupby(
        ["trace", "chunk_size", "repeat"]
    )["service_total_time_ns"].transform("mean")
    rows: List[Dict[str, Any]] = []
    for policy, group in frame.groupby("policy"):
        for order_position, local in group.groupby(
            "order_position"
        ):
            rows.append(
                {
                    "policy": policy,
                    "order_position": int(
                        order_position
                    ),
                    "n_cells": int(len(local)),
                    "median_normalized_time": float(
                        local[
                            "normalized_within_cell"
                        ].median()
                    ),
                    "mean_normalized_time": float(
                        local[
                            "normalized_within_cell"
                        ].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# J2: multi-seed traces, PH-BE development, and unified ablation
# ---------------------------------------------------------------------------

def seeded_category_probabilities(
    trace: str,
    n_events: int,
    trace_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(
        stable_seed(
            MASTER_SEED,
            "multiseed-category",
            trace,
            trace_seed,
        )
    )
    x = np.linspace(
        0.0,
        1.0,
        n_events,
        dtype=np.float64,
    )
    if trace == "R1_A_gradual_periodic":
        sigmoid = 1.0 / (
            1.0
            + np.exp(
                -(x - 0.50) / 0.12
            )
        )
        phase = float(
            rng.uniform(-0.20, 0.20)
        )
        point = np.clip(
            0.05
            + 0.65 * sigmoid
            + 0.045
            * np.sin(
                2.0
                * np.pi
                * 9.0
                * x
                + phase
            ),
            0.03,
            0.76,
        )
        aggregate = np.clip(
            0.16
            + 0.035
            * np.sin(
                2.0
                * np.pi
                * 4.0
                * x
                + 1.1
                + phase
            ),
            0.08,
            0.25,
        )
        scan = np.maximum(
            1.0 - point - aggregate,
            0.08,
        )
        probabilities = np.column_stack(
            [point, scan, aggregate]
        )
        return probabilities / probabilities.sum(
            axis=1,
            keepdims=True,
        )

    knot_step = 100
    n_knots = (
        math.ceil(n_events / knot_step)
        + 1
    )
    logits = np.zeros(
        (n_knots, 3),
        dtype=np.float64,
    )
    logits[0] = np.asarray(
        [-0.4, 0.9, -0.2],
        dtype=np.float64,
    )
    center = np.asarray(
        [-0.15, 0.50, -0.20],
        dtype=np.float64,
    )
    for knot in range(1, n_knots):
        logits[knot] = (
            0.985 * logits[knot - 1]
            + 0.015 * center
            + rng.normal(
                0.0,
                0.075,
                size=3,
            )
        )
        logits[knot, 0] += (
            0.08
            * math.sin(
                2.0 * math.pi * knot / 170.0
            )
        )
        logits[knot, 2] += (
            0.06
            * math.sin(
                2.0
                * math.pi
                * knot
                / 113.0
                + 0.7
            )
        )

    if trace == "R1_C_noisy_bursts":
        positions = np.arange(
            n_knots,
            dtype=np.float64,
        )
        for _ in range(24):
            center_knot = int(
                rng.integers(
                    10,
                    n_knots - 10,
                )
            )
            width = float(
                rng.uniform(2.0, 12.0)
            )
            category = int(
                rng.integers(0, 3)
            )
            amplitude = float(
                rng.uniform(0.65, 1.60)
            )
            logits[:, category] += (
                amplitude
                * np.exp(
                    -0.5
                    * (
                        (
                            positions
                            - center_knot
                        )
                        / width
                    )
                    ** 2
                )
            )

    knot_positions = np.arange(
        n_knots,
        dtype=np.float64,
    ) * knot_step
    event_positions = np.arange(
        n_events,
        dtype=np.float64,
    )
    interpolated = np.column_stack(
        [
            np.interp(
                event_positions,
                knot_positions,
                logits[:, category],
            )
            for category in range(3)
        ]
    )
    shifted = interpolated - interpolated.max(
        axis=1,
        keepdims=True,
    )
    exponential = np.exp(shifted)
    return exponential / exponential.sum(
        axis=1,
        keepdims=True,
    )


def expand_categories_to_query_classes(
    category_probabilities: np.ndarray,
    query_classes: Sequence[str],
) -> np.ndarray:
    output = np.zeros(
        (
            len(category_probabilities),
            len(query_classes),
        ),
        dtype=np.float64,
    )
    for category_position, category in enumerate(
        ("point", "scan", "aggregate")
    ):
        for query_class, weight in (
            CATEGORY_CLASS_WEIGHTS[
                category
            ].items()
        ):
            output[
                :,
                query_classes.index(
                    query_class
                ),
            ] += (
                category_probabilities[
                    :,
                    category_position,
                ]
                * float(weight)
            )
    return output / output.sum(
        axis=1,
        keepdims=True,
    )


def seeded_access_probabilities(
    trace: str,
    n_events: int,
    n_chunks: int,
    trace_seed: int,
) -> np.ndarray:
    block_size = 100
    n_blocks = math.ceil(
        n_events / block_size
    )
    rng = np.random.default_rng(
        stable_seed(
            MASTER_SEED,
            "multiseed-access",
            trace,
            trace_seed,
        )
    )
    base_logits = rng.normal(
        0.0,
        0.12,
        n_chunks,
    )
    phases = rng.uniform(
        0.0,
        2.0 * np.pi,
        n_chunks,
    )
    periods = rng.uniform(
        180.0,
        520.0,
        n_chunks,
    )
    autoregressive = np.zeros(
        (n_blocks, n_chunks),
        dtype=np.float64,
    )
    for block in range(1, n_blocks):
        autoregressive[block] = (
            0.992
            * autoregressive[block - 1]
            + rng.normal(
                0.0,
                0.012,
                n_chunks,
            )
        )
    block_positions = np.arange(
        n_blocks,
        dtype=np.float64,
    )[:, np.newaxis]
    logits = (
        base_logits[np.newaxis, :]
        + 0.22
        * np.sin(
            2.0
            * np.pi
            * block_positions
            / periods[np.newaxis, :]
            + phases[np.newaxis, :]
        )
        + autoregressive
    )
    logits -= logits.max(
        axis=1,
        keepdims=True,
    )
    exponential = np.exp(logits)
    return exponential / exponential.sum(
        axis=1,
        keepdims=True,
    )


def build_or_load_multiseed_plan(
    paths: ProjectPaths,
    modules: LoadedModules,
    context: R1Context,
    trace: str,
    chunk_size: int,
    trace_seed: int,
) -> Tuple[Any, Dict[str, Any]]:
    plan_dir = (
        paths.output
        / "j2_multiseed"
        / "frozen_plans"
    )
    plan_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    npz_path = (
        plan_dir
        / (
            f"{trace}__size_{chunk_size}__"
            f"seed_{trace_seed}.npz"
        )
    )
    json_path = npz_path.with_suffix(
        ".json"
    )

    size_manifest = (
        context.chunk_manifest
        .query("chunk_size == @chunk_size")
        .sort_values("building_rank")
    )
    chunk_ids = tuple(
        size_manifest[
            "chunk_id"
        ].astype(str)
    )
    n_chunks = len(chunk_ids)
    if n_chunks != 32:
        raise AssertionError(
            f"Expected 32 R1 chunks, found {n_chunks}"
        )
    n_events = (
        n_chunks
        * MULTISEED_EVENTS_PER_BUILDING
    )

    if npz_path.exists() and json_path.exists():
        manifest = read_json(json_path)
        if manifest[
            "trace_file_sha256"
        ] != sha256_file(npz_path):
            raise AssertionError(
                "Multi-seed trace hash mismatch"
            )
        with np.load(
            npz_path,
            allow_pickle=False,
        ) as archive:
            chunk_positions = np.asarray(
                archive[
                    "chunk_positions"
                ],
                dtype=np.int16,
            )
            local_indices = np.asarray(
                archive["local_indices"],
                dtype=np.int32,
            )
            query_class_positions = np.asarray(
                archive[
                    "query_class_positions"
                ],
                dtype=np.uint8,
            )
            query_choice_positions = np.asarray(
                archive[
                    "query_choice_positions"
                ],
                dtype=np.uint16,
            )
    else:
        query_classes = tuple(
            modules.core.QUERY_CLASSES
        )
        category = seeded_category_probabilities(
            trace,
            n_events,
            trace_seed,
        )
        class_probabilities = (
            expand_categories_to_query_classes(
                category,
                query_classes,
            )
        )
        access = seeded_access_probabilities(
            trace,
            n_events,
            n_chunks,
            trace_seed,
        )
        rng = np.random.default_rng(
            stable_seed(
                MASTER_SEED,
                "multiseed-sampling",
                trace,
                chunk_size,
                trace_seed,
            )
        )
        chunk_positions = np.empty(
            n_events,
            dtype=np.int16,
        )
        for start in range(
            0,
            n_events,
            100,
        ):
            end = min(
                n_events,
                start + 100,
            )
            chunk_positions[
                start:end
            ] = rng.choice(
                n_chunks,
                size=end - start,
                p=access[start // 100],
            ).astype(np.int16)

        cumulative = np.cumsum(
            class_probabilities,
            axis=1,
        )
        uniforms = rng.random(
            n_events
        )
        query_class_positions = np.sum(
            uniforms[:, np.newaxis]
            > cumulative,
            axis=1,
        ).astype(np.uint8)
        query_class_positions = np.minimum(
            query_class_positions,
            len(query_classes) - 1,
        )
        local_indices = np.empty(
            n_events,
            dtype=np.int32,
        )
        query_choice_positions = np.empty(
            n_events,
            dtype=np.uint16,
        )
        local_counts = np.zeros(
            n_chunks,
            dtype=np.int32,
        )
        for event_index in range(
            n_events
        ):
            chunk_position = int(
                chunk_positions[
                    event_index
                ]
            )
            local_indices[
                event_index
            ] = local_counts[
                chunk_position
            ]
            local_counts[
                chunk_position
            ] += 1
            query_class = query_classes[
                int(
                    query_class_positions[
                        event_index
                    ]
                )
            ]
            pool_size = len(
                context.query_pools[
                    chunk_ids[
                        chunk_position
                    ]
                ][query_class]
            )
            query_choice_positions[
                event_index
            ] = int(
                rng.integers(
                    0,
                    pool_size,
                )
            )

        temporary = npz_path.with_name(
            npz_path.name + ".tmp"
        )
        with open(
            temporary,
            "wb",
        ) as handle:
            np.savez_compressed(
                handle,
                chunk_positions=(
                    chunk_positions
                ),
                local_indices=(
                    local_indices
                ),
                query_class_positions=(
                    query_class_positions
                ),
                query_choice_positions=(
                    query_choice_positions
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temporary,
            npz_path,
        )
        counts = np.bincount(
            chunk_positions,
            minlength=n_chunks,
        )
        manifest = {
            "trace_name": trace,
            "chunk_size": int(
                chunk_size
            ),
            "trace_seed": int(
                trace_seed
            ),
            "n_chunks": int(
                n_chunks
            ),
            "n_events": int(
                n_events
            ),
            "minimum_local_queries": int(
                counts.min()
            ),
            "median_local_queries": float(
                np.median(counts)
            ),
            "maximum_local_queries": int(
                counts.max()
            ),
            "query_class_counts": {
                query_classes[position]: int(
                    np.sum(
                        query_class_positions
                        == position
                    )
                )
                for position in range(
                    len(query_classes)
                )
            },
            "trace_file_sha256": sha256_file(
                npz_path
            ),
            "phase_boundaries_visible": False,
            "future_probabilities_visible": False,
        }
        write_json_atomic(
            json_path,
            manifest,
        )

    plan = modules.core.LightweightTracePlan(
        trace_name=trace,
        trace_seed=int(trace_seed),
        chunk_size=int(chunk_size),
        chunk_ids=chunk_ids,
        chunk_positions=chunk_positions,
        local_indices=local_indices,
        query_class_positions=(
            query_class_positions
        ),
        query_choice_positions=(
            query_choice_positions
        ),
        workload_positions=np.zeros(
            len(chunk_positions),
            dtype=np.uint8,
        ),
        workload_names=(
            "continuous_nonstationary",
        ),
    )
    return plan, manifest


def ph_grid() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    index = 0
    for delta in PH_DELTAS:
        for threshold in PH_THRESHOLDS:
            for window_cap in PH_WINDOW_CAPS:
                for deadband in PH_DEADBANDS:
                    for horizon in PH_HORIZONS:
                        rows.append(
                            {
                                "config_id": (
                                    f"PH_{index:03d}"
                                ),
                                "delta": float(
                                    delta
                                ),
                                "threshold": float(
                                    threshold
                                ),
                                "window_cap": int(
                                    window_cap
                                ),
                                "deadband": float(
                                    deadband
                                ),
                                "prediction_horizon": int(
                                    horizon
                                ),
                            }
                        )
                        index += 1
    return rows


def develop_page_hinkley_baseline(
    paths: ProjectPaths,
    modules: LoadedModules,
    freeze_hash: str,
) -> Dict[str, Any]:
    output = (
        paths.output
        / "j2_multiseed"
        / "ph_be_development"
    )
    output.mkdir(
        parents=True,
        exist_ok=True,
    )
    selected_path = (
        output
        / "ph_be_frozen_config.json"
    )
    complete_path = (
        output / "complete.json"
    )
    if checkpoint_complete(
        complete_path,
        freeze_hash,
    ):
        return read_json(selected_path)

    pems04 = modules.s1.load_pems04_context(
        paths.project,
        modules.core,
        modules.e1,
        modules.base,
        paths.results["a0"],
        modules.a2,
    )
    configurations = ph_grid()
    codecs = tuple(
        sorted(
            modules.base.representation_registry(
                3
            )
        )
    )
    query_classes = tuple(
        modules.core.QUERY_CLASSES
    )
    initial_weights = initial_weights_from_base(
        modules,
        query_classes,
    )
    rows: List[Dict[str, Any]] = []

    plans = list(
        modules.s1.build_plans(
            pems04,
            modules.core,
            modules.base,
            modules.core.TRACE_NAMES,
            modules.core.TRACE_SEEDS,
        )
    )
    for config_index, config in enumerate(
        configurations,
        start=1,
    ):
        for (
            chunk_size,
            trace_seed,
            trace_name,
            plan,
        ) in plans:
            chunk_ids = tuple(
                plan.chunk_ids
            )
            profile = modules.s1.profile_array(
                pems04["profiles"],
                chunk_ids,
                codecs,
                query_classes,
            )
            initial = modules.s1.scan_best_positions(
                modules.e1,
                modules.base,
                pems04["profiles"],
                chunk_ids,
                codecs,
            )
            migration_cost, migration_bytes = (
                modules.s1.migration_matrix(
                    pems04[
                        "migration_costs"
                    ],
                    pems04[
                        "migration_bytes"
                    ],
                    chunk_ids,
                    codecs,
                )
            )
            initial_result = static_model_result(
                plan,
                profile,
                initial,
                "initial_best",
            )
            result = simulate_controller_model(
                policy_name=(
                    "page_hinkley_be"
                ),
                controller_factory=(
                    lambda initial_position,
                    local_config=dict(
                        config
                    ): PageHinkleyBEController(
                        initial_position,
                        local_config,
                        query_classes,
                    )
                ),
                plan=plan,
                evaluation_profile=profile,
                decision_profile=profile,
                evaluation_migration_cost=(
                    migration_cost
                ),
                decision_migration_cost=(
                    migration_cost
                ),
                migration_bytes=(
                    migration_bytes
                ),
                initial_positions=initial,
                codecs=codecs,
                use_write_budget=False,
            )
            rows.append(
                {
                    **config,
                    "dataset": "PEMS04",
                    "trace": trace_name,
                    "trace_seed": int(
                        trace_seed
                    ),
                    "chunk_size": int(
                        chunk_size
                    ),
                    "n_queries": int(
                        plan.n_events
                    ),
                    "total_time_ns": float(
                        result[
                            "total_time_ns"
                        ]
                    ),
                    "initial_best_total_time_ns": float(
                        initial_result[
                            "total_time_ns"
                        ]
                    ),
                    "ratio_to_initial_best": float(
                        result[
                            "total_time_ns"
                        ]
                        / initial_result[
                            "total_time_ns"
                        ]
                    ),
                    "migration_count": int(
                        result[
                            "migration_count"
                        ]
                    ),
                    "rewrite_bytes": int(
                        result[
                            "rewrite_bytes"
                        ]
                    ),
                    "unamortized_migration_rate": float(
                        result[
                            "unamortized_migration_rate"
                        ]
                    ),
                    "stale_migration_rate": float(
                        result[
                            "stale_migration_rate"
                        ]
                    ),
                }
            )
        print(
            f"[J2 PH development "
            f"{config_index}/{len(configurations)}] "
            f"{config['config_id']}"
        )

    all_results = pd.DataFrame(
        rows
    )
    write_csv_atomic(
        output
        / "ph_be_grid_all_results.csv",
        all_results,
    )
    summary_rows: List[
        Dict[str, Any]
    ] = []
    for config_id, group in all_results.groupby(
        "config_id"
    ):
        action = group[
            group["trace"].isin(
                ("shape_only", "joint_orthogonal")
            )
        ]
        brief = group[
            group["trace"] == "brief_burst"
        ]
        stable = group[
            group["trace"] == "stable_scan"
        ]
        first = group.iloc[0]
        summary_rows.append(
            {
                "config_id": str(
                    config_id
                ),
                "delta": float(
                    first["delta"]
                ),
                "threshold": float(
                    first["threshold"]
                ),
                "window_cap": int(
                    first["window_cap"]
                ),
                "deadband": float(
                    first["deadband"]
                ),
                "prediction_horizon": int(
                    first[
                        "prediction_horizon"
                    ]
                ),
                "action_geometric_mean_ratio": geometric_mean(
                    action[
                        "ratio_to_initial_best"
                    ]
                ),
                "action_migration_count": int(
                    action[
                        "migration_count"
                    ].sum()
                ),
                "action_rewrite_bytes": int(
                    action[
                        "rewrite_bytes"
                    ].sum()
                ),
                "brief_median_ratio": float(
                    brief[
                        "ratio_to_initial_best"
                    ].median()
                ),
                "stable_median_ratio": float(
                    stable[
                        "ratio_to_initial_best"
                    ].median()
                ),
            }
        )
    summary = pd.DataFrame(
        summary_rows
    )
    summary["safe"] = (
        (
            summary[
                "brief_median_ratio"
            ]
            <= PH_SAFETY_LIMIT
        )
        & (
            summary[
                "stable_median_ratio"
            ]
            <= PH_SAFETY_LIMIT
        )
    )
    safe = summary[
        summary["safe"]
    ].copy()
    if safe.empty:
        raise RuntimeError(
            "No safe PH-BE configuration."
        )
    best_score = float(
        safe[
            "action_geometric_mean_ratio"
        ].min()
    )
    equivalent = safe[
        safe[
            "action_geometric_mean_ratio"
        ]
        <= best_score
        * (
            1.0
            + PH_TIE_TOLERANCE
        )
    ].copy()
    equivalent = equivalent.sort_values(
        [
            "action_migration_count",
            "action_rewrite_bytes",
            "threshold",
            "deadband",
            "prediction_horizon",
            "window_cap",
            "delta",
            "config_id",
        ],
        ascending=[
            True,
            True,
            False,
            False,
            False,
            False,
            True,
            True,
        ],
    )
    selected_summary = (
        equivalent.iloc[0].to_dict()
    )
    selected = next(
        dict(config)
        for config in configurations
        if config["config_id"]
        == selected_summary[
            "config_id"
        ]
    )
    selected.update(
        {
            "selection_dataset": (
                "PEMS04 only"
            ),
            "selection_summary": (
                selected_summary
            ),
            "best_action_score": (
                best_score
            ),
            "equivalent_config_count": int(
                len(equivalent)
            ),
            "frozen_for_multiseed": True,
            "revision_freeze_sha256": (
                freeze_hash
            ),
        }
    )
    write_csv_atomic(
        output
        / "ph_be_grid_summary.csv",
        summary,
    )
    write_json_atomic(
        selected_path,
        selected,
    )
    write_json_atomic(
        complete_path,
        {
            "complete": True,
            "freeze_sha256": freeze_hash,
            "selected_config_sha256": (
                sha256_file(
                    selected_path
                )
            ),
            "completed_at_ns": (
                time.time_ns()
            ),
        },
    )
    return selected


def build_multiseed_plans(
    paths: ProjectPaths,
    modules: LoadedModules,
    context: R1Context,
) -> Tuple[
    Dict[Tuple[str, int, int], Any],
    pd.DataFrame,
]:
    plans: Dict[
        Tuple[str, int, int],
        Any,
    ] = {}
    rows: List[Dict[str, Any]] = []
    for trace_seed in MULTISEED_TRACE_SEEDS:
        for trace in TRACE_FAMILIES:
            for size in CHUNK_SIZES:
                plan, manifest = (
                    build_or_load_multiseed_plan(
                        paths,
                        modules,
                        context,
                        trace,
                        size,
                        trace_seed,
                    )
                )
                plans[
                    (
                        trace,
                        size,
                        trace_seed,
                    )
                ] = plan
                rows.append(manifest)
    frame = pd.DataFrame(rows)
    write_csv_atomic(
        paths.output
        / "j2_multiseed"
        / "multiseed_trace_manifest.csv",
        frame,
    )
    return plans, frame


def run_multiseed_and_ablation(
    paths: ProjectPaths,
    modules: LoadedModules,
    context: R1Context,
    plans: Mapping[
        Tuple[str, int, int],
        Any,
    ],
    ph_config: Mapping[str, Any],
    freeze_hash: str,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    output = (
        paths.output / "j2_multiseed"
    )
    checkpoint_root = (
        output / "_cell_checkpoints"
    )
    checkpoint_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    codecs = tuple(
        sorted(
            modules.base.representation_registry(
                3
            )
        )
    )
    query_classes = tuple(
        modules.core.QUERY_CLASSES
    )
    shape_config = read_json(
        paths.results["s1"]
        / "shape_ewma_be_frozen_config.json"
    )
    afhc_config = read_json(
        paths.results["v074"]
        / "hotcold_frozen_config.json"
    )
    initial_weights = initial_weights_from_base(
        modules,
        query_classes,
    )

    global_frames: List[
        pd.DataFrame
    ] = []
    chunk_frames: List[
        pd.DataFrame
    ] = []
    migration_frames: List[
        pd.DataFrame
    ] = []
    ablation_frames: List[
        pd.DataFrame
    ] = []

    total_cells = len(plans)
    for cell_index, (
        key,
        plan,
    ) in enumerate(
        sorted(plans.items()),
        start=1,
    ):
        trace, size, trace_seed = key
        checkpoint = (
            checkpoint_root
            / (
                f"{trace}__size_{size}__"
                f"seed_{trace_seed}"
            )
        )
        global_path = checkpoint / "global.csv"
        chunk_path = checkpoint / "chunk.csv"
        migration_path = (
            checkpoint / "migrations.csv"
        )
        ablation_path = (
            checkpoint / "ablation.csv"
        )
        complete_path = (
            checkpoint / "complete.json"
        )
        if (
            checkpoint_complete(
                complete_path,
                freeze_hash,
            )
            and global_path.exists()
            and chunk_path.exists()
            and migration_path.exists()
            and ablation_path.exists()
        ):
            global_frames.append(
                pd.read_csv(global_path)
            )
            chunk_frames.append(
                pd.read_csv(chunk_path)
            )
            migration_frames.append(
                pd.read_csv(
                    migration_path
                )
            )
            ablation_frames.append(
                pd.read_csv(
                    ablation_path
                )
            )
            print(
                f"[J2 resume {cell_index}/{total_cells}] "
                f"{trace} size={size} seed={trace_seed}"
            )
            continue

        shutil.rmtree(
            checkpoint,
            ignore_errors=True,
        )
        checkpoint.mkdir(
            parents=True,
            exist_ok=True,
        )
        chunk_ids = tuple(
            plan.chunk_ids
        )
        profile = modules.s1.profile_array(
            context.profiles,
            chunk_ids,
            codecs,
            query_classes,
        )
        initial = modules.s1.scan_best_positions(
            modules.e1,
            modules.base,
            context.profiles,
            chunk_ids,
            codecs,
        )
        migration_cost, migration_bytes = (
            modules.s1.migration_matrix(
                context.migration_costs,
                context.migration_bytes,
                chunk_ids,
                codecs,
            )
        )
        budget_bytes = int(
            migration_bytes.max()
        )
        best_positions = (
            best_static_position_array(
                plan,
                profile,
            )
        )

        results: Dict[
            str,
            Dict[str, Any],
        ] = {}
        results["initial_best"] = (
            static_model_result(
                plan,
                profile,
                initial,
                "initial_best",
            )
        )
        results["best_static"] = (
            static_model_result(
                plan,
                profile,
                best_positions,
                "best_static",
            )
        )
        results[
            "shape_ewma_be_frozen"
        ] = simulate_controller_model(
            policy_name=(
                "shape_ewma_be_frozen"
            ),
            controller_factory=(
                lambda initial_position: ShapeEWMAController(
                    initial_position,
                    initial_weights,
                    shape_config,
                )
            ),
            plan=plan,
            evaluation_profile=profile,
            decision_profile=profile,
            evaluation_migration_cost=(
                migration_cost
            ),
            decision_migration_cost=(
                migration_cost
            ),
            migration_bytes=(
                migration_bytes
            ),
            initial_positions=initial,
            codecs=codecs,
            use_write_budget=True,
        )
        results[
            "repressdb_frozen"
        ] = simulate_controller_model(
            policy_name=(
                "repressdb_frozen"
            ),
            controller_factory=(
                lambda initial_position: RepressController(
                    initial_position,
                    RepressVariant(
                        name="full_repressdb"
                    ),
                )
            ),
            plan=plan,
            evaluation_profile=profile,
            decision_profile=profile,
            evaluation_migration_cost=(
                migration_cost
            ),
            decision_migration_cost=(
                migration_cost
            ),
            migration_bytes=(
                migration_bytes
            ),
            initial_positions=initial,
            codecs=codecs,
            use_write_budget=True,
        )
        results["page_hinkley_be"] = (
            simulate_controller_model(
                policy_name=(
                    "page_hinkley_be"
                ),
                controller_factory=(
                    lambda initial_position: PageHinkleyBEController(
                        initial_position,
                        ph_config,
                        query_classes,
                    )
                ),
                plan=plan,
                evaluation_profile=profile,
                decision_profile=profile,
                evaluation_migration_cost=(
                    migration_cost
                ),
                decision_migration_cost=(
                    migration_cost
                ),
                migration_bytes=(
                    migration_bytes
                ),
                initial_positions=initial,
                codecs=codecs,
                use_write_budget=True,
            )
        )

        r1_context_mapping = {
            "s1": modules.s1,
        }
        results["afhc_d_frozen"] = (
            modules.r1.afhc_result(
                r1_context_mapping,
                plan,
                profile,
                initial,
                migration_cost,
                migration_bytes,
                codecs,
                afhc_config,
                modules.core,
                budget_bytes,
            )
        )
        modules.r1.common_stale_diagnostic(
            plan,
            results[
                "afhc_d_frozen"
            ],
            profile,
            codecs,
        )
        results["hindsight_oracle"] = (
            modules.r1.oracle_result(
                plan,
                profile,
                initial,
                migration_cost,
                migration_bytes,
            )
        )

        global_rows: List[
            Dict[str, Any]
        ] = []
        chunk_rows: List[
            Dict[str, Any]
        ] = []
        migration_rows: List[
            Dict[str, Any]
        ] = []
        for policy, result in results.items():
            global_rows.append(
                result_to_global_row(
                    result,
                    trace,
                    size,
                    trace_seed,
                    policy,
                    len(chunk_ids),
                    plan.n_events,
                )
            )
            chunk_rows.extend(
                chunk_results_to_rows(
                    result,
                    plan,
                    context.metadata,
                    trace_seed,
                    policy,
                )
            )
            migration_rows.extend(
                migration_records_to_rows(
                    result,
                    trace,
                    size,
                    trace_seed,
                    policy,
                )
            )

        ablation_rows: List[
            Dict[str, Any]
        ] = []
        if trace_seed == MULTISEED_TRACE_SEEDS[0]:
            for variant in REPRESS_VARIANTS:
                variant_result = (
                    simulate_controller_model(
                        policy_name=(
                            variant.name
                        ),
                        controller_factory=(
                            lambda initial_position,
                            local_variant=variant: RepressController(
                                initial_position,
                                local_variant,
                            )
                        ),
                        plan=plan,
                        evaluation_profile=profile,
                        decision_profile=profile,
                        evaluation_migration_cost=(
                            migration_cost
                        ),
                        decision_migration_cost=(
                            migration_cost
                        ),
                        migration_bytes=(
                            migration_bytes
                        ),
                        initial_positions=initial,
                        codecs=codecs,
                        use_write_budget=(
                            variant.use_write_budget
                        ),
                    )
                )
                ablation_rows.append(
                    result_to_global_row(
                        variant_result,
                        trace,
                        size,
                        trace_seed,
                        variant.name,
                        len(chunk_ids),
                        plan.n_events,
                    )
                )

        global_frame = pd.DataFrame(
            global_rows
        )
        chunk_frame = pd.DataFrame(
            chunk_rows
        )
        migration_frame = pd.DataFrame(
            migration_rows
        )
        if ablation_rows:
            ablation_frame = pd.DataFrame(
                ablation_rows
            )
        else:
            ablation_frame = pd.DataFrame(
                columns=[
                    "trace",
                    "chunk_size",
                    "trace_seed",
                    "policy",
                    "n_buildings",
                    "n_queries",
                    "query_time_ns",
                    "migration_time_ns",
                    "total_time_ns",
                    "migration_count",
                    "rewrite_bytes",
                    "budget_denied_proposals",
                    "budget_violation_count",
                    "unamortized_migration_count",
                    "unamortized_migration_rate",
                    "stale_assessable_migration_count",
                    "stale_migration_count",
                    "stale_migration_rate",
                ]
            )
        write_csv_atomic(
            global_path,
            global_frame,
        )
        write_csv_atomic(
            chunk_path,
            chunk_frame,
        )
        write_csv_atomic(
            migration_path,
            migration_frame,
        )
        write_csv_atomic(
            ablation_path,
            ablation_frame,
        )
        write_json_atomic(
            complete_path,
            {
                "complete": True,
                "freeze_sha256": (
                    freeze_hash
                ),
                "trace": trace,
                "chunk_size": int(size),
                "trace_seed": int(
                    trace_seed
                ),
                "completed_at_ns": (
                    time.time_ns()
                ),
            },
        )
        global_frames.append(
            global_frame
        )
        chunk_frames.append(
            chunk_frame
        )
        migration_frames.append(
            migration_frame
        )
        ablation_frames.append(
            ablation_frame
        )
        print(
            f"[J2 {cell_index}/{total_cells}] "
            f"{trace} size={size} seed={trace_seed}"
        )

    return (
        pd.concat(
            global_frames,
            ignore_index=True,
        ),
        pd.concat(
            chunk_frames,
            ignore_index=True,
        ),
        pd.concat(
            migration_frames,
            ignore_index=True,
        ),
        pd.concat(
            ablation_frames,
            ignore_index=True,
        ),
    )


def multiseed_paired_statistics(
    global_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    comparisons = (
        (
            "repressdb_frozen",
            "shape_ewma_be_frozen",
        ),
        (
            "repressdb_frozen",
            "page_hinkley_be",
        ),
        (
            "repressdb_frozen",
            "afhc_d_frozen",
        ),
        (
            "repressdb_frozen",
            "initial_best",
        ),
        (
            "repressdb_frozen",
            "best_static",
        ),
    )
    for (
        trace,
        size,
    ), group in global_frame.groupby(
        ["trace", "chunk_size"]
    ):
        pivot = group.pivot(
            index="trace_seed",
            columns="policy",
            values="total_time_ns",
        ).dropna()
        for numerator, denominator in comparisons:
            if (
                numerator not in pivot
                or denominator not in pivot
            ):
                continue
            ratio = (
                pivot[numerator].to_numpy(
                    dtype=np.float64
                )
                / pivot[
                    denominator
                ].to_numpy(
                    dtype=np.float64
                )
            )
            low, high = percentile_interval(
                ratio,
                (
                    "multiseed",
                    trace,
                    size,
                    numerator,
                    denominator,
                ),
                MULTISEED_BOOTSTRAPS,
                statistic="geometric_mean",
            )
            rows.append(
                {
                    "trace": trace,
                    "chunk_size": int(size),
                    "numerator_policy": (
                        numerator
                    ),
                    "denominator_policy": (
                        denominator
                    ),
                    "n_trace_seeds": int(
                        len(ratio)
                    ),
                    "geometric_mean_ratio": geometric_mean(
                        ratio
                    ),
                    "median_ratio": float(
                        np.median(ratio)
                    ),
                    "minimum_ratio": float(
                        np.min(ratio)
                    ),
                    "maximum_ratio": float(
                        np.max(ratio)
                    ),
                    "ci_low": low,
                    "ci_high": high,
                    "fraction_numerator_faster": float(
                        np.mean(ratio < 1.0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def ablation_statistics(
    ablation_frame: pd.DataFrame,
) -> pd.DataFrame:
    full = ablation_frame[
        ablation_frame["policy"]
        == "full_repressdb"
    ][
        [
            "trace",
            "chunk_size",
            "trace_seed",
            "total_time_ns",
            "migration_count",
            "unamortized_migration_rate",
            "stale_migration_rate",
            "budget_denied_proposals",
        ]
    ].rename(
        columns={
            "total_time_ns": (
                "full_total_time_ns"
            ),
            "migration_count": (
                "full_migration_count"
            ),
            "unamortized_migration_rate": (
                "full_unamortized_rate"
            ),
            "stale_migration_rate": (
                "full_stale_rate"
            ),
            "budget_denied_proposals": (
                "full_budget_denied"
            ),
        }
    )
    merged = ablation_frame.merge(
        full,
        on=[
            "trace",
            "chunk_size",
            "trace_seed",
        ],
        how="left",
        validate="many_to_one",
    )
    merged["total_ratio_to_full"] = (
        merged["total_time_ns"]
        / merged[
            "full_total_time_ns"
        ]
    )
    rows: List[Dict[str, Any]] = []
    for policy, group in merged.groupby(
        "policy"
    ):
        ratio = group[
            "total_ratio_to_full"
        ].to_numpy(
            dtype=np.float64
        )
        low, high = percentile_interval(
            ratio,
            ("ablation", policy),
            MULTISEED_BOOTSTRAPS,
            statistic="geometric_mean",
        )
        rows.append(
            {
                "variant": policy,
                "n_cells": int(
                    len(group)
                ),
                "geometric_mean_total_ratio_to_full": geometric_mean(
                    ratio
                ),
                "ci_low": low,
                "ci_high": high,
                "total_migrations": int(
                    group[
                        "migration_count"
                    ].sum()
                ),
                "median_unamortized_rate": float(
                    group[
                        "unamortized_migration_rate"
                    ].median()
                ),
                "median_stale_rate": float(
                    group[
                        "stale_migration_rate"
                    ].median()
                ),
                "total_budget_denials": int(
                    group[
                        "budget_denied_proposals"
                    ].sum()
                ),
                "budget_violations": int(
                    group[
                        "budget_violation_count"
                    ].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# J3: process termination and corruption fuzz
# ---------------------------------------------------------------------------

def prepare_crash_assets(
    paths: ProjectPaths,
    modules: LoadedModules,
    freeze_hash: str,
) -> Dict[str, Any]:
    root = (
        paths.output
        / "j3_crash_fuzz"
    )
    root.mkdir(
        parents=True,
        exist_ok=True,
    )
    base_store = root / "_base_store"
    values_path = root / "crash_chunk_values.npz"
    marker_path = (
        base_store / "base_store_complete.json"
    )

    chunks, source_metadata = (
        modules.b1.load_frozen_chunks(
            paths.project,
            modules.base,
        )
    )
    chunk = chunks[0]
    if not values_path.exists():
        temporary = values_path.with_name(
            values_path.name + ".tmp"
        )
        with open(
            temporary,
            "wb",
        ) as handle:
            np.savez_compressed(
                handle,
                chunk_id=np.asarray(
                    [chunk.chunk_id],
                    dtype="U128",
                ),
                series_index=np.asarray(
                    [chunk.series_index],
                    dtype=np.int32,
                ),
                values=np.asarray(
                    chunk.values,
                    dtype=np.float32,
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temporary,
            values_path,
        )

    if not checkpoint_complete(
        marker_path,
        freeze_hash,
    ):
        shutil.rmtree(
            base_store,
            ignore_errors=True,
        )
        base_store.mkdir(
            parents=True,
            exist_ok=True,
        )
        with modules.b1.SingleActiveChunkStore(
            base_store,
            modules.base,
            create=True,
            recover_on_open=False,
        ) as store:
            store.create_chunk(
                chunk_id=chunk.chunk_id,
                series_index=int(
                    chunk.series_index
                ),
                values=chunk.values,
                codec="whole_lz4",
            )
            integrity = (
                store.integrity_check()
            )
            if not integrity["pass"]:
                raise RuntimeError(
                    f"Crash base store invalid: {integrity}"
                )
            store.connection.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            )
            store.connection.commit()
        sqlite_checkpoint_database(
            base_store / "catalog.sqlite3"
        )
        write_json_atomic(
            marker_path,
            {
                "complete": True,
                "freeze_sha256": freeze_hash,
                "chunk_id": chunk.chunk_id,
                "initial_codec": (
                    "whole_lz4"
                ),
                "source_metadata": (
                    source_metadata
                ),
                "integrity": integrity,
            },
        )
    return {
        "root": str(root),
        "base_store": str(base_store),
        "values_path": str(values_path),
        "chunk_id": str(chunk.chunk_id),
        "initial_codec": "whole_lz4",
    }


def load_crash_chunk(
    modules: LoadedModules,
    values_path: Path,
) -> Any:
    with np.load(
        values_path,
        allow_pickle=False,
    ) as archive:
        chunk_id = str(
            archive["chunk_id"][0]
        )
        series_index = int(
            archive["series_index"][0]
        )
        values = np.ascontiguousarray(
            archive["values"],
            dtype=np.float32,
        )
    return modules.b1.FrozenChunk(
        chunk_id=chunk_id,
        series_index=series_index,
        chunk_start=0,
        chunk_size=int(values.size),
        values=values,
        raw_checksum=modules.b1.raw_array_sha256(values),
    )


def run_crash_trace_worker(
    config_path: Path,
) -> int:
    config = read_json(config_path)
    project = Path(
        str(config["project_dir"])
    )
    paths = resolve_project_paths(project)
    modules = load_all_modules(
        paths,
        f"crashtrace_{os.getpid()}",
    )
    store_root = Path(
        str(config["store_root"])
    )
    chunk_id = str(
        config["chunk_id"]
    )
    target_codec = str(
        config["target_codec"]
    )
    kill_after = config.get(
        "kill_after_line_events"
    )
    result_path = Path(
        str(config["result_path"])
    )

    line_count = 0
    active = False

    def trace_function(
        frame: Any,
        event: str,
        argument: Any,
    ) -> Any:
        nonlocal line_count
        if active and event == "line":
            line_count += 1
            if (
                kill_after is not None
                and line_count
                >= int(kill_after)
            ):
                os._exit(137)
        return trace_function

    with modules.b1.SingleActiveChunkStore(
        store_root,
        modules.base,
        create=True,
        recover_on_open=False,
    ) as store:
        sys.settrace(trace_function)
        active = True
        try:
            result = store.migrate(
                chunk_id,
                target_codec,
            )
        finally:
            active = False
            sys.settrace(None)
        write_json_atomic(
            result_path,
            {
                "line_count": int(
                    line_count
                ),
                "migration_result": (
                    result
                ),
            },
        )
    return 0


def run_crash_corruption_worker(
    config_path: Path,
) -> int:
    config = read_json(config_path)
    project = Path(
        str(config["project_dir"])
    )
    paths = resolve_project_paths(project)
    modules = load_all_modules(
        paths,
        f"crashcorrupt_{os.getpid()}",
    )
    store_root = Path(
        str(config["store_root"])
    )
    chunk_id = str(
        config["chunk_id"]
    )
    target_codec = str(
        config["target_codec"]
    )
    corruption = str(
        config["corruption"]
    )

    raw_store = modules.b1.SingleActiveChunkStore(
        store_root,
        modules.base,
        create=True,
        recover_on_open=False,
    )
    instrumented = InstrumentedSingleActiveChunkStore(
        raw_store,
        modules.b1,
    )

    def fault_hook(
        stage: str,
        context: Dict[str, Any],
    ) -> None:
        if stage != "after_catalog_commit":
            return
        if corruption == "payload_truncation":
            path = Path(
                context["final_data"]
            )
            with open(
                path,
                "r+b",
            ) as handle:
                handle.truncate(
                    max(
                        1,
                        path.stat().st_size // 3,
                    )
                )
                handle.flush()
                os.fsync(handle.fileno())
        elif corruption == "metadata_corruption":
            metadata_path = context.get(
                "final_metadata_path"
            )
            if metadata_path is None:
                raise RuntimeError(
                    "Metadata corruption target has no metadata file."
                )
            metadata_path = Path(
                metadata_path
            )
            payload = bytearray(
                metadata_path.read_bytes()
            )
            if not payload:
                raise RuntimeError(
                    "Metadata file is empty."
                )
            index = len(payload) // 2
            payload[index] ^= 0xA5
            with open(
                metadata_path,
                "wb",
            ) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            raise ValueError(corruption)
        os._exit(137)

    try:
        instrumented.migrate_profiled(
            chunk_id,
            target_codec,
            fault_hook=fault_hook,
        )
    finally:
        # os._exit should prevent this block from completing.
        with contextlib.suppress(Exception):
            instrumented.store.close()
    return 3


def verify_crash_trial(
    paths: ProjectPaths,
    modules: LoadedModules,
    trial_store: Path,
    chunk: Any,
    trial_type: str,
    trial_index: int,
    target_codec: str,
    child_exit_code: Optional[int],
    kill_after: Optional[int],
) -> Dict[str, Any]:
    pre_state = ""
    pre_active_codec = ""
    pre_log_states = ""
    recovery_actions: Dict[str, int] = {}
    verification_error = ""
    try:
        store = modules.b1.SingleActiveChunkStore(
            trial_store,
            modules.base,
            create=True,
            recover_on_open=False,
        )
        try:
            row = store._row(
                chunk.chunk_id
            )
            pre_state = str(
                row["migration_state"]
            )
            pre_active_codec = str(
                row["active_codec"]
            )
            log_frame = (
                store.migration_log_dataframe()
            )
            pre_log_states = "|".join(
                log_frame["state"]
                .astype(str)
                .tolist()
            )
            recovery_actions = (
                store.recover()
            )
            integrity = (
                store.integrity_check()
            )
            recovered_values = (
                store.read_all(
                    chunk.chunk_id
                )
            )
            full_array_ok = bool(
                np.array_equal(
                    recovered_values,
                    np.ascontiguousarray(
                        chunk.values,
                        dtype=np.float32,
                    ),
                )
            )
            query_ok_count = 0
            for query_index in range(
                CRASH_TRUTH_QUERIES_PER_TRIAL
            ):
                query = modules.b1.build_query(
                    modules.base,
                    chunk,
                    (
                        trial_index
                        * CRASH_TRUTH_QUERIES_PER_TRIAL
                        + query_index
                    ),
                )
                actual, _, _, _ = (
                    store.execute_query(
                        chunk.chunk_id,
                        query,
                    )
                )
                expected = (
                    modules.b1.expected_query_result(
                        modules.base,
                        chunk,
                        query,
                    )
                )
                query_ok_count += int(
                    modules.b1.query_result_equal(
                        modules.base,
                        actual,
                        expected,
                    )
                )
            final_row = store._row(
                chunk.chunk_id
            )
            final_active_codec = str(
                final_row["active_codec"]
            )
        finally:
            store.close()

        pass_value = bool(
            integrity["pass"]
            and full_array_ok
            and query_ok_count
            == CRASH_TRUTH_QUERIES_PER_TRIAL
        )
        return {
            "trial_type": trial_type,
            "trial_index": int(
                trial_index
            ),
            "target_codec": (
                target_codec
            ),
            "kill_after_line_events": (
                int(kill_after)
                if kill_after
                is not None
                else math.nan
            ),
            "child_exit_code": (
                int(child_exit_code)
                if child_exit_code
                is not None
                else math.nan
            ),
            "pre_recovery_state": (
                pre_state
            ),
            "pre_recovery_active_codec": (
                pre_active_codec
            ),
            "pre_recovery_log_states": (
                pre_log_states
            ),
            "final_active_codec": (
                final_active_codec
            ),
            "recovery_actions": (
                json.dumps(
                    recovery_actions,
                    sort_keys=True,
                )
            ),
            "integrity_pass": int(
                integrity["pass"]
            ),
            "full_array_ok": int(
                full_array_ok
            ),
            "query_ok_count": int(
                query_ok_count
            ),
            "temporary_file_count": int(
                integrity[
                    "temporary_file_count"
                ]
            ),
            "invalid_chunk_count": len(
                integrity[
                    "invalid_chunks"
                ]
            ),
            "non_idle_chunk_count": len(
                integrity[
                    "non_idle_chunks"
                ]
            ),
            "wrong_payload_count": len(
                integrity[
                    "chunks_without_exactly_one_payload"
                ]
            ),
            "pass": int(pass_value),
            "verification_error": "",
        }
    except Exception:
        verification_error = traceback.format_exc()
        return {
            "trial_type": trial_type,
            "trial_index": int(
                trial_index
            ),
            "target_codec": (
                target_codec
            ),
            "kill_after_line_events": (
                int(kill_after)
                if kill_after
                is not None
                else math.nan
            ),
            "child_exit_code": (
                int(child_exit_code)
                if child_exit_code
                is not None
                else math.nan
            ),
            "pre_recovery_state": (
                pre_state
            ),
            "pre_recovery_active_codec": (
                pre_active_codec
            ),
            "pre_recovery_log_states": (
                pre_log_states
            ),
            "final_active_codec": "",
            "recovery_actions": (
                json.dumps(
                    recovery_actions,
                    sort_keys=True,
                )
            ),
            "integrity_pass": 0,
            "full_array_ok": 0,
            "query_ok_count": 0,
            "temporary_file_count": math.nan,
            "invalid_chunk_count": math.nan,
            "non_idle_chunk_count": math.nan,
            "wrong_payload_count": math.nan,
            "pass": 0,
            "verification_error": (
                verification_error
            ),
        }


def calibrate_crash_line_counts(
    paths: ProjectPaths,
    crash_assets: Mapping[str, Any],
    freeze_hash: str,
) -> Dict[str, int]:
    output = (
        paths.output
        / "j3_crash_fuzz"
        / "line_count_calibration.json"
    )
    if output.exists():
        data = read_json(output)
        if data.get(
            "freeze_sha256"
        ) == freeze_hash:
            return {
                str(key): int(value)
                for key, value in data[
                    "line_counts"
                ].items()
            }

    targets = (
        "whole_zstd",
        "block_lz4_64",
        "block_lz4_256",
        "block_lz4_1024",
    )
    line_counts: Dict[str, int] = {}
    root = (
        paths.output
        / "j3_crash_fuzz"
        / "_line_calibration"
    )
    shutil.rmtree(
        root,
        ignore_errors=True,
    )
    root.mkdir(
        parents=True,
        exist_ok=True,
    )
    for target in targets:
        trial_store = (
            root / target
        )
        shutil.copytree(
            Path(
                str(
                    crash_assets[
                        "base_store"
                    ]
                )
            ),
            trial_store,
        )
        result_path = (
            root
            / f"{target}_result.json"
        )
        config_path = (
            root
            / f"{target}_config.json"
        )
        write_json_atomic(
            config_path,
            {
                "project_dir": str(
                    paths.project
                ),
                "store_root": str(
                    trial_store
                ),
                "chunk_id": str(
                    crash_assets[
                        "chunk_id"
                    ]
                ),
                "target_codec": target,
                "result_path": str(
                    result_path
                ),
                "kill_after_line_events": None,
            },
        )
        exit_code = (
            run_subprocess_checked(
                [
                    sys.executable,
                    str(
                        Path(
                            __file__
                        ).resolve()
                    ),
                    "--internal-worker",
                    "crash-trace",
                    "--config",
                    str(config_path),
                ],
                root
                / f"{target}.log",
            )
        )
        if (
            exit_code != 0
            or not result_path.exists()
        ):
            raise RuntimeError(
                f"Crash line calibration "
                f"failed for {target}"
            )
        line_counts[target] = int(
            read_json(result_path)[
                "line_count"
            ]
        )
    write_json_atomic(
        output,
        {
            "freeze_sha256": (
                freeze_hash
            ),
            "line_counts": (
                line_counts
            ),
        },
    )
    shutil.rmtree(
        root,
        ignore_errors=True,
    )
    return line_counts


def run_crash_fuzz(
    paths: ProjectPaths,
    modules: LoadedModules,
    crash_assets: Mapping[str, Any],
    freeze_hash: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    root = (
        paths.output
        / "j3_crash_fuzz"
    )
    records_path = (
        root / "crash_fuzz_records.jsonl"
    )
    progress_path = (
        root / "crash_fuzz_progress.json"
    )
    line_counts = (
        calibrate_crash_line_counts(
            paths,
            crash_assets,
            freeze_hash,
        )
    )
    chunk = load_crash_chunk(
        modules,
        Path(
            str(
                crash_assets[
                    "values_path"
                ]
            )
        ),
    )

    completed_keys: set[
        Tuple[str, int]
    ] = set()
    existing_rows: List[
        Dict[str, Any]
    ] = []
    if records_path.exists():
        with open(
            records_path,
            "r",
            encoding="utf-8",
        ) as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                existing_rows.append(row)
                completed_keys.add(
                    (
                        str(
                            row["trial_type"]
                        ),
                        int(
                            row["trial_index"]
                        ),
                    )
                )

    trial_plan: List[
        Dict[str, Any]
    ] = []
    targets = (
        "whole_zstd",
        "block_lz4_64",
        "block_lz4_256",
        "block_lz4_1024",
    )
    for index in range(
        CRASH_RANDOM_KILL_TRIALS
    ):
        target = targets[
            index % len(targets)
        ]
        maximum = int(
            line_counts[target]
        )
        low = max(
            1,
            int(maximum * 0.02),
        )
        high = max(
            low + 1,
            int(maximum * 0.98),
        )
        rng = np.random.default_rng(
            stable_seed(
                MASTER_SEED,
                "crash-kill",
                index,
                target,
            )
        )
        kill_after = int(
            rng.integers(
                low,
                high,
            )
        )
        trial_plan.append(
            {
                "trial_type": (
                    "random_process_termination"
                ),
                "trial_index": index,
                "target_codec": target,
                "kill_after_line_events": (
                    kill_after
                ),
            }
        )
    for index in range(
        CRASH_PAYLOAD_TRUNCATION_TRIALS
    ):
        trial_plan.append(
            {
                "trial_type": (
                    "payload_truncation_after_commit"
                ),
                "trial_index": index,
                "target_codec": (
                    "block_lz4_256"
                ),
            }
        )
    for index in range(
        CRASH_METADATA_CORRUPTION_TRIALS
    ):
        trial_plan.append(
            {
                "trial_type": (
                    "metadata_corruption_after_commit"
                ),
                "trial_index": index,
                "target_codec": (
                    "block_lz4_1024"
                ),
            }
        )
    for index in range(
        CRASH_ORPHAN_TRIALS
    ):
        trial_plan.append(
            {
                "trial_type": (
                    "orphan_file_recovery"
                ),
                "trial_index": index,
                "target_codec": "",
            }
        )
    if len(trial_plan) != CRASH_TOTAL_TRIALS:
        raise AssertionError(
            f"Crash trial plan count "
            f"{len(trial_plan)} != "
            f"{CRASH_TOTAL_TRIALS}"
        )

    trial_root = root / "_trials"
    trial_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    for plan_index, trial in enumerate(
        trial_plan,
        start=1,
    ):
        key = (
            str(trial["trial_type"]),
            int(trial["trial_index"]),
        )
        if key in completed_keys:
            continue
        local_root = (
            trial_root
            / (
                f"{trial['trial_type']}__"
                f"{int(trial['trial_index']):05d}"
            )
        )
        shutil.rmtree(
            local_root,
            ignore_errors=True,
        )
        shutil.copytree(
            Path(
                str(
                    crash_assets[
                        "base_store"
                    ]
                )
            ),
            local_root,
        )
        child_exit_code: Optional[int] = None
        kill_after = trial.get(
            "kill_after_line_events"
        )
        target = str(
            trial.get(
                "target_codec",
                "",
            )
        )
        try:
            if trial[
                "trial_type"
            ] == "random_process_termination":
                config_path = (
                    local_root
                    / "worker_config.json"
                )
                result_path = (
                    local_root
                    / "worker_result.json"
                )
                write_json_atomic(
                    config_path,
                    {
                        "project_dir": str(
                            paths.project
                        ),
                        "store_root": str(
                            local_root
                        ),
                        "chunk_id": (
                            crash_assets[
                                "chunk_id"
                            ]
                        ),
                        "target_codec": target,
                        "result_path": str(
                            result_path
                        ),
                        "kill_after_line_events": int(
                            kill_after
                        ),
                    },
                )
                child_exit_code = (
                    run_subprocess_checked(
                        [
                            sys.executable,
                            str(
                                Path(
                                    __file__
                                ).resolve()
                            ),
                            "--internal-worker",
                            "crash-trace",
                            "--config",
                            str(
                                config_path
                            ),
                        ],
                        local_root
                        / "worker.log",
                    )
                )
            elif trial[
                "trial_type"
            ] in (
                "payload_truncation_after_commit",
                "metadata_corruption_after_commit",
            ):
                corruption = (
                    "payload_truncation"
                    if trial[
                        "trial_type"
                    ].startswith(
                        "payload"
                    )
                    else (
                        "metadata_corruption"
                    )
                )
                config_path = (
                    local_root
                    / "worker_config.json"
                )
                write_json_atomic(
                    config_path,
                    {
                        "project_dir": str(
                            paths.project
                        ),
                        "store_root": str(
                            local_root
                        ),
                        "chunk_id": (
                            crash_assets[
                                "chunk_id"
                            ]
                        ),
                        "target_codec": target,
                        "corruption": (
                            corruption
                        ),
                    },
                )
                child_exit_code = (
                    run_subprocess_checked(
                        [
                            sys.executable,
                            str(
                                Path(
                                    __file__
                                ).resolve()
                            ),
                            "--internal-worker",
                            "crash-corrupt",
                            "--config",
                            str(
                                config_path
                            ),
                        ],
                        local_root
                        / "worker.log",
                    )
                )
            elif trial[
                "trial_type"
            ] == "orphan_file_recovery":
                orphan_staging = (
                    local_root
                    / "staging"
                    / chunk.chunk_id
                    / (
                        f"orphan_{trial['trial_index']}.bin"
                    )
                )
                orphan_final = (
                    local_root
                    / "chunks"
                    / chunk.chunk_id
                    / (
                        f"orphan_{trial['trial_index']}.bin"
                    )
                )
                orphan_staging.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                orphan_final.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                orphan_staging.write_bytes(
                    os.urandom(257)
                )
                orphan_final.write_bytes(
                    os.urandom(509)
                )
            else:
                raise ValueError(
                    trial["trial_type"]
                )

            record = verify_crash_trial(
                paths,
                modules,
                local_root,
                chunk,
                str(
                    trial[
                        "trial_type"
                    ]
                ),
                int(
                    trial[
                        "trial_index"
                    ]
                ),
                target,
                child_exit_code,
                (
                    int(kill_after)
                    if kill_after
                    is not None
                    else None
                ),
            )
            if (
                trial["trial_type"]
                != "orphan_file_recovery"
                and child_exit_code == 0
            ):
                record[
                    "verification_error"
                ] = (
                    record.get(
                        "verification_error",
                        ""
                    )
                    + "\nChild completed without abrupt termination."
                ).strip()
                record["pass"] = 0
            append_jsonl_fsync(
                records_path,
                record,
            )
            existing_rows.append(record)
            completed_keys.add(key)
            write_json_atomic(
                progress_path,
                {
                    "freeze_sha256": (
                        freeze_hash
                    ),
                    "completed_trials": int(
                        len(completed_keys)
                    ),
                    "total_trials": int(
                        CRASH_TOTAL_TRIALS
                    ),
                    "last_key": list(
                        key
                    ),
                },
            )
            print(
                f"[J3 {len(completed_keys)}/"
                f"{CRASH_TOTAL_TRIALS}] "
                f"{trial['trial_type']} "
                f"trial={trial['trial_index']} "
                f"pass={record['pass']}"
            )
        finally:
            shutil.rmtree(
                local_root,
                ignore_errors=True,
            )

    frame = pd.DataFrame(
        existing_rows
    ).sort_values(
        [
            "trial_type",
            "trial_index",
        ]
    ).reset_index(drop=True)
    summary = (
        frame.groupby(
            "trial_type",
            dropna=False,
        )
        .agg(
            trials=("pass", "size"),
            passed=("pass", "sum"),
            integrity_passed=(
                "integrity_pass",
                "sum",
            ),
            full_array_passed=(
                "full_array_ok",
                "sum",
            ),
            median_query_ok=(
                "query_ok_count",
                "median",
            ),
        )
        .reset_index()
    )
    summary["pass_rate"] = (
        summary["passed"]
        / summary["trials"]
    )
    write_csv_atomic(
        root
        / "crash_fuzz_records.csv",
        frame,
    )
    write_csv_atomic(
        root
        / "crash_fuzz_summary.csv",
        summary,
    )
    write_json_atomic(
        root
        / "complete.json",
        {
            "complete": True,
            "freeze_sha256": (
                freeze_hash
            ),
            "total_trials": int(
                len(frame)
            ),
            "all_pass": bool(
                len(frame)
                == CRASH_TOTAL_TRIALS
                and (
                    frame["pass"] == 1
                ).all()
            ),
            "line_count_calibration": (
                line_counts
            ),
            "records_sha256": (
                sha256_file(
                    root
                    / "crash_fuzz_records.csv"
                )
            ),
        },
    )
    return frame, summary


# ---------------------------------------------------------------------------
# J4: calibration overhead, sampled profiles, and cost-table errors
# ---------------------------------------------------------------------------

def process_io_snapshot() -> Dict[str, int]:
    if psutil is None:
        return {
            "read_bytes": 0,
            "write_bytes": 0,
            "read_count": 0,
            "write_count": 0,
        }
    counters = psutil.Process(
        os.getpid()
    ).io_counters()
    return {
        "read_bytes": int(
            getattr(
                counters,
                "read_bytes",
                0,
            )
        ),
        "write_bytes": int(
            getattr(
                counters,
                "write_bytes",
                0,
            )
        ),
        "read_count": int(
            getattr(
                counters,
                "read_count",
                0,
            )
        ),
        "write_count": int(
            getattr(
                counters,
                "write_count",
                0,
            )
        ),
    }


def process_cpu_snapshot() -> Dict[str, float]:
    if psutil is None:
        return {
            "user_seconds": 0.0,
            "system_seconds": 0.0,
        }
    cpu = psutil.Process(
        os.getpid()
    ).cpu_times()
    return {
        "user_seconds": float(
            cpu.user
        ),
        "system_seconds": float(
            cpu.system
        ),
    }


def difference_mapping(
    after: Mapping[str, Any],
    before: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        key: after.get(key, 0)
        - before.get(key, 0)
        for key in set(after)
        | set(before)
    }


def run_fresh_calibration_overhead(
    paths: ProjectPaths,
    modules: LoadedModules,
    context: R1Context,
    freeze_hash: str,
) -> Dict[str, Any]:
    root = (
        paths.output
        / "j4_calibration"
        / "fresh_full_calibration"
    )
    summary_path = (
        root
        / "calibration_overhead_summary.json"
    )
    complete_path = root / "complete.json"
    if checkpoint_complete(
        complete_path,
        freeze_hash,
    ):
        return read_json(summary_path)

    # An interrupted run cannot produce an unbiased full wall/I/O total.
    # Restart this stage from an empty directory.
    shutil.rmtree(
        root,
        ignore_errors=True,
    )
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    reps = modules.base.representation_registry(
        3
    )
    io_before = process_io_snapshot()
    cpu_before = process_cpu_snapshot()
    disk_before = directory_size_bytes(
        root
    )
    wall_start = time.perf_counter_ns()

    calibration_freeze = sha256_json(
        {
            "revision_freeze": freeze_hash,
            "stage": (
                "fresh_full_calibration"
            ),
            "chunk_manifest_sha256": (
                sha256_file(
                    paths.results["r1"]
                    / "r1_chunk_manifest.csv"
                )
            ),
        }
    )
    (
        stored_by_chunk,
        profiles,
        encode,
        decode,
        migration_costs,
        migration_bytes,
        calibration_frame,
        codec_frame,
        migration_frame,
    ) = modules.a2.prepare_calibration(
        "BDG2_JSS_FRESH_CALIBRATION",
        root,
        calibration_freeze,
        modules.core,
        modules.e1,
        modules.base,
        reps,
        context.chunks,
        context.metadata,
        context.query_pools,
        modules.v074,
    )

    wall_elapsed_ns = (
        time.perf_counter_ns()
        - wall_start
    )
    io_after = process_io_snapshot()
    cpu_after = process_cpu_snapshot()
    disk_after = directory_size_bytes(
        root
    )
    io_delta = difference_mapping(
        io_after,
        io_before,
    )
    cpu_delta = difference_mapping(
        cpu_after,
        cpu_before,
    )
    candidate_storage = (
        root / "_candidate_storage"
    )
    candidate_bytes = directory_size_bytes(
        candidate_storage
    )
    raw_bytes = int(
        context.chunk_manifest[
            "raw_bytes"
        ].sum()
    )
    represented_bytes = int(
        (
            codec_frame[
                "payload_size_bytes"
            ]
            + codec_frame[
                "metadata_size_bytes"
            ]
        ).sum()
    )
    summary = {
        "stage": (
            "fresh_full_calibration"
        ),
        "n_buildings": int(
            context.chunk_manifest[
                "building_id"
            ].nunique()
        ),
        "n_chunks": int(
            context.chunk_manifest[
                "chunk_id"
            ].nunique()
        ),
        "chunk_sizes": list(
            CHUNK_SIZES
        ),
        "candidate_codec_count": int(
            codec_frame[
                "codec"
            ].nunique()
        ),
        "query_class_count": int(
            calibration_frame[
                "query_class"
            ].nunique()
        ),
        "directed_migration_pairs": int(
            migration_frame[
                [
                    "current_codec",
                    "target_codec",
                ]
            ].drop_duplicates().shape[0]
        ),
        "wall_elapsed_ns": int(
            wall_elapsed_ns
        ),
        "wall_elapsed_minutes": float(
            wall_elapsed_ns
            / 60e9
        ),
        "cpu_user_seconds": float(
            cpu_delta[
                "user_seconds"
            ]
        ),
        "cpu_system_seconds": float(
            cpu_delta[
                "system_seconds"
            ]
        ),
        "process_read_bytes": int(
            io_delta["read_bytes"]
        ),
        "process_write_bytes": int(
            io_delta["write_bytes"]
        ),
        "process_read_count": int(
            io_delta["read_count"]
        ),
        "process_write_count": int(
            io_delta["write_count"]
        ),
        "output_directory_bytes": int(
            disk_after - disk_before
        ),
        "candidate_storage_bytes": int(
            candidate_bytes
        ),
        "raw_chunk_bytes": int(
            raw_bytes
        ),
        "all_candidate_representation_bytes": int(
            represented_bytes
        ),
        "candidate_storage_over_raw_ratio": float(
            candidate_bytes
            / max(raw_bytes, 1)
        ),
        "calibration_record_count": int(
            len(calibration_frame)
        ),
        "codec_record_count": int(
            len(codec_frame)
        ),
        "migration_record_count": int(
            len(migration_frame)
        ),
        "calibration_records_sha256": sha256_file(
            root
            / "calibration_records.csv"
        ),
        "codec_records_sha256": sha256_file(
            root
            / "codec_records.csv"
        ),
        "migration_records_sha256": sha256_file(
            root
            / "measured_migration_costs.csv"
        ),
    }
    write_json_atomic(
        summary_path,
        summary,
    )
    write_json_atomic(
        complete_path,
        {
            "complete": True,
            "freeze_sha256": (
                freeze_hash
            ),
            "summary_sha256": (
                sha256_file(
                    summary_path
                )
            ),
            "completed_at_ns": (
                time.time_ns()
            ),
        },
    )
    return summary


def generic_sampled_decision_tables(
    true_profile: np.ndarray,
    true_migration_cost: np.ndarray,
    sample_positions: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    sample_positions = np.asarray(
        list(sample_positions),
        dtype=np.int64,
    )
    median_profile = np.median(
        true_profile[
            sample_positions
        ],
        axis=0,
    )
    median_migration = np.median(
        true_migration_cost[
            sample_positions
        ],
        axis=0,
    )
    decision_profile = np.broadcast_to(
        median_profile,
        true_profile.shape,
    ).copy()
    decision_migration = np.broadcast_to(
        median_migration,
        true_migration_cost.shape,
    ).copy()
    for position in range(
        decision_migration.shape[1]
    ):
        decision_migration[
            :,
            position,
            position,
        ] = 0.0
    return (
        decision_profile,
        decision_migration,
    )


def perturb_decision_tables(
    true_profile: np.ndarray,
    true_migration_cost: np.ndarray,
    error_level: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if error_level == 0:
        return (
            true_profile.copy(),
            true_migration_cost.copy(),
        )
    rng = np.random.default_rng(
        stable_seed(
            MASTER_SEED,
            "cost-error",
            seed,
            error_level,
        )
    )
    profile_multiplier = rng.uniform(
        1.0 - error_level,
        1.0 + error_level,
        size=true_profile.shape,
    )
    migration_multiplier = rng.uniform(
        1.0 - error_level,
        1.0 + error_level,
        size=true_migration_cost.shape,
    )
    decision_profile = (
        true_profile
        * profile_multiplier
    )
    decision_migration = (
        true_migration_cost
        * migration_multiplier
    )
    for position in range(
        decision_migration.shape[1]
    ):
        decision_migration[
            :,
            position,
            position,
        ] = 0.0
    return (
        decision_profile,
        decision_migration,
    )


def migration_signature_set(
    result: Mapping[str, Any],
) -> set[Tuple[str, int, str]]:
    signatures: set[
        Tuple[str, int, str]
    ] = set()
    for episode in result.get(
        "migration_records",
        [],
    ):
        if dataclasses.is_dataclass(
            episode
        ):
            signatures.add(
                (
                    str(
                        episode.chunk_id
                    ),
                    int(
                        episode.start_global_event_index
                    ),
                    str(
                        episode.target_codec
                    ),
                )
            )
        else:
            signatures.add(
                (
                    str(
                        episode[
                            "chunk_id"
                        ]
                    ),
                    int(
                        episode[
                            "start_global_event_index"
                        ]
                    ),
                    str(
                        episode[
                            "target_codec"
                        ]
                    ),
                )
            )
    return signatures


def signature_agreement(
    reference: set[
        Tuple[str, int, str]
    ],
    candidate: set[
        Tuple[str, int, str]
    ],
) -> Dict[str, float]:
    intersection = len(
        reference & candidate
    )
    union = len(
        reference | candidate
    )
    precision = (
        intersection
        / len(candidate)
        if candidate
        else (
            1.0
            if not reference
            else 0.0
        )
    )
    recall = (
        intersection
        / len(reference)
        if reference
        else (
            1.0
            if not candidate
            else 0.0
        )
    )
    jaccard = (
        intersection / union
        if union
        else 1.0
    )
    return {
        "migration_precision": float(
            precision
        ),
        "migration_recall": float(
            recall
        ),
        "migration_jaccard": float(
            jaccard
        ),
    }


def simulate_shape_and_repress_with_decision_tables(
    modules: LoadedModules,
    plan: Any,
    true_profile: np.ndarray,
    decision_profile: np.ndarray,
    true_migration_cost: np.ndarray,
    decision_migration_cost: np.ndarray,
    migration_bytes: np.ndarray,
    initial: np.ndarray,
    codecs: Sequence[str],
    query_classes: Sequence[str],
    shape_config: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    initial_weights = initial_weights_from_base(
        modules,
        query_classes,
    )
    shape = simulate_controller_model(
        policy_name=(
            "shape_ewma_be_frozen"
        ),
        controller_factory=(
            lambda initial_position: ShapeEWMAController(
                initial_position,
                initial_weights,
                shape_config,
            )
        ),
        plan=plan,
        evaluation_profile=(
            true_profile
        ),
        decision_profile=(
            decision_profile
        ),
        evaluation_migration_cost=(
            true_migration_cost
        ),
        decision_migration_cost=(
            decision_migration_cost
        ),
        migration_bytes=(
            migration_bytes
        ),
        initial_positions=initial,
        codecs=codecs,
        use_write_budget=True,
    )
    repress = simulate_controller_model(
        policy_name=(
            "repressdb_frozen"
        ),
        controller_factory=(
            lambda initial_position: RepressController(
                initial_position,
                RepressVariant(
                    name="full_repressdb"
                ),
            )
        ),
        plan=plan,
        evaluation_profile=(
            true_profile
        ),
        decision_profile=(
            decision_profile
        ),
        evaluation_migration_cost=(
            true_migration_cost
        ),
        decision_migration_cost=(
            decision_migration_cost
        ),
        migration_bytes=(
            migration_bytes
        ),
        initial_positions=initial,
        codecs=codecs,
        use_write_budget=True,
    )
    return {
        "shape_ewma_be_frozen": shape,
        "repressdb_frozen": repress,
    }


def run_calibration_robustness(
    paths: ProjectPaths,
    modules: LoadedModules,
    context: R1Context,
    freeze_hash: str,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    root = (
        paths.output
        / "j4_calibration"
        / "robustness"
    )
    checkpoint_root = (
        root / "_checkpoints"
    )
    checkpoint_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    codecs = tuple(
        sorted(
            modules.base.representation_registry(
                3
            )
        )
    )
    query_classes = tuple(
        modules.core.QUERY_CLASSES
    )
    shape_config = read_json(
        paths.results["s1"]
        / "shape_ewma_be_frozen_config.json"
    )
    plans = {
        (
            trace,
            size,
        ): load_frozen_r1_plan(
            paths,
            modules,
            trace,
            size,
        )
        for trace in CALIBRATION_ROBUSTNESS_TRACES
        for size in CHUNK_SIZES
    }

    full_reference_rows: List[
        Dict[str, Any]
    ] = []
    full_reference_results: Dict[
        Tuple[str, int, str],
        Dict[str, Any],
    ] = {}
    prepared: Dict[
        Tuple[str, int],
        Dict[str, Any],
    ] = {}

    for (
        trace,
        size,
    ), plan in plans.items():
        chunk_ids = tuple(
            plan.chunk_ids
        )
        true_profile = (
            modules.s1.profile_array(
                context.profiles,
                chunk_ids,
                codecs,
                query_classes,
            )
        )
        initial = (
            modules.s1.scan_best_positions(
                modules.e1,
                modules.base,
                context.profiles,
                chunk_ids,
                codecs,
            )
        )
        (
            true_migration_cost,
            migration_bytes,
        ) = modules.s1.migration_matrix(
            context.migration_costs,
            context.migration_bytes,
            chunk_ids,
            codecs,
        )
        full = (
            simulate_shape_and_repress_with_decision_tables(
                modules,
                plan,
                true_profile,
                true_profile,
                true_migration_cost,
                true_migration_cost,
                migration_bytes,
                initial,
                codecs,
                query_classes,
                shape_config,
            )
        )
        prepared[
            (
                trace,
                size,
            )
        ] = {
            "plan": plan,
            "true_profile": (
                true_profile
            ),
            "initial": initial,
            "true_migration_cost": (
                true_migration_cost
            ),
            "migration_bytes": (
                migration_bytes
            ),
        }
        for policy, result in full.items():
            full_reference_results[
                (
                    trace,
                    size,
                    policy,
                )
            ] = result
            full_reference_rows.append(
                {
                    "trace": trace,
                    "chunk_size": int(
                        size
                    ),
                    "policy": policy,
                    "total_time_ns": float(
                        result[
                            "total_time_ns"
                        ]
                    ),
                    "migration_count": int(
                        result[
                            "migration_count"
                        ]
                    ),
                    "rewrite_bytes": int(
                        result[
                            "rewrite_bytes"
                        ]
                    ),
                    "unamortized_rate": float(
                        result[
                            "unamortized_migration_rate"
                        ]
                    ),
                    "stale_rate": float(
                        result[
                            "stale_migration_rate"
                        ]
                    ),
                }
            )
    reference_frame = pd.DataFrame(
        full_reference_rows
    )
    write_csv_atomic(
        root
        / "full_profile_reference.csv",
        reference_frame,
    )

    sampled_rows: List[
        Dict[str, Any]
    ] = []
    for fraction in CALIBRATION_SAMPLE_FRACTIONS:
        for sample_seed in CALIBRATION_SAMPLE_SEEDS:
            for (
                trace,
                size,
            ), local in prepared.items():
                checkpoint = (
                    checkpoint_root
                    / (
                        f"sample_fraction_{fraction:.2f}__"
                        f"seed_{sample_seed}__"
                        f"{trace}__size_{size}.json"
                    )
                )
                if checkpoint_complete(
                    checkpoint,
                    freeze_hash,
                ):
                    sampled_rows.extend(
                        read_json(
                            checkpoint
                        )["rows"]
                    )
                    continue

                plan = local["plan"]
                n_chunks = len(
                    plan.chunk_ids
                )
                sample_count = max(
                    1,
                    int(
                        round(
                            fraction
                            * n_chunks
                        )
                    ),
                )
                rng = np.random.default_rng(
                    stable_seed(
                        MASTER_SEED,
                        "profile-sample",
                        fraction,
                        sample_seed,
                        trace,
                        size,
                    )
                )
                sample_positions = np.sort(
                    rng.choice(
                        n_chunks,
                        size=sample_count,
                        replace=False,
                    )
                )
                (
                    decision_profile,
                    decision_migration,
                ) = generic_sampled_decision_tables(
                    local[
                        "true_profile"
                    ],
                    local[
                        "true_migration_cost"
                    ],
                    sample_positions,
                )
                results = (
                    simulate_shape_and_repress_with_decision_tables(
                        modules,
                        plan,
                        local[
                            "true_profile"
                        ],
                        decision_profile,
                        local[
                            "true_migration_cost"
                        ],
                        decision_migration,
                        local[
                            "migration_bytes"
                        ],
                        local[
                            "initial"
                        ],
                        codecs,
                        query_classes,
                        shape_config,
                    )
                )
                local_rows: List[
                    Dict[str, Any]
                ] = []
                for policy, result in results.items():
                    reference = (
                        full_reference_results[
                            (
                                trace,
                                size,
                                policy,
                            )
                        ]
                    )
                    agreement = (
                        signature_agreement(
                            migration_signature_set(
                                reference
                            ),
                            migration_signature_set(
                                result
                            ),
                        )
                    )
                    row = {
                        "trace": trace,
                        "chunk_size": int(
                            size
                        ),
                        "policy": policy,
                        "sample_fraction": float(
                            fraction
                        ),
                        "sample_seed": int(
                            sample_seed
                        ),
                        "sampled_chunk_count": int(
                            sample_count
                        ),
                        "total_time_ns": float(
                            result[
                                "total_time_ns"
                            ]
                        ),
                        "full_profile_total_time_ns": float(
                            reference[
                                "total_time_ns"
                            ]
                        ),
                        "total_ratio_to_full_profile": float(
                            result[
                                "total_time_ns"
                            ]
                            / reference[
                                "total_time_ns"
                            ]
                        ),
                        "migration_count": int(
                            result[
                                "migration_count"
                            ]
                        ),
                        "full_profile_migration_count": int(
                            reference[
                                "migration_count"
                            ]
                        ),
                        "unamortized_rate": float(
                            result[
                                "unamortized_migration_rate"
                            ]
                        ),
                        "stale_rate": float(
                            result[
                                "stale_migration_rate"
                            ]
                        ),
                        **agreement,
                    }
                    local_rows.append(
                        row
                    )
                    sampled_rows.append(
                        row
                    )
                write_json_atomic(
                    checkpoint,
                    {
                        "complete": True,
                        "freeze_sha256": (
                            freeze_hash
                        ),
                        "rows": local_rows,
                    },
                )
                print(
                    f"[J4 sample] fraction={fraction:.2f} "
                    f"seed={sample_seed} "
                    f"{trace} size={size}"
                )

    error_rows: List[
        Dict[str, Any]
    ] = []
    for error_level in COST_ERROR_LEVELS:
        for error_seed in COST_ERROR_SEEDS:
            for (
                trace,
                size,
            ), local in prepared.items():
                checkpoint = (
                    checkpoint_root
                    / (
                        f"error_{error_level:.2f}__"
                        f"seed_{error_seed}__"
                        f"{trace}__size_{size}.json"
                    )
                )
                if checkpoint_complete(
                    checkpoint,
                    freeze_hash,
                ):
                    error_rows.extend(
                        read_json(
                            checkpoint
                        )["rows"]
                    )
                    continue

                (
                    decision_profile,
                    decision_migration,
                ) = perturb_decision_tables(
                    local[
                        "true_profile"
                    ],
                    local[
                        "true_migration_cost"
                    ],
                    error_level,
                    error_seed,
                )
                results = (
                    simulate_shape_and_repress_with_decision_tables(
                        modules,
                        local["plan"],
                        local[
                            "true_profile"
                        ],
                        decision_profile,
                        local[
                            "true_migration_cost"
                        ],
                        decision_migration,
                        local[
                            "migration_bytes"
                        ],
                        local[
                            "initial"
                        ],
                        codecs,
                        query_classes,
                        shape_config,
                    )
                )
                local_rows: List[
                    Dict[str, Any]
                ] = []
                for policy, result in results.items():
                    reference = (
                        full_reference_results[
                            (
                                trace,
                                size,
                                policy,
                            )
                        ]
                    )
                    agreement = (
                        signature_agreement(
                            migration_signature_set(
                                reference
                            ),
                            migration_signature_set(
                                result
                            ),
                        )
                    )
                    row = {
                        "trace": trace,
                        "chunk_size": int(
                            size
                        ),
                        "policy": policy,
                        "error_level": float(
                            error_level
                        ),
                        "error_seed": int(
                            error_seed
                        ),
                        "total_time_ns": float(
                            result[
                                "total_time_ns"
                            ]
                        ),
                        "full_profile_total_time_ns": float(
                            reference[
                                "total_time_ns"
                            ]
                        ),
                        "total_ratio_to_full_profile": float(
                            result[
                                "total_time_ns"
                            ]
                            / reference[
                                "total_time_ns"
                            ]
                        ),
                        "migration_count": int(
                            result[
                                "migration_count"
                            ]
                        ),
                        "full_profile_migration_count": int(
                            reference[
                                "migration_count"
                            ]
                        ),
                        "unamortized_rate": float(
                            result[
                                "unamortized_migration_rate"
                            ]
                        ),
                        "stale_rate": float(
                            result[
                                "stale_migration_rate"
                            ]
                        ),
                        **agreement,
                    }
                    local_rows.append(
                        row
                    )
                    error_rows.append(
                        row
                    )
                write_json_atomic(
                    checkpoint,
                    {
                        "complete": True,
                        "freeze_sha256": (
                            freeze_hash
                        ),
                        "rows": local_rows,
                    },
                )
                print(
                    f"[J4 error] level={error_level:.2f} "
                    f"seed={error_seed} "
                    f"{trace} size={size}"
                )

    sampled_frame = pd.DataFrame(
        sampled_rows
    )
    error_frame = pd.DataFrame(
        error_rows
    )
    sampled_summary = (
        sampled_frame.groupby(
            [
                "sample_fraction",
                "policy",
            ]
        )
        .agg(
            n_cells=(
                "total_ratio_to_full_profile",
                "size",
            ),
            geometric_mean_total_ratio=(
                "total_ratio_to_full_profile",
                geometric_mean,
            ),
            median_total_ratio=(
                "total_ratio_to_full_profile",
                "median",
            ),
            median_migration_jaccard=(
                "migration_jaccard",
                "median",
            ),
            median_migration_precision=(
                "migration_precision",
                "median",
            ),
            median_migration_recall=(
                "migration_recall",
                "median",
            ),
            median_unamortized_rate=(
                "unamortized_rate",
                "median",
            ),
            median_stale_rate=(
                "stale_rate",
                "median",
            ),
        )
        .reset_index()
    )
    error_summary = (
        error_frame.groupby(
            [
                "error_level",
                "policy",
            ]
        )
        .agg(
            n_cells=(
                "total_ratio_to_full_profile",
                "size",
            ),
            geometric_mean_total_ratio=(
                "total_ratio_to_full_profile",
                geometric_mean,
            ),
            median_total_ratio=(
                "total_ratio_to_full_profile",
                "median",
            ),
            p95_total_ratio=(
                "total_ratio_to_full_profile",
                lambda values: float(
                    np.quantile(
                        values,
                        0.95,
                    )
                ),
            ),
            median_migration_jaccard=(
                "migration_jaccard",
                "median",
            ),
            median_unamortized_rate=(
                "unamortized_rate",
                "median",
            ),
            median_stale_rate=(
                "stale_rate",
                "median",
            ),
        )
        .reset_index()
    )

    write_csv_atomic(
        root
        / "sampled_profile_results.csv",
        sampled_frame,
    )
    write_csv_atomic(
        root
        / "sampled_profile_summary.csv",
        sampled_summary,
    )
    write_csv_atomic(
        root
        / "cost_error_results.csv",
        error_frame,
    )
    write_csv_atomic(
        root
        / "cost_error_summary.csv",
        error_summary,
    )
    write_json_atomic(
        root / "complete.json",
        {
            "complete": True,
            "freeze_sha256": (
                freeze_hash
            ),
            "sampled_rows": int(
                len(sampled_frame)
            ),
            "error_rows": int(
                len(error_frame)
            ),
            "sampled_results_sha256": (
                sha256_file(
                    root
                    / "sampled_profile_results.csv"
                )
            ),
            "error_results_sha256": (
                sha256_file(
                    root
                    / "cost_error_results.csv"
                )
            ),
        },
    )
    return (
        sampled_frame,
        error_frame,
        pd.concat(
            [
                sampled_summary.assign(
                    study="sampled_profile"
                ),
                error_summary.assign(
                    study="cost_error"
                ),
            ],
            ignore_index=True,
            sort=False,
        ),
    )


# ---------------------------------------------------------------------------
# Consolidation, checks, and main entry point
# ---------------------------------------------------------------------------

def flatten_result_files_for_manifest(
    root: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(
            part.startswith("_store")
            or part
            in (
                "_trials",
                "_base_stores",
                "_candidate_storage",
                "_calibration_checkpoints",
                "_migration_trials",
                "_cell_checkpoints",
                "_checkpoints",
                "_patched_core",
            )
            for part in relative.parts
        ):
            continue
        rows.append(
            {
                "relative_path": str(
                    relative
                ).replace("\\", "/"),
                "size_bytes": int(
                    path.stat().st_size
                ),
                "sha256": sha256_file(
                    path
                ),
            }
        )
    return pd.DataFrame(rows)


def create_revision_result_archives(
    paths: ProjectPaths,
) -> Dict[str, Any]:
    output = paths.output
    for previous_archive in (
        output / "RePressDB_JSS_REVISION_CORE_RESULTS_V151.zip",
        output / "RePressDB_JSS_REVISION_LATENCY_ARRAYS_V151.zip",
    ):
        previous_archive.unlink(missing_ok=True)
    manifest = flatten_result_files_for_manifest(
        output
    )
    manifest_path = (
        output / "result_file_manifest.csv"
    )
    write_csv_atomic(
        manifest_path,
        manifest,
    )

    core_zip = (
        output
        / "RePressDB_JSS_REVISION_CORE_RESULTS_V151.zip"
    )
    latency_zip = (
        output
        / "RePressDB_JSS_REVISION_LATENCY_ARRAYS_V151.zip"
    )
    with zipfile.ZipFile(
        core_zip,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(
            output.rglob("*")
        ):
            if not path.is_file():
                continue
            relative = path.relative_to(
                output
            )
            if path == core_zip or path == latency_zip:
                continue
            if (
                path.suffix.lower() == ".npz"
                and path.name
                == "latencies.npz"
            ):
                continue
            if any(
                part
                in (
                    "_store",
                    "_trials",
                    "_base_stores",
                    "_candidate_storage",
                    "_calibration_checkpoints",
                    "_migration_trials",
                    "_cell_checkpoints",
                    "_checkpoints",
                    "_patched_core",
                )
                for part in relative.parts
            ):
                continue
            archive.write(
                path,
                arcname=str(relative).replace(
                    "\\",
                    "/",
                ),
            )

    latency_files = list(
        output.rglob("latencies.npz")
    )
    if latency_files:
        with zipfile.ZipFile(
            latency_zip,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for path in latency_files:
                relative = path.relative_to(
                    output
                )
                archive.write(
                    path,
                    arcname=str(
                        relative
                    ).replace("\\", "/"),
                )
    elif latency_zip.exists():
        latency_zip.unlink()

    return {
        "manifest_path": str(
            manifest_path
        ),
        "manifest_sha256": sha256_file(
            manifest_path
        ),
        "core_zip": str(
            core_zip
        ),
        "core_zip_sha256": sha256_file(
            core_zip
        ),
        "core_zip_bytes": int(
            core_zip.stat().st_size
        ),
        "latency_zip": (
            str(latency_zip)
            if latency_zip.exists()
            else None
        ),
        "latency_zip_sha256": (
            sha256_file(
                latency_zip
            )
            if latency_zip.exists()
            else None
        ),
        "latency_zip_bytes": (
            int(
                latency_zip.stat().st_size
            )
            if latency_zip.exists()
            else 0
        ),
    }


def create_protocol_and_freeze(
    paths: ProjectPaths,
    parent_hashes: Mapping[str, str],
    worker_assets: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str]:
    protocol = {
        "version": VERSION,
        "target_journal": TARGET_JOURNAL,
        "study": (
            "JSS major-revision experiment suite"
        ),
        "parent_hashes": dict(
            parent_hashes
        ),
        "script_sha256": sha256_file(
            Path(__file__).resolve()
        ),
        "j1_end_to_end": {
            "building_count": (
                R2_BUILDING_COUNT
            ),
            "selection_rule": (
                "deterministic round-robin across frozen "
                "R1 meter-type and site strata, with "
                "within-stratum SHA-256 ordering"
            ),
            "trace_families": list(
                TRACE_FAMILIES
            ),
            "chunk_sizes": list(
                CHUNK_SIZES
            ),
            "policies": list(
                CORE_POLICIES
            ),
            "repeats": (
                R2_REPEATS
            ),
            "policy_order": (
                "three cyclic Latin-square orders, each policy occupying each order position once "
                "within every trace-size stratum"
            ),
            "fresh_process_per_policy_run": True,
            "fresh_store_copy_per_policy_run": True,
            "truth_check_every_query": True,
            "fresh_prefix_queries": (
                R2_FRESH_PREFIX_QUERIES
            ),
            "memory_pressure_subset": {
                "trace": (
                    R2_MEMORY_PRESSURE_TRACE
                ),
                "chunk_size": (
                    R2_MEMORY_PRESSURE_SIZE
                ),
                "bytes": (
                    R2_MEMORY_PRESSURE_BYTES
                ),
                "repeats": (
                    R2_REPEATS
                ),
            },
        },
        "j2_multiseed": {
            "trace_seeds": list(
                MULTISEED_TRACE_SEEDS
            ),
            "events_per_building": (
                MULTISEED_EVENTS_PER_BUILDING
            ),
            "building_count": 32,
            "trace_families": list(
                TRACE_FAMILIES
            ),
            "chunk_sizes": list(
                CHUNK_SIZES
            ),
            "policies": [
                "initial_best",
                "best_static",
                "afhc_d_frozen",
                "shape_ewma_be_frozen",
                "repressdb_frozen",
                "page_hinkley_be",
                "hindsight_oracle",
            ],
            "primary_statistical_unit": (
                "independently frozen trace realization"
            ),
            "ph_be_development": {
                "dataset": (
                    "PEMS04 only"
                ),
                "deltas": list(
                    PH_DELTAS
                ),
                "thresholds": list(
                    PH_THRESHOLDS
                ),
                "window_caps": list(
                    PH_WINDOW_CAPS
                ),
                "deadbands": list(
                    PH_DEADBANDS
                ),
                "horizons": list(
                    PH_HORIZONS
                ),
                "safety_limit": (
                    PH_SAFETY_LIMIT
                ),
                "tie_tolerance": (
                    PH_TIE_TOLERANCE
                ),
            },
            "ablation_variants": [
                dataclasses.asdict(
                    variant
                )
                for variant in REPRESS_VARIANTS
            ],
        },
        "j3_crash_fuzz": {
            "total_trials": (
                CRASH_TOTAL_TRIALS
            ),
            "random_process_termination_trials": (
                CRASH_RANDOM_KILL_TRIALS
            ),
            "payload_truncation_trials": (
                CRASH_PAYLOAD_TRUNCATION_TRIALS
            ),
            "metadata_corruption_trials": (
                CRASH_METADATA_CORRUPTION_TRIALS
            ),
            "orphan_file_trials": (
                CRASH_ORPHAN_TRIALS
            ),
            "truth_queries_per_trial": (
                CRASH_TRUTH_QUERIES_PER_TRIAL
            ),
            "claim_boundary": (
                "process-termination and corruption "
                "fuzz; not physical power-loss testing"
            ),
        },
        "j4_calibration": {
            "fresh_full_calibration": True,
            "sample_fractions": list(
                CALIBRATION_SAMPLE_FRACTIONS
            ),
            "sample_seeds": list(
                CALIBRATION_SAMPLE_SEEDS
            ),
            "bounded_error_levels": list(
                COST_ERROR_LEVELS
            ),
            "error_seeds": list(
                COST_ERROR_SEEDS
            ),
            "decision_table_rule": (
                "sampled/perturbed table is used only "
                "for decisions; true measured table "
                "is used for evaluation"
            ),
        },
        "shared_budget": {
            "interval_global_queries": (
                BUDGET_INTERVAL_GLOBAL_QUERIES
            ),
            "bytes_per_interval": (
                "maximum measured directed rewrite "
                "size in each trace-size plan"
            ),
        },
        "worker_assets": {
            "selected_buildings_sha256": (
                worker_assets[
                    "selected_buildings_sha256"
                ]
            ),
            "values_sha256": {
                size: sha256_file(
                    Path(path)
                )
                for size, path in worker_assets[
                    "values_paths"
                ].items()
            },
            "plan_sha256": {
                key: sha256_file(
                    Path(path)
                )
                for key, path in worker_assets[
                    "plan_paths"
                ].items()
            },
        },
    }
    protocol_path = (
        paths.output
        / "jss_revision_protocol.json"
    )
    write_json_atomic(
        protocol_path,
        protocol,
    )
    freeze_material = {
        "protocol_sha256": sha256_file(
            protocol_path
        ),
        "script_sha256": (
            protocol[
                "script_sha256"
            ]
        ),
        "parent_hashes": dict(
            parent_hashes
        ),
        "worker_assets": (
            protocol[
                "worker_assets"
            ]
        ),
    }
    freeze_material_path = (
        paths.output
        / "jss_revision_freeze_material.json"
    )
    write_json_atomic(
        freeze_material_path,
        freeze_material,
    )
    freeze_hash = sha256_file(
        freeze_material_path
    )
    (
        paths.output
        / "jss_revision_experiment_freeze_sha256.txt"
    ).write_text(
        freeze_hash + "\n",
        encoding="utf-8",
    )
    return protocol, freeze_hash


def build_acceptance_checks(
    paths: ProjectPaths,
    r2_summary: pd.DataFrame,
    r2_memory_summary: pd.DataFrame,
    ph_config: Mapping[str, Any],
    multiseed_global: pd.DataFrame,
    ablation_global: pd.DataFrame,
    crash_records: pd.DataFrame,
    calibration_overhead: Mapping[str, Any],
    sampled_frame: pd.DataFrame,
    error_frame: pd.DataFrame,
) -> Dict[str, bool]:
    expected_r2 = (
        len(TRACE_FAMILIES)
        * len(CHUNK_SIZES)
        * R2_REPEATS
        * len(CORE_POLICIES)
    )
    expected_memory = (
        R2_REPEATS
        * len(CORE_POLICIES)
    )
    expected_multiseed = (
        len(MULTISEED_TRACE_SEEDS)
        * len(TRACE_FAMILIES)
        * len(CHUNK_SIZES)
        * 7
    )
    expected_ablation = (
        len(TRACE_FAMILIES)
        * len(CHUNK_SIZES)
        * len(REPRESS_VARIANTS)
    )
    expected_sampled = (
        len(CALIBRATION_SAMPLE_FRACTIONS)
        * len(CALIBRATION_SAMPLE_SEEDS)
        * len(CALIBRATION_ROBUSTNESS_TRACES)
        * len(CHUNK_SIZES)
        * 2
    )
    expected_error = (
        len(COST_ERROR_LEVELS)
        * len(COST_ERROR_SEEDS)
        * len(CALIBRATION_ROBUSTNESS_TRACES)
        * len(CHUNK_SIZES)
        * 2
    )
    checks = {
        "JSS_1_r2_all_54_primary_runs_complete": bool(
            len(r2_summary)
            == expected_r2
        ),
        "JSS_2_r2_all_queries_truth_checked_and_no_budget_violation": bool(
            (
                r2_summary[
                    "all_queries_truth_checked"
                ]
                == True
            ).all()
            and (
                r2_summary[
                    "budget_violation_count"
                ]
                == 0
            ).all()
        ),
        "JSS_3_r2_memory_pressure_9_runs_complete": bool(
            len(r2_memory_summary)
            == expected_memory
            and (
                r2_memory_summary[
                    "all_queries_truth_checked"
                ]
                == True
            ).all()
        ),
        "JSS_4_ph_be_frozen_only_on_pems04": bool(
            ph_config.get(
                "selection_dataset"
            )
            == "PEMS04 only"
            and ph_config.get(
                "frozen_for_multiseed"
            )
            is True
        ),
        "JSS_5_multiseed_all_expected_policy_runs_complete": bool(
            len(multiseed_global)
            == expected_multiseed
            and (
                multiseed_global[
                    "budget_violation_count"
                ]
                == 0
            ).all()
        ),
        "JSS_6_unified_ablation_complete": bool(
            len(ablation_global)
            == expected_ablation
            and set(
                ablation_global[
                    "policy"
                ].unique()
            )
            == {
                variant.name
                for variant in REPRESS_VARIANTS
            }
        ),
        "JSS_7_all_500_crash_fuzz_trials_pass": bool(
            len(crash_records)
            == CRASH_TOTAL_TRIALS
            and (
                crash_records["pass"] == 1
            ).all()
        ),
        "JSS_8_fresh_full_calibration_measured": bool(
            int(
                calibration_overhead[
                    "n_chunks"
                ]
            )
            == 64
            and float(
                calibration_overhead[
                    "wall_elapsed_ns"
                ]
            )
            > 0
            and int(
                calibration_overhead[
                    "candidate_storage_bytes"
                ]
            )
            > 0
        ),
        "JSS_9_sampled_and_error_robustness_complete": bool(
            len(sampled_frame)
            == expected_sampled
            and len(error_frame)
            == expected_error
            and np.isfinite(
                sampled_frame[
                    "total_ratio_to_full_profile"
                ]
            ).all()
            and np.isfinite(
                error_frame[
                    "total_ratio_to_full_profile"
                ]
            ).all()
        ),
        "JSS_10_all_core_result_files_exist": bool(
            all(
                path.exists()
                for path in (
                    paths.output
                    / "j1_end_to_end"
                    / "r2_end_to_end_summary.csv",
                    paths.output
                    / "j2_multiseed"
                    / "multiseed_global_results.csv",
                    paths.output
                    / "j2_multiseed"
                    / "unified_ablation_results.csv",
                    paths.output
                    / "j3_crash_fuzz"
                    / "crash_fuzz_records.csv",
                    paths.output
                    / "j4_calibration"
                    / "fresh_full_calibration"
                    / "calibration_overhead_summary.json",
                    paths.output
                    / "j4_calibration"
                    / "robustness"
                    / "cost_error_results.csv",
                )
            )
        ),
    }
    checks["all_ten"] = bool(
        all(checks.values())
    )
    return checks


def run_master() -> int:
    project = Path(
        __file__
    ).resolve().parent
    paths = resolve_project_paths(
        project
    )
    modules = load_all_modules(
        paths,
        "master",
    )
    parent_hashes = verify_parent_freezes(
        paths,
        modules,
    )
    environment_report = collect_environment_report(
        paths
    )

    print("=" * 118)
    print(
        "RePressDB JSS Major-Revision "
        "Experiment Suite v1.5.1"
    )
    print(
        f"Output: {paths.output.resolve()}"
    )
    print("=" * 118)

    context = load_r1_context(
        paths,
        modules,
    )
    worker_assets = (
        save_compact_worker_assets(
            paths,
            modules,
            context,
        )
    )
    protocol, freeze_hash = (
        create_protocol_and_freeze(
            paths,
            parent_hashes,
            worker_assets,
        )
    )

    # J1.
    print("\n[J1] Preparing base stores")
    base_stores = create_r2_base_stores(
        paths,
        modules,
        context,
        worker_assets,
        freeze_hash,
    )
    primary_cells = build_r2_cells(
        paths,
        worker_assets,
        base_stores,
        freeze_hash,
        memory_pressure=False,
    )
    run_r2_cells(
        paths,
        primary_cells,
    )
    (
        r2_summary,
        r2_chunk,
        r2_migrations,
    ) = collect_r2_results(
        paths,
        primary_cells,
        "primary",
    )

    memory_cells = build_r2_cells(
        paths,
        worker_assets,
        base_stores,
        freeze_hash,
        memory_pressure=True,
    )
    run_r2_cells(
        paths,
        memory_cells,
    )
    (
        r2_memory_summary,
        r2_memory_chunk,
        r2_memory_migrations,
    ) = collect_r2_results(
        paths,
        memory_cells,
        "memory_pressure",
    )

    j1_root = (
        paths.output
        / "j1_end_to_end"
    )
    combined_summary = pd.concat(
        [
            r2_summary,
            r2_memory_summary,
        ],
        ignore_index=True,
    )
    combined_chunk = pd.concat(
        [
            r2_chunk,
            r2_memory_chunk,
        ],
        ignore_index=True,
    )
    migration_frames = [
        frame
        for frame in (
            r2_migrations,
            r2_memory_migrations,
        )
        if not frame.empty
    ]
    combined_migrations = (
        pd.concat(
            migration_frames,
            ignore_index=True,
        )
        if migration_frames
        else pd.DataFrame()
    )
    r2_statistics = paired_r2_statistics(
        combined_summary
    )
    order_effects = r2_order_effects(
        combined_summary
    )
    r2_building_hierarchical = hierarchical_paired_ratio_statistics(
        frame=combined_chunk,
        cluster_column="repeat",
        unit_column="building_id",
        value_column="service_total_ns",
        group_columns=("result_group", "trace", "chunk_size"),
        comparisons=(
            ("repressdb_frozen", "shape_ewma_be_frozen"),
            ("repressdb_frozen", "initial_best"),
        ),
        seed_token="r2-building-hierarchical",
    )
    write_csv_atomic(
        j1_root
        / "r2_end_to_end_summary.csv",
        combined_summary,
    )
    write_csv_atomic(
        j1_root
        / "r2_end_to_end_chunk_results.csv",
        combined_chunk,
    )
    write_csv_atomic(
        j1_root
        / "r2_end_to_end_migrations.csv",
        combined_migrations,
    )
    write_csv_atomic(
        j1_root
        / "r2_paired_statistics.csv",
        r2_statistics,
    )
    write_csv_atomic(
        j1_root
        / "r2_order_effects.csv",
        order_effects,
    )
    write_csv_atomic(
        j1_root
        / "r2_building_hierarchical_statistics.csv",
        r2_building_hierarchical,
    )

    # J2.
    print("\n[J2] Developing Page-Hinkley-BE on PEMS04")
    ph_config = develop_page_hinkley_baseline(
        paths,
        modules,
        freeze_hash,
    )
    print(
        "[J2] Frozen PH-BE:",
        json.dumps(
            ph_config,
            ensure_ascii=False,
        ),
    )
    plans, plan_manifest = (
        build_multiseed_plans(
            paths,
            modules,
            context,
        )
    )
    (
        multiseed_global,
        multiseed_chunk,
        multiseed_migrations,
        ablation_global,
    ) = run_multiseed_and_ablation(
        paths,
        modules,
        context,
        plans,
        ph_config,
        freeze_hash,
    )
    multiseed_stats = (
        multiseed_paired_statistics(
            multiseed_global
        )
    )
    multiseed_building_hierarchical = hierarchical_paired_ratio_statistics(
        frame=multiseed_chunk,
        cluster_column="trace_seed",
        unit_column="building_id",
        value_column="total_time_ns",
        group_columns=("trace", "chunk_size"),
        comparisons=(
            ("repressdb_frozen", "shape_ewma_be_frozen"),
            ("repressdb_frozen", "page_hinkley_be"),
            ("repressdb_frozen", "initial_best"),
        ),
        seed_token="multiseed-building-hierarchical",
    )
    ablation_stats = (
        ablation_statistics(
            ablation_global
        )
    )
    j2_root = (
        paths.output
        / "j2_multiseed"
    )
    write_csv_atomic(
        j2_root
        / "multiseed_global_results.csv",
        multiseed_global,
    )
    write_csv_atomic(
        j2_root
        / "multiseed_chunk_results.csv",
        multiseed_chunk,
    )
    write_csv_atomic(
        j2_root
        / "multiseed_migration_records.csv",
        multiseed_migrations,
    )
    write_csv_atomic(
        j2_root
        / "multiseed_paired_statistics.csv",
        multiseed_stats,
    )
    write_csv_atomic(
        j2_root
        / "multiseed_building_hierarchical_statistics.csv",
        multiseed_building_hierarchical,
    )
    write_csv_atomic(
        j2_root
        / "unified_ablation_results.csv",
        ablation_global,
    )
    write_csv_atomic(
        j2_root
        / "unified_ablation_summary.csv",
        ablation_stats,
    )

    # J3.
    print("\n[J3] Running process-termination and corruption fuzz")
    crash_assets = prepare_crash_assets(
        paths,
        modules,
        freeze_hash,
    )
    crash_records, crash_summary = (
        run_crash_fuzz(
            paths,
            modules,
            crash_assets,
            freeze_hash,
        )
    )

    # J4.
    print("\n[J4] Measuring fresh calibration overhead")
    calibration_overhead = (
        run_fresh_calibration_overhead(
            paths,
            modules,
            context,
            freeze_hash,
        )
    )
    print("\n[J4] Running sampled/error sensitivity")
    (
        sampled_frame,
        error_frame,
        calibration_robustness_summary,
    ) = run_calibration_robustness(
        paths,
        modules,
        context,
        freeze_hash,
    )

    checks = build_acceptance_checks(
        paths,
        r2_summary,
        r2_memory_summary,
        ph_config,
        multiseed_global,
        ablation_global,
        crash_records,
        calibration_overhead,
        sampled_frame,
        error_frame,
    )
    assessment = {
        "version": VERSION,
        "target_journal": (
            TARGET_JOURNAL
        ),
        "status": (
            "JSS revision experiments complete"
            if checks["all_ten"]
            else (
                "JSS revision experiments completed "
                "with failed checks"
            )
        ),
        "experiment_freeze_sha256": (
            freeze_hash
        ),
        "parent_hashes": (
            parent_hashes
        ),
        "environment_report_sha256": sha256_file(
            paths.output / "environment_report.json"
        ),
        "acceptance_checks": checks,
        "j1_primary_runs": int(
            len(r2_summary)
        ),
        "j1_memory_pressure_runs": int(
            len(r2_memory_summary)
        ),
        "j2_multiseed_runs": int(
            len(multiseed_global)
        ),
        "j2_ablation_runs": int(
            len(ablation_global)
        ),
        "j3_crash_trials": int(
            len(crash_records)
        ),
        "j4_calibration_overhead": (
            calibration_overhead
        ),
        "ph_be_frozen_config": (
            ph_config
        ),
        "claim_boundaries": {
            "fresh_prefix": (
                "fresh process and fresh copied "
                "store; not a guaranteed OS page-"
                "cache flush"
            ),
            "memory_pressure": (
                "controlled 1-GiB competing-memory "
                "allocation; not a complete memory-"
                "constrained deployment model"
            ),
            "crash_fuzz": (
                "abrupt process termination and "
                "file corruption; not physical "
                "power-loss testing"
            ),
            "multiseed_replay": (
                "measured-cost replay; actual "
                "end-to-end evidence is reported "
                "separately in J1"
            ),
            "logical_scale": (
                "B3 remains catalog/routing scale, "
                "not independent physical 10-GiB data"
            ),
        },
        "next_gate": (
            "Upload the core result ZIP and revise "
            "the JSS manuscript."
            if checks["all_ten"]
            else (
                "Inspect failed checks before "
                "manuscript revision."
            )
        ),
    }
    assessment_path = (
        paths.output
        / "jss_revision_assessment.json"
    )
    write_json_atomic(
        assessment_path,
        assessment,
    )
    write_json_atomic(
        paths.project
        / "JSS_REVISION_EXPERIMENT_ASSESSMENT_V151.json",
        assessment,
    )

    archive_info = (
        create_revision_result_archives(
            paths
        )
    )
    assessment[
        "result_archives"
    ] = archive_info
    write_json_atomic(
        assessment_path,
        assessment,
    )
    write_json_atomic(
        paths.project
        / "JSS_REVISION_EXPERIMENT_ASSESSMENT_V151.json",
        assessment,
    )

    print("=" * 118)
    print(
        "RePressDB JSS revision suite finished"
    )
    print(
        json.dumps(
            checks,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        "Core upload package:",
        archive_info["core_zip"],
    )
    if archive_info["latency_zip"]:
        print(
            "Latency archive:",
            archive_info[
                "latency_zip"
            ],
        )
    print(
        "Assessment:",
        assessment_path,
    )
    print("=" * 118)
    return 0 if checks["all_ten"] else 2


def parse_internal_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        add_help=True
    )
    parser.add_argument(
        "--internal-worker",
        choices=(
            "r2",
            "crash-trace",
            "crash-corrupt",
        ),
        default=None,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_internal_arguments()
    if arguments.internal_worker is None:
        return run_master()
    if arguments.config is None:
        raise ValueError(
            "Internal worker requires --config."
        )
    config_path = Path(
        arguments.config
    )
    if arguments.internal_worker == "r2":
        return run_r2_worker(
            config_path
        )
    if arguments.internal_worker == "crash-trace":
        return run_crash_trace_worker(
            config_path
        )
    if arguments.internal_worker == "crash-corrupt":
        return run_crash_corruption_worker(
            config_path
        )
    raise ValueError(
        arguments.internal_worker
    )


if __name__ == "__main__":
    raise SystemExit(main())
