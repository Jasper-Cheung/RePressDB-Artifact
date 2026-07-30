#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
RePressDB J1-R Active-View Cache Confirmation v1.6.2
====================================================

PURPOSE
-------
The completed JSS v1.5.1 experiment suite showed that the frozen RePressDB
controller remained effective in measured-cost replay, but the first complete
single-active-store run was slower because the query path repeatedly:

    1. queried SQLite for the same active catalog row;
    2. reconstructed StoredChunk;
    3. reread and reparsed block metadata.

This program changes only that implementation path.

It introduces a version-keyed active-view cache containing:

    catalog row
    active version
    StoredChunk
    parsed metadata
    representation object
    payload handle
    metadata handle

The cache is invalidated after:

    migration catalog commit and cleanup
    recovery
    rollback through recovery
    store close

No controller parameter, query trace, selected building, write budget, codec,
or migration rule is changed.

EXPERIMENT
----------
Primary confirmation:

    8 already frozen BDG2 buildings
    3 frozen R1 trace families
    chunk sizes 8192 and 16384
    initial-best, frozen Shape-EWMA-BE, frozen RePressDB
    3 balanced cyclic policy orders
    54 end-to-end policy runs

Each run uses:

    a fresh Python process
    an independent copy of the same base store
    a deterministic one-query-per-chunk active-view prewarm
    every measured query checked against original float32 truth
    real catalog, payload, metadata, migration, fsync, validation, and cleanup

The prewarm cost is reported separately. The measured service interval begins
only after every active view is loaded once. Cache reload after a committed
migration remains inside the measured interval.

Optional controlled memory-pressure confirmation:

    noisy-burst
    chunk size 16384
    1 GiB competing allocation
    9 policy runs

It is disabled in v1.6.1 because this 16-GiB host could not establish sufficient uncontended headroom. The earlier diagnostic remains archived.

CALIBRATION
-----------
Decision and migration tables come from:

    results_jss_major_revision_v151/
    j4_calibration/fresh_full_calibration/

The older R1 calibration table is not used for J1-R decisions.

ENVIRONMENT GATE
----------------
Primary timing does not begin until stable 2.5 GiB of physical-memory headroom is
available. The program waits for up to 30 minutes and then stops with a clear
message. This prevents a repeat of the v1.5.1 run that started with only about
0.4 GiB available.

RUN
---
Place this file in the RePressDB project root and run it directly in PyCharm.

Required parent result:

    results_jss_major_revision_v151/
    jss_revision_assessment.json

Required parent script:

    RUN_THIS_ONLY_REPRESSDB_JSS_MAJOR_REVISION_EXPERIMENTS_V1_5_1.py

Output:

    results_jss_j1r_active_view_cache_v160/

RESUME
------
Every policy cell is checkpointed. Rerun this same file after interruption.
Do not delete the output directory unless restarting J1-R from the beginning.

CLAIM BOUNDARIES
----------------
The active-view prewarm matches the frozen warm_slicedmeta execution boundary.
It is not a cold-cache experiment.

The optional memory-pressure subset is a controlled competing allocation, not
a complete low-memory deployment model.

The script does not require RePressDB to outperform a baseline to pass its
integrity checks. Negative results remain valid results.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import dataclasses
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import psutil
except Exception:
    psutil = None


VERSION = "1.6.2"
TARGET_JOURNAL = "Journal of Systems and Software"

PARENT_SCRIPT_NAME = (
    "RUN_THIS_ONLY_REPRESSDB_JSS_MAJOR_REVISION_EXPERIMENTS_V1_5_1.py"
)
PARENT_RESULT_DIR_NAME = "results_jss_major_revision_v151"
OUTPUT_DIR_NAME = "results_jss_j1r_active_view_cache_v162"

TRACE_FAMILIES = (
    "R1_A_gradual_periodic",
    "R1_B_mean_reverting",
    "R1_C_noisy_bursts",
)
CHUNK_SIZES = (8192, 16384)
POLICIES = (
    "initial_best",
    "shape_ewma_be_frozen",
    "repressdb_frozen",
)
POLICY_ORDERS = (
    POLICIES,
    (
        "shape_ewma_be_frozen",
        "repressdb_frozen",
        "initial_best",
    ),
    (
        "repressdb_frozen",
        "initial_best",
        "shape_ewma_be_frozen",
    ),
)
REPEATS = 3
BUILDING_COUNT = 8

FRESH_PREFIX_QUERIES = 5000
BOOTSTRAP_REPLICATES = 20_000

PRIMARY_START_MIN_AVAILABLE_MEMORY_BYTES = int(2.5 * 1024**3)
PRIMARY_CELL_MIN_AVAILABLE_MEMORY_BYTES = int(1.5 * 1024**3)
MEMORY_PRESSURE_MIN_AVAILABLE_BYTES = 6 * 1024**3
MEMORY_PRESSURE_BYTES = 1 * 1024**3
PRIMARY_MEMORY_WAIT_TIMEOUT_SECONDS = 10 * 60
CELL_MEMORY_WAIT_TIMEOUT_SECONDS = 5 * 60
MEMORY_WAIT_INTERVAL_SECONDS = 5
MEMORY_GATE_CONSECUTIVE_SAMPLES = 3
RUN_OPTIONAL_MEMORY_PRESSURE = False

MEMORY_PRESSURE_TRACE = "R1_C_noisy_bursts"
MEMORY_PRESSURE_SIZE = 16384

BUDGET_INTERVAL_GLOBAL_QUERIES = 1000


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
    payload = json.dumps(
        json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_seed(*parts: Any) -> int:
    token = "|".join(str(part) for part in parts)
    return int.from_bytes(
        hashlib.sha256(token.encode("utf-8")).digest()[:8],
        "little",
        signed=False,
    )


def load_module(path: Path, module_name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    specification = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    if specification is None or specification.loader is None:
        raise ImportError(path)
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


def bootstrap_geometric_mean_interval(
    values: Sequence[float],
    seed_parts: Sequence[Any],
) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array) & (array > 0)]
    if array.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(
        stable_seed("j1r-bootstrap", *seed_parts)
    )
    indices = rng.integers(
        0,
        array.size,
        size=(BOOTSTRAP_REPLICATES, array.size),
    )
    estimates = np.exp(
        np.mean(np.log(array[indices]), axis=1)
    )
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def checkpoint_complete(path: Path, freeze_hash: str) -> bool:
    if not path.exists():
        return False
    try:
        marker = read_json(path)
    except Exception:
        return False
    return bool(
        marker.get("complete", False)
        and marker.get("freeze_sha256") == freeze_hash
    )


def sqlite_checkpoint_database(database_path: Path) -> None:
    if not database_path.exists():
        return
    connection = sqlite3.connect(str(database_path))
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.commit()
    finally:
        connection.close()


def run_subprocess_checked(
    arguments: Sequence[str],
    log_path: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(arguments),
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parent),
            env=os.environ.copy(),
        )
        return int(process.wait())


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            with contextlib.suppress(FileNotFoundError):
                total += int(child.stat().st_size)
    return total


def available_memory_bytes() -> int:
    if psutil is not None:
        return int(psutil.virtual_memory().available)

    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(
            ctypes.byref(status)
        ):
            return int(status.ullAvailPhys)

    # Linux/macOS fallback through sysconf.
    if hasattr(os, "sysconf"):
        with contextlib.suppress(Exception):
            pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return pages * page_size
    return 0


def wait_for_available_memory(
    minimum_bytes: int,
    timeout_seconds: int,
    stage_name: str,
) -> Dict[str, Any]:
    """
    Require stable headroom instead of accepting a one-sample transient spike.
    Three consecutive samples are required.
    """
    start = time.monotonic()
    observations: List[Dict[str, Any]] = []
    consecutive = 0
    maximum_available = 0
    while True:
        available = available_memory_bytes()
        maximum_available = max(maximum_available, available)
        elapsed = time.monotonic() - start
        if available >= minimum_bytes:
            consecutive += 1
        else:
            consecutive = 0
        observations.append(
            {
                "elapsed_seconds": float(elapsed),
                "available_bytes": int(available),
                "consecutive_passing_samples": int(consecutive),
            }
        )
        print(
            f"[memory gate] {stage_name}: "
            f"available={available / 1024**3:.2f} GiB, "
            f"required={minimum_bytes / 1024**3:.2f} GiB, "
            f"stable={consecutive}/{MEMORY_GATE_CONSECUTIVE_SAMPLES}"
        )
        if consecutive >= MEMORY_GATE_CONSECUTIVE_SAMPLES:
            recent = observations[-MEMORY_GATE_CONSECUTIVE_SAMPLES:]
            return {
                "passed": True,
                "stage": stage_name,
                "minimum_bytes": int(minimum_bytes),
                "required_consecutive_samples": int(
                    MEMORY_GATE_CONSECUTIVE_SAMPLES
                ),
                "available_bytes_at_pass": int(available),
                "minimum_recent_passing_bytes": int(
                    min(row["available_bytes"] for row in recent)
                ),
                "maximum_observed_bytes": int(maximum_available),
                "waited_seconds": float(elapsed),
                "observations": observations,
            }
        if elapsed >= timeout_seconds:
            return {
                "passed": False,
                "stage": stage_name,
                "minimum_bytes": int(minimum_bytes),
                "required_consecutive_samples": int(
                    MEMORY_GATE_CONSECUTIVE_SAMPLES
                ),
                "available_bytes_at_timeout": int(available),
                "maximum_observed_bytes": int(maximum_available),
                "waited_seconds": float(elapsed),
                "observations": observations,
            }
        time.sleep(MEMORY_WAIT_INTERVAL_SECONDS)


def touch_memory(buffer: bytearray, stride: int = 4096) -> int:
    checksum = 0
    for position in range(0, len(buffer), stride):
        buffer[position] = (buffer[position] + 1) % 251
        checksum += int(buffer[position])
    return checksum


@dataclasses.dataclass
class J1RPaths:
    project: Path
    output: Path
    parent_script: Path
    parent_results: Path
    fresh_calibration: Path
    worker_assets: Path


def resolve_paths(project: Path) -> J1RPaths:
    parent_script = project / PARENT_SCRIPT_NAME
    parent_results = project / PARENT_RESULT_DIR_NAME
    fresh_calibration = (
        parent_results
        / "j4_calibration"
        / "fresh_full_calibration"
    )
    worker_assets = parent_results / "_worker_assets"
    required = [
        parent_script,
        parent_results / "jss_revision_assessment.json",
        parent_results / "jss_revision_experiment_freeze_sha256.txt",
        fresh_calibration / "calibration_records.csv",
        fresh_calibration / "codec_records.csv",
        fresh_calibration / "measured_migration_costs.csv",
        worker_assets / "r2_selected_buildings.csv",
        worker_assets / "r2_values_size_8192.npz",
        worker_assets / "r2_values_size_16384.npz",
    ]
    for trace in TRACE_FAMILIES:
        for size in CHUNK_SIZES:
            required.append(
                worker_assets / f"r2_plan_{trace}_size_{size}.npz"
            )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    output = project / OUTPUT_DIR_NAME
    output.mkdir(parents=True, exist_ok=True)
    return J1RPaths(
        project=project,
        output=output,
        parent_script=parent_script,
        parent_results=parent_results,
        fresh_calibration=fresh_calibration,
        worker_assets=worker_assets,
    )


def verify_parent(paths: J1RPaths) -> Dict[str, Any]:
    assessment = read_json(
        paths.parent_results / "jss_revision_assessment.json"
    )
    if not bool(
        assessment.get("acceptance_checks", {}).get("all_ten", False)
    ):
        raise RuntimeError(
            "The v1.5.1 parent experiment is not fully accepted."
        )
    parent_freeze = (
        paths.parent_results
        / "jss_revision_experiment_freeze_sha256.txt"
    ).read_text(encoding="utf-8").strip()

    calibration_summary = read_json(
        paths.fresh_calibration
        / "calibration_overhead_summary.json"
    )
    return {
        "parent_experiment_freeze_sha256": parent_freeze,
        "parent_assessment_sha256": sha256_file(
            paths.parent_results / "jss_revision_assessment.json"
        ),
        "parent_script_sha256": sha256_file(paths.parent_script),
        "fresh_calibration_records_sha256": sha256_file(
            paths.fresh_calibration / "calibration_records.csv"
        ),
        "fresh_codec_records_sha256": sha256_file(
            paths.fresh_calibration / "codec_records.csv"
        ),
        "fresh_migration_records_sha256": sha256_file(
            paths.fresh_calibration / "measured_migration_costs.csv"
        ),
        "fresh_calibration_summary": calibration_summary,
    }


# ---------------------------------------------------------------------------
# Module loading and frozen data access
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Loaded:
    parent: Any
    paths: Any
    modules: Any


def load_parent_modules(paths: J1RPaths, tag: str) -> Loaded:
    parent = load_module(
        paths.parent_script,
        f"repressdb_jss_v151_{tag}_{os.getpid()}",
    )
    parent_paths = parent.resolve_project_paths(paths.project)
    modules = parent.load_all_modules(
        parent_paths,
        f"j1r_{tag}_{os.getpid()}",
    )
    return Loaded(
        parent=parent,
        paths=parent_paths,
        modules=modules,
    )


def load_fresh_tables(
    paths: J1RPaths,
    loaded: Loaded,
) -> Dict[str, Any]:
    calibration_frame = pd.read_csv(
        paths.fresh_calibration / "calibration_records.csv"
    )
    codec_frame = pd.read_csv(
        paths.fresh_calibration / "codec_records.csv"
    )
    migration_frame = pd.read_csv(
        paths.fresh_calibration / "measured_migration_costs.csv"
    )
    profiles = loaded.modules.s1.load_profiles(calibration_frame)
    encode, decode = loaded.modules.s1.load_encode_decode_maps(
        codec_frame
    )
    migration_costs, migration_bytes = (
        loaded.modules.s1.parse_migration_maps(
            migration_frame
        )
    )
    return {
        "calibration_frame": calibration_frame,
        "codec_frame": codec_frame,
        "migration_frame": migration_frame,
        "profiles": profiles,
        "encode": encode,
        "decode": decode,
        "migration_costs": migration_costs,
        "migration_bytes": migration_bytes,
    }


def load_chunks_and_plan(
    paths: J1RPaths,
    loaded: Loaded,
    trace: str,
    chunk_size: int,
) -> Tuple[List[Any], Any]:
    values_path = (
        paths.worker_assets / f"r2_values_size_{chunk_size}.npz"
    )
    plan_path = (
        paths.worker_assets
        / f"r2_plan_{trace}_size_{chunk_size}.npz"
    )
    chunks, plan = loaded.parent.load_compact_chunks_and_plan(
        loaded.modules,
        values_path,
        plan_path,
        trace,
        chunk_size,
    )
    if len(chunks) != BUILDING_COUNT:
        raise AssertionError(
            f"Expected {BUILDING_COUNT} compact chunks, got {len(chunks)}"
        )
    return chunks, plan


def metadata_for_plan(
    paths: J1RPaths,
    plan: Any,
) -> Dict[str, Dict[str, str]]:
    selected = pd.read_csv(
        paths.worker_assets / "r2_selected_buildings.csv"
    )
    selected = selected.sort_values("building_rank").head(BUILDING_COUNT)
    by_building = {
        str(row.building_id): {
            "building_id": str(row.building_id),
            "meter_type": str(row.meter_type),
            "site_id": str(row.site_id),
        }
        for row in selected.itertuples(index=False)
    }
    frozen_manifest = pd.read_csv(
        paths.parent_results.parent
        / "results_r1_continuous_nonstationary_v140"
        / "r1_chunk_manifest.csv"
    )
    output: Dict[str, Dict[str, str]] = {}
    for row in frozen_manifest.itertuples(index=False):
        chunk_id = str(row.chunk_id)
        if chunk_id not in set(plan.chunk_ids):
            continue
        building_id = str(row.building_id)
        output[chunk_id] = {
            "building_id": building_id,
            "meter_type": str(row.meter_type),
            "site_id": str(row.site_id),
        }
    if set(output) != set(plan.chunk_ids):
        missing = sorted(set(plan.chunk_ids) - set(output))
        raise AssertionError(f"Missing metadata for chunks: {missing}")
    return output


# ---------------------------------------------------------------------------
# Active-view cache implementation
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ActiveView:
    chunk_id: str
    active_version: int
    row: Any
    stored: Any
    representation: Any
    payload_handle: Any
    metadata_handle: Any

    def close(self) -> None:
        for handle in (self.payload_handle, self.metadata_handle):
            if handle is not None:
                with contextlib.suppress(Exception):
                    handle.close()


class ActiveViewCachedStore:
    """
    Composition wrapper around the frozen B1 SingleActiveChunkStore.

    We deliberately use composition instead of editing B1 in place. The
    migration and recovery protocol remains the frozen B1 implementation.
    Only the read-side active view is cached.
    """

    def __init__(
        self,
        raw_store: Any,
        base: Any,
    ):
        self.raw = raw_store
        self.base = base
        self._views: Dict[str, ActiveView] = {}
        self._lock = raw_store._lock
        self._counters: Dict[str, int] = {
            "catalog_row_loads": 0,
            "stored_chunk_builds": 0,
            "metadata_parses": 0,
            "payload_handle_opens": 0,
            "metadata_handle_opens": 0,
            "active_view_cache_invalidations": 0,
            "query_cache_hits": 0,
            "query_cache_misses": 0,
            "query_calls": 0,
        }

    @property
    def root(self) -> Path:
        return self.raw.root

    @property
    def connection(self) -> sqlite3.Connection:
        return self.raw.connection

    @property
    def representations(self) -> Mapping[str, Any]:
        return self.raw.representations

    @property
    def mode(self) -> Any:
        return self.raw.mode

    def _build_view(self, chunk_id: str) -> ActiveView:
        row = self.raw._row(chunk_id)
        self._counters["catalog_row_loads"] += 1

        data_path = self.raw.root / str(row["active_data_path"])
        metadata_path = (
            self.raw.root / str(row["active_metadata_path"])
            if row["active_metadata_path"] is not None
            else None
        )
        codec = str(row["active_codec"])

        if metadata_path is not None:
            metadata = self.base.parse_block_metadata(
                metadata_path.read_bytes()
            )
            self._counters["metadata_parses"] += 1
        elif codec == "whole_zstd":
            metadata = {"level": 3}
        else:
            metadata = {}

        stored = self.base.StoredChunk(
            codec=codec,
            n=int(row["n"]),
            data_path=data_path,
            metadata_path=metadata_path,
            payload_size_bytes=int(row["active_payload_bytes"]),
            metadata_size_bytes=int(row["active_metadata_bytes"]),
            metadata=metadata,
        )
        self._counters["stored_chunk_builds"] += 1

        payload = open(data_path, "rb")
        self._counters["payload_handle_opens"] += 1
        metadata_handle = (
            open(metadata_path, "rb")
            if metadata_path is not None
            else None
        )
        if metadata_handle is not None:
            self._counters["metadata_handle_opens"] += 1

        view = ActiveView(
            chunk_id=chunk_id,
            active_version=int(row["active_version"]),
            row=row,
            stored=stored,
            representation=self.representations[codec],
            payload_handle=payload,
            metadata_handle=metadata_handle,
        )
        self._views[chunk_id] = view
        return view

    def _view(self, chunk_id: str) -> ActiveView:
        cached = self._views.get(chunk_id)
        if cached is not None:
            return cached
        return self._build_view(chunk_id)

    def invalidate_handle(self, chunk_id: str) -> None:
        with self._lock:
            view = self._views.pop(chunk_id, None)
            if view is not None:
                view.close()
                self._counters[
                    "active_view_cache_invalidations"
                ] += 1
            # B1's original handle cache is not used by this wrapper, but clear
            # it as a defensive measure.
            self.raw.invalidate_handle(chunk_id)

    def invalidate_all_handles(self) -> None:
        with self._lock:
            for chunk_id in list(self._views):
                self.invalidate_handle(chunk_id)
            self.raw.invalidate_all_handles()

    def reset_query_counters(self) -> None:
        self._counters["query_cache_hits"] = 0
        self._counters["query_cache_misses"] = 0
        self._counters["query_calls"] = 0

    def counter_snapshot(self) -> Dict[str, int]:
        return {
            key: int(value)
            for key, value in self._counters.items()
        }

    def execute_query(
        self,
        chunk_id: str,
        query: Any,
    ) -> Tuple[Any, int, Any, Any]:
        with self._lock:
            self._counters["query_calls"] += 1
            cached = self._views.get(chunk_id)
            if cached is None:
                self._counters["query_cache_misses"] += 1
                view = self._build_view(chunk_id)
            else:
                self._counters["query_cache_hits"] += 1
                view = cached

            start_ns = time.perf_counter_ns()
            result, stats = self.base.run_query_once(
                view.representation,
                view.stored,
                query,
                self.mode,
                view.payload_handle,
                view.metadata_handle,
            )
            latency_ns = time.perf_counter_ns() - start_ns
            return result, int(latency_ns), stats, view.row

    def active_stored(self, chunk_id: str) -> Any:
        with self._lock:
            return self._view(chunk_id).stored

    def previous_stored(self, chunk_id: str) -> Optional[Any]:
        return self.raw.previous_stored(chunk_id)

    def read_all(self, chunk_id: str) -> np.ndarray:
        with self._lock:
            stored = self._view(chunk_id).stored
            representation = self.representations[stored.codec]
            stats = self.base.IOStats()
            with open(stored.data_path, "rb") as payload:
                return representation.decode_all(
                    stored,
                    stored.metadata,
                    payload,
                    stats,
                )

    def integrity_check(self) -> Dict[str, Any]:
        return self.raw.integrity_check()

    def recover(self) -> Dict[str, int]:
        with self._lock:
            self.invalidate_all_handles()
            result = self.raw.recover()
            self.invalidate_all_handles()
            return result

    def close(self) -> None:
        with self._lock:
            self.invalidate_all_handles()
            self.raw.close()


class CachedStoreAdapter:
    """
    Exposes the interface expected by the frozen v1.5.1 migration profiler.

    Attribute lookup falls back to the original B1 store. Methods that affect
    the active version use the cached wrapper.
    """

    def __init__(
        self,
        cached: ActiveViewCachedStore,
    ):
        self.cached = cached

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.cached, name):
            return getattr(self.cached, name)
        return getattr(self.cached.raw, name)


class CachedInstrumentedMigration:
    def __init__(
        self,
        parent_instrumented_class: Any,
        cached_store: ActiveViewCachedStore,
        b1: Any,
    ):
        self.cached_store = cached_store
        self.adapter = CachedStoreAdapter(cached_store)
        self.inner = parent_instrumented_class(
            self.adapter,
            b1,
        )

    def migrate_profiled(
        self,
        chunk_id: str,
        target_codec: str,
    ) -> Dict[str, Any]:
        result = self.inner.migrate_profiled(
            chunk_id,
            target_codec,
        )
        # The inner protocol loads the new view for post-commit validation
        # before cleanup changes migration_state back to idle. Remove that
        # temporary view so the next query loads one stable idle view.
        self.cached_store.invalidate_handle(chunk_id)
        return result


# ---------------------------------------------------------------------------
# Protocol, base stores, and cells
# ---------------------------------------------------------------------------

def create_protocol(
    paths: J1RPaths,
    parent_info: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str]:
    protocol = {
        "version": VERSION,
        "target_journal": TARGET_JOURNAL,
        "experiment": "J1-R active-view cache confirmation with stable adaptive memory gate",
        "parent": dict(parent_info),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "implementation_change": {
            "cache_key": "(chunk_id, active_version)",
            "cached_objects": [
                "catalog row",
                "StoredChunk",
                "parsed metadata",
                "representation object",
                "payload handle",
                "metadata handle",
            ],
            "query_catalog_lookup_on_cache_hit": False,
            "query_metadata_parse_on_cache_hit": False,
            "invalidation": [
                "migration completion",
                "recovery",
                "store close",
            ],
        },
        "data": {
            "building_count": BUILDING_COUNT,
            "selection": (
                "same eight already frozen v1.5.1 J1 buildings"
            ),
            "traces": list(TRACE_FAMILIES),
            "chunk_sizes": list(CHUNK_SIZES),
            "worker_assets_sha256": {
                path.name: sha256_file(path)
                for path in sorted(paths.worker_assets.glob("*.npz"))
            },
        },
        "policies": list(POLICIES),
        "repeats": REPEATS,
        "policy_orders": [list(order) for order in POLICY_ORDERS],
        "prewarm": {
            "one_truth_checked_query_per_chunk": True,
            "included_in_measured_service_time": False,
            "reported_separately": True,
        },
        "calibration": {
            "source": str(paths.fresh_calibration),
            "calibration_records_sha256": parent_info[
                "fresh_calibration_records_sha256"
            ],
            "codec_records_sha256": parent_info[
                "fresh_codec_records_sha256"
            ],
            "migration_records_sha256": parent_info[
                "fresh_migration_records_sha256"
            ],
        },
        "memory_gate": {
            "primary_start_min_available_bytes": (
                PRIMARY_START_MIN_AVAILABLE_MEMORY_BYTES
            ),
            "primary_cell_min_available_bytes": (
                PRIMARY_CELL_MIN_AVAILABLE_MEMORY_BYTES
            ),
            "required_consecutive_samples": (
                MEMORY_GATE_CONSECUTIVE_SAMPLES
            ),
            "primary_wait_timeout_seconds": (
                PRIMARY_MEMORY_WAIT_TIMEOUT_SECONDS
            ),
            "cell_wait_timeout_seconds": (
                CELL_MEMORY_WAIT_TIMEOUT_SECONDS
            ),
            "memory_pressure_min_available_bytes": (
                MEMORY_PRESSURE_MIN_AVAILABLE_BYTES
            ),
            "memory_pressure_bytes": MEMORY_PRESSURE_BYTES,
        },
        "claim_boundaries": {
            "cache": (
                "version-keyed in-process active view; not a distributed "
                "catalog cache"
            ),
            "prewarm": (
                "matches warm_slicedmeta; not a cold-cache experiment"
            ),
            "memory_pressure": (
                "optional competing allocation; skipped when the memory "
                "gate is not met"
            ),
            "outcome": (
                "performance superiority is not an acceptance condition"
            ),
        },
    }
    protocol_path = paths.output / "j1r_protocol.json"
    write_json_atomic(protocol_path, protocol)
    freeze_material = {
        "protocol_sha256": sha256_file(protocol_path),
        "script_sha256": protocol["script_sha256"],
        "parent_experiment_freeze_sha256": parent_info[
            "parent_experiment_freeze_sha256"
        ],
        "fresh_calibration_records_sha256": parent_info[
            "fresh_calibration_records_sha256"
        ],
        "fresh_migration_records_sha256": parent_info[
            "fresh_migration_records_sha256"
        ],
    }
    freeze_material_path = paths.output / "j1r_freeze_material.json"
    write_json_atomic(freeze_material_path, freeze_material)
    freeze_hash = sha256_file(freeze_material_path)
    (
        paths.output / "j1r_experiment_freeze_sha256.txt"
    ).write_text(freeze_hash + "\n", encoding="utf-8")
    return protocol, freeze_hash


def create_base_stores(
    paths: J1RPaths,
    loaded: Loaded,
    fresh: Mapping[str, Any],
    freeze_hash: str,
) -> Dict[int, Path]:
    root = paths.output / "_base_stores"
    root.mkdir(parents=True, exist_ok=True)
    outputs: Dict[int, Path] = {}

    for size in CHUNK_SIZES:
        base_root = root / f"size_{size}"
        marker = base_root / "complete.json"
        if checkpoint_complete(marker, freeze_hash):
            outputs[size] = base_root
            continue

        shutil.rmtree(base_root, ignore_errors=True)
        base_root.mkdir(parents=True, exist_ok=True)

        chunks, _ = load_chunks_and_plan(
            paths,
            loaded,
            TRACE_FAMILIES[0],
            size,
        )
        with loaded.modules.b1.SingleActiveChunkStore(
            base_root,
            loaded.modules.base,
            create=True,
            recover_on_open=False,
        ) as store:
            for chunk in chunks:
                initial_codec, _ = loaded.modules.e1.choose_best_codec(
                    fresh["profiles"][chunk.chunk_id],
                    loaded.modules.base.WORKLOADS["scan_heavy"],
                )
                store.create_chunk(
                    chunk_id=chunk.chunk_id,
                    series_index=int(chunk.series_index),
                    values=chunk.values,
                    codec=initial_codec,
                )
            integrity = store.integrity_check()
            if not integrity["pass"]:
                raise RuntimeError(
                    f"J1-R base store integrity failed: {integrity}"
                )
            store.connection.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            )
            store.connection.commit()

        sqlite_checkpoint_database(base_root / "catalog.sqlite3")
        write_json_atomic(
            marker,
            {
                "complete": True,
                "freeze_sha256": freeze_hash,
                "chunk_size": size,
                "n_chunks": len(chunks),
                "fresh_calibration_records_sha256": sha256_file(
                    paths.fresh_calibration / "calibration_records.csv"
                ),
                "integrity": integrity,
            },
        )
        outputs[size] = base_root
    return outputs


def build_cells(
    paths: J1RPaths,
    base_stores: Mapping[int, Path],
    freeze_hash: str,
    group: str,
) -> List[Dict[str, Any]]:
    if group == "primary":
        traces = TRACE_FAMILIES
        sizes = CHUNK_SIZES
        memory_pressure = False
    elif group == "memory_pressure":
        traces = (MEMORY_PRESSURE_TRACE,)
        sizes = (MEMORY_PRESSURE_SIZE,)
        memory_pressure = True
    else:
        raise ValueError(group)

    cells: List[Dict[str, Any]] = []
    for trace in traces:
        for size in sizes:
            for repeat, order in enumerate(POLICY_ORDERS):
                for order_position, policy in enumerate(order):
                    output_dir = (
                        paths.output
                        / group
                        / trace
                        / f"size_{size}"
                        / f"repeat_{repeat:02d}"
                        / f"order_{order_position}_{policy}"
                    )
                    config = {
                        "project_dir": str(paths.project),
                        "output_dir": str(output_dir),
                        "parent_result_dir": str(paths.parent_results),
                        "fresh_calibration_dir": str(paths.fresh_calibration),
                        "worker_assets_dir": str(paths.worker_assets),
                        "base_store": str(base_stores[size]),
                        "trace": trace,
                        "chunk_size": size,
                        "repeat": repeat,
                        "policy": policy,
                        "order_position": order_position,
                        "policy_order": list(order),
                        "result_group": group,
                        "memory_pressure": memory_pressure,
                        "memory_pressure_bytes": (
                            MEMORY_PRESSURE_BYTES if memory_pressure else 0
                        ),
                        "freeze_sha256": freeze_hash,
                    }
                    config_path = output_dir / "worker_config.json"
                    write_json_atomic(config_path, config)
                    cells.append(
                        {
                            **config,
                            "config_path": str(config_path),
                        }
                    )
    return cells


# ---------------------------------------------------------------------------
# Internal end-to-end worker
# ---------------------------------------------------------------------------

def run_worker(config_path: Path) -> int:
    config = read_json(config_path)
    project = Path(str(config["project_dir"]))
    paths = resolve_paths(project)
    loaded = load_parent_modules(paths, "worker")
    fresh = load_fresh_tables(paths, loaded)

    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_path = output_dir / "complete.json"
    freeze_hash = str(config["freeze_sha256"])
    if checkpoint_complete(complete_path, freeze_hash):
        return 0

    trace = str(config["trace"])
    chunk_size = int(config["chunk_size"])
    policy = str(config["policy"])
    chunks, plan = load_chunks_and_plan(
        paths,
        loaded,
        trace,
        chunk_size,
    )
    metadata = metadata_for_plan(paths, plan)
    chunks_by_id = {
        str(chunk.chunk_id): chunk
        for chunk in chunks
    }

    query_pools = loaded.modules.a2.build_query_pools(
        loaded.modules.base,
        loaded.modules.e1,
        loaded.modules.core,
        chunks,
    )
    codecs = tuple(
        sorted(
            loaded.modules.base.representation_registry(3)
        )
    )
    codec_to_position = {
        codec: position
        for position, codec in enumerate(codecs)
    }
    query_classes = tuple(loaded.modules.core.QUERY_CLASSES)
    initial_weights = loaded.parent.initial_weights_from_base(
        loaded.modules,
        query_classes,
    )
    shape_config = read_json(
        loaded.paths.results["s1"]
        / "shape_ewma_be_frozen_config.json"
    )

    chunk_ids = tuple(plan.chunk_ids)
    profile_array = loaded.modules.s1.profile_array(
        fresh["profiles"],
        chunk_ids,
        codecs,
        query_classes,
    )
    migration_cost_array, migration_byte_array = (
        loaded.modules.s1.migration_matrix(
            fresh["migration_costs"],
            fresh["migration_bytes"],
            chunk_ids,
            codecs,
        )
    )
    budget_bytes = int(migration_byte_array.max())

    active_positions: List[int] = []
    controllers: List[Optional[Any]] = []
    for chunk_id in chunk_ids:
        initial_codec, _ = loaded.modules.e1.choose_best_codec(
            fresh["profiles"][chunk_id],
            loaded.modules.base.WORKLOADS["scan_heavy"],
        )
        initial_position = codec_to_position[initial_codec]
        active_positions.append(initial_position)
        if policy == "initial_best":
            controller = None
        elif policy == "shape_ewma_be_frozen":
            controller = loaded.parent.ShapeEWMAController(
                initial_position,
                initial_weights,
                shape_config,
            )
        elif policy == "repressdb_frozen":
            controller = loaded.parent.RepressController(
                initial_position,
                loaded.parent.RepressVariant(
                    name="full_repressdb"
                ),
            )
        else:
            raise ValueError(policy)
        controllers.append(controller)

    memory_buffer: Optional[bytearray] = None
    memory_checksum = 0
    if bool(config.get("memory_pressure", False)):
        available_before = available_memory_bytes()
        if available_before < MEMORY_PRESSURE_MIN_AVAILABLE_BYTES:
            raise RuntimeError(
                "Memory-pressure worker started without the required "
                f"{MEMORY_PRESSURE_MIN_AVAILABLE_BYTES / 1024**3:.1f} GiB "
                f"available; observed {available_before / 1024**3:.2f} GiB."
            )
        memory_buffer = bytearray(int(config["memory_pressure_bytes"]))
        memory_checksum = touch_memory(memory_buffer)

    run_store = output_dir / "_store"
    shutil.rmtree(run_store, ignore_errors=True)
    shutil.copytree(
        Path(str(config["base_store"])),
        run_store,
    )
    sqlite_checkpoint_database(run_store / "catalog.sqlite3")

    raw_store = loaded.modules.b1.SingleActiveChunkStore(
        run_store,
        loaded.modules.base,
        create=True,
        recover_on_open=True,
    )
    cached_store = ActiveViewCachedStore(
        raw_store,
        loaded.modules.base,
    )
    migration_executor = CachedInstrumentedMigration(
        loaded.parent.InstrumentedSingleActiveChunkStore,
        cached_store,
        loaded.modules.b1,
    )

    # Deterministic one-query-per-chunk prewarm.
    first_event_by_chunk: Dict[str, int] = {}
    for event_index in range(plan.n_events):
        chunk_position = int(plan.chunk_positions[event_index])
        chunk_id = str(plan.chunk_ids[chunk_position])
        first_event_by_chunk.setdefault(chunk_id, event_index)
        if len(first_event_by_chunk) == len(chunk_ids):
            break
    if len(first_event_by_chunk) != len(chunk_ids):
        raise AssertionError("Not every chunk appears in the measured plan.")

    prewarm_start = time.perf_counter_ns()
    prewarm_query_outer_ns = 0
    prewarm_codec_ns = 0
    prewarm_truth_ns = 0
    prewarm_ok = 0
    for chunk_id in chunk_ids:
        event_index = first_event_by_chunk[chunk_id]
        query_position = int(
            plan.query_class_positions[event_index]
        )
        query_class = query_classes[query_position]
        choice = int(plan.query_choice_positions[event_index])
        query = query_pools[chunk_id][query_class][choice]

        start = time.perf_counter_ns()
        actual, codec_ns, _, _ = cached_store.execute_query(
            chunk_id,
            query,
        )
        prewarm_query_outer_ns += time.perf_counter_ns() - start
        prewarm_codec_ns += int(codec_ns)

        truth_start = time.perf_counter_ns()
        expected = loaded.modules.base.execute_on_array(
            chunks_by_id[chunk_id].values,
            query,
        )
        prewarm_truth_ns += time.perf_counter_ns() - truth_start
        ok = loaded.modules.base.results_equal(actual, expected)
        prewarm_ok += int(ok)
        if not ok:
            raise AssertionError(
                f"Prewarm truth mismatch: {chunk_id}"
            )
    prewarm_elapsed_ns = time.perf_counter_ns() - prewarm_start
    prewarm_counters = cached_store.counter_snapshot()
    cached_store.reset_query_counters()
    measured_counter_start = cached_store.counter_snapshot()

    n_events = int(plan.n_events)
    query_outer_latency = np.empty(n_events, dtype=np.int64)
    query_codec_latency = np.empty(n_events, dtype=np.int64)
    controller_latency = np.zeros(n_events, dtype=np.int64)
    truth_latency = np.empty(n_events, dtype=np.int64)
    result_ok = np.zeros(n_events, dtype=np.uint8)
    logical_bytes = np.zeros(n_events, dtype=np.int64)

    per_chunk: Dict[str, Dict[str, int]] = {
        chunk_id: {
            "n_queries": 0,
            "query_outer_ns": 0,
            "query_codec_ns": 0,
            "controller_ns": 0,
            "migration_ns": 0,
            "migration_count": 0,
            "rewrite_bytes": 0,
            "truth_check_ns": 0,
        }
        for chunk_id in chunk_ids
    }
    migration_rows: List[Dict[str, Any]] = []
    stage_totals: Dict[str, int] = {}

    interval_id = 0
    budget_remaining = budget_bytes
    budget_spent = 0
    budget_denied = 0
    budget_violations = 0

    measured_start = time.perf_counter_ns()
    try:
        for event_index in range(n_events):
            new_interval = (
                event_index // BUDGET_INTERVAL_GLOBAL_QUERIES
            )
            if new_interval != interval_id:
                if budget_spent > budget_bytes:
                    budget_violations += 1
                interval_id = new_interval
                budget_remaining = budget_bytes
                budget_spent = 0

            chunk_position = int(plan.chunk_positions[event_index])
            chunk_id = str(plan.chunk_ids[chunk_position])
            query_position = int(
                plan.query_class_positions[event_index]
            )
            query_class = query_classes[query_position]
            choice = int(plan.query_choice_positions[event_index])
            query = query_pools[chunk_id][query_class][choice]

            query_start = time.perf_counter_ns()
            actual, codec_ns, stats, _ = cached_store.execute_query(
                chunk_id,
                query,
            )
            outer_ns = time.perf_counter_ns() - query_start
            query_outer_latency[event_index] = int(outer_ns)
            query_codec_latency[event_index] = int(codec_ns)
            logical_bytes[event_index] = int(
                getattr(
                    stats,
                    "total_logical_bytes",
                    getattr(stats, "logical_bytes", 0),
                )
            )

            truth_start = time.perf_counter_ns()
            expected = loaded.modules.base.execute_on_array(
                chunks_by_id[chunk_id].values,
                query,
            )
            truth_ns = time.perf_counter_ns() - truth_start
            truth_latency[event_index] = int(truth_ns)
            ok = loaded.modules.base.results_equal(actual, expected)
            result_ok[event_index] = int(ok)
            if not ok:
                raise AssertionError(
                    f"Measured truth mismatch: {policy}/{trace}/"
                    f"{chunk_size}/{event_index}/{chunk_id}"
                )

            local = per_chunk[chunk_id]
            local["n_queries"] += 1
            local["query_outer_ns"] += int(outer_ns)
            local["query_codec_ns"] += int(codec_ns)
            local["truth_check_ns"] += int(truth_ns)

            controller = controllers[chunk_position]
            proposal: Optional[int] = None
            decision: Optional[Dict[str, Any]] = None
            if controller is not None:
                controller_start = time.perf_counter_ns()
                proposal, decision = controller.observe_and_decide(
                    query_position=query_position,
                    decision_profile=profile_array[chunk_position],
                    decision_migration_cost=(
                        migration_cost_array[chunk_position]
                    ),
                    codecs=codecs,
                    global_event_index=event_index,
                    chunk_id=chunk_id,
                )
                controller_ns = (
                    time.perf_counter_ns() - controller_start
                )
                controller_latency[event_index] = int(controller_ns)
                local["controller_ns"] += int(controller_ns)

            if proposal is not None:
                current_position = int(
                    active_positions[chunk_position]
                )
                target_position = int(proposal)
                rewrite_estimate = int(
                    migration_byte_array[
                        chunk_position,
                        current_position,
                        target_position,
                    ]
                )
                if rewrite_estimate <= budget_remaining:
                    migration_result = (
                        migration_executor.migrate_profiled(
                            chunk_id,
                            codecs[target_position],
                        )
                    )
                    migration_ns = int(
                        migration_result["migration_time_ns"]
                    )
                    actual_rewrite = int(
                        migration_result["rewrite_bytes"]
                    )
                    active_positions[chunk_position] = target_position
                    controller.committed_switch(target_position)

                    budget_remaining -= actual_rewrite
                    budget_spent += actual_rewrite
                    local["migration_ns"] += migration_ns
                    local["migration_count"] += 1
                    local["rewrite_bytes"] += actual_rewrite

                    for stage, value in migration_result[
                        "stage_times_ns"
                    ].items():
                        stage_totals[stage] = (
                            stage_totals.get(stage, 0) + int(value)
                        )
                    migration_rows.append(
                        {
                            "result_group": str(config["result_group"]),
                            "trace": trace,
                            "chunk_size": chunk_size,
                            "repeat": int(config["repeat"]),
                            "policy": policy,
                            "order_position": int(
                                config["order_position"]
                            ),
                            "global_event_index": event_index,
                            "chunk_id": chunk_id,
                            "building_id": metadata[chunk_id][
                                "building_id"
                            ],
                            "source_codec": str(
                                migration_result["from_codec"]
                            ),
                            "target_codec": str(
                                migration_result["target_codec"]
                            ),
                            "migration_time_ns": migration_ns,
                            "rewrite_bytes": actual_rewrite,
                            "budget_remaining_after": budget_remaining,
                            "decision_reason": (
                                str(decision.get("reason", ""))
                                if decision is not None
                                else ""
                            ),
                            **{
                                f"stage_{stage}": int(value)
                                for stage, value in migration_result[
                                    "stage_times_ns"
                                ].items()
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
                memory_checksum += touch_memory(memory_buffer)

        if budget_spent > budget_bytes:
            budget_violations += 1
        measured_elapsed_ns = (
            time.perf_counter_ns() - measured_start
        )
        integrity = cached_store.integrity_check()
        if not integrity["pass"]:
            raise RuntimeError(
                f"Final store integrity failed: {integrity}"
            )
    finally:
        measured_counter_end = cached_store.counter_snapshot()
        with contextlib.suppress(Exception):
            cached_store.connection.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            )
            cached_store.connection.commit()
        cached_store.close()

    if int(result_ok.sum()) != n_events:
        raise AssertionError("Not every measured query passed truth checking.")

    measured_counter_delta = {
        key: int(
            measured_counter_end.get(key, 0)
            - measured_counter_start.get(key, 0)
        )
        for key in measured_counter_end
    }

    query_outer_total = int(query_outer_latency.sum())
    query_codec_total = int(query_codec_latency.sum())
    controller_total = int(controller_latency.sum())
    truth_total = int(truth_latency.sum())
    migration_total = int(
        sum(row["migration_time_ns"] for row in migration_rows)
    )
    service_total = (
        query_outer_total + controller_total + migration_total
    )
    dispatch_total = query_outer_total - query_codec_total

    prefix_n = min(FRESH_PREFIX_QUERIES, n_events)
    prefix_migration = int(
        sum(
            row["migration_time_ns"]
            for row in migration_rows
            if int(row["global_event_index"]) < prefix_n
        )
    )
    prefix_service = int(
        query_outer_latency[:prefix_n].sum()
        + controller_latency[:prefix_n].sum()
        + prefix_migration
    )
    steady_service = int(service_total - prefix_service)

    chunk_rows: List[Dict[str, Any]] = []
    for chunk_id in chunk_ids:
        local = per_chunk[chunk_id]
        chunk_rows.append(
            {
                "result_group": str(config["result_group"]),
                "trace": trace,
                "chunk_size": chunk_size,
                "repeat": int(config["repeat"]),
                "policy": policy,
                "order_position": int(config["order_position"]),
                "chunk_id": chunk_id,
                "building_id": metadata[chunk_id]["building_id"],
                "meter_type": metadata[chunk_id]["meter_type"],
                "site_id": metadata[chunk_id]["site_id"],
                **local,
                "query_catalog_dispatch_ns": int(
                    local["query_outer_ns"] - local["query_codec_ns"]
                ),
                "service_total_ns": int(
                    local["query_outer_ns"]
                    + local["controller_ns"]
                    + local["migration_ns"]
                ),
            }
        )

    latency_path = output_dir / "latencies.npz"
    temporary = latency_path.with_name(latency_path.name + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(
            handle,
            query_outer_latency_ns=query_outer_latency,
            query_codec_latency_ns=query_codec_latency,
            controller_latency_ns=controller_latency,
            truth_check_latency_ns=truth_latency,
            logical_bytes=logical_bytes,
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
        "result_group",
        "trace",
        "chunk_size",
        "repeat",
        "policy",
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
    migration_frame = (
        pd.DataFrame(migration_rows)
        if migration_rows
        else pd.DataFrame(columns=migration_columns)
    )
    write_csv_atomic(
        output_dir / "migration_results.csv",
        migration_frame,
    )

    query_calls = int(measured_counter_delta["query_calls"])
    query_hits = int(measured_counter_delta["query_cache_hits"])
    query_misses = int(measured_counter_delta["query_cache_misses"])
    query_hit_rate = (
        query_hits / query_calls if query_calls else math.nan
    )
    migration_count = len(migration_rows)

    summary = {
        "version": VERSION,
        "result_group": str(config["result_group"]),
        "trace": trace,
        "chunk_size": chunk_size,
        "repeat": int(config["repeat"]),
        "policy": policy,
        "order_position": int(config["order_position"]),
        "policy_order": list(config["policy_order"]),
        "memory_pressure": bool(config["memory_pressure"]),
        "memory_pressure_bytes": int(config["memory_pressure_bytes"]),
        "memory_touch_checksum": int(memory_checksum),
        "n_buildings": len(chunk_ids),
        "n_queries": n_events,
        "all_queries_truth_checked": True,
        "prewarm_queries": len(chunk_ids),
        "prewarm_all_truth_checked": bool(
            prewarm_ok == len(chunk_ids)
        ),
        "prewarm_elapsed_ns": int(prewarm_elapsed_ns),
        "prewarm_query_outer_ns": int(prewarm_query_outer_ns),
        "prewarm_codec_ns": int(prewarm_codec_ns),
        "prewarm_truth_ns": int(prewarm_truth_ns),
        "prewarm_counters": prewarm_counters,
        "measured_process_elapsed_ns": int(measured_elapsed_ns),
        "query_outer_time_ns": query_outer_total,
        "query_codec_time_ns": query_codec_total,
        "query_catalog_dispatch_time_ns": int(dispatch_total),
        "controller_time_ns": controller_total,
        "migration_time_ns": migration_total,
        "service_total_time_ns": int(service_total),
        "truth_check_time_ns": truth_total,
        "throughput_qps": float(
            n_events / max(service_total / 1e9, 1e-12)
        ),
        "query_p50_ns": float(
            np.quantile(query_outer_latency, 0.50)
        ),
        "query_p95_ns": float(
            np.quantile(query_outer_latency, 0.95)
        ),
        "query_p99_ns": float(
            np.quantile(query_outer_latency, 0.99)
        ),
        "query_max_ns": int(query_outer_latency.max()),
        "fresh_prefix_queries": prefix_n,
        "fresh_prefix_service_time_ns": prefix_service,
        "steady_queries": n_events - prefix_n,
        "steady_service_time_ns": steady_service,
        "fresh_prefix_throughput_qps": float(
            prefix_n / max(prefix_service / 1e9, 1e-12)
        ),
        "steady_throughput_qps": float(
            (n_events - prefix_n)
            / max(steady_service / 1e9, 1e-12)
        ),
        "migration_count": migration_count,
        "rewrite_bytes": int(
            sum(row["rewrite_bytes"] for row in migration_rows)
        ),
        "budget_bytes_per_interval": budget_bytes,
        "budget_denied_proposals": budget_denied,
        "budget_violation_count": budget_violations,
        "migration_stage_totals_ns": stage_totals,
        "measured_cache_counters": measured_counter_delta,
        "query_active_view_cache_hits": query_hits,
        "query_active_view_cache_misses": query_misses,
        "query_active_view_cache_hit_rate": float(query_hit_rate),
        "query_cache_miss_bound": migration_count,
        "query_cache_miss_bound_pass": bool(
            query_misses <= migration_count
        ),
        "store_integrity": integrity,
        "fresh_calibration_records_sha256": sha256_file(
            paths.fresh_calibration / "calibration_records.csv"
        ),
        "fresh_migration_records_sha256": sha256_file(
            paths.fresh_calibration / "measured_migration_costs.csv"
        ),
        "latency_npz_sha256": sha256_file(latency_path),
    }
    summary_path = output_dir / "summary.json"
    write_json_atomic(summary_path, summary)
    write_json_atomic(
        complete_path,
        {
            "complete": True,
            "freeze_sha256": freeze_hash,
            "summary_sha256": sha256_file(summary_path),
            "completed_at_ns": time.time_ns(),
        },
    )
    shutil.rmtree(run_store, ignore_errors=True)
    return 0


# ---------------------------------------------------------------------------
# Cell execution and result collection
# ---------------------------------------------------------------------------

def run_cells(
    cells: Sequence[Mapping[str, Any]],
    minimum_available_memory: int,
) -> None:
    total = len(cells)
    for index, cell in enumerate(cells, start=1):
        output_dir = Path(str(cell["output_dir"]))
        complete_path = output_dir / "complete.json"
        freeze_hash = str(cell["freeze_sha256"])
        if checkpoint_complete(complete_path, freeze_hash):
            print(
                f"[J1-R resume {index}/{total}] "
                f"{cell['result_group']} {cell['trace']} "
                f"size={cell['chunk_size']} repeat={cell['repeat']} "
                f"policy={cell['policy']}"
            )
            continue

        gate = wait_for_available_memory(
            minimum_available_memory,
            CELL_MEMORY_WAIT_TIMEOUT_SECONDS,
            (
                f"{cell['result_group']} cell {index}/{total}: "
                f"{cell['trace']} size={cell['chunk_size']} "
                f"policy={cell['policy']}"
            ),
        )
        write_json_atomic(
            output_dir / "memory_gate.json",
            gate,
        )
        if not gate["passed"]:
            raise RuntimeError(
                "The memory gate timed out. Close memory-intensive "
                "applications and rerun this same file."
            )

        exit_code = run_subprocess_checked(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--internal-worker",
                "--config",
                str(cell["config_path"]),
            ],
            output_dir / "worker.log",
        )
        if exit_code != 0:
            raise RuntimeError(
                f"J1-R worker failed with exit code {exit_code}. "
                f"See {output_dir / 'worker.log'}"
            )
        print(
            f"[J1-R {index}/{total}] "
            f"{cell['result_group']} {cell['trace']} "
            f"size={cell['chunk_size']} repeat={cell['repeat']} "
            f"policy={cell['policy']}"
        )


def flatten_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    excluded = {
        "policy_order",
        "prewarm_counters",
        "migration_stage_totals_ns",
        "measured_cache_counters",
        "store_integrity",
    }
    flat = {
        key: value
        for key, value in summary.items()
        if key not in excluded
    }
    flat["policy_order"] = json.dumps(
        summary.get("policy_order", []),
        ensure_ascii=False,
    )
    for prefix, nested_key in (
        ("prewarm_counter", "prewarm_counters"),
        ("migration_stage", "migration_stage_totals_ns"),
        ("cache_counter", "measured_cache_counters"),
    ):
        for key, value in summary.get(nested_key, {}).items():
            flat[f"{prefix}_{key}"] = value
    integrity = summary.get("store_integrity", {})
    flat["store_integrity_pass"] = bool(
        integrity.get("pass", False)
    )
    flat["store_temporary_file_count"] = int(
        integrity.get("temporary_file_count", -1)
    )
    flat["store_invalid_chunk_count"] = len(
        integrity.get("invalid_chunks", [])
    )
    flat["store_non_idle_chunk_count"] = len(
        integrity.get("non_idle_chunks", [])
    )
    flat["store_wrong_payload_count"] = len(
        integrity.get("chunks_without_exactly_one_payload", [])
    )
    return flat


def collect_results(
    cells: Sequence[Mapping[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, Any]] = []
    chunk_frames: List[pd.DataFrame] = []
    migration_frames: List[pd.DataFrame] = []

    for cell in cells:
        root = Path(str(cell["output_dir"]))
        summary_path = root / "summary.json"
        latency_path = root / "latencies.npz"
        summary = read_json(summary_path)
        if summary["latency_npz_sha256"] != sha256_file(latency_path):
            raise AssertionError(
                f"Latency SHA mismatch: {latency_path}"
            )
        with np.load(latency_path, allow_pickle=False) as archive:
            query_outer = np.asarray(
                archive["query_outer_latency_ns"],
                dtype=np.int64,
            )
            query_codec = np.asarray(
                archive["query_codec_latency_ns"],
                dtype=np.int64,
            )
            controller = np.asarray(
                archive["controller_latency_ns"],
                dtype=np.int64,
            )
            result_ok = np.asarray(
                archive["result_ok"],
                dtype=np.uint8,
            )
        if len(query_outer) != int(summary["n_queries"]):
            raise AssertionError(
                f"Latency length mismatch: {latency_path}"
            )
        if int(result_ok.sum()) != len(result_ok):
            raise AssertionError(
                f"Truth flags failed: {latency_path}"
            )
        if int(query_outer.sum()) != int(summary["query_outer_time_ns"]):
            raise AssertionError(
                f"Query sum mismatch: {latency_path}"
            )
        if int(query_codec.sum()) != int(summary["query_codec_time_ns"]):
            raise AssertionError(
                f"Codec sum mismatch: {latency_path}"
            )
        if int(controller.sum()) != int(summary["controller_time_ns"]):
            raise AssertionError(
                f"Controller sum mismatch: {latency_path}"
            )
        recomputed_p99 = float(np.quantile(query_outer, 0.99))
        if not math.isclose(
            recomputed_p99,
            float(summary["query_p99_ns"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise AssertionError(
                f"p99 mismatch: {latency_path}"
            )

        summary_rows.append(flatten_summary(summary))
        chunk_frames.append(
            pd.read_csv(root / "chunk_results.csv")
        )
        migration_path = root / "migration_results.csv"
        try:
            migration_frame = pd.read_csv(migration_path)
        except pd.errors.EmptyDataError:
            migration_frame = pd.DataFrame()
        if not migration_frame.empty:
            migration_frames.append(migration_frame)

    summary_frame = pd.DataFrame(summary_rows)
    chunk_frame = pd.concat(chunk_frames, ignore_index=True)
    migration_frame = (
        pd.concat(migration_frames, ignore_index=True)
        if migration_frames
        else pd.DataFrame()
    )
    return summary_frame, chunk_frame, migration_frame


def paired_statistics(
    summary_frame: pd.DataFrame,
) -> pd.DataFrame:
    metrics = (
        "service_total_time_ns",
        "throughput_qps",
        "query_catalog_dispatch_time_ns",
        "query_codec_time_ns",
        "query_p99_ns",
        "fresh_prefix_service_time_ns",
        "steady_service_time_ns",
    )
    comparisons = (
        ("repressdb_frozen", "shape_ewma_be_frozen"),
        ("repressdb_frozen", "initial_best"),
        ("shape_ewma_be_frozen", "initial_best"),
    )
    rows: List[Dict[str, Any]] = []

    for (
        result_group,
        trace,
        size,
    ), group in summary_frame.groupby(
        ["result_group", "trace", "chunk_size"]
    ):
        for metric in metrics:
            pivot = group.pivot(
                index="repeat",
                columns="policy",
                values=metric,
            ).dropna()
            for numerator, denominator in comparisons:
                if (
                    numerator not in pivot.columns
                    or denominator not in pivot.columns
                ):
                    continue
                ratio = (
                    pivot[numerator].to_numpy(dtype=np.float64)
                    / pivot[denominator].to_numpy(dtype=np.float64)
                )
                low, high = bootstrap_geometric_mean_interval(
                    ratio,
                    (
                        result_group,
                        trace,
                        size,
                        metric,
                        numerator,
                        denominator,
                    ),
                )
                rows.append(
                    {
                        "result_group": result_group,
                        "trace": trace,
                        "chunk_size": int(size),
                        "metric": metric,
                        "numerator_policy": numerator,
                        "denominator_policy": denominator,
                        "n_paired_repeats": len(ratio),
                        "geometric_mean_ratio": geometric_mean(ratio),
                        "median_ratio": float(np.median(ratio)),
                        "minimum_ratio": float(np.min(ratio)),
                        "maximum_ratio": float(np.max(ratio)),
                        "ci_low": low,
                        "ci_high": high,
                        "fraction_ratio_below_one": float(
                            np.mean(ratio < 1.0)
                        ),
                    }
                )

    # Aggregate across all 18 primary paired repeat cells.
    primary = summary_frame[
        summary_frame["result_group"] == "primary"
    ]
    for metric in metrics:
        pivot = primary.pivot(
            index=["trace", "chunk_size", "repeat"],
            columns="policy",
            values=metric,
        ).dropna()
        for numerator, denominator in comparisons:
            if (
                numerator not in pivot.columns
                or denominator not in pivot.columns
            ):
                continue
            ratio = (
                pivot[numerator].to_numpy(dtype=np.float64)
                / pivot[denominator].to_numpy(dtype=np.float64)
            )
            low, high = bootstrap_geometric_mean_interval(
                ratio,
                (
                    "primary_aggregate",
                    metric,
                    numerator,
                    denominator,
                ),
            )
            rows.append(
                {
                    "result_group": "primary_aggregate",
                    "trace": "all",
                    "chunk_size": 0,
                    "metric": metric,
                    "numerator_policy": numerator,
                    "denominator_policy": denominator,
                    "n_paired_repeats": len(ratio),
                    "geometric_mean_ratio": geometric_mean(ratio),
                    "median_ratio": float(np.median(ratio)),
                    "minimum_ratio": float(np.min(ratio)),
                    "maximum_ratio": float(np.max(ratio)),
                    "ci_low": low,
                    "ci_high": high,
                    "fraction_ratio_below_one": float(
                        np.mean(ratio < 1.0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def order_effects(summary_frame: pd.DataFrame) -> pd.DataFrame:
    primary = summary_frame[
        summary_frame["result_group"] == "primary"
    ].copy()
    primary["within_cell_normalized_time"] = (
        primary["service_total_time_ns"]
        / primary.groupby(
            ["trace", "chunk_size", "repeat"]
        )["service_total_time_ns"].transform("mean")
    )
    return (
        primary.groupby(["policy", "order_position"])
        .agg(
            n_cells=("within_cell_normalized_time", "size"),
            median_normalized_time=(
                "within_cell_normalized_time",
                "median",
            ),
            mean_normalized_time=(
                "within_cell_normalized_time",
                "mean",
            ),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Fresh-calibration model replay and legacy comparison
# ---------------------------------------------------------------------------

def run_fresh_model_replay(
    paths: J1RPaths,
    loaded: Loaded,
    fresh: Mapping[str, Any],
) -> pd.DataFrame:
    codecs = tuple(
        sorted(
            loaded.modules.base.representation_registry(3)
        )
    )
    query_classes = tuple(loaded.modules.core.QUERY_CLASSES)
    initial_weights = loaded.parent.initial_weights_from_base(
        loaded.modules,
        query_classes,
    )
    shape_config = read_json(
        loaded.paths.results["s1"]
        / "shape_ewma_be_frozen_config.json"
    )
    rows: List[Dict[str, Any]] = []

    for trace in TRACE_FAMILIES:
        for size in CHUNK_SIZES:
            _, plan = load_chunks_and_plan(
                paths,
                loaded,
                trace,
                size,
            )
            chunk_ids = tuple(plan.chunk_ids)
            profile = loaded.modules.s1.profile_array(
                fresh["profiles"],
                chunk_ids,
                codecs,
                query_classes,
            )
            initial = loaded.modules.s1.scan_best_positions(
                loaded.modules.e1,
                loaded.modules.base,
                fresh["profiles"],
                chunk_ids,
                codecs,
            )
            migration_cost, migration_bytes = (
                loaded.modules.s1.migration_matrix(
                    fresh["migration_costs"],
                    fresh["migration_bytes"],
                    chunk_ids,
                    codecs,
                )
            )
            results = {
                "initial_best": loaded.parent.static_model_result(
                    plan,
                    profile,
                    initial,
                    "initial_best",
                ),
                "shape_ewma_be_frozen": (
                    loaded.parent.simulate_controller_model(
                        policy_name="shape_ewma_be_frozen",
                        controller_factory=(
                            lambda initial_position: (
                                loaded.parent.ShapeEWMAController(
                                    initial_position,
                                    initial_weights,
                                    shape_config,
                                )
                            )
                        ),
                        plan=plan,
                        evaluation_profile=profile,
                        decision_profile=profile,
                        evaluation_migration_cost=migration_cost,
                        decision_migration_cost=migration_cost,
                        migration_bytes=migration_bytes,
                        initial_positions=initial,
                        codecs=codecs,
                        use_write_budget=True,
                    )
                ),
                "repressdb_frozen": (
                    loaded.parent.simulate_controller_model(
                        policy_name="repressdb_frozen",
                        controller_factory=(
                            lambda initial_position: (
                                loaded.parent.RepressController(
                                    initial_position,
                                    loaded.parent.RepressVariant(
                                        name="full_repressdb"
                                    ),
                                )
                            )
                        ),
                        plan=plan,
                        evaluation_profile=profile,
                        decision_profile=profile,
                        evaluation_migration_cost=migration_cost,
                        decision_migration_cost=migration_cost,
                        migration_bytes=migration_bytes,
                        initial_positions=initial,
                        codecs=codecs,
                        use_write_budget=True,
                    )
                ),
            }
            initial_total = float(
                results["initial_best"]["total_time_ns"]
            )
            for policy, result in results.items():
                rows.append(
                    {
                        "trace": trace,
                        "chunk_size": size,
                        "policy": policy,
                        "model_query_time_ns": float(
                            result["query_time_ns"]
                        ),
                        "model_migration_time_ns": float(
                            result["migration_time_ns"]
                        ),
                        "model_total_time_ns": float(
                            result["total_time_ns"]
                        ),
                        "model_ratio_to_initial": float(
                            result["total_time_ns"] / initial_total
                        ),
                        "model_migration_count": int(
                            result["migration_count"]
                        ),
                        "model_rewrite_bytes": int(
                            result["rewrite_bytes"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def model_actual_validation(
    actual_summary: pd.DataFrame,
    model_frame: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    actual = (
        actual_summary[
            actual_summary["result_group"] == "primary"
        ]
        .groupby(["trace", "chunk_size", "policy"])
        .agg(
            actual_median_service_time_ns=(
                "service_total_time_ns",
                "median",
            ),
            actual_geometric_mean_service_time_ns=(
                "service_total_time_ns",
                geometric_mean,
            ),
            actual_median_migration_count=(
                "migration_count",
                "median",
            ),
        )
        .reset_index()
    )
    merged = actual.merge(
        model_frame,
        on=["trace", "chunk_size", "policy"],
        how="inner",
        validate="one_to_one",
    )
    merged["actual_over_model_ratio"] = (
        merged["actual_median_service_time_ns"]
        / merged["model_total_time_ns"]
    )

    ratio_rows: List[Dict[str, Any]] = []
    for trace in TRACE_FAMILIES:
        for size in CHUNK_SIZES:
            local = merged[
                (merged["trace"] == trace)
                & (merged["chunk_size"] == size)
            ].set_index("policy")
            for numerator, denominator in (
                ("repressdb_frozen", "shape_ewma_be_frozen"),
                ("repressdb_frozen", "initial_best"),
                ("shape_ewma_be_frozen", "initial_best"),
            ):
                model_ratio = float(
                    local.loc[numerator, "model_total_time_ns"]
                    / local.loc[denominator, "model_total_time_ns"]
                )
                actual_ratio = float(
                    local.loc[
                        numerator,
                        "actual_median_service_time_ns",
                    ]
                    / local.loc[
                        denominator,
                        "actual_median_service_time_ns",
                    ]
                )
                ratio_rows.append(
                    {
                        "trace": trace,
                        "chunk_size": size,
                        "numerator_policy": numerator,
                        "denominator_policy": denominator,
                        "model_ratio": model_ratio,
                        "actual_ratio": actual_ratio,
                        "same_direction_relative_to_one": bool(
                            (model_ratio < 1.0) == (actual_ratio < 1.0)
                        ),
                        "absolute_log_ratio_error": float(
                            abs(math.log(actual_ratio / model_ratio))
                        ),
                    }
                )
    return merged, pd.DataFrame(ratio_rows)


def legacy_before_after(
    paths: J1RPaths,
    fixed_summary: pd.DataFrame,
) -> pd.DataFrame:
    legacy_path = (
        paths.parent_results
        / "j1_end_to_end"
        / "r2_end_to_end_summary.csv"
    )
    legacy = pd.read_csv(legacy_path)
    legacy = legacy[
        legacy["result_group"] == "primary"
    ].copy()
    fixed = fixed_summary[
        fixed_summary["result_group"] == "primary"
    ].copy()

    keys = ["trace", "chunk_size", "repeat", "policy"]
    metrics = [
        "service_total_time_ns",
        "query_catalog_dispatch_time_ns",
        "query_codec_time_ns",
        "query_p99_ns",
        "migration_time_ns",
    ]
    left = fixed[keys + metrics].rename(
        columns={metric: f"fixed_{metric}" for metric in metrics}
    )
    right = legacy[keys + metrics].rename(
        columns={metric: f"legacy_{metric}" for metric in metrics}
    )
    merged = left.merge(
        right,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    for metric in metrics:
        merged[f"fixed_over_legacy_{metric}"] = (
            merged[f"fixed_{metric}"]
            / merged[f"legacy_{metric}"]
        )
    return merged


# ---------------------------------------------------------------------------
# Assessment, archives, and main
# ---------------------------------------------------------------------------

def environment_report() -> Dict[str, Any]:
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "available_memory_bytes": available_memory_bytes(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "timer": "time.perf_counter_ns",
    }
    if psutil is not None:
        with contextlib.suppress(Exception):
            memory = psutil.virtual_memory()
            report["total_memory_bytes"] = int(memory.total)
            report["available_memory_bytes"] = int(memory.available)
        with contextlib.suppress(Exception):
            report["cpu_count_logical"] = int(psutil.cpu_count())
            report["cpu_count_physical"] = int(
                psutil.cpu_count(logical=False) or 0
            )
        with contextlib.suppress(Exception):
            report["cpu_percent_at_start"] = float(
                psutil.cpu_percent(interval=1.0)
            )
    return report


def result_manifest(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(
            part in ("_store", "_base_stores")
            for part in relative.parts
        ):
            continue
        rows.append(
            {
                "relative_path": str(relative).replace("\\", "/"),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def create_archives(paths: J1RPaths) -> Dict[str, Any]:
    manifest = result_manifest(paths.output)
    manifest_path = paths.output / "j1r_result_manifest.csv"
    write_csv_atomic(manifest_path, manifest)

    core_zip = (
        paths.output / "RePressDB_J1R_CORE_RESULTS_V162.zip"
    )
    latency_zip = (
        paths.output / "RePressDB_J1R_LATENCY_ARRAYS_V162.zip"
    )

    with zipfile.ZipFile(
        core_zip,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(paths.output.rglob("*")):
            if not path.is_file():
                continue
            if path in (core_zip, latency_zip):
                continue
            relative = path.relative_to(paths.output)
            if path.name == "latencies.npz":
                continue
            if any(
                part in ("_store", "_base_stores")
                for part in relative.parts
            ):
                continue
            archive.write(
                path,
                arcname=str(relative).replace("\\", "/"),
            )

    latency_files = list(paths.output.rglob("latencies.npz"))
    if latency_files:
        with zipfile.ZipFile(
            latency_zip,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for path in latency_files:
                relative = path.relative_to(paths.output)
                archive.write(
                    path,
                    arcname=str(relative).replace("\\", "/"),
                )

    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "core_zip": str(core_zip),
        "core_zip_sha256": sha256_file(core_zip),
        "core_zip_bytes": int(core_zip.stat().st_size),
        "latency_zip": (
            str(latency_zip) if latency_zip.exists() else None
        ),
        "latency_zip_sha256": (
            sha256_file(latency_zip)
            if latency_zip.exists()
            else None
        ),
        "latency_zip_bytes": (
            int(latency_zip.stat().st_size)
            if latency_zip.exists()
            else 0
        ),
    }


def acceptance_checks(
    primary_summary: pd.DataFrame,
    memory_summary: pd.DataFrame,
    memory_stage: Mapping[str, Any],
    model_frame: pd.DataFrame,
    model_ratio_frame: pd.DataFrame,
    before_after: pd.DataFrame,
    paths: J1RPaths,
) -> Dict[str, bool]:
    expected_primary = (
        len(TRACE_FAMILIES)
        * len(CHUNK_SIZES)
        * REPEATS
        * len(POLICIES)
    )
    expected_memory = REPEATS * len(POLICIES)

    checks = {
        "J1R_1_all_54_primary_runs_complete": bool(
            len(primary_summary) == expected_primary
        ),
        "J1R_2_every_primary_query_and_prewarm_correct": bool(
            (
                primary_summary["all_queries_truth_checked"] == True
            ).all()
            and (
                primary_summary["prewarm_all_truth_checked"] == True
            ).all()
        ),
        "J1R_3_all_primary_stores_integrity_pass": bool(
            (primary_summary["store_integrity_pass"] == True).all()
            and (
                primary_summary["store_temporary_file_count"] == 0
            ).all()
            and (
                primary_summary["store_invalid_chunk_count"] == 0
            ).all()
            and (
                primary_summary["store_non_idle_chunk_count"] == 0
            ).all()
            and (
                primary_summary["store_wrong_payload_count"] == 0
            ).all()
        ),
        "J1R_4_no_primary_write_budget_violation": bool(
            (
                primary_summary["budget_violation_count"] == 0
            ).all()
        ),
        "J1R_5_query_cache_misses_bounded_by_committed_migrations": bool(
            (
                primary_summary["query_cache_miss_bound_pass"] == True
            ).all()
            and (
                primary_summary["query_active_view_cache_misses"]
                <= primary_summary["migration_count"]
            ).all()
        ),
        "J1R_6_no_per_query_catalog_or_metadata_reconstruction": bool(
            (
                primary_summary["query_active_view_cache_hits"]
                + primary_summary["query_active_view_cache_misses"]
                == primary_summary["n_queries"]
            ).all()
            and (
                primary_summary["cache_counter_catalog_row_loads"]
                <= 2 * primary_summary["migration_count"]
            ).all()
            and (
                primary_summary["cache_counter_stored_chunk_builds"]
                <= 2 * primary_summary["migration_count"]
            ).all()
        ),
        "J1R_7_fresh_j4_calibration_used_in_every_run": bool(
            primary_summary[
                "fresh_calibration_records_sha256"
            ].nunique()
            == 1
            and primary_summary[
                "fresh_migration_records_sha256"
            ].nunique()
            == 1
            and primary_summary[
                "fresh_calibration_records_sha256"
            ].iloc[0]
            == sha256_file(
                paths.fresh_calibration / "calibration_records.csv"
            )
        ),
        "J1R_8_model_actual_validation_complete": bool(
            len(model_frame)
            == len(TRACE_FAMILIES)
            * len(CHUNK_SIZES)
            * len(POLICIES)
            and len(model_ratio_frame)
            == len(TRACE_FAMILIES)
            * len(CHUNK_SIZES)
            * 3
            and np.isfinite(
                model_ratio_frame["actual_ratio"]
            ).all()
        ),
        "J1R_9_legacy_before_after_diagnostic_complete": bool(
            len(before_after) == expected_primary
            and np.isfinite(
                before_after[
                    "fixed_over_legacy_service_total_time_ns"
                ]
            ).all()
        ),
        "J1R_10_memory_pressure_completed_or_transparently_skipped": bool(
            (
                memory_stage.get("status") == "complete"
                and len(memory_summary) == expected_memory
                and (
                    memory_summary["all_queries_truth_checked"] == True
                ).all()
            )
            or (
                memory_stage.get("status") == "skipped_memory_gate"
                and len(memory_summary) == 0
            )
        ),
    }
    checks["all_ten"] = bool(all(checks.values()))
    return checks


def run_master() -> int:
    project = Path(__file__).resolve().parent
    paths = resolve_paths(project)
    parent_info = verify_parent(paths)
    loaded = load_parent_modules(paths, "master")
    fresh = load_fresh_tables(paths, loaded)

    print("=" * 118)
    print("RePressDB J1-R Active-View Cache Confirmation v1.6.2")
    print(f"Output: {paths.output.resolve()}")
    print("=" * 118)

    environment = environment_report()
    write_json_atomic(
        paths.output / "environment_report.json",
        environment,
    )

    primary_gate = wait_for_available_memory(
        PRIMARY_START_MIN_AVAILABLE_MEMORY_BYTES,
        PRIMARY_MEMORY_WAIT_TIMEOUT_SECONDS,
        "primary J1-R stage",
    )
    write_json_atomic(
        paths.output / "primary_memory_gate.json",
        primary_gate,
    )
    if not primary_gate["passed"]:
        raise RuntimeError(
            "Primary J1-R did not start because stable 2.5-GiB "
            "headroom was not observed within 10 minutes. Close memory-"
            "intensive applications and rerun this same file."
        )

    protocol, freeze_hash = create_protocol(
        paths,
        parent_info,
    )
    base_stores = create_base_stores(
        paths,
        loaded,
        fresh,
        freeze_hash,
    )

    # Primary 54 cells.
    primary_cells = build_cells(
        paths,
        base_stores,
        freeze_hash,
        "primary",
    )
    run_cells(
        primary_cells,
        PRIMARY_CELL_MIN_AVAILABLE_MEMORY_BYTES,
    )
    (
        primary_summary,
        primary_chunk,
        primary_migrations,
    ) = collect_results(primary_cells)

    # Optional memory-pressure stage is disabled for this focused repair.
    # The v1.5.1 host never established 4 GiB of headroom, so forcing an
    # additional 1-GiB allocation would recreate the paging confound.
    memory_cells = []
    memory_summary = pd.DataFrame(columns=primary_summary.columns)
    memory_chunk = pd.DataFrame(columns=primary_chunk.columns)
    memory_migrations = pd.DataFrame(
        columns=primary_migrations.columns
    )
    memory_gate = {
        "passed": False,
        "stage": "optional memory-pressure J1-R stage",
        "minimum_bytes": int(MEMORY_PRESSURE_MIN_AVAILABLE_BYTES),
        "disabled_by_protocol": True,
    }
    memory_stage: Dict[str, Any] = {
        "status": "skipped_memory_gate",
        "memory_gate": memory_gate,
        "n_runs": 0,
        "reason": (
            "The optional 1-GiB competing-allocation subset was disabled "
            "because the host could not establish sufficient uncontended "
            "memory headroom. The primary 54-cell confirmation remains the "
            "pre-registered J1-R result."
        ),
    }
    write_json_atomic(
        paths.output / "memory_pressure_stage.json",
        memory_stage,
    )

    all_summary = pd.concat(
        [primary_summary, memory_summary],
        ignore_index=True,
    )
    all_chunk = pd.concat(
        [primary_chunk, memory_chunk],
        ignore_index=True,
    )
    migration_frames = [
        frame
        for frame in (primary_migrations, memory_migrations)
        if not frame.empty
    ]
    all_migrations = (
        pd.concat(migration_frames, ignore_index=True)
        if migration_frames
        else pd.DataFrame()
    )

    statistics = paired_statistics(all_summary)
    order_frame = order_effects(primary_summary)
    model_frame = run_fresh_model_replay(
        paths,
        loaded,
        fresh,
    )
    model_actual_frame, model_ratio_frame = (
        model_actual_validation(
            primary_summary,
            model_frame,
        )
    )
    before_after = legacy_before_after(
        paths,
        primary_summary,
    )

    write_csv_atomic(
        paths.output / "j1r_end_to_end_summary.csv",
        all_summary,
    )
    write_csv_atomic(
        paths.output / "j1r_end_to_end_chunk_results.csv",
        all_chunk,
    )
    write_csv_atomic(
        paths.output / "j1r_end_to_end_migrations.csv",
        all_migrations,
    )
    write_csv_atomic(
        paths.output / "j1r_paired_statistics.csv",
        statistics,
    )
    write_csv_atomic(
        paths.output / "j1r_order_effects.csv",
        order_frame,
    )
    write_csv_atomic(
        paths.output / "j1r_fresh_calibration_model_results.csv",
        model_frame,
    )
    write_csv_atomic(
        paths.output / "j1r_model_actual_validation.csv",
        model_actual_frame,
    )
    write_csv_atomic(
        paths.output / "j1r_model_actual_ratio_validation.csv",
        model_ratio_frame,
    )
    write_csv_atomic(
        paths.output / "j1r_legacy_before_after.csv",
        before_after,
    )

    checks = acceptance_checks(
        primary_summary,
        memory_summary,
        memory_stage,
        model_frame,
        model_ratio_frame,
        before_after,
        paths,
    )

    primary_aggregate = statistics[
        (statistics["result_group"] == "primary_aggregate")
        & (statistics["metric"] == "service_total_time_ns")
    ].to_dict("records")
    dispatch_before_after = {
        policy: {
            "geometric_mean_fixed_over_legacy_dispatch": geometric_mean(
                group[
                    "fixed_over_legacy_query_catalog_dispatch_time_ns"
                ]
            ),
            "geometric_mean_fixed_over_legacy_service": geometric_mean(
                group[
                    "fixed_over_legacy_service_total_time_ns"
                ]
            ),
            "geometric_mean_fixed_over_legacy_p99": geometric_mean(
                group["fixed_over_legacy_query_p99_ns"]
            ),
        }
        for policy, group in before_after.groupby("policy")
    }

    assessment = {
        "version": VERSION,
        "target_journal": TARGET_JOURNAL,
        "status": (
            "J1-R complete"
            if checks["all_ten"]
            else "J1-R completed with failed integrity checks"
        ),
        "experiment_freeze_sha256": freeze_hash,
        "parent": parent_info,
        "implementation_change": protocol["implementation_change"],
        "primary_memory_gate": primary_gate,
        "memory_pressure_stage": memory_stage,
        "primary_runs": len(primary_summary),
        "memory_pressure_runs": len(memory_summary),
        "total_measured_queries": int(
            primary_summary["n_queries"].sum()
            + (
                memory_summary["n_queries"].sum()
                if not memory_summary.empty
                else 0
            )
        ),
        "acceptance_checks": checks,
        "primary_aggregate_comparisons": primary_aggregate,
        "legacy_before_after_by_policy": dispatch_before_after,
        "model_actual_direction_agreement": {
            "agreeing_cells": int(
                model_ratio_frame[
                    "same_direction_relative_to_one"
                ].sum()
            ),
            "total_cells": int(len(model_ratio_frame)),
        },
        "claim_boundaries": protocol["claim_boundaries"],
        "next_gate": (
            "Upload the J1-R core result ZIP. J2, J3, and J4 remain frozen."
            if checks["all_ten"]
            else (
                "Inspect failed J1-R checks before manuscript revision. "
                "Do not retune the policies."
            )
        ),
    }
    assessment_path = paths.output / "j1r_assessment.json"
    write_json_atomic(assessment_path, assessment)
    write_json_atomic(
        project / "J1R_ACTIVE_VIEW_CACHE_ASSESSMENT_V162.json",
        assessment,
    )

    archives = create_archives(paths)
    assessment["result_archives"] = archives
    write_json_atomic(assessment_path, assessment)
    write_json_atomic(
        project / "J1R_ACTIVE_VIEW_CACHE_ASSESSMENT_V162.json",
        assessment,
    )

    print("=" * 118)
    print("J1-R finished")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(f"Assessment: {assessment_path}")
    print(f"Core result ZIP: {archives['core_zip']}")
    if archives["latency_zip"]:
        print(f"Latency ZIP: {archives['latency_zip']}")
    print("=" * 118)
    return 0 if checks["all_ten"] else 2


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--internal-worker",
        action="store_true",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.internal_worker:
        if arguments.config is None:
            raise ValueError("--config is required for internal worker")
        return run_worker(Path(arguments.config))
    return run_master()


if __name__ == "__main__":
    raise SystemExit(main())
