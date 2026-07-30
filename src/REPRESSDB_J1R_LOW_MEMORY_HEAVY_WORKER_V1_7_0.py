#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RePressDB J1-R Low-Memory Heavy Worker v1.7.0

This file is not the user entry point. Run:
    RUN_THIS_ONLY_REPRESSDB_J1R_LOW_MEMORY_PAIRED_V1_7_0.py

The lightweight launcher starts this heavy worker in short-lived isolated
processes. Only one NumPy/Pandas/RePressDB process exists at a time.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sqlite3
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


VERSION = "1.7.0"
J1R_SCRIPT_NAME = (
    "RUN_THIS_ONLY_REPRESSDB_J1R_ACTIVE_VIEW_CACHE_CONFIRM_V1_6_2.py"
)
OUTPUT_DIR_NAME = "results_jss_j1r_low_memory_paired_v170"
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
IMPLEMENTATIONS = ("legacy", "cached")
BOOTSTRAP_REPLICATES = 20_000


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
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
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    token = "|".join(str(part) for part in parts)
    return int.from_bytes(
        hashlib.sha256(token.encode("utf-8")).digest()[:8],
        "little",
        signed=False,
    )


def load_module(path: Path, module_name: str) -> Any:
    specification = importlib.util.spec_from_file_location(module_name, path)
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


def bootstrap_gm_interval(
    values: Sequence[float],
    seed_parts: Sequence[Any],
) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array) & (array > 0)]
    if array.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(stable_seed("v170-bootstrap", *seed_parts))
    indices = rng.integers(
        0,
        array.size,
        size=(BOOTSTRAP_REPLICATES, array.size),
    )
    estimates = np.exp(np.mean(np.log(array[indices]), axis=1))
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def load_j1r(project: Path, tag: str) -> Any:
    path = project / J1R_SCRIPT_NAME
    if not path.exists():
        raise FileNotFoundError(path)
    return load_module(path, f"repressdb_j1r_v162_{tag}_{os.getpid()}")


def make_v170_paths(j1r: Any, project: Path) -> Any:
    original = j1r.resolve_paths(project)
    output = project / OUTPUT_DIR_NAME
    output.mkdir(parents=True, exist_ok=True)
    return j1r.J1RPaths(
        project=original.project,
        output=output,
        parent_script=original.parent_script,
        parent_results=original.parent_results,
        fresh_calibration=original.fresh_calibration,
        worker_assets=original.worker_assets,
    )


class LegacyReadPathStore:
    """
    Interface-compatible replacement for ActiveViewCachedStore.

    It preserves the original B1 read path:
        execute_query -> _row -> active_stored -> _row -> metadata parse

    The wrapper only adds counters. It does not cache catalog rows, StoredChunk,
    metadata, or active views.
    """

    def __init__(self, raw_store: Any, base: Any):
        self.raw = raw_store
        self.base = base
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

        self._original_row = raw_store._row
        self._original_active_stored = raw_store.active_stored

        def counted_row(chunk_id: str) -> Any:
            self._counters["catalog_row_loads"] += 1
            return self._original_row(chunk_id)

        def counted_active_stored(chunk_id: str) -> Any:
            stored = self._original_active_stored(chunk_id)
            self._counters["stored_chunk_builds"] += 1
            if getattr(stored, "metadata_path", None) is not None:
                self._counters["metadata_parses"] += 1
            return stored

        raw_store._row = counted_row
        raw_store.active_stored = counted_active_stored

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

    def execute_query(
        self,
        chunk_id: str,
        query: Any,
    ) -> Tuple[Any, int, Any, Any]:
        self._counters["query_calls"] += 1
        self._counters["query_cache_misses"] += 1
        return self.raw.execute_query(chunk_id, query)

    def active_stored(self, chunk_id: str) -> Any:
        return self.raw.active_stored(chunk_id)

    def previous_stored(self, chunk_id: str) -> Optional[Any]:
        return self.raw.previous_stored(chunk_id)

    def read_all(self, chunk_id: str) -> np.ndarray:
        return self.raw.read_all(chunk_id)

    def integrity_check(self) -> Dict[str, Any]:
        return self.raw.integrity_check()

    def invalidate_handle(self, chunk_id: str) -> None:
        self.raw.invalidate_handle(chunk_id)
        self._counters["active_view_cache_invalidations"] += 1

    def invalidate_all_handles(self) -> None:
        self.raw.invalidate_all_handles()

    def recover(self) -> Dict[str, int]:
        return self.raw.recover()

    def reset_query_counters(self) -> None:
        self._counters["query_cache_hits"] = 0
        self._counters["query_cache_misses"] = 0
        self._counters["query_calls"] = 0

    def counter_snapshot(self) -> Dict[str, int]:
        return {key: int(value) for key, value in self._counters.items()}

    def close(self) -> None:
        self.raw.close()


def create_protocol_and_freeze(
    output: Path,
    j1r: Any,
    paths: Any,
    parent_info: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str]:
    protocol = {
        "version": VERSION,
        "experiment": (
            "low-memory paired legacy-versus-active-view-cache "
            "end-to-end confirmation"
        ),
        "architecture": {
            "launcher": (
                "standard-library-only parent; no NumPy, Pandas, or RePressDB "
                "modules remain resident while a cell worker runs"
            ),
            "worker": (
                "one isolated heavy Python process per implementation-policy "
                "cell; process exits before the next cell starts"
            ),
            "aggregation": "separate final heavy process",
        },
        "parent": dict(parent_info),
        "j1r_script_sha256": sha256_file(paths.project / J1R_SCRIPT_NAME),
        "data": {
            "building_count": 8,
            "trace_families": list(TRACE_FAMILIES),
            "chunk_sizes": list(CHUNK_SIZES),
            "worker_asset_hashes": {
                path.name: sha256_file(path)
                for path in sorted(paths.worker_assets.glob("*.npz"))
            },
        },
        "policies": list(POLICIES),
        "policy_orders": [list(order) for order in POLICY_ORDERS],
        "implementations": {
            "legacy": (
                "frozen B1 query path with per-query catalog lookups and "
                "StoredChunk/metadata reconstruction"
            ),
            "cached": (
                "version-keyed active-view cache from J1-R v1.6.2"
            ),
        },
        "pairing": {
            "same trace, size, repeat, policy, base-store snapshot, "
            "fresh J4 calibration, and query realization": True,
            "implementation_order": (
                "balanced deterministically across the 54 pair units"
            ),
        },
        "calibration": {
            "source": str(paths.fresh_calibration),
            "calibration_records_sha256": parent_info[
                "fresh_calibration_records_sha256"
            ],
            "migration_records_sha256": parent_info[
                "fresh_migration_records_sha256"
            ],
        },
        "claim_boundaries": {
            "cache": (
                "in-process version-keyed active view; no distributed cache"
            ),
            "prewarm": (
                "one truth-checked query per chunk outside measured service "
                "time; matches warm_slicedmeta"
            ),
            "memory_pressure": (
                "no additional competing-memory allocation in v1.7.0"
            ),
            "performance": (
                "legacy and cached implementations are rerun on the same "
                "machine and in paired temporal proximity"
            ),
        },
    }
    protocol_path = output / "v170_protocol.json"
    write_json_atomic(protocol_path, protocol)
    freeze_material = {
        "protocol_sha256": sha256_file(protocol_path),
        "j1r_script_sha256": protocol["j1r_script_sha256"],
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
    freeze_path = output / "v170_freeze_material.json"
    write_json_atomic(freeze_path, freeze_material)
    freeze_hash = sha256_file(freeze_path)
    (
        output / "v170_experiment_freeze_sha256.txt"
    ).write_text(freeze_hash + "\n", encoding="utf-8")
    return protocol, freeze_hash


def implementation_order_for_pair(
    trace_index: int,
    size_index: int,
    repeat: int,
    policy_position: int,
) -> Tuple[str, str]:
    parity = (
        trace_index + size_index + repeat + policy_position
    ) % 2
    return (
        ("legacy", "cached")
        if parity == 0
        else ("cached", "legacy")
    )


def setup(project: Path) -> int:
    j1r = load_j1r(project, "setup")
    paths = make_v170_paths(j1r, project)
    parent_info = j1r.verify_parent(paths)
    loaded = j1r.load_parent_modules(paths, "v170_setup")
    fresh = j1r.load_fresh_tables(paths, loaded)
    protocol, freeze_hash = create_protocol_and_freeze(
        paths.output,
        j1r,
        paths,
        parent_info,
    )
    base_stores = j1r.create_base_stores(
        paths,
        loaded,
        fresh,
        freeze_hash,
    )

    pairs: List[Dict[str, Any]] = []
    run_sequence: List[Dict[str, Any]] = []
    pair_index = 0
    implementation_first_counts = {"legacy": 0, "cached": 0}

    for trace_index, trace in enumerate(TRACE_FAMILIES):
        for size_index, size in enumerate(CHUNK_SIZES):
            for repeat, policy_order in enumerate(POLICY_ORDERS):
                for policy_position, policy in enumerate(policy_order):
                    pair_id = (
                        f"pair_{pair_index:03d}__{trace}__size_{size}__"
                        f"repeat_{repeat:02d}__policy_{policy}"
                    )
                    implementation_order = implementation_order_for_pair(
                        trace_index,
                        size_index,
                        repeat,
                        policy_position,
                    )
                    implementation_first_counts[implementation_order[0]] += 1
                    pair_root = (
                        paths.output
                        / "pairs"
                        / trace
                        / f"size_{size}"
                        / f"repeat_{repeat:02d}"
                        / f"order_{policy_position}_{policy}"
                    )
                    cells: List[Dict[str, Any]] = []
                    for implementation_position, implementation in enumerate(
                        implementation_order
                    ):
                        output_dir = pair_root / implementation
                        config = {
                            "project_dir": str(project),
                            "output_dir": str(output_dir),
                            "parent_result_dir": str(paths.parent_results),
                            "fresh_calibration_dir": str(
                                paths.fresh_calibration
                            ),
                            "worker_assets_dir": str(paths.worker_assets),
                            "base_store": str(base_stores[size]),
                            "trace": trace,
                            "chunk_size": int(size),
                            "repeat": int(repeat),
                            "policy": policy,
                            "order_position": int(policy_position),
                            "policy_order": list(policy_order),
                            "result_group": "primary",
                            "memory_pressure": False,
                            "memory_pressure_bytes": 0,
                            "freeze_sha256": freeze_hash,
                            "implementation": implementation,
                            "implementation_position": int(
                                implementation_position
                            ),
                            "implementation_order": list(
                                implementation_order
                            ),
                            "pair_id": pair_id,
                            "paired_suite_version": VERSION,
                        }
                        config_path = output_dir / "worker_config.json"
                        write_json_atomic(config_path, config)
                        cell = {
                            **config,
                            "config_path": str(config_path),
                        }
                        cells.append(cell)
                        run_sequence.append(cell)
                    pairs.append(
                        {
                            "pair_index": pair_index,
                            "pair_id": pair_id,
                            "trace": trace,
                            "chunk_size": int(size),
                            "repeat": int(repeat),
                            "policy": policy,
                            "policy_position": int(policy_position),
                            "implementation_order": list(
                                implementation_order
                            ),
                            "cells": cells,
                        }
                    )
                    pair_index += 1

    if len(pairs) != 54 or len(run_sequence) != 108:
        raise AssertionError(
            f"Unexpected plan size: {len(pairs)} pairs, "
            f"{len(run_sequence)} workers"
        )
    if implementation_first_counts != {"legacy": 27, "cached": 27}:
        raise AssertionError(
            f"Implementation-order imbalance: {implementation_first_counts}"
        )

    plan = {
        "version": VERSION,
        "freeze_sha256": freeze_hash,
        "project_dir": str(project),
        "output_dir": str(paths.output),
        "pairs": pairs,
        "run_sequence": run_sequence,
        "pair_count": len(pairs),
        "worker_count": len(run_sequence),
        "implementation_first_counts": implementation_first_counts,
        "protocol_sha256": sha256_file(paths.output / "v170_protocol.json"),
    }
    write_json_atomic(paths.output / "experiment_plan.json", plan)
    write_json_atomic(
        paths.output / "setup_complete.json",
        {
            "complete": True,
            "freeze_sha256": freeze_hash,
            "plan_sha256": sha256_file(
                paths.output / "experiment_plan.json"
            ),
            "base_stores": {
                str(size): str(path)
                for size, path in base_stores.items()
            },
        },
    )
    print(
        f"SETUP_OK pairs={len(pairs)} workers={len(run_sequence)} "
        f"output={paths.output}"
    )
    return 0


def update_worker_summary(
    output_dir: Path,
    implementation: str,
    config: Mapping[str, Any],
) -> None:
    summary_path = output_dir / "summary.json"
    complete_path = output_dir / "complete.json"
    summary = read_json(summary_path)
    summary.update(
        {
            "paired_suite_version": VERSION,
            "implementation": implementation,
            "pair_id": str(config["pair_id"]),
            "implementation_position": int(
                config["implementation_position"]
            ),
            "implementation_order": list(
                config["implementation_order"]
            ),
            "read_path": (
                "legacy_b1"
                if implementation == "legacy"
                else "active_view_cached"
            ),
        }
    )
    write_json_atomic(summary_path, summary)
    marker = read_json(complete_path)
    marker["summary_sha256"] = sha256_file(summary_path)
    marker["implementation"] = implementation
    marker["pair_id"] = str(config["pair_id"])
    write_json_atomic(complete_path, marker)

    for filename in ("chunk_results.csv", "migration_results.csv"):
        path = output_dir / filename
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame()
        frame["implementation"] = implementation
        frame["pair_id"] = str(config["pair_id"])
        write_csv_atomic(path, frame)


def run_cell(project: Path, config_path: Path) -> int:
    config = read_json(config_path)
    implementation = str(config["implementation"])
    j1r = load_j1r(project, f"cell_{implementation}")

    if implementation == "legacy":
        j1r.ActiveViewCachedStore = LegacyReadPathStore
    elif implementation != "cached":
        raise ValueError(implementation)

    exit_code = int(j1r.run_worker(config_path))
    if exit_code != 0:
        return exit_code

    output_dir = Path(str(config["output_dir"]))
    update_worker_summary(output_dir, implementation, config)
    print(
        f"CELL_OK pair={config['pair_id']} "
        f"implementation={implementation} policy={config['policy']}"
    )
    return 0


def read_cell_outputs(
    cell: Mapping[str, Any],
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    root = Path(str(cell["output_dir"]))
    summary_path = root / "summary.json"
    complete_path = root / "complete.json"
    latency_path = root / "latencies.npz"
    if not (summary_path.exists() and complete_path.exists()):
        raise FileNotFoundError(f"Incomplete cell: {root}")
    marker = read_json(complete_path)
    if marker.get("summary_sha256") != sha256_file(summary_path):
        raise AssertionError(f"Summary hash mismatch: {root}")
    summary = read_json(summary_path)
    if summary.get("latency_npz_sha256") != sha256_file(latency_path):
        raise AssertionError(f"Latency hash mismatch: {root}")

    with np.load(latency_path, allow_pickle=False) as archive:
        query_outer = np.asarray(
            archive["query_outer_latency_ns"], dtype=np.int64
        )
        query_codec = np.asarray(
            archive["query_codec_latency_ns"], dtype=np.int64
        )
        controller = np.asarray(
            archive["controller_latency_ns"], dtype=np.int64
        )
        result_ok = np.asarray(archive["result_ok"], dtype=np.uint8)
    if int(result_ok.sum()) != len(result_ok):
        raise AssertionError(f"Truth failure: {root}")
    if int(query_outer.sum()) != int(summary["query_outer_time_ns"]):
        raise AssertionError(f"Query-time mismatch: {root}")
    if int(query_codec.sum()) != int(summary["query_codec_time_ns"]):
        raise AssertionError(f"Codec-time mismatch: {root}")
    if int(controller.sum()) != int(summary["controller_time_ns"]):
        raise AssertionError(f"Controller-time mismatch: {root}")

    flat = {
        key: value
        for key, value in summary.items()
        if key
        not in {
            "policy_order",
            "implementation_order",
            "prewarm_counters",
            "migration_stage_totals_ns",
            "measured_cache_counters",
            "store_integrity",
        }
    }
    flat["policy_order"] = json.dumps(
        summary.get("policy_order", []), ensure_ascii=False
    )
    flat["implementation_order"] = json.dumps(
        summary.get("implementation_order", []), ensure_ascii=False
    )
    for prefix, nested_key in (
        ("prewarm_counter", "prewarm_counters"),
        ("migration_stage", "migration_stage_totals_ns"),
        ("cache_counter", "measured_cache_counters"),
    ):
        for key, value in summary.get(nested_key, {}).items():
            flat[f"{prefix}_{key}"] = value
    integrity = summary.get("store_integrity", {})
    flat["store_integrity_pass"] = bool(integrity.get("pass", False))
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

    chunk_path = root / "chunk_results.csv"
    migration_path = root / "migration_results.csv"
    try:
        chunk_frame = pd.read_csv(chunk_path)
    except pd.errors.EmptyDataError:
        chunk_frame = pd.DataFrame()
    try:
        migration_frame = pd.read_csv(migration_path)
    except pd.errors.EmptyDataError:
        migration_frame = pd.DataFrame()
    return flat, chunk_frame, migration_frame


def paired_implementation_statistics(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    metrics = (
        "service_total_time_ns",
        "throughput_qps",
        "query_catalog_dispatch_time_ns",
        "query_codec_time_ns",
        "controller_time_ns",
        "migration_time_ns",
        "query_p99_ns",
        "fresh_prefix_service_time_ns",
        "steady_service_time_ns",
    )
    rows: List[Dict[str, Any]] = []
    for (policy, trace, size), group in summary.groupby(
        ["policy", "trace", "chunk_size"]
    ):
        for metric in metrics:
            pivot = group.pivot(
                index="repeat",
                columns="implementation",
                values=metric,
            ).dropna()
            ratio = (
                pivot["cached"].to_numpy(dtype=np.float64)
                / pivot["legacy"].to_numpy(dtype=np.float64)
            )
            low, high = bootstrap_gm_interval(
                ratio,
                ("implementation", policy, trace, size, metric),
            )
            rows.append(
                {
                    "scope": "trace_size_policy",
                    "policy": policy,
                    "trace": trace,
                    "chunk_size": int(size),
                    "metric": metric,
                    "numerator": "cached",
                    "denominator": "legacy",
                    "n_pairs": int(len(ratio)),
                    "geometric_mean_ratio": geometric_mean(ratio),
                    "median_ratio": float(np.median(ratio)),
                    "minimum_ratio": float(np.min(ratio)),
                    "maximum_ratio": float(np.max(ratio)),
                    "ci_low": low,
                    "ci_high": high,
                    "fraction_cached_lower": float(np.mean(ratio < 1.0)),
                }
            )

    for policy, group in summary.groupby("policy"):
        for metric in metrics:
            pivot = group.pivot(
                index=["trace", "chunk_size", "repeat"],
                columns="implementation",
                values=metric,
            ).dropna()
            ratio = (
                pivot["cached"].to_numpy(dtype=np.float64)
                / pivot["legacy"].to_numpy(dtype=np.float64)
            )
            low, high = bootstrap_gm_interval(
                ratio,
                ("implementation_aggregate", policy, metric),
            )
            rows.append(
                {
                    "scope": "policy_aggregate",
                    "policy": policy,
                    "trace": "all",
                    "chunk_size": 0,
                    "metric": metric,
                    "numerator": "cached",
                    "denominator": "legacy",
                    "n_pairs": int(len(ratio)),
                    "geometric_mean_ratio": geometric_mean(ratio),
                    "median_ratio": float(np.median(ratio)),
                    "minimum_ratio": float(np.min(ratio)),
                    "maximum_ratio": float(np.max(ratio)),
                    "ci_low": low,
                    "ci_high": high,
                    "fraction_cached_lower": float(np.mean(ratio < 1.0)),
                }
            )
    return pd.DataFrame(rows)


def policy_statistics_by_implementation(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    j1r = None
    # Reimplement locally so this aggregate remains independent from v1.6.2
    metrics = (
        "service_total_time_ns",
        "throughput_qps",
        "query_catalog_dispatch_time_ns",
        "query_codec_time_ns",
        "query_p99_ns",
    )
    comparisons = (
        ("repressdb_frozen", "shape_ewma_be_frozen"),
        ("repressdb_frozen", "initial_best"),
        ("shape_ewma_be_frozen", "initial_best"),
    )
    output_rows: List[Dict[str, Any]] = []
    for implementation, implementation_group in summary.groupby(
        "implementation"
    ):
        for (trace, size), group in implementation_group.groupby(
            ["trace", "chunk_size"]
        ):
            for metric in metrics:
                pivot = group.pivot(
                    index="repeat",
                    columns="policy",
                    values=metric,
                ).dropna()
                for numerator, denominator in comparisons:
                    ratio = (
                        pivot[numerator].to_numpy(dtype=np.float64)
                        / pivot[denominator].to_numpy(dtype=np.float64)
                    )
                    low, high = bootstrap_gm_interval(
                        ratio,
                        (
                            "policy",
                            implementation,
                            trace,
                            size,
                            metric,
                            numerator,
                            denominator,
                        ),
                    )
                    output_rows.append(
                        {
                            "implementation": implementation,
                            "scope": "trace_size",
                            "trace": trace,
                            "chunk_size": int(size),
                            "metric": metric,
                            "numerator_policy": numerator,
                            "denominator_policy": denominator,
                            "n_pairs": int(len(ratio)),
                            "geometric_mean_ratio": geometric_mean(ratio),
                            "median_ratio": float(np.median(ratio)),
                            "ci_low": low,
                            "ci_high": high,
                            "fraction_ratio_below_one": float(
                                np.mean(ratio < 1.0)
                            ),
                        }
                    )
        for metric in metrics:
            pivot = implementation_group.pivot(
                index=["trace", "chunk_size", "repeat"],
                columns="policy",
                values=metric,
            ).dropna()
            for numerator, denominator in comparisons:
                ratio = (
                    pivot[numerator].to_numpy(dtype=np.float64)
                    / pivot[denominator].to_numpy(dtype=np.float64)
                )
                low, high = bootstrap_gm_interval(
                    ratio,
                    (
                        "policy_aggregate",
                        implementation,
                        metric,
                        numerator,
                        denominator,
                    ),
                )
                output_rows.append(
                    {
                        "implementation": implementation,
                        "scope": "aggregate",
                        "trace": "all",
                        "chunk_size": 0,
                        "metric": metric,
                        "numerator_policy": numerator,
                        "denominator_policy": denominator,
                        "n_pairs": int(len(ratio)),
                        "geometric_mean_ratio": geometric_mean(ratio),
                        "median_ratio": float(np.median(ratio)),
                        "ci_low": low,
                        "ci_high": high,
                        "fraction_ratio_below_one": float(
                            np.mean(ratio < 1.0)
                        ),
                    }
                )
    return pd.DataFrame(output_rows)


def migration_signature(frame: pd.DataFrame) -> Tuple[Tuple[Any, ...], ...]:
    if frame.empty:
        return tuple()
    columns = [
        "global_event_index",
        "chunk_id",
        "source_codec",
        "target_codec",
        "rewrite_bytes",
    ]
    local = frame[columns].sort_values(
        ["global_event_index", "chunk_id", "target_codec"]
    )
    return tuple(
        tuple(row)
        for row in local.itertuples(index=False, name=None)
    )


def migration_equivalence(
    plan: Mapping[str, Any],
    migration_frames: Mapping[str, pd.DataFrame],
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    summary_index = summary.set_index(
        ["pair_id", "implementation"]
    )
    for pair in plan["pairs"]:
        pair_id = str(pair["pair_id"])
        legacy = migration_frames.get(
            f"{pair_id}|legacy", pd.DataFrame()
        )
        cached = migration_frames.get(
            f"{pair_id}|cached", pd.DataFrame()
        )
        legacy_signature = migration_signature(legacy)
        cached_signature = migration_signature(cached)
        legacy_summary = summary_index.loc[(pair_id, "legacy")]
        cached_summary = summary_index.loc[(pair_id, "cached")]
        rows.append(
            {
                "pair_id": pair_id,
                "trace": pair["trace"],
                "chunk_size": int(pair["chunk_size"]),
                "repeat": int(pair["repeat"]),
                "policy": pair["policy"],
                "legacy_migration_count": int(
                    legacy_summary["migration_count"]
                ),
                "cached_migration_count": int(
                    cached_summary["migration_count"]
                ),
                "legacy_rewrite_bytes": int(
                    legacy_summary["rewrite_bytes"]
                ),
                "cached_rewrite_bytes": int(
                    cached_summary["rewrite_bytes"]
                ),
                "exact_migration_signature_match": bool(
                    legacy_signature == cached_signature
                ),
            }
        )
    return pd.DataFrame(rows)


def implementation_order_summary(
    plan: Mapping[str, Any],
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for implementation_position, group in summary.groupby(
        ["implementation", "implementation_position"]
    ):
        implementation, position = implementation_position
        rows.append(
            {
                "implementation": implementation,
                "implementation_position": int(position),
                "n_cells": int(len(group)),
                "median_service_time_ns": float(
                    group["service_total_time_ns"].median()
                ),
                "geometric_mean_service_time_ns": geometric_mean(
                    group["service_total_time_ns"]
                ),
            }
        )
    return pd.DataFrame(rows)


def create_archives(output: Path) -> Dict[str, Any]:
    manifest_rows: List[Dict[str, Any]] = []
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output)
        if any(part in ("_base_stores", "_store") for part in relative.parts):
            continue
        manifest_rows.append(
            {
                "relative_path": str(relative).replace("\\", "/"),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output / "v170_result_manifest.csv"
    write_csv_atomic(manifest_path, manifest)

    core_zip = output / "RePressDB_J1R_LOW_MEMORY_PAIRED_CORE_V170.zip"
    latency_zip = (
        output / "RePressDB_J1R_LOW_MEMORY_PAIRED_LATENCIES_V170.zip"
    )

    with zipfile.ZipFile(
        core_zip,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(output.rglob("*")):
            if not path.is_file() or path in (core_zip, latency_zip):
                continue
            relative = path.relative_to(output)
            if path.name == "latencies.npz":
                continue
            if any(
                part in ("_base_stores", "_store")
                for part in relative.parts
            ):
                continue
            archive.write(
                path,
                arcname=str(relative).replace("\\", "/"),
            )

    latency_files = list(output.rglob("latencies.npz"))
    if latency_files:
        with zipfile.ZipFile(
            latency_zip,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for path in latency_files:
                relative = path.relative_to(output)
                archive.write(
                    path,
                    arcname=str(relative).replace("\\", "/"),
                )
    return {
        "core_zip": str(core_zip),
        "core_zip_sha256": sha256_file(core_zip),
        "core_zip_bytes": int(core_zip.stat().st_size),
        "latency_zip": str(latency_zip),
        "latency_zip_sha256": sha256_file(latency_zip),
        "latency_zip_bytes": int(latency_zip.stat().st_size),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def aggregate(project: Path) -> int:
    j1r = load_j1r(project, "aggregate")
    paths = make_v170_paths(j1r, project)
    plan_path = paths.output / "experiment_plan.json"
    plan = read_json(plan_path)

    summary_rows: List[Dict[str, Any]] = []
    chunk_frames: List[pd.DataFrame] = []
    migration_frames_list: List[pd.DataFrame] = []
    migration_frames_by_cell: Dict[str, pd.DataFrame] = {}

    for cell in plan["run_sequence"]:
        flat, chunk_frame, migration_frame = read_cell_outputs(cell)
        summary_rows.append(flat)
        if not chunk_frame.empty:
            chunk_frames.append(chunk_frame)
        if not migration_frame.empty:
            migration_frames_list.append(migration_frame)
        migration_frames_by_cell[
            f"{cell['pair_id']}|{cell['implementation']}"
        ] = migration_frame

    summary = pd.DataFrame(summary_rows)
    chunks = (
        pd.concat(chunk_frames, ignore_index=True)
        if chunk_frames
        else pd.DataFrame()
    )
    migrations = (
        pd.concat(migration_frames_list, ignore_index=True)
        if migration_frames_list
        else pd.DataFrame()
    )

    implementation_stats = paired_implementation_statistics(summary)
    policy_stats = policy_statistics_by_implementation(summary)
    migration_match = migration_equivalence(
        plan,
        migration_frames_by_cell,
        summary,
    )
    order_summary = implementation_order_summary(plan, summary)

    write_csv_atomic(
        paths.output / "v170_end_to_end_summary.csv", summary
    )
    write_csv_atomic(
        paths.output / "v170_end_to_end_chunk_results.csv", chunks
    )
    write_csv_atomic(
        paths.output / "v170_end_to_end_migrations.csv", migrations
    )
    write_csv_atomic(
        paths.output / "v170_implementation_paired_statistics.csv",
        implementation_stats,
    )
    write_csv_atomic(
        paths.output / "v170_policy_statistics_by_implementation.csv",
        policy_stats,
    )
    write_csv_atomic(
        paths.output / "v170_migration_decision_equivalence.csv",
        migration_match,
    )
    write_csv_atomic(
        paths.output / "v170_implementation_order_summary.csv",
        order_summary,
    )

    expected_rows = 108
    legacy = summary[summary["implementation"] == "legacy"]
    cached = summary[summary["implementation"] == "cached"]

    pair_counts = summary.groupby("pair_id")["implementation"].nunique()
    first_counts = {
        implementation: int(
            sum(
                pair["implementation_order"][0] == implementation
                for pair in plan["pairs"]
            )
        )
        for implementation in IMPLEMENTATIONS
    }

    cached_catalog_bound = (
        cached["cache_counter_catalog_row_loads"]
        <= 2 * cached["migration_count"]
    ).all()
    cached_build_bound = (
        cached["cache_counter_stored_chunk_builds"]
        <= 2 * cached["migration_count"]
    ).all()
    legacy_catalog_expected = (
        legacy["cache_counter_catalog_row_loads"]
        >= 2 * legacy["n_queries"]
    ).all()

    calibration_hash_match = (
        summary["fresh_calibration_records_sha256"].nunique() == 1
        and summary["fresh_migration_records_sha256"].nunique() == 1
        and summary["fresh_calibration_records_sha256"].iloc[0]
        == sha256_file(
            paths.fresh_calibration / "calibration_records.csv"
        )
    )

    checks = {
        "V170_1_all_108_workers_complete": bool(
            len(summary) == expected_rows
            and (pair_counts == 2).all()
        ),
        "V170_2_every_query_and_prewarm_correct": bool(
            (summary["all_queries_truth_checked"] == True).all()
            and (summary["prewarm_all_truth_checked"] == True).all()
        ),
        "V170_3_all_stores_integrity_pass": bool(
            (summary["store_integrity_pass"] == True).all()
            and (summary["store_temporary_file_count"] == 0).all()
            and (summary["store_invalid_chunk_count"] == 0).all()
            and (summary["store_non_idle_chunk_count"] == 0).all()
            and (summary["store_wrong_payload_count"] == 0).all()
        ),
        "V170_4_no_write_budget_violation": bool(
            (summary["budget_violation_count"] == 0).all()
        ),
        "V170_5_legacy_and_cached_migration_decisions_identical": bool(
            len(migration_match) == 54
            and (
                migration_match["exact_migration_signature_match"] == True
            ).all()
            and (
                migration_match["legacy_migration_count"]
                == migration_match["cached_migration_count"]
            ).all()
            and (
                migration_match["legacy_rewrite_bytes"]
                == migration_match["cached_rewrite_bytes"]
            ).all()
        ),
        "V170_6_cached_path_removes_per_query_catalog_reconstruction": bool(
            cached_catalog_bound
            and cached_build_bound
            and legacy_catalog_expected
        ),
        "V170_7_implementation_order_balanced": bool(
            first_counts == {"legacy": 27, "cached": 27}
        ),
        "V170_8_fresh_j4_calibration_identical_in_all_cells": bool(
            calibration_hash_match
        ),
        "V170_9_latency_arrays_recomputed_and_verified": bool(
            len(summary) == expected_rows
            and np.isfinite(summary["query_p99_ns"]).all()
        ),
        "V170_10_paired_performance_tables_complete": bool(
            len(implementation_stats) > 0
            and len(policy_stats) > 0
            and len(order_summary) == 4
        ),
    }
    checks["all_ten"] = bool(all(checks.values()))

    aggregate_service = implementation_stats[
        (implementation_stats["scope"] == "policy_aggregate")
        & (
            implementation_stats["metric"]
            == "service_total_time_ns"
        )
    ].to_dict("records")
    aggregate_dispatch = implementation_stats[
        (implementation_stats["scope"] == "policy_aggregate")
        & (
            implementation_stats["metric"]
            == "query_catalog_dispatch_time_ns"
        )
    ].to_dict("records")
    cached_policy_service = policy_stats[
        (policy_stats["implementation"] == "cached")
        & (policy_stats["scope"] == "aggregate")
        & (policy_stats["metric"] == "service_total_time_ns")
    ].to_dict("records")

    assessment = {
        "version": VERSION,
        "status": (
            "low-memory paired J1-R complete"
            if checks["all_ten"]
            else "low-memory paired J1-R completed with failed checks"
        ),
        "experiment_freeze_sha256": plan["freeze_sha256"],
        "pair_count": int(plan["pair_count"]),
        "worker_count": int(plan["worker_count"]),
        "total_measured_queries": int(summary["n_queries"].sum()),
        "implementation_first_counts": first_counts,
        "acceptance_checks": checks,
        "cached_over_legacy_service_by_policy": aggregate_service,
        "cached_over_legacy_dispatch_by_policy": aggregate_dispatch,
        "cached_policy_service_comparisons": cached_policy_service,
        "claim_boundaries": read_json(
            paths.output / "v170_protocol.json"
        )["claim_boundaries"],
        "next_gate": (
            "Upload the core ZIP and assessment for manuscript revision."
            if checks["all_ten"]
            else "Inspect the failed check before manuscript revision."
        ),
    }
    assessment_path = paths.output / "v170_assessment.json"
    write_json_atomic(assessment_path, assessment)
    write_json_atomic(
        project / "J1R_LOW_MEMORY_PAIRED_ASSESSMENT_V170.json",
        assessment,
    )

    archives = create_archives(paths.output)
    assessment["result_archives"] = archives
    write_json_atomic(assessment_path, assessment)
    write_json_atomic(
        project / "J1R_LOW_MEMORY_PAIRED_ASSESSMENT_V170.json",
        assessment,
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(f"AGGREGATE_OK assessment={assessment_path}")
    print(f"CORE_ZIP={archives['core_zip']}")
    return 0 if checks["all_ten"] else 2


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=("setup", "cell", "aggregate"),
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--config")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    project = Path(arguments.project).resolve()
    if arguments.mode == "setup":
        return setup(project)
    if arguments.mode == "cell":
        if arguments.config is None:
            raise ValueError("--config is required for cell mode")
        return run_cell(project, Path(arguments.config))
    if arguments.mode == "aggregate":
        return aggregate(project)
    raise ValueError(arguments.mode)


if __name__ == "__main__":
    raise SystemExit(main())
