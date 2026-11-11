#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
RePressDB A0+A1 v0.7.0
AF-HC protocol freeze and PEMS04 development-grid selection.

Target journal: Journal of Big Data.

DIRECT-RUN VERSION
------------------
Open this file in PyCharm and click the green Run button.
No command-line arguments and no source-code edits are required.

Required dependency files in the same directory:
  1) RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT_V0_2_0.py
  2) RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT.py
  3) RUN_THIS_ONLY_REPRESSDB_E0C_V0_2_0.py

What this program does
----------------------
A0: freezes the nearest-baseline protocol.
  * deterministically selects 32 PEMS04 series and 32 PEMS08 series;
  * creates paired 8,192/16,384 chunks for each selected series;
  * creates balanced orthogonal trace-group assignments;
  * detects the prior BDG2 E5 selection manifest when available;
  * writes all manifests and a SHA-256 protocol hash.

A1: selects an Access-Frequency Hot/Cold (AF-HC) baseline on PEMS04 only.
  * AF-HC uses only per-chunk access frequency;
  * AF-HC never observes point/scan/aggregate labels;
  * AF-HC does not use RePressDB's dual windows, cost model, confirmation,
    break-even rule, cooldown, or future phase information;
  * 72 configurations are compared under a predeclared grid;
  * grid selection uses calibrated query costs and physically measured
    directed migration costs, avoiding 72 repeated wall-clock replays;
  * the selected configuration is then audited using real file-backed queries,
    actual decode+encode+fsync migrations, and per-query correctness checks;
  * the selected configuration is frozen for later PEMS08 and BDG2 holdouts.

The frozen RePressDB controller is not retuned.

Primary output directory
------------------------
results_a0_a1_hotcold_pems04_v070/
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

VERSION = "0.7.0"
TARGET_JOURNAL = "Journal of Big Data"
SEED = 20260718
TRACE_SEEDS = (20260718, 20260719, 20260720)

OUTPUT_DIR_NAME = "results_a0_a1_hotcold_pems04_v070"
SUMMARY_JSON_NAME = "A0_A1_HOTCOLD_PEMS04_ASSESSMENT_v070.json"

N_SERIES = 32
CHUNK_SIZES = (8192, 16384)
QUERY_CLASSES = (
    "point",
    "short_scan",
    "medium_scan",
    "long_scan",
    "short_agg",
    "medium_agg",
    "long_agg",
    "threshold",
)

TRACE_NAMES = (
    "shape_only",
    "frequency_only",
    "joint_orthogonal",
    "brief_burst",
    "stable_scan",
)

# AF-HC grid is frozen before seeing PEMS08/BDG2 results.
ADAPTATION_PERIODS = (100, 300, 500, 1000)
HOT_FRACTIONS = (0.25, 0.50, 0.75)
COLD_CODECS = ("whole_lz4", "whole_zstd")
HOT_CODECS = (
    "block_lz4_64",
    "block_lz4_256",
    "block_lz4_1024",
)
FREQUENCY_EMA_ALPHA = 0.50

# Grid safety and tie-breaking rules are protocol constants, not tuned outputs.
MAX_STABLE_RATIO = 1.05
MAX_BURST_RATIO = 1.05
MAX_BURST_MEDIAN_MIGRATIONS_PER_CHUNK = 2.0
TIE_SCORE_TOLERANCE = 0.005  # 0.5%

# Frozen RePressDB constants, copied from E1 v0.2 protocol.
LONG_WINDOW = 300
SHORT_WINDOW = 100
MIN_OBSERVATIONS = 200
DECISION_INTERVAL = 100
CONFIRMATION_ROUNDS = 3
COOLDOWN_QUERIES = 300
PREDICTION_HORIZON = 1200
MIN_RELATIVE_IMPROVEMENT = 0.08
REWRITE_SAFETY_FACTOR = 1.25

# Actual audit runs all five traces, both sizes, and all three seeds.
AUDIT_POLICIES = (
    "initial_best",
    "afhc_frozen",
    "repressdb",
)

MIGRATION_MEASURE_REPEATS = 2
BOOTSTRAP_ITERATIONS = 5000

# Existing project paths.
BDG2_ROOT = Path(
    r"E:\IPM\大学\科研\E\buds-lab-building-data-genome-project-2-3d0cbaf"
)
BDG2_E5_RESULT_DIR_NAME = "results_e5_bdg2_frozen_crossdomain_v060"

QUICK_TEST = os.environ.get("REPRESSDB_A1_QUICK", "0") == "1"


@dataclass(frozen=True)
class LightweightTracePlan:
    trace_name: str
    trace_seed: int
    chunk_size: int
    chunk_ids: Tuple[str, ...]
    chunk_positions: np.ndarray
    local_indices: np.ndarray
    query_class_positions: np.ndarray
    query_choice_positions: np.ndarray
    workload_positions: np.ndarray
    workload_names: Tuple[str, ...]

    @property
    def n_events(self) -> int:
        return int(self.chunk_positions.size)


@dataclass
class RepressState:
    active_codec: str
    recent_classes: Deque[str] = field(
        default_factory=lambda: deque(maxlen=LONG_WINDOW)
    )
    local_access_count: int = 0
    last_switch_local_index: int = -10**9
    pending_candidate: Optional[str] = None
    pending_rounds: int = 0
    query_cost_ns: float = 0.0
    migration_cost_ns: float = 0.0
    migration_count: int = 0
    rewrite_bytes: int = 0


@dataclass
class AFHCState:
    active_codec: str
    query_cost_ns: float = 0.0
    migration_cost_ns: float = 0.0
    migration_count: int = 0
    rewrite_bytes: int = 0


class MultiChunkHandles:
    def __init__(
        self,
        stored_by_chunk: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.payload: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        for chunk_id, stored_map in stored_by_chunk.items():
            self.payload[chunk_id] = {}
            self.metadata[chunk_id] = {}
            for codec, stored in stored_map.items():
                self.payload[chunk_id][codec] = open(
                    stored.data_path, "rb"
                )
                self.metadata[chunk_id][codec] = (
                    open(stored.metadata_path, "rb")
                    if stored.metadata_path is not None
                    else None
                )

    def close(self) -> None:
        for per_chunk in self.payload.values():
            for handle in per_chunk.values():
                if handle is not None:
                    handle.close()
        for per_chunk in self.metadata.values():
            for handle in per_chunk.values():
                if handle is not None:
                    handle.close()
        self.payload.clear()
        self.metadata.clear()

    def __enter__(self) -> "MultiChunkHandles":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def stable_seed(*parts: Any) -> int:
    token = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def stable_hash(*parts: Any) -> str:
    token = "|".join(str(part) for part in parts)
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(
            json_safe(value),
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        )
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_module(filename: str, module_name: str) -> Any:
    path = Path(__file__).resolve().parent / filename
    if not path.exists():
        raise FileNotFoundError(
            f"找不到依赖文件：{filename}\n"
            "请解压完整包，并保证所有 Python 文件位于同一目录。"
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_repressdb_modules() -> Tuple[Any, Any, Any]:
    e1v2 = load_module(
        "RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT_V0_2_0.py",
        "repressdb_e1v2_for_afhc",
    )
    e1 = e1v2.load_e1_module()
    base = e1.load_base_module()
    if getattr(e1v2, "VERSION", None) != "0.2.0":
        raise RuntimeError("需要 E1 v0.2.0 冻结控制器。")
    if getattr(e1, "VERSION", None) != "0.1.0":
        raise RuntimeError("需要 E1 v0.1.0 基础执行器。")
    return e1v2, e1, base


def deterministic_series_chunks(
    base: Any,
    matrix: np.ndarray,
    dataset_label: str,
    selected_original_indices: Sequence[int],
    chunk_sizes: Sequence[int] = CHUNK_SIZES,
) -> Tuple[List[Any], pd.DataFrame]:
    chunks: List[Any] = []
    rows: List[Dict[str, Any]] = []
    time_length, local_series_count = matrix.shape
    if local_series_count != len(selected_original_indices):
        raise AssertionError("Selected-series metadata does not match matrix width")

    for local_index, original_index in enumerate(selected_original_indices):
        for chunk_size in chunk_sizes:
            if chunk_size > time_length:
                raise ValueError(
                    f"{dataset_label}: chunk_size={chunk_size} exceeds length={time_length}"
                )
            maximum_start = time_length - chunk_size
            rng = np.random.default_rng(
                stable_seed(
                    SEED,
                    dataset_label,
                    int(original_index),
                    int(chunk_size),
                    "paired_chunk_start",
                )
            )
            start = (
                int(rng.integers(0, maximum_start + 1))
                if maximum_start > 0
                else 0
            )
            values = np.ascontiguousarray(
                matrix[start : start + chunk_size, local_index],
                dtype=np.float32,
            )
            chunk_id = (
                f"{dataset_label.lower()}_cs{chunk_size}_"
                f"orig{int(original_index)}_t{start}"
            )
            chunk = base.ChunkUnit(
                chunk_id=chunk_id,
                series_index=int(original_index),
                chunk_start=int(start),
                chunk_size=int(chunk_size),
                values=values,
            )
            chunks.append(chunk)
            rows.append(
                {
                    "dataset": dataset_label,
                    "chunk_id": chunk_id,
                    "original_series_index": int(original_index),
                    "local_series_index": int(local_index),
                    "chunk_size": int(chunk_size),
                    "chunk_start": int(start),
                    "chunk_end_exclusive": int(start + chunk_size),
                    "raw_bytes": int(values.nbytes),
                    "selection_seed": SEED,
                }
            )
    return chunks, pd.DataFrame(rows)


def build_trace_assignments(
    chunk_manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (dataset, chunk_size), group in chunk_manifest.groupby(
        ["dataset", "chunk_size"], sort=True
    ):
        ordered = group.copy()
        ordered["assignment_key"] = [
            stable_hash(
                SEED,
                dataset,
                int(chunk_size),
                int(row.original_series_index),
                "orthogonal_assignment",
            )
            for row in ordered.itertuples(index=False)
        ]
        ordered = ordered.sort_values(
            ["assignment_key", "original_series_index"]
        ).reset_index(drop=True)

        if len(ordered) != N_SERIES:
            raise AssertionError(
                f"{dataset}/{chunk_size}: expected {N_SERIES} chunks, got {len(ordered)}"
            )

        for rank, row in enumerate(ordered.itertuples(index=False)):
            # Alternating assignments ensure every frequency tie-rank region
            # contains both query-shape groups.
            shape_group = "point_drift" if rank % 2 == 0 else "scan_stable"
            frequency_group = "hot" if rank % 2 == 0 else "cold"
            joint_group = (
                "hot_point",
                "hot_scan",
                "cold_point",
                "cold_scan",
            )[rank % 4]
            burst_group = "burst" if rank % 2 == 0 else "scan_stable"

            base_row = {
                "dataset": dataset,
                "chunk_id": row.chunk_id,
                "original_series_index": int(row.original_series_index),
                "chunk_size": int(chunk_size),
                "assignment_rank": int(rank),
                "frequency_tie_break_key": row.assignment_key,
            }
            for trace_name, group_name in (
                ("shape_only", shape_group),
                ("frequency_only", frequency_group),
                ("joint_orthogonal", joint_group),
                ("brief_burst", burst_group),
                ("stable_scan", "scan_stable"),
            ):
                rows.append(
                    {
                        **base_row,
                        "trace": trace_name,
                        "group": group_name,
                    }
                )
    return pd.DataFrame(rows)


def locate_bdg2_selection_manifest(project_dir: Path) -> Optional[Path]:
    candidates = [
        project_dir
        / BDG2_E5_RESULT_DIR_NAME
        / "bdg2_selected_series.csv",
        project_dir / "bdg2_selected_series.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_bdg2_future_manifest(project_dir: Path) -> pd.DataFrame:
    path = locate_bdg2_selection_manifest(project_dir)
    if path is None:
        return pd.DataFrame(
            columns=[
                "dataset",
                "building_id",
                "source_file",
                "source_column",
                "meter_type",
                "site_id",
                "longest_run_start",
                "longest_run_length",
                "status",
            ]
        )

    frame = pd.read_csv(path)
    required = {
        "building_id",
        "source_file",
        "source_column",
        "longest_run_start",
        "longest_run_length",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"BDG2 selection manifest misses columns: {sorted(missing)}"
        )

    frame = frame.drop_duplicates("building_id").copy()
    frame["selection_key"] = [
        stable_hash(SEED, building_id, "afhc_bdg2_holdout")
        for building_id in frame["building_id"].astype(str)
    ]
    frame = frame.sort_values(
        ["selection_key", "building_id"]
    ).head(N_SERIES)
    frame["dataset"] = "BDG2"
    frame["status"] = "reserved_for_A2_holdout"

    keep = [
        "dataset",
        "building_id",
        "source_file",
        "source_column",
        "meter_type",
        "site_id",
        "primary_use",
        "longest_run_start",
        "longest_run_length",
        "selection_key",
        "status",
    ]
    for column in keep:
        if column not in frame:
            frame[column] = ""
    return frame[keep].reset_index(drop=True)


def group_for_chunk(
    assignments: pd.DataFrame,
    chunk_id: str,
    trace_name: str,
) -> str:
    local = assignments[
        (assignments["chunk_id"] == chunk_id)
        & (assignments["trace"] == trace_name)
    ]
    if len(local) != 1:
        raise ValueError(
            f"Assignment is not unique: chunk={chunk_id}, trace={trace_name}"
        )
    return str(local.iloc[0]["group"])


def local_access_schedule(trace_name: str, group: str) -> Tuple[int, int]:
    """Return (number of global rounds, access stride)."""
    if trace_name in {"shape_only", "brief_burst", "stable_scan"}:
        return 3200, 1
    if trace_name == "frequency_only":
        return (3200, 1) if group == "hot" else (3200, 4)
    if trace_name == "joint_orthogonal":
        return (
            (3200, 1)
            if group.startswith("hot_")
            else (3200, 4)
        )
    raise ValueError(f"Unknown trace: {trace_name}")


def workload_for_local_access(
    trace_name: str,
    group: str,
    local_index: int,
    total_local_accesses: int,
) -> str:
    if trace_name == "shape_only":
        if group == "point_drift":
            boundary = total_local_accesses // 4
            return "scan_heavy" if local_index < boundary else "point_heavy"
        return "scan_heavy"

    if trace_name == "frequency_only":
        boundary = total_local_accesses // 4
        return "scan_heavy" if local_index < boundary else "point_heavy"

    if trace_name == "joint_orthogonal":
        if group.endswith("_point"):
            boundary = total_local_accesses // 4
            return "scan_heavy" if local_index < boundary else "point_heavy"
        return "scan_heavy"

    if trace_name == "brief_burst":
        if group == "burst":
            if local_index < 1000:
                return "scan_heavy"
            if local_index < 1250:
                return "point_heavy"
        return "scan_heavy"

    if trace_name == "stable_scan":
        return "scan_heavy"

    raise ValueError(f"Unknown trace: {trace_name}")


def build_trace_plan(
    base: Any,
    chunks: Sequence[Any],
    query_pools: Mapping[str, Mapping[str, List[Any]]],
    assignments: pd.DataFrame,
    trace_name: str,
    trace_seed: int,
) -> LightweightTracePlan:
    ordered_chunks = sorted(chunks, key=lambda chunk: chunk.chunk_id)
    chunk_ids = tuple(chunk.chunk_id for chunk in ordered_chunks)
    n_chunks = len(chunk_ids)
    if n_chunks != N_SERIES and not QUICK_TEST:
        raise AssertionError(f"Expected {N_SERIES} chunks, got {n_chunks}")

    group_by_position = [
        group_for_chunk(assignments, chunk_id, trace_name)
        for chunk_id in chunk_ids
    ]

    local_counts: List[int] = []
    strides: List[int] = []
    maximum_rounds = 0
    for group in group_by_position:
        rounds, stride = local_access_schedule(trace_name, group)
        if QUICK_TEST:
            rounds = max(160, rounds // 20)
            # Keep a 4:1 frequency contrast in quick mode.
            stride = min(stride, 4)
        maximum_rounds = max(maximum_rounds, rounds)
        strides.append(stride)
        local_counts.append((rounds + stride - 1) // stride)

    workload_names = tuple(sorted(base.WORKLOADS))
    workload_to_position = {
        name: index for index, name in enumerate(workload_names)
    }
    query_class_to_position = {
        name: index for index, name in enumerate(QUERY_CLASSES)
    }

    local_query_classes: List[np.ndarray] = []
    local_query_choices: List[np.ndarray] = []
    local_workloads: List[np.ndarray] = []

    for position, chunk_id in enumerate(chunk_ids):
        total_local = local_counts[position]
        group = group_by_position[position]
        rng = np.random.default_rng(
            stable_seed(
                trace_seed,
                chunk_id,
                trace_name,
                "local_query_sequence",
            )
        )
        qclasses = np.empty(total_local, dtype=np.uint8)
        qchoices = np.empty(total_local, dtype=np.uint16)
        workloads = np.empty(total_local, dtype=np.uint8)

        for local_index in range(total_local):
            workload = workload_for_local_access(
                trace_name,
                group,
                local_index,
                total_local,
            )
            weights = base.WORKLOADS[workload]
            classes = [
                query_class
                for query_class in QUERY_CLASSES
                if float(weights.get(query_class, 0.0)) > 0.0
            ]
            probabilities = np.asarray(
                [float(weights[query_class]) for query_class in classes],
                dtype=float,
            )
            probabilities /= probabilities.sum()
            query_class = str(rng.choice(classes, p=probabilities))
            pool = query_pools[chunk_id][query_class]
            choice = int(rng.integers(0, len(pool)))

            qclasses[local_index] = query_class_to_position[query_class]
            qchoices[local_index] = choice
            workloads[local_index] = workload_to_position[workload]

        local_query_classes.append(qclasses)
        local_query_choices.append(qchoices)
        local_workloads.append(workloads)

    chunk_positions: List[int] = []
    local_indices: List[int] = []
    qclass_positions: List[int] = []
    qchoice_positions: List[int] = []
    workload_positions: List[int] = []
    next_local = [0 for _ in range(n_chunks)]
    global_rng = np.random.default_rng(
        stable_seed(trace_seed, trace_name, "global_round_order")
    )

    for global_round in range(maximum_rounds):
        eligible = [
            position
            for position in range(n_chunks)
            if global_round % strides[position] == 0
            and next_local[position] < local_counts[position]
        ]
        if not eligible:
            continue
        order = np.asarray(eligible, dtype=int)
        global_rng.shuffle(order)
        for position_value in order:
            position = int(position_value)
            local_index = next_local[position]
            chunk_positions.append(position)
            local_indices.append(local_index)
            qclass_positions.append(
                int(local_query_classes[position][local_index])
            )
            qchoice_positions.append(
                int(local_query_choices[position][local_index])
            )
            workload_positions.append(
                int(local_workloads[position][local_index])
            )
            next_local[position] += 1

    if next_local != local_counts:
        raise AssertionError(
            f"Trace construction incomplete: expected={local_counts}, observed={next_local}"
        )

    return LightweightTracePlan(
        trace_name=trace_name,
        trace_seed=int(trace_seed),
        chunk_size=int(ordered_chunks[0].chunk_size),
        chunk_ids=chunk_ids,
        chunk_positions=np.asarray(chunk_positions, dtype=np.int16),
        local_indices=np.asarray(local_indices, dtype=np.int32),
        query_class_positions=np.asarray(qclass_positions, dtype=np.uint8),
        query_choice_positions=np.asarray(qchoice_positions, dtype=np.uint16),
        workload_positions=np.asarray(workload_positions, dtype=np.uint8),
        workload_names=workload_names,
    )


def measure_directed_migrations(
    e1: Any,
    base: Any,
    reps: Mapping[str, Any],
    chunks_by_id: Mapping[str, Any],
    stored_by_chunk: Mapping[str, Mapping[str, Any]],
    migration_root: Path,
) -> Tuple[
    Dict[str, Dict[Tuple[str, str], int]],
    Dict[str, Dict[Tuple[str, str], int]],
    List[Dict[str, Any]],
]:
    cost_map: Dict[str, Dict[Tuple[str, str], int]] = {}
    byte_map: Dict[str, Dict[Tuple[str, str], int]] = {}
    rows: List[Dict[str, Any]] = []

    for chunk_index, (chunk_id, chunk) in enumerate(
        sorted(chunks_by_id.items()), start=1
    ):
        stored_map = stored_by_chunk[chunk_id]
        codecs = sorted(stored_map)
        cost_map[chunk_id] = {}
        byte_map[chunk_id] = {}

        for current_codec in codecs:
            for target_codec in codecs:
                key = (current_codec, target_codec)
                if current_codec == target_codec:
                    cost_map[chunk_id][key] = 0
                    byte_map[chunk_id][key] = 0
                    rows.append(
                        {
                            "chunk_id": chunk_id,
                            "series_index": int(chunk.series_index),
                            "chunk_size": int(chunk.chunk_size),
                            "current_codec": current_codec,
                            "target_codec": target_codec,
                            "median_migration_ns": 0,
                            "minimum_migration_ns": 0,
                            "maximum_migration_ns": 0,
                            "rewrite_bytes": 0,
                            "n_repeats": 0,
                        }
                    )
                    continue

                times: List[int] = []
                bytes_values: List[int] = []
                for repeat in range(MIGRATION_MEASURE_REPEATS):
                    migration_ns, rewrite_bytes = e1.perform_actual_migration(
                        base,
                        reps,
                        stored_map,
                        chunk,
                        current_codec,
                        target_codec,
                        migration_root / chunk_id,
                        (
                            f"measure__{current_codec}__to__"
                            f"{target_codec}__r{repeat}"
                        ),
                    )
                    times.append(int(migration_ns))
                    bytes_values.append(int(rewrite_bytes))

                median_ns = int(
                    np.median(np.asarray(times, dtype=np.int64))
                )
                median_bytes = int(
                    np.median(np.asarray(bytes_values, dtype=np.int64))
                )
                cost_map[chunk_id][key] = median_ns
                byte_map[chunk_id][key] = median_bytes
                rows.append(
                    {
                        "chunk_id": chunk_id,
                        "series_index": int(chunk.series_index),
                        "chunk_size": int(chunk.chunk_size),
                        "current_codec": current_codec,
                        "target_codec": target_codec,
                        "median_migration_ns": median_ns,
                        "minimum_migration_ns": int(min(times)),
                        "maximum_migration_ns": int(max(times)),
                        "rewrite_bytes": median_bytes,
                        "n_repeats": int(MIGRATION_MEASURE_REPEATS),
                    }
                )

        if (
            chunk_index == 1
            or chunk_index == len(chunks_by_id)
            or chunk_index % max(1, len(chunks_by_id) // 10) == 0
        ):
            print(
                f"[migration calibration] {chunk_index}/{len(chunks_by_id)} chunks"
            )

    return cost_map, byte_map, rows


def profile_arrays(
    profiles: Mapping[str, Mapping[str, Mapping[str, float]]],
    chunk_ids: Sequence[str],
    codecs: Sequence[str],
) -> np.ndarray:
    array = np.empty(
        (len(chunk_ids), len(codecs), len(QUERY_CLASSES)),
        dtype=np.float64,
    )
    for chunk_position, chunk_id in enumerate(chunk_ids):
        for codec_position, codec in enumerate(codecs):
            for query_position, query_class in enumerate(QUERY_CLASSES):
                array[chunk_position, codec_position, query_position] = float(
                    profiles[chunk_id][codec][query_class]
                )
    return array


def scan_best_codec_positions(
    e1: Any,
    base: Any,
    profiles: Mapping[str, Mapping[str, Mapping[str, float]]],
    chunk_ids: Sequence[str],
    codecs: Sequence[str],
) -> np.ndarray:
    codec_to_position = {
        codec: index for index, codec in enumerate(codecs)
    }
    positions = []
    for chunk_id in chunk_ids:
        codec, _ = e1.choose_best_codec(
            profiles[chunk_id], base.WORKLOADS["scan_heavy"]
        )
        positions.append(codec_to_position[codec])
    return np.asarray(positions, dtype=np.int16)


def trace_query_count_matrix(
    plan: LightweightTracePlan,
    n_chunks: int,
) -> np.ndarray:
    counts = np.zeros(
        (n_chunks, len(QUERY_CLASSES)), dtype=np.int64
    )
    np.add.at(
        counts,
        (
            plan.chunk_positions.astype(int),
            plan.query_class_positions.astype(int),
        ),
        1,
    )
    return counts


def simulate_initial_best(
    plan: LightweightTracePlan,
    profiles_array: np.ndarray,
    initial_codec_positions: np.ndarray,
) -> Dict[str, Any]:
    counts = trace_query_count_matrix(plan, profiles_array.shape[0])
    query_cost = float(
        np.sum(
            counts
            * profiles_array[
                np.arange(profiles_array.shape[0]),
                initial_codec_positions,
                :,
            ]
        )
    )
    return {
        "query_time_ns": query_cost,
        "migration_time_ns": 0.0,
        "total_time_ns": query_cost,
        "migration_count": 0,
        "rewrite_bytes": 0,
        "per_chunk_query_time_ns": np.sum(
            counts
            * profiles_array[
                np.arange(profiles_array.shape[0]),
                initial_codec_positions,
                :,
            ],
            axis=1,
        ),
        "per_chunk_migration_time_ns": np.zeros(
            profiles_array.shape[0], dtype=float
        ),
        "per_chunk_migration_count": np.zeros(
            profiles_array.shape[0], dtype=int
        ),
        "per_chunk_rewrite_bytes": np.zeros(
            profiles_array.shape[0], dtype=int
        ),
    }


def classify_hot_chunks(
    scores: np.ndarray,
    hot_fraction: float,
) -> np.ndarray:
    """
    Percentile-based hot classification using a strict threshold.

    Ties at the threshold remain cold. Consequently, when every chunk has the
    same access frequency, AF-HC correctly has no evidence that any subset is
    hotter than another and performs no arbitrary tie-breaking migration.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return np.zeros(0, dtype=bool)
    threshold = float(
        np.quantile(scores, 1.0 - float(hot_fraction))
    )
    return scores > threshold


def simulate_afhc(
    plan: LightweightTracePlan,
    profiles_array: np.ndarray,
    codecs: Sequence[str],
    migration_costs: Mapping[str, Mapping[Tuple[str, str], int]],
    migration_bytes: Mapping[str, Mapping[Tuple[str, str], int]],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    n_chunks = len(plan.chunk_ids)
    codec_to_position = {
        codec: index for index, codec in enumerate(codecs)
    }
    cold_codec = str(config["cold_codec"])
    hot_codec = str(config["hot_codec"])
    cold_position = codec_to_position[cold_codec]
    hot_position = codec_to_position[hot_codec]
    period = int(config["adaptation_period"])
    hot_fraction = float(config["hot_fraction"])

    active = np.full(n_chunks, cold_position, dtype=np.int16)
    scores = np.zeros(n_chunks, dtype=np.float64)
    per_chunk_query = np.zeros(n_chunks, dtype=np.float64)
    per_chunk_migration = np.zeros(n_chunks, dtype=np.float64)
    per_chunk_migration_count = np.zeros(n_chunks, dtype=np.int64)
    per_chunk_rewrite = np.zeros(n_chunks, dtype=np.int64)

    total_events = plan.n_events
    for start in range(0, total_events, period):
        end = min(total_events, start + period)
        chunk_segment = plan.chunk_positions[start:end].astype(int)
        query_segment = plan.query_class_positions[start:end].astype(int)

        counts = np.zeros(
            (n_chunks, len(QUERY_CLASSES)), dtype=np.int64
        )
        np.add.at(counts, (chunk_segment, query_segment), 1)
        costs = np.sum(
            counts
            * profiles_array[
                np.arange(n_chunks), active, :
            ],
            axis=1,
        )
        per_chunk_query += costs

        accesses = np.bincount(
            chunk_segment, minlength=n_chunks
        ).astype(float)
        scores = (
            FREQUENCY_EMA_ALPHA * accesses
            + (1.0 - FREQUENCY_EMA_ALPHA) * scores
        )
        hot_mask = classify_hot_chunks(scores, hot_fraction)
        target = np.where(hot_mask, hot_position, cold_position)

        for chunk_position in np.flatnonzero(target != active):
            chunk_id = plan.chunk_ids[int(chunk_position)]
            old_codec = codecs[int(active[chunk_position])]
            new_codec = codecs[int(target[chunk_position])]
            key = (old_codec, new_codec)
            per_chunk_migration[chunk_position] += float(
                migration_costs[chunk_id][key]
            )
            per_chunk_migration_count[chunk_position] += 1
            per_chunk_rewrite[chunk_position] += int(
                migration_bytes[chunk_id][key]
            )
            active[chunk_position] = target[chunk_position]

    total_query = float(per_chunk_query.sum())
    total_migration = float(per_chunk_migration.sum())
    return {
        "query_time_ns": total_query,
        "migration_time_ns": total_migration,
        "total_time_ns": total_query + total_migration,
        "migration_count": int(per_chunk_migration_count.sum()),
        "rewrite_bytes": int(per_chunk_rewrite.sum()),
        "per_chunk_query_time_ns": per_chunk_query,
        "per_chunk_migration_time_ns": per_chunk_migration,
        "per_chunk_migration_count": per_chunk_migration_count,
        "per_chunk_rewrite_bytes": per_chunk_rewrite,
    }


def decide_repressdb_model(
    e1: Any,
    e1v2: Any,
    profile: Mapping[str, Mapping[str, float]],
    state: RepressState,
    local_query_index: int,
    encode_ns: Mapping[str, int],
    decode_ns: Mapping[str, int],
) -> Tuple[Optional[str], Dict[str, Any]]:
    recent_long = list(state.recent_classes)
    recent_short = recent_long[-SHORT_WINDOW:]
    long_weights = e1v2.observed_weights(
        recent_long, e1.QUERY_CLASSES
    )
    short_weights = e1v2.observed_weights(
        recent_short, e1.QUERY_CLASSES
    )

    long_target, long_target_cost = e1.choose_best_codec(
        profile, long_weights
    )
    short_target, short_target_cost = e1.choose_best_codec(
        profile, short_weights
    )
    current_long_cost = e1.workload_cost(
        profile, state.active_codec, long_weights
    )
    current_short_cost = e1.workload_cost(
        profile, state.active_codec, short_weights
    )
    long_relative = (
        (current_long_cost - long_target_cost) / current_long_cost
        if current_long_cost > 0
        else 0.0
    )
    short_relative = (
        (current_short_cost - short_target_cost) / current_short_cost
        if current_short_cost > 0
        else 0.0
    )
    conservative_relative = min(long_relative, short_relative)
    conservative_gain = max(
        0.0,
        min(
            current_long_cost - long_target_cost,
            current_short_cost - short_target_cost,
        ),
    )
    target = long_target
    rewrite_estimate = e1.migration_estimate_ns(
        state.active_codec, target, encode_ns, decode_ns
    )
    predicted_saving = conservative_gain * PREDICTION_HORIZON

    detail = {
        "current_expected_ns": current_long_cost,
        "target_expected_ns": long_target_cost,
        "relative_improvement": conservative_relative,
        "predicted_saving_ns": predicted_saving,
        "rewrite_estimate_ns": rewrite_estimate,
        "reason": "",
    }

    if long_target != short_target:
        state.pending_candidate = None
        state.pending_rounds = 0
        detail["reason"] = "timescale_disagreement"
        return None, detail
    if target == state.active_codec:
        state.pending_candidate = None
        state.pending_rounds = 0
        detail["reason"] = "current_layout_is_best"
        return None, detail
    if (
        local_query_index - state.last_switch_local_index
        < COOLDOWN_QUERIES
    ):
        state.pending_candidate = None
        state.pending_rounds = 0
        detail["reason"] = "cooldown"
        return None, detail
    if conservative_relative < MIN_RELATIVE_IMPROVEMENT:
        state.pending_candidate = None
        state.pending_rounds = 0
        detail["reason"] = "relative_gain_too_small"
        return None, detail
    if (
        predicted_saving
        < REWRITE_SAFETY_FACTOR * rewrite_estimate
    ):
        state.pending_candidate = None
        state.pending_rounds = 0
        detail["reason"] = "break_even_not_met"
        return None, detail

    if state.pending_candidate == target:
        state.pending_rounds += 1
    else:
        state.pending_candidate = target
        state.pending_rounds = 1

    if state.pending_rounds < CONFIRMATION_ROUNDS:
        detail["reason"] = "awaiting_confirmation"
        return None, detail

    state.pending_candidate = None
    state.pending_rounds = 0
    detail["reason"] = "switch"
    return target, detail


def simulate_repressdb(
    e1: Any,
    e1v2: Any,
    base: Any,
    plan: LightweightTracePlan,
    profiles: Mapping[str, Mapping[str, Mapping[str, float]]],
    encode_ns_by_chunk: Mapping[str, Mapping[str, int]],
    decode_ns_by_chunk: Mapping[str, Mapping[str, int]],
    migration_costs: Mapping[str, Mapping[Tuple[str, str], int]],
    migration_bytes: Mapping[str, Mapping[Tuple[str, str], int]],
) -> Dict[str, Any]:
    states: List[RepressState] = []
    for chunk_id in plan.chunk_ids:
        initial_codec, _ = e1.choose_best_codec(
            profiles[chunk_id], base.WORKLOADS["scan_heavy"]
        )
        states.append(RepressState(active_codec=initial_codec))

    for event_index in range(plan.n_events):
        position = int(plan.chunk_positions[event_index])
        query_position = int(
            plan.query_class_positions[event_index]
        )
        query_class = QUERY_CLASSES[query_position]
        state = states[position]
        chunk_id = plan.chunk_ids[position]

        state.query_cost_ns += float(
            profiles[chunk_id][state.active_codec][query_class]
        )
        state.recent_classes.append(query_class)
        state.local_access_count += 1

        if (
            state.local_access_count >= MIN_OBSERVATIONS
            and state.local_access_count % DECISION_INTERVAL == 0
        ):
            local_query_index = state.local_access_count - 1
            target, _ = decide_repressdb_model(
                e1,
                e1v2,
                profiles[chunk_id],
                state,
                local_query_index,
                encode_ns_by_chunk[chunk_id],
                decode_ns_by_chunk[chunk_id],
            )
            if target is not None and target != state.active_codec:
                key = (state.active_codec, target)
                state.migration_cost_ns += float(
                    migration_costs[chunk_id][key]
                )
                state.migration_count += 1
                state.rewrite_bytes += int(
                    migration_bytes[chunk_id][key]
                )
                state.active_codec = target
                state.last_switch_local_index = local_query_index

    per_query = np.asarray(
        [state.query_cost_ns for state in states], dtype=float
    )
    per_migration = np.asarray(
        [state.migration_cost_ns for state in states], dtype=float
    )
    per_count = np.asarray(
        [state.migration_count for state in states], dtype=int
    )
    per_bytes = np.asarray(
        [state.rewrite_bytes for state in states], dtype=int
    )
    total_query = float(per_query.sum())
    total_migration = float(per_migration.sum())
    return {
        "query_time_ns": total_query,
        "migration_time_ns": total_migration,
        "total_time_ns": total_query + total_migration,
        "migration_count": int(per_count.sum()),
        "rewrite_bytes": int(per_bytes.sum()),
        "per_chunk_query_time_ns": per_query,
        "per_chunk_migration_time_ns": per_migration,
        "per_chunk_migration_count": per_count,
        "per_chunk_rewrite_bytes": per_bytes,
    }


def config_grid() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    config_index = 0
    for period in ADAPTATION_PERIODS:
        for hot_fraction in HOT_FRACTIONS:
            for cold_codec in COLD_CODECS:
                for hot_codec in HOT_CODECS:
                    config_id = f"AFHC_{config_index:03d}"
                    rows.append(
                        {
                            "config_id": config_id,
                            "adaptation_period": int(period),
                            "hot_fraction": float(hot_fraction),
                            "cold_codec": cold_codec,
                            "hot_codec": hot_codec,
                            "frequency_ema_alpha": FREQUENCY_EMA_ALPHA,
                            "hot_rule": (
                                "score > percentile(score, 1-hot_fraction); "
                                "ties at threshold remain cold"
                            ),
                        }
                    )
                    config_index += 1
    if len(rows) != 72:
        raise AssertionError(f"Expected 72 configs, got {len(rows)}")
    return rows


def summarize_grid(
    grid_run_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    safety_rows: List[Dict[str, Any]] = []

    for config_id, group in grid_run_df.groupby("config_id", sort=True):
        first = group.iloc[0]
        stable = group[group["trace"] == "stable_scan"]
        burst = group[group["trace"] == "brief_burst"]

        stable_ratio = float(stable["ratio_to_initial_best"].median())
        burst_ratio = float(burst["ratio_to_initial_best"].median())
        burst_migrations_per_chunk = float(
            burst["median_migrations_per_chunk"].median()
        )
        score = float(group["ratio_to_initial_best"].median())
        median_migrations = float(
            group["median_migrations_per_chunk"].median()
        )
        median_rewrite_bytes = float(group["rewrite_bytes"].median())

        failures: List[str] = []
        if stable_ratio > MAX_STABLE_RATIO:
            failures.append("stable_ratio_above_1.05")
        if burst_ratio > MAX_BURST_RATIO:
            failures.append("burst_ratio_above_1.05")
        if (
            burst_migrations_per_chunk
            > MAX_BURST_MEDIAN_MIGRATIONS_PER_CHUNK
        ):
            failures.append("burst_migrations_per_chunk_above_2")

        row = {
            "config_id": config_id,
            "adaptation_period": int(first["adaptation_period"]),
            "hot_fraction": float(first["hot_fraction"]),
            "cold_codec": str(first["cold_codec"]),
            "hot_codec": str(first["hot_codec"]),
            "composite_median_ratio": score,
            "stable_median_ratio": stable_ratio,
            "burst_median_ratio": burst_ratio,
            "burst_median_migrations_per_chunk": (
                burst_migrations_per_chunk
            ),
            "overall_median_migrations_per_chunk": median_migrations,
            "overall_median_rewrite_bytes": median_rewrite_bytes,
            "safety_pass": len(failures) == 0,
            "failure_reasons": "|".join(failures),
        }
        rows.append(row)
        if failures:
            safety_rows.append(row)

    summary = pd.DataFrame(rows)
    safe = summary[summary["safety_pass"]].copy()
    if safe.empty:
        raise RuntimeError(
            "No AF-HC configuration passed the predeclared safety filter."
        )

    # Primary order. A second pass applies the 0.5% score-equivalence rule.
    safe = safe.sort_values(
        [
            "composite_median_ratio",
            "overall_median_migrations_per_chunk",
            "overall_median_rewrite_bytes",
            "adaptation_period",
            "config_id",
        ],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)
    best_score = float(safe.iloc[0]["composite_median_ratio"])
    equivalent = safe[
        safe["composite_median_ratio"]
        <= best_score * (1.0 + TIE_SCORE_TOLERANCE)
    ].copy()
    equivalent = equivalent.sort_values(
        [
            "overall_median_migrations_per_chunk",
            "overall_median_rewrite_bytes",
            "adaptation_period",
            "config_id",
        ],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    selected = equivalent.iloc[0].to_dict()

    summary = summary.merge(
        safe[["config_id"]].reset_index().rename(
            columns={"index": "safe_rank_zero_based"}
        ),
        on="config_id",
        how="left",
    )
    summary["selected"] = summary["config_id"] == selected["config_id"]

    return (
        summary.sort_values("config_id").reset_index(drop=True),
        pd.DataFrame(safety_rows),
        safe,
        selected,
    )


def actual_query_for_event(
    plan: LightweightTracePlan,
    event_index: int,
    query_pools: Mapping[str, Mapping[str, List[Any]]],
) -> Tuple[str, str, Any]:
    position = int(plan.chunk_positions[event_index])
    chunk_id = plan.chunk_ids[position]
    query_class = QUERY_CLASSES[
        int(plan.query_class_positions[event_index])
    ]
    choice = int(plan.query_choice_positions[event_index])
    query = query_pools[chunk_id][query_class][choice]
    return chunk_id, query_class, query


def execute_actual_migration(
    e1: Any,
    base: Any,
    reps: Mapping[str, Any],
    stored_map: Mapping[str, Any],
    chunk: Any,
    old_codec: str,
    target_codec: str,
    migration_root: Path,
    token: str,
) -> Tuple[int, int]:
    return e1.perform_actual_migration(
        base,
        reps,
        stored_map,
        chunk,
        old_codec,
        target_codec,
        migration_root,
        token,
    )


def run_actual_policy(
    e1: Any,
    e1v2: Any,
    base: Any,
    reps: Mapping[str, Any],
    chunks_by_id: Mapping[str, Any],
    stored_by_chunk: Mapping[str, Mapping[str, Any]],
    profiles: Mapping[str, Mapping[str, Mapping[str, float]]],
    encode_ns_by_chunk: Mapping[str, Mapping[str, int]],
    decode_ns_by_chunk: Mapping[str, Mapping[str, int]],
    query_pools: Mapping[str, Mapping[str, List[Any]]],
    plan: LightweightTracePlan,
    policy: str,
    frozen_config: Mapping[str, Any],
    migration_root: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    mode = base.CACHE_MODES["warm_slicedmeta"]
    codecs = tuple(sorted(reps))
    chunk_ids = plan.chunk_ids

    if policy == "initial_best":
        active: Dict[str, str] = {
            chunk_id: e1.choose_best_codec(
                profiles[chunk_id], base.WORKLOADS["scan_heavy"]
            )[0]
            for chunk_id in chunk_ids
        }
    elif policy == "afhc_frozen":
        active = {
            chunk_id: str(frozen_config["cold_codec"])
            for chunk_id in chunk_ids
        }
    elif policy == "repressdb":
        active = {
            chunk_id: e1.choose_best_codec(
                profiles[chunk_id], base.WORKLOADS["scan_heavy"]
            )[0]
            for chunk_id in chunk_ids
        }
    else:
        raise ValueError(f"Unknown audit policy: {policy}")

    per_chunk_query = {chunk_id: 0 for chunk_id in chunk_ids}
    per_chunk_migration = {chunk_id: 0 for chunk_id in chunk_ids}
    per_chunk_migrations = {chunk_id: 0 for chunk_id in chunk_ids}
    per_chunk_rewrite = {chunk_id: 0 for chunk_id in chunk_ids}
    per_chunk_latencies: Dict[str, List[int]] = {
        chunk_id: [] for chunk_id in chunk_ids
    }
    event_rows: List[Dict[str, Any]] = []

    repress_states: Dict[str, RepressState] = {}
    if policy == "repressdb":
        repress_states = {
            chunk_id: RepressState(active_codec=active[chunk_id])
            for chunk_id in chunk_ids
        }

    afhc_scores = np.zeros(len(chunk_ids), dtype=float)
    afhc_interval_counts = np.zeros(len(chunk_ids), dtype=float)
    chunk_to_position = {
        chunk_id: index for index, chunk_id in enumerate(chunk_ids)
    }
    period = int(frozen_config["adaptation_period"])
    hot_fraction = float(frozen_config["hot_fraction"])
    hot_codec = str(frozen_config["hot_codec"])
    cold_codec = str(frozen_config["cold_codec"])

    with MultiChunkHandles(stored_by_chunk) as handles:
        # Warm every chunk/codec with the first trace query for that chunk.
        first_event_by_chunk: Dict[str, int] = {}
        for event_index in range(plan.n_events):
            chunk_id = plan.chunk_ids[
                int(plan.chunk_positions[event_index])
            ]
            first_event_by_chunk.setdefault(chunk_id, event_index)
            if len(first_event_by_chunk) == len(chunk_ids):
                break

        for chunk_id, event_index in first_event_by_chunk.items():
            _, _, query = actual_query_for_event(
                plan, event_index, query_pools
            )
            chunk = chunks_by_id[chunk_id]
            truth = base.execute_on_array(chunk.values, query)
            for codec, rep in reps.items():
                result, _ = base.run_query_once(
                    rep,
                    stored_by_chunk[chunk_id][codec],
                    query,
                    mode,
                    handles.payload[chunk_id][codec],
                    handles.metadata[chunk_id][codec],
                )
                if not base.results_equal(result, truth):
                    raise AssertionError(
                        f"Warm-up query failed: {chunk_id}, {codec}"
                    )

        for event_index in range(plan.n_events):
            chunk_id, query_class, query = actual_query_for_event(
                plan, event_index, query_pools
            )
            chunk = chunks_by_id[chunk_id]
            codec = active[chunk_id]
            rep = reps[codec]
            stored = stored_by_chunk[chunk_id][codec]

            start_ns = time.perf_counter_ns()
            result, stats = base.run_query_once(
                rep,
                stored,
                query,
                mode,
                handles.payload[chunk_id][codec],
                handles.metadata[chunk_id][codec],
            )
            latency_ns = time.perf_counter_ns() - start_ns
            truth = base.execute_on_array(chunk.values, query)
            if not base.results_equal(result, truth):
                raise AssertionError(
                    f"Actual audit query incorrect: policy={policy}, "
                    f"chunk={chunk_id}, event={event_index}, codec={codec}"
                )

            per_chunk_query[chunk_id] += int(latency_ns)
            per_chunk_latencies[chunk_id].append(int(latency_ns))

            if policy == "repressdb":
                state = repress_states[chunk_id]
                state.recent_classes.append(query_class)
                state.local_access_count += 1
                if (
                    state.local_access_count >= MIN_OBSERVATIONS
                    and state.local_access_count % DECISION_INTERVAL == 0
                ):
                    local_query_index = state.local_access_count - 1
                    target, detail = decide_repressdb_model(
                        e1,
                        e1v2,
                        profiles[chunk_id],
                        state,
                        local_query_index,
                        encode_ns_by_chunk[chunk_id],
                        decode_ns_by_chunk[chunk_id],
                    )
                    if target is not None and target != state.active_codec:
                        old_codec = state.active_codec
                        migration_ns, rewrite_bytes = execute_actual_migration(
                            e1,
                            base,
                            reps,
                            stored_by_chunk[chunk_id],
                            chunk,
                            old_codec,
                            target,
                            migration_root / chunk_id,
                            (
                                f"{plan.trace_name}__seed{plan.trace_seed}__"
                                f"{policy}__event{event_index}__"
                                f"m{per_chunk_migrations[chunk_id]}"
                            ),
                        )
                        active[chunk_id] = target
                        state.active_codec = target
                        state.last_switch_local_index = local_query_index
                        per_chunk_migration[chunk_id] += int(migration_ns)
                        per_chunk_migrations[chunk_id] += 1
                        per_chunk_rewrite[chunk_id] += int(rewrite_bytes)
                        event_rows.append(
                            {
                                "policy": policy,
                                "trace": plan.trace_name,
                                "trace_seed": int(plan.trace_seed),
                                "chunk_size": int(plan.chunk_size),
                                "global_event_index": int(event_index),
                                "chunk_id": chunk_id,
                                "local_query_index": int(local_query_index),
                                "old_codec": old_codec,
                                "target_codec": target,
                                "reason": detail["reason"],
                                "actual_migration_ns": int(migration_ns),
                                "rewrite_bytes": int(rewrite_bytes),
                            }
                        )

            elif policy == "afhc_frozen":
                afhc_interval_counts[
                    chunk_to_position[chunk_id]
                ] += 1.0
                is_boundary = (
                    (event_index + 1) % period == 0
                    or event_index == plan.n_events - 1
                )
                if is_boundary:
                    afhc_scores = (
                        FREQUENCY_EMA_ALPHA * afhc_interval_counts
                        + (1.0 - FREQUENCY_EMA_ALPHA) * afhc_scores
                    )
                    hot_mask = classify_hot_chunks(
                        afhc_scores, hot_fraction
                    )
                    for position, candidate_chunk_id in enumerate(chunk_ids):
                        target = hot_codec if hot_mask[position] else cold_codec
                        if target == active[candidate_chunk_id]:
                            continue
                        old_codec = active[candidate_chunk_id]
                        candidate_chunk = chunks_by_id[candidate_chunk_id]
                        migration_ns, rewrite_bytes = execute_actual_migration(
                            e1,
                            base,
                            reps,
                            stored_by_chunk[candidate_chunk_id],
                            candidate_chunk,
                            old_codec,
                            target,
                            migration_root / candidate_chunk_id,
                            (
                                f"{plan.trace_name}__seed{plan.trace_seed}__"
                                f"{policy}__event{event_index}__"
                                f"m{per_chunk_migrations[candidate_chunk_id]}"
                            ),
                        )
                        active[candidate_chunk_id] = target
                        per_chunk_migration[candidate_chunk_id] += int(
                            migration_ns
                        )
                        per_chunk_migrations[candidate_chunk_id] += 1
                        per_chunk_rewrite[candidate_chunk_id] += int(
                            rewrite_bytes
                        )
                        event_rows.append(
                            {
                                "policy": policy,
                                "trace": plan.trace_name,
                                "trace_seed": int(plan.trace_seed),
                                "chunk_size": int(plan.chunk_size),
                                "global_event_index": int(event_index),
                                "chunk_id": candidate_chunk_id,
                                "local_query_index": math.nan,
                                "old_codec": old_codec,
                                "target_codec": target,
                                "reason": "frequency_percentile_reclassification",
                                "actual_migration_ns": int(migration_ns),
                                "rewrite_bytes": int(rewrite_bytes),
                            }
                        )
                    afhc_interval_counts[:] = 0.0

            # Keep a sparse execution diagnostic instead of a huge per-query log.
            if (
                event_index % 5000 == 0
                or event_index == plan.n_events - 1
            ):
                event_rows.append(
                    {
                        "policy": policy,
                        "trace": plan.trace_name,
                        "trace_seed": int(plan.trace_seed),
                        "chunk_size": int(plan.chunk_size),
                        "global_event_index": int(event_index),
                        "chunk_id": chunk_id,
                        "local_query_index": int(
                            plan.local_indices[event_index]
                        ),
                        "old_codec": codec,
                        "target_codec": active[chunk_id],
                        "reason": "query_checkpoint",
                        "actual_migration_ns": 0,
                        "rewrite_bytes": 0,
                        "query_class": query_class,
                        "workload": plan.workload_names[
                            int(plan.workload_positions[event_index])
                        ],
                        "query_latency_ns": int(latency_ns),
                        "payload_bytes": int(stats.payload_bytes),
                        "metadata_bytes": int(stats.metadata_bytes),
                    }
                )

    chunk_rows: List[Dict[str, Any]] = []
    all_latencies: List[int] = []
    for chunk_id in chunk_ids:
        latencies = np.asarray(
            per_chunk_latencies[chunk_id], dtype=np.int64
        )
        all_latencies.extend(per_chunk_latencies[chunk_id])
        query_ns = int(per_chunk_query[chunk_id])
        migration_ns = int(per_chunk_migration[chunk_id])
        chunk = chunks_by_id[chunk_id]
        chunk_rows.append(
            {
                "policy": policy,
                "trace": plan.trace_name,
                "trace_seed": int(plan.trace_seed),
                "chunk_id": chunk_id,
                "series_index": int(chunk.series_index),
                "chunk_size": int(chunk.chunk_size),
                "n_queries": int(latencies.size),
                "query_time_ns": query_ns,
                "migration_time_ns": migration_ns,
                "total_time_ns": query_ns + migration_ns,
                "median_query_latency_ns": float(np.median(latencies)),
                "p95_query_latency_ns": float(np.quantile(latencies, 0.95)),
                "p99_query_latency_ns": float(np.quantile(latencies, 0.99)),
                "migration_count": int(per_chunk_migrations[chunk_id]),
                "rewrite_bytes": int(per_chunk_rewrite[chunk_id]),
                "initial_codec": (
                    str(frozen_config["cold_codec"])
                    if policy == "afhc_frozen"
                    else e1.choose_best_codec(
                        profiles[chunk_id],
                        base.WORKLOADS["scan_heavy"],
                    )[0]
                ),
                "final_codec": active[chunk_id],
            }
        )

    all_latency_array = np.asarray(all_latencies, dtype=np.int64)
    total_query = int(sum(per_chunk_query.values()))
    total_migration = int(sum(per_chunk_migration.values()))
    total_time = total_query + total_migration
    summary = {
        "policy": policy,
        "trace": plan.trace_name,
        "trace_seed": int(plan.trace_seed),
        "chunk_size": int(plan.chunk_size),
        "n_chunks": int(len(chunk_ids)),
        "n_queries": int(plan.n_events),
        "query_time_ns": total_query,
        "migration_time_ns": total_migration,
        "total_time_ns": total_time,
        "throughput_qps": float(
            plan.n_events / (total_time / 1e9)
            if total_time > 0
            else math.inf
        ),
        "median_query_latency_ns": float(
            np.median(all_latency_array)
        ),
        "p95_query_latency_ns": float(
            np.quantile(all_latency_array, 0.95)
        ),
        "p99_query_latency_ns": float(
            np.quantile(all_latency_array, 0.99)
        ),
        "migration_count": int(sum(per_chunk_migrations.values())),
        "rewrite_bytes": int(sum(per_chunk_rewrite.values())),
    }
    return summary, chunk_rows, event_rows


def bootstrap_median_ci(
    values: np.ndarray,
    seed: int,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=float)
    for index in range(iterations):
        samples[index] = np.median(
            rng.choice(values, size=values.size, replace=True)
        )
    return (
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def actual_clustered_comparisons(
    chunk_run_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    series_df = (
        chunk_run_df.groupby(
            [
                "trace",
                "chunk_size",
                "series_index",
                "policy",
            ],
            as_index=False,
        )["total_time_ns"]
        .median()
    )

    for (trace, chunk_size), group in series_df.groupby(
        ["trace", "chunk_size"], sort=True
    ):
        pivot = group.pivot(
            index="series_index",
            columns="policy",
            values="total_time_ns",
        ).dropna()
        for policy, reference in (
            ("repressdb", "afhc_frozen"),
            ("repressdb", "initial_best"),
            ("afhc_frozen", "initial_best"),
        ):
            if policy not in pivot or reference not in pivot:
                continue
            policy_values = pivot[policy].to_numpy(dtype=float)
            reference_values = pivot[reference].to_numpy(dtype=float)
            ratios = policy_values / reference_values
            try:
                p_value = float(
                    wilcoxon(
                        policy_values,
                        reference_values,
                        alternative="two-sided",
                        zero_method="wilcox",
                    ).pvalue
                )
            except ValueError:
                p_value = 1.0
            ci_low, ci_high = bootstrap_median_ci(
                ratios,
                stable_seed(
                    SEED,
                    trace,
                    chunk_size,
                    policy,
                    reference,
                    "actual_clustered_bootstrap",
                ),
            )
            rows.append(
                {
                    "trace": trace,
                    "chunk_size": int(chunk_size),
                    "policy": policy,
                    "reference_policy": reference,
                    "n_independent_series": int(len(pivot)),
                    "median_ratio_policy_over_reference": float(
                        np.median(ratios)
                    ),
                    "mean_ratio_policy_over_reference": float(
                        np.mean(ratios)
                    ),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "fraction_policy_faster": float(
                        np.mean(ratios < 1.0)
                    ),
                    "wilcoxon_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def protocol_payload(
    pems04_meta: Mapping[str, Any],
    pems08_meta: Mapping[str, Any],
    bdg2_manifest: pd.DataFrame,
) -> Dict[str, Any]:
    return {
        "version": VERSION,
        "target_journal": TARGET_JOURNAL,
        "study_name": "AF-HC nearest-baseline protocol",
        "development_dataset": "PEMS04",
        "holdout_datasets": ["PEMS08", "BDG2"],
        "seed": SEED,
        "trace_seeds": list(TRACE_SEEDS),
        "n_independent_entities_per_dataset": N_SERIES,
        "chunk_sizes": list(CHUNK_SIZES),
        "trace_names": list(TRACE_NAMES),
        "trace_design": {
            "shape_only": {
                "access_frequency": "equal for all chunks",
                "point_drift_group": "800 local scan + 2400 local point",
                "scan_stable_group": "3200 local scan",
            },
            "frequency_only": {
                "query_shape": "all chunks scan-to-point",
                "hot_group": "3200 local accesses",
                "cold_group": "800 local accesses spread every fourth global round",
            },
            "joint_orthogonal": {
                "groups": [
                    "hot_point",
                    "hot_scan",
                    "cold_point",
                    "cold_scan",
                ],
                "entities_per_group": 8,
            },
            "brief_burst": {
                "burst_group": "1000 scan + 250 point + 1950 scan",
                "control_group": "3200 scan",
            },
            "stable_scan": {
                "all_chunks": "3200 local scan accesses",
            },
        },
        "afhc_information_boundary": {
            "allowed": [
                "per-chunk access counts",
                "frequency EMA",
                "fixed reclassification period",
                "fixed hot/cold percentile",
                "fixed hot/cold codecs",
            ],
            "forbidden": [
                "query class or query shape",
                "point/scan/aggregate proportions",
                "dual-timescale layout estimates",
                "predicted saving",
                "break-even",
                "confirmation rounds",
                "cooldown",
                "future phase labels",
            ],
        },
        "afhc_grid": {
            "adaptation_periods": list(ADAPTATION_PERIODS),
            "hot_fractions": list(HOT_FRACTIONS),
            "cold_codecs": list(COLD_CODECS),
            "hot_codecs": list(HOT_CODECS),
            "frequency_ema_alpha": FREQUENCY_EMA_ALPHA,
            "n_configurations": 72,
            "hot_classification": (
                "strictly above the (1-hot_fraction) percentile; "
                "threshold ties remain cold"
            ),
        },
        "grid_selection": {
            "safety_filters": {
                "stable_ratio_vs_initial_best_max": MAX_STABLE_RATIO,
                "brief_burst_ratio_vs_initial_best_max": MAX_BURST_RATIO,
                "brief_burst_median_migrations_per_chunk_max": (
                    MAX_BURST_MEDIAN_MIGRATIONS_PER_CHUNK
                ),
            },
            "objective": (
                "minimum median AF-HC/initial-best total-cost ratio across "
                "five traces, two chunk sizes, and three trace seeds"
            ),
            "equivalence_tolerance": TIE_SCORE_TOLERANCE,
            "tie_break_order": [
                "fewer migrations",
                "fewer rewrite bytes",
                "longer adaptation period",
                "lexicographically smaller config id",
            ],
        },
        "cost_measurement": {
            "query_grid_selection": (
                "per-chunk codec/query-class calibrated median latency"
            ),
            "migration_grid_selection": (
                "physically measured directed decode+encode+fsync cost"
            ),
            "selected_config_audit": (
                "real file-backed warm_slicedmeta replay with all query "
                "results checked against NumPy truth"
            ),
        },
        "repressdb_frozen_controller": {
            "long_window": LONG_WINDOW,
            "short_window": SHORT_WINDOW,
            "minimum_observations": MIN_OBSERVATIONS,
            "decision_interval": DECISION_INTERVAL,
            "confirmation_rounds": CONFIRMATION_ROUNDS,
            "cooldown_queries": COOLDOWN_QUERIES,
            "prediction_horizon": PREDICTION_HORIZON,
            "minimum_relative_improvement": MIN_RELATIVE_IMPROVEMENT,
            "rewrite_safety_factor": REWRITE_SAFETY_FACTOR,
        },
        "pems04_source": pems04_meta,
        "pems08_source": pems08_meta,
        "bdg2_reserved_manifest_status": {
            "found": bool(not bdg2_manifest.empty),
            "n_reserved_buildings": int(
                bdg2_manifest["building_id"].nunique()
                if not bdg2_manifest.empty
                else 0
            ),
            "bdg2_root": str(BDG2_ROOT),
        },
        "anti_leakage": (
            "AF-HC parameters are selected only on PEMS04. The resulting "
            "hotcold_frozen_config.json must be used unchanged on PEMS08 and BDG2."
        ),
    }


def build_model_actual_validation(
    grid_run_df: pd.DataFrame,
    model_policy_df: pd.DataFrame,
    actual_summary_df: pd.DataFrame,
    selected_config_id: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    keys = ["trace", "trace_seed", "chunk_size"]

    selected_grid = grid_run_df[
        grid_run_df["config_id"] == selected_config_id
    ][keys + ["total_time_ns"]].rename(
        columns={"total_time_ns": "model_afhc_total_ns"}
    )
    model_pivot = model_policy_df.pivot(
        index=keys, columns="policy", values="total_time_ns"
    ).reset_index().rename(
        columns={
            "initial_best": "model_initial_total_ns",
            "repressdb": "model_repressdb_total_ns",
        }
    )
    actual_pivot = actual_summary_df.pivot(
        index=keys, columns="policy", values="total_time_ns"
    ).reset_index().rename(
        columns={
            "initial_best": "actual_initial_total_ns",
            "afhc_frozen": "actual_afhc_total_ns",
            "repressdb": "actual_repressdb_total_ns",
        }
    )

    merged = (
        selected_grid.merge(model_pivot, on=keys, how="inner")
        .merge(actual_pivot, on=keys, how="inner")
    )
    for row in merged.itertuples(index=False):
        model_afhc_ratio = float(
            row.model_afhc_total_ns / row.model_initial_total_ns
        )
        actual_afhc_ratio = float(
            row.actual_afhc_total_ns / row.actual_initial_total_ns
        )
        model_repress_ratio = float(
            row.model_repressdb_total_ns / row.model_initial_total_ns
        )
        actual_repress_ratio = float(
            row.actual_repressdb_total_ns / row.actual_initial_total_ns
        )
        rows.append(
            {
                "trace": row.trace,
                "trace_seed": int(row.trace_seed),
                "chunk_size": int(row.chunk_size),
                "model_afhc_ratio_to_initial": model_afhc_ratio,
                "actual_afhc_ratio_to_initial": actual_afhc_ratio,
                "absolute_afhc_ratio_error": abs(
                    model_afhc_ratio - actual_afhc_ratio
                ),
                "model_repressdb_ratio_to_initial": model_repress_ratio,
                "actual_repressdb_ratio_to_initial": actual_repress_ratio,
                "absolute_repressdb_ratio_error": abs(
                    model_repress_ratio - actual_repress_ratio
                ),
            }
        )
    return pd.DataFrame(rows)


def build_assessment(
    selected_config: Mapping[str, Any],
    grid_summary_df: pd.DataFrame,
    grid_run_df: pd.DataFrame,
    actual_summary_df: pd.DataFrame,
    actual_comparison_df: pd.DataFrame,
    model_actual_validation_df: pd.DataFrame,
    protocol_hash: str,
) -> Dict[str, Any]:
    selected_id = str(selected_config["config_id"])
    selected_grid = grid_run_df[
        grid_run_df["config_id"] == selected_id
    ]

    actual_key_results: Dict[str, Any] = {}
    for chunk_size in CHUNK_SIZES:
        for trace in TRACE_NAMES:
            local = actual_comparison_df[
                (actual_comparison_df["trace"] == trace)
                & (actual_comparison_df["chunk_size"] == chunk_size)
                & (actual_comparison_df["policy"] == "repressdb")
                & (
                    actual_comparison_df["reference_policy"]
                    == "afhc_frozen"
                )
            ]
            if len(local) != 1:
                continue
            row = local.iloc[0]
            actual_key_results[f"{trace}|{chunk_size}"] = {
                "median_ratio_repressdb_over_afhc": float(
                    row["median_ratio_policy_over_reference"]
                ),
                "bootstrap_ci_low": float(row["bootstrap_ci_low"]),
                "bootstrap_ci_high": float(row["bootstrap_ci_high"]),
                "fraction_repressdb_faster": float(
                    row["fraction_policy_faster"]
                ),
                "wilcoxon_p": float(row["wilcoxon_p"]),
            }

    shape_signal = actual_key_results.get("shape_only|16384", {})
    frequency_signal = actual_key_results.get("frequency_only|16384", {})
    stable_signal = actual_key_results.get("stable_scan|16384", {})

    median_afhc_model_error = (
        float(
            model_actual_validation_df[
                "absolute_afhc_ratio_error"
            ].median()
        )
        if not model_actual_validation_df.empty
        else math.nan
    )
    median_repress_model_error = (
        float(
            model_actual_validation_df[
                "absolute_repressdb_ratio_error"
            ].median()
        )
        if not model_actual_validation_df.empty
        else math.nan
    )

    development_checks = {
        "D1_exactly_72_grid_configurations": bool(
            grid_summary_df["config_id"].nunique() == 72
        ),
        "D2_at_least_one_configuration_passes_safety": bool(
            grid_summary_df["safety_pass"].any()
        ),
        "D3_one_and_only_one_frozen_configuration": bool(
            grid_summary_df["selected"].sum() == 1
        ),
        "D4_actual_audit_has_all_policy_trace_size_seed_cells": bool(
            len(actual_summary_df)
            == len(AUDIT_POLICIES)
            * len(TRACE_NAMES)
            * len(CHUNK_SIZES)
            * len(TRACE_SEEDS)
        ),
        "D5_shape_only_development_signal_repressdb_faster": bool(
            shape_signal
            and shape_signal["median_ratio_repressdb_over_afhc"] < 1.0
        ),
        "D6_frequency_only_development_noncatastrophic": bool(
            frequency_signal
            and frequency_signal["median_ratio_repressdb_over_afhc"] <= 1.10
        ),
        "D7_stable_development_noncatastrophic": bool(
            stable_signal
            and stable_signal["median_ratio_repressdb_over_afhc"] <= 1.10
        ),
        "D8_calibrated_replay_tracks_actual_ratios": bool(
            np.isfinite(median_afhc_model_error)
            and np.isfinite(median_repress_model_error)
            and median_afhc_model_error <= 0.10
            and median_repress_model_error <= 0.10
        ),
    }

    return {
        "version": VERSION,
        "target_journal": TARGET_JOURNAL,
        "status": "A0 protocol frozen and A1 PEMS04 AF-HC development selection complete",
        "study_role": (
            "Nearest-baseline development selection only. PEMS08 and BDG2 "
            "remain untouched holdouts for A2."
        ),
        "protocol_sha256": protocol_hash,
        "selected_configuration": dict(selected_config),
        "selected_grid_results": {
            "composite_median_ratio": float(
                selected_grid["ratio_to_initial_best"].median()
            ),
            "stable_median_ratio": float(
                selected_grid[
                    selected_grid["trace"] == "stable_scan"
                ]["ratio_to_initial_best"].median()
            ),
            "brief_burst_median_ratio": float(
                selected_grid[
                    selected_grid["trace"] == "brief_burst"
                ]["ratio_to_initial_best"].median()
            ),
            "median_migrations_per_chunk": float(
                selected_grid["median_migrations_per_chunk"].median()
            ),
        },
        "actual_audit_repressdb_vs_afhc": actual_key_results,
        "calibrated_replay_validation": {
            "median_absolute_afhc_ratio_error": median_afhc_model_error,
            "median_absolute_repressdb_ratio_error": (
                median_repress_model_error
            ),
        },
        "development_checks": {
            **development_checks,
            "all_eight": bool(all(development_checks.values())),
        },
        "interpretation_boundary": (
            "A1 can establish that a fair AF-HC configuration has been selected "
            "and that the implementation behaves as intended on PEMS04. It cannot "
            "establish the paper's nearest-baseline claim; that claim requires the "
            "frozen PEMS08 and BDG2 A2 holdouts."
        ),
        "next_gate": (
            "Run A2 with hotcold_frozen_config.json unchanged. Do not alter AF-HC "
            "or RePressDB parameters after inspecting A2 results."
        ),
    }


def main() -> int:
    e1v2, e1, base = load_repressdb_modules()
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / OUTPUT_DIR_NAME
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_dir = output_dir / "_candidate_storage"
    migration_calibration_dir = output_dir / "_migration_calibration"
    actual_migration_dir = output_dir / "_actual_audit_migrations"
    storage_dir.mkdir(parents=True, exist_ok=True)
    migration_calibration_dir.mkdir(parents=True, exist_ok=True)
    actual_migration_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 104)
    print("RePressDB A0+A1 v0.7.0 — AF-HC protocol and PEMS04 development selection")
    print(f"Target journal: {TARGET_JOURNAL}")
    print("No controller retuning. PEMS08 and BDG2 remain holdouts.")
    print("=" * 104)

    # ------------------------------------------------------------------
    # A0: deterministic manifests and protocol freeze.
    # ------------------------------------------------------------------
    local_n_series = 8 if QUICK_TEST else N_SERIES
    local_chunk_sizes = (8192,) if QUICK_TEST else CHUNK_SIZES

    if QUICK_TEST:
        pems04_matrix, pems04_meta = base.synthetic_dataset(
            SEED, local_n_series
        )
        pems08_matrix, pems08_meta = base.synthetic_dataset(
            SEED + 1, local_n_series
        )
        pems04_meta = dict(pems04_meta)
        pems08_meta = dict(pems08_meta)
        pems04_meta["selected_original_series_indices"] = list(
            range(local_n_series)
        )
        pems08_meta["selected_original_series_indices"] = list(
            range(local_n_series)
        )
        pems04_meta["source"] = "synthetic_quick_pems04"
        pems08_meta["source"] = "synthetic_quick_pems08"
    else:
        pems04_dir, pems08_dir = base.default_data_dirs()
        pems04_file = base.auto_detect_data_file(
            pems04_dir, "PEMS04_AFHC_PROTOCOL"
        )
        pems08_file = base.auto_detect_data_file(
            pems08_dir, "PEMS08_AFHC_PROTOCOL"
        )
        pems04_matrix, pems04_meta = base.load_dataset(
            pems04_file,
            feature_index=0,
            max_series=local_n_series,
            seed=SEED,
            excluded_series=None,
        )
        pems08_matrix, pems08_meta = base.load_dataset(
            pems08_file,
            feature_index=0,
            max_series=local_n_series,
            seed=SEED,
            excluded_series=None,
        )

    pems04_chunks, pems04_manifest = deterministic_series_chunks(
        base,
        pems04_matrix,
        "PEMS04",
        pems04_meta["selected_original_series_indices"],
        chunk_sizes=local_chunk_sizes,
    )
    _, pems08_manifest = deterministic_series_chunks(
        base,
        pems08_matrix,
        "PEMS08",
        pems08_meta["selected_original_series_indices"],
        chunk_sizes=local_chunk_sizes,
    )

    # In quick mode, build assignments without the production 32-series assertion.
    if QUICK_TEST:
        assignment_rows: List[Dict[str, Any]] = []
        for (dataset, chunk_size), group in pd.concat(
            [pems04_manifest, pems08_manifest], ignore_index=True
        ).groupby(["dataset", "chunk_size"]):
            ordered = group.sort_values("original_series_index").reset_index(drop=True)
            for rank, row in enumerate(ordered.itertuples(index=False)):
                for trace_name, group_name in (
                    ("shape_only", "point_drift" if rank % 2 == 0 else "scan_stable"),
                    ("frequency_only", "hot" if rank % 2 == 0 else "cold"),
                    (
                        "joint_orthogonal",
                        ("hot_point", "hot_scan", "cold_point", "cold_scan")[rank % 4],
                    ),
                    ("brief_burst", "burst" if rank % 2 == 0 else "scan_stable"),
                    ("stable_scan", "scan_stable"),
                ):
                    assignment_rows.append(
                        {
                            "dataset": dataset,
                            "chunk_id": row.chunk_id,
                            "original_series_index": int(row.original_series_index),
                            "chunk_size": int(chunk_size),
                            "assignment_rank": int(rank),
                            "frequency_tie_break_key": stable_hash(row.chunk_id),
                            "trace": trace_name,
                            "group": group_name,
                        }
                    )
        assignments = pd.DataFrame(assignment_rows)
    else:
        assignments = build_trace_assignments(
            pd.concat(
                [pems04_manifest, pems08_manifest], ignore_index=True
            )
        )

    bdg2_manifest = build_bdg2_future_manifest(project_dir)

    pems04_manifest.to_csv(
        output_dir / "pems04_selected_chunks.csv", index=False
    )
    pems08_manifest.to_csv(
        output_dir / "pems08_selected_chunks.csv", index=False
    )
    bdg2_manifest.to_csv(
        output_dir / "bdg2_selected_buildings.csv", index=False
    )
    assignments.to_csv(
        output_dir / "trace_assignment_manifest.csv", index=False
    )

    protocol = protocol_payload(
        pems04_meta, pems08_meta, bdg2_manifest
    )
    protocol["quick_test"] = QUICK_TEST
    write_json(output_dir / "hotcold_protocol.json", protocol)

    protocol_hash = hashlib.sha256()
    for filename in (
        "hotcold_protocol.json",
        "pems04_selected_chunks.csv",
        "pems08_selected_chunks.csv",
        "bdg2_selected_buildings.csv",
        "trace_assignment_manifest.csv",
    ):
        path = output_dir / filename
        protocol_hash.update(filename.encode("utf-8"))
        protocol_hash.update(sha256_file(path).encode("ascii"))
    protocol_sha256 = protocol_hash.hexdigest()
    (output_dir / "protocol_sha256.txt").write_text(
        protocol_sha256 + "\n", encoding="utf-8"
    )

    print(f"[A0] protocol hash: {protocol_sha256}")
    print(
        f"[A0] PEMS04 chunks={len(pems04_chunks)}, "
        f"PEMS08 reserved chunks={len(pems08_manifest)}, "
        f"BDG2 reserved buildings={len(bdg2_manifest)}"
    )

    # ------------------------------------------------------------------
    # A1 calibration: encode/query profiles and directed migration costs.
    # ------------------------------------------------------------------
    reps = base.representation_registry(3)
    codecs = tuple(sorted(reps))
    chunks_by_id = {
        chunk.chunk_id: chunk for chunk in pems04_chunks
    }
    chunks_by_size: Dict[int, List[Any]] = {
        int(size): sorted(
            [chunk for chunk in pems04_chunks if chunk.chunk_size == size],
            key=lambda chunk: chunk.chunk_id,
        )
        for size in local_chunk_sizes
    }

    stored_by_chunk: Dict[str, Dict[str, Any]] = {}
    profiles: Dict[str, Dict[str, Dict[str, float]]] = {}
    encode_ns_by_chunk: Dict[str, Dict[str, int]] = {}
    decode_ns_by_chunk: Dict[str, Dict[str, int]] = {}
    query_pools: Dict[str, Dict[str, List[Any]]] = {}
    calibration_rows: List[Dict[str, Any]] = []
    codec_rows: List[Dict[str, Any]] = []

    for chunk_index, chunk in enumerate(pems04_chunks, start=1):
        chunk_dir = storage_dir / chunk.chunk_id
        stored_map, encode_ns, decode_ns = e1.encode_candidates(
            base, reps, chunk, chunk_dir
        )
        stored_by_chunk[chunk.chunk_id] = stored_map
        encode_ns_by_chunk[chunk.chunk_id] = {
            str(codec): int(value)
            for codec, value in encode_ns.items()
        }
        decode_ns_by_chunk[chunk.chunk_id] = {
            str(codec): int(value)
            for codec, value in decode_ns.items()
        }

        for codec, stored in stored_map.items():
            codec_rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "series_index": int(chunk.series_index),
                    "chunk_start": int(chunk.chunk_start),
                    "chunk_size": int(chunk.chunk_size),
                    "codec": codec,
                    "raw_size_bytes": int(chunk.values.nbytes),
                    "payload_size_bytes": int(stored.payload_size_bytes),
                    "metadata_size_bytes": int(stored.metadata_size_bytes),
                    "stored_size_bytes": int(stored.stored_size_bytes),
                    "compression_ratio": float(
                        stored.stored_size_bytes / chunk.values.nbytes
                    ),
                    "encode_ns": int(encode_ns[codec]),
                    "full_decode_ns": int(decode_ns[codec]),
                }
            )

        queries = base.build_queries(
            chunk.values,
            chunk.chunk_id,
            SEED,
            1 if QUICK_TEST else e1.WINDOWS_PER_FRACTION,
        )
        pool = {
            query_class: [
                query
                for query in queries
                if query.query_class == query_class
            ]
            for query_class in QUERY_CLASSES
        }
        if any(not values for values in pool.values()):
            raise AssertionError(
                f"Incomplete query pool: {chunk.chunk_id}"
            )
        query_pools[chunk.chunk_id] = pool

        profile, local_calibration = e1.calibrate_chunk(
            base, reps, stored_map, chunk, pool
        )
        profiles[chunk.chunk_id] = profile
        for row in local_calibration:
            row["series_index"] = int(chunk.series_index)
            row["chunk_start"] = int(chunk.chunk_start)
        calibration_rows.extend(local_calibration)

        if (
            chunk_index == 1
            or chunk_index == len(pems04_chunks)
            or chunk_index % max(1, len(pems04_chunks) // 10) == 0
        ):
            print(
                f"[A1 calibration] {chunk_index}/{len(pems04_chunks)} chunks"
            )

    migration_costs, migration_bytes, migration_rows = (
        measure_directed_migrations(
            e1,
            base,
            reps,
            chunks_by_id,
            stored_by_chunk,
            migration_calibration_dir,
        )
    )

    pd.DataFrame(calibration_rows).to_csv(
        output_dir / "calibration_records.csv", index=False
    )
    pd.DataFrame(codec_rows).to_csv(
        output_dir / "codec_records.csv", index=False
    )
    pd.DataFrame(migration_rows).to_csv(
        output_dir / "measured_migration_costs.csv", index=False
    )

    # ------------------------------------------------------------------
    # A1 model replay and 72-configuration selection.
    # ------------------------------------------------------------------
    grid_configs = config_grid()
    grid_run_rows: List[Dict[str, Any]] = []
    model_policy_rows: List[Dict[str, Any]] = []
    trace_manifest_rows: List[Dict[str, Any]] = []

    local_trace_seeds = (SEED,) if QUICK_TEST else TRACE_SEEDS
    local_trace_names = (
        ("shape_only", "stable_scan")
        if QUICK_TEST
        else TRACE_NAMES
    )

    for chunk_size in local_chunk_sizes:
        size_chunks = chunks_by_size[int(chunk_size)]
        size_chunk_ids = tuple(chunk.chunk_id for chunk in size_chunks)
        size_profiles_array = profile_arrays(
            profiles, size_chunk_ids, codecs
        )
        initial_positions = scan_best_codec_positions(
            e1, base, profiles, size_chunk_ids, codecs
        )
        size_assignments = assignments[
            (assignments["dataset"] == "PEMS04")
            & (assignments["chunk_size"] == chunk_size)
        ]

        for trace_seed in local_trace_seeds:
            for trace_name in local_trace_names:
                plan = build_trace_plan(
                    base,
                    size_chunks,
                    query_pools,
                    size_assignments,
                    trace_name,
                    int(trace_seed),
                )
                trace_hash = hashlib.sha256()
                for array in (
                    plan.chunk_positions,
                    plan.local_indices,
                    plan.query_class_positions,
                    plan.query_choice_positions,
                    plan.workload_positions,
                ):
                    trace_hash.update(array.tobytes())
                trace_manifest_rows.append(
                    {
                        "dataset": "PEMS04",
                        "trace": trace_name,
                        "trace_seed": int(trace_seed),
                        "chunk_size": int(chunk_size),
                        "n_chunks": int(len(size_chunks)),
                        "n_global_queries": int(plan.n_events),
                        "trace_sha256": trace_hash.hexdigest(),
                    }
                )

                initial_result = simulate_initial_best(
                    plan, size_profiles_array, initial_positions
                )
                repress_result = simulate_repressdb(
                    e1,
                    e1v2,
                    base,
                    plan,
                    profiles,
                    encode_ns_by_chunk,
                    decode_ns_by_chunk,
                    migration_costs,
                    migration_bytes,
                )
                for policy, result in (
                    ("initial_best", initial_result),
                    ("repressdb", repress_result),
                ):
                    model_policy_rows.append(
                        {
                            "trace": trace_name,
                            "trace_seed": int(trace_seed),
                            "chunk_size": int(chunk_size),
                            "policy": policy,
                            "n_queries": int(plan.n_events),
                            "query_time_ns": float(result["query_time_ns"]),
                            "migration_time_ns": float(
                                result["migration_time_ns"]
                            ),
                            "total_time_ns": float(result["total_time_ns"]),
                            "migration_count": int(result["migration_count"]),
                            "rewrite_bytes": int(result["rewrite_bytes"]),
                        }
                    )

                for config in grid_configs:
                    afhc_result = simulate_afhc(
                        plan,
                        size_profiles_array,
                        codecs,
                        migration_costs,
                        migration_bytes,
                        config,
                    )
                    ratio = float(
                        afhc_result["total_time_ns"]
                        / initial_result["total_time_ns"]
                    )
                    grid_run_rows.append(
                        {
                            **config,
                            "trace": trace_name,
                            "trace_seed": int(trace_seed),
                            "chunk_size": int(chunk_size),
                            "n_chunks": int(len(size_chunks)),
                            "n_queries": int(plan.n_events),
                            "query_time_ns": float(
                                afhc_result["query_time_ns"]
                            ),
                            "migration_time_ns": float(
                                afhc_result["migration_time_ns"]
                            ),
                            "total_time_ns": float(
                                afhc_result["total_time_ns"]
                            ),
                            "initial_best_total_time_ns": float(
                                initial_result["total_time_ns"]
                            ),
                            "ratio_to_initial_best": ratio,
                            "migration_count": int(
                                afhc_result["migration_count"]
                            ),
                            "median_migrations_per_chunk": float(
                                np.median(
                                    afhc_result[
                                        "per_chunk_migration_count"
                                    ]
                                )
                            ),
                            "mean_migrations_per_chunk": float(
                                np.mean(
                                    afhc_result[
                                        "per_chunk_migration_count"
                                    ]
                                )
                            ),
                            "rewrite_bytes": int(
                                afhc_result["rewrite_bytes"]
                            ),
                        }
                    )

                print(
                    f"[A1 model] size={chunk_size}, trace={trace_name}, "
                    f"seed={trace_seed}, queries={plan.n_events}"
                )

    grid_run_df = pd.DataFrame(grid_run_rows)
    model_policy_df = pd.DataFrame(model_policy_rows)
    trace_manifest_df = pd.DataFrame(trace_manifest_rows)
    (
        grid_summary_df,
        grid_failures_df,
        grid_ranked_df,
        selected_config_summary,
    ) = summarize_grid(grid_run_df)

    selected_config_id = str(selected_config_summary["config_id"])
    selected_config = next(
        config
        for config in grid_configs
        if config["config_id"] == selected_config_id
    )
    selected_config.update(
        {
            "protocol_sha256": protocol_sha256,
            "selection_dataset": "PEMS04",
            "selection_version": VERSION,
            "selection_summary": {
                key: json_safe(value)
                for key, value in selected_config_summary.items()
            },
            "frozen_for_holdouts": True,
        }
    )

    grid_run_df.to_csv(
        output_dir / "hotcold_grid_all_results.csv", index=False
    )
    grid_summary_df.to_csv(
        output_dir / "hotcold_grid_config_summary.csv", index=False
    )
    grid_failures_df.to_csv(
        output_dir / "hotcold_grid_safety_failures.csv", index=False
    )
    grid_ranked_df.to_csv(
        output_dir / "hotcold_grid_ranked.csv", index=False
    )
    model_policy_df.to_csv(
        output_dir / "model_repressdb_initial_results.csv", index=False
    )
    trace_manifest_df.to_csv(
        output_dir / "generated_trace_manifest.csv", index=False
    )
    write_json(
        output_dir / "hotcold_frozen_config.json", selected_config
    )

    experiment_freeze = hashlib.sha256()
    experiment_freeze.update(protocol_sha256.encode("ascii"))
    for freeze_file in (
        output_dir / "generated_trace_manifest.csv",
        output_dir / "hotcold_frozen_config.json",
        Path(__file__).resolve(),
    ):
        experiment_freeze.update(freeze_file.name.encode("utf-8"))
        experiment_freeze.update(sha256_file(freeze_file).encode("ascii"))
    experiment_freeze_sha256 = experiment_freeze.hexdigest()
    (output_dir / "experiment_freeze_sha256.txt").write_text(
        experiment_freeze_sha256 + "\n", encoding="utf-8"
    )
    selected_config["experiment_freeze_sha256"] = (
        experiment_freeze_sha256
    )
    write_json(
        output_dir / "hotcold_frozen_config.json", selected_config
    )

    print("-" * 104)
    print("[A1] frozen AF-HC configuration:")
    print(json.dumps(selected_config, ensure_ascii=False, indent=2))
    print("-" * 104)

    # ------------------------------------------------------------------
    # A1 actual file-backed audit of selected AF-HC vs frozen RePressDB.
    # ------------------------------------------------------------------
    actual_summary_rows: List[Dict[str, Any]] = []
    actual_chunk_rows: List[Dict[str, Any]] = []
    actual_event_rows: List[Dict[str, Any]] = []

    for chunk_size in local_chunk_sizes:
        size_chunks = chunks_by_size[int(chunk_size)]
        size_assignments = assignments[
            (assignments["dataset"] == "PEMS04")
            & (assignments["chunk_size"] == chunk_size)
        ]

        for trace_seed in local_trace_seeds:
            for trace_name in local_trace_names:
                plan = build_trace_plan(
                    base,
                    size_chunks,
                    query_pools,
                    size_assignments,
                    trace_name,
                    int(trace_seed),
                )
                policy_order = list(AUDIT_POLICIES)
                random.Random(
                    stable_seed(
                        trace_seed,
                        chunk_size,
                        trace_name,
                        "actual_policy_order",
                    )
                ).shuffle(policy_order)

                for policy in policy_order:
                    print(
                        f"[A1 actual] size={chunk_size}, trace={trace_name}, "
                        f"seed={trace_seed}, policy={policy}, queries={plan.n_events}"
                    )
                    summary, chunk_rows, event_rows = run_actual_policy(
                        e1,
                        e1v2,
                        base,
                        reps,
                        chunks_by_id,
                        stored_by_chunk,
                        profiles,
                        encode_ns_by_chunk,
                        decode_ns_by_chunk,
                        query_pools,
                        plan,
                        policy,
                        selected_config,
                        actual_migration_dir,
                    )
                    actual_summary_rows.append(summary)
                    actual_chunk_rows.extend(chunk_rows)
                    actual_event_rows.extend(event_rows)

    actual_summary_df = pd.DataFrame(actual_summary_rows)
    actual_chunk_df = pd.DataFrame(actual_chunk_rows)
    actual_event_df = pd.DataFrame(actual_event_rows)
    actual_comparison_df = actual_clustered_comparisons(
        actual_chunk_df
    )
    model_actual_validation_df = build_model_actual_validation(
        grid_run_df,
        model_policy_df,
        actual_summary_df,
        selected_config_id,
    )

    actual_summary_df.to_csv(
        output_dir / "actual_audit_global_run_records.csv", index=False
    )
    actual_chunk_df.to_csv(
        output_dir / "actual_audit_chunk_run_records.csv", index=False
    )
    actual_event_df.to_csv(
        output_dir / "actual_audit_event_diagnostics.csv", index=False
    )
    actual_comparison_df.to_csv(
        output_dir
        / "actual_audit_series_clustered_comparisons.csv",
        index=False,
    )
    model_actual_validation_df.to_csv(
        output_dir / "model_actual_validation.csv",
        index=False,
    )

    assessment = build_assessment(
        selected_config,
        grid_summary_df,
        grid_run_df,
        actual_summary_df,
        actual_comparison_df,
        model_actual_validation_df,
        protocol_sha256,
    )
    write_json(
        output_dir / "hotcold_dev_assessment.json", assessment
    )
    write_json(project_dir / SUMMARY_JSON_NAME, assessment)

    run_config = {
        "version": VERSION,
        "target_journal": TARGET_JOURNAL,
        "protocol_sha256": protocol_sha256,
        "seed": SEED,
        "trace_seeds": list(local_trace_seeds),
        "n_series": int(local_n_series),
        "chunk_sizes": list(local_chunk_sizes),
        "trace_names": list(local_trace_names),
        "afhc_grid": {
            "adaptation_periods": list(ADAPTATION_PERIODS),
            "hot_fractions": list(HOT_FRACTIONS),
            "cold_codecs": list(COLD_CODECS),
            "hot_codecs": list(HOT_CODECS),
            "frequency_ema_alpha": FREQUENCY_EMA_ALPHA,
        },
        "selected_configuration": selected_config,
        "query_calibration": {
            "windows_per_fraction": (
                1 if QUICK_TEST else e1.WINDOWS_PER_FRACTION
            ),
            "queries_per_class": (
                1 if QUICK_TEST else e1.CALIBRATION_QUERIES_PER_CLASS
            ),
            "repeats": (
                1 if QUICK_TEST else e1.CALIBRATION_REPEATS
            ),
        },
        "migration_measure_repeats": MIGRATION_MEASURE_REPEATS,
        "repressdb_frozen_controller": {
            "long_window": LONG_WINDOW,
            "short_window": SHORT_WINDOW,
            "minimum_observations": MIN_OBSERVATIONS,
            "decision_interval": DECISION_INTERVAL,
            "confirmation_rounds": CONFIRMATION_ROUNDS,
            "cooldown_queries": COOLDOWN_QUERIES,
            "prediction_horizon": PREDICTION_HORIZON,
            "minimum_relative_improvement": MIN_RELATIVE_IMPROVEMENT,
            "rewrite_safety_factor": REWRITE_SAFETY_FACTOR,
        },
        "representations": list(codecs),
        "primary_execution_mode": "warm_slicedmeta",
        "grid_metric": (
            "calibrated per-query-class median query time plus physically "
            "measured directed migration time"
        ),
        "actual_audit": (
            "real file-backed replay with actual decode+encode+fsync migration "
            "and every query checked against NumPy truth"
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "timer": "time.perf_counter_ns",
        },
    }
    write_json(output_dir / "run_config.json", run_config)

    # Candidate files and migration trials are reproducible temporary artifacts.
    shutil.rmtree(storage_dir, ignore_errors=True)
    shutil.rmtree(migration_calibration_dir, ignore_errors=True)
    shutil.rmtree(actual_migration_dir, ignore_errors=True)

    print("=" * 104)
    print("A0+A1 complete")
    print("Development checks:")
    print(json.dumps(assessment["development_checks"], ensure_ascii=False, indent=2))
    print(f"Results: {output_dir.resolve()}")
    print("Upload these files after completion:")
    print("  hotcold_dev_assessment.json")
    print("  hotcold_frozen_config.json")
    print("  hotcold_grid_ranked.csv")
    print("  hotcold_grid_all_results.csv")
    print("  actual_audit_series_clustered_comparisons.csv")
    print("  actual_audit_global_run_records.csv")
    print("  actual_audit_chunk_run_records.csv")
    print("  model_actual_validation.csv")
    print("  protocol_sha256.txt")
    print("  experiment_freeze_sha256.txt")
    print("  hotcold_protocol.json")
    print("  run_config.json")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
