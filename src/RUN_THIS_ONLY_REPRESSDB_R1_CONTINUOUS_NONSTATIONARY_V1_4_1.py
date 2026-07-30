#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
RePressDB R1 v1.4.1
Continuous nonstationary replay over real BDG2 values.

Run after S1. This script never retunes Shape-EWMA-BE.

Required result directories:
  results_s1_shape_ewma_be_v130/
  results_b2_full_bdg2_scale_v100/
  results_a0_a1_hotcold_pems04_v070/
  results_a0_a1_hotcold_deadband_pems04_v074/

Main design:
  * 32 independent BDG2 buildings;
  * paired chunks 8192 and 16384;
  * three frozen, phase-free continuous query streams;
  * 640,000 global queries per stream, about 20,000/building;
  * initial-best, best-static, AF-HC-D, Shape-EWMA-BE,
    RePressDB, and hindsight oracle;
  * one identical hard write budget for deployable online policies;
  * real file-backed audit on R1-C, 8 buildings, both sizes,
    initial-best/Shape-EWMA-BE/RePressDB.

The query streams are semi-synthetic. The underlying values are real BDG2
measurements. Do not describe the query streams as production logs.

IMPLEMENTATION REPAIR IN v1.4.1
-------------------------------
Create the _patched_core directory before build_corrected_core writes the
generated module. No scientific parameter, trace, baseline, dataset, budget,
metric definition, or acceptance threshold is changed.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

VERSION = "1.4.1"
TARGET_JOURNAL = "Journal of Big Data"
SEED = 20260728

S1_SCRIPT = "RUN_THIS_ONLY_REPRESSDB_S1_SHAPE_EWMA_BE_V1_3_0.py"
V070_SCRIPT = "RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_DEV_PEMS04_V0_7_0.py"
V073_SCRIPT = "RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_DEADBAND_RECOVER_V0_7_3.py"
V074_SCRIPT = "RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_V0_7_4.py"
A2_SCRIPT = "RUN_THIS_ONLY_REPRESSDB_A2_PEMS08_BDG2_HOLDOUT_V0_8_0.py"
E5_SCRIPT = "RUN_THIS_ONLY_REPRESSDB_E5_BDG2_FROZEN_DIRECT_V0_6_0.py"

S1_DIR = "results_s1_shape_ewma_be_v130"
B2_DIR = "results_b2_full_bdg2_scale_v100"
V070_DIR = "results_a0_a1_hotcold_pems04_v070"
V074_DIR = "results_a0_a1_hotcold_deadband_pems04_v074"
OUTPUT_DIR = "results_r1_continuous_nonstationary_v140"
SUMMARY_JSON = "R1_CONTINUOUS_NONSTATIONARY_ASSESSMENT_v140.json"

N_BUILDINGS = 32
CHUNK_SIZES = (8192, 16384)
LOCAL_QUERIES_PER_BUILDING = 20_000
N_EVENTS = N_BUILDINGS * LOCAL_QUERIES_PER_BUILDING
TRACES = (
    "R1_A_gradual_periodic",
    "R1_B_mean_reverting",
    "R1_C_noisy_bursts",
)
POLICIES = (
    "initial_best",
    "best_static",
    "afhc_d_frozen",
    "shape_ewma_be_frozen",
    "repressdb_frozen",
    "hindsight_oracle",
)
ONLINE_POLICIES = (
    "afhc_d_frozen",
    "shape_ewma_be_frozen",
    "repressdb_frozen",
)
BUDGET_INTERVAL = 1000
DECISION_INTERVAL = 100
MIN_OBSERVATIONS = 200
CUMULATIVE_EVERY = 5000
ACTUAL_TRACE = "R1_C_noisy_bursts"
ACTUAL_BUILDINGS = 8
ACTUAL_POLICIES = (
    "initial_best",
    "shape_ewma_be_frozen",
    "repressdb_frozen",
)
BOOTSTRAPS = 5000

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


def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(json_safe(value), f, ensure_ascii=False, indent=2, allow_nan=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def stable_seed(*parts: Any) -> int:
    token = "|".join(str(x) for x in parts)
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little")


def stable_hash(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(x) for x in parts).encode()).hexdigest()


def load_module(path: Path, name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def geometric_mean(values: Iterable[float]) -> float:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a) & (a > 0)]
    return float(np.exp(np.mean(np.log(a)))) if a.size else math.nan


def bootstrap_median_ci(values: Sequence[float], *seed_parts: Any) -> Tuple[float, float]:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not a.size:
        return math.nan, math.nan
    rng = np.random.default_rng(stable_seed(SEED, "boot", *seed_parts))
    idx = rng.integers(0, len(a), size=(BOOTSTRAPS, len(a)))
    est = np.median(a[idx], axis=1)
    return float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))


def select_buildings(eligible: pd.DataFrame, excluded: set[str]) -> pd.DataFrame:
    f = eligible.copy()
    f["building_id"] = f["building_id"].astype(str)
    f = f[pd.to_numeric(f["longest_run_length"], errors="coerce") >= 16384].copy()
    f["selection_hash"] = [
        stable_hash(
            "R1",
            r.building_id,
            r.meter_type,
            r.source_file,
            r.source_column,
        )
        for r in f.itertuples(index=False)
    ]
    f = (
        f.sort_values(
            ["building_id", "longest_run_length", "selection_hash"],
            ascending=[True, False, True],
        )
        .drop_duplicates("building_id", keep="first")
    )
    no_overlap = f[~f["building_id"].isin(excluded)].copy()
    pool = no_overlap if len(no_overlap) >= N_BUILDINGS else f
    fallback = len(no_overlap) < N_BUILDINGS

    groups = {
        str(mt): g.sort_values("selection_hash").to_dict("records")
        for mt, g in pool.groupby("meter_type")
    }
    order = sorted(groups, key=lambda k: (-len(groups[k]), k))
    cursor = {k: 0 for k in order}
    selected: List[Dict[str, Any]] = []
    while len(selected) < N_BUILDINGS and order:
        alive: List[str] = []
        for key in order:
            i = cursor[key]
            if i < len(groups[key]):
                selected.append(groups[key][i])
                cursor[key] += 1
                if len(selected) == N_BUILDINGS:
                    break
            if cursor[key] < len(groups[key]):
                alive.append(key)
        order = alive
    if len(selected) != N_BUILDINGS:
        raise RuntimeError(f"Only {len(selected)} eligible independent buildings")
    out = pd.DataFrame(selected).sort_values("selection_hash").reset_index(drop=True)
    out["building_rank"] = np.arange(len(out), dtype=int)
    out["excluded_a2_building"] = out["building_id"].isin(excluded)
    out["overlap_fallback_used"] = fallback
    return out


def load_chunks(
    selected: pd.DataFrame,
    e5: Any,
    base: Any,
) -> Tuple[List[Any], pd.DataFrame, Dict[str, Dict[str, Any]]]:
    chunks: List[Any] = []
    rows: List[Dict[str, Any]] = []
    metadata: Dict[str, Dict[str, Any]] = {}

    for source_file, group in selected.groupby("source_file", sort=True):
        columns = group["source_column"].astype(str).tolist()
        frame = e5.read_selected_columns(Path(str(source_file)), columns)
        for row in group.to_dict("records"):
            values = pd.to_numeric(frame[str(row["source_column"])], errors="coerce").to_numpy(
                dtype=np.float32
            )
            start = int(row["longest_run_start"])
            full = np.ascontiguousarray(values[start : start + 16384], dtype=np.float32)
            if full.size != 16384 or not np.isfinite(full).all():
                raise AssertionError(f"Invalid selected run: {row['building_id']}")
            for size in CHUNK_SIZES:
                local = np.ascontiguousarray(full[:size], dtype=np.float32)
                cid = f"bdg2_r1_cs{size}_{str(row['selection_hash'])[:16]}"
                chunks.append(
                    base.ChunkUnit(
                        chunk_id=cid,
                        series_index=int(row["building_rank"]),
                        chunk_start=start,
                        chunk_size=size,
                        values=local,
                    )
                )
                metadata[cid] = {
                    "unit_type": "building",
                    "unit_id": str(row["building_id"]),
                    "building_id": str(row["building_id"]),
                    "meter_type": str(row["meter_type"]),
                    "site_id": str(row["site_id"]),
                    "source_file": str(row["source_file"]),
                    "source_column": str(row["source_column"]),
                }
                rows.append(
                    {
                        "chunk_id": cid,
                        "building_rank": int(row["building_rank"]),
                        "building_id": str(row["building_id"]),
                        "meter_type": str(row["meter_type"]),
                        "site_id": str(row["site_id"]),
                        "source_file": str(row["source_file"]),
                        "source_column": str(row["source_column"]),
                        "chunk_size": size,
                        "chunk_start": start,
                        "chunk_end_exclusive": start + size,
                        "raw_bytes": int(local.nbytes),
                        "raw_sha256": hashlib.sha256(local.tobytes()).hexdigest(),
                    }
                )
        print(f"[R1 load] {source_file}: {len(group)} buildings")

    manifest = pd.DataFrame(rows).sort_values(
        ["chunk_size", "building_rank"]
    ).reset_index(drop=True)
    if manifest["building_id"].nunique() != N_BUILDINGS:
        raise AssertionError("Not exactly 32 independent buildings")
    return chunks, manifest, metadata


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def category_probabilities(trace: str, n: int) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(SEED, trace, "category"))
    x = np.linspace(0.0, 1.0, n, dtype=float)

    if trace == "R1_A_gradual_periodic":
        sig = 1.0 / (1.0 + np.exp(-(x - 0.50) / 0.12))
        point = np.clip(0.05 + 0.65 * sig + 0.045 * np.sin(2 * np.pi * 9 * x), 0.03, 0.76)
        agg = np.clip(0.16 + 0.035 * np.sin(2 * np.pi * 4 * x + 1.1), 0.08, 0.25)
        scan = np.maximum(1.0 - point - agg, 0.08)
        p = np.column_stack([point, scan, agg])
        return p / p.sum(axis=1, keepdims=True)

    step = 100
    knots = math.ceil(n / step) + 1
    logits = np.zeros((knots, 3), dtype=float)
    logits[0] = [-0.4, 0.9, -0.2]
    center = np.asarray([-0.15, 0.50, -0.20])
    for i in range(1, knots):
        logits[i] = 0.985 * logits[i - 1] + 0.015 * center + rng.normal(0, 0.075, 3)
        logits[i, 0] += 0.08 * math.sin(2 * math.pi * i / 170)
        logits[i, 2] += 0.06 * math.sin(2 * math.pi * i / 113 + 0.7)

    if trace == "R1_C_noisy_bursts":
        positions = np.arange(knots, dtype=float)
        for _ in range(24):
            center_i = int(rng.integers(10, knots - 10))
            width = float(rng.uniform(2.0, 12.0))
            category = int(rng.integers(0, 3))
            amp = float(rng.uniform(0.65, 1.60))
            logits[:, category] += amp * np.exp(-0.5 * ((positions - center_i) / width) ** 2)

    knot_pos = np.arange(knots) * step
    event_pos = np.arange(n)
    interpolated = np.column_stack(
        [np.interp(event_pos, knot_pos, logits[:, j]) for j in range(3)]
    )
    return softmax(interpolated)


def class_probabilities(category_p: np.ndarray, query_classes: Sequence[str]) -> np.ndarray:
    out = np.zeros((len(category_p), len(query_classes)), dtype=float)
    for cat_i, cat in enumerate(("point", "scan", "aggregate")):
        for qclass, weight in CATEGORY_CLASS_WEIGHTS[cat].items():
            out[:, query_classes.index(qclass)] += category_p[:, cat_i] * weight
    return out / out.sum(axis=1, keepdims=True)


def access_probabilities(trace: str, n: int, n_chunks: int) -> np.ndarray:
    block = 100
    n_blocks = math.ceil(n / block)
    rng = np.random.default_rng(stable_seed(SEED, trace, "access"))
    base = rng.normal(0, 0.12, n_chunks)
    phase = rng.uniform(0, 2 * np.pi, n_chunks)
    period = rng.uniform(180, 520, n_chunks)
    ar = np.zeros((n_blocks, n_chunks), dtype=float)
    for i in range(1, n_blocks):
        ar[i] = 0.992 * ar[i - 1] + rng.normal(0, 0.012, n_chunks)
    t = np.arange(n_blocks)[:, None]
    logits = base[None, :] + 0.22 * np.sin(2 * np.pi * t / period[None, :] + phase[None, :]) + ar
    return softmax(logits)


def build_or_load_plan(
    output: Path,
    core: Any,
    chunks: Sequence[Any],
    query_pools: Mapping[str, Mapping[str, List[Any]]],
    trace: str,
    size: int,
) -> Tuple[Any, Dict[str, Any]]:
    trace_dir = output / "frozen_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    npz_path = trace_dir / f"{trace}__size_{size}.npz"
    json_path = trace_dir / f"{trace}__size_{size}.json"

    size_chunks = sorted(
        [c for c in chunks if int(c.chunk_size) == size],
        key=lambda c: c.chunk_id,
    )
    chunk_ids = tuple(c.chunk_id for c in size_chunks)
    if len(chunk_ids) != N_BUILDINGS:
        raise AssertionError(f"Expected 32 chunks at size {size}")

    if npz_path.exists() and json_path.exists():
        with np.load(npz_path, allow_pickle=False) as a:
            cp = np.asarray(a["chunk_positions"], dtype=np.int16)
            li = np.asarray(a["local_indices"], dtype=np.int32)
            qp = np.asarray(a["query_class_positions"], dtype=np.uint8)
            qc = np.asarray(a["query_choice_positions"], dtype=np.uint16)
        manifest = json.loads(json_path.read_text(encoding="utf-8"))
        if manifest["trace_file_sha256"] != sha256_file(npz_path):
            raise AssertionError("Frozen trace hash mismatch")
    else:
        qclasses = tuple(core.QUERY_CLASSES)
        rng = np.random.default_rng(stable_seed(SEED, trace, size, "sampling"))
        cp = np.empty(N_EVENTS, dtype=np.int16)
        access = access_probabilities(trace, N_EVENTS, len(chunk_ids))
        for start in range(0, N_EVENTS, 100):
            end = min(N_EVENTS, start + 100)
            cp[start:end] = rng.choice(
                len(chunk_ids), size=end - start, p=access[start // 100]
            ).astype(np.int16)

        probs = class_probabilities(category_probabilities(trace, N_EVENTS), qclasses)
        cumulative = np.cumsum(probs, axis=1)
        qp = np.sum(rng.random(N_EVENTS)[:, None] > cumulative, axis=1).astype(np.uint8)
        qp = np.minimum(qp, len(qclasses) - 1)

        li = np.empty(N_EVENTS, dtype=np.int32)
        qc = np.empty(N_EVENTS, dtype=np.uint16)
        counters = np.zeros(len(chunk_ids), dtype=np.int32)
        for i in range(N_EVENTS):
            pos = int(cp[i])
            li[i] = counters[pos]
            counters[pos] += 1
            qclass = qclasses[int(qp[i])]
            qc[i] = int(rng.integers(0, len(query_pools[chunk_ids[pos]][qclass])))

        tmp = npz_path.with_name(npz_path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez_compressed(
                f,
                chunk_positions=cp,
                local_indices=li,
                query_class_positions=qp,
                query_choice_positions=qc,
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, npz_path)

        local_counts = np.bincount(cp, minlength=len(chunk_ids))
        manifest = {
            "trace_name": trace,
            "chunk_size": size,
            "n_chunks": len(chunk_ids),
            "n_events": N_EVENTS,
            "expected_local_queries_per_building": LOCAL_QUERIES_PER_BUILDING,
            "minimum_local_queries": int(local_counts.min()),
            "median_local_queries": float(np.median(local_counts)),
            "maximum_local_queries": int(local_counts.max()),
            "query_class_counts": {
                qclasses[j]: int(np.sum(qp == j)) for j in range(len(qclasses))
            },
            "trace_file_sha256": sha256_file(npz_path),
            "phase_boundaries_provided_to_policy": False,
            "future_probabilities_visible_to_policy": False,
        }
        write_json(json_path, manifest)

    plan = core.LightweightTracePlan(
        trace_name=trace,
        trace_seed=int(stable_seed(SEED, trace, size) % (2**31 - 1)),
        chunk_size=size,
        chunk_ids=chunk_ids,
        chunk_positions=cp,
        local_indices=li,
        query_class_positions=qp,
        query_choice_positions=qc,
        workload_positions=np.zeros(len(cp), dtype=np.uint8),
        workload_names=("continuous_nonstationary",),
    )
    return plan, manifest


def static_result(
    plan: Any,
    profile: np.ndarray,
    positions: np.ndarray,
    policy: str,
) -> Dict[str, Any]:
    n_chunks = len(plan.chunk_ids)
    per_query = np.zeros(n_chunks, dtype=float)
    cumulative: List[Dict[str, Any]] = []
    running = 0.0
    for i in range(plan.n_events):
        c = int(plan.chunk_positions[i])
        q = int(plan.query_class_positions[i])
        cost = float(profile[c, int(positions[c]), q])
        per_query[c] += cost
        running += cost
        if i % CUMULATIVE_EVERY == 0 or i == plan.n_events - 1:
            cumulative.append(
                {
                    "global_event_index": i,
                    "policy": policy,
                    "cumulative_query_time_ns": running,
                    "cumulative_migration_time_ns": 0.0,
                    "cumulative_total_time_ns": running,
                    "cumulative_migrations": 0,
                    "cumulative_budget_denials": 0,
                }
            )
    total = float(per_query.sum())
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
        "per_chunk_query_time_ns": per_query,
        "per_chunk_migration_time_ns": np.zeros(n_chunks, dtype=float),
        "per_chunk_migration_count": np.zeros(n_chunks, dtype=np.int64),
        "per_chunk_rewrite_bytes": np.zeros(n_chunks, dtype=np.int64),
        "migration_records": [],
        "decision_records": [],
        "cumulative_records": cumulative,
    }


def best_static_positions(plan: Any, profile: np.ndarray) -> np.ndarray:
    n_chunks, _, n_classes = profile.shape
    counts = np.zeros((n_chunks, n_classes), dtype=np.int64)
    np.add.at(
        counts,
        (
            plan.chunk_positions.astype(np.int64),
            plan.query_class_positions.astype(np.int64),
        ),
        1,
    )
    costs = np.einsum("ncq,nq->nc", profile, counts, optimize=True)
    return np.argmin(costs, axis=1).astype(np.int16)


def oracle_result(
    plan: Any,
    profile: np.ndarray,
    initial: np.ndarray,
    migration_cost: np.ndarray,
    migration_bytes: np.ndarray,
) -> Dict[str, Any]:
    """
    Non-deployable hindsight lower bound.

    Each chunk is partitioned into its own consecutive 100-query blocks.
    Dynamic programming chooses one codec per block and includes the measured
    directed migration cost between consecutive block layouts.
    """
    n_chunks, n_codecs, _ = profile.shape
    per_query = np.zeros(n_chunks, dtype=float)
    per_migration = np.zeros(n_chunks, dtype=float)
    per_count = np.zeros(n_chunks, dtype=np.int64)
    per_rewrite = np.zeros(n_chunks, dtype=np.int64)
    schedules: List[Dict[int, int]] = []

    for c in range(n_chunks):
        event_indices = np.flatnonzero(plan.chunk_positions == c)
        sequence = plan.query_class_positions[event_indices].astype(np.int64)
        n_blocks = max(1, math.ceil(len(sequence) / DECISION_INTERVAL))
        block_cost = np.zeros((n_blocks, n_codecs), dtype=float)
        for b in range(n_blocks):
            start = b * DECISION_INTERVAL
            end = min(len(sequence), start + DECISION_INTERVAL)
            counts = np.bincount(sequence[start:end], minlength=profile.shape[2])
            block_cost[b] = profile[c] @ counts

        dp = np.full((n_blocks, n_codecs), np.inf)
        previous = np.full((n_blocks, n_codecs), -1, dtype=np.int16)
        source0 = int(initial[c])
        for target in range(n_codecs):
            dp[0, target] = block_cost[0, target] + migration_cost[c, source0, target]
            previous[0, target] = source0

        for b in range(1, n_blocks):
            transitions = dp[b - 1][:, None] + migration_cost[c]
            previous[b] = np.argmin(transitions, axis=0).astype(np.int16)
            dp[b] = block_cost[b] + np.min(transitions, axis=0)

        path = np.empty(n_blocks, dtype=np.int16)
        path[-1] = int(np.argmin(dp[-1]))
        for b in range(n_blocks - 1, 0, -1):
            path[b - 1] = previous[b, path[b]]

        active = source0
        schedule: Dict[int, int] = {}
        query_total = 0.0
        migration_total = 0.0
        for b in range(n_blocks):
            target = int(path[b])
            if target != active:
                migration_total += float(migration_cost[c, active, target])
                per_count[c] += 1
                per_rewrite[c] += int(migration_bytes[c, active, target])
                schedule[b * DECISION_INTERVAL] = target
                active = target
            query_total += float(block_cost[b, active])
        per_query[c] = query_total
        per_migration[c] = migration_total
        schedules.append(schedule)

    active = initial.copy()
    local_count = np.zeros(n_chunks, dtype=np.int32)
    running_query = 0.0
    running_migration = 0.0
    running_count = 0
    cumulative: List[Dict[str, Any]] = []
    for i in range(plan.n_events):
        c = int(plan.chunk_positions[i])
        local = int(local_count[c])
        if local in schedules[c]:
            target = int(schedules[c][local])
            source = int(active[c])
            running_migration += float(migration_cost[c, source, target])
            running_count += 1
            active[c] = target
        q = int(plan.query_class_positions[i])
        running_query += float(profile[c, int(active[c]), q])
        local_count[c] += 1
        if i % CUMULATIVE_EVERY == 0 or i == plan.n_events - 1:
            cumulative.append(
                {
                    "global_event_index": i,
                    "policy": "hindsight_oracle",
                    "cumulative_query_time_ns": running_query,
                    "cumulative_migration_time_ns": running_migration,
                    "cumulative_total_time_ns": running_query + running_migration,
                    "cumulative_migrations": running_count,
                    "cumulative_budget_denials": 0,
                }
            )

    tq = float(per_query.sum())
    tm = float(per_migration.sum())
    return {
        "query_time_ns": tq,
        "migration_time_ns": tm,
        "total_time_ns": tq + tm,
        "migration_count": int(per_count.sum()),
        "rewrite_bytes": int(per_rewrite.sum()),
        "budget_denied_proposals": 0,
        "budget_violation_count": 0,
        "unamortized_migration_count": 0,
        "unamortized_migration_rate": 0.0,
        "stale_assessable_migration_count": 0,
        "stale_migration_count": 0,
        "stale_migration_rate": 0.0,
        "per_chunk_query_time_ns": per_query,
        "per_chunk_migration_time_ns": per_migration,
        "per_chunk_migration_count": per_count,
        "per_chunk_rewrite_bytes": per_rewrite,
        "migration_records": [],
        "decision_records": [],
        "cumulative_records": cumulative,
    }


@dataclasses.dataclass
class OnlineState:
    active_position: int
    local_access_count: int = 0
    ewma: Optional[np.ndarray] = None
    repress_state: Optional[Any] = None
    query_time_ns: float = 0.0
    migration_time_ns: float = 0.0
    migration_count: int = 0
    rewrite_bytes: int = 0
    open_episode: Optional[Dict[str, Any]] = None
    migration_records: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    decision_records: List[Dict[str, Any]] = dataclasses.field(default_factory=list)


def close_episode(state: OnlineState, end_event: int) -> None:
    if state.open_episode is None:
        return
    episode = state.open_episode
    episode["end_global_event_index"] = int(end_event)
    episode["unamortized"] = bool(
        float(episode["realized_query_saving_ns"])
        < float(episode["migration_cost_ns"])
    )
    state.migration_records.append(episode)
    state.open_episode = None


def assign_stale_flags(states: Sequence[OnlineState]) -> None:
    """
    A migration is stale only when the target is not the estimated best codec
    at both of the first two later decision opportunities for that chunk.
    """
    for state in states:
        for migration in state.migration_records:
            later = [
                row
                for row in state.decision_records
                if int(row["global_event_index"])
                > int(migration["start_global_event_index"])
            ][:2]
            migration["stale_assessable"] = len(later) == 2
            migration["stale"] = bool(
                len(later) == 2
                and all(
                    str(row["estimated_best_codec"])
                    != str(migration["target_codec"])
                    for row in later
                )
            )


def online_result(
    policy: str,
    context: Mapping[str, Any],
    plan: Any,
    profile: np.ndarray,
    initial: np.ndarray,
    migration_cost: np.ndarray,
    migration_bytes: np.ndarray,
    codecs: Sequence[str],
    query_classes: Sequence[str],
    shape_config: Mapping[str, Any],
    core: Any,
    e1: Any,
    e1v2: Any,
    base: Any,
    budget_bytes: int,
) -> Dict[str, Any]:
    if policy not in ("shape_ewma_be_frozen", "repressdb_frozen"):
        raise ValueError(policy)

    init_weights = context["s1"].normalized_initial_weights(base, query_classes)
    states: List[OnlineState] = []
    for c, chunk_id in enumerate(plan.chunk_ids):
        active = int(initial[c])
        if policy == "shape_ewma_be_frozen":
            states.append(OnlineState(active_position=active, ewma=init_weights.copy()))
        else:
            states.append(
                OnlineState(
                    active_position=active,
                    repress_state=core.RepressState(active_codec=codecs[active]),
                )
            )

    alpha = float(shape_config["alpha"])
    deadband = float(shape_config["deadband"])
    horizon = int(shape_config["prediction_horizon"])

    interval_id = 0
    remaining = int(budget_bytes)
    spent = 0
    denials = 0
    violations = 0
    running_query = 0.0
    running_migration = 0.0
    cumulative: List[Dict[str, Any]] = []

    for i in range(plan.n_events):
        new_interval = i // BUDGET_INTERVAL
        if new_interval != interval_id:
            if spent > budget_bytes:
                violations += 1
            interval_id = new_interval
            remaining = int(budget_bytes)
            spent = 0

        c = int(plan.chunk_positions[i])
        q = int(plan.query_class_positions[i])
        qclass = query_classes[q]
        state = states[c]
        current = int(state.active_position)
        query_ns = float(profile[c, current, q])
        state.query_time_ns += query_ns
        running_query += query_ns

        if state.open_episode is not None:
            source = int(state.open_episode["source_position"])
            target = int(state.open_episode["target_position"])
            state.open_episode["realized_query_saving_ns"] += float(
                profile[c, source, q] - profile[c, target, q]
            )

        state.local_access_count += 1
        proposal: Optional[int] = None
        estimated_best = current
        reason = ""
        relative = 0.0
        predicted_saving = 0.0
        migration_estimate = 0.0
        decision_due = False

        if policy == "shape_ewma_be_frozen":
            assert state.ewma is not None
            state.ewma *= 1.0 - alpha
            state.ewma[q] += alpha
            decision_due = (
                state.local_access_count >= MIN_OBSERVATIONS
                and state.local_access_count % DECISION_INTERVAL == 0
            )
            if decision_due:
                expected = profile[c] @ state.ewma
                estimated_best = int(np.argmin(expected))
                current_cost = float(expected[current])
                target_cost = float(expected[estimated_best])
                gain = current_cost - target_cost
                relative = gain / current_cost if current_cost > 0 else 0.0
                migration_estimate = float(
                    migration_cost[c, current, estimated_best]
                )
                predicted_saving = gain * horizon
                if estimated_best == current:
                    reason = "current_layout_is_best"
                elif relative < deadband:
                    reason = "deadband"
                elif predicted_saving <= migration_estimate:
                    reason = "break_even_not_met"
                else:
                    reason = "proposal"
                    proposal = estimated_best
        else:
            assert state.repress_state is not None
            rs = state.repress_state
            rs.recent_classes.append(qclass)
            rs.local_access_count += 1
            decision_due = (
                rs.local_access_count >= core.MIN_OBSERVATIONS
                and rs.local_access_count % core.DECISION_INTERVAL == 0
            )
            if decision_due:
                candidate, detail = core.decide_repressdb_model(
                    e1,
                    e1v2,
                    context["profiles"][plan.chunk_ids[c]],
                    rs,
                    rs.local_access_count - 1,
                    context["encode"][plan.chunk_ids[c]],
                    context["decode"][plan.chunk_ids[c]],
                )
                short_weights = e1v2.observed_weights(
                    list(rs.recent_classes)[-core.SHORT_WINDOW :],
                    e1.QUERY_CLASSES,
                )
                best_codec, _ = e1.choose_best_codec(
                    context["profiles"][plan.chunk_ids[c]],
                    short_weights,
                )
                estimated_best = codecs.index(best_codec)
                reason = str(detail["reason"])
                relative = float(detail["relative_improvement"])
                predicted_saving = float(detail["predicted_saving_ns"])
                migration_estimate = float(detail["rewrite_estimate_ns"])
                if candidate is not None:
                    proposal = codecs.index(candidate)

        if decision_due:
            row = {
                "global_event_index": i,
                "chunk_id": str(plan.chunk_ids[c]),
                "local_query_index": state.local_access_count - 1,
                "old_codec": str(codecs[current]),
                "estimated_best_codec": str(codecs[estimated_best]),
                "candidate_codec": (
                    str(codecs[proposal]) if proposal is not None else str(codecs[current])
                ),
                "reason": reason,
                "relative_improvement": relative,
                "predicted_saving_ns": predicted_saving,
                "migration_estimate_ns": migration_estimate,
                "budget_remaining_before": remaining,
                "decision": "stay",
            }

            if proposal is not None and proposal != current:
                rewrite = int(migration_bytes[c, current, proposal])
                if rewrite <= remaining:
                    close_episode(state, i)
                    migration_ns = float(migration_cost[c, current, proposal])
                    state.migration_time_ns += migration_ns
                    state.migration_count += 1
                    state.rewrite_bytes += rewrite
                    state.active_position = proposal
                    running_migration += migration_ns
                    remaining -= rewrite
                    spent += rewrite
                    row["decision"] = "switch"
                    state.open_episode = {
                        "chunk_id": str(plan.chunk_ids[c]),
                        "policy": policy,
                        "start_global_event_index": i,
                        "start_local_query_index": state.local_access_count - 1,
                        "source_position": current,
                        "target_position": proposal,
                        "source_codec": str(codecs[current]),
                        "target_codec": str(codecs[proposal]),
                        "migration_cost_ns": migration_ns,
                        "rewrite_bytes": rewrite,
                        "realized_query_saving_ns": 0.0,
                        "end_global_event_index": None,
                        "unamortized": None,
                    }
                    if policy == "repressdb_frozen" and state.repress_state is not None:
                        state.repress_state.active_codec = codecs[proposal]
                        state.repress_state.last_switch_local_index = (
                            state.repress_state.local_access_count - 1
                        )
                else:
                    denials += 1
                    row["decision"] = "budget_denied"
                    row["reason"] = "write_budget"
            row["budget_remaining_after"] = remaining
            state.decision_records.append(row)

        if i % CUMULATIVE_EVERY == 0 or i == plan.n_events - 1:
            cumulative.append(
                {
                    "global_event_index": i,
                    "policy": policy,
                    "cumulative_query_time_ns": running_query,
                    "cumulative_migration_time_ns": running_migration,
                    "cumulative_total_time_ns": running_query + running_migration,
                    "cumulative_migrations": int(sum(s.migration_count for s in states)),
                    "cumulative_budget_denials": denials,
                }
            )

    if spent > budget_bytes:
        violations += 1
    for state in states:
        close_episode(state, plan.n_events)
    assign_stale_flags(states)

    migrations = [row for state in states for row in state.migration_records]
    decisions = [row for state in states for row in state.decision_records]
    per_query = np.asarray([s.query_time_ns for s in states], dtype=float)
    per_migration = np.asarray([s.migration_time_ns for s in states], dtype=float)
    per_count = np.asarray([s.migration_count for s in states], dtype=np.int64)
    per_rewrite = np.asarray([s.rewrite_bytes for s in states], dtype=np.int64)
    unamortized = sum(bool(row["unamortized"]) for row in migrations)
    assessable = sum(bool(row.get("stale_assessable", False)) for row in migrations)
    stale = sum(bool(row.get("stale", False)) for row in migrations)
    tq = float(per_query.sum())
    tm = float(per_migration.sum())
    return {
        "query_time_ns": tq,
        "migration_time_ns": tm,
        "total_time_ns": tq + tm,
        "migration_count": int(per_count.sum()),
        "rewrite_bytes": int(per_rewrite.sum()),
        "budget_denied_proposals": int(denials),
        "budget_violation_count": int(violations),
        "unamortized_migration_count": int(unamortized),
        "unamortized_migration_rate": (
            float(unamortized / len(migrations)) if migrations else 0.0
        ),
        "stale_assessable_migration_count": int(assessable),
        "stale_migration_count": int(stale),
        "stale_migration_rate": float(stale / assessable) if assessable else 0.0,
        "per_chunk_query_time_ns": per_query,
        "per_chunk_migration_time_ns": per_migration,
        "per_chunk_migration_count": per_count,
        "per_chunk_rewrite_bytes": per_rewrite,
        "migration_records": migrations,
        "decision_records": decisions,
        "cumulative_records": cumulative,
    }


def afhc_result(
    context: Mapping[str, Any],
    plan: Any,
    profile: np.ndarray,
    initial: np.ndarray,
    migration_cost: np.ndarray,
    migration_bytes: np.ndarray,
    codecs: Sequence[str],
    frozen: Mapping[str, Any],
    core: Any,
    budget_bytes: int,
) -> Dict[str, Any]:
    n = len(plan.chunk_ids)
    active = initial.copy()
    query_time = np.zeros(n, dtype=float)
    migration_time = np.zeros(n, dtype=float)
    migration_count = np.zeros(n, dtype=np.int64)
    rewrite_total = np.zeros(n, dtype=np.int64)
    scores = np.zeros(n, dtype=float)
    period_count = np.zeros(n, dtype=float)

    period = int(frozen["adaptation_period"])
    hot_fraction = float(frozen["hot_fraction"])
    hot_pos = codecs.index(str(frozen["hot_codec"]))
    cold_pos = codecs.index(str(frozen["cold_codec"]))

    interval_id = 0
    remaining = int(budget_bytes)
    spent = 0
    denials = 0
    violations = 0
    running_query = 0.0
    running_migration = 0.0
    cumulative: List[Dict[str, Any]] = []

    for i in range(plan.n_events):
        new_interval = i // BUDGET_INTERVAL
        if new_interval != interval_id:
            if spent > budget_bytes:
                violations += 1
            interval_id = new_interval
            remaining = int(budget_bytes)
            spent = 0

        c = int(plan.chunk_positions[i])
        q = int(plan.query_class_positions[i])
        value = float(profile[c, int(active[c]), q])
        query_time[c] += value
        running_query += value
        period_count[c] += 1.0

        boundary = (i + 1) % period == 0 or i == plan.n_events - 1
        if boundary:
            scores = (
                core.FREQUENCY_EMA_ALPHA * period_count
                + (1.0 - core.FREQUENCY_EMA_ALPHA) * scores
            )
            hot_mask, cold_mask, _ = core.classify_hot_cold_deadband(
                scores,
                hot_fraction,
                core.AFHC_DEADBAND_FACTOR,
            )
            target = active.copy()
            target[hot_mask] = hot_pos
            target[cold_mask] = cold_pos
            for c2 in np.flatnonzero(target != active):
                source = int(active[c2])
                destination = int(target[c2])
                rewrite = int(migration_bytes[c2, source, destination])
                if rewrite <= remaining:
                    migration_ns = float(
                        migration_cost[c2, source, destination]
                    )
                    migration_time[c2] += migration_ns
                    migration_count[c2] += 1
                    rewrite_total[c2] += rewrite
                    running_migration += migration_ns
                    remaining -= rewrite
                    spent += rewrite
                    active[c2] = destination
                else:
                    denials += 1
            period_count[:] = 0.0

        if i % CUMULATIVE_EVERY == 0 or i == plan.n_events - 1:
            cumulative.append(
                {
                    "global_event_index": i,
                    "policy": "afhc_d_frozen",
                    "cumulative_query_time_ns": running_query,
                    "cumulative_migration_time_ns": running_migration,
                    "cumulative_total_time_ns": running_query + running_migration,
                    "cumulative_migrations": int(migration_count.sum()),
                    "cumulative_budget_denials": denials,
                }
            )

    if spent > budget_bytes:
        violations += 1
    tq = float(query_time.sum())
    tm = float(migration_time.sum())
    return {
        "query_time_ns": tq,
        "migration_time_ns": tm,
        "total_time_ns": tq + tm,
        "migration_count": int(migration_count.sum()),
        "rewrite_bytes": int(rewrite_total.sum()),
        "budget_denied_proposals": int(denials),
        "budget_violation_count": int(violations),
        "unamortized_migration_count": 0,
        "unamortized_migration_rate": math.nan,
        "stale_assessable_migration_count": 0,
        "stale_migration_count": 0,
        "stale_migration_rate": math.nan,
        "per_chunk_query_time_ns": query_time,
        "per_chunk_migration_time_ns": migration_time,
        "per_chunk_migration_count": migration_count,
        "per_chunk_rewrite_bytes": rewrite_total,
        "migration_records": [],
        "decision_records": [],
        "cumulative_records": cumulative,
    }



def common_stale_diagnostic(
    plan: Any,
    result: Dict[str, Any],
    profile: np.ndarray,
    codecs: Sequence[str],
) -> None:
    """
    Apply one policy-independent stale-migration diagnostic.

    At every common local decision point (>=200 observations and divisible by
    100), estimate the query-class distribution from the most recent 100
    observed local queries. The best codec is the one with minimum expected
    measured query cost under that distribution.

    A migration is stale when its target is not this best codec at both of the
    first two later common decision points for the same chunk.
    """
    records = result.get("migration_records", [])
    if not records:
        result["stale_assessable_migration_count"] = 0
        result["stale_migration_count"] = 0
        result["stale_migration_rate"] = 0.0
        return

    decision_map: Dict[str, List[Tuple[int, str]]] = {
        str(chunk_id): [] for chunk_id in plan.chunk_ids
    }
    local_history: Dict[str, List[int]] = {
        str(chunk_id): [] for chunk_id in plan.chunk_ids
    }

    for event_index in range(plan.n_events):
        chunk_position = int(plan.chunk_positions[event_index])
        chunk_id = str(plan.chunk_ids[chunk_position])
        query_position = int(plan.query_class_positions[event_index])
        history = local_history[chunk_id]
        history.append(query_position)
        local_count = len(history)
        if (
            local_count >= MIN_OBSERVATIONS
            and local_count % DECISION_INTERVAL == 0
        ):
            recent = np.asarray(
                history[-DECISION_INTERVAL:],
                dtype=np.int64,
            )
            weights = np.bincount(
                recent,
                minlength=profile.shape[2],
            ).astype(np.float64)
            weights /= weights.sum()
            expected = profile[chunk_position] @ weights
            best_codec = str(codecs[int(np.argmin(expected))])
            decision_map[chunk_id].append(
                (int(event_index), best_codec)
            )

    assessable = 0
    stale = 0
    for migration in records:
        chunk_id = str(migration["chunk_id"])
        start_event = int(migration["start_global_event_index"])
        target_codec = str(migration["target_codec"])
        later = [
            item
            for item in decision_map[chunk_id]
            if item[0] > start_event
        ][:2]
        migration["stale_assessable"] = len(later) == 2
        migration["stale_diagnostic"] = (
            "common_recent_100_query_cost"
        )
        migration["later_best_codecs"] = [
            codec for _, codec in later
        ]
        migration["stale"] = bool(
            len(later) == 2
            and all(
                best_codec != target_codec
                for _, best_codec in later
            )
        )
        assessable += int(migration["stale_assessable"])
        stale += int(migration["stale"])

    result["stale_assessable_migration_count"] = int(assessable)
    result["stale_migration_count"] = int(stale)
    result["stale_migration_rate"] = (
        float(stale / assessable)
        if assessable
        else 0.0
    )


def afhc_result(
    context: Mapping[str, Any],
    plan: Any,
    profile: np.ndarray,
    initial: np.ndarray,
    migration_cost: np.ndarray,
    migration_bytes: np.ndarray,
    codecs: Sequence[str],
    frozen: Mapping[str, Any],
    core: Any,
    budget_bytes: int,
) -> Dict[str, Any]:
    """
    Frozen AF-HC-D with the same external hard write budget.

    Unlike the earlier compact implementation, this version also records
    migration episodes so unamortized and common stale diagnostics can be
    computed under the same definitions used for the other online policies.
    """
    n = len(plan.chunk_ids)
    states = [
        OnlineState(active_position=int(initial[c]))
        for c in range(n)
    ]
    scores = np.zeros(n, dtype=np.float64)
    period_counts = np.zeros(n, dtype=np.float64)

    period = int(frozen["adaptation_period"])
    hot_fraction = float(frozen["hot_fraction"])
    hot_position = codecs.index(str(frozen["hot_codec"]))
    cold_position = codecs.index(str(frozen["cold_codec"]))

    interval_id = 0
    remaining = int(budget_bytes)
    spent = 0
    denials = 0
    violations = 0
    running_query = 0.0
    running_migration = 0.0
    cumulative: List[Dict[str, Any]] = []

    for event_index in range(plan.n_events):
        new_interval = event_index // BUDGET_INTERVAL
        if new_interval != interval_id:
            if spent > budget_bytes:
                violations += 1
            interval_id = new_interval
            remaining = int(budget_bytes)
            spent = 0

        chunk_position = int(plan.chunk_positions[event_index])
        query_position = int(plan.query_class_positions[event_index])
        state = states[chunk_position]
        current_position = int(state.active_position)
        query_ns = float(
            profile[
                chunk_position,
                current_position,
                query_position,
            ]
        )
        state.query_time_ns += query_ns
        state.local_access_count += 1
        running_query += query_ns
        period_counts[chunk_position] += 1.0

        if state.open_episode is not None:
            source_position = int(
                state.open_episode["source_position"]
            )
            target_position = int(
                state.open_episode["target_position"]
            )
            state.open_episode[
                "realized_query_saving_ns"
            ] += float(
                profile[
                    chunk_position,
                    source_position,
                    query_position,
                ]
                - profile[
                    chunk_position,
                    target_position,
                    query_position,
                ]
            )

        boundary = bool(
            (event_index + 1) % period == 0
            or event_index == plan.n_events - 1
        )
        if boundary:
            scores = (
                core.FREQUENCY_EMA_ALPHA
                * period_counts
                + (
                    1.0
                    - core.FREQUENCY_EMA_ALPHA
                )
                * scores
            )
            hot_mask, cold_mask, _ = (
                core.classify_hot_cold_deadband(
                    scores,
                    hot_fraction,
                    core.AFHC_DEADBAND_FACTOR,
                )
            )
            target_positions = np.asarray(
                [
                    hot_position
                    if hot_mask[c]
                    else (
                        cold_position
                        if cold_mask[c]
                        else states[c].active_position
                    )
                    for c in range(n)
                ],
                dtype=np.int16,
            )

            for c in range(n):
                local_state = states[c]
                source = int(local_state.active_position)
                target = int(target_positions[c])
                decision_row = {
                    "global_event_index": int(event_index),
                    "chunk_id": str(plan.chunk_ids[c]),
                    "local_query_index": int(
                        local_state.local_access_count - 1
                    ),
                    "old_codec": str(codecs[source]),
                    "estimated_best_codec": str(codecs[target]),
                    "candidate_codec": str(codecs[target]),
                    "reason": (
                        "frequency_deadband_hot"
                        if hot_mask[c]
                        else (
                            "frequency_deadband_cold"
                            if cold_mask[c]
                            else "frequency_deadband_neutral"
                        )
                    ),
                    "relative_improvement": math.nan,
                    "predicted_saving_ns": math.nan,
                    "migration_estimate_ns": (
                        float(
                            migration_cost[c, source, target]
                        )
                        if target != source
                        else 0.0
                    ),
                    "budget_remaining_before": int(remaining),
                    "decision": "stay",
                }

                if target != source:
                    rewrite = int(
                        migration_bytes[c, source, target]
                    )
                    if rewrite <= remaining:
                        close_episode(local_state, event_index)
                        migration_ns = float(
                            migration_cost[c, source, target]
                        )
                        local_state.migration_time_ns += (
                            migration_ns
                        )
                        local_state.migration_count += 1
                        local_state.rewrite_bytes += rewrite
                        local_state.active_position = target
                        running_migration += migration_ns
                        remaining -= rewrite
                        spent += rewrite
                        decision_row["decision"] = "switch"
                        local_state.open_episode = {
                            "chunk_id": str(plan.chunk_ids[c]),
                            "policy": "afhc_d_frozen",
                            "start_global_event_index": int(
                                event_index
                            ),
                            "start_local_query_index": int(
                                local_state.local_access_count
                                - 1
                            ),
                            "source_position": source,
                            "target_position": target,
                            "source_codec": str(codecs[source]),
                            "target_codec": str(codecs[target]),
                            "migration_cost_ns": migration_ns,
                            "rewrite_bytes": rewrite,
                            "realized_query_saving_ns": 0.0,
                            "end_global_event_index": None,
                            "unamortized": None,
                        }
                    else:
                        denials += 1
                        decision_row["decision"] = "budget_denied"
                        decision_row["reason"] = "write_budget"
                decision_row["budget_remaining_after"] = int(
                    remaining
                )
                local_state.decision_records.append(
                    decision_row
                )
            period_counts[:] = 0.0

        if (
            event_index % CUMULATIVE_EVERY == 0
            or event_index == plan.n_events - 1
        ):
            cumulative.append(
                {
                    "global_event_index": int(event_index),
                    "policy": "afhc_d_frozen",
                    "cumulative_query_time_ns": float(
                        running_query
                    ),
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
                    "cumulative_budget_denials": int(
                        denials
                    ),
                }
            )

    if spent > budget_bytes:
        violations += 1
    for state in states:
        close_episode(state, plan.n_events)

    migrations = [
        row
        for state in states
        for row in state.migration_records
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
            bool(row["unamortized"])
            for row in migrations
        )
    )
    total_query = float(per_query.sum())
    total_migration = float(per_migration.sum())
    return {
        "query_time_ns": total_query,
        "migration_time_ns": total_migration,
        "total_time_ns": total_query + total_migration,
        "migration_count": int(per_count.sum()),
        "rewrite_bytes": int(per_rewrite.sum()),
        "budget_denied_proposals": int(denials),
        "budget_violation_count": int(violations),
        "unamortized_migration_count": int(
            unamortized
        ),
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

def run_model_experiment(
    context: Mapping[str, Any],
    plans: Mapping[Tuple[str, int], Any],
    shape_config: Mapping[str, Any],
    afhc_config: Mapping[str, Any],
    core: Any,
    e1: Any,
    e1v2: Any,
    base: Any,
    output: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checkpoint_root = output / "_model_checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    codecs = tuple(sorted(base.representation_registry(3)))
    qclasses = tuple(core.QUERY_CLASSES)

    global_frames: List[pd.DataFrame] = []
    chunk_frames: List[pd.DataFrame] = []
    migration_frames: List[pd.DataFrame] = []
    decision_frames: List[pd.DataFrame] = []
    cumulative_frames: List[pd.DataFrame] = []

    for size in CHUNK_SIZES:
        for trace in TRACES:
            checkpoint = checkpoint_root / f"{trace}__size_{size}"
            paths = {
                "global": checkpoint / "global.csv",
                "chunk": checkpoint / "chunk.csv",
                "migrations": checkpoint / "migrations.csv",
                "decisions": checkpoint / "decisions.csv",
                "cumulative": checkpoint / "cumulative.csv",
                "complete": checkpoint / "complete.json",
            }
            if all(paths[k].exists() for k in paths):
                global_frames.append(pd.read_csv(paths["global"]))
                chunk_frames.append(pd.read_csv(paths["chunk"]))
                migration_frames.append(pd.read_csv(paths["migrations"]))
                decision_frames.append(pd.read_csv(paths["decisions"]))
                cumulative_frames.append(pd.read_csv(paths["cumulative"]))
                print(f"[R1 model resume] {trace}/{size}")
                continue

            shutil.rmtree(checkpoint, ignore_errors=True)
            checkpoint.mkdir(parents=True, exist_ok=True)
            plan = plans[(trace, size)]
            chunk_ids = tuple(plan.chunk_ids)
            profile = context["s1"].profile_array(
                context["profiles"],
                chunk_ids,
                codecs,
                qclasses,
            )
            initial = context["s1"].scan_best_positions(
                e1,
                base,
                context["profiles"],
                chunk_ids,
                codecs,
            )
            migration_cost, migration_bytes = context["s1"].migration_matrix(
                context["migration_costs"],
                context["migration_bytes"],
                chunk_ids,
                codecs,
            )
            budget_bytes = int(migration_bytes.max())
            if budget_bytes <= 0:
                raise AssertionError("Non-positive shared write budget")

            best = best_static_positions(plan, profile)
            results = {
                "initial_best": static_result(
                    plan, profile, initial, "initial_best"
                ),
                "best_static": static_result(
                    plan, profile, best, "best_static"
                ),
                "afhc_d_frozen": afhc_result(
                    context,
                    plan,
                    profile,
                    initial,
                    migration_cost,
                    migration_bytes,
                    codecs,
                    afhc_config,
                    core,
                    budget_bytes,
                ),
                "shape_ewma_be_frozen": online_result(
                    "shape_ewma_be_frozen",
                    context,
                    plan,
                    profile,
                    initial,
                    migration_cost,
                    migration_bytes,
                    codecs,
                    qclasses,
                    shape_config,
                    core,
                    e1,
                    e1v2,
                    base,
                    budget_bytes,
                ),
                "repressdb_frozen": online_result(
                    "repressdb_frozen",
                    context,
                    plan,
                    profile,
                    initial,
                    migration_cost,
                    migration_bytes,
                    codecs,
                    qclasses,
                    shape_config,
                    core,
                    e1,
                    e1v2,
                    base,
                    budget_bytes,
                ),
                "hindsight_oracle": oracle_result(
                    plan,
                    profile,
                    initial,
                    migration_cost,
                    migration_bytes,
                ),
            }

            for adaptive_policy in ONLINE_POLICIES:
                common_stale_diagnostic(
                    plan,
                    results[adaptive_policy],
                    profile,
                    codecs,
                )

            best_total = float(results["best_static"]["total_time_ns"])
            shape_total = float(
                results["shape_ewma_be_frozen"]["total_time_ns"]
            )
            repress_total = float(
                results["repressdb_frozen"]["total_time_ns"]
            )
            global_rows: List[Dict[str, Any]] = []
            chunk_rows: List[Dict[str, Any]] = []
            migration_rows: List[Dict[str, Any]] = []
            decision_rows: List[Dict[str, Any]] = []
            cumulative_rows: List[Dict[str, Any]] = []

            for policy, result in results.items():
                global_rows.append(
                    {
                        "trace": trace,
                        "chunk_size": size,
                        "policy": policy,
                        "n_buildings": N_BUILDINGS,
                        "n_queries": int(plan.n_events),
                        "budget_interval_global_queries": (
                            BUDGET_INTERVAL if policy in ONLINE_POLICIES else math.nan
                        ),
                        "budget_bytes_per_interval": (
                            budget_bytes if policy in ONLINE_POLICIES else math.nan
                        ),
                        "query_time_ns": float(result["query_time_ns"]),
                        "migration_time_ns": float(result["migration_time_ns"]),
                        "total_time_ns": float(result["total_time_ns"]),
                        "ratio_to_best_static": float(
                            result["total_time_ns"] / best_total
                        ),
                        "ratio_to_shape_ewma": float(
                            result["total_time_ns"] / shape_total
                        ),
                        "ratio_to_repressdb": float(
                            result["total_time_ns"] / repress_total
                        ),
                        "migration_count": int(result["migration_count"]),
                        "rewrite_bytes": int(result["rewrite_bytes"]),
                        "budget_denied_proposals": int(
                            result["budget_denied_proposals"]
                        ),
                        "budget_violation_count": int(
                            result["budget_violation_count"]
                        ),
                        "unamortized_migration_count": int(
                            result["unamortized_migration_count"]
                        ),
                        "unamortized_migration_rate": float(
                            result["unamortized_migration_rate"]
                        ),
                        "stale_assessable_migration_count": int(
                            result["stale_assessable_migration_count"]
                        ),
                        "stale_migration_count": int(
                            result["stale_migration_count"]
                        ),
                        "stale_migration_rate": float(
                            result["stale_migration_rate"]
                        ),
                    }
                )
                for c, chunk_id in enumerate(chunk_ids):
                    meta = context["metadata"][chunk_id]
                    chunk_rows.append(
                        {
                            "trace": trace,
                            "chunk_size": size,
                            "policy": policy,
                            "chunk_id": chunk_id,
                            "building_id": meta["building_id"],
                            "meter_type": meta["meter_type"],
                            "site_id": meta["site_id"],
                            "query_time_ns": float(
                                result["per_chunk_query_time_ns"][c]
                            ),
                            "migration_time_ns": float(
                                result["per_chunk_migration_time_ns"][c]
                            ),
                            "total_time_ns": float(
                                result["per_chunk_query_time_ns"][c]
                                + result["per_chunk_migration_time_ns"][c]
                            ),
                            "migration_count": int(
                                result["per_chunk_migration_count"][c]
                            ),
                            "rewrite_bytes": int(
                                result["per_chunk_rewrite_bytes"][c]
                            ),
                        }
                    )
                for row in result["migration_records"]:
                    migration_rows.append(
                        {
                            "trace": trace,
                            "chunk_size": size,
                            "policy": policy,
                            **row,
                        }
                    )
                for row in result["decision_records"]:
                    decision_rows.append(
                        {
                            "trace": trace,
                            "chunk_size": size,
                            "policy": policy,
                            **row,
                        }
                    )
                for row in result["cumulative_records"]:
                    cumulative_rows.append(
                        {
                            "trace": trace,
                            "chunk_size": size,
                            **row,
                        }
                    )

            frames = {
                "global": pd.DataFrame(global_rows),
                "chunk": pd.DataFrame(chunk_rows),
                "migrations": pd.DataFrame(migration_rows),
                "decisions": pd.DataFrame(decision_rows),
                "cumulative": pd.DataFrame(cumulative_rows),
            }
            for name, frame in frames.items():
                write_csv(paths[name], frame)
            write_json(
                paths["complete"],
                {
                    "trace": trace,
                    "chunk_size": size,
                    "n_queries": int(plan.n_events),
                    "completed_at_ns": time.time_ns(),
                },
            )
            global_frames.append(frames["global"])
            chunk_frames.append(frames["chunk"])
            migration_frames.append(frames["migrations"])
            decision_frames.append(frames["decisions"])
            cumulative_frames.append(frames["cumulative"])
            print(f"[R1 model] {trace}/{size} complete")

    return (
        pd.concat(global_frames, ignore_index=True),
        pd.concat(chunk_frames, ignore_index=True),
        pd.concat(migration_frames, ignore_index=True),
        pd.concat(decision_frames, ignore_index=True),
        pd.concat(cumulative_frames, ignore_index=True),
    )


def clustered_comparisons(chunk_frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    comparisons = (
        ("repressdb_frozen", "shape_ewma_be_frozen"),
        ("repressdb_frozen", "best_static"),
        ("shape_ewma_be_frozen", "best_static"),
        ("repressdb_frozen", "afhc_d_frozen"),
    )
    for (trace, size), group in chunk_frame.groupby(["trace", "chunk_size"]):
        pivot = group.pivot(
            index="building_id",
            columns="policy",
            values="total_time_ns",
        ).dropna()
        for numerator, denominator in comparisons:
            if numerator not in pivot or denominator not in pivot:
                continue
            ratio = (
                pivot[numerator].to_numpy(dtype=float)
                / pivot[denominator].to_numpy(dtype=float)
            )
            low, high = bootstrap_median_ci(
                ratio, trace, size, numerator, denominator
            )
            rows.append(
                {
                    "trace": trace,
                    "chunk_size": int(size),
                    "numerator_policy": numerator,
                    "denominator_policy": denominator,
                    "n_buildings": int(len(pivot)),
                    "median_ratio": float(np.median(ratio)),
                    "geometric_mean_ratio": geometric_mean(ratio),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "fraction_numerator_faster": float(np.mean(ratio < 1.0)),
                }
            )
    return pd.DataFrame(rows)


def filter_plan(core: Any, plan: Any, keep_ids: Sequence[str]) -> Any:
    keep = set(keep_ids)
    old_positions = [
        i for i, chunk_id in enumerate(plan.chunk_ids) if chunk_id in keep
    ]
    remap = {old: new for new, old in enumerate(old_positions)}
    mask = np.isin(
        plan.chunk_positions,
        np.asarray(old_positions, dtype=plan.chunk_positions.dtype),
    )
    mapped = np.asarray(
        [remap[int(x)] for x in plan.chunk_positions[mask]],
        dtype=np.int16,
    )
    return core.LightweightTracePlan(
        trace_name=plan.trace_name,
        trace_seed=plan.trace_seed,
        chunk_size=plan.chunk_size,
        chunk_ids=tuple(plan.chunk_ids[i] for i in old_positions),
        chunk_positions=mapped,
        local_indices=plan.local_indices[mask],
        query_class_positions=plan.query_class_positions[mask],
        query_choice_positions=plan.query_choice_positions[mask],
        workload_positions=plan.workload_positions[mask],
        workload_names=plan.workload_names,
    )


def query_for_event(
    core: Any,
    plan: Any,
    event_index: int,
    query_pools: Mapping[str, Mapping[str, List[Any]]],
) -> Tuple[str, str, Any]:
    c = int(plan.chunk_positions[event_index])
    chunk_id = str(plan.chunk_ids[c])
    qclass = str(
        core.QUERY_CLASSES[int(plan.query_class_positions[event_index])]
    )
    choice = int(plan.query_choice_positions[event_index])
    return chunk_id, qclass, query_pools[chunk_id][qclass][choice]


def reconstruct_stored(
    base: Any,
    chunks: Sequence[Any],
    codec_frame: pd.DataFrame,
    storage_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    output: Dict[str, Dict[str, Any]] = {}
    for chunk_id, group in codec_frame.groupby("chunk_id"):
        cid = str(chunk_id)
        local: Dict[str, Any] = {}
        for row in group.itertuples(index=False):
            codec = str(row.codec)
            stem = storage_dir / cid / codec
            data_path = stem.with_suffix(".bin")
            metadata_candidate = stem.with_suffix(".meta.bin")
            if not data_path.exists():
                raise FileNotFoundError(data_path)
            if metadata_candidate.exists():
                metadata_path: Optional[Path] = metadata_candidate
                metadata = base.parse_block_metadata(metadata_path.read_bytes())
            else:
                metadata_path = None
                metadata = {"level": 3} if codec == "whole_zstd" else {}
            local[codec] = base.StoredChunk(
                codec=codec,
                n=int(by_id[cid].values.size),
                data_path=data_path,
                metadata_path=metadata_path,
                payload_size_bytes=int(row.payload_size_bytes),
                metadata_size_bytes=int(row.metadata_size_bytes),
                metadata=metadata,
            )
        output[cid] = local
    return output


def actual_policy(
    policy: str,
    context: Mapping[str, Any],
    plan: Any,
    shape_config: Mapping[str, Any],
    core: Any,
    e1: Any,
    e1v2: Any,
    base: Any,
    reps: Mapping[str, Any],
    stored: Mapping[str, Mapping[str, Any]],
    migration_root: Path,
    budget_bytes: int,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if policy not in ACTUAL_POLICIES:
        raise ValueError(policy)

    codecs = tuple(sorted(reps))
    qclasses = tuple(core.QUERY_CLASSES)
    codec_pos = {codec: i for i, codec in enumerate(codecs)}
    init_weights = context["s1"].normalized_initial_weights(base, qclasses)
    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunk in context["chunks"]
        if chunk.chunk_id in set(plan.chunk_ids)
    }

    states: Dict[str, OnlineState] = {}
    for chunk_id in plan.chunk_ids:
        initial_codec, _ = e1.choose_best_codec(
            context["profiles"][chunk_id],
            base.WORKLOADS["scan_heavy"],
        )
        position = codec_pos[initial_codec]
        if policy == "shape_ewma_be_frozen":
            states[chunk_id] = OnlineState(
                active_position=position,
                ewma=init_weights.copy(),
            )
        elif policy == "repressdb_frozen":
            states[chunk_id] = OnlineState(
                active_position=position,
                repress_state=core.RepressState(active_codec=initial_codec),
            )
        else:
            states[chunk_id] = OnlineState(active_position=position)

    alpha = float(shape_config["alpha"])
    deadband = float(shape_config["deadband"])
    horizon = int(shape_config["prediction_horizon"])
    mode = base.CACHE_MODES["warm_slicedmeta"]
    latencies: Dict[str, List[int]] = {cid: [] for cid in plan.chunk_ids}
    events: List[Dict[str, Any]] = []
    interval_id = 0
    remaining = int(budget_bytes)
    spent = 0
    denials = 0
    violations = 0

    with core.MultiChunkHandles(stored) as handles:
        first_event: Dict[str, int] = {}
        for i in range(plan.n_events):
            chunk_id, _, _ = query_for_event(
                core, plan, i, context["query_pools"]
            )
            first_event.setdefault(chunk_id, i)
            if len(first_event) == len(plan.chunk_ids):
                break

        for chunk_id, i in first_event.items():
            _, _, query = query_for_event(
                core, plan, i, context["query_pools"]
            )
            truth = base.execute_on_array(chunks_by_id[chunk_id].values, query)
            for codec, rep in reps.items():
                result, _ = base.run_query_once(
                    rep,
                    stored[chunk_id][codec],
                    query,
                    mode,
                    handles.payload[chunk_id][codec],
                    handles.metadata[chunk_id][codec],
                )
                if not base.results_equal(result, truth):
                    raise AssertionError(
                        f"R1 actual warm-up failed: {chunk_id}/{codec}"
                    )

        for i in range(plan.n_events):
            new_interval = i // BUDGET_INTERVAL
            if new_interval != interval_id:
                if spent > budget_bytes:
                    violations += 1
                interval_id = new_interval
                remaining = int(budget_bytes)
                spent = 0

            chunk_id, qclass, query = query_for_event(
                core, plan, i, context["query_pools"]
            )
            state = states[chunk_id]
            current_codec = codecs[state.active_position]
            chunk = chunks_by_id[chunk_id]

            start = time.perf_counter_ns()
            result, stats = base.run_query_once(
                reps[current_codec],
                stored[chunk_id][current_codec],
                query,
                mode,
                handles.payload[chunk_id][current_codec],
                handles.metadata[chunk_id][current_codec],
            )
            latency = time.perf_counter_ns() - start
            truth = base.execute_on_array(chunk.values, query)
            if not base.results_equal(result, truth):
                raise AssertionError(
                    f"R1 actual query failed: {policy}/{chunk_id}/{i}"
                )
            state.query_time_ns += int(latency)
            latencies[chunk_id].append(int(latency))
            state.local_access_count += 1

            target_codec: Optional[str] = None
            reason = ""
            relative = 0.0
            estimated_migration = 0.0

            if policy == "shape_ewma_be_frozen":
                assert state.ewma is not None
                qpos = qclasses.index(qclass)
                state.ewma *= 1.0 - alpha
                state.ewma[qpos] += alpha
                if (
                    state.local_access_count >= MIN_OBSERVATIONS
                    and state.local_access_count % DECISION_INTERVAL == 0
                ):
                    profile = context["profiles"][chunk_id]
                    expected = np.asarray(
                        [
                            sum(
                                float(profile[codec][qclasses[j]])
                                * float(state.ewma[j])
                                for j in range(len(qclasses))
                            )
                            for codec in codecs
                        ]
                    )
                    current = state.active_position
                    target = int(np.argmin(expected))
                    gain = float(expected[current] - expected[target])
                    relative = (
                        gain / float(expected[current])
                        if expected[current] > 0
                        else 0.0
                    )
                    estimated_migration = (
                        float(
                            context["migration_costs"][chunk_id][
                                (codecs[current], codecs[target])
                            ]
                        )
                        if target != current
                        else 0.0
                    )
                    if target == current:
                        reason = "current_layout_is_best"
                    elif relative < deadband:
                        reason = "deadband"
                    elif gain * horizon <= estimated_migration:
                        reason = "break_even_not_met"
                    else:
                        target_codec = codecs[target]
                        reason = "proposal"

            elif policy == "repressdb_frozen":
                assert state.repress_state is not None
                rs = state.repress_state
                rs.recent_classes.append(qclass)
                rs.local_access_count += 1
                if (
                    rs.local_access_count >= core.MIN_OBSERVATIONS
                    and rs.local_access_count % core.DECISION_INTERVAL == 0
                ):
                    candidate, detail = core.decide_repressdb_model(
                        e1,
                        e1v2,
                        context["profiles"][chunk_id],
                        rs,
                        rs.local_access_count - 1,
                        context["encode"][chunk_id],
                        context["decode"][chunk_id],
                    )
                    target_codec = candidate
                    reason = str(detail["reason"])
                    relative = float(detail["relative_improvement"])
                    estimated_migration = float(detail["rewrite_estimate_ns"])

            if target_codec is not None and target_codec != current_codec:
                rewrite = int(
                    context["migration_bytes"][chunk_id][
                        (current_codec, target_codec)
                    ]
                )
                if rewrite <= remaining:
                    migration_ns, actual_rewrite = core.execute_actual_migration(
                        e1,
                        base,
                        reps,
                        stored[chunk_id],
                        chunk,
                        current_codec,
                        target_codec,
                        migration_root / chunk_id,
                        (
                            f"{plan.trace_name}__{policy}__event{i}__"
                            f"m{state.migration_count}"
                        ),
                    )
                    if int(actual_rewrite) != rewrite:
                        raise AssertionError(
                            "Actual rewrite bytes differ from calibration table"
                        )
                    state.active_position = codec_pos[target_codec]
                    state.migration_time_ns += int(migration_ns)
                    state.migration_count += 1
                    state.rewrite_bytes += rewrite
                    remaining -= rewrite
                    spent += rewrite
                    if policy == "repressdb_frozen" and state.repress_state is not None:
                        state.repress_state.active_codec = target_codec
                        state.repress_state.last_switch_local_index = (
                            state.repress_state.local_access_count - 1
                        )
                    events.append(
                        {
                            "policy": policy,
                            "trace": plan.trace_name,
                            "chunk_size": int(plan.chunk_size),
                            "global_event_index": i,
                            "chunk_id": chunk_id,
                            "event": "switch",
                            "old_codec": current_codec,
                            "target_codec": target_codec,
                            "reason": reason,
                            "relative_improvement": relative,
                            "estimated_migration_ns": estimated_migration,
                            "actual_migration_ns": int(migration_ns),
                            "rewrite_bytes": rewrite,
                            "budget_remaining_after": remaining,
                        }
                    )
                else:
                    denials += 1
                    events.append(
                        {
                            "policy": policy,
                            "trace": plan.trace_name,
                            "chunk_size": int(plan.chunk_size),
                            "global_event_index": i,
                            "chunk_id": chunk_id,
                            "event": "budget_denied",
                            "old_codec": current_codec,
                            "target_codec": target_codec,
                            "reason": "write_budget",
                            "rewrite_bytes": rewrite,
                            "budget_remaining_after": remaining,
                        }
                    )

            if i % 5000 == 0 or i == plan.n_events - 1:
                events.append(
                    {
                        "policy": policy,
                        "trace": plan.trace_name,
                        "chunk_size": int(plan.chunk_size),
                        "global_event_index": i,
                        "chunk_id": chunk_id,
                        "event": "query_checkpoint",
                        "old_codec": current_codec,
                        "target_codec": codecs[state.active_position],
                        "reason": "query_checkpoint",
                        "query_class": qclass,
                        "query_latency_ns": int(latency),
                        "payload_bytes": int(stats.payload_bytes),
                        "metadata_bytes": int(stats.metadata_bytes),
                    }
                )

    if spent > budget_bytes:
        violations += 1

    chunk_rows: List[Dict[str, Any]] = []
    all_latency: List[int] = []
    for chunk_id in plan.chunk_ids:
        state = states[chunk_id]
        local = np.asarray(latencies[chunk_id], dtype=np.int64)
        all_latency.extend(latencies[chunk_id])
        meta = context["metadata"][chunk_id]
        chunk_rows.append(
            {
                "policy": policy,
                "trace": plan.trace_name,
                "chunk_size": int(plan.chunk_size),
                "chunk_id": chunk_id,
                "building_id": meta["building_id"],
                "n_queries": int(local.size),
                "query_time_ns": int(state.query_time_ns),
                "migration_time_ns": int(state.migration_time_ns),
                "total_time_ns": int(
                    state.query_time_ns + state.migration_time_ns
                ),
                "median_query_latency_ns": float(np.median(local)),
                "p95_query_latency_ns": float(np.quantile(local, 0.95)),
                "p99_query_latency_ns": float(np.quantile(local, 0.99)),
                "migration_count": int(state.migration_count),
                "rewrite_bytes": int(state.rewrite_bytes),
                "final_codec": codecs[state.active_position],
            }
        )

    a = np.asarray(all_latency, dtype=np.int64)
    tq = int(sum(s.query_time_ns for s in states.values()))
    tm = int(sum(s.migration_time_ns for s in states.values()))
    summary = {
        "policy": policy,
        "trace": plan.trace_name,
        "chunk_size": int(plan.chunk_size),
        "n_buildings": len(plan.chunk_ids),
        "n_queries": int(plan.n_events),
        "query_time_ns": tq,
        "migration_time_ns": tm,
        "total_time_ns": tq + tm,
        "throughput_qps": float(
            plan.n_events / max((tq + tm) / 1e9, 1e-12)
        ),
        "median_query_latency_ns": float(np.median(a)),
        "p95_query_latency_ns": float(np.quantile(a, 0.95)),
        "p99_query_latency_ns": float(np.quantile(a, 0.99)),
        "migration_count": int(sum(s.migration_count for s in states.values())),
        "rewrite_bytes": int(sum(s.rewrite_bytes for s in states.values())),
        "budget_denied_proposals": int(denials),
        "budget_violation_count": int(violations),
    }
    return summary, pd.DataFrame(chunk_rows), pd.DataFrame(events)


def run_actual_audit(
    context: Mapping[str, Any],
    plans: Mapping[Tuple[str, int], Any],
    shape_config: Mapping[str, Any],
    core: Any,
    e1: Any,
    e1v2: Any,
    base: Any,
    output: Path,
    freeze_hash: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reps = base.representation_registry(3)
    root = output / "_actual_audit_checkpoints"
    summaries: List[pd.DataFrame] = []
    chunks_out: List[pd.DataFrame] = []
    events_out: List[pd.DataFrame] = []

    for size in CHUNK_SIZES:
        full = plans[(ACTUAL_TRACE, size)]
        plan = filter_plan(core, full, full.chunk_ids[:ACTUAL_BUILDINGS])
        codecs = tuple(sorted(reps))
        _, migration_bytes = context["s1"].migration_matrix(
            context["migration_costs"],
            context["migration_bytes"],
            plan.chunk_ids,
            codecs,
        )
        budget_bytes = int(migration_bytes.max())
        stored = {
            chunk_id: context["stored_by_chunk"][chunk_id]
            for chunk_id in plan.chunk_ids
        }

        for policy in ACTUAL_POLICIES:
            checkpoint = root / f"size_{size}__policy_{policy}"
            summary_path = checkpoint / "summary.csv"
            chunk_path = checkpoint / "chunks.csv"
            event_path = checkpoint / "events.csv"
            complete_path = checkpoint / "complete.json"

            if (
                summary_path.exists()
                and chunk_path.exists()
                and event_path.exists()
                and complete_path.exists()
            ):
                marker = json.loads(complete_path.read_text(encoding="utf-8"))
                if marker.get("experiment_freeze_sha256") == freeze_hash:
                    summaries.append(pd.read_csv(summary_path))
                    chunks_out.append(pd.read_csv(chunk_path))
                    events_out.append(pd.read_csv(event_path))
                    print(f"[R1 actual resume] {size}/{policy}")
                    continue

            shutil.rmtree(checkpoint, ignore_errors=True)
            checkpoint.mkdir(parents=True, exist_ok=True)
            summary, chunk_frame, event_frame = actual_policy(
                policy,
                context,
                plan,
                shape_config,
                core,
                e1,
                e1v2,
                base,
                reps,
                stored,
                checkpoint / "_migration_trials",
                budget_bytes,
            )
            summary_frame = pd.DataFrame([summary])
            write_csv(summary_path, summary_frame)
            write_csv(chunk_path, chunk_frame)
            write_csv(event_path, event_frame)
            write_json(
                complete_path,
                {
                    "experiment_freeze_sha256": freeze_hash,
                    "trace": ACTUAL_TRACE,
                    "chunk_size": size,
                    "policy": policy,
                    "n_buildings": ACTUAL_BUILDINGS,
                    "completed_at_ns": time.time_ns(),
                },
            )
            summaries.append(summary_frame)
            chunks_out.append(chunk_frame)
            events_out.append(event_frame)
            print(f"[R1 actual] {size}/{policy} complete")

    return (
        pd.concat(summaries, ignore_index=True),
        pd.concat(chunks_out, ignore_index=True),
        pd.concat(events_out, ignore_index=True),
    )


def model_audit_subset(
    context: Mapping[str, Any],
    plans: Mapping[Tuple[str, int], Any],
    shape_config: Mapping[str, Any],
    core: Any,
    e1: Any,
    e1v2: Any,
    base: Any,
) -> pd.DataFrame:
    codecs = tuple(sorted(base.representation_registry(3)))
    qclasses = tuple(core.QUERY_CLASSES)
    rows: List[Dict[str, Any]] = []

    for size in CHUNK_SIZES:
        full = plans[(ACTUAL_TRACE, size)]
        plan = filter_plan(core, full, full.chunk_ids[:ACTUAL_BUILDINGS])
        profile = context["s1"].profile_array(
            context["profiles"],
            plan.chunk_ids,
            codecs,
            qclasses,
        )
        initial = context["s1"].scan_best_positions(
            e1,
            base,
            context["profiles"],
            plan.chunk_ids,
            codecs,
        )
        migration_cost, migration_bytes = context["s1"].migration_matrix(
            context["migration_costs"],
            context["migration_bytes"],
            plan.chunk_ids,
            codecs,
        )
        budget_bytes = int(migration_bytes.max())
        results = {
            "initial_best": static_result(
                plan, profile, initial, "initial_best"
            ),
            "shape_ewma_be_frozen": online_result(
                "shape_ewma_be_frozen",
                context,
                plan,
                profile,
                initial,
                migration_cost,
                migration_bytes,
                codecs,
                qclasses,
                shape_config,
                core,
                e1,
                e1v2,
                base,
                budget_bytes,
            ),
            "repressdb_frozen": online_result(
                "repressdb_frozen",
                context,
                plan,
                profile,
                initial,
                migration_cost,
                migration_bytes,
                codecs,
                qclasses,
                shape_config,
                core,
                e1,
                e1v2,
                base,
                budget_bytes,
            ),
        }
        for policy, result in results.items():
            rows.append(
                {
                    "policy": policy,
                    "trace": ACTUAL_TRACE,
                    "chunk_size": size,
                    "model_total_time_ns": float(result["total_time_ns"]),
                    "model_migration_count": int(result["migration_count"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    project = Path(__file__).resolve().parent
    output = project / OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)

    paths = {
        "s1_script": project / S1_SCRIPT,
        "v070_script": project / V070_SCRIPT,
        "v073_script": project / V073_SCRIPT,
        "v074_script": project / V074_SCRIPT,
        "a2_script": project / A2_SCRIPT,
        "e5_script": project / E5_SCRIPT,
        "s1_dir": project / S1_DIR,
        "b2_dir": project / B2_DIR,
        "v070_dir": project / V070_DIR,
        "v074_dir": project / V074_DIR,
    }
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(p)

    print("=" * 116)
    print("RePressDB R1 v1.4.1 — continuous nonstationary replay")
    print(f"Target journal: {TARGET_JOURNAL}")
    print("=" * 116)

    s1 = load_module(paths["s1_script"], "repressdb_s1_for_r1_v140")
    v073 = load_module(paths["v073_script"], "repressdb_v073_for_r1_v140")
    v074 = load_module(paths["v074_script"], "repressdb_v074_for_r1_v140")
    a2 = load_module(paths["a2_script"], "repressdb_a2_for_r1_v140")
    e5 = load_module(paths["e5_script"], "repressdb_e5_for_r1_v140")

    patched_core_dir = output / "_patched_core"
    patched_core_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    core = v073.build_corrected_core(
        paths["v070_script"],
        patched_core_dir,
    )
    core.__file__ = str(paths["v070_script"].resolve())
    e1v2, e1, base = core.load_repressdb_modules()

    s1_assessment = json.loads(
        (paths["s1_dir"] / "s1_assessment.json").read_text(encoding="utf-8")
    )
    if not bool(
        s1_assessment.get("acceptance_checks", {}).get("all_ten", False)
    ):
        raise RuntimeError("S1 must pass all acceptance checks before R1")

    shape_config_path = (
        paths["s1_dir"] / "shape_ewma_be_frozen_config.json"
    )
    shape_config = json.loads(shape_config_path.read_text(encoding="utf-8"))
    s1_freeze = (
        paths["s1_dir"] / "s1_experiment_freeze_sha256.txt"
    ).read_text(encoding="utf-8").strip()
    if shape_config.get("s1_experiment_freeze_sha256") != s1_freeze:
        raise AssertionError("Shape-EWMA-BE config is not bound to the S1 freeze")

    afhc_config = json.loads(
        (paths["v074_dir"] / "hotcold_frozen_config.json").read_text(
            encoding="utf-8"
        )
    )
    afhc_freeze = (
        paths["v074_dir"] / "experiment_freeze_sha256.txt"
    ).read_text(encoding="utf-8").strip()
    b2_freeze = (
        paths["b2_dir"] / "b2_experiment_freeze_sha256.txt"
    ).read_text(encoding="utf-8").strip()

    selected_path = output / "r1_selected_buildings.csv"
    if selected_path.exists():
        selected = pd.read_csv(selected_path)
    else:
        eligible = pd.read_csv(
            paths["b2_dir"] / "bdg2_full_eligible_series.csv"
        )
        reserved = pd.read_csv(
            paths["v070_dir"] / "bdg2_selected_buildings.csv"
        )
        selected = select_buildings(
            eligible,
            set(reserved["building_id"].astype(str)),
        )
        write_csv(selected_path, selected)

    chunks, chunk_manifest, metadata = load_chunks(selected, e5, base)
    chunk_manifest_path = output / "r1_chunk_manifest.csv"
    write_csv(chunk_manifest_path, chunk_manifest)

    selection_summary = {
        "n_buildings": int(selected["building_id"].nunique()),
        "meter_type_counts": {
            str(k): int(v)
            for k, v in selected["meter_type"].value_counts().items()
        },
        "site_count": int(selected["site_id"].nunique()),
        "a2_overlap_count": int(selected["excluded_a2_building"].sum()),
        "overlap_fallback_used": bool(
            selected["overlap_fallback_used"].iloc[0]
        ),
        "selected_buildings_sha256": sha256_file(selected_path),
        "chunk_manifest_sha256": sha256_file(chunk_manifest_path),
    }
    write_json(output / "r1_selection_summary.json", selection_summary)

    protocol = {
        "version": VERSION,
        "target_journal": TARGET_JOURNAL,
        "study_name": "Continuous Nonstationary Replay",
        "query_stream_provenance": (
            "semi-synthetic query streams over real BDG2 values"
        ),
        "parent_s1_freeze_sha256": s1_freeze,
        "parent_afhc_freeze_sha256": afhc_freeze,
        "parent_b2_freeze_sha256": b2_freeze,
        "selection": selection_summary,
        "chunk_sizes": list(CHUNK_SIZES),
        "trace_names": list(TRACES),
        "global_queries_per_trace": N_EVENTS,
        "expected_local_queries_per_building": LOCAL_QUERIES_PER_BUILDING,
        "policies": list(POLICIES),
        "shape_ewma_config": shape_config,
        "repressdb_frozen": {
            "long_window": core.LONG_WINDOW,
            "short_window": core.SHORT_WINDOW,
            "minimum_observations": core.MIN_OBSERVATIONS,
            "decision_interval": core.DECISION_INTERVAL,
            "confirmation_rounds": core.CONFIRMATION_ROUNDS,
            "cooldown_queries": core.COOLDOWN_QUERIES,
            "prediction_horizon": core.PREDICTION_HORIZON,
            "minimum_relative_improvement": core.MIN_RELATIVE_IMPROVEMENT,
            "rewrite_safety_factor": core.REWRITE_SAFETY_FACTOR,
        },
        "shared_write_budget": {
            "interval_global_queries": BUDGET_INTERVAL,
            "bytes_per_interval_rule": (
                "maximum measured directed rewrite size in each size plan"
            ),
            "online_policies": list(ONLINE_POLICIES),
        },
        "actual_audit": {
            "trace": ACTUAL_TRACE,
            "building_count": ACTUAL_BUILDINGS,
            "chunk_sizes": list(CHUNK_SIZES),
            "policies": list(ACTUAL_POLICIES),
        },
        "metric_definitions": {
            "unamortized": (
                "realized old-minus-new query saving until next migration "
                "or trace end is below measured migration cost"
            ),
            "stale": (
                "target is not the estimated best codec at both of the first "
                "two later decision opportunities for the same chunk"
            ),
        },
    }
    protocol_path = output / "r1_protocol.json"
    write_json(protocol_path, protocol)
    h = hashlib.sha256()
    for value in (s1_freeze, afhc_freeze, b2_freeze):
        h.update(value.encode("ascii"))
    h.update(sha256_file(protocol_path).encode("ascii"))
    protocol_hash = h.hexdigest()
    (output / "r1_protocol_sha256.txt").write_text(
        protocol_hash + "\n", encoding="utf-8"
    )

    reps = base.representation_registry(3)
    query_pools = a2.build_query_pools(base, e1, core, chunks)

    calibration_dir = output / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    calibration_freeze = hashlib.sha256(
        json.dumps(
            {
                "protocol_sha256": protocol_hash,
                "selected_buildings_sha256": selection_summary[
                    "selected_buildings_sha256"
                ],
                "chunk_manifest_sha256": selection_summary[
                    "chunk_manifest_sha256"
                ],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    (
        stored_by_chunk,
        profiles,
        encode,
        decode,
        migration_costs,
        migration_bytes_map,
        calibration_frame,
        codec_frame,
        migration_frame,
    ) = a2.prepare_calibration(
        "BDG2_R1",
        calibration_dir,
        calibration_freeze,
        core,
        e1,
        base,
        reps,
        chunks,
        metadata,
        query_pools,
        v074,
    )

    plans: Dict[Tuple[str, int], Any] = {}
    trace_rows: List[Dict[str, Any]] = []
    for size in CHUNK_SIZES:
        for trace in TRACES:
            plan, manifest = build_or_load_plan(
                output,
                core,
                chunks,
                query_pools,
                trace,
                size,
            )
            plans[(trace, size)] = plan
            trace_rows.append(
                {
                    **manifest,
                    "trace_file": str(
                        output / "frozen_traces" / f"{trace}__size_{size}.npz"
                    ),
                }
            )
    trace_frame = pd.DataFrame(trace_rows)
    trace_manifest_path = output / "r1_trace_manifest.csv"
    write_csv(trace_manifest_path, trace_frame)

    freeze_material = {
        "r1_protocol_sha256": protocol_hash,
        "trace_manifest_sha256": sha256_file(trace_manifest_path),
        "shape_ewma_frozen_config_sha256": sha256_file(shape_config_path),
        "calibration_records_sha256": sha256_file(
            calibration_dir / "calibration_records.csv"
        ),
        "migration_costs_sha256": sha256_file(
            calibration_dir / "measured_migration_costs.csv"
        ),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "parent_s1_freeze_sha256": s1_freeze,
        "parent_afhc_freeze_sha256": afhc_freeze,
        "parent_b2_freeze_sha256": b2_freeze,
    }
    freeze_material_path = output / "r1_experiment_freeze_material.json"
    write_json(freeze_material_path, freeze_material)
    freeze_hash = sha256_file(freeze_material_path)
    (output / "r1_experiment_freeze_sha256.txt").write_text(
        freeze_hash + "\n", encoding="utf-8"
    )

    context = {
        "s1": s1,
        "chunks": chunks,
        "metadata": metadata,
        "query_pools": query_pools,
        "stored_by_chunk": stored_by_chunk,
        "profiles": profiles,
        "encode": encode,
        "decode": decode,
        "migration_costs": migration_costs,
        "migration_bytes": migration_bytes_map,
    }

    (
        global_frame,
        chunk_frame,
        migration_records,
        decision_records,
        cumulative_records,
    ) = run_model_experiment(
        context,
        plans,
        shape_config,
        afhc_config,
        core,
        e1,
        e1v2,
        base,
        output,
    )
    write_csv(output / "r1_model_global_results.csv", global_frame)
    write_csv(output / "r1_model_chunk_results.csv", chunk_frame)
    write_csv(output / "r1_migration_records.csv", migration_records)
    write_csv(output / "r1_decision_records.csv", decision_records)
    write_csv(output / "r1_cumulative_cost_curves.csv", cumulative_records)

    comparisons = clustered_comparisons(chunk_frame)
    write_csv(
        output / "r1_building_clustered_comparisons.csv",
        comparisons,
    )

    actual_summary, actual_chunk, actual_events = run_actual_audit(
        context,
        plans,
        shape_config,
        core,
        e1,
        e1v2,
        base,
        output,
        freeze_hash,
    )
    write_csv(output / "r1_actual_audit_summary.csv", actual_summary)
    write_csv(output / "r1_actual_audit_chunk.csv", actual_chunk)
    write_csv(output / "r1_actual_audit_events.csv", actual_events)

    model_subset = model_audit_subset(
        context,
        plans,
        shape_config,
        core,
        e1,
        e1v2,
        base,
    )
    validation = actual_summary.merge(
        model_subset,
        on=["policy", "trace", "chunk_size"],
        how="inner",
        validate="one_to_one",
    )
    validation["actual_over_model_ratio"] = (
        validation["total_time_ns"] / validation["model_total_time_ns"]
    )
    validation["absolute_log_ratio"] = np.abs(
        np.log(validation["actual_over_model_ratio"])
    )
    write_csv(output / "r1_model_actual_validation.csv", validation)

    expected_global = len(TRACES) * len(CHUNK_SIZES) * len(POLICIES)
    online = global_frame[global_frame["policy"].isin(ONLINE_POLICIES)]
    checks = {
        "R1_1_exactly_32_independent_buildings": bool(
            selected["building_id"].nunique() == N_BUILDINGS
        ),
        "R1_2_paired_8192_and_16384_chunks_complete": bool(
            len(chunk_manifest) == N_BUILDINGS * len(CHUNK_SIZES)
            and set(chunk_manifest["chunk_size"]) == set(CHUNK_SIZES)
        ),
        "R1_3_six_phase_free_traces_frozen": bool(
            len(trace_frame) == len(TRACES) * len(CHUNK_SIZES)
            and (trace_frame["n_events"] == N_EVENTS).all()
        ),
        "R1_4_s1_config_used_without_retuning": bool(
            shape_config["s1_experiment_freeze_sha256"] == s1_freeze
        ),
        "R1_5_all_36_model_policy_runs_complete": bool(
            len(global_frame) == expected_global
        ),
        "R1_6_no_online_write_budget_violations": bool(
            (online["budget_violation_count"] == 0).all()
        ),
        "R1_7_stale_and_unamortized_metrics_reported": bool(
            {
                "stale_migration_count",
                "unamortized_migration_count",
            }.issubset(global_frame.columns)
            and len(migration_records) > 0
        ),
        "R1_8_cumulative_cost_curves_complete": bool(
            cumulative_records[
                ["trace", "chunk_size", "policy"]
            ].drop_duplicates().shape[0]
            == expected_global
        ),
        "R1_9_actual_audit_six_runs_truth_checked": bool(
            len(actual_summary)
            == len(CHUNK_SIZES) * len(ACTUAL_POLICIES)
            and len(actual_events) > 0
            and (actual_summary["budget_violation_count"] == 0).all()
        ),
        "R1_10_building_clustered_and_model_actual_comparisons_reported": bool(
            len(comparisons) > 0
            and len(validation) == len(actual_summary)
            and np.isfinite(validation["actual_over_model_ratio"]).all()
        ),
    }
    checks["all_ten"] = bool(all(checks.values()))

    main_comparison = comparisons[
        (comparisons["numerator_policy"] == "repressdb_frozen")
        & (
            comparisons["denominator_policy"]
            == "shape_ewma_be_frozen"
        )
    ]
    assessment = {
        "version": VERSION,
        "target_journal": TARGET_JOURNAL,
        "status": (
            "R1 complete"
            if checks["all_ten"]
            else "R1 completed with failed acceptance checks"
        ),
        "r1_protocol_sha256": protocol_hash,
        "r1_experiment_freeze_sha256": freeze_hash,
        "parent_s1_freeze_sha256": s1_freeze,
        "parent_afhc_freeze_sha256": afhc_freeze,
        "parent_b2_freeze_sha256": b2_freeze,
        "selection_summary": selection_summary,
        "trace_summary": trace_frame.to_dict("records"),
        "global_policy_results": global_frame.to_dict("records"),
        "repressdb_vs_shape_ewma_building_clustered": (
            main_comparison.to_dict("records")
        ),
        "actual_model_validation": {
            "n_cells": int(len(validation)),
            "median_actual_over_model_ratio": float(
                validation["actual_over_model_ratio"].median()
            ),
            "median_absolute_log_ratio": float(
                validation["absolute_log_ratio"].median()
            ),
        },
        "acceptance_checks": checks,
        "claim_boundary": {
            "allowed": (
                "Frozen policies were evaluated on phase-free "
                "semi-synthetic continuous nonstationary query streams "
                "over real BDG2 values."
            ),
            "not_allowed": (
                "The generated query streams are real production logs."
            ),
            "oracle": "Hindsight lower bound only; not deployable.",
        },
        "interpretation_rule": {
            "repressdb_better_total_cost": (
                "dual-window and safety chain add independent value"
            ),
            "similar_total_cost_but_repressdb_lower_risk": (
                "primary value is migration risk control"
            ),
            "shape_ewma_faster_some_flows": (
                "speed-stability trade-off"
            ),
            "shape_ewma_dominates_cost_and_safety": (
                "RePressDB necessity is weakened; manuscript positioning "
                "must change and the outcome must not be hidden"
            ),
        },
        "next_gate": (
            "Stop adding experiments and proceed to final figures, tables, "
            "and manuscript."
            if checks["all_ten"]
            else "Diagnose R1 failures before final continuous-drift claims."
        ),
    }
    write_json(output / "r1_assessment.json", assessment)
    write_json(project / SUMMARY_JSON, assessment)

    write_json(
        output / "run_config.json",
        {
            "version": VERSION,
            "target_journal": TARGET_JOURNAL,
            "r1_protocol_sha256": protocol_hash,
            "r1_experiment_freeze_sha256": freeze_hash,
            "n_buildings": N_BUILDINGS,
            "chunk_sizes": list(CHUNK_SIZES),
            "trace_names": list(TRACES),
            "global_queries_per_trace": N_EVENTS,
            "policies": list(POLICIES),
            "online_write_budget": {
                "interval_global_queries": BUDGET_INTERVAL,
                "bytes_rule": "maximum directed rewrite",
            },
            "shape_ewma_config": shape_config,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "timer": "time.perf_counter_ns",
            },
        },
    )

    print("=" * 116)
    print("RePressDB R1 v1.4.1 finished")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(f"Results: {output.resolve()}")
    print("Upload:")
    print("  r1_assessment.json")
    print("  r1_selected_buildings.csv")
    print("  r1_trace_manifest.csv")
    print("  r1_model_global_results.csv")
    print("  r1_model_chunk_results.csv")
    print("  r1_migration_records.csv")
    print("  r1_cumulative_cost_curves.csv")
    print("  r1_building_clustered_comparisons.csv")
    print("  r1_actual_audit_summary.csv")
    print("  r1_model_actual_validation.csv")
    print("  r1_protocol.json")
    print("  r1_experiment_freeze_sha256.txt")
    print("=" * 116)
    return 0 if checks["all_ten"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
