#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RePressDB E1-A v0.1.0 — one-click PEMS04 online adaptation pilot.

Target: EDBT 2027 (CCF-B).

Run this file directly in PyCharm. No command-line arguments are required.
Keep RUN_THIS_ONLY_REPRESSDB_E0C_V0_2_0.py in the same directory.

What this experiment adds beyond E0-B/E0-C:
  * time-ordered workload traces;
  * online workload observation using only past queries;
  * cost-aware representation switching with hysteresis and cooldown;
  * actual file-backed query execution in warm_slicedmeta mode;
  * actual decode + encode + fsync migration measurements;
  * fixed, static, periodic-greedy, RePressDB, and oracle baselines.

This is a development-set pilot on PEMS04. Parameters must be frozen before
running the PEMS08 holdout version.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import statistics
import sys
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

VERSION = "0.1.0"
TARGET_VENUE = "EDBT 2027"
BASE_PROTOCOL = "E0-C v0.2.0 / warm_slicedmeta"
SEED = 20260718

# Direct-run configuration. These are intentionally fixed for the PEMS04 pilot.
CHUNK_SIZES = [2048, 8192, 16384]
MAX_SERIES = 12
MAX_CHUNKS_PER_SIZE = 6
WINDOWS_PER_FRACTION = 4
CALIBRATION_QUERIES_PER_CLASS = 4
CALIBRATION_REPEATS = 3

DECISION_INTERVAL = 100
OBSERVATION_WINDOW = 300
MIN_OBSERVATIONS = 200
CONFIRMATION_ROUNDS = 3
COOLDOWN_QUERIES = 300
PREDICTION_HORIZON = 1200
MIN_RELATIVE_IMPROVEMENT = 0.08
REWRITE_SAFETY_FACTOR = 1.25
TIMELINE_SAMPLE_EVERY = 25

OUTPUT_DIR_NAME = "results_e1_pems04_online_v010"
SUMMARY_JSON_NAME = "E1_PEMS04_ONLINE_ASSESSMENT_v010.json"

# Internal smoke-test mode used only during code validation. Normal direct runs
# leave this environment variable unset.
QUICK_TEST = os.environ.get("REPRESSDB_E1_QUICK", "0") == "1"

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

TRACE_SPECS: Dict[str, List[Tuple[str, int]]] = {
    "scan_to_point": [("scan_heavy", 800), ("point_heavy", 2400)],
    "scan_to_aggregate": [("scan_heavy", 800), ("aggregate_heavy", 2400)],
    "cyclic_scan_point": [
        ("scan_heavy", 800),
        ("point_heavy", 800),
        ("scan_heavy", 800),
        ("point_heavy", 800),
    ],
    "brief_point_burst": [
        ("scan_heavy", 1000),
        ("point_heavy", 250),
        ("scan_heavy", 1950),
    ],
}

POLICIES = (
    "fixed_whole_lz4",
    "fixed_block_lz4_256",
    "fixed_block_lz4_1024",
    "initial_best",
    "best_static",
    "periodic_greedy",
    "repressdb",
    "oracle",
)


@dataclass(frozen=True)
class TraceItem:
    index: int
    phase_id: int
    workload: str
    query: Any


@dataclass
class PolicyState:
    policy: str
    active_codec: str
    recent_classes: Deque[str]
    last_switch_query: int = -10**9
    pending_candidate: Optional[str] = None
    pending_rounds: int = 0
    migration_count: int = 0
    migration_time_ns: int = 0
    rewrite_bytes: int = 0
    query_time_ns: int = 0
    query_latencies_ns: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if self.query_latencies_ns is None:
            self.query_latencies_ns = []


class CandidateHandles:
    def __init__(self, stored_map: Mapping[str, Any]):
        self.payload: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        for codec, stored in stored_map.items():
            self.payload[codec] = open(stored.data_path, "rb")
            if stored.metadata_path is not None:
                self.metadata[codec] = open(stored.metadata_path, "rb")
            else:
                self.metadata[codec] = None

    def close(self) -> None:
        for handle in list(self.payload.values()) + list(self.metadata.values()):
            if handle is not None:
                handle.close()
        self.payload.clear()
        self.metadata.clear()

    def __enter__(self) -> "CandidateHandles":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def stable_seed(*parts: Any) -> int:
    token = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def load_base_module() -> Any:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "RUN_THIS_ONLY_REPRESSDB_E0C_V0_2_0.py",
        here / "RUN_THIS_ONLY_REPRESSDB_E0C_PEMS08_DIRECT.py",
    ]
    base_path = next((p for p in candidates if p.exists()), None)
    if base_path is None:
        names = " 或 ".join(p.name for p in candidates)
        raise FileNotFoundError(
            f"找不到 E0-C 基础代码。请把 {names} 与本文件放在同一文件夹。"
        )

    module_name = "repressdb_e0c_base_for_e1"
    spec = importlib.util.spec_from_file_location(module_name, base_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载基础模块：{base_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if getattr(module, "VERSION", None) != "0.2.0":
        raise RuntimeError(
            f"需要 E0-C v0.2.0，当前加载版本为 {getattr(module, 'VERSION', 'unknown')}"
        )
    return module


def workload_cost(
    profile: Mapping[str, Mapping[str, float]],
    codec: str,
    weights: Mapping[str, float],
) -> float:
    return float(sum(float(weights.get(qc, 0.0)) * float(profile[codec][qc]) for qc in QUERY_CLASSES))


def observed_weights(recent: Iterable[str]) -> Dict[str, float]:
    counts = Counter(recent)
    total = sum(counts.values())
    if total <= 0:
        return {qc: 0.0 for qc in QUERY_CLASSES}
    return {qc: counts.get(qc, 0) / total for qc in QUERY_CLASSES}


def choose_best_codec(
    profile: Mapping[str, Mapping[str, float]],
    weights: Mapping[str, float],
) -> Tuple[str, float]:
    scored = [(codec, workload_cost(profile, codec, weights)) for codec in sorted(profile)]
    return min(scored, key=lambda item: (item[1], item[0]))


def build_trace(base: Any, chunk: Any, trace_name: str, query_pool: Mapping[str, List[Any]]) -> List[TraceItem]:
    phases = TRACE_SPECS[trace_name]
    if QUICK_TEST:
        phases = [(name, max(60, length // 20)) for name, length in phases]

    rng = np.random.default_rng(stable_seed(SEED, chunk.chunk_id, trace_name, "trace"))
    trace: List[TraceItem] = []
    global_index = 0
    for phase_id, (workload_name, phase_length) in enumerate(phases):
        weights = base.WORKLOADS[workload_name]
        classes = [qc for qc in QUERY_CLASSES if weights.get(qc, 0.0) > 0.0]
        probabilities = np.asarray([weights[qc] for qc in classes], dtype=float)
        probabilities /= probabilities.sum()
        selected_classes = rng.choice(classes, size=phase_length, p=probabilities)
        for query_class in selected_classes:
            pool = query_pool[str(query_class)]
            query = pool[int(rng.integers(0, len(pool)))]
            trace.append(TraceItem(global_index, phase_id, workload_name, query))
            global_index += 1
    return trace


def phase_boundaries(trace: Sequence[TraceItem]) -> List[Tuple[int, int, int, str]]:
    if not trace:
        return []
    out: List[Tuple[int, int, int, str]] = []
    start = 0
    current_phase = trace[0].phase_id
    current_workload = trace[0].workload
    for i, item in enumerate(trace[1:], start=1):
        if item.phase_id != current_phase:
            out.append((current_phase, start, i, current_workload))
            start = i
            current_phase = item.phase_id
            current_workload = item.workload
    out.append((current_phase, start, len(trace), current_workload))
    return out


def encode_candidates(
    base: Any,
    reps: Mapping[str, Any],
    chunk: Any,
    chunk_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, int], Dict[str, int]]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    stored_map: Dict[str, Any] = {}
    encode_ns: Dict[str, int] = {}
    decode_ns: Dict[str, int] = {}
    for codec, rep in reps.items():
        stored, enc = base.encode_one(rep, chunk.values, chunk_dir / codec, 1)
        decoded = base.measure_full_decode_warm(rep, stored, 1, 3 if not QUICK_TEST else 1)
        decoded_array_stats = base.IOStats()
        with open(stored.data_path, "rb") as handle:
            roundtrip = rep.decode_all(stored, stored.metadata, handle, decoded_array_stats)
        if not np.array_equal(roundtrip.view(np.uint32), chunk.values.view(np.uint32)):
            raise AssertionError(f"Lossless round trip failed: {chunk.chunk_id}, {codec}")
        stored_map[codec] = stored
        encode_ns[codec] = int(enc)
        decode_ns[codec] = int(decoded[0])
    return stored_map, encode_ns, decode_ns


def calibrate_chunk(
    base: Any,
    reps: Mapping[str, Any],
    stored_map: Mapping[str, Any],
    chunk: Any,
    query_pool: Mapping[str, List[Any]],
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    mode = base.CACHE_MODES["warm_slicedmeta"]
    profile: Dict[str, Dict[str, float]] = {codec: {} for codec in reps}
    rows: List[Dict[str, Any]] = []

    with CandidateHandles(stored_map) as handles:
        for codec, rep in reps.items():
            stored = stored_map[codec]
            for query_class in QUERY_CLASSES:
                pool = query_pool[query_class]
                limit = min(len(pool), 1 if QUICK_TEST else CALIBRATION_QUERIES_PER_CLASS)
                selected = pool[:limit]
                timings: List[int] = []
                payload_bytes: List[int] = []
                metadata_bytes: List[int] = []
                for query in selected:
                    # Warm up once before measured repetitions.
                    result, _ = base.run_query_once(
                        rep,
                        stored,
                        query,
                        mode,
                        handles.payload[codec],
                        handles.metadata[codec],
                    )
                    truth = base.execute_on_array(chunk.values, query)
                    if not base.results_equal(result, truth):
                        raise AssertionError(
                            f"Calibration query incorrect: {chunk.chunk_id}, {codec}, {query.query_id}"
                        )
                    for _ in range(1 if QUICK_TEST else CALIBRATION_REPEATS):
                        start = time.perf_counter_ns()
                        result, stats = base.run_query_once(
                            rep,
                            stored,
                            query,
                            mode,
                            handles.payload[codec],
                            handles.metadata[codec],
                        )
                        elapsed = time.perf_counter_ns() - start
                        if not base.results_equal(result, truth):
                            raise AssertionError(
                                f"Calibration query incorrect: {chunk.chunk_id}, {codec}, {query.query_id}"
                            )
                        timings.append(int(elapsed))
                        payload_bytes.append(int(stats.payload_bytes))
                        metadata_bytes.append(int(stats.metadata_bytes))
                median_ns = float(np.median(np.asarray(timings, dtype=np.int64)))
                profile[codec][query_class] = median_ns
                rows.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "chunk_size": chunk.chunk_size,
                        "codec": codec,
                        "query_class": query_class,
                        "n_measurements": len(timings),
                        "median_latency_ns": median_ns,
                        "median_payload_bytes": float(np.median(payload_bytes)),
                        "median_metadata_bytes": float(np.median(metadata_bytes)),
                    }
                )
    return profile, rows


def migration_estimate_ns(
    current_codec: str,
    target_codec: str,
    encode_ns: Mapping[str, int],
    decode_ns: Mapping[str, int],
) -> int:
    if current_codec == target_codec:
        return 0
    return int(decode_ns[current_codec] + encode_ns[target_codec])


def perform_actual_migration(
    base: Any,
    reps: Mapping[str, Any],
    stored_map: Mapping[str, Any],
    chunk: Any,
    current_codec: str,
    target_codec: str,
    migration_dir: Path,
    token: str,
) -> Tuple[int, int]:
    if current_codec == target_codec:
        return 0, 0
    migration_dir.mkdir(parents=True, exist_ok=True)
    current_rep = reps[current_codec]
    current_stored = stored_map[current_codec]
    target_rep = reps[target_codec]

    read_stats = base.IOStats()
    start_ns = time.perf_counter_ns()
    with open(current_stored.data_path, "rb") as handle:
        values = current_rep.decode_all(
            current_stored, current_stored.metadata, handle, read_stats
        )
    trial_stem = migration_dir / f"migration__{token}__to_{target_codec}"
    trial = target_rep.encode_to_files(values, trial_stem)
    elapsed_ns = time.perf_counter_ns() - start_ns

    # Validate outside the measured migration interval.
    verify_stats = base.IOStats()
    with open(trial.data_path, "rb") as handle:
        verify = target_rep.decode_all(trial, trial.metadata, handle, verify_stats)
    if not np.array_equal(verify.view(np.uint32), chunk.values.view(np.uint32)):
        raise AssertionError(
            f"Migration validation failed: {chunk.chunk_id}, {current_codec}->{target_codec}"
        )

    rewrite_bytes = int(trial.stored_size_bytes)
    trial.data_path.unlink(missing_ok=True)
    if trial.metadata_path is not None:
        trial.metadata_path.unlink(missing_ok=True)
    return int(elapsed_ns), rewrite_bytes


def policy_initial_codec(
    base: Any,
    policy: str,
    profile: Mapping[str, Mapping[str, float]],
    trace: Sequence[TraceItem],
) -> str:
    if policy == "fixed_whole_lz4":
        return "whole_lz4"
    if policy == "fixed_block_lz4_256":
        return "block_lz4_256"
    if policy == "fixed_block_lz4_1024":
        return "block_lz4_1024"

    first_workload = trace[0].workload
    if policy in {"initial_best", "periodic_greedy", "repressdb", "oracle"}:
        return choose_best_codec(profile, base.WORKLOADS[first_workload])[0]

    if policy == "best_static":
        counts = Counter(item.query.query_class for item in trace)
        weights = {qc: counts.get(qc, 0) / len(trace) for qc in QUERY_CLASSES}
        return choose_best_codec(profile, weights)[0]

    raise ValueError(f"Unknown policy: {policy}")


def decide_periodic(
    profile: Mapping[str, Mapping[str, float]],
    state: PolicyState,
) -> Tuple[Optional[str], Dict[str, Any]]:
    weights = observed_weights(state.recent_classes)
    target, target_cost = choose_best_codec(profile, weights)
    current_cost = workload_cost(profile, state.active_codec, weights)
    return target, {
        "current_expected_ns": current_cost,
        "target_expected_ns": target_cost,
        "relative_improvement": (
            (current_cost - target_cost) / current_cost if current_cost > 0 else 0.0
        ),
        "predicted_saving_ns": math.nan,
        "rewrite_estimate_ns": math.nan,
        "reason": "periodic_best_of_recent_window",
    }


def decide_repressdb(
    profile: Mapping[str, Mapping[str, float]],
    state: PolicyState,
    query_index: int,
    encode_ns: Mapping[str, int],
    decode_ns: Mapping[str, int],
) -> Tuple[Optional[str], Dict[str, Any]]:
    weights = observed_weights(state.recent_classes)
    target, target_cost = choose_best_codec(profile, weights)
    current_cost = workload_cost(profile, state.active_codec, weights)
    relative = (current_cost - target_cost) / current_cost if current_cost > 0 else 0.0
    rewrite_est = migration_estimate_ns(
        state.active_codec, target, encode_ns, decode_ns
    )
    predicted_saving = max(0.0, current_cost - target_cost) * PREDICTION_HORIZON

    detail = {
        "current_expected_ns": current_cost,
        "target_expected_ns": target_cost,
        "relative_improvement": relative,
        "predicted_saving_ns": predicted_saving,
        "rewrite_estimate_ns": rewrite_est,
        "reason": "",
    }

    if target == state.active_codec:
        state.pending_candidate = None
        state.pending_rounds = 0
        detail["reason"] = "current_layout_is_best"
        return None, detail

    if query_index - state.last_switch_query < COOLDOWN_QUERIES:
        state.pending_candidate = None
        state.pending_rounds = 0
        detail["reason"] = "cooldown"
        return None, detail

    if relative < MIN_RELATIVE_IMPROVEMENT:
        state.pending_candidate = None
        state.pending_rounds = 0
        detail["reason"] = "relative_gain_too_small"
        return None, detail

    if predicted_saving < REWRITE_SAFETY_FACTOR * rewrite_est:
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


def oracle_target_for_phase(
    base: Any,
    profile: Mapping[str, Mapping[str, float]],
    current_codec: str,
    workload: str,
    phase_length: int,
    encode_ns: Mapping[str, int],
    decode_ns: Mapping[str, int],
) -> str:
    weights = base.WORKLOADS[workload]
    candidates: List[Tuple[float, str]] = []
    for codec in sorted(profile):
        query_cost = phase_length * workload_cost(profile, codec, weights)
        rewrite_cost = migration_estimate_ns(current_codec, codec, encode_ns, decode_ns)
        candidates.append((query_cost + rewrite_cost, codec))
    return min(candidates, key=lambda item: (item[0], item[1]))[1]


def run_policy(
    base: Any,
    reps: Mapping[str, Any],
    stored_map: Mapping[str, Any],
    chunk: Any,
    trace_name: str,
    trace: Sequence[TraceItem],
    profile: Mapping[str, Mapping[str, float]],
    encode_ns: Mapping[str, int],
    decode_ns: Mapping[str, int],
    policy: str,
    migration_root: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    mode = base.CACHE_MODES["warm_slicedmeta"]
    initial_codec = policy_initial_codec(base, policy, profile, trace)
    state = PolicyState(
        policy=policy,
        active_codec=initial_codec,
        recent_classes=deque(maxlen=OBSERVATION_WINDOW),
    )
    decision_rows: List[Dict[str, Any]] = []
    timeline_rows: List[Dict[str, Any]] = []
    phase_rows: List[Dict[str, Any]] = []

    boundaries = phase_boundaries(trace)
    boundary_by_start = {start: (phase_id, end, workload) for phase_id, start, end, workload in boundaries}
    current_phase_query_time = 0
    current_phase_migration_time = 0
    current_phase_start = 0
    current_phase_id = trace[0].phase_id
    current_phase_workload = trace[0].workload
    current_phase_migrations = 0

    with CandidateHandles(stored_map) as handles:
        # Warm candidate files before replay. This is the declared warm_slicedmeta model.
        warm_query = trace[0].query
        for codec, rep in reps.items():
            result, _ = base.run_query_once(
                rep,
                stored_map[codec],
                warm_query,
                mode,
                handles.payload[codec],
                handles.metadata[codec],
            )
            truth = base.execute_on_array(chunk.values, warm_query)
            if not base.results_equal(result, truth):
                raise AssertionError(f"Warm-up query failed: {chunk.chunk_id}, {codec}")

        for item in trace:
            # Flush the previous phase summary before processing a new phase.
            if item.phase_id != current_phase_id:
                phase_rows.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "chunk_size": chunk.chunk_size,
                        "trace": trace_name,
                        "policy": policy,
                        "phase_id": current_phase_id,
                        "workload": current_phase_workload,
                        "start_query": current_phase_start,
                        "end_query": item.index,
                        "n_queries": item.index - current_phase_start,
                        "query_time_ns": current_phase_query_time,
                        "migration_time_ns": current_phase_migration_time,
                        "migration_count": current_phase_migrations,
                    }
                )
                current_phase_query_time = 0
                current_phase_migration_time = 0
                current_phase_start = item.index
                current_phase_id = item.phase_id
                current_phase_workload = item.workload
                current_phase_migrations = 0

            # Oracle may act exactly at a known phase boundary.
            if policy == "oracle" and item.index in boundary_by_start:
                phase_id, phase_end, workload = boundary_by_start[item.index]
                target = oracle_target_for_phase(
                    base,
                    profile,
                    state.active_codec,
                    workload,
                    phase_end - item.index,
                    encode_ns,
                    decode_ns,
                )
                if target != state.active_codec:
                    old_codec = state.active_codec
                    migration_ns, rewritten = perform_actual_migration(
                        base,
                        reps,
                        stored_map,
                        chunk,
                        old_codec,
                        target,
                        migration_root,
                        f"{trace_name}__{policy}__q{item.index}__m{state.migration_count}",
                    )
                    state.active_codec = target
                    state.last_switch_query = item.index
                    state.migration_count += 1
                    state.migration_time_ns += migration_ns
                    state.rewrite_bytes += rewritten
                    current_phase_migration_time += migration_ns
                    current_phase_migrations += 1
                    decision_rows.append(
                        {
                            "chunk_id": chunk.chunk_id,
                            "chunk_size": chunk.chunk_size,
                            "trace": trace_name,
                            "policy": policy,
                            "query_index": item.index,
                            "phase_id": item.phase_id,
                            "workload": item.workload,
                            "old_codec": old_codec,
                            "candidate_codec": target,
                            "decision": "switch",
                            "reason": "oracle_phase_boundary",
                            "current_expected_ns": workload_cost(profile, old_codec, base.WORKLOADS[workload]),
                            "target_expected_ns": workload_cost(profile, target, base.WORKLOADS[workload]),
                            "relative_improvement": math.nan,
                            "predicted_saving_ns": math.nan,
                            "rewrite_estimate_ns": migration_estimate_ns(old_codec, target, encode_ns, decode_ns),
                            "actual_migration_ns": migration_ns,
                            "rewrite_bytes": rewritten,
                        }
                    )

            active_rep = reps[state.active_codec]
            active_stored = stored_map[state.active_codec]
            start_ns = time.perf_counter_ns()
            result, stats = base.run_query_once(
                active_rep,
                active_stored,
                item.query,
                mode,
                handles.payload[state.active_codec],
                handles.metadata[state.active_codec],
            )
            latency_ns = time.perf_counter_ns() - start_ns
            truth = base.execute_on_array(chunk.values, item.query)
            if not base.results_equal(result, truth):
                raise AssertionError(
                    f"Replay query incorrect: {chunk.chunk_id}, {trace_name}, {policy}, q={item.index}"
                )

            state.query_time_ns += int(latency_ns)
            state.query_latencies_ns.append(int(latency_ns))
            current_phase_query_time += int(latency_ns)
            state.recent_classes.append(item.query.query_class)

            switched_this_query = False
            if policy in {"periodic_greedy", "repressdb"}:
                should_decide = (
                    len(state.recent_classes) >= (MIN_OBSERVATIONS if not QUICK_TEST else 30)
                    and (item.index + 1) % (DECISION_INTERVAL if not QUICK_TEST else 20) == 0
                )
                if should_decide:
                    if policy == "periodic_greedy":
                        target, detail = decide_periodic(profile, state)
                    else:
                        target, detail = decide_repressdb(
                            profile,
                            state,
                            item.index,
                            encode_ns,
                            decode_ns,
                        )
                    old_codec = state.active_codec
                    actual_migration_ns = 0
                    rewritten = 0
                    decision = "stay"
                    if target is not None and target != state.active_codec:
                        actual_migration_ns, rewritten = perform_actual_migration(
                            base,
                            reps,
                            stored_map,
                            chunk,
                            state.active_codec,
                            target,
                            migration_root,
                            f"{trace_name}__{policy}__q{item.index}__m{state.migration_count}",
                        )
                        state.active_codec = target
                        state.last_switch_query = item.index
                        state.migration_count += 1
                        state.migration_time_ns += actual_migration_ns
                        state.rewrite_bytes += rewritten
                        current_phase_migration_time += actual_migration_ns
                        current_phase_migrations += 1
                        switched_this_query = True
                        decision = "switch"
                    decision_rows.append(
                        {
                            "chunk_id": chunk.chunk_id,
                            "chunk_size": chunk.chunk_size,
                            "trace": trace_name,
                            "policy": policy,
                            "query_index": item.index,
                            "phase_id": item.phase_id,
                            "workload": item.workload,
                            "old_codec": old_codec,
                            "candidate_codec": target if target is not None else old_codec,
                            "decision": decision,
                            "reason": detail["reason"],
                            "current_expected_ns": detail["current_expected_ns"],
                            "target_expected_ns": detail["target_expected_ns"],
                            "relative_improvement": detail["relative_improvement"],
                            "predicted_saving_ns": detail["predicted_saving_ns"],
                            "rewrite_estimate_ns": detail["rewrite_estimate_ns"],
                            "actual_migration_ns": actual_migration_ns,
                            "rewrite_bytes": rewritten,
                        }
                    )

            if (
                item.index % (TIMELINE_SAMPLE_EVERY if not QUICK_TEST else 10) == 0
                or switched_this_query
                or item.index == len(trace) - 1
            ):
                timeline_rows.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "chunk_size": chunk.chunk_size,
                        "trace": trace_name,
                        "policy": policy,
                        "query_index": item.index,
                        "phase_id": item.phase_id,
                        "workload": item.workload,
                        "query_class": item.query.query_class,
                        "active_codec": state.active_codec,
                        "latency_ns": int(latency_ns),
                        "payload_bytes": int(stats.payload_bytes),
                        "metadata_bytes": int(stats.metadata_bytes),
                        "cumulative_query_time_ns": state.query_time_ns,
                        "cumulative_migration_time_ns": state.migration_time_ns,
                        "cumulative_total_time_ns": state.query_time_ns + state.migration_time_ns,
                        "cumulative_migrations": state.migration_count,
                    }
                )

    # Final phase summary.
    phase_rows.append(
        {
            "chunk_id": chunk.chunk_id,
            "chunk_size": chunk.chunk_size,
            "trace": trace_name,
            "policy": policy,
            "phase_id": current_phase_id,
            "workload": current_phase_workload,
            "start_query": current_phase_start,
            "end_query": len(trace),
            "n_queries": len(trace) - current_phase_start,
            "query_time_ns": current_phase_query_time,
            "migration_time_ns": current_phase_migration_time,
            "migration_count": current_phase_migrations,
        }
    )

    latencies = np.asarray(state.query_latencies_ns, dtype=np.int64)
    summary = {
        "chunk_id": chunk.chunk_id,
        "series_index": chunk.series_index,
        "chunk_start": chunk.chunk_start,
        "chunk_size": chunk.chunk_size,
        "trace": trace_name,
        "policy": policy,
        "n_queries": len(trace),
        "initial_codec": initial_codec,
        "final_codec": state.active_codec,
        "query_time_ns": state.query_time_ns,
        "migration_time_ns": state.migration_time_ns,
        "total_time_ns": state.query_time_ns + state.migration_time_ns,
        "throughput_qps": len(trace) / ((state.query_time_ns + state.migration_time_ns) / 1e9),
        "median_query_latency_ns": float(np.median(latencies)),
        "p95_query_latency_ns": float(np.quantile(latencies, 0.95)),
        "p99_query_latency_ns": float(np.quantile(latencies, 0.99)),
        "migration_count": state.migration_count,
        "rewrite_bytes": state.rewrite_bytes,
    }
    return summary, decision_rows, timeline_rows, phase_rows


def aggregate_results(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, group in run_df.groupby(["trace", "chunk_size", "policy"], sort=True):
        trace, chunk_size, policy = keys
        total = group["total_time_ns"].to_numpy(dtype=float)
        rows.append(
            {
                "trace": trace,
                "chunk_size": int(chunk_size),
                "policy": policy,
                "n_chunks": len(group),
                "median_total_time_ns": float(np.median(total)),
                "mean_total_time_ns": float(np.mean(total)),
                "median_query_time_ns": float(group["query_time_ns"].median()),
                "median_migration_time_ns": float(group["migration_time_ns"].median()),
                "median_query_latency_ns": float(group["median_query_latency_ns"].median()),
                "median_p95_query_latency_ns": float(group["p95_query_latency_ns"].median()),
                "median_migration_count": float(group["migration_count"].median()),
                "median_rewrite_bytes": float(group["rewrite_bytes"].median()),
                "median_throughput_qps": float(group["throughput_qps"].median()),
            }
        )
    return pd.DataFrame(rows)


def make_assessment(
    aggregate_df: pd.DataFrame,
    run_df: pd.DataFrame,
    dataset_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    comparisons: Dict[str, Any] = {}
    for (trace, chunk_size), group in aggregate_df.groupby(["trace", "chunk_size"]):
        costs = group.set_index("policy")["median_total_time_ns"].to_dict()
        migrations = group.set_index("policy")["median_migration_count"].to_dict()
        repress = float(costs["repressdb"])
        comparisons[f"{trace}|{int(chunk_size)}"] = {
            "repressdb_vs_fixed_whole_ratio": repress / float(costs["fixed_whole_lz4"]),
            "repressdb_vs_initial_best_ratio": repress / float(costs["initial_best"]),
            "repressdb_vs_periodic_greedy_ratio": repress / float(costs["periodic_greedy"]),
            "repressdb_vs_best_static_ratio": repress / float(costs["best_static"]),
            "repressdb_vs_oracle_ratio": repress / float(costs["oracle"]),
            "repressdb_median_migrations": float(migrations["repressdb"]),
            "periodic_greedy_median_migrations": float(migrations["periodic_greedy"]),
        }

    stable_keys = [
        key for key in comparisons
        if key.startswith("scan_to_point|") or key.startswith("scan_to_aggregate|")
    ]
    large_stable_keys = [key for key in stable_keys if key.endswith("|8192") or key.endswith("|16384")]
    stable_improvement = all(
        comparisons[key]["repressdb_vs_initial_best_ratio"] < 1.0
        for key in large_stable_keys
    )
    oracle_gap_ok = all(
        comparisons[key]["repressdb_vs_oracle_ratio"] <= 1.30
        for key in large_stable_keys
    )
    anti_thrashing_keys = [
        key for key in comparisons
        if key.startswith("cyclic_scan_point|") or key.startswith("brief_point_burst|")
    ]
    fewer_migrations = all(
        comparisons[key]["repressdb_median_migrations"]
        <= comparisons[key]["periodic_greedy_median_migrations"]
        for key in anti_thrashing_keys
    )

    return {
        "version": VERSION,
        "target_venue": TARGET_VENUE,
        "dataset_label": dataset_meta.get("dataset_label"),
        "status": "PEMS04 development pilot",
        "primary_execution_mode": "warm_slicedmeta",
        "online_information_boundary": (
            "RePressDB and periodic-greedy use only the most recent observed query classes. "
            "Oracle alone knows phase boundaries and future phase lengths."
        ),
        "pilot_gates": {
            "G1_stable_drift_improves_over_initial_best_on_8192_and_16384": bool(stable_improvement),
            "G2_within_30_percent_of_oracle_on_stable_large_chunks": bool(oracle_gap_ok),
            "G3_no_more_migrations_than_periodic_on_cyclic_and_burst_traces": bool(fewer_migrations),
            "all_three": bool(stable_improvement and oracle_gap_ok and fewer_migrations),
        },
        "comparisons": comparisons,
        "important_limitations": [
            "This is a PEMS04 development-set pilot with one deterministic trace seed.",
            "Queries execute in the E0-C warm_slicedmeta mode; this is not a cold-SSD claim.",
            "The cost profile is calibrated per chunk before replay; PEMS08 must use frozen controller parameters.",
            "Migration time includes full decode plus target encode and fsync, but candidate files are prebuilt for query replay after the measured rewrite.",
        ],
        "next_gate": (
            "Inspect PEMS04. If the controller behaves sensibly, freeze all constants and run the unchanged PEMS08 holdout direct version."
        ),
    }


def write_json(path: Path, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=True)


def main() -> int:
    base = load_base_module()
    print("=" * 88)
    print(f"RePressDB E1-A v{VERSION} — PEMS04 online adaptation pilot")
    print(f"Target: {TARGET_VENUE}; primary mode: warm_slicedmeta")
    print("Direct run: no command-line arguments are needed.")
    print("=" * 88)

    output_dir = Path(OUTPUT_DIR_NAME)
    if QUICK_TEST:
        output_dir = Path("results_e1_quick_smoke")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    storage_dir = output_dir / "_storage"
    migration_dir = output_dir / "_migration_trials"
    storage_dir.mkdir(parents=True, exist_ok=True)
    migration_dir.mkdir(parents=True, exist_ok=True)

    if QUICK_TEST:
        matrix, dataset_meta = base.synthetic_dataset(SEED, 3)
        dataset_label = "SYNTHETIC_E1_SMOKE"
        local_chunk_sizes = [2048]
        local_max_chunks = 2
    else:
        pems04_dir, _ = base.default_data_dirs()
        data_file = base.auto_detect_data_file(pems04_dir, "PEMS04_E1_ONLINE_PILOT")
        matrix, dataset_meta = base.load_dataset(
            data_file,
            feature_index=0,
            max_series=MAX_SERIES,
            seed=SEED,
            excluded_series=None,
        )
        dataset_label = "PEMS04_E1_ONLINE_PILOT"
        local_chunk_sizes = CHUNK_SIZES
        local_max_chunks = MAX_CHUNKS_PER_SIZE

    chunks = base.extract_chunks(
        matrix,
        local_chunk_sizes,
        local_max_chunks,
        SEED,
    )
    reps = base.representation_registry(3)
    print(f"[{dataset_label}] chunks={len(chunks)}, traces={len(TRACE_SPECS)}, policies={len(POLICIES)}")

    calibration_rows: List[Dict[str, Any]] = []
    codec_rows: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []
    timeline_rows: List[Dict[str, Any]] = []
    phase_rows: List[Dict[str, Any]] = []

    for chunk_index, chunk in enumerate(chunks, start=1):
        chunk_dir = storage_dir / chunk.chunk_id
        stored_map, encode_ns, decode_ns = encode_candidates(
            base, reps, chunk, chunk_dir
        )
        for codec, stored in stored_map.items():
            codec_rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "series_index": chunk.series_index,
                    "chunk_start": chunk.chunk_start,
                    "chunk_size": chunk.chunk_size,
                    "codec": codec,
                    "raw_size_bytes": int(chunk.values.nbytes),
                    "payload_size_bytes": int(stored.payload_size_bytes),
                    "metadata_size_bytes": int(stored.metadata_size_bytes),
                    "stored_size_bytes": int(stored.stored_size_bytes),
                    "compression_ratio": float(stored.stored_size_bytes / chunk.values.nbytes),
                    "encode_ns": int(encode_ns[codec]),
                    "full_decode_ns": int(decode_ns[codec]),
                }
            )

        queries = base.build_queries(
            chunk.values,
            chunk.chunk_id,
            SEED,
            1 if QUICK_TEST else WINDOWS_PER_FRACTION,
        )
        query_pool = {
            query_class: [q for q in queries if q.query_class == query_class]
            for query_class in QUERY_CLASSES
        }
        if any(not values for values in query_pool.values()):
            raise AssertionError(f"Incomplete query pool: {chunk.chunk_id}")

        profile, local_calibration = calibrate_chunk(
            base, reps, stored_map, chunk, query_pool
        )
        calibration_rows.extend(local_calibration)

        for trace_name in TRACE_SPECS:
            trace = build_trace(base, chunk, trace_name, query_pool)
            order = list(POLICIES)
            random.Random(stable_seed(SEED, chunk.chunk_id, trace_name, "policy_order")).shuffle(order)
            for policy in order:
                summary, decisions, timeline, phases = run_policy(
                    base,
                    reps,
                    stored_map,
                    chunk,
                    trace_name,
                    trace,
                    profile,
                    encode_ns,
                    decode_ns,
                    policy,
                    migration_dir / chunk.chunk_id,
                )
                run_rows.append(summary)
                decision_rows.extend(decisions)
                timeline_rows.extend(timeline)
                phase_rows.extend(phases)

        if chunk_index == 1 or chunk_index == len(chunks) or chunk_index % max(1, len(chunks) // 10) == 0:
            print(f"[{dataset_label}] {chunk_index}/{len(chunks)} chunks complete")

    run_df = pd.DataFrame(run_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    codec_df = pd.DataFrame(codec_rows)
    decisions_df = pd.DataFrame(decision_rows)
    timeline_df = pd.DataFrame(timeline_rows)
    phases_df = pd.DataFrame(phase_rows)
    aggregate_df = aggregate_results(run_df)

    # Normalize each policy against RePressDB and oracle within the same trace/chunk-size stratum.
    for reference_policy, column_name in [
        ("repressdb", "ratio_to_repressdb"),
        ("oracle", "ratio_to_oracle"),
    ]:
        reference = aggregate_df[aggregate_df["policy"] == reference_policy][
            ["trace", "chunk_size", "median_total_time_ns"]
        ].rename(columns={"median_total_time_ns": "reference_time_ns"})
        aggregate_df = aggregate_df.merge(reference, on=["trace", "chunk_size"], how="left")
        aggregate_df[column_name] = aggregate_df["median_total_time_ns"] / aggregate_df["reference_time_ns"]
        aggregate_df = aggregate_df.drop(columns=["reference_time_ns"])

    dataset_meta = dict(dataset_meta)
    dataset_meta.update(
        {
            "dataset_label": dataset_label,
            "num_chunks": len(chunks),
            "chunk_sizes": local_chunk_sizes,
            "max_chunks_per_size": local_max_chunks,
            "trace_names": list(TRACE_SPECS),
            "policies": list(POLICIES),
            "primary_execution_mode": "warm_slicedmeta",
            "target_venue": TARGET_VENUE,
        }
    )
    assessment = make_assessment(aggregate_df, run_df, dataset_meta)

    calibration_df.to_csv(output_dir / "calibration_records.csv", index=False)
    codec_df.to_csv(output_dir / "codec_records.csv", index=False)
    run_df.to_csv(output_dir / "policy_run_records.csv", index=False)
    aggregate_df.to_csv(output_dir / "policy_aggregate_summary.csv", index=False)
    decisions_df.to_csv(output_dir / "decision_records.csv", index=False)
    timeline_df.to_csv(output_dir / "timeline_sample.csv", index=False)
    phases_df.to_csv(output_dir / "phase_records.csv", index=False)
    write_json(output_dir / "dataset_summary.json", dataset_meta)
    write_json(output_dir / "e1_assessment.json", assessment)

    config = {
        "version": VERSION,
        "target_venue": TARGET_VENUE,
        "base_protocol": BASE_PROTOCOL,
        "seed": SEED,
        "chunk_sizes": local_chunk_sizes,
        "max_series": 3 if QUICK_TEST else MAX_SERIES,
        "max_chunks_per_size": local_max_chunks,
        "windows_per_fraction": 1 if QUICK_TEST else WINDOWS_PER_FRACTION,
        "calibration_queries_per_class": 1 if QUICK_TEST else CALIBRATION_QUERIES_PER_CLASS,
        "calibration_repeats": 1 if QUICK_TEST else CALIBRATION_REPEATS,
        "decision_interval": 20 if QUICK_TEST else DECISION_INTERVAL,
        "observation_window": OBSERVATION_WINDOW,
        "min_observations": 30 if QUICK_TEST else MIN_OBSERVATIONS,
        "confirmation_rounds": CONFIRMATION_ROUNDS,
        "cooldown_queries": COOLDOWN_QUERIES,
        "prediction_horizon": PREDICTION_HORIZON,
        "min_relative_improvement": MIN_RELATIVE_IMPROVEMENT,
        "rewrite_safety_factor": REWRITE_SAFETY_FACTOR,
        "timeline_sample_every": 10 if QUICK_TEST else TIMELINE_SAMPLE_EVERY,
        "trace_specs": TRACE_SPECS,
        "policies": POLICIES,
        "representations": list(reps),
        "workloads": base.WORKLOADS,
        "measurement_scope": (
            "Time-ordered file-backed replay in warm_slicedmeta mode. Query latency is measured "
            "with perf_counter_ns. Each migration measures full decode plus target encode and fsync."
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "timer": "time.perf_counter_ns",
        },
    }
    write_json(output_dir / "run_config.json", config)
    write_json(Path(SUMMARY_JSON_NAME if not QUICK_TEST else "E1_QUICK_ASSESSMENT.json"), assessment)

    # Candidate and migration files are not part of the retained results.
    shutil.rmtree(storage_dir, ignore_errors=True)
    shutil.rmtree(migration_dir, ignore_errors=True)

    print("-" * 88)
    print(f"Pilot gates: {assessment['pilot_gates']}")
    print(f"Results: {output_dir.resolve()}")
    print("Done. Upload e1_assessment.json, policy_aggregate_summary.csv, policy_run_records.csv,")
    print("decision_records.csv, phase_records.csv, calibration_records.csv, and run_config.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
