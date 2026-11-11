#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RePressDB E4 v0.5.0
Global rewrite-budget scheduling under heterogeneous multi-chunk drift.

Target venue: EDBT 2027 (CCF-B).

DIRECT-RUN VERSION
------------------
Open this file in PyCharm and click the green Run button.
No command-line arguments and no source-code edits are required.

Keep these two dependency files in the same directory:
  1) RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT.py
  2) RUN_THIS_ONLY_REPRESSDB_E0C_V0_2_0.py

Research question
-----------------
Earlier experiments adapt one chunk independently. E4 asks what happens when
many chunks drift at the same time but the system has a finite rewrite budget.

A single global query stream accesses heterogeneous chunks:
  * early sustained scan -> point drift;
  * late sustained scan -> point drift;
  * brief point bursts;
  * cyclic scan/point changes;
  * scan-only controls.

The controller remains frozen:
  * long observation window = 300 local accesses;
  * short observation window = 100 local accesses;
  * decision interval = 100 local accesses;
  * minimum observations = 200;
  * confirmation rounds = 3;
  * cooldown = 300 local accesses;
  * prediction horizon = 1200;
  * minimum relative improvement = 8%;
  * rewrite safety factor = 1.25.

Compared policies
-----------------
  initial_best:
      Pick the initial scan-heavy layout once and never migrate.

  unbounded_repressdb:
      Execute every confirmed RePressDB migration proposal immediately.
      This is an adaptation upper-bound without a global rewrite budget.

  fifo_budget_{1,2,4}x:
      Queue confirmed proposals and spend the interval budget in FIFO order.

  benefit_budget_{1,2,4}x:
      Queue confirmed proposals and prioritize positive net benefit per rewrite
      byte, with a small aging term to avoid starvation.

Rewrite budget
--------------
The program physically measures directed codec migration costs and rewrite
bytes for every chunk. One budget unit is the maximum measured non-identity
rewrite byte count, so a 1x budget can always admit at least one migration in
an interval. Budgets reset every 1000 global queries.

Execution model
---------------
Queries are executed for real using the E0-C warm_slicedmeta mode.
Candidate representation files are prebuilt. Migration time and bytes are
physically measured once per directed codec pair and replayed when a policy
changes the active representation. This isolates scheduling effects from
repeated fsync noise while retaining measured rewrite cost.

Outputs
-------
results_e4_pems04_global_budget_v050/
  calibration_records.csv
  codec_records.csv
  measured_migration_costs.csv
  chunk_workload_assignments.csv
  policy_run_records.csv
  chunk_policy_summary.csv
  scheduler_events.csv
  policy_aggregate_summary.csv
  budget_comparison_summary.csv
  dataset_summary.json
  e4_assessment.json
  run_config.json
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

VERSION = "0.5.0"
TARGET_VENUE = "EDBT 2027"
BASE_E1_VERSION = "0.1.0"
SEED = 20260718

TRACE_SEEDS = (20260718, 20260719, 20260720)
CHUNK_SIZES = [8192, 16384]
MAX_SERIES = 24
MAX_CHUNKS_PER_SIZE = 12
GLOBAL_TRACE_LENGTH = 48000

DECISION_INTERVAL = 100
LONG_WINDOW = 300
SHORT_WINDOW = 100
MIN_OBSERVATIONS = 200
CONFIRMATION_ROUNDS = 3
COOLDOWN_QUERIES = 300
PREDICTION_HORIZON = 1200
MIN_RELATIVE_IMPROVEMENT = 0.08
REWRITE_SAFETY_FACTOR = 1.25

BUDGET_INTERVAL_GLOBAL_QUERIES = 1000
BUDGET_MULTIPLIERS = (1, 2, 4)
MIGRATION_MEASURE_REPEATS = 1

OUTPUT_DIR_NAME = "results_e4_pems04_global_budget_v050"
SUMMARY_JSON_NAME = "E4_PEMS04_GLOBAL_BUDGET_ASSESSMENT_v050.json"

QUICK_TEST = os.environ.get("REPRESSDB_E4_QUICK", "0") == "1"


@dataclass(frozen=True)
class GlobalTraceItem:
    global_index: int
    chunk_id: str
    workload: str
    query: Any


@dataclass
class ChunkOnlineState:
    active_codec: str
    recent_classes: Deque[str] = field(
        default_factory=lambda: deque(maxlen=LONG_WINDOW)
    )
    local_access_count: int = 0
    last_switch_access: int = -10**9
    pending_candidate: Optional[str] = None
    pending_rounds: int = 0
    queued_target: Optional[str] = None
    query_time_ns: int = 0
    query_count: int = 0
    migration_time_ns: int = 0
    migration_count: int = 0
    rewrite_bytes: int = 0
    proposal_count: int = 0
    executed_proposal_count: int = 0
    cancelled_proposal_count: int = 0


@dataclass
class MigrationProposal:
    proposal_id: str
    chunk_id: str
    old_codec: str
    target_codec: str
    created_global_index: int
    created_local_access: int
    predicted_saving_ns: float
    rewrite_estimate_ns: int
    rewrite_bytes: int
    net_benefit_ns: float
    relative_improvement: float
    wait_intervals: int = 0


class GlobalCandidateHandles:
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

    def __enter__(self) -> "GlobalCandidateHandles":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def stable_seed(*parts: Any) -> int:
    token = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def load_e1_module() -> Any:
    here = Path(__file__).resolve().parent
    path = here / "RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT.py"
    if not path.exists():
        raise FileNotFoundError(
            "找不到 RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT.py。"
            "请把它与 E4 主程序放在同一文件夹。"
        )
    spec = importlib.util.spec_from_file_location(
        "repressdb_e1_for_e4", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载基础文件：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if getattr(module, "VERSION", None) != BASE_E1_VERSION:
        raise RuntimeError(
            f"需要 E1 基础版本 {BASE_E1_VERSION}，当前为 "
            f"{getattr(module, 'VERSION', 'unknown')}"
        )
    return module


def workload_cost(
    e1: Any,
    profile: Mapping[str, Mapping[str, float]],
    codec: str,
    weights: Mapping[str, float],
) -> float:
    return float(
        sum(
            float(weights.get(query_class, 0.0))
            * float(profile[codec][query_class])
            for query_class in e1.QUERY_CLASSES
        )
    )


def observed_weights(
    e1: Any,
    query_classes: Iterable[str],
) -> Dict[str, float]:
    counts = Counter(query_classes)
    total = int(sum(counts.values()))
    if total <= 0:
        return {query_class: 0.0 for query_class in e1.QUERY_CLASSES}
    return {
        query_class: counts.get(query_class, 0) / total
        for query_class in e1.QUERY_CLASSES
    }


def choose_best_codec(
    e1: Any,
    profile: Mapping[str, Mapping[str, float]],
    weights: Mapping[str, float],
) -> Tuple[str, float]:
    values = [
        (
            codec,
            workload_cost(e1, profile, codec, weights),
        )
        for codec in sorted(profile)
    ]
    return min(values, key=lambda item: (item[1], item[0]))


def assign_workload_types(
    chunks: Sequence[Any],
) -> Dict[str, str]:
    """
    Deterministically assign heterogeneous workload patterns.

    Every eight chunks contain:
      3 early sustained drift
      2 late sustained drift
      1 brief burst
      1 cyclic drift
      1 scan-only control
    """
    types = (
        "early_sustained",
        "early_sustained",
        "early_sustained",
        "late_sustained",
        "late_sustained",
        "brief_burst",
        "cyclic",
        "scan_only",
    )
    ordered = sorted(
        chunks,
        key=lambda chunk: (
            int(chunk.chunk_size),
            int(chunk.series_index),
            int(chunk.chunk_start),
            chunk.chunk_id,
        ),
    )
    return {
        chunk.chunk_id: types[index % len(types)]
        for index, chunk in enumerate(ordered)
    }


def workload_at_global_fraction(
    workload_type: str,
    fraction: float,
) -> str:
    if workload_type == "early_sustained":
        return "scan_heavy" if fraction < 0.25 else "point_heavy"
    if workload_type == "late_sustained":
        return "scan_heavy" if fraction < 0.55 else "point_heavy"
    if workload_type == "brief_burst":
        return (
            "point_heavy"
            if 0.40 <= fraction < 0.48
            else "scan_heavy"
        )
    if workload_type == "cyclic":
        phase = min(3, int(fraction * 4.0))
        return (
            "scan_heavy"
            if phase in (0, 2)
            else "point_heavy"
        )
    if workload_type == "scan_only":
        return "scan_heavy"
    raise ValueError(f"Unknown workload type: {workload_type}")


def build_global_trace(
    e1: Any,
    base: Any,
    chunks: Sequence[Any],
    query_pools: Mapping[str, Mapping[str, List[Any]]],
    workload_types: Mapping[str, str],
    trace_seed: int,
    trace_length: int,
) -> Tuple[List[GlobalTraceItem], Dict[str, float]]:
    rng = np.random.default_rng(
        stable_seed(trace_seed, "global_trace")
    )
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    n = len(chunk_ids)

    ranks = np.arange(1, n + 1, dtype=float)
    zipf = 1.0 / np.power(ranks, 0.8)
    zipf /= zipf.sum()
    permutation = rng.permutation(n)
    popularity = np.empty(n, dtype=float)
    popularity[permutation] = zipf
    popularity = 0.80 * popularity + 0.20 / n
    popularity /= popularity.sum()

    items: List[GlobalTraceItem] = []
    for global_index in range(trace_length):
        chunk_position = int(
            rng.choice(n, p=popularity)
        )
        chunk_id = chunk_ids[chunk_position]
        fraction = (
            global_index / max(1, trace_length - 1)
        )
        workload = workload_at_global_fraction(
            workload_types[chunk_id],
            fraction,
        )
        weights = base.WORKLOADS[workload]
        classes = [
            query_class
            for query_class in e1.QUERY_CLASSES
            if float(weights.get(query_class, 0.0)) > 0.0
        ]
        probabilities = np.asarray(
            [float(weights[query_class]) for query_class in classes],
            dtype=float,
        )
        probabilities /= probabilities.sum()
        query_class = str(
            rng.choice(classes, p=probabilities)
        )
        pool = query_pools[chunk_id][query_class]
        query = pool[int(rng.integers(0, len(pool)))]
        items.append(
            GlobalTraceItem(
                global_index=global_index,
                chunk_id=chunk_id,
                workload=workload,
                query=query,
            )
        )

    return items, {
        chunk_id: float(popularity[index])
        for index, chunk_id in enumerate(chunk_ids)
    }


def measure_directed_migrations(
    e1: Any,
    base: Any,
    reps: Mapping[str, Any],
    chunks_by_id: Mapping[str, Any],
    stored_by_chunk: Mapping[str, Mapping[str, Any]],
    migration_root: Path,
    repeats: int,
) -> Tuple[
    Dict[str, Dict[Tuple[str, str], int]],
    Dict[str, Dict[Tuple[str, str], int]],
    List[Dict[str, Any]],
]:
    cost_map: Dict[str, Dict[Tuple[str, str], int]] = {}
    byte_map: Dict[str, Dict[Tuple[str, str], int]] = {}
    rows: List[Dict[str, Any]] = []

    for chunk_id, chunk in chunks_by_id.items():
        stored_map = stored_by_chunk[chunk_id]
        cost_map[chunk_id] = {}
        byte_map[chunk_id] = {}
        codecs = sorted(stored_map)

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
                            "chunk_start": int(chunk.chunk_start),
                            "chunk_size": int(chunk.chunk_size),
                            "current_codec": current_codec,
                            "target_codec": target_codec,
                            "n_repeats": 0,
                            "median_migration_ns": 0,
                            "min_migration_ns": 0,
                            "max_migration_ns": 0,
                            "rewrite_bytes": 0,
                        }
                    )
                    continue

                timings: List[int] = []
                bytes_values: List[int] = []
                for repeat in range(repeats):
                    elapsed_ns, rewritten = (
                        e1.perform_actual_migration(
                            base,
                            reps,
                            stored_map,
                            chunk,
                            current_codec,
                            target_codec,
                            migration_root / chunk_id,
                            (
                                f"measure__{current_codec}"
                                f"__to__{target_codec}__r{repeat}"
                            ),
                        )
                    )
                    timings.append(int(elapsed_ns))
                    bytes_values.append(int(rewritten))

                median_ns = int(
                    np.median(
                        np.asarray(timings, dtype=np.int64)
                    )
                )
                median_bytes = int(
                    np.median(
                        np.asarray(bytes_values, dtype=np.int64)
                    )
                )
                cost_map[chunk_id][key] = median_ns
                byte_map[chunk_id][key] = median_bytes
                rows.append(
                    {
                        "chunk_id": chunk_id,
                        "series_index": int(chunk.series_index),
                        "chunk_start": int(chunk.chunk_start),
                        "chunk_size": int(chunk.chunk_size),
                        "current_codec": current_codec,
                        "target_codec": target_codec,
                        "n_repeats": int(repeats),
                        "median_migration_ns": median_ns,
                        "min_migration_ns": int(min(timings)),
                        "max_migration_ns": int(max(timings)),
                        "rewrite_bytes": median_bytes,
                    }
                )

    return cost_map, byte_map, rows


def evaluate_chunk_controller(
    e1: Any,
    state: ChunkOnlineState,
    profile: Mapping[str, Mapping[str, float]],
    migration_costs: Mapping[Tuple[str, str], int],
    migration_bytes: Mapping[Tuple[str, str], int],
    global_index: int,
) -> Tuple[
    Optional[MigrationProposal],
    Dict[str, Any],
]:
    recent_long = list(state.recent_classes)
    long_weights = observed_weights(e1, recent_long)
    short_weights = observed_weights(
        e1,
        recent_long[-SHORT_WINDOW:],
    )

    long_target, long_target_cost = choose_best_codec(
        e1, profile, long_weights
    )
    short_target, short_target_cost = choose_best_codec(
        e1, profile, short_weights
    )
    current_long_cost = workload_cost(
        e1, profile, state.active_codec, long_weights
    )
    current_short_cost = workload_cost(
        e1, profile, state.active_codec, short_weights
    )
    long_relative = (
        (current_long_cost - long_target_cost)
        / current_long_cost
        if current_long_cost > 0
        else 0.0
    )
    short_relative = (
        (current_short_cost - short_target_cost)
        / current_short_cost
        if current_short_cost > 0
        else 0.0
    )
    conservative_relative = min(
        long_relative, short_relative
    )
    conservative_gain = max(
        0.0,
        min(
            current_long_cost - long_target_cost,
            current_short_cost - short_target_cost,
        ),
    )
    target = long_target
    rewrite_estimate = int(
        migration_costs.get(
            (state.active_codec, target), 0
        )
    )
    rewrite_byte_count = int(
        migration_bytes.get(
            (state.active_codec, target), 0
        )
    )
    predicted_saving = (
        conservative_gain * PREDICTION_HORIZON
    )
    net_benefit = (
        predicted_saving
        - REWRITE_SAFETY_FACTOR * rewrite_estimate
    )

    detail = {
        "global_index": int(global_index),
        "local_access_count": int(state.local_access_count),
        "old_codec": state.active_codec,
        "long_target_codec": long_target,
        "short_target_codec": short_target,
        "relative_improvement": float(
            conservative_relative
        ),
        "predicted_saving_ns": float(
            predicted_saving
        ),
        "rewrite_estimate_ns": int(
            rewrite_estimate
        ),
        "rewrite_bytes": int(rewrite_byte_count),
        "net_benefit_ns": float(net_benefit),
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
        state.local_access_count - state.last_switch_access
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
    state.proposal_count += 1

    proposal = MigrationProposal(
        proposal_id=(
            f"{global_index}:{state.local_access_count}:"
            f"{state.active_codec}->{target}"
        ),
        chunk_id="",
        old_codec=state.active_codec,
        target_codec=target,
        created_global_index=int(global_index),
        created_local_access=int(
            state.local_access_count
        ),
        predicted_saving_ns=float(predicted_saving),
        rewrite_estimate_ns=int(rewrite_estimate),
        rewrite_bytes=int(rewrite_byte_count),
        net_benefit_ns=float(net_benefit),
        relative_improvement=float(
            conservative_relative
        ),
    )
    detail["reason"] = "proposal"
    return proposal, detail


def proposal_priority(
    proposal: MigrationProposal,
    scheduling_policy: str,
) -> Tuple[float, int, str]:
    if scheduling_policy == "fifo":
        # The caller sorts FIFO proposals in ascending order.
        # Older proposals therefore use the smaller creation index.
        return (
            float(proposal.created_global_index),
            -proposal.wait_intervals,
            proposal.chunk_id,
        )

    density = proposal.net_benefit_ns / max(
        1, proposal.rewrite_bytes
    )
    aging = 1.0 + 0.05 * proposal.wait_intervals
    score = density * aging
    return (
        float(score),
        -proposal.created_global_index,
        proposal.chunk_id,
    )


def execute_proposal(
    proposal: MigrationProposal,
    state: ChunkOnlineState,
    global_index: int,
    policy_name: str,
) -> Dict[str, Any]:
    state.active_codec = proposal.target_codec
    state.last_switch_access = state.local_access_count
    state.migration_count += 1
    state.migration_time_ns += int(
        proposal.rewrite_estimate_ns
    )
    state.rewrite_bytes += int(proposal.rewrite_bytes)
    state.executed_proposal_count += 1
    state.queued_target = None

    return {
        "event": "execute",
        "policy": policy_name,
        "global_index": int(global_index),
        "chunk_id": proposal.chunk_id,
        "old_codec": proposal.old_codec,
        "target_codec": proposal.target_codec,
        "proposal_global_index": int(
            proposal.created_global_index
        ),
        "queue_delay_queries": int(
            global_index - proposal.created_global_index
        ),
        "predicted_saving_ns": float(
            proposal.predicted_saving_ns
        ),
        "rewrite_estimate_ns": int(
            proposal.rewrite_estimate_ns
        ),
        "rewrite_bytes": int(proposal.rewrite_bytes),
        "net_benefit_ns": float(
            proposal.net_benefit_ns
        ),
        "relative_improvement": float(
            proposal.relative_improvement
        ),
    }


def run_global_policy(
    e1: Any,
    base: Any,
    reps: Mapping[str, Any],
    chunks_by_id: Mapping[str, Any],
    stored_by_chunk: Mapping[str, Mapping[str, Any]],
    profiles: Mapping[
        str, Mapping[str, Mapping[str, float]]
    ],
    trace: Sequence[GlobalTraceItem],
    migration_costs: Mapping[
        str, Mapping[Tuple[str, str], int]
    ],
    migration_bytes: Mapping[
        str, Mapping[Tuple[str, str], int]
    ],
    policy_name: str,
    budget_bytes_per_interval: Optional[int],
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    mode = base.CACHE_MODES["warm_slicedmeta"]

    states: Dict[str, ChunkOnlineState] = {}
    for chunk_id, profile in profiles.items():
        initial_codec = choose_best_codec(
            e1,
            profile,
            base.WORKLOADS["scan_heavy"],
        )[0]
        states[chunk_id] = ChunkOnlineState(
            active_codec=initial_codec
        )

    scheduler_events: List[Dict[str, Any]] = []
    queued: Dict[str, MigrationProposal] = {}
    query_latencies: List[int] = []
    query_time_ns = 0
    migration_time_ns = 0
    total_rewrite_bytes = 0
    budget_violation_count = 0
    budget_spent_by_interval: Dict[int, int] = {}
    peak_queue_length = 0

    if policy_name.startswith("fifo_budget"):
        scheduling_policy = "fifo"
    elif policy_name.startswith("benefit_budget"):
        scheduling_policy = "benefit"
    elif policy_name == "unbounded_repressdb":
        scheduling_policy = "unbounded"
    elif policy_name == "initial_best":
        scheduling_policy = "none"
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    with GlobalCandidateHandles(
        stored_by_chunk
    ) as handles:
        # Warm every candidate file once before the timed replay.
        first_query_by_chunk: Dict[str, Any] = {}
        for item in trace:
            first_query_by_chunk.setdefault(
                item.chunk_id, item.query
            )
            if len(first_query_by_chunk) == len(
                chunks_by_id
            ):
                break

        for chunk_id, stored_map in stored_by_chunk.items():
            chunk = chunks_by_id[chunk_id]
            warm_query = first_query_by_chunk[chunk_id]
            truth = base.execute_on_array(
                chunk.values, warm_query
            )
            for codec, rep in reps.items():
                result, _ = base.run_query_once(
                    rep,
                    stored_map[codec],
                    warm_query,
                    mode,
                    handles.payload[chunk_id][codec],
                    handles.metadata[chunk_id][codec],
                )
                if not base.results_equal(result, truth):
                    raise AssertionError(
                        f"Warm-up failed: {chunk_id}, {codec}"
                    )

        for item in trace:
            chunk_id = item.chunk_id
            chunk = chunks_by_id[chunk_id]
            state = states[chunk_id]
            codec = state.active_codec
            rep = reps[codec]
            stored = stored_by_chunk[chunk_id][codec]

            start_ns = time.perf_counter_ns()
            result, _ = base.run_query_once(
                rep,
                stored,
                item.query,
                mode,
                handles.payload[chunk_id][codec],
                handles.metadata[chunk_id][codec],
            )
            latency_ns = (
                time.perf_counter_ns() - start_ns
            )
            truth = base.execute_on_array(
                chunk.values, item.query
            )
            if not base.results_equal(result, truth):
                raise AssertionError(
                    f"Incorrect global query: "
                    f"{chunk_id}, {codec}, "
                    f"{item.global_index}"
                )

            query_latencies.append(int(latency_ns))
            query_time_ns += int(latency_ns)
            state.query_time_ns += int(latency_ns)
            state.query_count += 1
            state.local_access_count += 1
            state.recent_classes.append(
                item.query.query_class
            )

            if (
                scheduling_policy != "none"
                and state.local_access_count
                >= MIN_OBSERVATIONS
                and state.local_access_count
                % DECISION_INTERVAL
                == 0
            ):
                proposal, detail = (
                    evaluate_chunk_controller(
                        e1,
                        state,
                        profiles[chunk_id],
                        migration_costs[chunk_id],
                        migration_bytes[chunk_id],
                        item.global_index,
                    )
                )
                detail.update(
                    {
                        "event": "decision",
                        "policy": policy_name,
                        "chunk_id": chunk_id,
                        "workload": item.workload,
                        "active_codec": state.active_codec,
                    }
                )
                scheduler_events.append(detail)

                if proposal is not None:
                    proposal.chunk_id = chunk_id
                    proposal.proposal_id = (
                        f"{policy_name}:{chunk_id}:"
                        f"{proposal.proposal_id}"
                    )

                    if scheduling_policy == "unbounded":
                        event = execute_proposal(
                            proposal,
                            state,
                            item.global_index,
                            policy_name,
                        )
                        scheduler_events.append(event)
                        migration_time_ns += int(
                            proposal.rewrite_estimate_ns
                        )
                        total_rewrite_bytes += int(
                            proposal.rewrite_bytes
                        )
                    else:
                        existing = queued.get(chunk_id)
                        if existing is not None:
                            state.cancelled_proposal_count += 1
                            scheduler_events.append(
                                {
                                    "event": "cancel_replace",
                                    "policy": policy_name,
                                    "global_index": int(
                                        item.global_index
                                    ),
                                    "chunk_id": chunk_id,
                                    "old_target_codec": (
                                        existing.target_codec
                                    ),
                                    "new_target_codec": (
                                        proposal.target_codec
                                    ),
                                }
                            )
                        queued[chunk_id] = proposal
                        state.queued_target = (
                            proposal.target_codec
                        )
                        scheduler_events.append(
                            {
                                "event": "enqueue",
                                "policy": policy_name,
                                "global_index": int(
                                    item.global_index
                                ),
                                "chunk_id": chunk_id,
                                "old_codec": proposal.old_codec,
                                "target_codec": (
                                    proposal.target_codec
                                ),
                                "predicted_saving_ns": float(
                                    proposal.predicted_saving_ns
                                ),
                                "rewrite_estimate_ns": int(
                                    proposal.rewrite_estimate_ns
                                ),
                                "rewrite_bytes": int(
                                    proposal.rewrite_bytes
                                ),
                                "net_benefit_ns": float(
                                    proposal.net_benefit_ns
                                ),
                            }
                        )

            # Invalidate stale queued proposals at each local decision point.
            if (
                scheduling_policy in {"fifo", "benefit"}
                and chunk_id in queued
                and state.local_access_count
                % DECISION_INTERVAL
                == 0
            ):
                queued_proposal = queued[chunk_id]
                recent_long = list(state.recent_classes)
                long_weights = observed_weights(
                    e1, recent_long
                )
                short_weights = observed_weights(
                    e1,
                    recent_long[-SHORT_WINDOW:],
                )
                long_target = choose_best_codec(
                    e1,
                    profiles[chunk_id],
                    long_weights,
                )[0]
                short_target = choose_best_codec(
                    e1,
                    profiles[chunk_id],
                    short_weights,
                )[0]
                if (
                    long_target != queued_proposal.target_codec
                    or short_target
                    != queued_proposal.target_codec
                    or state.active_codec
                    == queued_proposal.target_codec
                ):
                    queued.pop(chunk_id, None)
                    state.queued_target = None
                    state.cancelled_proposal_count += 1
                    scheduler_events.append(
                        {
                            "event": "cancel_stale",
                            "policy": policy_name,
                            "global_index": int(
                                item.global_index
                            ),
                            "chunk_id": chunk_id,
                            "queued_target_codec": (
                                queued_proposal.target_codec
                            ),
                            "long_target_codec": long_target,
                            "short_target_codec": short_target,
                        }
                    )

            peak_queue_length = max(
                peak_queue_length, len(queued)
            )

            is_budget_boundary = (
                scheduling_policy in {"fifo", "benefit"}
                and (
                    (item.global_index + 1)
                    % BUDGET_INTERVAL_GLOBAL_QUERIES
                    == 0
                    or item.global_index
                    == len(trace) - 1
                )
            )
            if is_budget_boundary:
                interval_id = (
                    item.global_index
                    // BUDGET_INTERVAL_GLOBAL_QUERIES
                )
                budget = int(
                    budget_bytes_per_interval or 0
                )
                remaining = budget
                spent = 0

                proposals = list(queued.values())
                for proposal in proposals:
                    proposal.wait_intervals += 1

                reverse = scheduling_policy == "benefit"
                proposals.sort(
                    key=lambda proposal: proposal_priority(
                        proposal,
                        scheduling_policy,
                    ),
                    reverse=reverse,
                )

                for proposal in proposals:
                    current_state = states[
                        proposal.chunk_id
                    ]
                    if (
                        current_state.active_codec
                        != proposal.old_codec
                    ):
                        queued.pop(
                            proposal.chunk_id, None
                        )
                        current_state.queued_target = None
                        current_state.cancelled_proposal_count += 1
                        continue

                    if proposal.rewrite_bytes > remaining:
                        continue

                    queued.pop(
                        proposal.chunk_id, None
                    )
                    event = execute_proposal(
                        proposal,
                        current_state,
                        item.global_index,
                        policy_name,
                    )
                    event["budget_interval"] = int(
                        interval_id
                    )
                    event["budget_bytes"] = int(budget)
                    event["budget_remaining_before"] = int(
                        remaining
                    )
                    scheduler_events.append(event)

                    remaining -= int(
                        proposal.rewrite_bytes
                    )
                    spent += int(proposal.rewrite_bytes)
                    migration_time_ns += int(
                        proposal.rewrite_estimate_ns
                    )
                    total_rewrite_bytes += int(
                        proposal.rewrite_bytes
                    )

                budget_spent_by_interval[
                    int(interval_id)
                ] = int(spent)
                if spent > budget:
                    budget_violation_count += 1

                scheduler_events.append(
                    {
                        "event": "budget_close",
                        "policy": policy_name,
                        "global_index": int(
                            item.global_index
                        ),
                        "budget_interval": int(
                            interval_id
                        ),
                        "budget_bytes": int(budget),
                        "spent_bytes": int(spent),
                        "remaining_bytes": int(
                            budget - spent
                        ),
                        "queued_after": int(len(queued)),
                    }
                )

    total_time_ns = query_time_ns + migration_time_ns
    latency_array = np.asarray(
        query_latencies, dtype=np.int64
    )

    chunk_rows: List[Dict[str, Any]] = []
    for chunk_id, state in states.items():
        chunk = chunks_by_id[chunk_id]
        chunk_rows.append(
            {
                "policy": policy_name,
                "chunk_id": chunk_id,
                "series_index": int(
                    chunk.series_index
                ),
                "chunk_start": int(
                    chunk.chunk_start
                ),
                "chunk_size": int(
                    chunk.chunk_size
                ),
                "query_count": int(state.query_count),
                "query_time_ns": int(
                    state.query_time_ns
                ),
                "migration_time_ns": int(
                    state.migration_time_ns
                ),
                "total_time_ns": int(
                    state.query_time_ns
                    + state.migration_time_ns
                ),
                "migration_count": int(
                    state.migration_count
                ),
                "rewrite_bytes": int(
                    state.rewrite_bytes
                ),
                "proposal_count": int(
                    state.proposal_count
                ),
                "executed_proposal_count": int(
                    state.executed_proposal_count
                ),
                "cancelled_proposal_count": int(
                    state.cancelled_proposal_count
                ),
                "final_codec": state.active_codec,
            }
        )

    summary = {
        "policy": policy_name,
        "n_queries": int(len(trace)),
        "query_time_ns": int(query_time_ns),
        "migration_time_ns": int(
            migration_time_ns
        ),
        "total_time_ns": int(total_time_ns),
        "throughput_qps": float(
            len(trace) / (total_time_ns / 1e9)
            if total_time_ns > 0
            else math.inf
        ),
        "median_query_latency_ns": float(
            np.median(latency_array)
        ),
        "p95_query_latency_ns": float(
            np.quantile(latency_array, 0.95)
        ),
        "p99_query_latency_ns": float(
            np.quantile(latency_array, 0.99)
        ),
        "migration_count": int(
            sum(
                state.migration_count
                for state in states.values()
            )
        ),
        "rewrite_bytes": int(total_rewrite_bytes),
        "proposal_count": int(
            sum(
                state.proposal_count
                for state in states.values()
            )
        ),
        "executed_proposal_count": int(
            sum(
                state.executed_proposal_count
                for state in states.values()
            )
        ),
        "cancelled_proposal_count": int(
            sum(
                state.cancelled_proposal_count
                for state in states.values()
            )
        ),
        "peak_queue_length": int(
            peak_queue_length
        ),
        "remaining_queue_length": int(len(queued)),
        "budget_bytes_per_interval": (
            int(budget_bytes_per_interval)
            if budget_bytes_per_interval
            is not None
            else math.nan
        ),
        "budget_violation_count": int(
            budget_violation_count
        ),
        "mean_budget_utilization": (
            float(
                np.mean(
                    [
                        spent
                        / max(
                            1,
                            int(
                                budget_bytes_per_interval
                            ),
                        )
                        for spent in budget_spent_by_interval.values()
                    ]
                )
            )
            if budget_spent_by_interval
            and budget_bytes_per_interval
            else math.nan
        ),
    }
    return summary, chunk_rows, scheduler_events


def aggregate_policy_runs(
    run_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for policy, group in run_df.groupby(
        "policy", sort=True
    ):
        rows.append(
            {
                "policy": policy,
                "n_trace_seeds": int(
                    group["trace_seed"].nunique()
                ),
                "median_total_time_ns": float(
                    group["total_time_ns"].median()
                ),
                "mean_total_time_ns": float(
                    group["total_time_ns"].mean()
                ),
                "median_query_time_ns": float(
                    group["query_time_ns"].median()
                ),
                "median_migration_time_ns": float(
                    group[
                        "migration_time_ns"
                    ].median()
                ),
                "median_migration_count": float(
                    group["migration_count"].median()
                ),
                "median_rewrite_bytes": float(
                    group["rewrite_bytes"].median()
                ),
                "median_p95_query_latency_ns": float(
                    group[
                        "p95_query_latency_ns"
                    ].median()
                ),
                "median_peak_queue_length": float(
                    group[
                        "peak_queue_length"
                    ].median()
                ),
                "median_budget_utilization": float(
                    group[
                        "mean_budget_utilization"
                    ].median()
                )
                if group[
                    "mean_budget_utilization"
                ].notna().any()
                else math.nan,
                "budget_violation_count": int(
                    group[
                        "budget_violation_count"
                    ].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_budget_comparisons(
    run_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    pivot = run_df.pivot(
        index="trace_seed",
        columns="policy",
        values="total_time_ns",
    )

    for multiplier in BUDGET_MULTIPLIERS:
        fifo = f"fifo_budget_{multiplier}x"
        benefit = f"benefit_budget_{multiplier}x"
        required = [
            "initial_best",
            "unbounded_repressdb",
            fifo,
            benefit,
        ]
        local = pivot.dropna(subset=required)
        if local.empty:
            continue

        for policy, reference in (
            (fifo, "initial_best"),
            (benefit, "initial_best"),
            (benefit, fifo),
            (benefit, "unbounded_repressdb"),
        ):
            ratios = (
                local[policy].to_numpy(dtype=float)
                / local[reference].to_numpy(dtype=float)
            )
            rows.append(
                {
                    "budget_multiplier": int(
                        multiplier
                    ),
                    "policy": policy,
                    "reference_policy": reference,
                    "n_trace_seeds": int(
                        len(ratios)
                    ),
                    "median_ratio_policy_over_reference": float(
                        np.median(ratios)
                    ),
                    "mean_ratio_policy_over_reference": float(
                        np.mean(ratios)
                    ),
                    "fraction_policy_faster": float(
                        np.mean(ratios < 1.0)
                    ),
                    "minimum_ratio": float(
                        np.min(ratios)
                    ),
                    "maximum_ratio": float(
                        np.max(ratios)
                    ),
                }
            )

    return pd.DataFrame(rows)


def make_assessment(
    aggregate_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    base_budget_bytes: int,
) -> Dict[str, Any]:
    def ratio(
        multiplier: int,
        policy: str,
        reference: str,
    ) -> float:
        local = comparison_df[
            (
                comparison_df[
                    "budget_multiplier"
                ]
                == multiplier
            )
            & (comparison_df["policy"] == policy)
            & (
                comparison_df[
                    "reference_policy"
                ]
                == reference
            )
        ]
        return (
            float(
                local.iloc[0][
                    "median_ratio_policy_over_reference"
                ]
            )
            if not local.empty
            else math.nan
        )

    fifo_wins = 0
    benefit_vs_fifo: Dict[str, Any] = {}
    for multiplier in BUDGET_MULTIPLIERS:
        value = ratio(
            multiplier,
            f"benefit_budget_{multiplier}x",
            f"fifo_budget_{multiplier}x",
        )
        benefit_vs_fifo[str(multiplier)] = value
        if np.isfinite(value) and value <= 1.0:
            fifo_wins += 1

    benefit_4x_vs_unbounded = ratio(
        4,
        "benefit_budget_4x",
        "unbounded_repressdb",
    )

    violations = int(
        aggregate_df[
            "budget_violation_count"
        ].sum()
    )

    checks = {
        "B1_no_budget_violations": bool(
            violations == 0
        ),
        "B2_benefit_scheduler_matches_or_beats_fifo_at_least_two_budget_levels": bool(
            fifo_wins >= 2
        ),
        "B3_benefit_4x_is_within_10_percent_of_unbounded": bool(
            np.isfinite(
                benefit_4x_vs_unbounded
            )
            and benefit_4x_vs_unbounded <= 1.10
        ),
        "B4_budgeted_scheduler_reduces_rewrite_bytes_relative_to_unbounded_at_1x": bool(
            float(
                aggregate_df.loc[
                    aggregate_df["policy"]
                    == "benefit_budget_1x",
                    "median_rewrite_bytes",
                ].iloc[0]
            )
            <= float(
                aggregate_df.loc[
                    aggregate_df["policy"]
                    == "unbounded_repressdb",
                    "median_rewrite_bytes",
                ].iloc[0]
            )
        ),
    }

    policy_summary = (
        aggregate_df.set_index("policy")
        .to_dict(orient="index")
    )

    return {
        "version": VERSION,
        "target_venue": TARGET_VENUE,
        "status": (
            "PEMS04 heterogeneous multi-chunk "
            "global rewrite-budget scheduling study"
        ),
        "study_role": (
            "System-level scheduler evaluation after "
            "freezing the per-chunk controller."
        ),
        "primary_execution_mode": "warm_slicedmeta",
        "base_budget_bytes_per_interval": int(
            base_budget_bytes
        ),
        "budget_interval_global_queries": int(
            BUDGET_INTERVAL_GLOBAL_QUERIES
        ),
        "descriptive_checks": {
            **checks,
            "all_four": bool(all(checks.values())),
        },
        "benefit_vs_fifo_ratio_by_budget": (
            benefit_vs_fifo
        ),
        "benefit_4x_vs_unbounded_ratio": (
            benefit_4x_vs_unbounded
        ),
        "policy_summary": policy_summary,
        "important_limitations": [
            "This is a PEMS04 system-mechanism experiment, not a new holdout.",
            "Global queries are replayed sequentially; migration time is added to end-to-end elapsed time rather than executed by a real background I/O thread.",
            "Migration costs and bytes are physically measured once per directed codec pair and replayed during policy comparison.",
            "Queries execute in warm_slicedmeta mode; this is not a cold-SSD claim.",
        ],
        "next_gate": (
            "Use E4 to establish the value of global "
            "benefit-aware migration scheduling. Do not "
            "retune the frozen per-chunk controller."
        ),
    }


def write_json(path: Path, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        )


def main() -> int:
    e1 = load_e1_module()
    base = e1.load_base_module()

    print("=" * 100)
    print(
        f"RePressDB E4 global rewrite-budget scheduling v{VERSION}"
    )
    print(f"Target: {TARGET_VENUE}")
    print(
        "Direct-run PEMS04 experiment; no command-line "
        "arguments are required."
    )
    print("=" * 100)

    output_dir = Path(
        "results_e4_quick_smoke"
        if QUICK_TEST
        else OUTPUT_DIR_NAME
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    storage_dir = output_dir / "_storage"
    migration_dir = (
        output_dir / "_migration_measurements"
    )
    storage_dir.mkdir(parents=True, exist_ok=True)
    migration_dir.mkdir(parents=True, exist_ok=True)

    if QUICK_TEST:
        matrix, dataset_meta = base.synthetic_dataset(
            SEED, 4
        )
        dataset_label = (
            "SYNTHETIC_E4_GLOBAL_BUDGET_SMOKE"
        )
        chunk_sizes = [2048, 8192]
        max_chunks_per_size = 2
        trace_seeds = (SEED,)
        trace_length = 2000
        migration_repeats = 1
    else:
        pems04_dir, _ = base.default_data_dirs()
        data_file = base.auto_detect_data_file(
            pems04_dir,
            "PEMS04_E4_GLOBAL_BUDGET",
        )
        matrix, dataset_meta = base.load_dataset(
            data_file,
            feature_index=0,
            max_series=MAX_SERIES,
            seed=SEED,
            excluded_series=None,
        )
        dataset_label = (
            "PEMS04_E4_GLOBAL_BUDGET_SCHEDULING"
        )
        chunk_sizes = CHUNK_SIZES
        max_chunks_per_size = (
            MAX_CHUNKS_PER_SIZE
        )
        trace_seeds = TRACE_SEEDS
        trace_length = GLOBAL_TRACE_LENGTH
        migration_repeats = (
            MIGRATION_MEASURE_REPEATS
        )

    chunks = base.extract_chunks(
        matrix,
        chunk_sizes,
        max_chunks_per_size,
        SEED,
    )
    chunks_by_id = {
        chunk.chunk_id: chunk for chunk in chunks
    }
    workload_types = assign_workload_types(chunks)
    reps = base.representation_registry(3)

    print(
        f"[{dataset_label}] chunks={len(chunks)}, "
        f"global queries/seed={trace_length}, "
        f"trace seeds={len(trace_seeds)}"
    )

    stored_by_chunk: Dict[
        str, Dict[str, Any]
    ] = {}
    profiles: Dict[
        str, Dict[str, Dict[str, float]]
    ] = {}
    query_pools: Dict[
        str, Dict[str, List[Any]]
    ] = {}
    calibration_rows: List[
        Dict[str, Any]
    ] = []
    codec_rows: List[Dict[str, Any]] = []

    for index, chunk in enumerate(chunks, start=1):
        chunk_dir = storage_dir / chunk.chunk_id
        stored_map, encode_ns, decode_ns = (
            e1.encode_candidates(
                base,
                reps,
                chunk,
                chunk_dir,
            )
        )
        stored_by_chunk[chunk.chunk_id] = (
            stored_map
        )

        for codec, stored in stored_map.items():
            codec_rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "series_index": int(
                        chunk.series_index
                    ),
                    "chunk_start": int(
                        chunk.chunk_start
                    ),
                    "chunk_size": int(
                        chunk.chunk_size
                    ),
                    "codec": codec,
                    "raw_size_bytes": int(
                        chunk.values.nbytes
                    ),
                    "payload_size_bytes": int(
                        stored.payload_size_bytes
                    ),
                    "metadata_size_bytes": int(
                        stored.metadata_size_bytes
                    ),
                    "stored_size_bytes": int(
                        stored.stored_size_bytes
                    ),
                    "compression_ratio": float(
                        stored.stored_size_bytes
                        / chunk.values.nbytes
                    ),
                    "encode_ns": int(
                        encode_ns[codec]
                    ),
                    "full_decode_ns": int(
                        decode_ns[codec]
                    ),
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
            for query_class in e1.QUERY_CLASSES
        }
        if any(not values for values in pool.values()):
            raise AssertionError(
                f"Incomplete query pool: "
                f"{chunk.chunk_id}"
            )
        query_pools[chunk.chunk_id] = pool

        profile, local_calibration = (
            e1.calibrate_chunk(
                base,
                reps,
                stored_map,
                chunk,
                pool,
            )
        )
        profiles[chunk.chunk_id] = profile
        for row in local_calibration:
            row.update(
                {
                    "series_index": int(
                        chunk.series_index
                    ),
                    "chunk_start": int(
                        chunk.chunk_start
                    ),
                    "workload_type": (
                        workload_types[
                            chunk.chunk_id
                        ]
                    ),
                }
            )
        calibration_rows.extend(
            local_calibration
        )

        if (
            index == 1
            or index == len(chunks)
            or index
            % max(1, len(chunks) // 8)
            == 0
        ):
            print(
                f"[calibration] "
                f"{index}/{len(chunks)} chunks"
            )

    migration_costs, migration_bytes, migration_rows = (
        measure_directed_migrations(
            e1,
            base,
            reps,
            chunks_by_id,
            stored_by_chunk,
            migration_dir,
            repeats=migration_repeats,
        )
    )

    nonzero_rewrite_bytes = [
        int(row["rewrite_bytes"])
        for row in migration_rows
        if row["current_codec"]
        != row["target_codec"]
        and int(row["rewrite_bytes"]) > 0
    ]
    if not nonzero_rewrite_bytes:
        raise RuntimeError(
            "No non-zero rewrite bytes measured"
        )
    base_budget_bytes = int(
        max(nonzero_rewrite_bytes)
    )

    if QUICK_TEST:
        budget_multipliers = (1, 2)
    else:
        budget_multipliers = BUDGET_MULTIPLIERS

    policies = [
        "initial_best",
        "unbounded_repressdb",
    ]
    for multiplier in budget_multipliers:
        policies.extend(
            [
                f"fifo_budget_{multiplier}x",
                f"benefit_budget_{multiplier}x",
            ]
        )

    run_rows: List[Dict[str, Any]] = []
    chunk_rows: List[Dict[str, Any]] = []
    scheduler_rows: List[Dict[str, Any]] = []
    assignment_rows: List[Dict[str, Any]] = []

    for chunk in chunks:
        assignment_rows.append(
            {
                "chunk_id": chunk.chunk_id,
                "series_index": int(
                    chunk.series_index
                ),
                "chunk_start": int(
                    chunk.chunk_start
                ),
                "chunk_size": int(
                    chunk.chunk_size
                ),
                "workload_type": (
                    workload_types[
                        chunk.chunk_id
                    ]
                ),
            }
        )

    for trace_seed in trace_seeds:
        trace, popularity = build_global_trace(
            e1,
            base,
            chunks,
            query_pools,
            workload_types,
            int(trace_seed),
            int(trace_length),
        )

        policy_order = list(policies)
        random.Random(
            stable_seed(
                trace_seed,
                "policy_order",
            )
        ).shuffle(policy_order)

        for policy_index, policy_name in enumerate(
            policy_order, start=1
        ):
            if policy_name.startswith(
                ("fifo_budget_", "benefit_budget_")
            ):
                multiplier = int(
                    policy_name.rsplit("_", 1)[1]
                    .removesuffix("x")
                )
                budget_bytes = int(
                    multiplier * base_budget_bytes
                )
            else:
                budget_bytes = None

            print(
                f"[trace seed {trace_seed}] "
                f"{policy_index}/{len(policy_order)} "
                f"{policy_name}"
            )

            summary, local_chunk_rows, events = (
                run_global_policy(
                    e1,
                    base,
                    reps,
                    chunks_by_id,
                    stored_by_chunk,
                    profiles,
                    trace,
                    migration_costs,
                    migration_bytes,
                    policy_name,
                    budget_bytes,
                )
            )
            summary["trace_seed"] = int(
                trace_seed
            )
            summary[
                "base_budget_bytes"
            ] = int(base_budget_bytes)
            run_rows.append(summary)

            for row in local_chunk_rows:
                row["trace_seed"] = int(
                    trace_seed
                )
                row["workload_type"] = (
                    workload_types[
                        row["chunk_id"]
                    ]
                )
                row["popularity_probability"] = (
                    float(
                        popularity[
                            row["chunk_id"]
                        ]
                    )
                )
            chunk_rows.extend(local_chunk_rows)

            for row in events:
                row["trace_seed"] = int(
                    trace_seed
                )
            scheduler_rows.extend(events)

    run_df = pd.DataFrame(run_rows)
    chunk_df = pd.DataFrame(chunk_rows)
    scheduler_df = pd.DataFrame(
        scheduler_rows
    )
    calibration_df = pd.DataFrame(
        calibration_rows
    )
    codec_df = pd.DataFrame(codec_rows)
    migration_df = pd.DataFrame(
        migration_rows
    )
    assignment_df = pd.DataFrame(
        assignment_rows
    )

    aggregate_df = aggregate_policy_runs(run_df)
    comparison_df = build_budget_comparisons(
        run_df
    )
    assessment = make_assessment(
        aggregate_df,
        comparison_df,
        base_budget_bytes,
    )

    dataset_meta = dict(dataset_meta)
    dataset_meta.update(
        {
            "dataset_label": dataset_label,
            "num_chunks": int(len(chunks)),
            "chunk_sizes": list(chunk_sizes),
            "max_chunks_per_size": int(
                max_chunks_per_size
            ),
            "trace_seeds": list(trace_seeds),
            "global_trace_length": int(
                trace_length
            ),
            "workload_types": (
                dict(
                    Counter(
                        workload_types.values()
                    )
                )
            ),
            "policies": policies,
            "budget_interval_global_queries": int(
                BUDGET_INTERVAL_GLOBAL_QUERIES
            ),
            "base_budget_bytes": int(
                base_budget_bytes
            ),
            "budget_multipliers": list(
                budget_multipliers
            ),
            "primary_execution_mode": (
                "warm_slicedmeta"
            ),
            "target_venue": TARGET_VENUE,
        }
    )

    calibration_df.to_csv(
        output_dir
        / "calibration_records.csv",
        index=False,
    )
    codec_df.to_csv(
        output_dir / "codec_records.csv",
        index=False,
    )
    migration_df.to_csv(
        output_dir
        / "measured_migration_costs.csv",
        index=False,
    )
    assignment_df.to_csv(
        output_dir
        / "chunk_workload_assignments.csv",
        index=False,
    )
    run_df.to_csv(
        output_dir
        / "policy_run_records.csv",
        index=False,
    )
    chunk_df.to_csv(
        output_dir
        / "chunk_policy_summary.csv",
        index=False,
    )
    scheduler_df.to_csv(
        output_dir
        / "scheduler_events.csv",
        index=False,
    )
    aggregate_df.to_csv(
        output_dir
        / "policy_aggregate_summary.csv",
        index=False,
    )
    comparison_df.to_csv(
        output_dir
        / "budget_comparison_summary.csv",
        index=False,
    )
    write_json(
        output_dir / "dataset_summary.json",
        dataset_meta,
    )
    write_json(
        output_dir / "e4_assessment.json",
        assessment,
    )

    config = {
        "version": VERSION,
        "target_venue": TARGET_VENUE,
        "base_e1_version": BASE_E1_VERSION,
        "seed": SEED,
        "trace_seeds": list(trace_seeds),
        "chunk_sizes": list(chunk_sizes),
        "max_series": (
            4 if QUICK_TEST else MAX_SERIES
        ),
        "max_chunks_per_size": int(
            max_chunks_per_size
        ),
        "global_trace_length": int(
            trace_length
        ),
        "decision_interval_local_queries": int(
            DECISION_INTERVAL
        ),
        "long_window_local_queries": int(
            LONG_WINDOW
        ),
        "short_window_local_queries": int(
            SHORT_WINDOW
        ),
        "minimum_observations": int(
            MIN_OBSERVATIONS
        ),
        "confirmation_rounds": int(
            CONFIRMATION_ROUNDS
        ),
        "cooldown_local_queries": int(
            COOLDOWN_QUERIES
        ),
        "prediction_horizon_local_queries": int(
            PREDICTION_HORIZON
        ),
        "minimum_relative_improvement": float(
            MIN_RELATIVE_IMPROVEMENT
        ),
        "rewrite_safety_factor": float(
            REWRITE_SAFETY_FACTOR
        ),
        "budget_interval_global_queries": int(
            BUDGET_INTERVAL_GLOBAL_QUERIES
        ),
        "base_budget_bytes": int(
            base_budget_bytes
        ),
        "budget_multipliers": list(
            budget_multipliers
        ),
        "migration_measure_repeats": int(
            migration_repeats
        ),
        "policies": policies,
        "representations": list(reps),
        "workloads": base.WORKLOADS,
        "measurement_scope": (
            "Real warm_slicedmeta global query "
            "replay; measured directed migration "
            "time and bytes are replayed when "
            "the active codec changes."
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "timer": "time.perf_counter_ns",
        },
    }
    write_json(
        output_dir / "run_config.json",
        config,
    )
    write_json(
        Path(
            "E4_QUICK_ASSESSMENT.json"
            if QUICK_TEST
            else SUMMARY_JSON_NAME
        ),
        assessment,
    )

    shutil.rmtree(
        storage_dir, ignore_errors=True
    )
    shutil.rmtree(
        migration_dir, ignore_errors=True
    )

    print("-" * 100)
    print(
        "Descriptive checks:",
        assessment["descriptive_checks"],
    )
    print(
        f"Results: {output_dir.resolve()}"
    )
    print(
        "Done. Upload e4_assessment.json, "
        "budget_comparison_summary.csv, "
        "policy_aggregate_summary.csv, "
        "policy_run_records.csv, "
        "chunk_policy_summary.csv, "
        "scheduler_events.csv, "
        "measured_migration_costs.csv, "
        "dataset_summary.json, and run_config.json."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
