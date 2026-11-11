#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
RePressDB S1 v1.3.1
Shape-EWMA-BE strong online baseline:
PEMS04 development selection, frozen PEMS08/BDG2 holdout evaluation,
and a pre-registered file-backed audit subset.

Target journal: Journal of Big Data.

DIRECT RUN
----------
Place this file in the RePressDB project root beside:

  RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_DEV_PEMS04_V0_7_0.py
  RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_DEADBAND_RECOVER_V0_7_3.py
  RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_V0_7_4.py
  RUN_THIS_ONLY_REPRESSDB_A2_PEMS08_BDG2_HOLDOUT_V0_8_0.py
  RUN_THIS_ONLY_REPRESSDB_E5_BDG2_FROZEN_DIRECT_V0_6_0.py
  RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT.py
  RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT_V0_2_0.py
  RUN_THIS_ONLY_REPRESSDB_E0C_V0_2_0.py

Keep the completed result directories:

  results_a0_a1_hotcold_pems04_v070/
  results_a0_a1_hotcold_deadband_pems04_v073/
  results_a0_a1_hotcold_deadband_pems04_v074/
  results_a2_hotcold_holdouts_v080/

Open this file in PyCharm and click Run.
No command-line arguments are required.

IMPLEMENTATION REPAIR IN v1.3.1
-------------------------------
Create the _patched_core directory before build_corrected_core writes the
generated module. No scientific parameter, trace, baseline, dataset, selection
rule, holdout rule, or acceptance threshold is changed.

SCIENTIFIC PURPOSE
------------------
Existing online competitors leave a gap:

  * periodic greedy uses query shape but is migration-cost unaware;
  * AF-HC-D is stable and cost-conscious operationally, but uses access
    frequency rather than query shape;
  * RePressDB combines query shape with two windows, agreement, confirmation,
    cooldown, and break-even safety.

Shape-EWMA-BE fills that gap with a deliberately simple strong baseline:

  1. one EWMA over the eight frozen query classes;
  2. the same measured codec/query-cost table available to RePressDB;
  3. the same measured directed migration-cost table;
  4. the same candidate representations and decision interval;
  5. migration only when:
       - another codec has minimum expected query cost;
       - relative improvement exceeds one deadband;
       - horizon-predicted saving exceeds estimated migration cost;
  6. no dual-window agreement;
  7. no multi-round confirmation;
  8. no cooldown.

PARAMETER SELECTION
-------------------
Only PEMS04 development traces are used.

Grid:
  alpha    in {0.01, 0.02, 0.05, 0.10, 0.20}
  deadband in {0.00, 0.01, 0.02, 0.05}
  horizon  in {300, 600, 1200, 2400}

80 configurations.

Primary action score:
  geometric mean of Shape-EWMA-BE / initial-best total-cost ratios on
  shape_only and joint_orthogonal traces, both chunk sizes, all three seeds.

Safety constraints:
  median ratio <= 1.02 on brief_burst;
  median ratio <= 1.02 on stable_scan.

Tie rule:
  among configurations within 0.5% of the best action score, prefer:
    fewer action migrations;
    fewer action rewrite bytes;
    larger deadband;
    longer horizon;
    smaller alpha;
    lexical configuration id.

No PEMS08, BDG2, or future R1 result may influence selection.

HOLDOUTS
--------
The selected configuration is frozen and evaluated without retuning on the
existing A2 PEMS08 and BDG2 holdouts, using the exact frozen A2 chunks,
assignments, measured query-cost profiles, and directed migration costs.

The full holdout is model-backed by previously measured file costs.
A pre-registered actual file-backed audit subset runs:

  datasets: PEMS08 and BDG2
  chunk sizes: 8192 and 16384
  trace seed: 20260718
  traces: shape_only, brief_burst, stable_scan
  policy: frozen Shape-EWMA-BE

This audit checks query truth and measures physical decode/encode/fsync
migration time. It is a model-validation subset, not a second tuning stage.

RESUME
------
Model stages are deterministic and inexpensive relative to calibration already
completed in A0/A2. File-backed audit runs are checkpointed individually.
Rerun the same file to resume.

Output directory:

  results_s1_shape_ewma_be_v130/
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
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


VERSION = "1.3.1"
TARGET_JOURNAL = "Journal of Big Data"
SEED = 20260727

V070_SCRIPT_NAME = "RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_DEV_PEMS04_V0_7_0.py"
V073_SCRIPT_NAME = "RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_DEADBAND_RECOVER_V0_7_3.py"
V074_SCRIPT_NAME = "RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_V0_7_4.py"
A2_SCRIPT_NAME = "RUN_THIS_ONLY_REPRESSDB_A2_PEMS08_BDG2_HOLDOUT_V0_8_0.py"
E5_SCRIPT_NAME = "RUN_THIS_ONLY_REPRESSDB_E5_BDG2_FROZEN_DIRECT_V0_6_0.py"

V070_RESULT_DIR_NAME = "results_a0_a1_hotcold_pems04_v070"
V073_RESULT_DIR_NAME = "results_a0_a1_hotcold_deadband_pems04_v073"
V074_RESULT_DIR_NAME = "results_a0_a1_hotcold_deadband_pems04_v074"
A2_RESULT_DIR_NAME = "results_a2_hotcold_holdouts_v080"

OUTPUT_DIR_NAME = "results_s1_shape_ewma_be_v130"
SUMMARY_JSON_NAME = "S1_SHAPE_EWMA_BE_ASSESSMENT_v130.json"

ALPHAS = (0.01, 0.02, 0.05, 0.10, 0.20)
DEADBANDS = (0.00, 0.01, 0.02, 0.05)
HORIZONS = (300, 600, 1200, 2400)

DECISION_INTERVAL = 100
MIN_OBSERVATIONS = 200
TIE_SCORE_TOLERANCE = 0.005
MAX_BRIEF_MEDIAN_RATIO = 1.02
MAX_STABLE_MEDIAN_RATIO = 1.02

ACTION_TRACES = ("shape_only", "joint_orthogonal")
SAFETY_TRACES = ("brief_burst", "stable_scan")
HOLDOUT_TRACE_NAMES = (
    "shape_only",
    "frequency_only",
    "joint_orthogonal",
    "brief_burst",
    "stable_scan",
)
TRACE_SEEDS = (20260718, 20260719, 20260720)
CHUNK_SIZES = (8192, 16384)

ACTUAL_AUDIT_TRACE_SEED = 20260718
ACTUAL_AUDIT_TRACES = (
    "shape_only",
    "brief_burst",
    "stable_scan",
)

BOOTSTRAP_REPLICATES = 5000


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
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_dataframe_atomic(path: Path, frame: pd.DataFrame) -> None:
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
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def load_module(path: Path, module_name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load Python module: {path}")
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


def bootstrap_median_ci(
    values: Sequence[float],
    seed_parts: Sequence[Any],
) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(
        stable_seed(SEED, "bootstrap", *seed_parts)
    )
    indices = rng.integers(
        0,
        array.size,
        size=(BOOTSTRAP_REPLICATES, array.size),
    )
    estimates = np.median(array[indices], axis=1)
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def parse_migration_maps(
    migration_frame: pd.DataFrame,
) -> Tuple[
    Dict[str, Dict[Tuple[str, str], int]],
    Dict[str, Dict[Tuple[str, str], int]],
]:
    costs: Dict[str, Dict[Tuple[str, str], int]] = {}
    rewrite: Dict[str, Dict[Tuple[str, str], int]] = {}
    for chunk_id, group in migration_frame.groupby("chunk_id"):
        local_cost: Dict[Tuple[str, str], int] = {}
        local_bytes: Dict[Tuple[str, str], int] = {}
        for row in group.itertuples(index=False):
            key = (str(row.current_codec), str(row.target_codec))
            local_cost[key] = int(row.median_migration_ns)
            local_bytes[key] = int(row.rewrite_bytes)
        costs[str(chunk_id)] = local_cost
        rewrite[str(chunk_id)] = local_bytes
    return costs, rewrite


def load_profiles(
    calibration_frame: pd.DataFrame,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    output: Dict[str, Dict[str, Dict[str, float]]] = {}
    for chunk_id, chunk_group in calibration_frame.groupby("chunk_id"):
        local: Dict[str, Dict[str, float]] = {}
        for codec, codec_group in chunk_group.groupby("codec"):
            local[str(codec)] = {
                str(row.query_class): float(row.median_latency_ns)
                for row in codec_group.itertuples(index=False)
            }
        output[str(chunk_id)] = local
    return output


def load_encode_decode_maps(
    codec_frame: pd.DataFrame,
) -> Tuple[
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, int]],
]:
    encode: Dict[str, Dict[str, int]] = {}
    decode: Dict[str, Dict[str, int]] = {}
    for chunk_id, group in codec_frame.groupby("chunk_id"):
        encode[str(chunk_id)] = {
            str(row.codec): int(row.encode_ns)
            for row in group.itertuples(index=False)
        }
        decode[str(chunk_id)] = {
            str(row.codec): int(row.full_decode_ns)
            for row in group.itertuples(index=False)
        }
    return encode, decode


def configuration_grid() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    config_index = 0
    for alpha in ALPHAS:
        for deadband in DEADBANDS:
            for horizon in HORIZONS:
                rows.append(
                    {
                        "config_id": f"EWMA_{config_index:03d}",
                        "alpha": float(alpha),
                        "deadband": float(deadband),
                        "prediction_horizon": int(horizon),
                        "decision_interval": DECISION_INTERVAL,
                        "min_observations": MIN_OBSERVATIONS,
                        "break_even_factor": 1.0,
                    }
                )
                config_index += 1
    return rows


def normalized_initial_weights(
    base: Any,
    query_classes: Sequence[str],
) -> np.ndarray:
    weights = np.asarray(
        [
            float(base.WORKLOADS["scan_heavy"].get(query_class, 0.0))
            for query_class in query_classes
        ],
        dtype=np.float64,
    )
    total = float(weights.sum())
    if total <= 0:
        raise AssertionError("scan_heavy has no query-class mass")
    return weights / total


def profile_array(
    profiles: Mapping[str, Mapping[str, Mapping[str, float]]],
    chunk_ids: Sequence[str],
    codecs: Sequence[str],
    query_classes: Sequence[str],
) -> np.ndarray:
    return np.asarray(
        [
            [
                [
                    float(profiles[chunk_id][codec][query_class])
                    for query_class in query_classes
                ]
                for codec in codecs
            ]
            for chunk_id in chunk_ids
        ],
        dtype=np.float64,
    )


def migration_matrix(
    migration_costs: Mapping[str, Mapping[Tuple[str, str], int]],
    migration_bytes: Mapping[str, Mapping[Tuple[str, str], int]],
    chunk_ids: Sequence[str],
    codecs: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    n_chunks = len(chunk_ids)
    n_codecs = len(codecs)
    costs = np.zeros(
        (n_chunks, n_codecs, n_codecs),
        dtype=np.float64,
    )
    rewrite = np.zeros(
        (n_chunks, n_codecs, n_codecs),
        dtype=np.int64,
    )
    for chunk_position, chunk_id in enumerate(chunk_ids):
        for source_position, source in enumerate(codecs):
            for target_position, target in enumerate(codecs):
                if source == target:
                    continue
                key = (source, target)
                costs[
                    chunk_position,
                    source_position,
                    target_position,
                ] = float(migration_costs[chunk_id][key])
                rewrite[
                    chunk_position,
                    source_position,
                    target_position,
                ] = int(migration_bytes[chunk_id][key])
    return costs, rewrite


def scan_best_positions(
    e1: Any,
    base: Any,
    profiles: Mapping[str, Mapping[str, Mapping[str, float]]],
    chunk_ids: Sequence[str],
    codecs: Sequence[str],
) -> np.ndarray:
    codec_to_position = {
        codec: position for position, codec in enumerate(codecs)
    }
    positions = []
    for chunk_id in chunk_ids:
        codec, _ = e1.choose_best_codec(
            profiles[chunk_id],
            base.WORKLOADS["scan_heavy"],
        )
        positions.append(codec_to_position[codec])
    return np.asarray(positions, dtype=np.int16)


def simulate_initial_best_model(
    plan: Any,
    profile_values: np.ndarray,
    initial_positions: np.ndarray,
) -> Dict[str, Any]:
    per_chunk = np.zeros(len(plan.chunk_ids), dtype=np.float64)
    for event_index in range(plan.n_events):
        chunk_position = int(plan.chunk_positions[event_index])
        query_position = int(
            plan.query_class_positions[event_index]
        )
        per_chunk[chunk_position] += profile_values[
            chunk_position,
            int(initial_positions[chunk_position]),
            query_position,
        ]
    total = float(per_chunk.sum())
    return {
        "query_time_ns": total,
        "migration_time_ns": 0.0,
        "total_time_ns": total,
        "migration_count": 0,
        "rewrite_bytes": 0,
        "per_chunk_query_time_ns": per_chunk,
        "per_chunk_migration_time_ns": np.zeros_like(per_chunk),
        "per_chunk_migration_count": np.zeros(
            len(plan.chunk_ids), dtype=np.int64
        ),
        "per_chunk_rewrite_bytes": np.zeros(
            len(plan.chunk_ids), dtype=np.int64
        ),
    }


def simulate_shape_ewma_grid(
    plan: Any,
    configurations: Sequence[Mapping[str, Any]],
    profile_values: np.ndarray,
    initial_positions: np.ndarray,
    migration_cost_values: np.ndarray,
    migration_byte_values: np.ndarray,
    initial_ewma: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Simulate all 80 configurations simultaneously.

    No global write budget is introduced in S1 because the existing A0/A2
    controller and baseline comparisons were conducted without E4's separate
    global scheduler. R1 later applies one identical external budget to both
    online policies.
    """
    n_configs = len(configurations)
    n_chunks = len(plan.chunk_ids)
    n_classes = profile_values.shape[2]

    alpha = np.asarray(
        [float(config["alpha"]) for config in configurations],
        dtype=np.float64,
    )
    deadband = np.asarray(
        [float(config["deadband"]) for config in configurations],
        dtype=np.float64,
    )
    horizon = np.asarray(
        [
            float(config["prediction_horizon"])
            for config in configurations
        ],
        dtype=np.float64,
    )

    active = np.repeat(
        initial_positions[np.newaxis, :],
        n_configs,
        axis=0,
    ).astype(np.int16, copy=True)
    ewma = np.broadcast_to(
        initial_ewma,
        (n_configs, n_chunks, n_classes),
    ).copy()
    local_counts = np.zeros(n_chunks, dtype=np.int32)

    query_cost = np.zeros(n_configs, dtype=np.float64)
    migration_cost = np.zeros(n_configs, dtype=np.float64)
    migration_count = np.zeros(n_configs, dtype=np.int64)
    rewrite_bytes = np.zeros(n_configs, dtype=np.int64)

    config_positions = np.arange(n_configs, dtype=np.int64)

    for event_index in range(plan.n_events):
        chunk_position = int(plan.chunk_positions[event_index])
        query_position = int(
            plan.query_class_positions[event_index]
        )

        active_local = active[:, chunk_position]
        query_cost += profile_values[
            chunk_position,
            active_local,
            query_position,
        ]

        ewma[:, chunk_position, :] *= (
            1.0 - alpha[:, np.newaxis]
        )
        ewma[:, chunk_position, query_position] += alpha

        local_counts[chunk_position] += 1
        local_count = int(local_counts[chunk_position])
        if (
            local_count < MIN_OBSERVATIONS
            or local_count % DECISION_INTERVAL != 0
        ):
            continue

        expected = np.einsum(
            "cq,kq->kc",
            profile_values[chunk_position],
            ewma[:, chunk_position, :],
            optimize=True,
        )
        target = np.argmin(expected, axis=1).astype(np.int16)
        current = active[:, chunk_position]
        current_cost = expected[config_positions, current]
        target_cost = expected[config_positions, target]
        improvement = current_cost - target_cost
        relative = np.divide(
            improvement,
            current_cost,
            out=np.zeros_like(improvement),
            where=current_cost > 0,
        )
        estimated_migration = migration_cost_values[
            chunk_position,
            current,
            target,
        ]
        should_switch = (
            (target != current)
            & (relative >= deadband)
            & (improvement * horizon > estimated_migration)
        )
        switching = np.flatnonzero(should_switch)
        if switching.size == 0:
            continue

        old_positions = current[switching]
        new_positions = target[switching]
        migration_cost[switching] += migration_cost_values[
            chunk_position,
            old_positions,
            new_positions,
        ]
        migration_count[switching] += 1
        rewrite_bytes[switching] += migration_byte_values[
            chunk_position,
            old_positions,
            new_positions,
        ]
        active[switching, chunk_position] = new_positions

    return {
        "query_time_ns": query_cost,
        "migration_time_ns": migration_cost,
        "total_time_ns": query_cost + migration_cost,
        "migration_count": migration_count,
        "rewrite_bytes": rewrite_bytes,
    }


@dataclasses.dataclass
class EWMAChunkState:
    active_position: int
    ewma: np.ndarray
    local_access_count: int = 0
    query_cost_ns: float = 0.0
    migration_cost_ns: float = 0.0
    migration_count: int = 0
    rewrite_bytes: int = 0
    migration_records: List[Dict[str, Any]] = dataclasses.field(
        default_factory=list
    )
    decision_records: List[Dict[str, Any]] = dataclasses.field(
        default_factory=list
    )
    open_episode: Optional[Dict[str, Any]] = None


def simulate_shape_ewma_detailed(
    plan: Any,
    configuration: Mapping[str, Any],
    profile_values: np.ndarray,
    initial_positions: np.ndarray,
    migration_cost_values: np.ndarray,
    migration_byte_values: np.ndarray,
    initial_ewma: np.ndarray,
    codecs: Sequence[str],
    query_classes: Sequence[str],
) -> Dict[str, Any]:
    alpha = float(configuration["alpha"])
    deadband = float(configuration["deadband"])
    horizon = int(configuration["prediction_horizon"])

    states = [
        EWMAChunkState(
            active_position=int(initial_positions[position]),
            ewma=np.asarray(initial_ewma, dtype=np.float64).copy(),
        )
        for position in range(len(plan.chunk_ids))
    ]

    global_decision_rows: List[Dict[str, Any]] = []
    cumulative_rows: List[Dict[str, Any]] = []
    cumulative_query = 0.0
    cumulative_migration = 0.0

    for event_index in range(plan.n_events):
        chunk_position = int(plan.chunk_positions[event_index])
        query_position = int(
            plan.query_class_positions[event_index]
        )
        state = states[chunk_position]
        old_active = state.active_position

        query_ns = float(
            profile_values[
                chunk_position,
                old_active,
                query_position,
            ]
        )
        state.query_cost_ns += query_ns
        cumulative_query += query_ns

        if state.open_episode is not None:
            previous_position = int(
                state.open_episode["source_position"]
            )
            current_position = int(
                state.open_episode["target_position"]
            )
            state.open_episode[
                "realized_query_saving_ns"
            ] += float(
                profile_values[
                    chunk_position,
                    previous_position,
                    query_position,
                ]
                - profile_values[
                    chunk_position,
                    current_position,
                    query_position,
                ]
            )

        state.ewma *= 1.0 - alpha
        state.ewma[query_position] += alpha
        state.local_access_count += 1

        if (
            state.local_access_count >= MIN_OBSERVATIONS
            and state.local_access_count % DECISION_INTERVAL == 0
        ):
            expected = profile_values[chunk_position] @ state.ewma
            target = int(np.argmin(expected))
            current = int(state.active_position)
            current_cost = float(expected[current])
            target_cost = float(expected[target])
            improvement = current_cost - target_cost
            relative = (
                improvement / current_cost
                if current_cost > 0
                else 0.0
            )
            migration_estimate = float(
                migration_cost_values[
                    chunk_position,
                    current,
                    target,
                ]
            )
            predicted_saving = improvement * horizon

            reason = "current_layout_is_best"
            switched = False
            if target != current:
                if relative < deadband:
                    reason = "deadband"
                elif predicted_saving <= migration_estimate:
                    reason = "break_even_not_met"
                else:
                    reason = "switch"
                    switched = True

            decision_row = {
                "global_event_index": int(event_index),
                "chunk_id": str(plan.chunk_ids[chunk_position]),
                "chunk_position": int(chunk_position),
                "local_query_index": int(
                    state.local_access_count - 1
                ),
                "old_codec": str(codecs[current]),
                "candidate_codec": str(codecs[target]),
                "estimated_best_codec": str(codecs[target]),
                "decision": "switch" if switched else "stay",
                "reason": reason,
                "current_expected_ns": current_cost,
                "target_expected_ns": target_cost,
                "relative_improvement": relative,
                "predicted_saving_ns": predicted_saving,
                "migration_estimate_ns": migration_estimate,
            }
            state.decision_records.append(decision_row)
            global_decision_rows.append(decision_row)

            if switched:
                if state.open_episode is not None:
                    state.open_episode["end_global_event_index"] = int(
                        event_index
                    )
                    state.open_episode["unamortized"] = bool(
                        float(
                            state.open_episode[
                                "realized_query_saving_ns"
                            ]
                        )
                        < float(
                            state.open_episode[
                                "migration_cost_ns"
                            ]
                        )
                    )
                    state.migration_records.append(
                        state.open_episode
                    )

                migration_ns = float(
                    migration_cost_values[
                        chunk_position,
                        current,
                        target,
                    ]
                )
                rewritten = int(
                    migration_byte_values[
                        chunk_position,
                        current,
                        target,
                    ]
                )
                state.migration_cost_ns += migration_ns
                state.migration_count += 1
                state.rewrite_bytes += rewritten
                cumulative_migration += migration_ns
                state.active_position = target
                state.open_episode = {
                    "chunk_id": str(
                        plan.chunk_ids[chunk_position]
                    ),
                    "start_global_event_index": int(
                        event_index
                    ),
                    "start_local_query_index": int(
                        state.local_access_count - 1
                    ),
                    "source_position": int(current),
                    "target_position": int(target),
                    "source_codec": str(codecs[current]),
                    "target_codec": str(codecs[target]),
                    "migration_cost_ns": migration_ns,
                    "rewrite_bytes": rewritten,
                    "realized_query_saving_ns": 0.0,
                    "end_global_event_index": None,
                    "unamortized": None,
                }

        if (
            event_index % 5000 == 0
            or event_index == plan.n_events - 1
        ):
            cumulative_rows.append(
                {
                    "global_event_index": int(event_index),
                    "cumulative_query_time_ns": float(
                        cumulative_query
                    ),
                    "cumulative_migration_time_ns": float(
                        cumulative_migration
                    ),
                    "cumulative_total_time_ns": float(
                        cumulative_query + cumulative_migration
                    ),
                    "cumulative_migrations": int(
                        sum(
                            item.migration_count
                            for item in states
                        )
                    ),
                }
            )

    all_migrations: List[Dict[str, Any]] = []
    all_decisions: List[Dict[str, Any]] = []
    for state in states:
        if state.open_episode is not None:
            state.open_episode[
                "end_global_event_index"
            ] = int(plan.n_events)
            state.open_episode["unamortized"] = bool(
                float(
                    state.open_episode[
                        "realized_query_saving_ns"
                    ]
                )
                < float(
                    state.open_episode[
                        "migration_cost_ns"
                    ]
                )
            )
            state.migration_records.append(
                state.open_episode
            )
            state.open_episode = None
        all_migrations.extend(state.migration_records)
        all_decisions.extend(state.decision_records)

    # Pre-registered stale-switch diagnostic:
    # inspect the first two later decision points for the same chunk.
    decisions_by_chunk: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_decisions:
        decisions_by_chunk.setdefault(
            str(row["chunk_id"]),
            [],
        ).append(row)
    for migration in all_migrations:
        later = [
            row
            for row in decisions_by_chunk[
                str(migration["chunk_id"])
            ]
            if int(row["global_event_index"])
            > int(
                migration[
                    "start_global_event_index"
                ]
            )
        ][:2]
        migration["stale_assessable"] = (
            len(later) == 2
        )
        migration["stale"] = bool(
            len(later) == 2
            and all(
                str(row["estimated_best_codec"])
                != str(
                    migration["target_codec"]
                )
                for row in later
            )
        )

    per_query = np.asarray(
        [state.query_cost_ns for state in states],
        dtype=np.float64,
    )
    per_migration = np.asarray(
        [state.migration_cost_ns for state in states],
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
    total_query = float(per_query.sum())
    total_migration = float(per_migration.sum())

    unamortized = int(
        sum(bool(row["unamortized"]) for row in all_migrations)
    )
    stale_assessable = int(
        sum(
            bool(row["stale_assessable"])
            for row in all_migrations
        )
    )
    stale = int(
        sum(bool(row["stale"]) for row in all_migrations)
    )

    return {
        "query_time_ns": total_query,
        "migration_time_ns": total_migration,
        "total_time_ns": total_query + total_migration,
        "migration_count": int(per_count.sum()),
        "rewrite_bytes": int(per_rewrite.sum()),
        "unamortized_migration_count": unamortized,
        "unamortized_migration_rate": (
            unamortized / len(all_migrations)
            if all_migrations
            else 0.0
        ),
        "stale_assessable_migration_count": stale_assessable,
        "stale_migration_count": stale,
        "stale_migration_rate": (
            stale / stale_assessable
            if stale_assessable
            else 0.0
        ),
        "per_chunk_query_time_ns": per_query,
        "per_chunk_migration_time_ns": per_migration,
        "per_chunk_migration_count": per_count,
        "per_chunk_rewrite_bytes": per_rewrite,
        "migration_records": all_migrations,
        "decision_records": global_decision_rows,
        "cumulative_records": cumulative_rows,
    }


def select_configuration(
    all_results: pd.DataFrame,
) -> Tuple[
    Dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    rows: List[Dict[str, Any]] = []
    for config_id, group in all_results.groupby("config_id"):
        action = group[
            group["trace"].isin(ACTION_TRACES)
        ]
        brief = group[
            group["trace"] == "brief_burst"
        ]
        stable = group[
            group["trace"] == "stable_scan"
        ]
        first = group.iloc[0]
        rows.append(
            {
                "config_id": str(config_id),
                "alpha": float(first["alpha"]),
                "deadband": float(first["deadband"]),
                "prediction_horizon": int(
                    first["prediction_horizon"]
                ),
                "action_geometric_mean_ratio": geometric_mean(
                    action["ratio_to_initial_best"]
                ),
                "action_median_ratio": float(
                    action[
                        "ratio_to_initial_best"
                    ].median()
                ),
                "action_migration_count": int(
                    action["migration_count"].sum()
                ),
                "action_rewrite_bytes": int(
                    action["rewrite_bytes"].sum()
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
                "brief_max_ratio": float(
                    brief[
                        "ratio_to_initial_best"
                    ].max()
                ),
                "stable_max_ratio": float(
                    stable[
                        "ratio_to_initial_best"
                    ].max()
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary["brief_safety_pass"] = (
        summary["brief_median_ratio"]
        <= MAX_BRIEF_MEDIAN_RATIO
    )
    summary["stable_safety_pass"] = (
        summary["stable_median_ratio"]
        <= MAX_STABLE_MEDIAN_RATIO
    )
    summary["safe"] = (
        summary["brief_safety_pass"]
        & summary["stable_safety_pass"]
    )

    safe = summary[summary["safe"]].copy()
    if safe.empty:
        raise RuntimeError(
            "No Shape-EWMA-BE configuration satisfies "
            "the pre-registered brief/stable safety constraints."
        )
    best_score = float(
        safe["action_geometric_mean_ratio"].min()
    )
    equivalent = safe[
        safe["action_geometric_mean_ratio"]
        <= best_score * (1.0 + TIE_SCORE_TOLERANCE)
    ].copy()
    equivalent = equivalent.sort_values(
        [
            "action_migration_count",
            "action_rewrite_bytes",
            "deadband",
            "prediction_horizon",
            "alpha",
            "config_id",
        ],
        ascending=[
            True,
            True,
            False,
            False,
            True,
            True,
        ],
    )
    selected_row = equivalent.iloc[0].to_dict()
    selected_row["best_action_score"] = best_score
    selected_row["tie_tolerance"] = TIE_SCORE_TOLERANCE
    selected_row["equivalent_config_count"] = int(
        len(equivalent)
    )
    ranked = summary.sort_values(
        [
            "safe",
            "action_geometric_mean_ratio",
            "action_migration_count",
            "action_rewrite_bytes",
        ],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    ranked["selection_rank"] = np.arange(
        1, len(ranked) + 1
    )
    return selected_row, summary, ranked


def load_pems04_context(
    project_dir: Path,
    core: Any,
    e1: Any,
    base: Any,
    v070_dir: Path,
    a2: Any,
) -> Dict[str, Any]:
    manifest = pd.read_csv(
        v070_dir / "pems04_selected_chunks.csv"
    )
    selected_indices = sorted(
        manifest["original_series_index"]
        .astype(int)
        .unique()
        .tolist()
    )

    pems04_dir, _ = base.default_data_dirs()
    data_file = base.auto_detect_data_file(
        pems04_dir,
        "PEMS04_SHAPE_EWMA_DEVELOPMENT",
    )
    matrix, source_meta = base.load_dataset(
        data_file,
        feature_index=0,
        max_series=len(selected_indices),
        seed=core.SEED,
        excluded_series=None,
    )
    if list(
        source_meta["selected_original_series_indices"]
    ) != selected_indices:
        raise AssertionError(
            "PEMS04 selected series differ from frozen A0 manifest"
        )
    chunks, rebuilt = core.deterministic_series_chunks(
        base,
        matrix,
        "PEMS04",
        selected_indices,
        chunk_sizes=CHUNK_SIZES,
    )
    compare_columns = [
        "dataset",
        "chunk_id",
        "original_series_index",
        "local_series_index",
        "chunk_size",
        "chunk_start",
        "chunk_end_exclusive",
        "raw_bytes",
        "selection_seed",
    ]
    expected = manifest[
        compare_columns
    ].sort_values(compare_columns).reset_index(drop=True)
    observed = rebuilt[
        compare_columns
    ].sort_values(compare_columns).reset_index(drop=True)
    if not expected.equals(observed):
        raise AssertionError(
            "Rebuilt PEMS04 chunks differ from frozen A0 manifest"
        )

    calibration = pd.read_csv(
        v070_dir / "calibration_records.csv"
    )
    codec = pd.read_csv(
        v070_dir / "codec_records.csv"
    )
    migration = pd.read_csv(
        v070_dir / "measured_migration_costs.csv"
    )
    profiles = load_profiles(calibration)
    encode, decode = load_encode_decode_maps(codec)
    migration_costs, migration_bytes = (
        parse_migration_maps(migration)
    )
    query_pools = a2.build_query_pools(
        base,
        e1,
        core,
        chunks,
    )
    assignments = pd.read_csv(
        v070_dir / "trace_assignment_manifest.csv"
    )
    assignments = assignments[
        assignments["dataset"] == "PEMS04"
    ].copy()

    return {
        "dataset": "PEMS04",
        "chunks": chunks,
        "profiles": profiles,
        "encode": encode,
        "decode": decode,
        "migration_costs": migration_costs,
        "migration_bytes": migration_bytes,
        "query_pools": query_pools,
        "assignments": assignments,
        "source_meta": source_meta,
        "calibration_frame": calibration,
        "codec_frame": codec,
        "migration_frame": migration,
    }


def load_holdout_contexts(
    project_dir: Path,
    core: Any,
    e1: Any,
    base: Any,
    a2: Any,
    e5: Any,
    v070_dir: Path,
    a2_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    pems08_manifest = pd.read_csv(
        v070_dir / "pems08_selected_chunks.csv"
    )
    pems08_chunks, _, pems08_metadata, pems08_source = (
        a2.load_pems08_chunks(
            core,
            base,
            pems08_manifest,
        )
    )

    bdg2_reserved = pd.read_csv(
        v070_dir / "bdg2_selected_buildings.csv"
    )
    bdg2_manifest = a2.build_bdg2_paired_manifest(
        bdg2_reserved
    )
    bdg2_chunks, _, bdg2_metadata, bdg2_source = (
        a2.load_bdg2_chunks(
            base,
            e5,
            bdg2_manifest,
        )
    )

    pems08_assignments = pd.read_csv(
        v070_dir / "trace_assignment_manifest.csv"
    )
    pems08_assignments = pems08_assignments[
        pems08_assignments["dataset"] == "PEMS08"
    ].copy()
    bdg2_assignments = core.build_trace_assignments(
        bdg2_manifest
    )

    contexts: Dict[str, Dict[str, Any]] = {}
    for dataset, chunks, metadata, source, assignments in (
        (
            "PEMS08",
            pems08_chunks,
            pems08_metadata,
            pems08_source,
            pems08_assignments,
        ),
        (
            "BDG2",
            bdg2_chunks,
            bdg2_metadata,
            bdg2_source,
            bdg2_assignments,
        ),
    ):
        dataset_dir = a2_dir / dataset.lower()
        calibration = pd.read_csv(
            dataset_dir / "calibration_records.csv"
        )
        codec = pd.read_csv(
            dataset_dir / "codec_records.csv"
        )
        migration = pd.read_csv(
            dataset_dir / "measured_migration_costs.csv"
        )
        contexts[dataset] = {
            "dataset": dataset,
            "chunks": chunks,
            "profiles": load_profiles(calibration),
            "encode": load_encode_decode_maps(codec)[0],
            "decode": load_encode_decode_maps(codec)[1],
            "migration_costs": parse_migration_maps(
                migration
            )[0],
            "migration_bytes": parse_migration_maps(
                migration
            )[1],
            "query_pools": a2.build_query_pools(
                base,
                e1,
                core,
                chunks,
            ),
            "assignments": assignments,
            "metadata": metadata,
            "source_meta": source,
            "calibration_frame": calibration,
            "codec_frame": codec,
            "migration_frame": migration,
            "candidate_storage_dir": (
                dataset_dir / "_candidate_storage"
            ),
        }
    return contexts


def plan_digest(plan: Any) -> str:
    digest = hashlib.sha256()
    for array in (
        plan.chunk_positions,
        plan.local_indices,
        plan.query_class_positions,
        plan.query_choice_positions,
        plan.workload_positions,
    ):
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_plans(
    context: Mapping[str, Any],
    core: Any,
    base: Any,
    trace_names: Sequence[str],
    trace_seeds: Sequence[int],
) -> Iterator[Tuple[int, int, str, Any]]:
    chunks = context["chunks"]
    for chunk_size in CHUNK_SIZES:
        size_chunks = sorted(
            [
                chunk
                for chunk in chunks
                if int(chunk.chunk_size)
                == int(chunk_size)
            ],
            key=lambda item: item.chunk_id,
        )
        size_assignments = context["assignments"][
            context["assignments"]["chunk_size"]
            == chunk_size
        ]
        for trace_seed in trace_seeds:
            for trace_name in trace_names:
                plan = core.build_trace_plan(
                    base,
                    size_chunks,
                    context["query_pools"],
                    size_assignments,
                    trace_name,
                    int(trace_seed),
                )
                yield (
                    chunk_size,
                    int(trace_seed),
                    trace_name,
                    plan,
                )


def run_development_grid(
    context: Mapping[str, Any],
    configurations: Sequence[Mapping[str, Any]],
    core: Any,
    e1: Any,
    e1v2: Any,
    base: Any,
    output_dir: Path,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Any],
]:
    cache_path = output_dir / "development_grid_all_results.csv"
    manifest_path = output_dir / "development_trace_manifest.csv"

    if cache_path.exists() and manifest_path.exists():
        all_results = pd.read_csv(cache_path)
        selected, summary, ranked = select_configuration(
            all_results
        )
        return all_results, summary, ranked, selected

    codecs = tuple(
        sorted(base.representation_registry(3))
    )
    query_classes = tuple(core.QUERY_CLASSES)
    initial_ewma = normalized_initial_weights(
        base,
        query_classes,
    )

    result_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []

    for (
        chunk_size,
        trace_seed,
        trace_name,
        plan,
    ) in build_plans(
        context,
        core,
        base,
        core.TRACE_NAMES,
        core.TRACE_SEEDS,
    ):
        chunk_ids = tuple(plan.chunk_ids)
        profiles = profile_array(
            context["profiles"],
            chunk_ids,
            codecs,
            query_classes,
        )
        initial_positions = scan_best_positions(
            e1,
            base,
            context["profiles"],
            chunk_ids,
            codecs,
        )
        migration_cost_values, migration_byte_values = (
            migration_matrix(
                context["migration_costs"],
                context["migration_bytes"],
                chunk_ids,
                codecs,
            )
        )
        initial = simulate_initial_best_model(
            plan,
            profiles,
            initial_positions,
        )
        grid = simulate_shape_ewma_grid(
            plan,
            configurations,
            profiles,
            initial_positions,
            migration_cost_values,
            migration_byte_values,
            initial_ewma,
        )
        for config_position, configuration in enumerate(
            configurations
        ):
            total = float(
                grid["total_time_ns"][config_position]
            )
            result_rows.append(
                {
                    **configuration,
                    "dataset": "PEMS04",
                    "trace": trace_name,
                    "trace_seed": trace_seed,
                    "chunk_size": chunk_size,
                    "n_chunks": len(chunk_ids),
                    "n_queries": int(plan.n_events),
                    "query_time_ns": float(
                        grid["query_time_ns"][
                            config_position
                        ]
                    ),
                    "migration_time_ns": float(
                        grid["migration_time_ns"][
                            config_position
                        ]
                    ),
                    "total_time_ns": total,
                    "initial_best_total_time_ns": float(
                        initial["total_time_ns"]
                    ),
                    "ratio_to_initial_best": float(
                        total
                        / initial["total_time_ns"]
                    ),
                    "migration_count": int(
                        grid["migration_count"][
                            config_position
                        ]
                    ),
                    "rewrite_bytes": int(
                        grid["rewrite_bytes"][
                            config_position
                        ]
                    ),
                }
            )
        manifest_rows.append(
            {
                "dataset": "PEMS04",
                "trace": trace_name,
                "trace_seed": trace_seed,
                "chunk_size": chunk_size,
                "n_chunks": len(chunk_ids),
                "n_queries": int(plan.n_events),
                "trace_sha256": plan_digest(plan),
            }
        )
        print(
            f"[S1 development] size={chunk_size}, "
            f"trace={trace_name}, seed={trace_seed}, "
            f"queries={plan.n_events}"
        )

    all_results = pd.DataFrame(result_rows)
    manifest = pd.DataFrame(manifest_rows)
    selected, summary, ranked = select_configuration(
        all_results
    )

    write_dataframe_atomic(cache_path, all_results)
    write_dataframe_atomic(manifest_path, manifest)
    write_dataframe_atomic(
        output_dir / "development_config_summary.csv",
        summary,
    )
    write_dataframe_atomic(
        output_dir / "development_config_ranked.csv",
        ranked,
    )
    return all_results, summary, ranked, selected


def evaluate_selected_on_context(
    context: Mapping[str, Any],
    selected_config: Mapping[str, Any],
    core: Any,
    e1: Any,
    e1v2: Any,
    base: Any,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    codecs = tuple(
        sorted(base.representation_registry(3))
    )
    query_classes = tuple(core.QUERY_CLASSES)
    initial_ewma = normalized_initial_weights(
        base,
        query_classes,
    )

    global_rows: List[Dict[str, Any]] = []
    chunk_rows: List[Dict[str, Any]] = []
    migration_rows: List[Dict[str, Any]] = []
    cumulative_rows: List[Dict[str, Any]] = []

    for (
        chunk_size,
        trace_seed,
        trace_name,
        plan,
    ) in build_plans(
        context,
        core,
        base,
        HOLDOUT_TRACE_NAMES,
        TRACE_SEEDS,
    ):
        chunk_ids = tuple(plan.chunk_ids)
        profiles = profile_array(
            context["profiles"],
            chunk_ids,
            codecs,
            query_classes,
        )
        initial_positions = scan_best_positions(
            e1,
            base,
            context["profiles"],
            chunk_ids,
            codecs,
        )
        migration_cost_values, migration_byte_values = (
            migration_matrix(
                context["migration_costs"],
                context["migration_bytes"],
                chunk_ids,
                codecs,
            )
        )

        initial = simulate_initial_best_model(
            plan,
            profiles,
            initial_positions,
        )
        repress = core.simulate_repressdb(
            e1,
            e1v2,
            base,
            plan,
            context["profiles"],
            context["encode"],
            context["decode"],
            context["migration_costs"],
            context["migration_bytes"],
        )
        shape = simulate_shape_ewma_detailed(
            plan,
            selected_config,
            profiles,
            initial_positions,
            migration_cost_values,
            migration_byte_values,
            initial_ewma,
            codecs,
            query_classes,
        )

        for policy, result in (
            ("initial_best", initial),
            ("shape_ewma_be", shape),
            ("repressdb", repress),
        ):
            global_rows.append(
                {
                    "dataset": str(context["dataset"]),
                    "trace": trace_name,
                    "trace_seed": trace_seed,
                    "chunk_size": chunk_size,
                    "policy": policy,
                    "n_chunks": len(chunk_ids),
                    "n_queries": int(plan.n_events),
                    "query_time_ns": float(
                        result["query_time_ns"]
                    ),
                    "migration_time_ns": float(
                        result["migration_time_ns"]
                    ),
                    "total_time_ns": float(
                        result["total_time_ns"]
                    ),
                    "migration_count": int(
                        result["migration_count"]
                    ),
                    "rewrite_bytes": int(
                        result["rewrite_bytes"]
                    ),
                    "unamortized_migration_count": int(
                        result.get(
                            "unamortized_migration_count",
                            0,
                        )
                    ),
                    "unamortized_migration_rate": float(
                        result.get(
                            "unamortized_migration_rate",
                            0.0,
                        )
                    ),
                    "stale_migration_count": int(
                        result.get(
                            "stale_migration_count",
                            0,
                        )
                    ),
                    "stale_migration_rate": float(
                        result.get(
                            "stale_migration_rate",
                            0.0,
                        )
                    ),
                }
            )
            for position, chunk_id in enumerate(
                chunk_ids
            ):
                metadata = context["metadata"].get(
                    chunk_id,
                    {
                        "unit_type": "series",
                        "unit_id": chunk_id,
                        "building_id": "",
                    },
                )
                chunk_rows.append(
                    {
                        "dataset": str(
                            context["dataset"]
                        ),
                        "trace": trace_name,
                        "trace_seed": trace_seed,
                        "chunk_size": chunk_size,
                        "policy": policy,
                        "chunk_id": chunk_id,
                        "unit_type": metadata.get(
                            "unit_type",
                            "series",
                        ),
                        "unit_id": metadata.get(
                            "unit_id",
                            chunk_id,
                        ),
                        "building_id": metadata.get(
                            "building_id",
                            "",
                        ),
                        "query_time_ns": float(
                            result[
                                "per_chunk_query_time_ns"
                            ][position]
                        ),
                        "migration_time_ns": float(
                            result[
                                "per_chunk_migration_time_ns"
                            ][position]
                        ),
                        "total_time_ns": float(
                            result[
                                "per_chunk_query_time_ns"
                            ][position]
                            + result[
                                "per_chunk_migration_time_ns"
                            ][position]
                        ),
                        "migration_count": int(
                            result[
                                "per_chunk_migration_count"
                            ][position]
                        ),
                        "rewrite_bytes": int(
                            result[
                                "per_chunk_rewrite_bytes"
                            ][position]
                        ),
                    }
                )

        for row in shape["migration_records"]:
            migration_rows.append(
                {
                    "dataset": str(context["dataset"]),
                    "trace": trace_name,
                    "trace_seed": trace_seed,
                    "chunk_size": chunk_size,
                    **row,
                }
            )
        for row in shape["cumulative_records"]:
            cumulative_rows.append(
                {
                    "dataset": str(context["dataset"]),
                    "trace": trace_name,
                    "trace_seed": trace_seed,
                    "chunk_size": chunk_size,
                    "policy": "shape_ewma_be",
                    **row,
                }
            )
        print(
            f"[S1 holdout model] dataset={context['dataset']}, "
            f"size={chunk_size}, trace={trace_name}, "
            f"seed={trace_seed}"
        )

    return (
        pd.DataFrame(global_rows),
        pd.DataFrame(chunk_rows),
        pd.DataFrame(migration_rows),
        pd.DataFrame(cumulative_rows),
    )


def reconstruct_stored_maps(
    base: Any,
    chunks: Sequence[Any],
    codec_frame: pd.DataFrame,
    storage_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    chunks_by_id = {
        chunk.chunk_id: chunk for chunk in chunks
    }
    output: Dict[str, Dict[str, Any]] = {}
    for chunk_id, group in codec_frame.groupby("chunk_id"):
        chunk_id = str(chunk_id)
        chunk = chunks_by_id[chunk_id]
        local: Dict[str, Any] = {}
        for row in group.itertuples(index=False):
            codec = str(row.codec)
            stem = storage_dir / chunk_id / codec
            data_path = stem.with_suffix(".bin")
            metadata_path_candidate = stem.with_suffix(
                ".meta.bin"
            )
            if not data_path.exists():
                raise FileNotFoundError(data_path)
            if metadata_path_candidate.exists():
                metadata_path: Optional[Path] = (
                    metadata_path_candidate
                )
                metadata = base.parse_block_metadata(
                    metadata_path.read_bytes()
                )
            else:
                metadata_path = None
                metadata = (
                    {"level": 3}
                    if codec == "whole_zstd"
                    else {}
                )
            local[codec] = base.StoredChunk(
                codec=codec,
                n=int(chunk.values.size),
                data_path=data_path,
                metadata_path=metadata_path,
                payload_size_bytes=int(
                    row.payload_size_bytes
                ),
                metadata_size_bytes=int(
                    row.metadata_size_bytes
                ),
                metadata=metadata,
            )
        output[chunk_id] = local
    return output


def actual_query_for_event(
    core: Any,
    plan: Any,
    event_index: int,
    query_pools: Mapping[str, Mapping[str, List[Any]]],
) -> Tuple[str, str, Any]:
    chunk_position = int(
        plan.chunk_positions[event_index]
    )
    chunk_id = str(plan.chunk_ids[chunk_position])
    query_class = str(
        core.QUERY_CLASSES[
            int(
                plan.query_class_positions[
                    event_index
                ]
            )
        ]
    )
    query_choice = int(
        plan.query_choice_positions[event_index]
    )
    return (
        chunk_id,
        query_class,
        query_pools[chunk_id][query_class][
            query_choice
        ],
    )


def run_actual_shape_ewma(
    core: Any,
    e1: Any,
    base: Any,
    reps: Mapping[str, Any],
    context: Mapping[str, Any],
    plan: Any,
    selected_config: Mapping[str, Any],
    stored_by_chunk: Mapping[str, Mapping[str, Any]],
    migration_root: Path,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    mode = base.CACHE_MODES["warm_slicedmeta"]
    codecs = tuple(sorted(reps))
    codec_to_position = {
        codec: position
        for position, codec in enumerate(codecs)
    }
    query_classes = tuple(core.QUERY_CLASSES)
    initial_ewma = normalized_initial_weights(
        base,
        query_classes,
    )
    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunk in context["chunks"]
        if chunk.chunk_id in set(plan.chunk_ids)
    }

    states: Dict[str, EWMAChunkState] = {}
    for chunk_id in plan.chunk_ids:
        initial_codec, _ = e1.choose_best_codec(
            context["profiles"][chunk_id],
            base.WORKLOADS["scan_heavy"],
        )
        states[chunk_id] = EWMAChunkState(
            active_position=codec_to_position[
                initial_codec
            ],
            ewma=initial_ewma.copy(),
        )

    alpha = float(selected_config["alpha"])
    deadband = float(
        selected_config["deadband"]
    )
    horizon = int(
        selected_config["prediction_horizon"]
    )
    per_chunk_latencies: Dict[str, List[int]] = {
        chunk_id: [] for chunk_id in plan.chunk_ids
    }
    event_rows: List[Dict[str, Any]] = []

    with core.MultiChunkHandles(
        stored_by_chunk
    ) as handles:
        first_event_by_chunk: Dict[str, int] = {}
        for event_index in range(plan.n_events):
            chunk_id, _, _ = actual_query_for_event(
                core,
                plan,
                event_index,
                context["query_pools"],
            )
            first_event_by_chunk.setdefault(
                chunk_id,
                event_index,
            )
            if len(first_event_by_chunk) == len(
                plan.chunk_ids
            ):
                break

        for chunk_id, event_index in (
            first_event_by_chunk.items()
        ):
            _, _, query = actual_query_for_event(
                core,
                plan,
                event_index,
                context["query_pools"],
            )
            truth = base.execute_on_array(
                chunks_by_id[chunk_id].values,
                query,
            )
            for codec, rep in reps.items():
                result, _ = base.run_query_once(
                    rep,
                    stored_by_chunk[chunk_id][codec],
                    query,
                    mode,
                    handles.payload[chunk_id][codec],
                    handles.metadata[chunk_id][codec],
                )
                if not base.results_equal(
                    result,
                    truth,
                ):
                    raise AssertionError(
                        f"Actual audit warm-up failed: "
                        f"{chunk_id}/{codec}"
                    )

        for event_index in range(plan.n_events):
            chunk_id, query_class, query = (
                actual_query_for_event(
                    core,
                    plan,
                    event_index,
                    context["query_pools"],
                )
            )
            state = states[chunk_id]
            codec = codecs[state.active_position]
            chunk = chunks_by_id[chunk_id]

            start_ns = time.perf_counter_ns()
            result, stats = base.run_query_once(
                reps[codec],
                stored_by_chunk[chunk_id][codec],
                query,
                mode,
                handles.payload[chunk_id][codec],
                handles.metadata[chunk_id][codec],
            )
            latency_ns = (
                time.perf_counter_ns()
                - start_ns
            )
            truth = base.execute_on_array(
                chunk.values,
                query,
            )
            if not base.results_equal(
                result,
                truth,
            ):
                raise AssertionError(
                    f"Actual audit query failed: "
                    f"{chunk_id}, event={event_index}"
                )
            state.query_cost_ns += int(latency_ns)
            per_chunk_latencies[chunk_id].append(
                int(latency_ns)
            )

            query_position = query_classes.index(
                query_class
            )
            state.ewma *= 1.0 - alpha
            state.ewma[query_position] += alpha
            state.local_access_count += 1

            if (
                state.local_access_count
                >= MIN_OBSERVATIONS
                and state.local_access_count
                % DECISION_INTERVAL
                == 0
            ):
                profile = context["profiles"][
                    chunk_id
                ]
                expected = np.asarray(
                    [
                        sum(
                            float(
                                profile[codec_name][
                                    query_classes[
                                        class_position
                                    ]
                                ]
                            )
                            * float(
                                state.ewma[
                                    class_position
                                ]
                            )
                            for class_position
                            in range(
                                len(query_classes)
                            )
                        )
                        for codec_name in codecs
                    ],
                    dtype=np.float64,
                )
                target_position = int(
                    np.argmin(expected)
                )
                current_position = int(
                    state.active_position
                )
                current_cost = float(
                    expected[current_position]
                )
                target_cost = float(
                    expected[target_position]
                )
                improvement = (
                    current_cost - target_cost
                )
                relative = (
                    improvement / current_cost
                    if current_cost > 0
                    else 0.0
                )
                target_codec = codecs[
                    target_position
                ]
                migration_estimate = (
                    float(
                        context["migration_costs"][
                            chunk_id
                        ][
                            (
                                codecs[
                                    current_position
                                ],
                                target_codec,
                            )
                        ]
                    )
                    if target_position
                    != current_position
                    else 0.0
                )
                should_switch = bool(
                    target_position
                    != current_position
                    and relative >= deadband
                    and improvement * horizon
                    > migration_estimate
                )
                if should_switch:
                    old_codec = codecs[
                        current_position
                    ]
                    (
                        migration_ns,
                        rewrite_bytes,
                    ) = core.execute_actual_migration(
                        e1,
                        base,
                        reps,
                        stored_by_chunk[chunk_id],
                        chunk,
                        old_codec,
                        target_codec,
                        migration_root / chunk_id,
                        (
                            f"{plan.trace_name}__"
                            f"seed{plan.trace_seed}__"
                            f"shape_ewma__"
                            f"event{event_index}__"
                            f"m{state.migration_count}"
                        ),
                    )
                    state.active_position = (
                        target_position
                    )
                    state.migration_cost_ns += int(
                        migration_ns
                    )
                    state.migration_count += 1
                    state.rewrite_bytes += int(
                        rewrite_bytes
                    )
                    event_rows.append(
                        {
                            "trace": plan.trace_name,
                            "trace_seed": int(
                                plan.trace_seed
                            ),
                            "chunk_size": int(
                                plan.chunk_size
                            ),
                            "global_event_index": int(
                                event_index
                            ),
                            "chunk_id": chunk_id,
                            "local_query_index": int(
                                state.local_access_count
                                - 1
                            ),
                            "old_codec": old_codec,
                            "target_codec": target_codec,
                            "reason": "switch",
                            "actual_migration_ns": int(
                                migration_ns
                            ),
                            "rewrite_bytes": int(
                                rewrite_bytes
                            ),
                            "estimated_migration_ns": (
                                migration_estimate
                            ),
                            "relative_improvement": (
                                relative
                            ),
                        }
                    )

            if (
                event_index % 5000 == 0
                or event_index
                == plan.n_events - 1
            ):
                event_rows.append(
                    {
                        "trace": plan.trace_name,
                        "trace_seed": int(
                            plan.trace_seed
                        ),
                        "chunk_size": int(
                            plan.chunk_size
                        ),
                        "global_event_index": int(
                            event_index
                        ),
                        "chunk_id": chunk_id,
                        "local_query_index": int(
                            state.local_access_count
                            - 1
                        ),
                        "old_codec": codec,
                        "target_codec": codecs[
                            state.active_position
                        ],
                        "reason": "query_checkpoint",
                        "actual_migration_ns": 0,
                        "rewrite_bytes": 0,
                        "query_class": query_class,
                        "query_latency_ns": int(
                            latency_ns
                        ),
                        "payload_bytes": int(
                            stats.payload_bytes
                        ),
                        "metadata_bytes": int(
                            stats.metadata_bytes
                        ),
                    }
                )

    chunk_rows: List[Dict[str, Any]] = []
    all_latencies: List[int] = []
    for chunk_id in plan.chunk_ids:
        state = states[chunk_id]
        latencies = np.asarray(
            per_chunk_latencies[chunk_id],
            dtype=np.int64,
        )
        all_latencies.extend(
            per_chunk_latencies[chunk_id]
        )
        metadata = context["metadata"].get(
            chunk_id,
            {},
        )
        chunk_rows.append(
            {
                "dataset": context["dataset"],
                "trace": plan.trace_name,
                "trace_seed": int(plan.trace_seed),
                "chunk_size": int(plan.chunk_size),
                "policy": "shape_ewma_be",
                "chunk_id": chunk_id,
                "unit_type": metadata.get(
                    "unit_type",
                    "series",
                ),
                "unit_id": metadata.get(
                    "unit_id",
                    chunk_id,
                ),
                "building_id": metadata.get(
                    "building_id",
                    "",
                ),
                "n_queries": int(latencies.size),
                "query_time_ns": int(
                    state.query_cost_ns
                ),
                "migration_time_ns": int(
                    state.migration_cost_ns
                ),
                "total_time_ns": int(
                    state.query_cost_ns
                    + state.migration_cost_ns
                ),
                "median_query_latency_ns": float(
                    np.median(latencies)
                ),
                "p95_query_latency_ns": float(
                    np.quantile(
                        latencies,
                        0.95,
                    )
                ),
                "p99_query_latency_ns": float(
                    np.quantile(
                        latencies,
                        0.99,
                    )
                ),
                "migration_count": int(
                    state.migration_count
                ),
                "rewrite_bytes": int(
                    state.rewrite_bytes
                ),
                "final_codec": codecs[
                    state.active_position
                ],
            }
        )

    all_latency = np.asarray(
        all_latencies,
        dtype=np.int64,
    )
    query_time = int(
        sum(
            state.query_cost_ns
            for state in states.values()
        )
    )
    migration_time = int(
        sum(
            state.migration_cost_ns
            for state in states.values()
        )
    )
    summary = {
        "dataset": context["dataset"],
        "trace": plan.trace_name,
        "trace_seed": int(plan.trace_seed),
        "chunk_size": int(plan.chunk_size),
        "policy": "shape_ewma_be",
        "n_chunks": len(plan.chunk_ids),
        "n_queries": int(plan.n_events),
        "query_time_ns": query_time,
        "migration_time_ns": migration_time,
        "total_time_ns": (
            query_time + migration_time
        ),
        "throughput_qps": float(
            plan.n_events
            / max(
                (query_time + migration_time)
                / 1e9,
                1e-12,
            )
        ),
        "median_query_latency_ns": float(
            np.median(all_latency)
        ),
        "p95_query_latency_ns": float(
            np.quantile(
                all_latency,
                0.95,
            )
        ),
        "p99_query_latency_ns": float(
            np.quantile(
                all_latency,
                0.99,
            )
        ),
        "migration_count": int(
            sum(
                state.migration_count
                for state in states.values()
            )
        ),
        "rewrite_bytes": int(
            sum(
                state.rewrite_bytes
                for state in states.values()
            )
        ),
    }
    return (
        summary,
        pd.DataFrame(chunk_rows),
        pd.DataFrame(event_rows),
    )


def actual_audit_checkpoint_path(
    output_dir: Path,
    dataset: str,
    chunk_size: int,
    trace_name: str,
) -> Path:
    return (
        output_dir
        / "_actual_audit_checkpoints"
        / (
            f"{dataset.lower()}__"
            f"size_{chunk_size}__"
            f"trace_{trace_name}"
        )
    )


def run_actual_audit(
    contexts: Mapping[str, Mapping[str, Any]],
    selected_config: Mapping[str, Any],
    core: Any,
    e1: Any,
    base: Any,
    output_dir: Path,
    freeze_hash: str,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    reps = base.representation_registry(3)
    summary_frames: List[pd.DataFrame] = []
    chunk_frames: List[pd.DataFrame] = []
    event_frames: List[pd.DataFrame] = []

    for dataset in ("PEMS08", "BDG2"):
        context = contexts[dataset]
        stored_by_chunk = reconstruct_stored_maps(
            base,
            context["chunks"],
            context["codec_frame"],
            context["candidate_storage_dir"],
        )

        for chunk_size in CHUNK_SIZES:
            size_chunks = sorted(
                [
                    chunk
                    for chunk in context["chunks"]
                    if int(chunk.chunk_size)
                    == chunk_size
                ],
                key=lambda item: item.chunk_id,
            )
            size_assignments = context["assignments"][
                context["assignments"]["chunk_size"]
                == chunk_size
            ]
            for trace_name in ACTUAL_AUDIT_TRACES:
                checkpoint = (
                    actual_audit_checkpoint_path(
                        output_dir,
                        dataset,
                        chunk_size,
                        trace_name,
                    )
                )
                summary_path = (
                    checkpoint / "summary.csv"
                )
                chunks_path = (
                    checkpoint / "chunks.csv"
                )
                events_path = (
                    checkpoint / "events.csv"
                )
                complete_path = (
                    checkpoint / "complete.json"
                )

                if (
                    complete_path.exists()
                    and summary_path.exists()
                    and chunks_path.exists()
                    and events_path.exists()
                ):
                    marker = json.loads(
                        complete_path.read_text(
                            encoding="utf-8"
                        )
                    )
                    if (
                        marker.get(
                            "experiment_freeze_sha256"
                        )
                        == freeze_hash
                    ):
                        summary_frames.append(
                            pd.read_csv(summary_path)
                        )
                        chunk_frames.append(
                            pd.read_csv(chunks_path)
                        )
                        event_frames.append(
                            pd.read_csv(events_path)
                        )
                        print(
                            f"[S1 actual audit resume] "
                            f"{dataset}/{chunk_size}/"
                            f"{trace_name}"
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
                plan = core.build_trace_plan(
                    base,
                    size_chunks,
                    context["query_pools"],
                    size_assignments,
                    trace_name,
                    ACTUAL_AUDIT_TRACE_SEED,
                )
                summary, chunks, events = (
                    run_actual_shape_ewma(
                        core,
                        e1,
                        base,
                        reps,
                        context,
                        plan,
                        selected_config,
                        {
                            chunk_id: stored_by_chunk[
                                chunk_id
                            ]
                            for chunk_id in plan.chunk_ids
                        },
                        checkpoint
                        / "_migration_trials",
                    )
                )
                summary_frame = pd.DataFrame(
                    [summary]
                )
                write_dataframe_atomic(
                    summary_path,
                    summary_frame,
                )
                write_dataframe_atomic(
                    chunks_path,
                    chunks,
                )
                write_dataframe_atomic(
                    events_path,
                    events,
                )
                write_json_atomic(
                    complete_path,
                    {
                        "experiment_freeze_sha256": (
                            freeze_hash
                        ),
                        "dataset": dataset,
                        "chunk_size": chunk_size,
                        "trace": trace_name,
                        "trace_seed": (
                            ACTUAL_AUDIT_TRACE_SEED
                        ),
                        "completed_at_ns": (
                            time.time_ns()
                        ),
                    },
                )
                summary_frames.append(
                    summary_frame
                )
                chunk_frames.append(chunks)
                event_frames.append(events)
                print(
                    f"[S1 actual audit] "
                    f"{dataset}/{chunk_size}/"
                    f"{trace_name} complete"
                )

    return (
        pd.concat(
            summary_frames,
            ignore_index=True,
        ),
        pd.concat(
            chunk_frames,
            ignore_index=True,
        ),
        pd.concat(
            event_frames,
            ignore_index=True,
        ),
    )


def clustered_comparisons(
    chunk_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (
        dataset,
        trace,
        chunk_size,
    ), group in chunk_frame.groupby(
        ["dataset", "trace", "chunk_size"]
    ):
        pivot = group.pivot_table(
            index=[
                "trace_seed",
                "chunk_id",
                "unit_id",
            ],
            columns="policy",
            values="total_time_ns",
            aggfunc="first",
        ).dropna()
        if (
            "shape_ewma_be" not in pivot
            or "repressdb" not in pivot
        ):
            continue
        ratio = (
            pivot["repressdb"].to_numpy(
                dtype=np.float64
            )
            / pivot["shape_ewma_be"].to_numpy(
                dtype=np.float64
            )
        )
        low, high = bootstrap_median_ci(
            ratio,
            (
                dataset,
                trace,
                chunk_size,
                "repress_over_ewma",
            ),
        )
        rows.append(
            {
                "dataset": dataset,
                "trace": trace,
                "chunk_size": int(chunk_size),
                "n_paired_chunk_runs": int(
                    len(pivot)
                ),
                "median_ratio_repressdb_over_shape_ewma": float(
                    np.median(ratio)
                ),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "fraction_repressdb_faster": float(
                    np.mean(ratio < 1.0)
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "v070_script": project_dir
        / V070_SCRIPT_NAME,
        "v073_script": project_dir
        / V073_SCRIPT_NAME,
        "v074_script": project_dir
        / V074_SCRIPT_NAME,
        "a2_script": project_dir
        / A2_SCRIPT_NAME,
        "e5_script": project_dir
        / E5_SCRIPT_NAME,
        "v070_dir": project_dir
        / V070_RESULT_DIR_NAME,
        "v073_dir": project_dir
        / V073_RESULT_DIR_NAME,
        "v074_dir": project_dir
        / V074_RESULT_DIR_NAME,
        "a2_dir": project_dir
        / A2_RESULT_DIR_NAME,
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    print("=" * 116)
    print(
        "RePressDB S1 v1.3.1 — "
        "Shape-EWMA-BE development and holdouts"
    )
    print(f"Target journal: {TARGET_JOURNAL}")
    print("=" * 116)

    v073 = load_module(
        paths["v073_script"],
        "repressdb_v073_for_s1",
    )
    v074 = load_module(
        paths["v074_script"],
        "repressdb_v074_for_s1",
    )
    a2 = load_module(
        paths["a2_script"],
        "repressdb_a2_for_s1",
    )
    e5 = load_module(
        paths["e5_script"],
        "repressdb_e5_for_s1",
    )
    patched_core_dir = output_dir / "_patched_core"
    patched_core_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    core = v073.build_corrected_core(
        paths["v070_script"],
        patched_core_dir,
    )
    core.__file__ = str(
        paths["v070_script"].resolve()
    )
    e1v2, e1, base = (
        core.load_repressdb_modules()
    )

    a2_assessment = json.loads(
        (
            paths["a2_dir"]
            / "a2_holdout_assessment.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    if not bool(
        a2_assessment.get(
            "integrity_checks",
            {},
        ).get(
            "I9_all_integrity_checks_pass",
            False,
        )
    ):
        # Older assessment variants may not expose I9 under this exact key.
        integrity = a2_assessment.get(
            "integrity_checks",
            {},
        )
        if not integrity or not all(
            bool(value)
            for value in integrity.values()
        ):
            raise RuntimeError(
                "A2 holdout result is not an "
                "integrity-passing frozen parent."
            )

    parent_hashes = {
        "v070_protocol_sha256": (
            paths["v070_dir"]
            / "protocol_sha256.txt"
        ).read_text(
            encoding="utf-8"
        ).strip(),
        "v074_experiment_freeze_sha256": (
            paths["v074_dir"]
            / "experiment_freeze_sha256.txt"
        ).read_text(
            encoding="utf-8"
        ).strip(),
        "a2_experiment_freeze_sha256": (
            paths["a2_dir"]
            / "a2_experiment_freeze_sha256.txt"
        ).read_text(
            encoding="utf-8"
        ).strip(),
    }

    protocol = {
        "version": VERSION,
        "target_journal": TARGET_JOURNAL,
        "study": (
            "Shape-EWMA-BE strong online "
            "baseline development and frozen "
            "holdout validation"
        ),
        "parent_hashes": parent_hashes,
        "grid": {
            "alphas": list(ALPHAS),
            "deadbands": list(DEADBANDS),
            "prediction_horizons": list(
                HORIZONS
            ),
            "configuration_count": (
                len(ALPHAS)
                * len(DEADBANDS)
                * len(HORIZONS)
            ),
        },
        "decision_rule": {
            "decision_interval": (
                DECISION_INTERVAL
            ),
            "minimum_observations": (
                MIN_OBSERVATIONS
            ),
            "single_ewma": True,
            "dual_window": False,
            "confirmation": False,
            "cooldown": False,
            "break_even_factor": 1.0,
            "migration_cost_information": (
                "previously measured directed "
                "cost table; no future actual "
                "migration time"
            ),
        },
        "selection": {
            "dataset": "PEMS04 only",
            "action_traces": list(
                ACTION_TRACES
            ),
            "primary_score": (
                "geometric mean ratio to "
                "initial-best"
            ),
            "brief_median_limit": (
                MAX_BRIEF_MEDIAN_RATIO
            ),
            "stable_median_limit": (
                MAX_STABLE_MEDIAN_RATIO
            ),
            "tie_tolerance": (
                TIE_SCORE_TOLERANCE
            ),
        },
        "holdouts": [
            "PEMS08",
            "BDG2",
        ],
        "actual_audit": {
            "trace_seed": (
                ACTUAL_AUDIT_TRACE_SEED
            ),
            "traces": list(
                ACTUAL_AUDIT_TRACES
            ),
            "chunk_sizes": list(
                CHUNK_SIZES
            ),
            "purpose": (
                "file-backed validation of "
                "the frozen model baseline"
            ),
        },
    }
    write_json_atomic(
        output_dir / "s1_protocol.json",
        protocol,
    )
    protocol_digest = hashlib.sha256()
    for value in parent_hashes.values():
        protocol_digest.update(
            value.encode("ascii")
        )
    protocol_digest.update(
        sha256_file(
            output_dir / "s1_protocol.json"
        ).encode("ascii")
    )
    protocol_hash = (
        protocol_digest.hexdigest()
    )
    (
        output_dir
        / "s1_protocol_sha256.txt"
    ).write_text(
        protocol_hash + "\n",
        encoding="utf-8",
    )

    configurations = configuration_grid()
    pems04 = load_pems04_context(
        project_dir,
        core,
        e1,
        base,
        paths["v070_dir"],
        a2,
    )
    (
        grid_results,
        grid_summary,
        grid_ranked,
        selected_summary,
    ) = run_development_grid(
        pems04,
        configurations,
        core,
        e1,
        e1v2,
        base,
        output_dir,
    )

    selected_config = next(
        dict(config)
        for config in configurations
        if config["config_id"]
        == selected_summary["config_id"]
    )
    selected_config.update(
        {
            "selection_dataset": "PEMS04",
            "selection_protocol_sha256": (
                protocol_hash
            ),
            "selection_summary": (
                selected_summary
            ),
            "frozen_for_holdouts_and_r1": True,
            "may_not_be_retuned_after_holdout": True,
        }
    )
    prechain_config_path = (
        output_dir
        / "shape_ewma_be_selected_config_prechain.json"
    )
    write_json_atomic(
        prechain_config_path,
        selected_config,
    )

    freeze_material = {
        "s1_protocol_sha256": (
            protocol_hash
        ),
        "selected_config_prechain_sha256": sha256_file(
            prechain_config_path
        ),
        "development_trace_manifest_sha256": (
            sha256_file(
                output_dir
                / "development_trace_manifest.csv"
            )
        ),
        "script_sha256": sha256_file(
            Path(__file__).resolve()
        ),
        **parent_hashes,
    }
    write_json_atomic(
        output_dir
        / "s1_experiment_freeze_material.json",
        freeze_material,
    )
    freeze_hash = sha256_file(
        output_dir
        / "s1_experiment_freeze_material.json"
    )
    (
        output_dir
        / "s1_experiment_freeze_sha256.txt"
    ).write_text(
        freeze_hash + "\n",
        encoding="utf-8",
    )
    selected_config[
        "s1_experiment_freeze_sha256"
    ] = freeze_hash
    write_json_atomic(
        output_dir
        / "shape_ewma_be_frozen_config.json",
        selected_config,
    )

    print("-" * 116)
    print("[S1] frozen Shape-EWMA-BE configuration")
    print(
        json.dumps(
            selected_config,
            ensure_ascii=False,
            indent=2,
        )
    )

    holdouts = load_holdout_contexts(
        project_dir,
        core,
        e1,
        base,
        a2,
        e5,
        paths["v070_dir"],
        paths["a2_dir"],
    )

    global_frames: List[pd.DataFrame] = []
    chunk_frames: List[pd.DataFrame] = []
    migration_frames: List[pd.DataFrame] = []
    cumulative_frames: List[pd.DataFrame] = []
    for dataset in ("PEMS08", "BDG2"):
        (
            global_frame,
            chunk_frame,
            migration_frame,
            cumulative_frame,
        ) = evaluate_selected_on_context(
            holdouts[dataset],
            selected_config,
            core,
            e1,
            e1v2,
            base,
        )
        global_frames.append(global_frame)
        chunk_frames.append(chunk_frame)
        migration_frames.append(migration_frame)
        cumulative_frames.append(
            cumulative_frame
        )

    holdout_global = pd.concat(
        global_frames,
        ignore_index=True,
    )
    holdout_chunk = pd.concat(
        chunk_frames,
        ignore_index=True,
    )
    holdout_migrations = pd.concat(
        migration_frames,
        ignore_index=True,
    )
    holdout_cumulative = pd.concat(
        cumulative_frames,
        ignore_index=True,
    )
    comparisons = clustered_comparisons(
        holdout_chunk
    )

    write_dataframe_atomic(
        output_dir / "holdout_model_global.csv",
        holdout_global,
    )
    write_dataframe_atomic(
        output_dir / "holdout_model_chunk.csv",
        holdout_chunk,
    )
    write_dataframe_atomic(
        output_dir
        / "holdout_shape_ewma_migrations.csv",
        holdout_migrations,
    )
    write_dataframe_atomic(
        output_dir
        / "holdout_cumulative_cost.csv",
        holdout_cumulative,
    )
    write_dataframe_atomic(
        output_dir
        / "holdout_repressdb_vs_shape_ewma.csv",
        comparisons,
    )

    (
        actual_summary,
        actual_chunk,
        actual_events,
    ) = run_actual_audit(
        holdouts,
        selected_config,
        core,
        e1,
        base,
        output_dir,
        freeze_hash,
    )
    write_dataframe_atomic(
        output_dir / "actual_audit_summary.csv",
        actual_summary,
    )
    write_dataframe_atomic(
        output_dir / "actual_audit_chunk.csv",
        actual_chunk,
    )
    write_dataframe_atomic(
        output_dir / "actual_audit_events.csv",
        actual_events,
    )

    model_selected = holdout_global[
        (
            holdout_global["policy"]
            == "shape_ewma_be"
        )
        & (
            holdout_global["trace_seed"]
            == ACTUAL_AUDIT_TRACE_SEED
        )
        & (
            holdout_global["trace"].isin(
                ACTUAL_AUDIT_TRACES
            )
        )
    ][
        [
            "dataset",
            "trace",
            "trace_seed",
            "chunk_size",
            "total_time_ns",
        ]
    ].rename(
        columns={
            "total_time_ns": (
                "model_total_time_ns"
            )
        }
    )
    validation = actual_summary.merge(
        model_selected,
        on=[
            "dataset",
            "trace",
            "trace_seed",
            "chunk_size",
        ],
        how="inner",
        validate="one_to_one",
    )
    validation[
        "actual_over_model_ratio"
    ] = (
        validation["total_time_ns"]
        / validation["model_total_time_ns"]
    )
    validation[
        "absolute_log_ratio"
    ] = np.abs(
        np.log(
            validation[
                "actual_over_model_ratio"
            ]
        )
    )
    write_dataframe_atomic(
        output_dir
        / "actual_model_validation.csv",
        validation,
    )

    holdout_summary_rows: List[
        Dict[str, Any]
    ] = []
    for (
        dataset,
        trace,
        chunk_size,
    ), group in holdout_global.groupby(
        ["dataset", "trace", "chunk_size"]
    ):
        pivot = group.pivot(
            index="trace_seed",
            columns="policy",
            values="total_time_ns",
        ).dropna()
        if (
            "shape_ewma_be" in pivot
            and "repressdb" in pivot
        ):
            ratios = (
                pivot["repressdb"]
                / pivot["shape_ewma_be"]
            ).to_numpy(
                dtype=np.float64
            )
            holdout_summary_rows.append(
                {
                    "dataset": dataset,
                    "trace": trace,
                    "chunk_size": int(
                        chunk_size
                    ),
                    "n_trace_seeds": int(
                        len(pivot)
                    ),
                    "geometric_mean_ratio_repressdb_over_shape_ewma": geometric_mean(
                        ratios
                    ),
                    "median_ratio_repressdb_over_shape_ewma": float(
                        np.median(ratios)
                    ),
                    "fraction_repressdb_faster": float(
                        np.mean(ratios < 1.0)
                    ),
                }
            )
    holdout_summary = pd.DataFrame(
        holdout_summary_rows
    )
    write_dataframe_atomic(
        output_dir
        / "holdout_global_comparison_summary.csv",
        holdout_summary,
    )

    selected_row = grid_summary[
        grid_summary["config_id"]
        == selected_config["config_id"]
    ].iloc[0]
    expected_holdout_cells = (
        2
        * len(CHUNK_SIZES)
        * len(HOLDOUT_TRACE_NAMES)
        * len(TRACE_SEEDS)
        * 3
    )
    checks = {
        "S1_1_exactly_80_configs_evaluated": bool(
            grid_results[
                "config_id"
            ].nunique()
            == 80
        ),
        "S1_2_selected_config_passes_brief_safety": bool(
            selected_row[
                "brief_safety_pass"
            ]
        ),
        "S1_3_selected_config_passes_stable_safety": bool(
            selected_row[
                "stable_safety_pass"
            ]
        ),
        "S1_4_config_frozen_before_holdout_outputs": bool(
            selected_config[
                "frozen_for_holdouts_and_r1"
            ]
        ),
        "S1_5_both_holdouts_complete": bool(
            set(
                holdout_global["dataset"]
            )
            == {"PEMS08", "BDG2"}
        ),
        "S1_6_all_expected_model_holdout_rows_present": bool(
            len(holdout_global)
            == expected_holdout_cells
        ),
        "S1_7_actual_audit_12_runs_complete": bool(
            len(actual_summary)
            == (
                2
                * len(CHUNK_SIZES)
                * len(ACTUAL_AUDIT_TRACES)
            )
        ),
        "S1_8_actual_audit_queries_all_truth_checked": bool(
            (
                actual_summary["n_queries"]
                > 0
            ).all()
            and len(actual_events) > 0
        ),
        "S1_9_model_actual_validation_complete": bool(
            len(validation)
            == len(actual_summary)
            and np.isfinite(
                validation[
                    "actual_over_model_ratio"
                ]
            ).all()
        ),
        "S1_10_holdout_repressdb_comparison_reported": bool(
            len(comparisons) > 0
            and len(holdout_summary) > 0
        ),
    }
    checks["all_ten"] = bool(
        all(checks.values())
    )

    assessment = {
        "version": VERSION,
        "target_journal": TARGET_JOURNAL,
        "status": (
            "S1 complete"
            if checks["all_ten"]
            else (
                "S1 completed with failed "
                "acceptance checks"
            )
        ),
        "s1_protocol_sha256": (
            protocol_hash
        ),
        "s1_experiment_freeze_sha256": (
            freeze_hash
        ),
        "parent_hashes": parent_hashes,
        "frozen_config": selected_config,
        "development_selection": {
            "safe_config_count": int(
                grid_summary["safe"].sum()
            ),
            "selected_action_score": float(
                selected_row[
                    "action_geometric_mean_ratio"
                ]
            ),
            "selected_brief_median_ratio": float(
                selected_row[
                    "brief_median_ratio"
                ]
            ),
            "selected_stable_median_ratio": float(
                selected_row[
                    "stable_median_ratio"
                ]
            ),
        },
        "holdout_comparison_summary": (
            holdout_summary.to_dict(
                orient="records"
            )
        ),
        "actual_model_validation": {
            "n_cells": int(
                len(validation)
            ),
            "median_actual_over_model_ratio": float(
                validation[
                    "actual_over_model_ratio"
                ].median()
            ),
            "median_absolute_log_ratio": float(
                validation[
                    "absolute_log_ratio"
                ].median()
            ),
        },
        "acceptance_checks": checks,
        "scientific_interpretation_rule": {
            "if_repressdb_lower_total_cost": (
                "dual-window safety chain adds "
                "independent value"
            ),
            "if_total_cost_similar_but_repressdb_fewer_risky_migrations": (
                "RePressDB value is mainly "
                "risk control"
            ),
            "if_shape_ewma_faster_in_some_traces": (
                "speed-stability trade-off"
            ),
            "if_shape_ewma_dominates_cost_and_safety": (
                "RePressDB necessity is "
                "substantively weakened; do "
                "not hide this outcome"
            ),
        },
        "next_gate": (
            "Run R1 with this frozen config; "
            "do not retune."
            if checks["all_ten"]
            else (
                "Diagnose S1 failures before "
                "running R1."
            )
        ),
    }
    write_json_atomic(
        output_dir / "s1_assessment.json",
        assessment,
    )
    write_json_atomic(
        project_dir / SUMMARY_JSON_NAME,
        assessment,
    )

    run_config = {
        "version": VERSION,
        "target_journal": TARGET_JOURNAL,
        "s1_protocol_sha256": (
            protocol_hash
        ),
        "s1_experiment_freeze_sha256": (
            freeze_hash
        ),
        "grid": configurations,
        "selected_config": selected_config,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "timer": "time.perf_counter_ns",
        },
    }
    write_json_atomic(
        output_dir / "run_config.json",
        run_config,
    )

    print("=" * 116)
    print("RePressDB S1 v1.3.1 finished")
    print(
        json.dumps(
            checks,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Results: {output_dir.resolve()}")
    print("Upload:")
    print("  s1_assessment.json")
    print("  shape_ewma_be_frozen_config.json")
    print("  development_config_ranked.csv")
    print("  holdout_model_global.csv")
    print("  holdout_model_chunk.csv")
    print("  holdout_repressdb_vs_shape_ewma.csv")
    print("  actual_audit_summary.csv")
    print("  actual_model_validation.csv")
    print("  s1_protocol.json")
    print("  s1_experiment_freeze_sha256.txt")
    print("=" * 116)
    return 0 if checks["all_ten"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
