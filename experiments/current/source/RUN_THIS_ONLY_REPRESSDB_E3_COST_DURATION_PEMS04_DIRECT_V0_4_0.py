#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RePressDB E3 v0.4.0 — one-click rewrite-cost and drift-duration sensitivity study.

Target venue: EDBT 2027 (CCF-B).

Run this file directly in PyCharm. No command-line arguments are required.

Keep these dependency files in the same directory:
  1) RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT.py
  2) RUN_THIS_ONLY_REPRESSDB_E0C_V0_2_0.py

Purpose
-------
E2 showed that the break-even component was rarely activated under the original
trace set. E3 deliberately varies:

  * the duration of a sustained scan -> point drift; and
  * a multiplier applied to measured rewrite costs.

The online controller itself is NOT retuned. Long/short windows, confirmation,
cooldown, prediction horizon, minimum relative improvement, and safety factor
remain frozen from E1 v0.2.

Policies
--------
  * initial_best:
      Select once from the initial scan-heavy workload and never migrate.
  * full_repressdb:
      Frozen dual-window controller including the measured-cost break-even gate.
  * no_break_even:
      Identical controller, but the break-even gate is disabled.

Measurement model
-----------------
Query replay is real and file-backed in warm_slicedmeta mode.

For every chunk, each directed codec migration is physically measured before
the sensitivity replay:
    decode current representation
    -> encode target representation
    -> flush/fsync
    -> validate losslessness

Sensitivity runs then replay these measured migration costs using multipliers
{0.5, 1, 2, 4, 8}. Candidate representation files are already materialized.
This prevents repeated OS scheduling noise from being mistaken for a rewrite
cost-multiplier effect.

Statistical unit
----------------
Source series. Trace seeds and multiple chunks from one series are repeated
measurements and are aggregated before paired inference.
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
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

VERSION = "0.4.0"
TARGET_VENUE = "EDBT 2027"
BASE_E1_VERSION = "0.1.0"
SEED = 20260718

TRACE_SEEDS = (20260718, 20260719)
CHUNK_SIZES = [2048, 8192, 16384]
MAX_SERIES = 12
MAX_CHUNKS_PER_SIZE = 4

INITIAL_SCAN_QUERIES = 800
DRIFT_DURATIONS = (400, 800, 1600, 3200)
REWRITE_COST_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0, 8.0)

POLICIES = (
    "initial_best",
    "full_repressdb",
    "no_break_even",
)

SHORT_OBSERVATION_WINDOW = 100
MIGRATION_MEASURE_REPEATS = 2

OUTPUT_DIR_NAME = "results_e3_pems04_cost_duration_v040"
SUMMARY_JSON_NAME = "E3_PEMS04_COST_DURATION_ASSESSMENT_v040.json"

QUICK_TEST = os.environ.get("REPRESSDB_E3_QUICK", "0") == "1"


def stable_seed(*parts: Any) -> int:
    token = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def load_e1_module() -> Any:
    here = Path(__file__).resolve().parent
    path = here / "RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT.py"
    if not path.exists():
        raise FileNotFoundError(
            "找不到 RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT.py。"
            "请把它与本文件放在同一文件夹。"
        )
    spec = importlib.util.spec_from_file_location("repressdb_e1_e3_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载基础代码：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if getattr(module, "VERSION", None) != BASE_E1_VERSION:
        raise RuntimeError(
            f"需要 E1 基础版本 {BASE_E1_VERSION}，当前为 "
            f"{getattr(module, 'VERSION', 'unknown')}"
        )
    return module


def observed_weights(
    query_classes: Iterable[str],
    all_classes: Sequence[str],
) -> Dict[str, float]:
    counts = Counter(query_classes)
    total = sum(counts.values())
    if total <= 0:
        return {qc: 0.0 for qc in all_classes}
    return {qc: counts.get(qc, 0) / total for qc in all_classes}


def build_scan_to_point_trace(
    e1: Any,
    base: Any,
    chunk: Any,
    query_pool: Mapping[str, List[Any]],
    trace_seed: int,
    drift_duration: int,
    initial_scan_queries: int,
) -> List[Any]:
    phases = [
        ("scan_heavy", int(initial_scan_queries)),
        ("point_heavy", int(drift_duration)),
    ]
    rng = np.random.default_rng(
        stable_seed(
            trace_seed,
            chunk.chunk_id,
            "scan_to_point_duration",
            drift_duration,
        )
    )
    trace: List[Any] = []
    global_index = 0
    for phase_id, (workload_name, phase_length) in enumerate(phases):
        weights = base.WORKLOADS[workload_name]
        classes = [
            qc for qc in e1.QUERY_CLASSES
            if float(weights.get(qc, 0.0)) > 0.0
        ]
        probabilities = np.asarray(
            [float(weights[qc]) for qc in classes],
            dtype=float,
        )
        probabilities /= probabilities.sum()
        selected_classes = rng.choice(
            classes,
            size=int(phase_length),
            p=probabilities,
        )
        for query_class in selected_classes:
            pool = query_pool[str(query_class)]
            query = pool[int(rng.integers(0, len(pool)))]
            trace.append(
                e1.TraceItem(
                    global_index,
                    phase_id,
                    workload_name,
                    query,
                )
            )
            global_index += 1
    return trace


def measure_directed_migration_costs(
    e1: Any,
    base: Any,
    reps: Mapping[str, Any],
    stored_map: Mapping[str, Any],
    chunk: Any,
    migration_root: Path,
    repeats: int,
) -> Tuple[
    Dict[Tuple[str, str], int],
    Dict[Tuple[str, str], int],
    List[Dict[str, Any]],
]:
    costs: Dict[Tuple[str, str], int] = {}
    rewrite_bytes: Dict[Tuple[str, str], int] = {}
    rows: List[Dict[str, Any]] = []

    codecs = sorted(stored_map)
    for current_codec in codecs:
        for target_codec in codecs:
            key = (current_codec, target_codec)
            if current_codec == target_codec:
                costs[key] = 0
                rewrite_bytes[key] = 0
                rows.append(
                    {
                        "chunk_id": chunk.chunk_id,
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
            byte_values: List[int] = []
            for repeat in range(repeats):
                elapsed_ns, rewritten = e1.perform_actual_migration(
                    base,
                    reps,
                    stored_map,
                    chunk,
                    current_codec,
                    target_codec,
                    migration_root,
                    (
                        f"measure__{chunk.chunk_id}__"
                        f"{current_codec}__to__{target_codec}__r{repeat}"
                    ),
                )
                timings.append(int(elapsed_ns))
                byte_values.append(int(rewritten))

            median_ns = int(
                np.median(np.asarray(timings, dtype=np.int64))
            )
            median_bytes = int(
                np.median(np.asarray(byte_values, dtype=np.int64))
            )
            costs[key] = median_ns
            rewrite_bytes[key] = median_bytes
            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
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
    return costs, rewrite_bytes, rows


def decide_cost_sensitivity(
    profile: Mapping[str, Mapping[str, float]],
    state: Any,
    query_index: int,
    encode_ns: Mapping[str, int],
    decode_ns: Mapping[str, int],
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Frozen E1 v0.2 dual-window controller.

    The only experimental factors are:
      * whether the break-even gate is enabled; and
      * the multiplier applied to a physically measured migration cost.
    """
    e1 = decide_cost_sensitivity.e1
    migration_costs = decide_cost_sensitivity.migration_costs
    multiplier = float(decide_cost_sensitivity.rewrite_multiplier)
    use_break_even = bool(decide_cost_sensitivity.use_break_even)

    recent_long = list(state.recent_classes)
    long_weights = observed_weights(
        recent_long,
        e1.QUERY_CLASSES,
    )
    long_target, long_target_cost = e1.choose_best_codec(
        profile,
        long_weights,
    )
    current_long_cost = e1.workload_cost(
        profile,
        state.active_codec,
        long_weights,
    )
    long_relative = (
        (current_long_cost - long_target_cost) / current_long_cost
        if current_long_cost > 0
        else 0.0
    )

    recent_short = recent_long[-SHORT_OBSERVATION_WINDOW:]
    short_weights = observed_weights(
        recent_short,
        e1.QUERY_CLASSES,
    )
    short_target, short_target_cost = e1.choose_best_codec(
        profile,
        short_weights,
    )
    current_short_cost = e1.workload_cost(
        profile,
        state.active_codec,
        short_weights,
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
    base_rewrite_ns = int(
        migration_costs.get((state.active_codec, target), 0)
    )
    scaled_rewrite_ns = int(round(base_rewrite_ns * multiplier))
    predicted_saving = (
        conservative_gain * int(e1.PREDICTION_HORIZON)
    )

    detail = {
        "current_expected_ns": current_long_cost,
        "target_expected_ns": long_target_cost,
        "relative_improvement": conservative_relative,
        "predicted_saving_ns": predicted_saving,
        "rewrite_estimate_ns": scaled_rewrite_ns,
        "base_measured_rewrite_ns": base_rewrite_ns,
        "rewrite_cost_multiplier": multiplier,
        "use_break_even": use_break_even,
        "long_target_codec": long_target,
        "short_target_codec": short_target,
        "long_relative_improvement": long_relative,
        "short_relative_improvement": short_relative,
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

    if query_index - state.last_switch_query < int(e1.COOLDOWN_QUERIES):
        state.pending_candidate = None
        state.pending_rounds = 0
        detail["reason"] = "cooldown"
        return None, detail

    if conservative_relative < float(e1.MIN_RELATIVE_IMPROVEMENT):
        state.pending_candidate = None
        state.pending_rounds = 0
        detail["reason"] = "relative_gain_too_small"
        return None, detail

    if use_break_even and (
        predicted_saving
        < float(e1.REWRITE_SAFETY_FACTOR) * scaled_rewrite_ns
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

    if state.pending_rounds < int(e1.CONFIRMATION_ROUNDS):
        detail["reason"] = "awaiting_confirmation"
        return None, detail

    state.pending_candidate = None
    state.pending_rounds = 0
    detail["reason"] = "switch"
    return target, detail


def replay_measured_migration(
    base: Any,
    reps: Mapping[str, Any],
    stored_map: Mapping[str, Any],
    chunk: Any,
    current_codec: str,
    target_codec: str,
    migration_dir: Path,
    token: str,
) -> Tuple[int, int]:
    costs = replay_measured_migration.migration_costs
    byte_map = replay_measured_migration.rewrite_bytes
    multiplier = float(replay_measured_migration.rewrite_multiplier)
    key = (current_codec, target_codec)
    return (
        int(round(float(costs[key]) * multiplier)),
        int(byte_map[key]),
    )


def run_sensitivity_policy(
    e1: Any,
    base: Any,
    reps: Mapping[str, Any],
    stored_map: Mapping[str, Any],
    chunk: Any,
    trace_name: str,
    trace: Sequence[Any],
    profile: Mapping[str, Mapping[str, float]],
    encode_ns: Mapping[str, int],
    decode_ns: Mapping[str, int],
    policy_name: str,
    migration_root: Path,
    migration_costs: Mapping[Tuple[str, str], int],
    rewrite_bytes: Mapping[Tuple[str, str], int],
    rewrite_multiplier: float,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    decide_cost_sensitivity.migration_costs = migration_costs
    decide_cost_sensitivity.rewrite_multiplier = float(
        rewrite_multiplier
    )
    decide_cost_sensitivity.use_break_even = (
        policy_name == "full_repressdb"
    )
    replay_measured_migration.migration_costs = migration_costs
    replay_measured_migration.rewrite_bytes = rewrite_bytes
    replay_measured_migration.rewrite_multiplier = float(
        rewrite_multiplier
    )

    if policy_name == "initial_best":
        base_policy = "initial_best"
    elif policy_name in {"full_repressdb", "no_break_even"}:
        base_policy = "repressdb"
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    summary, decisions, timeline, phases = e1.run_policy(
        base,
        reps,
        stored_map,
        chunk,
        trace_name,
        trace,
        profile,
        encode_ns,
        decode_ns,
        base_policy,
        migration_root,
    )

    summary["policy"] = policy_name
    for row in decisions:
        row["policy"] = policy_name
    for row in timeline:
        row["policy"] = policy_name
    for row in phases:
        row["policy"] = policy_name
    return summary, decisions, timeline, phases


def bootstrap_median_ci(
    values: np.ndarray,
    seed: int,
    iterations: int = 4000,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=float)
    for i in range(iterations):
        samples[i] = np.median(
            rng.choice(values, size=values.size, replace=True)
        )
    return (
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def aggregate_results(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    keys = [
        "chunk_size",
        "drift_duration",
        "rewrite_cost_multiplier",
        "policy",
    ]
    for group_keys, group in run_df.groupby(keys, sort=True):
        chunk_size, duration, multiplier, policy = group_keys
        rows.append(
            {
                "chunk_size": int(chunk_size),
                "drift_duration": int(duration),
                "rewrite_cost_multiplier": float(multiplier),
                "policy": str(policy),
                "n_runs": int(len(group)),
                "n_chunks": int(group["chunk_id"].nunique()),
                "n_series": int(group["series_index"].nunique()),
                "median_total_time_ns": float(
                    group["total_time_ns"].median()
                ),
                "median_query_time_ns": float(
                    group["query_time_ns"].median()
                ),
                "median_migration_time_ns": float(
                    group["migration_time_ns"].median()
                ),
                "median_migration_count": float(
                    group["migration_count"].median()
                ),
                "switch_fraction": float(
                    (group["migration_count"] > 0).mean()
                ),
                "median_rewrite_bytes": float(
                    group["rewrite_bytes"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def series_clustered_comparisons(
    run_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    series_df = (
        run_df.groupby(
            [
                "chunk_size",
                "drift_duration",
                "rewrite_cost_multiplier",
                "series_index",
                "policy",
            ],
            as_index=False,
        )["total_time_ns"]
        .median()
    )

    for keys, group in series_df.groupby(
        [
            "chunk_size",
            "drift_duration",
            "rewrite_cost_multiplier",
        ],
        sort=True,
    ):
        chunk_size, duration, multiplier = keys
        pivot = group.pivot(
            index="series_index",
            columns="policy",
            values="total_time_ns",
        ).dropna()

        comparisons = (
            ("full_repressdb", "initial_best"),
            ("no_break_even", "initial_best"),
            ("full_repressdb", "no_break_even"),
        )
        for policy, reference in comparisons:
            if policy not in pivot or reference not in pivot:
                continue
            values_a = pivot[policy].to_numpy(dtype=float)
            values_b = pivot[reference].to_numpy(dtype=float)
            ratios = values_a / values_b
            try:
                p_value = float(
                    wilcoxon(
                        values_a,
                        values_b,
                        alternative="two-sided",
                        zero_method="wilcox",
                    ).pvalue
                )
            except ValueError:
                p_value = 1.0
            lo, hi = bootstrap_median_ci(
                ratios,
                stable_seed(
                    SEED,
                    chunk_size,
                    duration,
                    multiplier,
                    policy,
                    reference,
                    "series_bootstrap",
                ),
            )
            rows.append(
                {
                    "chunk_size": int(chunk_size),
                    "drift_duration": int(duration),
                    "rewrite_cost_multiplier": float(multiplier),
                    "policy": policy,
                    "reference_policy": reference,
                    "n_independent_series": int(len(pivot)),
                    "median_ratio_policy_over_reference": float(
                        np.median(ratios)
                    ),
                    "mean_ratio_policy_over_reference": float(
                        np.mean(ratios)
                    ),
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "fraction_policy_faster": float(
                        np.mean(ratios < 1.0)
                    ),
                    "wilcoxon_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def build_crossover_summary(
    run_df: pd.DataFrame,
    comparisons_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def comparison_ratio(
        size: int,
        duration: int,
        multiplier: float,
        policy: str,
        reference: str,
    ) -> float:
        local = comparisons_df[
            (comparisons_df["chunk_size"] == size)
            & (comparisons_df["drift_duration"] == duration)
            & np.isclose(
                comparisons_df["rewrite_cost_multiplier"],
                multiplier,
            )
            & (comparisons_df["policy"] == policy)
            & (comparisons_df["reference_policy"] == reference)
        ]
        return (
            float(local.iloc[0]["median_ratio_policy_over_reference"])
            if not local.empty
            else math.nan
        )

    for size in sorted(run_df["chunk_size"].unique()):
        for duration in sorted(run_df["drift_duration"].unique()):
            for multiplier in sorted(
                run_df["rewrite_cost_multiplier"].unique()
            ):
                local = run_df[
                    (run_df["chunk_size"] == size)
                    & (run_df["drift_duration"] == duration)
                    & np.isclose(
                        run_df["rewrite_cost_multiplier"],
                        multiplier,
                    )
                ]
                pivot = local.pivot_table(
                    index=["chunk_id", "trace_seed"],
                    columns="policy",
                    values=[
                        "total_time_ns",
                        "migration_count",
                    ],
                    aggfunc="first",
                ).dropna()

                if pivot.empty:
                    continue

                full_switch = (
                    pivot[("migration_count", "full_repressdb")] > 0
                )
                no_break_switch = (
                    pivot[("migration_count", "no_break_even")] > 0
                )
                prevented = no_break_switch & (~full_switch)
                if prevented.any():
                    full_wins_prevented = float(
                        (
                            pivot.loc[
                                prevented,
                                ("total_time_ns", "full_repressdb"),
                            ]
                            <
                            pivot.loc[
                                prevented,
                                ("total_time_ns", "no_break_even"),
                            ]
                        ).mean()
                    )
                else:
                    full_wins_prevented = math.nan

                rows.append(
                    {
                        "chunk_size": int(size),
                        "drift_duration": int(duration),
                        "rewrite_cost_multiplier": float(multiplier),
                        "full_vs_initial_ratio": comparison_ratio(
                            int(size),
                            int(duration),
                            float(multiplier),
                            "full_repressdb",
                            "initial_best",
                        ),
                        "no_break_even_vs_initial_ratio": comparison_ratio(
                            int(size),
                            int(duration),
                            float(multiplier),
                            "no_break_even",
                            "initial_best",
                        ),
                        "full_vs_no_break_even_ratio": comparison_ratio(
                            int(size),
                            int(duration),
                            float(multiplier),
                            "full_repressdb",
                            "no_break_even",
                        ),
                        "full_switch_fraction": float(full_switch.mean()),
                        "no_break_even_switch_fraction": float(
                            no_break_switch.mean()
                        ),
                        "break_even_prevented_switch_fraction": float(
                            prevented.mean()
                        ),
                        "full_wins_given_prevented_switch_fraction": (
                            full_wins_prevented
                        ),
                    }
                )
    return pd.DataFrame(rows)


def make_assessment(
    crossover_df: pd.DataFrame,
    migration_cost_df: pd.DataFrame,
    dataset_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    def row(
        size: int,
        duration: int,
        multiplier: float,
    ) -> pd.Series:
        local = crossover_df[
            (crossover_df["chunk_size"] == size)
            & (crossover_df["drift_duration"] == duration)
            & np.isclose(
                crossover_df["rewrite_cost_multiplier"],
                multiplier,
            )
        ]
        if len(local) != 1:
            raise ValueError(
                f"Missing sensitivity cell: {size}, {duration}, {multiplier}"
            )
        return local.iloc[0]

    baseline_long = {}
    for size in (8192, 16384):
        if size in set(crossover_df["chunk_size"]):
            r = row(size, int(crossover_df["drift_duration"].max()), 1.0)
            baseline_long[str(size)] = {
                "full_vs_initial_ratio": float(
                    r["full_vs_initial_ratio"]
                ),
                "full_switch_fraction": float(
                    r["full_switch_fraction"]
                ),
            }

    high_cost_short = crossover_df[
        (crossover_df["drift_duration"] <= 800)
        & (crossover_df["rewrite_cost_multiplier"] >= 4.0)
    ].copy()
    if not high_cost_short.empty:
        median_full_vs_no_break = float(
            high_cost_short["full_vs_no_break_even_ratio"].median()
        )
        mean_prevented = float(
            high_cost_short[
                "break_even_prevented_switch_fraction"
            ].mean()
        )
    else:
        median_full_vs_no_break = math.nan
        mean_prevented = math.nan

    monotonic_cells = 0
    monotonic_pass = 0
    for (size, duration), group in crossover_df.groupby(
        ["chunk_size", "drift_duration"]
    ):
        ordered = group.sort_values("rewrite_cost_multiplier")
        rates = ordered["full_switch_fraction"].to_numpy(dtype=float)
        monotonic_cells += 1
        if np.all(np.diff(rates) <= 1e-12):
            monotonic_pass += 1

    migration_nonzero = migration_cost_df[
        migration_cost_df["current_codec"]
        != migration_cost_df["target_codec"]
    ]["median_migration_ns"].to_numpy(dtype=float)

    findings = {
        "baseline_cost_long_drift": baseline_long,
        "high_cost_short_drift": {
            "median_full_vs_no_break_even_ratio": (
                median_full_vs_no_break
            ),
            "mean_break_even_prevented_switch_fraction": (
                mean_prevented
            ),
        },
        "switch_rate_monotonicity": {
            "n_size_duration_cells": int(monotonic_cells),
            "n_nonincreasing_cells": int(monotonic_pass),
            "fraction_nonincreasing": (
                float(monotonic_pass / monotonic_cells)
                if monotonic_cells
                else math.nan
            ),
        },
        "measured_migration_cost_ns": {
            "n_directed_nonidentity_measurements": int(
                migration_nonzero.size
            ),
            "median": float(np.median(migration_nonzero)),
            "p95": float(np.quantile(migration_nonzero, 0.95)),
            "maximum": float(np.max(migration_nonzero)),
        },
    }

    descriptive_checks = {
        "D1_baseline_long_scan_to_point_remains_beneficial_at_8192_and_16384": bool(
            all(
                v["full_vs_initial_ratio"] < 1.0
                for v in baseline_long.values()
            )
            if baseline_long
            else False
        ),
        "D2_full_switch_rate_is_nonincreasing_with_rewrite_cost_in_most_cells": bool(
            monotonic_cells > 0
            and monotonic_pass / monotonic_cells >= 0.75
        ),
        "D3_break_even_prevents_some_high_cost_short_drift_switches": bool(
            np.isfinite(mean_prevented)
            and mean_prevented > 0.0
        ),
        "D4_break_even_is_not_worse_than_no_break_even_in_high_cost_short_drift": bool(
            np.isfinite(median_full_vs_no_break)
            and median_full_vs_no_break <= 1.0
        ),
    }

    return {
        "version": VERSION,
        "target_venue": TARGET_VENUE,
        "dataset_label": dataset_meta["dataset_label"],
        "status": (
            "PEMS04 rewrite-cost and drift-duration sensitivity study"
        ),
        "study_role": (
            "Post-freeze sensitivity analysis. It characterizes operating "
            "regimes and does not retune the E1 v0.2 controller."
        ),
        "primary_execution_mode": "warm_slicedmeta",
        "migration_cost_model": (
            "Each directed codec migration is physically measured per chunk; "
            "sensitivity replay applies a declared multiplier to the measured "
            "cost while executing all queries against prebuilt candidate files."
        ),
        "statistical_unit": (
            "Source-series clustered paired inference; trace seeds and "
            "multiple chunks within one series are repeated measurements."
        ),
        "descriptive_checks": {
            **descriptive_checks,
            "all_four": bool(all(descriptive_checks.values())),
        },
        "findings": findings,
        "important_limitations": [
            "PEMS04 is the development dataset; E3 is a mechanism sensitivity study, not a new holdout.",
            "Rewrite-cost multipliers are controlled counterfactuals applied to physically measured migration costs.",
            "The controller uses a fixed 1200-query prediction horizon and does not know the true remaining drift duration.",
            "Queries execute in warm_slicedmeta mode; this is not a cold-SSD claim.",
        ],
        "next_gate": (
            "Use the cost-duration surface to report when break-even gating "
            "matters. Do not retune the frozen controller from E3."
        ),
    }


def write_json(path: Path, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            value,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        )


def main() -> int:
    e1 = load_e1_module()
    base = e1.load_base_module()
    e1.QUICK_TEST = QUICK_TEST

    original_decide = e1.decide_repressdb
    original_migrate = e1.perform_actual_migration
    e1.decide_repressdb = decide_cost_sensitivity
    e1.perform_actual_migration = replay_measured_migration
    decide_cost_sensitivity.e1 = e1

    print("=" * 96)
    print(
        f"RePressDB E3 cost-duration sensitivity v{VERSION}"
    )
    print(f"Target: {TARGET_VENUE}")
    print(
        "Direct-run PEMS04 experiment; no command-line arguments are required."
    )
    print("=" * 96)

    output_dir = Path(
        "results_e3_quick_smoke"
        if QUICK_TEST
        else OUTPUT_DIR_NAME
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_dir = output_dir / "_storage"
    migration_measure_dir = (
        output_dir / "_migration_measurement_trials"
    )
    replay_migration_dir = output_dir / "_replay_migration"
    storage_dir.mkdir(parents=True, exist_ok=True)
    migration_measure_dir.mkdir(parents=True, exist_ok=True)
    replay_migration_dir.mkdir(parents=True, exist_ok=True)

    if QUICK_TEST:
        matrix, dataset_meta = base.synthetic_dataset(SEED, 4)
        dataset_label = "SYNTHETIC_E3_COST_DURATION_SMOKE"
        chunk_sizes = [2048, 8192]
        max_chunks_per_size = 1
        trace_seeds = (SEED,)
        durations = (120, 240)
        multipliers = (1.0, 4.0)
        initial_scan_queries = 80
        migration_repeats = 1
    else:
        pems04_dir, _ = base.default_data_dirs()
        data_file = base.auto_detect_data_file(
            pems04_dir,
            "PEMS04_E3_COST_DURATION",
        )
        matrix, dataset_meta = base.load_dataset(
            data_file,
            feature_index=0,
            max_series=MAX_SERIES,
            seed=SEED,
            excluded_series=None,
        )
        dataset_label = "PEMS04_E3_COST_DURATION_SENSITIVITY"
        chunk_sizes = CHUNK_SIZES
        max_chunks_per_size = MAX_CHUNKS_PER_SIZE
        trace_seeds = TRACE_SEEDS
        durations = DRIFT_DURATIONS
        multipliers = REWRITE_COST_MULTIPLIERS
        initial_scan_queries = INITIAL_SCAN_QUERIES
        migration_repeats = MIGRATION_MEASURE_REPEATS

    chunks = base.extract_chunks(
        matrix,
        chunk_sizes,
        max_chunks_per_size,
        SEED,
    )
    reps = base.representation_registry(3)

    print(
        f"[{dataset_label}] chunks={len(chunks)}, "
        f"durations={len(durations)}, "
        f"rewrite multipliers={len(multipliers)}, "
        f"policies={len(POLICIES)}"
    )

    calibration_rows: List[Dict[str, Any]] = []
    codec_rows: List[Dict[str, Any]] = []
    migration_cost_rows: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []
    phase_rows: List[Dict[str, Any]] = []

    try:
        for chunk_index, chunk in enumerate(chunks, start=1):
            chunk_dir = storage_dir / chunk.chunk_id
            stored_map, encode_ns, decode_ns = e1.encode_candidates(
                base,
                reps,
                chunk,
                chunk_dir,
            )

            for codec, stored in stored_map.items():
                codec_rows.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "series_index": int(chunk.series_index),
                        "chunk_start": int(chunk.chunk_start),
                        "chunk_size": int(chunk.chunk_size),
                        "codec": codec,
                        "raw_size_bytes": int(chunk.values.nbytes),
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
            query_pool = {
                query_class: [
                    query
                    for query in queries
                    if query.query_class == query_class
                ]
                for query_class in e1.QUERY_CLASSES
            }
            if any(not values for values in query_pool.values()):
                raise AssertionError(
                    f"Incomplete query pool: {chunk.chunk_id}"
                )

            profile, local_calibration = e1.calibrate_chunk(
                base,
                reps,
                stored_map,
                chunk,
                query_pool,
            )
            for row in local_calibration:
                row["series_index"] = int(chunk.series_index)
                row["chunk_start"] = int(chunk.chunk_start)
            calibration_rows.extend(local_calibration)

            # Temporarily restore the physical migration function while
            # measuring each directed migration pair.
            e1.perform_actual_migration = original_migrate
            migration_costs, rewrite_byte_map, local_cost_rows = (
                measure_directed_migration_costs(
                    e1,
                    base,
                    reps,
                    stored_map,
                    chunk,
                    migration_measure_dir / chunk.chunk_id,
                    repeats=migration_repeats,
                )
            )
            migration_cost_rows.extend(local_cost_rows)
            e1.perform_actual_migration = replay_measured_migration

            for trace_seed in trace_seeds:
                for duration in durations:
                    trace = build_scan_to_point_trace(
                        e1,
                        base,
                        chunk,
                        query_pool,
                        int(trace_seed),
                        int(duration),
                        int(initial_scan_queries),
                    )
                    trace_name = (
                        f"scan_to_point_d{int(duration)}"
                    )

                    for multiplier in multipliers:
                        policy_order = list(POLICIES)
                        random.Random(
                            stable_seed(
                                trace_seed,
                                chunk.chunk_id,
                                duration,
                                multiplier,
                                "policy_order",
                            )
                        ).shuffle(policy_order)

                        for policy_name in policy_order:
                            summary, decisions, _, phases = (
                                run_sensitivity_policy(
                                    e1,
                                    base,
                                    reps,
                                    stored_map,
                                    chunk,
                                    trace_name,
                                    trace,
                                    profile,
                                    encode_ns,
                                    decode_ns,
                                    policy_name,
                                    replay_migration_dir
                                    / chunk.chunk_id,
                                    migration_costs,
                                    rewrite_byte_map,
                                    float(multiplier),
                                )
                            )
                            summary.update(
                                {
                                    "trace_seed": int(trace_seed),
                                    "series_index": int(
                                        chunk.series_index
                                    ),
                                    "drift_duration": int(duration),
                                    "rewrite_cost_multiplier": float(
                                        multiplier
                                    ),
                                }
                            )
                            run_rows.append(summary)

                            for row in decisions:
                                row.update(
                                    {
                                        "trace_seed": int(trace_seed),
                                        "series_index": int(
                                            chunk.series_index
                                        ),
                                        "drift_duration": int(duration),
                                        "rewrite_cost_multiplier": float(
                                            multiplier
                                        ),
                                    }
                                )
                            decision_rows.extend(decisions)

                            for row in phases:
                                row.update(
                                    {
                                        "trace_seed": int(trace_seed),
                                        "series_index": int(
                                            chunk.series_index
                                        ),
                                        "drift_duration": int(duration),
                                        "rewrite_cost_multiplier": float(
                                            multiplier
                                        ),
                                    }
                                )
                            phase_rows.extend(phases)

            if (
                chunk_index == 1
                or chunk_index == len(chunks)
                or chunk_index
                % max(1, len(chunks) // 12)
                == 0
            ):
                print(
                    f"[{dataset_label}] "
                    f"{chunk_index}/{len(chunks)} chunks complete"
                )
    finally:
        e1.decide_repressdb = original_decide
        e1.perform_actual_migration = original_migrate

    calibration_df = pd.DataFrame(calibration_rows)
    codec_df = pd.DataFrame(codec_rows)
    migration_cost_df = pd.DataFrame(migration_cost_rows)
    run_df = pd.DataFrame(run_rows)
    decisions_df = pd.DataFrame(decision_rows)
    phase_df = pd.DataFrame(phase_rows)

    aggregate_df = aggregate_results(run_df)
    comparisons_df = series_clustered_comparisons(run_df)
    crossover_df = build_crossover_summary(
        run_df,
        comparisons_df,
    )

    dataset_meta = dict(dataset_meta)
    dataset_meta.update(
        {
            "dataset_label": dataset_label,
            "num_chunks": len(chunks),
            "chunk_sizes": list(chunk_sizes),
            "max_chunks_per_size": int(max_chunks_per_size),
            "trace_seeds": list(trace_seeds),
            "initial_scan_queries": int(initial_scan_queries),
            "drift_durations": list(durations),
            "rewrite_cost_multipliers": list(multipliers),
            "policies": list(POLICIES),
            "primary_execution_mode": "warm_slicedmeta",
            "target_venue": TARGET_VENUE,
        }
    )

    assessment = make_assessment(
        crossover_df,
        migration_cost_df,
        dataset_meta,
    )

    calibration_df.to_csv(
        output_dir / "calibration_records.csv",
        index=False,
    )
    codec_df.to_csv(
        output_dir / "codec_records.csv",
        index=False,
    )
    migration_cost_df.to_csv(
        output_dir / "measured_migration_costs.csv",
        index=False,
    )
    run_df.to_csv(
        output_dir / "policy_run_records.csv",
        index=False,
    )
    aggregate_df.to_csv(
        output_dir / "policy_aggregate_summary.csv",
        index=False,
    )
    comparisons_df.to_csv(
        output_dir
        / "policy_paired_comparisons_series_clustered.csv",
        index=False,
    )
    crossover_df.to_csv(
        output_dir / "cost_duration_crossover_summary.csv",
        index=False,
    )
    decisions_df.to_csv(
        output_dir / "decision_records.csv",
        index=False,
    )
    phase_df.to_csv(
        output_dir / "phase_records.csv",
        index=False,
    )
    write_json(
        output_dir / "dataset_summary.json",
        dataset_meta,
    )
    write_json(
        output_dir / "e3_assessment.json",
        assessment,
    )

    config = {
        "version": VERSION,
        "target_venue": TARGET_VENUE,
        "base_e1_version": BASE_E1_VERSION,
        "base_protocol": e1.BASE_PROTOCOL,
        "seed": SEED,
        "trace_seeds": list(trace_seeds),
        "chunk_sizes": list(chunk_sizes),
        "max_series": 4 if QUICK_TEST else MAX_SERIES,
        "max_chunks_per_size": int(max_chunks_per_size),
        "initial_scan_queries": int(initial_scan_queries),
        "drift_durations": list(durations),
        "rewrite_cost_multipliers": list(multipliers),
        "policies": list(POLICIES),
        "migration_measure_repeats": int(
            migration_repeats
        ),
        "decision_interval": (
            20 if QUICK_TEST else e1.DECISION_INTERVAL
        ),
        "long_observation_window": int(
            e1.OBSERVATION_WINDOW
        ),
        "short_observation_window": int(
            SHORT_OBSERVATION_WINDOW
        ),
        "min_observations": (
            30 if QUICK_TEST else e1.MIN_OBSERVATIONS
        ),
        "confirmation_rounds": int(
            e1.CONFIRMATION_ROUNDS
        ),
        "cooldown_queries": int(
            e1.COOLDOWN_QUERIES
        ),
        "prediction_horizon": int(
            e1.PREDICTION_HORIZON
        ),
        "minimum_relative_improvement": float(
            e1.MIN_RELATIVE_IMPROVEMENT
        ),
        "rewrite_safety_factor": float(
            e1.REWRITE_SAFETY_FACTOR
        ),
        "representations": list(reps),
        "workloads": base.WORKLOADS,
        "statistical_unit": (
            "Source series; trace seeds and chunks within series "
            "are repeated measurements."
        ),
        "study_role": (
            "Cost-duration sensitivity only; frozen controller "
            "is not retuned."
        ),
        "measurement_scope": (
            "Real warm_slicedmeta query replay. Directed codec "
            "migration costs are physically measured per chunk, "
            "then replayed under controlled cost multipliers."
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "timer": "time.perf_counter_ns",
        },
    }
    write_json(
        output_dir / "run_config.json",
        config,
    )
    write_json(
        Path(
            "E3_QUICK_ASSESSMENT.json"
            if QUICK_TEST
            else SUMMARY_JSON_NAME
        ),
        assessment,
    )

    shutil.rmtree(storage_dir, ignore_errors=True)
    shutil.rmtree(
        migration_measure_dir,
        ignore_errors=True,
    )
    shutil.rmtree(
        replay_migration_dir,
        ignore_errors=True,
    )

    print("-" * 96)
    print(
        f"Descriptive checks: "
        f"{assessment['descriptive_checks']}"
    )
    print(f"Results: {output_dir.resolve()}")
    print(
        "Done. Upload e3_assessment.json, "
        "cost_duration_crossover_summary.csv, "
        "policy_paired_comparisons_series_clustered.csv, "
        "measured_migration_costs.csv, "
        "policy_run_records.csv, decision_records.csv, "
        "dataset_summary.json, and run_config.json."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
