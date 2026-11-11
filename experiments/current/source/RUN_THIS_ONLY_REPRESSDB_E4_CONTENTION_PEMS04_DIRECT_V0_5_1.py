#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RePressDB E4 v0.5.1
Synchronized-contention validation for global rewrite-budget scheduling.

DIRECT-RUN VERSION
------------------
Open this file in PyCharm and click Run. No arguments are required.

Required files in the same directory:
  1) RUN_THIS_ONLY_REPRESSDB_E4_GLOBAL_BUDGET_PEMS04_DIRECT_V0_5_0.py
  2) RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT.py
  3) RUN_THIS_ONLY_REPRESSDB_E0C_V0_2_0.py

Why this follow-up exists
-------------------------
E4 v0.5.0 produced too little scheduler contention:
  * only 4--6 migrations per seed;
  * peak queue length <= 2;
  * 2x/4x budgets almost never queued work;
  * FIFO and benefit-aware policies often executed identical schedules.

This follow-up does NOT retune the frozen per-chunk controller. It changes only
the scheduler stress workload:

  * 24 chunks receive equal access frequency;
  * 18 chunks undergo the same sustained scan -> point transition;
  * 6 chunks remain scan-only controls;
  * each chunk receives 3200 local accesses;
  * the transition occurs after 800 local accesses;
  * confirmed proposals therefore arrive in a concentrated interval.

The base E4 experiment still executes every query for correctness and records
wall-clock latency. Because policy runs occur sequentially and individual
queries are very short, scheduler comparisons use a deterministic calibrated
cost replay as the PRIMARY metric:

  calibrated query-class median for active codec
  + physically measured migration time.

Raw wall-clock results are retained as a diagnostic only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

VERSION = "0.5.1"
TARGET_VENUE = "EDBT 2027"

OUTPUT_DIR_NAME = "results_e4_pems04_contention_v051"
SUMMARY_JSON_NAME = "E4_PEMS04_CONTENTION_ASSESSMENT_v051.json"

LOCAL_ACCESSES_PER_CHUNK = 3200
DRIFT_START_LOCAL_ACCESS = 800
SUSTAINED_CHUNK_FRACTION = 0.75
BUDGET_INTERVAL_GLOBAL_QUERIES = 1000

QUICK_TEST = os.environ.get("REPRESSDB_E4_QUICK", "0") == "1"


def stable_seed(*parts: Any) -> int:
    token = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def load_module(filename: str, module_name: str) -> Any:
    path = Path(__file__).resolve().parent / filename
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {filename}。请把完整包中的所有 Python 文件放在同一目录。"
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def synchronized_assignments(chunks: Sequence[Any]) -> Dict[str, str]:
    ordered = sorted(
        chunks,
        key=lambda chunk: (
            int(chunk.chunk_size),
            int(chunk.series_index),
            int(chunk.chunk_start),
            chunk.chunk_id,
        ),
    )
    n_sustained = int(round(len(ordered) * SUSTAINED_CHUNK_FRACTION))
    n_sustained = max(1, min(len(ordered), n_sustained))
    return {
        chunk.chunk_id: (
            "synchronized_sustained"
            if index < n_sustained
            else "scan_only"
        )
        for index, chunk in enumerate(ordered)
    }


def synchronized_global_trace(
    e1: Any,
    base: Any,
    chunks: Sequence[Any],
    query_pools: Mapping[str, Mapping[str, List[Any]]],
    workload_types: Mapping[str, str],
    trace_seed: int,
    trace_length: int,
) -> Tuple[List[Any], Dict[str, float]]:
    """
    Equal-frequency round-robin trace with a randomized order inside each cycle.

    Every chunk is accessed exactly once per local-access cycle. This makes
    confirmed migration proposals arrive close together and creates an actual
    global scheduling problem.
    """
    n_chunks = len(chunks)
    if n_chunks <= 0:
        raise ValueError("No chunks")
    local_cycles = trace_length // n_chunks
    if local_cycles <= 0:
        raise ValueError("Trace is too short")

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    rng = np.random.default_rng(
        stable_seed(trace_seed, "synchronized_contention_trace")
    )
    items: List[Any] = []
    global_index = 0

    # Scale the phase boundary in quick mode while retaining 25% scan warm-up.
    if local_cycles == LOCAL_ACCESSES_PER_CHUNK:
        drift_start = DRIFT_START_LOCAL_ACCESS
    else:
        drift_start = max(1, int(round(local_cycles * 0.25)))

    for local_access in range(local_cycles):
        order = rng.permutation(n_chunks)
        for position in order:
            chunk_id = chunk_ids[int(position)]
            workload_type = workload_types[chunk_id]
            if (
                workload_type == "synchronized_sustained"
                and local_access >= drift_start
            ):
                workload = "point_heavy"
            else:
                workload = "scan_heavy"

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
            query_class = str(rng.choice(classes, p=probabilities))
            pool = query_pools[chunk_id][query_class]
            query = pool[int(rng.integers(0, len(pool)))]

            items.append(
                synchronized_global_trace.e4.GlobalTraceItem(
                    global_index=global_index,
                    chunk_id=chunk_id,
                    workload=workload,
                    query=query,
                )
            )
            global_index += 1

    popularity = {
        chunk_id: 1.0 / n_chunks for chunk_id in chunk_ids
    }
    return items, popularity


def reconstruct_profile(calibration_df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
    profiles: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in calibration_df.itertuples(index=False):
        profiles.setdefault(str(row.chunk_id), {}).setdefault(
            str(row.codec), {}
        )[str(row.query_class)] = float(row.median_latency_ns)
    return profiles


def workload_cost(
    profile: Mapping[str, Mapping[str, float]],
    codec: str,
    weights: Mapping[str, float],
) -> float:
    return float(
        sum(
            float(weights.get(query_class, 0.0))
            * float(profile[codec][query_class])
            for query_class in profile[codec]
        )
    )


def initial_scan_codec(
    profile: Mapping[str, Mapping[str, float]],
    scan_weights: Mapping[str, float],
) -> str:
    candidates = [
        (codec, workload_cost(profile, codec, scan_weights))
        for codec in sorted(profile)
    ]
    return min(candidates, key=lambda item: (item[1], item[0]))[0]


def build_modeled_run_records(
    e4: Any,
    e1: Any,
    base: Any,
    output_dir: Path,
    chunks: Sequence[Any],
    query_pools: Mapping[str, Mapping[str, List[Any]]],
    assignments: Mapping[str, str],
    trace_seeds: Sequence[int],
    trace_length: int,
) -> pd.DataFrame:
    calibration_df = pd.read_csv(output_dir / "calibration_records.csv")
    raw_runs = pd.read_csv(output_dir / "policy_run_records.csv")
    events = pd.read_csv(output_dir / "scheduler_events.csv")
    profiles = reconstruct_profile(calibration_df)

    execute_events = events[events["event"] == "execute"].copy()
    execute_events["global_index"] = execute_events["global_index"].astype(int)

    rows: List[Dict[str, Any]] = []
    policies = sorted(raw_runs["policy"].unique())

    for trace_seed in trace_seeds:
        trace, _ = synchronized_global_trace(
            e1,
            base,
            chunks,
            query_pools,
            assignments,
            int(trace_seed),
            int(trace_length),
        )

        for policy in policies:
            active = {
                chunk.chunk_id: initial_scan_codec(
                    profiles[chunk.chunk_id],
                    base.WORKLOADS["scan_heavy"],
                )
                for chunk in chunks
            }

            policy_events = execute_events[
                (execute_events["trace_seed"] == trace_seed)
                & (execute_events["policy"] == policy)
            ].sort_values(["global_index", "chunk_id"])
            events_by_index: Dict[int, List[Any]] = {}
            for event in policy_events.itertuples(index=False):
                events_by_index.setdefault(int(event.global_index), []).append(event)

            modeled_query_ns = 0.0
            for item in trace:
                codec = active[item.chunk_id]
                modeled_query_ns += float(
                    profiles[item.chunk_id][codec][item.query.query_class]
                )

                # In the online run, a migration at this global index happens
                # after the current query has completed.
                for event in events_by_index.get(item.global_index, []):
                    active[str(event.chunk_id)] = str(event.target_codec)

            raw = raw_runs[
                (raw_runs["trace_seed"] == trace_seed)
                & (raw_runs["policy"] == policy)
            ]
            if len(raw) != 1:
                raise ValueError(
                    f"Run record is not unique: seed={trace_seed}, policy={policy}"
                )
            raw_row = raw.iloc[0]
            migration_ns = float(raw_row["migration_time_ns"])
            modeled_total_ns = modeled_query_ns + migration_ns

            row = raw_row.to_dict()
            row.update(
                {
                    "modeled_query_time_ns": modeled_query_ns,
                    "modeled_total_time_ns": modeled_total_ns,
                    "modeled_throughput_qps": (
                        float(raw_row["n_queries"])
                        / (modeled_total_ns / 1e9)
                        if modeled_total_ns > 0
                        else math.inf
                    ),
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def comparison_table(modeled_df: pd.DataFrame) -> pd.DataFrame:
    pivot = modeled_df.pivot(
        index="trace_seed",
        columns="policy",
        values="modeled_total_time_ns",
    )
    rows: List[Dict[str, Any]] = []

    for multiplier in e4_global.BUDGET_MULTIPLIERS:
        fifo = f"fifo_budget_{multiplier}x"
        benefit = f"benefit_budget_{multiplier}x"
        required = [
            "initial_best",
            "unbounded_repressdb",
            fifo,
            benefit,
        ]
        local = pivot.dropna(subset=required)
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
                    "budget_multiplier": int(multiplier),
                    "policy": policy,
                    "reference_policy": reference,
                    "n_trace_seeds": int(len(ratios)),
                    "median_ratio_policy_over_reference": float(np.median(ratios)),
                    "mean_ratio_policy_over_reference": float(np.mean(ratios)),
                    "fraction_policy_faster": float(np.mean(ratios < 1.0)),
                    "minimum_ratio": float(np.min(ratios)),
                    "maximum_ratio": float(np.max(ratios)),
                }
            )
    return pd.DataFrame(rows)


def aggregate_modeled(modeled_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for policy, group in modeled_df.groupby("policy", sort=True):
        rows.append(
            {
                "policy": policy,
                "n_trace_seeds": int(group["trace_seed"].nunique()),
                "median_modeled_total_time_ns": float(
                    group["modeled_total_time_ns"].median()
                ),
                "mean_modeled_total_time_ns": float(
                    group["modeled_total_time_ns"].mean()
                ),
                "median_modeled_query_time_ns": float(
                    group["modeled_query_time_ns"].median()
                ),
                "median_migration_time_ns": float(
                    group["migration_time_ns"].median()
                ),
                "median_migration_count": float(
                    group["migration_count"].median()
                ),
                "median_rewrite_bytes": float(
                    group["rewrite_bytes"].median()
                ),
                "median_peak_queue_length": float(
                    group["peak_queue_length"].median()
                ),
                "budget_violation_count": int(
                    group["budget_violation_count"].sum()
                ),
                "median_raw_wallclock_total_ns": float(
                    group["total_time_ns"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_assessment(
    aggregate_df: pd.DataFrame,
    comparisons_df: pd.DataFrame,
    raw_assessment: Mapping[str, Any],
) -> Dict[str, Any]:
    def ratio(multiplier: int, policy: str, reference: str) -> float:
        local = comparisons_df[
            (comparisons_df["budget_multiplier"] == multiplier)
            & (comparisons_df["policy"] == policy)
            & (comparisons_df["reference_policy"] == reference)
        ]
        return (
            float(local.iloc[0]["median_ratio_policy_over_reference"])
            if not local.empty
            else math.nan
        )

    benefit_vs_fifo = {
        str(multiplier): ratio(
            multiplier,
            f"benefit_budget_{multiplier}x",
            f"fifo_budget_{multiplier}x",
        )
        for multiplier in e4_global.BUDGET_MULTIPLIERS
    }
    n_benefit_not_worse = sum(
        np.isfinite(value) and value <= 1.0
        for value in benefit_vs_fifo.values()
    )
    benefit_4x_vs_unbounded = ratio(
        4, "benefit_budget_4x", "unbounded_repressdb"
    )

    peak_queue = float(aggregate_df["median_peak_queue_length"].max())
    violations = int(aggregate_df["budget_violation_count"].sum())

    initial = float(
        aggregate_df.loc[
            aggregate_df["policy"] == "initial_best",
            "median_modeled_total_time_ns",
        ].iloc[0]
    )
    benefit_1x = float(
        aggregate_df.loc[
            aggregate_df["policy"] == "benefit_budget_1x",
            "median_modeled_total_time_ns",
        ].iloc[0]
    )
    benefit_4x = float(
        aggregate_df.loc[
            aggregate_df["policy"] == "benefit_budget_4x",
            "median_modeled_total_time_ns",
        ].iloc[0]
    )

    checks = {
        "C1_meaningful_contention_peak_queue_at_least_5": bool(
            peak_queue >= 5
        ),
        "C2_no_budget_violations": bool(violations == 0),
        "C3_benefit_matches_or_beats_fifo_at_least_two_budget_levels": bool(
            n_benefit_not_worse >= 2
        ),
        "C4_benefit_4x_within_10_percent_of_unbounded": bool(
            np.isfinite(benefit_4x_vs_unbounded)
            and benefit_4x_vs_unbounded <= 1.10
        ),
        "C5_budgeted_adaptation_beats_initial_best_at_1x_and_4x": bool(
            benefit_1x < initial and benefit_4x < initial
        ),
    }

    return {
        "version": VERSION,
        "target_venue": TARGET_VENUE,
        "status": (
            "PEMS04 synchronized-contention global rewrite-budget study"
        ),
        "study_role": (
            "Corrected scheduler stress experiment after E4 v0.5.0 showed "
            "insufficient contention."
        ),
        "primary_scheduler_metric": (
            "Calibrated query-class median replay plus physically measured "
            "migration time."
        ),
        "raw_wallclock_role": (
            "Diagnostic only; microsecond query timing varies with sequential "
            "policy execution order."
        ),
        "contention_design": {
            "local_accesses_per_chunk": LOCAL_ACCESSES_PER_CHUNK,
            "drift_start_local_access": DRIFT_START_LOCAL_ACCESS,
            "sustained_chunk_fraction": SUSTAINED_CHUNK_FRACTION,
            "equal_access_frequency": True,
        },
        "descriptive_checks": {
            **checks,
            "all_five": bool(all(checks.values())),
        },
        "benefit_vs_fifo_ratio_by_budget": benefit_vs_fifo,
        "benefit_4x_vs_unbounded_ratio": benefit_4x_vs_unbounded,
        "peak_queue_length": peak_queue,
        "modeled_total_ratio": {
            "benefit_1x_vs_initial_best": benefit_1x / initial,
            "benefit_4x_vs_initial_best": benefit_4x / initial,
        },
        "raw_v050_diagnostic": dict(raw_assessment),
        "important_limitations": [
            "This is a synchronized contention stress workload, not a naturally observed production trace.",
            "Calibrated replay uses per-chunk, per-codec, per-query-class median latency.",
            "Queries are still physically executed to validate correctness; calibrated cost is used only to isolate scheduler ordering effects.",
            "Migration time and rewrite bytes are physically measured and retained.",
            "The per-chunk controller remains frozen.",
        ],
    }


def write_json(path: Path, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=True)


def main() -> int:
    global e4_global

    e4 = load_module(
        "RUN_THIS_ONLY_REPRESSDB_E4_GLOBAL_BUDGET_PEMS04_DIRECT_V0_5_0.py",
        "repressdb_e4_v050_base",
    )
    e1 = e4.load_e1_module()
    base = e1.load_base_module()
    e4_global = e4
    synchronized_global_trace.e4 = e4

    # Preserve original values for transparency in case the module is reused.
    original_values = {
        "VERSION": e4.VERSION,
        "OUTPUT_DIR_NAME": e4.OUTPUT_DIR_NAME,
        "SUMMARY_JSON_NAME": e4.SUMMARY_JSON_NAME,
        "GLOBAL_TRACE_LENGTH": e4.GLOBAL_TRACE_LENGTH,
        "BUDGET_INTERVAL_GLOBAL_QUERIES": (
            e4.BUDGET_INTERVAL_GLOBAL_QUERIES
        ),
        "assign_workload_types": e4.assign_workload_types,
        "build_global_trace": e4.build_global_trace,
    }

    if QUICK_TEST:
        # The base quick path uses four chunks and a 2000-query trace.
        # The synchronized builder scales the phase boundary automatically.
        pass
    else:
        e4.GLOBAL_TRACE_LENGTH = (
            LOCAL_ACCESSES_PER_CHUNK
            * (len(e4.CHUNK_SIZES) * e4.MAX_CHUNKS_PER_SIZE)
        )

    e4.VERSION = VERSION
    e4.OUTPUT_DIR_NAME = OUTPUT_DIR_NAME
    e4.SUMMARY_JSON_NAME = SUMMARY_JSON_NAME
    e4.BUDGET_INTERVAL_GLOBAL_QUERIES = (
        BUDGET_INTERVAL_GLOBAL_QUERIES
    )
    e4.assign_workload_types = synchronized_assignments
    e4.build_global_trace = synchronized_global_trace

    print("=" * 100)
    print("RePressDB E4 v0.5.1 synchronized-contention validation")
    print("The base experiment will run first; calibrated scheduler replay follows.")
    print("=" * 100)

    try:
        return_code = e4.main()
        if return_code != 0:
            return int(return_code)

        output_dir = Path(
            "results_e4_quick_smoke" if QUICK_TEST else OUTPUT_DIR_NAME
        )
        if not output_dir.exists():
            raise FileNotFoundError(f"Result directory not found: {output_dir}")

        calibration_df = pd.read_csv(output_dir / "calibration_records.csv")
        assignment_df = pd.read_csv(
            output_dir / "chunk_workload_assignments.csv"
        )
        dataset_summary = json.loads(
            (output_dir / "dataset_summary.json").read_text(encoding="utf-8")
        )
        raw_assessment_path = output_dir / "e4_assessment.json"
        raw_assessment = json.loads(
            raw_assessment_path.read_text(encoding="utf-8")
        )

        # Rebuild chunks/query pools deterministically so the calibrated replay
        # uses exactly the same trace generator and query instances.
        if QUICK_TEST:
            matrix, _ = base.synthetic_dataset(e4.SEED, 4)
            chunk_sizes = [2048, 8192]
            max_chunks = 2
            trace_seeds = (e4.SEED,)
            trace_length = 2000
        else:
            pems04_dir, _ = base.default_data_dirs()
            data_file = base.auto_detect_data_file(
                pems04_dir, "PEMS04_E4_CONTENTION_REBUILD"
            )
            matrix, _ = base.load_dataset(
                data_file,
                feature_index=0,
                max_series=e4.MAX_SERIES,
                seed=e4.SEED,
                excluded_series=None,
            )
            chunk_sizes = e4.CHUNK_SIZES
            max_chunks = e4.MAX_CHUNKS_PER_SIZE
            trace_seeds = e4.TRACE_SEEDS
            trace_length = e4.GLOBAL_TRACE_LENGTH

        chunks = base.extract_chunks(
            matrix, chunk_sizes, max_chunks, e4.SEED
        )
        reps = base.representation_registry(3)
        query_pools: Dict[str, Dict[str, List[Any]]] = {}
        for chunk in chunks:
            queries = base.build_queries(
                chunk.values,
                chunk.chunk_id,
                e4.SEED,
                1 if QUICK_TEST else e1.WINDOWS_PER_FRACTION,
            )
            query_pools[chunk.chunk_id] = {
                query_class: [
                    query
                    for query in queries
                    if query.query_class == query_class
                ]
                for query_class in e1.QUERY_CLASSES
            }

        assignments = {
            str(row.chunk_id): str(row.workload_type)
            for row in assignment_df.itertuples(index=False)
        }

        modeled_df = build_modeled_run_records(
            e4,
            e1,
            base,
            output_dir,
            chunks,
            query_pools,
            assignments,
            trace_seeds,
            trace_length,
        )
        comparisons_df = comparison_table(modeled_df)
        aggregate_df = aggregate_modeled(modeled_df)
        assessment = build_assessment(
            aggregate_df, comparisons_df, raw_assessment
        )

        # Preserve the original wall-clock assessment separately.
        shutil.copy2(
            raw_assessment_path,
            output_dir / "e4_wallclock_diagnostic_assessment.json",
        )

        modeled_df.to_csv(
            output_dir / "policy_run_records_modeled.csv", index=False
        )
        aggregate_df.to_csv(
            output_dir / "policy_aggregate_summary_modeled.csv", index=False
        )
        comparisons_df.to_csv(
            output_dir / "budget_comparison_summary_modeled.csv", index=False
        )
        write_json(raw_assessment_path, assessment)
        write_json(Path(SUMMARY_JSON_NAME), assessment)

        dataset_summary.update(
            {
                "experiment_version": VERSION,
                "contention_design": {
                    "local_accesses_per_chunk": (
                        trace_length // len(chunks)
                    ),
                    "drift_start_fraction": 0.25,
                    "equal_access_frequency": True,
                    "workload_type_counts": dict(
                        assignment_df["workload_type"].value_counts()
                    ),
                },
                "primary_scheduler_metric": (
                    "calibrated query cost + measured migration cost"
                ),
            }
        )
        write_json(output_dir / "dataset_summary.json", dataset_summary)

        print("-" * 100)
        print("Corrected contention assessment:")
        print(assessment["descriptive_checks"])
        print(f"Results: {output_dir.resolve()}")
        print(
            "Upload e4_assessment.json, "
            "budget_comparison_summary_modeled.csv, "
            "policy_aggregate_summary_modeled.csv, "
            "policy_run_records_modeled.csv, scheduler_events.csv, "
            "chunk_policy_summary.csv, measured_migration_costs.csv, "
            "dataset_summary.json, and run_config.json."
        )
        return 0
    finally:
        for name, value in original_values.items():
            setattr(e4, name, value)


if __name__ == "__main__":
    raise SystemExit(main())
