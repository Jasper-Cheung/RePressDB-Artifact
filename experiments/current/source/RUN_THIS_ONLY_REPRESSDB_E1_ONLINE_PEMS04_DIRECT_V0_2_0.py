#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RePressDB E1-A v0.2.0 — one-click PEMS04 multi-seed online adaptation experiment.

Target: EDBT 2027 (CCF-B).

Run this file directly in PyCharm. No command-line arguments are required.
Keep these two files in the same directory:
  1) RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT.py
  2) RUN_THIS_ONLY_REPRESSDB_E0C_V0_2_0.py

Changes from E1-A v0.1.0:
  * fixes stale-window migration after a short workload burst;
  * requires long-window and short-window layout recommendations to agree;
  * uses conservative benefit estimated from both time scales;
  * replays three deterministic trace seeds with one calibration per chunk;
  * adds paired policy statistics and post-hoc switch-alignment diagnostics.
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

VERSION = "0.2.0"
TARGET_VENUE = "EDBT 2027"
BASE_E1_VERSION = "0.1.0"
SEED = 20260718
TRACE_SEEDS = (20260718, 20260719, 20260720)
SHORT_OBSERVATION_WINDOW = 100
OUTPUT_DIR_NAME = "results_e1_pems04_online_v020"
SUMMARY_JSON_NAME = "E1_PEMS04_ONLINE_ASSESSMENT_v020.json"
QUICK_TEST = os.environ.get("REPRESSDB_E1_QUICK", "0") == "1"


def stable_seed(*parts: Any) -> int:
    token = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def load_e1_module() -> Any:
    here = Path(__file__).resolve().parent
    path = here / "RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT.py"
    if not path.exists():
        raise FileNotFoundError(
            "找不到 E1 v0.1.0 基础代码。请把 "
            "RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT.py 与本文件放在同一文件夹。"
        )
    spec = importlib.util.spec_from_file_location("repressdb_e1_v010_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载基础代码：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if getattr(module, "VERSION", None) != BASE_E1_VERSION:
        raise RuntimeError(
            f"需要 E1 基础版本 {BASE_E1_VERSION}，当前为 {getattr(module, 'VERSION', 'unknown')}"
        )
    return module


def observed_weights(query_classes: Iterable[str], all_classes: Sequence[str]) -> Dict[str, float]:
    counts = Counter(query_classes)
    total = sum(counts.values())
    if total <= 0:
        return {qc: 0.0 for qc in all_classes}
    return {qc: counts.get(qc, 0) / total for qc in all_classes}


def decide_repressdb_two_timescale(
    profile: Mapping[str, Mapping[str, float]],
    state: Any,
    query_index: int,
    encode_ns: Mapping[str, int],
    decode_ns: Mapping[str, int],
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Cost-aware decision with a short-window stale-signal veto."""
    e1 = decide_repressdb_two_timescale.e1
    recent_long = list(state.recent_classes)
    recent_short = recent_long[-SHORT_OBSERVATION_WINDOW:]
    long_weights = observed_weights(recent_long, e1.QUERY_CLASSES)
    short_weights = observed_weights(recent_short, e1.QUERY_CLASSES)

    long_target, long_target_cost = e1.choose_best_codec(profile, long_weights)
    short_target, short_target_cost = e1.choose_best_codec(profile, short_weights)
    current_long_cost = e1.workload_cost(profile, state.active_codec, long_weights)
    current_short_cost = e1.workload_cost(profile, state.active_codec, short_weights)

    long_relative = (
        (current_long_cost - long_target_cost) / current_long_cost
        if current_long_cost > 0 else 0.0
    )
    short_relative = (
        (current_short_cost - short_target_cost) / current_short_cost
        if current_short_cost > 0 else 0.0
    )
    conservative_relative = min(long_relative, short_relative)
    target = long_target
    rewrite_est = e1.migration_estimate_ns(state.active_codec, target, encode_ns, decode_ns)
    conservative_gain = max(
        0.0,
        min(
            current_long_cost - long_target_cost,
            current_short_cost - short_target_cost,
        ),
    )
    predicted_saving = conservative_gain * e1.PREDICTION_HORIZON

    detail = {
        "current_expected_ns": current_long_cost,
        "target_expected_ns": long_target_cost,
        "relative_improvement": conservative_relative,
        "predicted_saving_ns": predicted_saving,
        "rewrite_estimate_ns": rewrite_est,
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

    if query_index - state.last_switch_query < e1.COOLDOWN_QUERIES:
        state.pending_candidate = None
        state.pending_rounds = 0
        detail["reason"] = "cooldown"
        return None, detail

    if conservative_relative < e1.MIN_RELATIVE_IMPROVEMENT:
        state.pending_candidate = None
        state.pending_rounds = 0
        detail["reason"] = "relative_gain_too_small"
        return None, detail

    if predicted_saving < e1.REWRITE_SAFETY_FACTOR * rewrite_est:
        state.pending_candidate = None
        state.pending_rounds = 0
        detail["reason"] = "break_even_not_met"
        return None, detail

    if state.pending_candidate == target:
        state.pending_rounds += 1
    else:
        state.pending_candidate = target
        state.pending_rounds = 1

    if state.pending_rounds < e1.CONFIRMATION_ROUNDS:
        detail["reason"] = "awaiting_confirmation"
        return None, detail

    state.pending_candidate = None
    state.pending_rounds = 0
    detail["reason"] = "switch"
    return target, detail


def build_trace(
    e1: Any,
    base: Any,
    chunk: Any,
    trace_name: str,
    query_pool: Mapping[str, List[Any]],
    trace_seed: int,
) -> List[Any]:
    phases = e1.TRACE_SPECS[trace_name]
    if QUICK_TEST:
        phases = [(name, max(60, length // 20)) for name, length in phases]
    rng = np.random.default_rng(stable_seed(trace_seed, chunk.chunk_id, trace_name, "trace"))
    trace: List[Any] = []
    global_index = 0
    for phase_id, (workload_name, phase_length) in enumerate(phases):
        weights = base.WORKLOADS[workload_name]
        classes = [qc for qc in e1.QUERY_CLASSES if weights.get(qc, 0.0) > 0.0]
        probabilities = np.asarray([weights[qc] for qc in classes], dtype=float)
        probabilities /= probabilities.sum()
        selected_classes = rng.choice(classes, size=phase_length, p=probabilities)
        for query_class in selected_classes:
            pool = query_pool[str(query_class)]
            query = pool[int(rng.integers(0, len(pool)))]
            trace.append(e1.TraceItem(global_index, phase_id, workload_name, query))
            global_index += 1
    return trace


def bootstrap_median_ci(values: np.ndarray, seed: int, iterations: int = 3000) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=float)
    for i in range(iterations):
        samples[i] = np.median(rng.choice(values, size=values.size, replace=True))
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def paired_policy_statistics(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    references = ("fixed_whole_lz4", "initial_best", "best_static", "periodic_greedy", "oracle")
    for (trace, chunk_size), group in run_df.groupby(["trace", "chunk_size"]):
        pivot = group.pivot(
            index=["trace_seed", "chunk_id"], columns="policy", values="total_time_ns"
        ).dropna()
        if "repressdb" not in pivot:
            continue
        for reference in references:
            if reference not in pivot:
                continue
            ratio = pivot["repressdb"].to_numpy(dtype=float) / pivot[reference].to_numpy(dtype=float)
            try:
                p = float(
                    wilcoxon(
                        pivot["repressdb"], pivot[reference], alternative="two-sided", zero_method="wilcox"
                    ).pvalue
                )
            except ValueError:
                p = 1.0
            lo, hi = bootstrap_median_ci(
                ratio, stable_seed(SEED, trace, chunk_size, reference, "bootstrap")
            )
            rows.append(
                {
                    "trace": trace,
                    "chunk_size": int(chunk_size),
                    "reference_policy": reference,
                    "n_paired_runs": int(len(pivot)),
                    "median_ratio_repressdb_over_reference": float(np.median(ratio)),
                    "mean_ratio_repressdb_over_reference": float(np.mean(ratio)),
                    "bootstrap_95ci_median_ratio": f"[{lo:.9g}, {hi:.9g}]",
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "fraction_repressdb_faster": float(np.mean(ratio < 1.0)),
                    "wilcoxon_p": p,
                }
            )
    return pd.DataFrame(rows)


def add_switch_diagnostics(
    e1: Any,
    base: Any,
    decisions_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
) -> pd.DataFrame:
    if decisions_df.empty:
        return decisions_df
    profiles: Dict[str, Dict[str, Dict[str, float]]] = {}
    for chunk_id, group in calibration_df.groupby("chunk_id"):
        profile: Dict[str, Dict[str, float]] = {}
        for codec, local in group.groupby("codec"):
            profile[str(codec)] = {
                str(row.query_class): float(row.median_latency_ns)
                for row in local.itertuples(index=False)
            }
        profiles[str(chunk_id)] = profile

    phase_bounds: Dict[Tuple[str, int], Tuple[int, int]] = {}
    for trace_name, phases in e1.TRACE_SPECS.items():
        start = 0
        for phase_id, (_, length) in enumerate(phases):
            phase_bounds[(trace_name, phase_id)] = (start, start + length)
            start += length

    out = decisions_df.copy()
    phase_best: List[str] = []
    aligned: List[Optional[bool]] = []
    since_start: List[int] = []
    remaining: List[int] = []
    stale_burst: List[bool] = []
    for row in out.itertuples(index=False):
        profile = profiles[str(row.chunk_id)]
        best, _ = e1.choose_best_codec(profile, base.WORKLOADS[str(row.workload)])
        phase_best.append(best)
        is_switch = str(row.decision) == "switch"
        aligned.append(bool(str(row.candidate_codec) == best) if is_switch else None)
        start, end = phase_bounds[(str(row.trace), int(row.phase_id))]
        since_start.append(int(row.query_index) - start)
        remaining.append(end - int(row.query_index) - 1)
        stale_burst.append(
            bool(
                is_switch
                and str(row.policy) == "repressdb"
                and str(row.trace) == "brief_point_burst"
                and int(row.phase_id) == 2
                and str(row.candidate_codec) != best
            )
        )
    out["phase_best_codec"] = phase_best
    out["switch_aligned_with_current_phase"] = aligned
    out["queries_since_phase_start"] = since_start
    out["queries_remaining_in_phase"] = remaining
    out["stale_switch_after_brief_burst"] = stale_burst
    return out


def make_assessment(
    aggregate_df: pd.DataFrame,
    run_df: pd.DataFrame,
    decisions_df: pd.DataFrame,
    paired_df: pd.DataFrame,
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

    paired_lookup = {
        (str(r.trace), int(r.chunk_size), str(r.reference_policy)): r
        for r in paired_df.itertuples(index=False)
    }

    required_improvements = ({
        ("scan_to_point", 8192): 0.95,
        ("scan_to_point", 16384): 0.95,
        ("scan_to_aggregate", 16384): 0.95,
    } if not QUICK_TEST else {})
    stable_gain = all(
        paired_lookup[(trace, size, "initial_best")].median_ratio_repressdb_over_reference <= limit
        for (trace, size), limit in required_improvements.items()
    )
    aggregate_8192_nonregression = (
        True if QUICK_TEST else
        paired_lookup[("scan_to_aggregate", 8192, "initial_best")]
        .median_ratio_repressdb_over_reference <= 1.05
    )
    oracle_gap_ok = (
        True if QUICK_TEST else all(
            paired_lookup[(trace, size, "oracle")].median_ratio_repressdb_over_reference <= 1.30
            for trace in ("scan_to_point", "scan_to_aggregate")
            for size in (8192, 16384)
        )
    )

    anti_keys = [
        key for key in comparisons
        if key.startswith("cyclic_scan_point|") or key.startswith("brief_point_burst|")
    ]
    anti_thrashing = all(
        comparisons[key]["repressdb_median_migrations"]
        <= comparisons[key]["periodic_greedy_median_migrations"]
        for key in anti_keys
    )

    repress_switches = decisions_df[
        (decisions_df["policy"] == "repressdb") & (decisions_df["decision"] == "switch")
    ]
    stale_count = int(repress_switches["stale_switch_after_brief_burst"].sum())
    aligned_fraction = (
        float(repress_switches["switch_aligned_with_current_phase"].astype(bool).mean())
        if not repress_switches.empty else 1.0
    )
    stale_gate = stale_count == 0
    alignment_gate = aligned_fraction >= 0.95

    # Each trace seed must independently improve on initial-best for the strongest stable drifts.
    pivot = run_df.pivot(
        index=["trace_seed", "chunk_id", "trace", "chunk_size"],
        columns="policy",
        values="total_time_ns",
    ).dropna()
    per_seed: Dict[str, Any] = {}
    seed_robust = True
    for trace_seed in sorted(run_df["trace_seed"].unique()):
        seed_rows: Dict[str, float] = {}
        seed_checks = (
            (("scan_to_point", 8192), ("scan_to_point", 16384), ("scan_to_aggregate", 16384))
            if not QUICK_TEST else (("scan_to_point", 2048),)
        )
        for trace, size in seed_checks:
            idx = pivot.index
            mask = (
                (idx.get_level_values("trace_seed") == trace_seed)
                & (idx.get_level_values("trace") == trace)
                & (idx.get_level_values("chunk_size") == size)
            )
            local = pivot.loc[mask]
            ratio = float(np.median(local["repressdb"] / local["initial_best"]))
            seed_rows[f"{trace}|{size}"] = ratio
            seed_robust = seed_robust and ratio < 1.0
        per_seed[str(int(trace_seed))] = seed_rows

    gates = {
        "G1_stable_large_drift_gain": bool(stable_gain and aggregate_8192_nonregression),
        "G2_within_30_percent_of_oracle_on_stable_large_chunks": bool(oracle_gap_ok),
        "G3_no_more_migrations_than_periodic_on_cyclic_and_burst": bool(anti_thrashing),
        "G4_no_stale_post_burst_switches": bool(stale_gate),
        "G5_at_least_95_percent_switches_align_with_current_phase": bool(alignment_gate),
        "G6_stable_gain_repeats_in_all_trace_seeds": bool(seed_robust),
    }

    return {
        "version": VERSION,
        "target_venue": TARGET_VENUE,
        "dataset_label": dataset_meta["dataset_label"],
        "status": "PEMS04 multi-seed development experiment",
        "primary_execution_mode": "warm_slicedmeta",
        "controller_change": (
            "Long-window and short-window best codecs must agree; benefit is computed conservatively "
            "from the smaller predicted gain across the two windows."
        ),
        "online_information_boundary": (
            "RePressDB and periodic-greedy use only observed query classes. Oracle alone knows phase "
            "boundaries and future phase lengths. Phase labels are used only for post-hoc diagnostics."
        ),
        "development_gates": {**gates, "all_six": bool(all(gates.values()))},
        "comparisons": comparisons,
        "switch_diagnostics": {
            "n_repressdb_switches": int(len(repress_switches)),
            "switch_alignment_fraction": aligned_fraction,
            "stale_post_burst_switch_count": stale_count,
        },
        "per_trace_seed_stable_ratios_vs_initial_best": per_seed,
        "important_limitations": [
            "PEMS04 is the development dataset; controller constants may be frozen only after this run.",
            "Queries execute in warm_slicedmeta mode; this is not a cold-SSD claim.",
            "The cost profile is calibrated per chunk before replay.",
            "Migration time includes full decode plus target encode and fsync; candidate files are prebuilt for replay.",
        ],
        "next_gate": (
            "If all six development gates pass, freeze v0.2.0 and run the unchanged PEMS08 holdout version."
        ),
    }


def write_json(path: Path, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=True)


def main() -> int:
    e1 = load_e1_module()
    base = e1.load_base_module()
    decide_repressdb_two_timescale.e1 = e1
    e1.decide_repressdb = decide_repressdb_two_timescale

    print("=" * 92)
    print(f"RePressDB E1-A v{VERSION} — PEMS04 multi-seed online adaptation")
    print(f"Target: {TARGET_VENUE}; primary mode: warm_slicedmeta")
    print("Two-timescale controller: long window 300 + short window 100")
    print("Direct run: no command-line arguments are needed.")
    print("=" * 92)

    output_dir = Path("results_e1_v020_quick_smoke" if QUICK_TEST else OUTPUT_DIR_NAME)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    storage_dir = output_dir / "_storage"
    migration_dir = output_dir / "_migration_trials"
    storage_dir.mkdir(parents=True, exist_ok=True)
    migration_dir.mkdir(parents=True, exist_ok=True)

    if QUICK_TEST:
        matrix, dataset_meta = base.synthetic_dataset(SEED, 3)
        dataset_label = "SYNTHETIC_E1_V020_SMOKE"
        local_chunk_sizes = [2048]
        local_max_chunks = 2
        trace_seeds = (SEED,)
    else:
        pems04_dir, _ = base.default_data_dirs()
        data_file = base.auto_detect_data_file(pems04_dir, "PEMS04_E1_ONLINE_MULTI_SEED")
        matrix, dataset_meta = base.load_dataset(
            data_file, feature_index=0, max_series=e1.MAX_SERIES, seed=SEED, excluded_series=None
        )
        dataset_label = "PEMS04_E1_ONLINE_MULTI_SEED"
        local_chunk_sizes = e1.CHUNK_SIZES
        local_max_chunks = e1.MAX_CHUNKS_PER_SIZE
        trace_seeds = TRACE_SEEDS

    chunks = base.extract_chunks(matrix, local_chunk_sizes, local_max_chunks, SEED)
    reps = base.representation_registry(3)
    print(
        f"[{dataset_label}] chunks={len(chunks)}, trace_seeds={len(trace_seeds)}, "
        f"traces={len(e1.TRACE_SPECS)}, policies={len(e1.POLICIES)}"
    )

    calibration_rows: List[Dict[str, Any]] = []
    codec_rows: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []
    timeline_rows: List[Dict[str, Any]] = []
    phase_rows: List[Dict[str, Any]] = []

    for chunk_index, chunk in enumerate(chunks, start=1):
        chunk_dir = storage_dir / chunk.chunk_id
        stored_map, encode_ns, decode_ns = e1.encode_candidates(base, reps, chunk, chunk_dir)
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
            1 if QUICK_TEST else e1.WINDOWS_PER_FRACTION,
        )
        query_pool = {
            query_class: [q for q in queries if q.query_class == query_class]
            for query_class in e1.QUERY_CLASSES
        }
        if any(not values for values in query_pool.values()):
            raise AssertionError(f"Incomplete query pool: {chunk.chunk_id}")

        profile, local_calibration = e1.calibrate_chunk(
            base, reps, stored_map, chunk, query_pool
        )
        calibration_rows.extend(local_calibration)

        for trace_seed in trace_seeds:
            for trace_name in e1.TRACE_SPECS:
                trace = build_trace(e1, base, chunk, trace_name, query_pool, int(trace_seed))
                order = list(e1.POLICIES)
                random.Random(
                    stable_seed(trace_seed, chunk.chunk_id, trace_name, "policy_order")
                ).shuffle(order)
                for policy in order:
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
                        policy,
                        migration_dir / chunk.chunk_id,
                    )
                    summary["trace_seed"] = int(trace_seed)
                    for row in decisions:
                        row["trace_seed"] = int(trace_seed)
                    for row in phases:
                        row["trace_seed"] = int(trace_seed)
                    for row in timeline:
                        row["trace_seed"] = int(trace_seed)
                    run_rows.append(summary)
                    decision_rows.extend(decisions)
                    phase_rows.extend(phases)
                    if policy in {"repressdb", "periodic_greedy", "oracle"}:
                        timeline_rows.extend(timeline)

        if (
            chunk_index == 1
            or chunk_index == len(chunks)
            or chunk_index % max(1, len(chunks) // 10) == 0
        ):
            print(f"[{dataset_label}] {chunk_index}/{len(chunks)} chunks complete")

    run_df = pd.DataFrame(run_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    codec_df = pd.DataFrame(codec_rows)
    decisions_df = pd.DataFrame(decision_rows)
    timeline_df = pd.DataFrame(timeline_rows)
    phases_df = pd.DataFrame(phase_rows)
    decisions_df = add_switch_diagnostics(e1, base, decisions_df, calibration_df)
    aggregate_df = e1.aggregate_results(run_df)

    for reference_policy, column_name in [
        ("repressdb", "ratio_to_repressdb"),
        ("oracle", "ratio_to_oracle"),
    ]:
        reference = aggregate_df[aggregate_df["policy"] == reference_policy][
            ["trace", "chunk_size", "median_total_time_ns"]
        ].rename(columns={"median_total_time_ns": "reference_time_ns"})
        aggregate_df = aggregate_df.merge(reference, on=["trace", "chunk_size"], how="left")
        aggregate_df[column_name] = (
            aggregate_df["median_total_time_ns"] / aggregate_df["reference_time_ns"]
        )
        aggregate_df = aggregate_df.drop(columns=["reference_time_ns"])

    paired_df = paired_policy_statistics(run_df)

    dataset_meta = dict(dataset_meta)
    dataset_meta.update(
        {
            "dataset_label": dataset_label,
            "num_chunks": len(chunks),
            "chunk_sizes": local_chunk_sizes,
            "max_chunks_per_size": local_max_chunks,
            "trace_seeds": list(trace_seeds),
            "trace_names": list(e1.TRACE_SPECS),
            "policies": list(e1.POLICIES),
            "primary_execution_mode": "warm_slicedmeta",
            "target_venue": TARGET_VENUE,
        }
    )
    assessment = make_assessment(
        aggregate_df, run_df, decisions_df, paired_df, dataset_meta
    )

    calibration_df.to_csv(output_dir / "calibration_records.csv", index=False)
    codec_df.to_csv(output_dir / "codec_records.csv", index=False)
    run_df.to_csv(output_dir / "policy_run_records.csv", index=False)
    aggregate_df.to_csv(output_dir / "policy_aggregate_summary.csv", index=False)
    paired_df.to_csv(output_dir / "policy_paired_comparisons.csv", index=False)
    decisions_df.to_csv(output_dir / "decision_records.csv", index=False)
    timeline_df.to_csv(output_dir / "timeline_sample.csv", index=False)
    phases_df.to_csv(output_dir / "phase_records.csv", index=False)
    write_json(output_dir / "dataset_summary.json", dataset_meta)
    write_json(output_dir / "e1_assessment.json", assessment)

    config = {
        "version": VERSION,
        "target_venue": TARGET_VENUE,
        "base_e1_version": BASE_E1_VERSION,
        "base_protocol": e1.BASE_PROTOCOL,
        "seed": SEED,
        "trace_seeds": list(trace_seeds),
        "chunk_sizes": local_chunk_sizes,
        "max_series": 3 if QUICK_TEST else e1.MAX_SERIES,
        "max_chunks_per_size": local_max_chunks,
        "windows_per_fraction": 1 if QUICK_TEST else e1.WINDOWS_PER_FRACTION,
        "calibration_queries_per_class": (
            1 if QUICK_TEST else e1.CALIBRATION_QUERIES_PER_CLASS
        ),
        "calibration_repeats": 1 if QUICK_TEST else e1.CALIBRATION_REPEATS,
        "decision_interval": 20 if QUICK_TEST else e1.DECISION_INTERVAL,
        "long_observation_window": e1.OBSERVATION_WINDOW,
        "short_observation_window": SHORT_OBSERVATION_WINDOW,
        "min_observations": 30 if QUICK_TEST else e1.MIN_OBSERVATIONS,
        "confirmation_rounds": e1.CONFIRMATION_ROUNDS,
        "cooldown_queries": e1.COOLDOWN_QUERIES,
        "prediction_horizon": e1.PREDICTION_HORIZON,
        "min_relative_improvement": e1.MIN_RELATIVE_IMPROVEMENT,
        "rewrite_safety_factor": e1.REWRITE_SAFETY_FACTOR,
        "trace_specs": e1.TRACE_SPECS,
        "policies": e1.POLICIES,
        "representations": list(reps),
        "workloads": base.WORKLOADS,
        "controller_rule": (
            "Long-window and short-window candidate codecs must agree. Relative improvement and "
            "predicted saving use the conservative minimum across both windows."
        ),
        "measurement_scope": (
            "Time-ordered file-backed replay in warm_slicedmeta mode. Query latency is measured "
            "with perf_counter_ns. Each migration measures full decode plus target encode and fsync."
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
    write_json(output_dir / "run_config.json", config)
    write_json(
        Path("E1_V020_QUICK_ASSESSMENT.json" if QUICK_TEST else SUMMARY_JSON_NAME),
        assessment,
    )

    shutil.rmtree(storage_dir, ignore_errors=True)
    shutil.rmtree(migration_dir, ignore_errors=True)

    print("-" * 92)
    print(f"Development gates: {assessment['development_gates']}")
    print(f"Switch diagnostics: {assessment['switch_diagnostics']}")
    print(f"Results: {output_dir.resolve()}")
    print("Done. Upload e1_assessment.json, policy_paired_comparisons.csv, policy_aggregate_summary.csv,")
    print("policy_run_records.csv, decision_records.csv, phase_records.csv, and run_config.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
