#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RePressDB E2 v0.3.0 — one-click controller ablation on PEMS04.

Target venue: EDBT 2027 (CCF-B).

Run this file directly in PyCharm. No command-line arguments are required.
Keep these two dependency files in the same directory:
  1) RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT.py
  2) RUN_THIS_ONLY_REPRESSDB_E0C_V0_2_0.py

Purpose
-------
This experiment explains why each online-controller component is needed.
It does not retune the frozen full RePressDB controller.

Ablations
---------
  * full_repressdb: frozen E1 v0.2 controller.
  * no_short_window: removes the short-window agreement veto.
  * no_confirmation: switches after one qualifying decision instead of three.
  * no_break_even: ignores measured rewrite-cost amortization.
  * no_cooldown: removes the 300-query post-switch cooldown.
  * periodic_greedy: immediately adopts the recent-window best layout.
  * initial_best: chooses from the first workload and never migrates.

Primary traces
--------------
  * scan_to_point: sustained one-way drift; measures adaptation benefit.
  * cyclic_scan_point: repeated drift; measures thrashing/adaptation delay.
  * brief_point_burst: transient drift; measures unnecessary migrations.

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

VERSION = "0.3.0"
TARGET_VENUE = "EDBT 2027"
BASE_E1_VERSION = "0.1.0"
SEED = 20260718
TRACE_SEEDS = (20260718, 20260719, 20260720)
SHORT_OBSERVATION_WINDOW = 100

CHUNK_SIZES = [8192, 16384]
MAX_SERIES = 24
MAX_CHUNKS_PER_SIZE = 12
TRACE_NAMES = ("scan_to_point", "cyclic_scan_point", "brief_point_burst")

ABLATION_POLICIES = (
    "initial_best",
    "periodic_greedy",
    "full_repressdb",
    "no_short_window",
    "no_confirmation",
    "no_break_even",
    "no_cooldown",
)

OUTPUT_DIR_NAME = "results_e2_pems04_ablation_v030"
SUMMARY_JSON_NAME = "E2_PEMS04_ABLATION_ASSESSMENT_v030.json"
QUICK_TEST = os.environ.get("REPRESSDB_E2_QUICK", "0") == "1"


def stable_seed(*parts: Any) -> int:
    token = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def load_e1_module() -> Any:
    here = Path(__file__).resolve().parent
    path = here / "RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT.py"
    if not path.exists():
        raise FileNotFoundError(
            "找不到基础代码 RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT.py。"
            "请把它与本文件放在同一文件夹。"
        )
    spec = importlib.util.spec_from_file_location("repressdb_e1_ablation_base", path)
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


def observed_weights(query_classes: Iterable[str], all_classes: Sequence[str]) -> Dict[str, float]:
    counts = Counter(query_classes)
    total = sum(counts.values())
    if total <= 0:
        return {qc: 0.0 for qc in all_classes}
    return {qc: counts.get(qc, 0) / total for qc in all_classes}


def controller_options(policy_variant: str, e1: Any) -> Dict[str, Any]:
    options = {
        "use_short_window": True,
        "confirmation_rounds": int(e1.CONFIRMATION_ROUNDS),
        "use_break_even": True,
        "cooldown_queries": int(e1.COOLDOWN_QUERIES),
    }
    if policy_variant == "no_short_window":
        options["use_short_window"] = False
    elif policy_variant == "no_confirmation":
        options["confirmation_rounds"] = 1
    elif policy_variant == "no_break_even":
        options["use_break_even"] = False
    elif policy_variant == "no_cooldown":
        options["cooldown_queries"] = 0
    elif policy_variant != "full_repressdb":
        raise ValueError(f"Unknown RePressDB variant: {policy_variant}")
    return options


def decide_ablation(
    profile: Mapping[str, Mapping[str, float]],
    state: Any,
    query_index: int,
    encode_ns: Mapping[str, int],
    decode_ns: Mapping[str, int],
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Shared controller with one component removed according to ACTIVE_VARIANT."""
    e1 = decide_ablation.e1
    variant = decide_ablation.active_variant
    options = controller_options(variant, e1)

    recent_long = list(state.recent_classes)
    long_weights = observed_weights(recent_long, e1.QUERY_CLASSES)
    long_target, long_target_cost = e1.choose_best_codec(profile, long_weights)
    current_long_cost = e1.workload_cost(profile, state.active_codec, long_weights)
    long_relative = (
        (current_long_cost - long_target_cost) / current_long_cost
        if current_long_cost > 0 else 0.0
    )
    long_gain = max(0.0, current_long_cost - long_target_cost)

    target = long_target
    target_cost = long_target_cost
    conservative_relative = long_relative
    conservative_gain = long_gain
    short_target: Optional[str] = None
    short_relative = math.nan

    if options["use_short_window"]:
        recent_short = recent_long[-SHORT_OBSERVATION_WINDOW:]
        short_weights = observed_weights(recent_short, e1.QUERY_CLASSES)
        short_target, short_target_cost = e1.choose_best_codec(profile, short_weights)
        current_short_cost = e1.workload_cost(profile, state.active_codec, short_weights)
        short_relative = (
            (current_short_cost - short_target_cost) / current_short_cost
            if current_short_cost > 0 else 0.0
        )
        conservative_relative = min(long_relative, short_relative)
        conservative_gain = max(
            0.0,
            min(
                current_long_cost - long_target_cost,
                current_short_cost - short_target_cost,
            ),
        )

    rewrite_est = e1.migration_estimate_ns(state.active_codec, target, encode_ns, decode_ns)
    predicted_saving = conservative_gain * e1.PREDICTION_HORIZON

    detail = {
        "current_expected_ns": current_long_cost,
        "target_expected_ns": target_cost,
        "relative_improvement": conservative_relative,
        "predicted_saving_ns": predicted_saving,
        "rewrite_estimate_ns": rewrite_est,
        "reason": "",
        "ablation_variant": variant,
        "long_target_codec": long_target,
        "short_target_codec": short_target,
        "long_relative_improvement": long_relative,
        "short_relative_improvement": short_relative,
    }

    if options["use_short_window"] and long_target != short_target:
        state.pending_candidate = None
        state.pending_rounds = 0
        detail["reason"] = "timescale_disagreement"
        return None, detail

    if target == state.active_codec:
        state.pending_candidate = None
        state.pending_rounds = 0
        detail["reason"] = "current_layout_is_best"
        return None, detail

    if query_index - state.last_switch_query < int(options["cooldown_queries"]):
        state.pending_candidate = None
        state.pending_rounds = 0
        detail["reason"] = "cooldown"
        return None, detail

    if conservative_relative < e1.MIN_RELATIVE_IMPROVEMENT:
        state.pending_candidate = None
        state.pending_rounds = 0
        detail["reason"] = "relative_gain_too_small"
        return None, detail

    if options["use_break_even"] and (
        predicted_saving < e1.REWRITE_SAFETY_FACTOR * rewrite_est
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

    if state.pending_rounds < int(options["confirmation_rounds"]):
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
        phases = [(name, max(60, length // 25)) for name, length in phases]
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


def run_named_policy(
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
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if policy_name in {"initial_best", "periodic_greedy"}:
        base_policy = policy_name
    else:
        base_policy = "repressdb"
        decide_ablation.active_variant = policy_name

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

    if base_policy == "repressdb":
        summary["policy"] = policy_name
        for row in decisions:
            row["policy"] = policy_name
            row["ablation_variant"] = policy_name
        for row in timeline:
            row["policy"] = policy_name
        for row in phases:
            row["policy"] = policy_name
    else:
        for row in decisions:
            row["ablation_variant"] = policy_name
    return summary, decisions, timeline, phases


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
    for trace_name in TRACE_NAMES:
        start = 0
        for phase_id, (_, length) in enumerate(e1.TRACE_SPECS[trace_name]):
            phase_bounds[(trace_name, phase_id)] = (start, start + length)
            start += length

    out = decisions_df.copy()
    phase_best: List[str] = []
    aligned: List[Optional[bool]] = []
    near_optimal_ratio: List[float] = []
    stale_burst: List[bool] = []
    for row in out.itertuples(index=False):
        profile = profiles[str(row.chunk_id)]
        weights = base.WORKLOADS[str(row.workload)]
        best, best_cost = e1.choose_best_codec(profile, weights)
        candidate = str(row.candidate_codec)
        candidate_cost = e1.workload_cost(profile, candidate, weights)
        ratio = candidate_cost / best_cost if best_cost > 0 else 1.0
        is_switch = str(row.decision) == "switch"
        phase_best.append(best)
        aligned.append(bool(candidate == best) if is_switch else None)
        near_optimal_ratio.append(float(ratio) if is_switch else math.nan)
        stale_burst.append(
            bool(
                is_switch
                and str(row.trace) == "brief_point_burst"
                and int(row.phase_id) == 2
                and candidate != best
            )
        )
    out["phase_best_codec"] = phase_best
    out["switch_aligned_with_current_phase"] = aligned
    out["candidate_cost_ratio_to_phase_best"] = near_optimal_ratio
    out["stale_switch_after_brief_burst"] = stale_burst
    return out


def aggregate_results(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (trace, chunk_size, policy), group in run_df.groupby(
        ["trace", "chunk_size", "policy"], sort=True
    ):
        rows.append(
            {
                "trace": trace,
                "chunk_size": int(chunk_size),
                "policy": policy,
                "n_runs": int(len(group)),
                "n_chunks": int(group["chunk_id"].nunique()),
                "n_series": int(group["series_index"].nunique()),
                "median_total_time_ns": float(group["total_time_ns"].median()),
                "median_query_time_ns": float(group["query_time_ns"].median()),
                "median_migration_time_ns": float(group["migration_time_ns"].median()),
                "median_migration_count": float(group["migration_count"].median()),
                "p95_migration_time_ns": float(group["migration_time_ns"].quantile(0.95)),
                "median_rewrite_bytes": float(group["rewrite_bytes"].median()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_median_ci(values: np.ndarray, seed: int, iterations: int = 4000) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=float)
    for i in range(iterations):
        samples[i] = np.median(rng.choice(values, size=values.size, replace=True))
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def series_clustered_comparisons(run_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated seeds/chunks within source series, then perform paired inference."""
    rows: List[Dict[str, Any]] = []
    references = ("full_repressdb", "initial_best", "periodic_greedy")

    # First aggregate repeated measurements belonging to the same source series.
    series_df = (
        run_df.groupby(
            ["trace", "chunk_size", "series_index", "policy"], as_index=False
        )["total_time_ns"]
        .median()
    )

    for (trace, chunk_size), group in series_df.groupby(["trace", "chunk_size"]):
        pivot = group.pivot(index="series_index", columns="policy", values="total_time_ns").dropna()
        for policy in ABLATION_POLICIES:
            if policy not in pivot:
                continue
            for reference in references:
                if reference == policy or reference not in pivot:
                    continue
                ratios = pivot[policy].to_numpy(dtype=float) / pivot[reference].to_numpy(dtype=float)
                try:
                    p_value = float(
                        wilcoxon(
                            pivot[policy],
                            pivot[reference],
                            alternative="two-sided",
                            zero_method="wilcox",
                        ).pvalue
                    )
                except ValueError:
                    p_value = 1.0
                lo, hi = bootstrap_median_ci(
                    ratios,
                    stable_seed(SEED, trace, chunk_size, policy, reference, "series_bootstrap"),
                )
                rows.append(
                    {
                        "trace": trace,
                        "chunk_size": int(chunk_size),
                        "policy": policy,
                        "reference_policy": reference,
                        "n_independent_series": int(len(pivot)),
                        "median_ratio_policy_over_reference": float(np.median(ratios)),
                        "mean_ratio_policy_over_reference": float(np.mean(ratios)),
                        "bootstrap_ci_low": lo,
                        "bootstrap_ci_high": hi,
                        "fraction_policy_faster": float(np.mean(ratios < 1.0)),
                        "wilcoxon_p": p_value,
                    }
                )
    return pd.DataFrame(rows)


def policy_diagnostics(run_df: pd.DataFrame, decisions_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if decisions_df.empty:
        decisions_df = pd.DataFrame(
            columns=[
                "trace", "chunk_size", "policy", "decision",
                "switch_aligned_with_current_phase",
                "candidate_cost_ratio_to_phase_best",
                "stale_switch_after_brief_burst",
            ]
        )
    adaptive = [
        "periodic_greedy",
        "full_repressdb",
        "no_short_window",
        "no_confirmation",
        "no_break_even",
        "no_cooldown",
    ]
    for trace in TRACE_NAMES:
        for chunk_size in CHUNK_SIZES:
            for policy in adaptive:
                runs = run_df[
                    (run_df["trace"] == trace)
                    & (run_df["chunk_size"] == chunk_size)
                    & (run_df["policy"] == policy)
                ]
                switches = decisions_df[
                    (decisions_df["trace"] == trace)
                    & (decisions_df["chunk_size"] == chunk_size)
                    & (decisions_df["policy"] == policy)
                    & (decisions_df["decision"] == "switch")
                ]
                rows.append(
                    {
                        "trace": trace,
                        "chunk_size": int(chunk_size),
                        "policy": policy,
                        "n_runs": int(len(runs)),
                        "median_migration_count": float(runs["migration_count"].median()) if not runs.empty else math.nan,
                        "mean_migration_count": float(runs["migration_count"].mean()) if not runs.empty else math.nan,
                        "median_migration_time_ns": float(runs["migration_time_ns"].median()) if not runs.empty else math.nan,
                        "n_switches": int(len(switches)),
                        "exact_phase_best_fraction": (
                            float(switches["switch_aligned_with_current_phase"].astype(bool).mean())
                            if not switches.empty else 1.0
                        ),
                        "within_2_percent_phase_best_fraction": (
                            float((switches["candidate_cost_ratio_to_phase_best"] <= 1.02).mean())
                            if not switches.empty else 1.0
                        ),
                        "stale_post_burst_switch_count": int(
                            switches["stale_switch_after_brief_burst"].sum()
                        ) if not switches.empty else 0,
                    }
                )
    return pd.DataFrame(rows)


def make_assessment(
    aggregate_df: pd.DataFrame,
    comparisons_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    dataset_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    def ratio(trace: str, size: int, policy: str, reference: str) -> float:
        local = comparisons_df[
            (comparisons_df["trace"] == trace)
            & (comparisons_df["chunk_size"] == size)
            & (comparisons_df["policy"] == policy)
            & (comparisons_df["reference_policy"] == reference)
        ]
        return float(local.iloc[0]["median_ratio_policy_over_reference"])

    def migrations(trace: str, size: int, policy: str) -> float:
        local = diagnostics_df[
            (diagnostics_df["trace"] == trace)
            & (diagnostics_df["chunk_size"] == size)
            & (diagnostics_df["policy"] == policy)
        ]
        return float(local.iloc[0]["median_migration_count"])

    component_findings: Dict[str, Any] = {}
    for policy in ("no_short_window", "no_confirmation", "no_break_even", "no_cooldown"):
        component_findings[policy] = {
            "scan_to_point_8192_ratio_vs_full": ratio("scan_to_point", 8192, policy, "full_repressdb"),
            "scan_to_point_16384_ratio_vs_full": ratio("scan_to_point", 16384, policy, "full_repressdb"),
            "cyclic_8192_ratio_vs_full": ratio("cyclic_scan_point", 8192, policy, "full_repressdb"),
            "cyclic_16384_ratio_vs_full": ratio("cyclic_scan_point", 16384, policy, "full_repressdb"),
            "brief_burst_8192_median_migrations": migrations("brief_point_burst", 8192, policy),
            "brief_burst_16384_median_migrations": migrations("brief_point_burst", 16384, policy),
        }

    full_summary: Dict[str, Any] = {}
    for trace in TRACE_NAMES:
        for size in CHUNK_SIZES:
            full_summary[f"{trace}|{size}"] = {
                "ratio_vs_initial_best": ratio(trace, size, "full_repressdb", "initial_best"),
                "ratio_vs_periodic_greedy": ratio(trace, size, "full_repressdb", "periodic_greedy"),
                "median_migrations": migrations(trace, size, "full_repressdb"),
            }

    diagnostic_lookup = {
        (str(r.trace), int(r.chunk_size), str(r.policy)): r
        for r in diagnostics_df.itertuples(index=False)
    }

    gates = {
        "A1_full_improves_scan_to_point_at_both_scales": bool(
            full_summary["scan_to_point|8192"]["ratio_vs_initial_best"] < 1.0
            and full_summary["scan_to_point|16384"]["ratio_vs_initial_best"] < 1.0
        ),
        "A2_full_avoids_brief_burst_migrations": bool(
            migrations("brief_point_burst", 8192, "full_repressdb") == 0.0
            and migrations("brief_point_burst", 16384, "full_repressdb") == 0.0
        ),
        "A3_no_short_window_exposes_burst_risk": bool(
            migrations("brief_point_burst", 8192, "no_short_window")
            + migrations("brief_point_burst", 16384, "no_short_window")
            > 0.0
        ),
        "A4_full_switches_are_near_phase_optimal": bool(
            diagnostic_lookup[("scan_to_point", 8192, "full_repressdb")]
            .within_2_percent_phase_best_fraction >= 0.95
            and diagnostic_lookup[("scan_to_point", 16384, "full_repressdb")]
            .within_2_percent_phase_best_fraction >= 0.95
        ),
    }

    return {
        "version": VERSION,
        "target_venue": TARGET_VENUE,
        "dataset_label": dataset_meta["dataset_label"],
        "status": "PEMS04 controller-component ablation",
        "study_role": (
            "Mechanism ablation after freezing the full E1 v0.2 controller; not used for retuning."
        ),
        "primary_execution_mode": "warm_slicedmeta",
        "statistical_unit": "Source-series clustered paired inference.",
        "ablation_gates": {**gates, "all_four": bool(all(gates.values()))},
        "full_controller_summary": full_summary,
        "component_findings": component_findings,
        "interpretation": {
            "short_window": "Expected to prevent stale migration after a brief workload burst.",
            "confirmation": "Expected to suppress noisy one-round layout changes.",
            "break_even": "Expected to avoid rewrites whose predicted future saving cannot amortize measured migration cost.",
            "cooldown": "Expected to reduce repeated switching after a recent migration.",
        },
        "important_limitations": [
            "PEMS04 is the development dataset; this experiment explains mechanisms rather than providing a new holdout claim.",
            "Only 8192 and 16384 chunks are used because E1 showed the strongest and most relevant adaptation effects there.",
            "Queries execute in warm_slicedmeta mode; this is not a cold-SSD claim.",
            "Migration time includes full decode, target encode, file flush/fsync, and validation is outside the timed interval.",
        ],
        "next_gate": (
            "Use the ablation to select the mechanism story for the paper. Do not retune the frozen full controller from these results."
        ),
    }


def write_json(path: Path, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=True)


def main() -> int:
    e1 = load_e1_module()
    base = e1.load_base_module()
    e1.QUICK_TEST = QUICK_TEST
    decide_ablation.e1 = e1
    e1.decide_repressdb = decide_ablation

    print("=" * 96)
    print(f"RePressDB E2 controller ablation v{VERSION}")
    print(f"Target: {TARGET_VENUE}")
    print("Direct-run PEMS04 experiment; no command-line arguments are required.")
    print("=" * 96)

    output_dir = Path("results_e2_quick_smoke" if QUICK_TEST else OUTPUT_DIR_NAME)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_dir = output_dir / "_storage"
    migration_dir = output_dir / "_migration_trials"
    storage_dir.mkdir(parents=True, exist_ok=True)
    migration_dir.mkdir(parents=True, exist_ok=True)

    if QUICK_TEST:
        matrix, dataset_meta = base.synthetic_dataset(SEED, 4)
        dataset_label = "SYNTHETIC_E2_ABLATION_SMOKE"
        chunk_sizes = [8192, 16384]
        max_chunks_per_size = 1
        trace_names = TRACE_NAMES
        trace_seeds = (SEED,)
    else:
        pems04_dir, _ = base.default_data_dirs()
        data_file = base.auto_detect_data_file(pems04_dir, "PEMS04_E2_ABLATION")
        matrix, dataset_meta = base.load_dataset(
            data_file,
            feature_index=0,
            max_series=MAX_SERIES,
            seed=SEED,
            excluded_series=None,
        )
        dataset_label = "PEMS04_E2_CONTROLLER_ABLATION"
        chunk_sizes = CHUNK_SIZES
        max_chunks_per_size = MAX_CHUNKS_PER_SIZE
        trace_names = TRACE_NAMES
        trace_seeds = TRACE_SEEDS

    chunks = base.extract_chunks(matrix, chunk_sizes, max_chunks_per_size, SEED)
    reps = base.representation_registry(3)
    print(
        f"[{dataset_label}] chunks={len(chunks)}, trace_seeds={len(trace_seeds)}, "
        f"traces={len(trace_names)}, policies={len(ABLATION_POLICIES)}"
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

        profile, local_calibration = e1.calibrate_chunk(base, reps, stored_map, chunk, query_pool)
        calibration_rows.extend(local_calibration)

        for trace_seed in trace_seeds:
            for trace_name in trace_names:
                trace = build_trace(e1, base, chunk, trace_name, query_pool, int(trace_seed))
                policy_order = list(ABLATION_POLICIES)
                random.Random(
                    stable_seed(trace_seed, chunk.chunk_id, trace_name, "ablation_policy_order")
                ).shuffle(policy_order)

                for policy_name in policy_order:
                    summary, decisions, timeline, phases = run_named_policy(
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
                        migration_dir / chunk.chunk_id,
                    )
                    summary["trace_seed"] = int(trace_seed)
                    summary["series_index"] = int(chunk.series_index)
                    for row in decisions:
                        row["trace_seed"] = int(trace_seed)
                        row["series_index"] = int(chunk.series_index)
                    for row in phases:
                        row["trace_seed"] = int(trace_seed)
                        row["series_index"] = int(chunk.series_index)
                    for row in timeline:
                        row["trace_seed"] = int(trace_seed)
                        row["series_index"] = int(chunk.series_index)
                    run_rows.append(summary)
                    decision_rows.extend(decisions)
                    phase_rows.extend(phases)
                    if policy_name in {
                        "full_repressdb",
                        "no_short_window",
                        "periodic_greedy",
                    }:
                        timeline_rows.extend(timeline)

        if (
            chunk_index == 1
            or chunk_index == len(chunks)
            or chunk_index % max(1, len(chunks) // 12) == 0
        ):
            print(f"[{dataset_label}] {chunk_index}/{len(chunks)} chunks complete")

    calibration_df = pd.DataFrame(calibration_rows)
    codec_df = pd.DataFrame(codec_rows)
    run_df = pd.DataFrame(run_rows)
    decisions_df = pd.DataFrame(decision_rows)
    timeline_df = pd.DataFrame(timeline_rows)
    phase_df = pd.DataFrame(phase_rows)

    decisions_df = add_switch_diagnostics(e1, base, decisions_df, calibration_df)
    aggregate_df = aggregate_results(run_df)
    comparisons_df = series_clustered_comparisons(run_df)
    diagnostics_df = policy_diagnostics(run_df, decisions_df)

    dataset_meta = dict(dataset_meta)
    dataset_meta.update(
        {
            "dataset_label": dataset_label,
            "num_chunks": len(chunks),
            "chunk_sizes": chunk_sizes,
            "max_chunks_per_size": max_chunks_per_size,
            "trace_seeds": list(trace_seeds),
            "trace_names": list(trace_names),
            "policies": list(ABLATION_POLICIES),
            "primary_execution_mode": "warm_slicedmeta",
            "target_venue": TARGET_VENUE,
        }
    )
    assessment = make_assessment(
        aggregate_df,
        comparisons_df,
        diagnostics_df,
        dataset_meta,
    )

    calibration_df.to_csv(output_dir / "calibration_records.csv", index=False)
    codec_df.to_csv(output_dir / "codec_records.csv", index=False)
    run_df.to_csv(output_dir / "policy_run_records.csv", index=False)
    aggregate_df.to_csv(output_dir / "policy_aggregate_summary.csv", index=False)
    comparisons_df.to_csv(output_dir / "policy_paired_comparisons_series_clustered.csv", index=False)
    diagnostics_df.to_csv(output_dir / "policy_switch_diagnostics.csv", index=False)
    decisions_df.to_csv(output_dir / "decision_records.csv", index=False)
    timeline_df.to_csv(output_dir / "timeline_sample.csv", index=False)
    phase_df.to_csv(output_dir / "phase_records.csv", index=False)
    write_json(output_dir / "dataset_summary.json", dataset_meta)
    write_json(output_dir / "e2_assessment.json", assessment)

    config = {
        "version": VERSION,
        "target_venue": TARGET_VENUE,
        "base_e1_version": BASE_E1_VERSION,
        "base_protocol": e1.BASE_PROTOCOL,
        "seed": SEED,
        "trace_seeds": list(trace_seeds),
        "chunk_sizes": chunk_sizes,
        "max_series": 4 if QUICK_TEST else MAX_SERIES,
        "max_chunks_per_size": max_chunks_per_size,
        "trace_names": list(trace_names),
        "ablation_policies": list(ABLATION_POLICIES),
        "decision_interval": 20 if QUICK_TEST else e1.DECISION_INTERVAL,
        "long_observation_window": e1.OBSERVATION_WINDOW,
        "short_observation_window": SHORT_OBSERVATION_WINDOW,
        "min_observations": 30 if QUICK_TEST else e1.MIN_OBSERVATIONS,
        "confirmation_rounds_full": e1.CONFIRMATION_ROUNDS,
        "cooldown_queries_full": e1.COOLDOWN_QUERIES,
        "prediction_horizon": e1.PREDICTION_HORIZON,
        "min_relative_improvement": e1.MIN_RELATIVE_IMPROVEMENT,
        "rewrite_safety_factor": e1.REWRITE_SAFETY_FACTOR,
        "trace_specs": {name: e1.TRACE_SPECS[name] for name in trace_names},
        "representations": list(reps),
        "workloads": base.WORKLOADS,
        "statistical_unit": "Source series; trace seeds and chunks within series are repeated measurements.",
        "study_role": "Mechanism ablation only; frozen full controller is not retuned.",
        "measurement_scope": (
            "Time-ordered file-backed replay in warm_slicedmeta mode. Query latency uses "
            "perf_counter_ns. Migration measures full decode plus target encode and fsync."
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
        Path("E2_QUICK_ASSESSMENT.json" if QUICK_TEST else SUMMARY_JSON_NAME),
        assessment,
    )

    shutil.rmtree(storage_dir, ignore_errors=True)
    shutil.rmtree(migration_dir, ignore_errors=True)

    print("-" * 96)
    print(f"Ablation gates: {assessment['ablation_gates']}")
    print(f"Results: {output_dir.resolve()}")
    print("Done. Upload e2_assessment.json, policy_paired_comparisons_series_clustered.csv,")
    print("policy_switch_diagnostics.csv, policy_aggregate_summary.csv, policy_run_records.csv,")
    print("decision_records.csv, phase_records.csv, and run_config.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
