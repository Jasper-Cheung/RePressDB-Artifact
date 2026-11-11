#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RePressDB E2 — rebuild the corrected ablation assessment.

Run this file directly in PyCharm. No command-line arguments are required.

Expected result directory beside this script:
    results_e2_pems04_ablation_v030/
        e2_assessment.json
        policy_switch_diagnostics.csv
        policy_paired_comparisons_series_clustered.csv

This script does NOT rerun the benchmark and does NOT overwrite the original
assessment. It corrects the A3 gate, which must use actual switch/stale-switch
counts rather than only the median migration count.

Outputs:
    results_e2_pems04_ablation_v030/
        e2_assessment_corrected.json
        e2_ablation_mechanism_summary.csv
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import pandas as pd

RESULT_DIR_NAME = "results_e2_pems04_ablation_v030"


def find_result_dir() -> Path:
    here = Path(__file__).resolve().parent
    preferred = here / RESULT_DIR_NAME
    if preferred.exists():
        return preferred

    required = {
        "e2_assessment.json",
        "policy_switch_diagnostics.csv",
        "policy_paired_comparisons_series_clustered.csv",
    }
    if required.issubset({p.name for p in here.iterdir() if p.is_file()}):
        return here

    candidates = []
    for p in here.rglob("e2_assessment.json"):
        parent = p.parent
        if (
            (parent / "policy_switch_diagnostics.csv").exists()
            and (parent / "policy_paired_comparisons_series_clustered.csv").exists()
        ):
            candidates.append(parent)

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"找不到 {RESULT_DIR_NAME} 及其结果文件。"
            "请把本脚本放在该结果目录的上一级文件夹。"
        )
    raise RuntimeError(f"找到多个候选结果目录，请只保留一个：{candidates}")


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=True)


def get_diag(
    diagnostics: pd.DataFrame,
    trace: str,
    chunk_size: int,
    policy: str,
) -> pd.Series:
    local = diagnostics[
        (diagnostics["trace"] == trace)
        & (diagnostics["chunk_size"] == chunk_size)
        & (diagnostics["policy"] == policy)
    ]
    if len(local) != 1:
        raise ValueError(
            f"诊断记录不唯一：trace={trace}, chunk_size={chunk_size}, policy={policy}"
        )
    return local.iloc[0]


def get_comparison(
    comparisons: pd.DataFrame,
    trace: str,
    chunk_size: int,
    policy: str,
    reference: str,
) -> pd.Series:
    local = comparisons[
        (comparisons["trace"] == trace)
        & (comparisons["chunk_size"] == chunk_size)
        & (comparisons["policy"] == policy)
        & (comparisons["reference_policy"] == reference)
    ]
    if len(local) != 1:
        raise ValueError(
            "配对比较记录不唯一："
            f"trace={trace}, chunk_size={chunk_size}, "
            f"policy={policy}, reference={reference}"
        )
    return local.iloc[0]


def main() -> int:
    result_dir = find_result_dir()
    original = read_json(result_dir / "e2_assessment.json")
    diagnostics = pd.read_csv(result_dir / "policy_switch_diagnostics.csv")
    comparisons = pd.read_csv(
        result_dir / "policy_paired_comparisons_series_clustered.csv"
    )

    required_diag = {
        "trace",
        "chunk_size",
        "policy",
        "median_migration_count",
        "mean_migration_count",
        "n_switches",
        "within_2_percent_phase_best_fraction",
        "stale_post_burst_switch_count",
    }
    missing_diag = required_diag.difference(diagnostics.columns)
    if missing_diag:
        raise ValueError(
            f"policy_switch_diagnostics.csv 缺少列：{sorted(missing_diag)}"
        )

    required_cmp = {
        "trace",
        "chunk_size",
        "policy",
        "reference_policy",
        "n_independent_series",
        "median_ratio_policy_over_reference",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "fraction_policy_faster",
        "wilcoxon_p",
    }
    missing_cmp = required_cmp.difference(comparisons.columns)
    if missing_cmp:
        raise ValueError(
            "policy_paired_comparisons_series_clustered.csv 缺少列："
            f"{sorted(missing_cmp)}"
        )

    full_burst = {
        size: get_diag(
            diagnostics, "brief_point_burst", size, "full_repressdb"
        )
        for size in (8192, 16384)
    }
    no_short_burst = {
        size: get_diag(
            diagnostics, "brief_point_burst", size, "no_short_window"
        )
        for size in (8192, 16384)
    }
    no_confirmation_burst = {
        size: get_diag(
            diagnostics, "brief_point_burst", size, "no_confirmation"
        )
        for size in (8192, 16384)
    }

    # Correct A3:
    # The original assessment used only the median migration count.
    # A component can expose a real risk even when fewer than half of runs switch,
    # which leaves the median at zero. Use actual switch and stale-switch counts.
    full_avoids_burst = all(
        int(row["n_switches"]) == 0
        and int(row["stale_post_burst_switch_count"]) == 0
        for row in full_burst.values()
    )
    no_short_exposes_risk = any(
        int(row["n_switches"]) > 0
        or int(row["stale_post_burst_switch_count"]) > 0
        for row in no_short_burst.values()
    )

    corrected_gates = dict(original.get("ablation_gates", {}))
    corrected_gates.pop("all_four", None)
    corrected_gates["A2_full_avoids_brief_burst_migrations"] = bool(
        full_avoids_burst
    )
    corrected_gates["A3_no_short_window_exposes_burst_risk"] = bool(
        full_avoids_burst and no_short_exposes_risk
    )
    corrected_gates["all_four"] = bool(all(corrected_gates.values()))

    rows = []

    for size in (8192, 16384):
        full_row = full_burst[size]
        short_row = no_short_burst[size]
        confirm_row = no_confirmation_burst[size]

        cmp_short = get_comparison(
            comparisons,
            "brief_point_burst",
            size,
            "no_short_window",
            "full_repressdb",
        )
        cmp_confirm = get_comparison(
            comparisons,
            "brief_point_burst",
            size,
            "no_confirmation",
            "full_repressdb",
        )

        rows.append(
            {
                "mechanism": "short_window",
                "trace": "brief_point_burst",
                "chunk_size": size,
                "full_n_switches": int(full_row["n_switches"]),
                "ablated_policy": "no_short_window",
                "ablated_n_switches": int(short_row["n_switches"]),
                "ablated_mean_migrations": float(
                    short_row["mean_migration_count"]
                ),
                "ablated_stale_switches": int(
                    short_row["stale_post_burst_switch_count"]
                ),
                "median_time_ratio_ablated_over_full": float(
                    cmp_short["median_ratio_policy_over_reference"]
                ),
                "bootstrap_ci_low": float(cmp_short["bootstrap_ci_low"]),
                "bootstrap_ci_high": float(cmp_short["bootstrap_ci_high"]),
                "wilcoxon_p": float(cmp_short["wilcoxon_p"]),
            }
        )
        rows.append(
            {
                "mechanism": "confirmation",
                "trace": "brief_point_burst",
                "chunk_size": size,
                "full_n_switches": int(full_row["n_switches"]),
                "ablated_policy": "no_confirmation",
                "ablated_n_switches": int(confirm_row["n_switches"]),
                "ablated_mean_migrations": float(
                    confirm_row["mean_migration_count"]
                ),
                "ablated_stale_switches": int(
                    confirm_row["stale_post_burst_switch_count"]
                ),
                "median_time_ratio_ablated_over_full": float(
                    cmp_confirm["median_ratio_policy_over_reference"]
                ),
                "bootstrap_ci_low": float(cmp_confirm["bootstrap_ci_low"]),
                "bootstrap_ci_high": float(cmp_confirm["bootstrap_ci_high"]),
                "wilcoxon_p": float(cmp_confirm["wilcoxon_p"]),
            }
        )

    # Add the main adaptation comparisons and the two components that were
    # effectively inactive under these traces.
    for size in (8192, 16384):
        for policy in (
            "full_repressdb",
            "no_confirmation",
            "no_break_even",
            "no_cooldown",
        ):
            reference = (
                "initial_best"
                if policy == "full_repressdb"
                else "full_repressdb"
            )
            cmp_row = get_comparison(
                comparisons,
                "scan_to_point",
                size,
                policy,
                reference,
            )
            rows.append(
                {
                    "mechanism": (
                        "full_controller"
                        if policy == "full_repressdb"
                        else policy.removeprefix("no_")
                    ),
                    "trace": "scan_to_point",
                    "chunk_size": size,
                    "full_n_switches": math.nan,
                    "ablated_policy": policy,
                    "ablated_n_switches": math.nan,
                    "ablated_mean_migrations": math.nan,
                    "ablated_stale_switches": math.nan,
                    "median_time_ratio_ablated_over_full": float(
                        cmp_row["median_ratio_policy_over_reference"]
                    ),
                    "bootstrap_ci_low": float(cmp_row["bootstrap_ci_low"]),
                    "bootstrap_ci_high": float(cmp_row["bootstrap_ci_high"]),
                    "wilcoxon_p": float(cmp_row["wilcoxon_p"]),
                }
            )

    mechanism_df = pd.DataFrame(rows)
    mechanism_df.to_csv(
        result_dir / "e2_ablation_mechanism_summary.csv", index=False
    )

    corrected = dict(original)
    corrected["assessment_revision"] = "0.3.1-statistical-rebuild"
    corrected["original_experiment_version"] = original.get("version", "0.3.0")
    corrected["experiment_rerun"] = False
    corrected["ablation_gates"] = corrected_gates
    corrected["corrected_A3_definition"] = (
        "The full controller must have zero burst switches, while removing "
        "the short-window veto exposes at least one actual or stale post-burst "
        "switch. Median migration count alone is insufficient because a "
        "minority of affected runs can still leave the median at zero."
    )
    corrected["short_window_diagnostics"] = {
        str(size): {
            "full_n_switches": int(full_burst[size]["n_switches"]),
            "no_short_window_n_switches": int(
                no_short_burst[size]["n_switches"]
            ),
            "no_short_window_mean_migration_count": float(
                no_short_burst[size]["mean_migration_count"]
            ),
            "no_short_window_stale_post_burst_switch_count": int(
                no_short_burst[size]["stale_post_burst_switch_count"]
            ),
        }
        for size in (8192, 16384)
    }
    corrected["confirmation_tradeoff"] = {
        "finding": (
            "Removing confirmation accelerates sustained and cyclic adaptation, "
            "but increases unnecessary burst migrations and significantly harms "
            "the 16384 brief-burst case."
        ),
        "brief_burst_8192": {
            "n_switches": int(no_confirmation_burst[8192]["n_switches"]),
            "stale_switches": int(
                no_confirmation_burst[8192][
                    "stale_post_burst_switch_count"
                ]
            ),
            "ratio_vs_full": float(
                get_comparison(
                    comparisons,
                    "brief_point_burst",
                    8192,
                    "no_confirmation",
                    "full_repressdb",
                )["median_ratio_policy_over_reference"]
            ),
        },
        "brief_burst_16384": {
            "n_switches": int(no_confirmation_burst[16384]["n_switches"]),
            "stale_switches": int(
                no_confirmation_burst[16384][
                    "stale_post_burst_switch_count"
                ]
            ),
            "ratio_vs_full": float(
                get_comparison(
                    comparisons,
                    "brief_point_burst",
                    16384,
                    "no_confirmation",
                    "full_repressdb",
                )["median_ratio_policy_over_reference"]
            ),
            "wilcoxon_p": float(
                get_comparison(
                    comparisons,
                    "brief_point_burst",
                    16384,
                    "no_confirmation",
                    "full_repressdb",
                )["wilcoxon_p"]
            ),
        },
    }
    corrected["inactive_or_weak_components_under_current_traces"] = {
        "break_even": (
            "Removing break-even changes median total time by about 1% or less "
            "in the tested traces; these traces do not isolate its value."
        ),
        "cooldown": (
            "Removing cooldown changes median total time by about 1% or less; "
            "confirmation and dual-window agreement already suppress most "
            "rapid reversals in this workload set."
        ),
    }
    corrected["paper_interpretation"] = [
        "The full controller significantly improves sustained scan-to-point drift.",
        "Dual-window agreement prevents a subset of stale post-burst switches.",
        "Three-round confirmation trades some adaptation speed for burst stability.",
        "Break-even and cooldown are not empirically identifiable from this ablation alone and must not be claimed as independently essential.",
    ]

    write_json(result_dir / "e2_assessment_corrected.json", corrected)

    print("=" * 88)
    print("RePressDB E2 corrected assessment")
    print("=" * 88)
    print(f"Result directory: {result_dir.resolve()}")
    print(f"Corrected gates: {corrected_gates}")
    print(
        "Generated: e2_assessment_corrected.json, "
        "e2_ablation_mechanism_summary.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
