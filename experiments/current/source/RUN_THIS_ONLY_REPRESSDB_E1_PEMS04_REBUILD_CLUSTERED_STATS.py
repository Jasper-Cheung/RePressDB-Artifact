#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuild formal chunk-level paired statistics for the completed PEMS04 E1 v0.2.0 run.

Run directly in PyCharm. No command-line arguments are required.
Expected input:
  results_e1_pems04_online_v020/policy_run_records.csv

The three trace seeds are repeated measurements on the same chunk. Formal
inference therefore uses chunk as the independent unit: for each chunk, the
policy-time ratio is first aggregated across seeds, then bootstrap confidence
intervals and Wilcoxon tests are computed across chunks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SEED = 20260718
RESULT_DIR = Path("results_e1_pems04_online_v020")
INPUT_FILE = RESULT_DIR / "policy_run_records.csv"
OUTPUT_FILE = RESULT_DIR / "policy_paired_comparisons_clustered.csv"
SUMMARY_FILE = RESULT_DIR / "clustered_statistics_summary.json"
REFERENCES = (
    "fixed_whole_lz4",
    "initial_best",
    "best_static",
    "periodic_greedy",
    "oracle",
)


def stable_seed(*parts: Any) -> int:
    token = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def bootstrap_median_ci(values: np.ndarray, seed: int, iterations: int = 10000) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=float)
    for i in range(iterations):
        samples[i] = np.median(rng.choice(values, size=values.size, replace=True))
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> int:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"找不到 {INPUT_FILE}。请把本文件放在包含 results_e1_pems04_online_v020 的项目目录中。"
        )

    run_df = pd.read_csv(INPUT_FILE)
    required = {"chunk_id", "trace_seed", "trace", "chunk_size", "policy", "total_time_ns"}
    missing = sorted(required - set(run_df.columns))
    if missing:
        raise ValueError(f"policy_run_records.csv 缺少列：{missing}")

    rows: List[Dict[str, Any]] = []
    for (trace, chunk_size), group in run_df.groupby(["trace", "chunk_size"]):
        pivot = group.pivot(
            index=["chunk_id", "trace_seed"], columns="policy", values="total_time_ns"
        ).dropna()
        if "repressdb" not in pivot:
            continue
        for reference in REFERENCES:
            if reference not in pivot:
                continue
            ratio_df = (
                (pivot["repressdb"] / pivot[reference])
                .rename("ratio")
                .reset_index()
            )
            chunk_ratios = (
                ratio_df.groupby("chunk_id", as_index=False)["ratio"].median()["ratio"]
                .to_numpy(dtype=float)
            )
            try:
                p_value = float(
                    wilcoxon(chunk_ratios - 1.0, alternative="two-sided", zero_method="wilcox").pvalue
                )
            except ValueError:
                p_value = 1.0
            lo, hi = bootstrap_median_ci(
                chunk_ratios,
                stable_seed(SEED, trace, chunk_size, reference, "cluster_bootstrap"),
            )
            rows.append(
                {
                    "trace": trace,
                    "chunk_size": int(chunk_size),
                    "reference_policy": reference,
                    "independent_unit": "chunk",
                    "n_independent_chunks": int(chunk_ratios.size),
                    "n_trace_seeds_per_chunk": int(ratio_df["trace_seed"].nunique()),
                    "median_ratio_repressdb_over_reference": float(np.median(chunk_ratios)),
                    "mean_ratio_repressdb_over_reference": float(np.mean(chunk_ratios)),
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "fraction_repressdb_faster": float(np.mean(chunk_ratios < 1.0)),
                    "wilcoxon_p": p_value,
                }
            )

    out = pd.DataFrame(rows).sort_values(["trace", "chunk_size", "reference_policy"])
    out.to_csv(OUTPUT_FILE, index=False)

    focus = out[
        (out["reference_policy"] == "initial_best")
        & (
            ((out["trace"] == "scan_to_point") & out["chunk_size"].isin([8192, 16384]))
            | ((out["trace"] == "scan_to_aggregate") & (out["chunk_size"] == 16384))
        )
    ]
    summary = {
        "input": str(INPUT_FILE),
        "output": str(OUTPUT_FILE),
        "independent_unit": "chunk",
        "repeated_measurement_unit": "trace_seed",
        "n_rows": int(len(out)),
        "formal_note": (
            "Each chunk is one independent unit. The three trace seeds are aggregated within chunk "
            "before bootstrap confidence intervals and Wilcoxon signed-rank tests are computed."
        ),
        "key_stable_drift_comparisons_vs_initial_best": focus.to_dict("records"),
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=True)

    print("Chunk-level statistics rebuilt successfully.")
    print(f"Output: {OUTPUT_FILE.resolve()}")
    print(f"Summary: {SUMMARY_FILE.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
