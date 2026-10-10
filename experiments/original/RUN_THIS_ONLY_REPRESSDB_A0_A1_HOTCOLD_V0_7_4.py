#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
RePressDB A0+A1 v0.7.4
Action-sensitive AF-HC-D selection and three-repeat paired actual audit.

Target journal: Journal of Big Data.

DIRECT-RUN
----------
Place this file in the RePressDB project root beside:

  RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_DEADBAND_RECOVER_V0_7_3.py
  RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_DEV_PEMS04_V0_7_0.py
  RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT_V0_2_0.py
  RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT.py
  RUN_THIS_ONLY_REPRESSDB_E0C_V0_2_0.py

Keep both completed directories:

  results_a0_a1_hotcold_pems04_v070/
  results_a0_a1_hotcold_deadband_pems04_v073/

Open this file in PyCharm and click Run.

What v0.7.4 changes
-------------------
v0.7.3 correctly implemented AF-HC-D, but its configuration objective took
the median over all five traces. Three traces are deliberate no-action cases
for an access-frequency-only method, so 18 of 30 cells were exactly 1.0 and
the objective could not distinguish configurations.

v0.7.4 keeps all safety filters unchanged, but selects the configuration only
from the two traces in which access frequency is informative:

  frequency_only
  joint_orthogonal

The selection score is the geometric mean of AF-HC-D / initial-best total-cost
ratios across:

  2 traces × 2 chunk sizes × 3 trace seeds = 12 action cells.

Safety is still evaluated on:

  stable_scan ratio <= 1.05
  brief_burst ratio <= 1.05
  brief_burst median migrations/chunk <= 2

No RePressDB controller constant changes.

Actual audit
------------
The selected configuration is audited using real file-backed execution:

  3 policies
  × 5 traces
  × 2 chunk sizes
  × 3 trace seeds
  × 3 repeats
  = 270 complete policy runs.

Each run performs its own full warm-up. Policy order is independently
randomized inside every repeat. Ratios are first paired within the exact
repeat/seed/trace/size/series cell, then aggregated across repeats and seeds.

Checkpointing
-------------
Every completed policy run is written to an atomic checkpoint directory.
If PyCharm, Windows, or the machine stops, rerun this same file. Completed
runs with the same experiment-freeze hash are skipped.

The program never reads PEMS08 or BDG2 values.
"""

from __future__ import annotations

import gc
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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


VERSION = "0.7.4"
TARGET_JOURNAL = "Journal of Big Data"

V073_SCRIPT_NAME = (
    "RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_DEADBAND_RECOVER_V0_7_3.py"
)
V070_SCRIPT_NAME = (
    "RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_DEV_PEMS04_V0_7_0.py"
)
V070_RESULT_DIR_NAME = "results_a0_a1_hotcold_pems04_v070"
V073_RESULT_DIR_NAME = "results_a0_a1_hotcold_deadband_pems04_v073"
OUTPUT_DIR_NAME = "results_a0_a1_hotcold_deadband_pems04_v074"
SUMMARY_JSON_NAME = (
    "A0_A1_HOTCOLD_DEADBAND_PEMS04_ASSESSMENT_v074.json"
)

ACTION_TRACES = ("frequency_only", "joint_orthogonal")
NO_ACTION_TRACES = ("shape_only", "brief_burst", "stable_scan")
ACTUAL_REPEATS = 3
NO_ACTION_RATIO_LOW = 0.95
NO_ACTION_RATIO_HIGH = 1.05
MODEL_ACTUAL_MEDIAN_ERROR_LIMIT = 0.10
BOOTSTRAP_ITERATIONS = 5000
POLICY_PAUSE_SECONDS = 0.05

REQUIRED_V073_FILES = (
    "v073_deadband_protocol.json",
    "deadband_protocol_sha256.txt",
    "hotcold_grid_all_results.csv",
    "hotcold_grid_config_summary.csv",
    "model_repressdb_initial_results.csv",
    "generated_trace_manifest.csv",
)

REQUIRED_V070_FILES = (
    "protocol_sha256.txt",
    "hotcold_protocol.json",
    "pems04_selected_chunks.csv",
    "pems08_selected_chunks.csv",
    "bdg2_selected_buildings.csv",
    "trace_assignment_manifest.csv",
    "calibration_records.csv",
    "codec_records.csv",
    "measured_migration_costs.csv",
)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(
            json_safe(value),
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        )
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def atomic_write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, module_name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing Python file: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def stable_seed(*parts: Any) -> int:
    token = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def verify_required_files(
    directory: Path,
    filenames: Sequence[str],
    require_candidate_storage: bool = False,
) -> None:
    if not directory.exists():
        raise FileNotFoundError(f"Missing result directory: {directory}")
    missing = [
        filename
        for filename in filenames
        if not (directory / filename).exists()
    ]
    if require_candidate_storage:
        storage = directory / "_candidate_storage"
        if not storage.exists():
            missing.append("_candidate_storage/")
    if missing:
        raise FileNotFoundError(
            f"Incomplete result directory: {directory}\n"
            f"Missing: {missing}"
        )


def geometric_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan
    if np.any(array <= 0.0):
        raise ValueError("Cost ratios must be strictly positive")
    return float(np.exp(np.mean(np.log(array))))


def bootstrap_median_ci(
    values: np.ndarray,
    seed: int,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=float)
    for index in range(iterations):
        samples[index] = np.median(
            rng.choice(array, size=array.size, replace=True)
        )
    return (
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def verify_v073_deadband_hash(
    v070_dir: Path,
    v073_dir: Path,
) -> str:
    parent_hash = (
        v070_dir / "protocol_sha256.txt"
    ).read_text(encoding="utf-8").strip()
    stored_deadband_hash = (
        v073_dir / "deadband_protocol_sha256.txt"
    ).read_text(encoding="utf-8").strip()

    digest = hashlib.sha256()
    digest.update(parent_hash.encode("ascii"))
    digest.update(
        sha256_file(
            v073_dir / "v073_deadband_protocol.json"
        ).encode("ascii")
    )
    recomputed = digest.hexdigest()
    if recomputed != stored_deadband_hash:
        raise AssertionError(
            "v0.7.3 deadband protocol hash mismatch.\n"
            f"Stored:     {stored_deadband_hash}\n"
            f"Recomputed: {recomputed}"
        )
    return stored_deadband_hash


def select_action_sensitive_configuration(
    core: Any,
    grid_run_df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Any],
]:
    required_columns = {
        "config_id",
        "adaptation_period",
        "hot_fraction",
        "cold_codec",
        "hot_codec",
        "trace",
        "trace_seed",
        "chunk_size",
        "ratio_to_initial_best",
        "median_migrations_per_chunk",
        "rewrite_bytes",
    }
    missing = required_columns.difference(grid_run_df.columns)
    if missing:
        raise ValueError(f"Grid results missing columns: {sorted(missing)}")

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for config_id, group in grid_run_df.groupby(
        "config_id",
        sort=True,
    ):
        first = group.iloc[0]
        stable = group[group["trace"] == "stable_scan"]
        burst = group[group["trace"] == "brief_burst"]
        action = group[
            group["trace"].isin(ACTION_TRACES)
        ].copy()

        if len(action) != (
            len(ACTION_TRACES)
            * len(core.CHUNK_SIZES)
            * len(core.TRACE_SEEDS)
        ):
            raise AssertionError(
                f"Action-cell count mismatch for {config_id}: "
                f"{len(action)}"
            )

        stable_ratio = float(
            stable["ratio_to_initial_best"].median()
        )
        burst_ratio = float(
            burst["ratio_to_initial_best"].median()
        )
        burst_migrations = float(
            burst["median_migrations_per_chunk"].median()
        )

        action_score = geometric_mean(
            action["ratio_to_initial_best"].to_numpy(dtype=float)
        )
        action_median_ratio = float(
            action["ratio_to_initial_best"].median()
        )
        action_mean_ratio = float(
            action["ratio_to_initial_best"].mean()
        )
        action_migrations = float(
            action["median_migrations_per_chunk"].median()
        )
        action_rewrite_bytes = float(
            action["rewrite_bytes"].median()
        )

        all_trace_migrations = float(
            group["median_migrations_per_chunk"].median()
        )
        all_trace_rewrite = float(
            group["rewrite_bytes"].median()
        )

        reasons: List[str] = []
        if stable_ratio > core.MAX_STABLE_RATIO:
            reasons.append("stable_ratio_above_1.05")
        if burst_ratio > core.MAX_BURST_RATIO:
            reasons.append("burst_ratio_above_1.05")
        if (
            burst_migrations
            > core.MAX_BURST_MEDIAN_MIGRATIONS_PER_CHUNK
        ):
            reasons.append(
                "burst_migrations_per_chunk_above_2"
            )

        row = {
            "config_id": str(config_id),
            "adaptation_period": int(
                first["adaptation_period"]
            ),
            "hot_fraction": float(first["hot_fraction"]),
            "cold_codec": str(first["cold_codec"]),
            "hot_codec": str(first["hot_codec"]),
            "action_geometric_mean_ratio": action_score,
            "action_median_ratio": action_median_ratio,
            "action_mean_ratio": action_mean_ratio,
            "action_median_migrations_per_chunk": (
                action_migrations
            ),
            "action_median_rewrite_bytes": (
                action_rewrite_bytes
            ),
            "stable_median_ratio": stable_ratio,
            "burst_median_ratio": burst_ratio,
            "burst_median_migrations_per_chunk": (
                burst_migrations
            ),
            "all_trace_median_migrations_per_chunk": (
                all_trace_migrations
            ),
            "all_trace_median_rewrite_bytes": (
                all_trace_rewrite
            ),
            "safety_pass": len(reasons) == 0,
            "failure_reasons": "|".join(reasons),
            "n_safety_failures": int(len(reasons)),
            "stable_excess_over_limit": max(
                0.0,
                stable_ratio - core.MAX_STABLE_RATIO,
            ),
            "burst_excess_over_limit": max(
                0.0,
                burst_ratio - core.MAX_BURST_RATIO,
            ),
            "burst_migration_excess_over_limit": max(
                0.0,
                burst_migrations
                - core.MAX_BURST_MEDIAN_MIGRATIONS_PER_CHUNK,
            ),
        }
        rows.append(row)
        if reasons:
            failures.append(row)

    summary = pd.DataFrame(rows)
    if summary["config_id"].nunique() != 72:
        raise AssertionError(
            "Expected exactly 72 AF-HC-D configurations"
        )

    safe = summary[summary["safety_pass"]].copy()
    if safe.empty:
        diagnostic = summary.sort_values(
            [
                "n_safety_failures",
                "stable_excess_over_limit",
                "burst_excess_over_limit",
                "burst_migration_excess_over_limit",
                "action_geometric_mean_ratio",
                "action_median_migrations_per_chunk",
                "action_median_rewrite_bytes",
                "adaptation_period",
                "config_id",
            ],
            ascending=[
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
            ],
        ).reset_index(drop=True)
        selected = diagnostic.iloc[0].to_dict()
        selected["selection_status"] = (
            "diagnostic_only_no_safe_configuration"
        )
        return (
            summary.sort_values("config_id").reset_index(
                drop=True
            ),
            pd.DataFrame(failures),
            diagnostic,
            selected,
        )

    safe = safe.sort_values(
        [
            "action_geometric_mean_ratio",
            "action_median_migrations_per_chunk",
            "action_median_rewrite_bytes",
            "adaptation_period",
            "config_id",
        ],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)

    best_score = float(
        safe.iloc[0]["action_geometric_mean_ratio"]
    )
    equivalent = safe[
        safe["action_geometric_mean_ratio"]
        <= best_score * (1.0 + core.TIE_SCORE_TOLERANCE)
    ].copy()

    equivalent = equivalent.sort_values(
        [
            "action_median_migrations_per_chunk",
            "action_median_rewrite_bytes",
            "adaptation_period",
            "config_id",
        ],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)

    selected = equivalent.iloc[0].to_dict()
    selected["selection_status"] = (
        "frozen_action_sensitive_configuration"
    )
    selected["best_action_score_before_tie_band"] = (
        best_score
    )
    selected["tie_band_relative_tolerance"] = float(
        core.TIE_SCORE_TOLERANCE
    )
    selected["n_configs_in_tie_band"] = int(
        len(equivalent)
    )

    rank_map = {
        str(row.config_id): int(index + 1)
        for index, row in enumerate(
            safe.itertuples(index=False)
        )
    }
    summary["safe_action_rank_one_based"] = (
        summary["config_id"].map(rank_map)
    )
    summary["selected"] = (
        summary["config_id"]
        == str(selected["config_id"])
    )

    return (
        summary.sort_values("config_id").reset_index(
            drop=True
        ),
        pd.DataFrame(failures),
        safe,
        selected,
    )


def load_profiles(
    calibration_df: pd.DataFrame,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    profiles: Dict[
        str,
        Dict[str, Dict[str, float]],
    ] = {}
    for chunk_id, chunk_group in calibration_df.groupby(
        "chunk_id"
    ):
        local: Dict[str, Dict[str, float]] = {}
        for codec, codec_group in chunk_group.groupby(
            "codec"
        ):
            local[str(codec)] = {
                str(row.query_class): float(
                    row.median_latency_ns
                )
                for row in codec_group.itertuples(
                    index=False
                )
            }
        profiles[str(chunk_id)] = local
    return profiles


def load_encode_decode_maps(
    codec_df: pd.DataFrame,
) -> Tuple[
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, int]],
]:
    encode: Dict[str, Dict[str, int]] = {}
    decode: Dict[str, Dict[str, int]] = {}
    for chunk_id, group in codec_df.groupby("chunk_id"):
        encode[str(chunk_id)] = {
            str(row.codec): int(row.encode_ns)
            for row in group.itertuples(index=False)
        }
        decode[str(chunk_id)] = {
            str(row.codec): int(row.full_decode_ns)
            for row in group.itertuples(index=False)
        }
    return encode, decode


def reconstruct_stored_maps(
    base: Any,
    chunks_by_id: Mapping[str, Any],
    codec_df: pd.DataFrame,
    source_storage_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    stored_by_chunk: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for chunk_id, group in codec_df.groupby("chunk_id"):
        chunk_id = str(chunk_id)
        chunk = chunks_by_id[chunk_id]
        local: Dict[str, Any] = {}

        for row in group.itertuples(index=False):
            codec = str(row.codec)
            stem = (
                source_storage_dir / chunk_id / codec
            )
            data_path = stem.with_suffix(".bin")
            metadata_path = stem.with_suffix(
                ".meta.bin"
            )

            if not data_path.exists():
                raise FileNotFoundError(
                    f"Missing cached payload: {data_path}"
                )

            if metadata_path.exists():
                metadata_raw = metadata_path.read_bytes()
                metadata = base.parse_block_metadata(
                    metadata_raw
                )
                actual_metadata_path: Optional[Path] = (
                    metadata_path
                )
                metadata_size = int(
                    metadata_path.stat().st_size
                )
            else:
                metadata = (
                    {"level": 3}
                    if codec == "whole_zstd"
                    else {}
                )
                actual_metadata_path = None
                metadata_size = 0

            stored = base.StoredChunk(
                codec=codec,
                n=int(chunk.values.size),
                data_path=data_path,
                metadata_path=actual_metadata_path,
                payload_size_bytes=int(
                    data_path.stat().st_size
                ),
                metadata_size_bytes=metadata_size,
                metadata=metadata,
            )
            expected_size = int(row.stored_size_bytes)
            if (
                int(stored.stored_size_bytes)
                != expected_size
            ):
                raise AssertionError(
                    f"Cached size mismatch for "
                    f"{chunk_id}/{codec}"
                )
            local[codec] = stored

        stored_by_chunk[chunk_id] = local

    return stored_by_chunk


def checkpoint_token(
    repeat_index: int,
    trace_seed: int,
    trace: str,
    chunk_size: int,
    policy: str,
) -> str:
    return (
        f"repeat_{repeat_index:02d}"
        f"__seed_{trace_seed}"
        f"__trace_{trace}"
        f"__size_{chunk_size}"
        f"__policy_{policy}"
    )


def checkpoint_is_complete(
    run_dir: Path,
    freeze_hash: str,
) -> bool:
    marker = run_dir / "complete.json"
    summary_path = run_dir / "summary.json"
    chunks_path = run_dir / "chunks.csv"
    events_path = run_dir / "events.csv"
    if not (
        marker.exists()
        and summary_path.exists()
        and chunks_path.exists()
        and events_path.exists()
    ):
        return False
    try:
        payload = json.loads(
            marker.read_text(encoding="utf-8")
        )
    except Exception:
        return False
    return (
        payload.get("experiment_freeze_sha256")
        == freeze_hash
    )


def write_checkpoint(
    run_dir: Path,
    summary: Mapping[str, Any],
    chunk_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    freeze_hash: str,
) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "summary.json", dict(summary))
    atomic_write_dataframe(
        run_dir / "chunks.csv",
        pd.DataFrame(chunk_rows),
    )
    atomic_write_dataframe(
        run_dir / "events.csv",
        pd.DataFrame(event_rows),
    )
    write_json(
        run_dir / "complete.json",
        {
            "experiment_freeze_sha256": freeze_hash,
            "completed_at_unix_ns": time.time_ns(),
        },
    )


def load_all_checkpoints(
    checkpoint_root: Path,
    expected_tokens: Sequence[str],
    freeze_hash: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: List[Dict[str, Any]] = []
    chunk_frames: List[pd.DataFrame] = []
    event_frames: List[pd.DataFrame] = []

    missing: List[str] = []
    for token in expected_tokens:
        run_dir = checkpoint_root / token
        if not checkpoint_is_complete(
            run_dir,
            freeze_hash,
        ):
            missing.append(token)
            continue

        summaries.append(
            json.loads(
                (run_dir / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        chunk_frames.append(
            pd.read_csv(run_dir / "chunks.csv")
        )
        event_frames.append(
            pd.read_csv(run_dir / "events.csv")
        )

    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} completed audit runs. "
            f"First missing: {missing[:3]}"
        )

    summary_df = pd.DataFrame(summaries)
    chunk_df = pd.concat(
        chunk_frames,
        ignore_index=True,
    )
    event_df = pd.concat(
        event_frames,
        ignore_index=True,
    )
    return summary_df, chunk_df, event_df


def build_exact_paired_ratios(
    chunk_run_df: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "repeat_index",
        "trace_seed",
        "trace",
        "chunk_size",
        "series_index",
    ]
    pivot = chunk_run_df.pivot(
        index=keys,
        columns="policy",
        values="total_time_ns",
    ).reset_index()

    rows: List[Dict[str, Any]] = []
    for policy, reference in (
        ("repressdb", "afhc_frozen"),
        ("repressdb", "initial_best"),
        ("afhc_frozen", "initial_best"),
    ):
        if (
            policy not in pivot.columns
            or reference not in pivot.columns
        ):
            continue
        local = pivot[
            keys + [policy, reference]
        ].dropna()
        for row in local.itertuples(index=False):
            policy_time = float(getattr(row, policy))
            reference_time = float(
                getattr(row, reference)
            )
            rows.append(
                {
                    "repeat_index": int(
                        row.repeat_index
                    ),
                    "trace_seed": int(row.trace_seed),
                    "trace": str(row.trace),
                    "chunk_size": int(row.chunk_size),
                    "series_index": int(
                        row.series_index
                    ),
                    "policy": policy,
                    "reference_policy": reference,
                    "policy_total_time_ns": policy_time,
                    "reference_total_time_ns": (
                        reference_time
                    ),
                    "paired_ratio": float(
                        policy_time / reference_time
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_series_clustered_comparisons(
    exact_ratio_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    group_columns = [
        "trace",
        "chunk_size",
        "policy",
        "reference_policy",
    ]
    for keys, group in exact_ratio_df.groupby(
        group_columns,
        sort=True,
    ):
        trace, chunk_size, policy, reference = keys

        series_ratios = (
            group.groupby(
                "series_index",
                as_index=False,
            )["paired_ratio"]
            .median()
        )
        ratios = series_ratios[
            "paired_ratio"
        ].to_numpy(dtype=float)

        ci_low, ci_high = bootstrap_median_ci(
            ratios,
            stable_seed(
                VERSION,
                trace,
                chunk_size,
                policy,
                reference,
                "bootstrap",
            ),
        )

        try:
            log_ratios = np.log(ratios)
            if np.allclose(log_ratios, 0.0):
                p_value = 1.0
            else:
                p_value = float(
                    wilcoxon(
                        log_ratios,
                        alternative="two-sided",
                        zero_method="wilcox",
                    ).pvalue
                )
        except ValueError:
            p_value = 1.0

        repeat_medians = (
            group.groupby("repeat_index")[
                "paired_ratio"
            ].median()
        )

        rows.append(
            {
                "trace": str(trace),
                "chunk_size": int(chunk_size),
                "policy": str(policy),
                "reference_policy": str(reference),
                "n_independent_series": int(
                    len(series_ratios)
                ),
                "n_exact_paired_cells": int(
                    len(group)
                ),
                "median_ratio_policy_over_reference": (
                    float(np.median(ratios))
                ),
                "mean_ratio_policy_over_reference": (
                    float(np.mean(ratios))
                ),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "fraction_policy_faster": float(
                    np.mean(ratios < 1.0)
                ),
                "wilcoxon_log_ratio_p": p_value,
                "repeat_median_ratio_min": float(
                    repeat_medians.min()
                ),
                "repeat_median_ratio_max": float(
                    repeat_medians.max()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_no_action_consistency(
    exact_ratio_df: pd.DataFrame,
    actual_summary_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    local = exact_ratio_df[
        exact_ratio_df["trace"].isin(
            NO_ACTION_TRACES
        )
        & (
            exact_ratio_df["policy"]
            == "afhc_frozen"
        )
        & (
            exact_ratio_df["reference_policy"]
            == "initial_best"
        )
    ].copy()

    cell_rows: List[Dict[str, Any]] = []
    for (trace, chunk_size), group in local.groupby(
        ["trace", "chunk_size"],
        sort=True,
    ):
        ratios = group["paired_ratio"].to_numpy(
            dtype=float
        )
        cell_rows.append(
            {
                "trace": str(trace),
                "chunk_size": int(chunk_size),
                "n_exact_paired_cells": int(
                    len(group)
                ),
                "median_afhc_over_initial": float(
                    np.median(ratios)
                ),
                "p05_afhc_over_initial": float(
                    np.quantile(ratios, 0.05)
                ),
                "p95_afhc_over_initial": float(
                    np.quantile(ratios, 0.95)
                ),
                "minimum_afhc_over_initial": float(
                    np.min(ratios)
                ),
                "maximum_afhc_over_initial": float(
                    np.max(ratios)
                ),
            }
        )

    cell_df = pd.DataFrame(cell_rows)

    all_ratios = local[
        "paired_ratio"
    ].to_numpy(dtype=float)
    overall_median = (
        float(np.median(all_ratios))
        if all_ratios.size
        else math.nan
    )

    afhc_no_action_runs = actual_summary_df[
        actual_summary_df["trace"].isin(
            NO_ACTION_TRACES
        )
        & (
            actual_summary_df["policy"]
            == "afhc_frozen"
        )
    ]
    total_migrations = int(
        afhc_no_action_runs[
            "migration_count"
        ].sum()
    )

    assessment = {
        "no_action_traces": list(
            NO_ACTION_TRACES
        ),
        "n_exact_paired_cells": int(
            len(local)
        ),
        "overall_median_afhc_over_initial": (
            overall_median
        ),
        "required_interval": [
            NO_ACTION_RATIO_LOW,
            NO_ACTION_RATIO_HIGH,
        ],
        "overall_median_within_required_interval": bool(
            np.isfinite(overall_median)
            and NO_ACTION_RATIO_LOW
            <= overall_median
            <= NO_ACTION_RATIO_HIGH
        ),
        "afhc_total_migrations_in_no_action_traces": (
            total_migrations
        ),
        "zero_no_action_migrations": bool(
            total_migrations == 0
        ),
    }
    assessment["pass"] = bool(
        assessment[
            "overall_median_within_required_interval"
        ]
        and assessment["zero_no_action_migrations"]
    )
    return cell_df, assessment


def build_repeated_model_actual_validation(
    grid_run_df: pd.DataFrame,
    model_policy_df: pd.DataFrame,
    actual_summary_df: pd.DataFrame,
    selected_config_id: str,
) -> pd.DataFrame:
    keys = ["trace", "trace_seed", "chunk_size"]

    selected_grid = grid_run_df[
        grid_run_df["config_id"]
        == selected_config_id
    ][keys + ["total_time_ns"]].rename(
        columns={
            "total_time_ns": "model_afhc_total_ns"
        }
    )

    model_pivot = model_policy_df.pivot(
        index=keys,
        columns="policy",
        values="total_time_ns",
    ).reset_index().rename(
        columns={
            "initial_best": (
                "model_initial_total_ns"
            ),
            "repressdb": (
                "model_repressdb_total_ns"
            ),
        }
    )

    actual_pivot = actual_summary_df.pivot(
        index=[
            "repeat_index",
            *keys,
        ],
        columns="policy",
        values="total_time_ns",
    ).reset_index().rename(
        columns={
            "initial_best": (
                "actual_initial_total_ns"
            ),
            "afhc_frozen": (
                "actual_afhc_total_ns"
            ),
            "repressdb": (
                "actual_repressdb_total_ns"
            ),
        }
    )

    actual_pivot[
        "actual_afhc_ratio_to_initial"
    ] = (
        actual_pivot["actual_afhc_total_ns"]
        / actual_pivot["actual_initial_total_ns"]
    )
    actual_pivot[
        "actual_repressdb_ratio_to_initial"
    ] = (
        actual_pivot[
            "actual_repressdb_total_ns"
        ]
        / actual_pivot["actual_initial_total_ns"]
    )

    actual_aggregated = (
        actual_pivot.groupby(keys, as_index=False)
        .agg(
            actual_afhc_ratio_to_initial=(
                "actual_afhc_ratio_to_initial",
                "median",
            ),
            actual_repressdb_ratio_to_initial=(
                "actual_repressdb_ratio_to_initial",
                "median",
            ),
            actual_afhc_repeat_min=(
                "actual_afhc_ratio_to_initial",
                "min",
            ),
            actual_afhc_repeat_max=(
                "actual_afhc_ratio_to_initial",
                "max",
            ),
            actual_repressdb_repeat_min=(
                "actual_repressdb_ratio_to_initial",
                "min",
            ),
            actual_repressdb_repeat_max=(
                "actual_repressdb_ratio_to_initial",
                "max",
            ),
        )
    )

    merged = (
        selected_grid.merge(
            model_pivot,
            on=keys,
            how="inner",
        )
        .merge(
            actual_aggregated,
            on=keys,
            how="inner",
        )
    )

    merged[
        "model_afhc_ratio_to_initial"
    ] = (
        merged["model_afhc_total_ns"]
        / merged["model_initial_total_ns"]
    )
    merged[
        "model_repressdb_ratio_to_initial"
    ] = (
        merged["model_repressdb_total_ns"]
        / merged["model_initial_total_ns"]
    )

    merged["absolute_afhc_ratio_error"] = (
        merged["model_afhc_ratio_to_initial"]
        - merged["actual_afhc_ratio_to_initial"]
    ).abs()
    merged[
        "absolute_repressdb_ratio_error"
    ] = (
        merged[
            "model_repressdb_ratio_to_initial"
        ]
        - merged[
            "actual_repressdb_ratio_to_initial"
        ]
    ).abs()

    return merged


def comparison_lookup(
    comparison_df: pd.DataFrame,
    trace: str,
    chunk_size: int,
    policy: str,
    reference: str,
) -> Dict[str, Any]:
    local = comparison_df[
        (comparison_df["trace"] == trace)
        & (
            comparison_df["chunk_size"]
            == int(chunk_size)
        )
        & (comparison_df["policy"] == policy)
        & (
            comparison_df["reference_policy"]
            == reference
        )
    ]
    if len(local) != 1:
        return {}
    row = local.iloc[0]
    return {
        "median_ratio": float(
            row[
                "median_ratio_policy_over_reference"
            ]
        ),
        "bootstrap_ci_low": float(
            row["bootstrap_ci_low"]
        ),
        "bootstrap_ci_high": float(
            row["bootstrap_ci_high"]
        ),
        "fraction_policy_faster": float(
            row["fraction_policy_faster"]
        ),
        "wilcoxon_log_ratio_p": float(
            row["wilcoxon_log_ratio_p"]
        ),
        "n_independent_series": int(
            row["n_independent_series"]
        ),
    }


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    v073_script = project_dir / V073_SCRIPT_NAME
    v070_script = project_dir / V070_SCRIPT_NAME
    v070_dir = project_dir / V070_RESULT_DIR_NAME
    v073_dir = project_dir / V073_RESULT_DIR_NAME
    output_dir = project_dir / OUTPUT_DIR_NAME

    verify_required_files(
        v070_dir,
        REQUIRED_V070_FILES,
        require_candidate_storage=True,
    )
    verify_required_files(
        v073_dir,
        REQUIRED_V073_FILES,
        require_candidate_storage=False,
    )
    if not v073_script.exists():
        raise FileNotFoundError(
            f"Missing v0.7.3 script: {v073_script}"
        )
    if not v070_script.exists():
        raise FileNotFoundError(
            f"Missing v0.7.0 script: {v070_script}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 112)
    print(
        "RePressDB A0+A1 v0.7.4 — action-sensitive selection "
        "and repeated paired audit"
    )
    print(f"Target journal: {TARGET_JOURNAL}")
    print(
        "Reuse v0.7.3 grid and v0.7.0 calibration; "
        "PEMS08 and BDG2 remain untouched."
    )
    print("=" * 112)

    v073 = load_module(
        v073_script,
        "repressdb_a0_a1_v073_wrapper_for_v074",
    )
    core = v073.build_corrected_core(
        v070_script,
        output_dir,
    )
    core.__file__ = str(v070_script.resolve())
    e1v2, e1, base = core.load_repressdb_modules()

    deadband_hash = verify_v073_deadband_hash(
        v070_dir,
        v073_dir,
    )
    parent_protocol_hash = (
        v070_dir / "protocol_sha256.txt"
    ).read_text(encoding="utf-8").strip()

    grid_run_df = pd.read_csv(
        v073_dir / "hotcold_grid_all_results.csv"
    )
    model_policy_df = pd.read_csv(
        v073_dir
        / "model_repressdb_initial_results.csv"
    )
    generated_trace_manifest = pd.read_csv(
        v073_dir / "generated_trace_manifest.csv"
    )

    (
        grid_summary_df,
        grid_failures_df,
        grid_ranked_df,
        selected_summary,
    ) = select_action_sensitive_configuration(
        core,
        grid_run_df,
    )

    atomic_write_dataframe(
        output_dir
        / "hotcold_grid_config_summary_v074.csv",
        grid_summary_df,
    )
    atomic_write_dataframe(
        output_dir
        / "hotcold_grid_safety_failures_v074.csv",
        grid_failures_df,
    )
    atomic_write_dataframe(
        output_dir / "hotcold_grid_ranked_v074.csv",
        grid_ranked_df,
    )
    atomic_write_dataframe(
        output_dir / "hotcold_grid_all_results.csv",
        grid_run_df,
    )
    atomic_write_dataframe(
        output_dir
        / "model_repressdb_initial_results.csv",
        model_policy_df,
    )
    atomic_write_dataframe(
        output_dir / "generated_trace_manifest.csv",
        generated_trace_manifest,
    )

    safe_count = int(
        grid_summary_df["safety_pass"].sum()
    )
    if safe_count == 0:
        diagnostic = {
            "version": VERSION,
            "status": (
                "no_safe_configuration_under_unchanged_safety_rules"
            ),
            "deadband_protocol_sha256": deadband_hash,
            "best_diagnostic_configuration": (
                selected_summary
            ),
            "decision": (
                "Stop. Do not relax safety filters and do not "
                "run holdouts."
            ),
        }
        write_json(
            output_dir
            / "hotcold_no_safe_config_diagnostic_v074.json",
            diagnostic,
        )
        write_json(
            project_dir / SUMMARY_JSON_NAME,
            diagnostic,
        )
        print(json.dumps(
            diagnostic,
            ensure_ascii=False,
            indent=2,
        ))
        return 2

    selected_config_id = str(
        selected_summary["config_id"]
    )
    grid_configs = core.config_grid()
    selected_config = next(
        dict(config)
        for config in grid_configs
        if str(config["config_id"])
        == selected_config_id
    )

    selection_protocol = {
        "version": VERSION,
        "parent_deadband_protocol_sha256": (
            deadband_hash
        ),
        "parent_a0_protocol_sha256": (
            parent_protocol_hash
        ),
        "change_classification": (
            "selection-objective correction and measurement "
            "reliability improvement"
        ),
        "unchanged": {
            "AF_HC_D_deadband_factor": 1.5,
            "AF_HC_D_information_boundary": True,
            "grid_size": 72,
            "data_selection": True,
            "trace_assignments": True,
            "trace_seeds": list(core.TRACE_SEEDS),
            "chunk_sizes": list(core.CHUNK_SIZES),
            "stable_ratio_limit": (
                core.MAX_STABLE_RATIO
            ),
            "brief_burst_ratio_limit": (
                core.MAX_BURST_RATIO
            ),
            "brief_burst_migration_limit": (
                core.MAX_BURST_MEDIAN_MIGRATIONS_PER_CHUNK
            ),
            "RePressDB_controller": (
                "unchanged and frozen"
            ),
        },
        "selection": {
            "action_traces": list(ACTION_TRACES),
            "number_of_action_cells": 12,
            "score": (
                "geometric mean of AF-HC-D / initial-best "
                "total-cost ratios"
            ),
            "equivalence_tolerance": float(
                core.TIE_SCORE_TOLERANCE
            ),
            "tie_break_order": [
                "fewer action migrations",
                "fewer action rewrite bytes",
                "longer adaptation period",
                "lexicographically smaller config id",
            ],
            "safety_traces": [
                "stable_scan",
                "brief_burst",
            ],
        },
        "actual_audit": {
            "repeats": ACTUAL_REPEATS,
            "policies": list(core.AUDIT_POLICIES),
            "full_warmup_before_every_policy_run": True,
            "policy_order_randomized_per_repeat": True,
            "pairing": (
                "ratios are computed within exact repeat, "
                "seed, trace, size, and series cells"
            ),
            "no_action_consistency_interval": [
                NO_ACTION_RATIO_LOW,
                NO_ACTION_RATIO_HIGH,
            ],
            "model_actual_median_error_limit": (
                MODEL_ACTUAL_MEDIAN_ERROR_LIMIT
            ),
            "checkpoint_resume": True,
        },
        "holdout_status": (
            "PEMS08 and BDG2 are not accessed by this program."
        ),
    }
    write_json(
        output_dir / "v074_selection_audit_protocol.json",
        selection_protocol,
    )

    protocol_digest = hashlib.sha256()
    protocol_digest.update(
        deadband_hash.encode("ascii")
    )
    protocol_digest.update(
        sha256_file(
            output_dir
            / "v074_selection_audit_protocol.json"
        ).encode("ascii")
    )
    v074_protocol_hash = protocol_digest.hexdigest()
    (
        output_dir / "v074_protocol_sha256.txt"
    ).write_text(
        v074_protocol_hash + "\n",
        encoding="utf-8",
    )

    selected_config.update(
        {
            "selection_dataset": "PEMS04",
            "selection_version": VERSION,
            "frozen_for_holdouts": True,
            "deadband_factor": 1.5,
            "deadband_protocol_sha256": (
                deadband_hash
            ),
            "v074_protocol_sha256": (
                v074_protocol_hash
            ),
            "selection_objective": (
                "geometric mean over frequency_only and "
                "joint_orthogonal action cells"
            ),
            "selection_summary": {
                key: json_safe(value)
                for key, value
                in selected_summary.items()
            },
            "initialization": (
                "per-chunk scan-optimal representation, "
                "identical to initial-best and RePressDB"
            ),
        }
    )

    freeze_material = {
        "v074_protocol_sha256": v074_protocol_hash,
        "selected_configuration": selected_config,
        "generated_trace_manifest_sha256": (
            sha256_file(
                v073_dir
                / "generated_trace_manifest.csv"
            )
        ),
        "script_sha256": sha256_file(
            Path(__file__).resolve()
        ),
    }
    write_json(
        output_dir / "experiment_freeze_material.json",
        freeze_material,
    )
    freeze_hash = sha256_file(
        output_dir / "experiment_freeze_material.json"
    )
    (
        output_dir / "experiment_freeze_sha256.txt"
    ).write_text(
        freeze_hash + "\n",
        encoding="utf-8",
    )
    selected_config[
        "experiment_freeze_sha256"
    ] = freeze_hash
    write_json(
        output_dir / "hotcold_frozen_config.json",
        selected_config,
    )

    print("-" * 112)
    print(
        "[v0.7.4] selected AF-HC-D configuration:"
    )
    print(json.dumps(
        selected_config,
        ensure_ascii=False,
        indent=2,
    ))
    print("-" * 112)

    # Rebuild the exact frozen PEMS04 chunks.
    pems04_manifest = pd.read_csv(
        v070_dir / "pems04_selected_chunks.csv"
    )
    assignments = pd.read_csv(
        v070_dir / "trace_assignment_manifest.csv"
    )
    calibration_df = pd.read_csv(
        v070_dir / "calibration_records.csv"
    )
    codec_df = pd.read_csv(
        v070_dir / "codec_records.csv"
    )

    pems04_dir, _ = base.default_data_dirs()
    pems04_file = base.auto_detect_data_file(
        pems04_dir,
        "PEMS04_AFHC_V074",
    )
    matrix, metadata = base.load_dataset(
        pems04_file,
        feature_index=0,
        max_series=core.N_SERIES,
        seed=core.SEED,
        excluded_series=None,
    )
    chunks, rebuilt_manifest = (
        core.deterministic_series_chunks(
            base,
            matrix,
            "PEMS04",
            metadata[
                "selected_original_series_indices"
            ],
            chunk_sizes=core.CHUNK_SIZES,
        )
    )

    compare_columns = [
        "dataset",
        "chunk_id",
        "original_series_index",
        "chunk_start",
        "chunk_size",
    ]
    expected_manifest = (
        pems04_manifest[compare_columns]
        .sort_values(compare_columns)
        .reset_index(drop=True)
    )
    actual_manifest = (
        rebuilt_manifest[compare_columns]
        .sort_values(compare_columns)
        .reset_index(drop=True)
    )
    if not expected_manifest.equals(actual_manifest):
        raise AssertionError(
            "Rebuilt PEMS04 manifest differs from "
            "the frozen A0 manifest."
        )

    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunk in chunks
    }
    chunks_by_size = {
        int(size): sorted(
            [
                chunk
                for chunk in chunks
                if int(chunk.chunk_size)
                == int(size)
            ],
            key=lambda item: item.chunk_id,
        )
        for size in core.CHUNK_SIZES
    }

    reps = base.representation_registry(3)
    profiles = load_profiles(calibration_df)
    (
        encode_ns_by_chunk,
        decode_ns_by_chunk,
    ) = load_encode_decode_maps(codec_df)
    stored_by_chunk = reconstruct_stored_maps(
        base,
        chunks_by_id,
        codec_df,
        v070_dir / "_candidate_storage",
    )

    query_pools: Dict[
        str,
        Dict[str, List[Any]],
    ] = {}
    for index, chunk in enumerate(chunks, start=1):
        queries = base.build_queries(
            chunk.values,
            chunk.chunk_id,
            core.SEED,
            e1.WINDOWS_PER_FRACTION,
        )
        pool = {
            query_class: [
                query
                for query in queries
                if query.query_class
                == query_class
            ]
            for query_class
            in core.QUERY_CLASSES
        }
        if any(not values for values in pool.values()):
            raise AssertionError(
                f"Incomplete query pool: "
                f"{chunk.chunk_id}"
            )
        query_pools[chunk.chunk_id] = pool

        if (
            index == 1
            or index == len(chunks)
            or index % 8 == 0
        ):
            print(
                f"[v0.7.4] query pools "
                f"{index}/{len(chunks)}"
            )

    checkpoint_root = (
        output_dir / "_actual_checkpoints"
    )
    checkpoint_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    migration_root = (
        output_dir / "_actual_migration_trials"
    )
    migration_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected_tokens: List[str] = []
    total_runs = (
        ACTUAL_REPEATS
        * len(core.TRACE_SEEDS)
        * len(core.TRACE_NAMES)
        * len(core.CHUNK_SIZES)
        * len(core.AUDIT_POLICIES)
    )
    completed_counter = 0

    for repeat_index in range(ACTUAL_REPEATS):
        for chunk_size in core.CHUNK_SIZES:
            size_chunks = chunks_by_size[
                int(chunk_size)
            ]
            size_assignments = assignments[
                (
                    assignments["dataset"]
                    == "PEMS04"
                )
                & (
                    assignments["chunk_size"]
                    == int(chunk_size)
                )
            ]

            for trace_seed in core.TRACE_SEEDS:
                for trace_name in core.TRACE_NAMES:
                    plan = core.build_trace_plan(
                        base,
                        size_chunks,
                        query_pools,
                        size_assignments,
                        trace_name,
                        int(trace_seed),
                    )

                    policy_order = list(
                        core.AUDIT_POLICIES
                    )
                    random.Random(
                        stable_seed(
                            VERSION,
                            repeat_index,
                            trace_seed,
                            chunk_size,
                            trace_name,
                            "policy_order",
                        )
                    ).shuffle(policy_order)

                    for order_position, policy in enumerate(
                        policy_order
                    ):
                        token = checkpoint_token(
                            repeat_index,
                            int(trace_seed),
                            str(trace_name),
                            int(chunk_size),
                            str(policy),
                        )
                        expected_tokens.append(token)
                        run_dir = checkpoint_root / token

                        if checkpoint_is_complete(
                            run_dir,
                            freeze_hash,
                        ):
                            completed_counter += 1
                            print(
                                f"[audit resume] "
                                f"{completed_counter}/{total_runs} "
                                f"{token}"
                            )
                            continue

                        if run_dir.exists():
                            shutil.rmtree(run_dir)

                        print(
                            f"[actual audit] "
                            f"{completed_counter + 1}/"
                            f"{total_runs} "
                            f"repeat={repeat_index + 1}/"
                            f"{ACTUAL_REPEATS}, "
                            f"size={chunk_size}, "
                            f"trace={trace_name}, "
                            f"seed={trace_seed}, "
                            f"policy={policy}"
                        )

                        gc.collect()
                        if POLICY_PAUSE_SECONDS > 0:
                            time.sleep(
                                POLICY_PAUSE_SECONDS
                            )

                        (
                            summary,
                            chunk_rows,
                            event_rows,
                        ) = core.run_actual_policy(
                            e1,
                            e1v2,
                            base,
                            reps,
                            chunks_by_id,
                            stored_by_chunk,
                            profiles,
                            encode_ns_by_chunk,
                            decode_ns_by_chunk,
                            query_pools,
                            plan,
                            policy,
                            selected_config,
                            migration_root
                            / f"repeat_{repeat_index:02d}"
                            / token,
                        )

                        common = {
                            "repeat_index": int(
                                repeat_index
                            ),
                            "repeat_one_based": int(
                                repeat_index + 1
                            ),
                            "policy_order_position": int(
                                order_position
                            ),
                            "actual_run_token": token,
                            "experiment_freeze_sha256": (
                                freeze_hash
                            ),
                        }
                        summary.update(common)
                        for row in chunk_rows:
                            row.update(common)
                        for row in event_rows:
                            row.update(common)

                        write_checkpoint(
                            run_dir,
                            summary,
                            chunk_rows,
                            event_rows,
                            freeze_hash,
                        )
                        completed_counter += 1

    (
        actual_summary_df,
        actual_chunk_df,
        actual_event_df,
    ) = load_all_checkpoints(
        checkpoint_root,
        expected_tokens,
        freeze_hash,
    )

    if len(actual_summary_df) != total_runs:
        raise AssertionError(
            f"Expected {total_runs} summary runs, "
            f"found {len(actual_summary_df)}"
        )

    duplicate_keys = [
        "repeat_index",
        "trace_seed",
        "trace",
        "chunk_size",
        "policy",
    ]
    if actual_summary_df.duplicated(
        duplicate_keys
    ).any():
        raise AssertionError(
            "Duplicate actual-audit run keys"
        )

    atomic_write_dataframe(
        output_dir
        / "actual_audit_global_run_records_repeated.csv",
        actual_summary_df,
    )
    atomic_write_dataframe(
        output_dir
        / "actual_audit_chunk_run_records_repeated.csv",
        actual_chunk_df,
    )
    atomic_write_dataframe(
        output_dir
        / "actual_audit_event_diagnostics_repeated.csv",
        actual_event_df,
    )

    exact_ratio_df = build_exact_paired_ratios(
        actual_chunk_df
    )
    comparison_df = (
        build_series_clustered_comparisons(
            exact_ratio_df
        )
    )
    atomic_write_dataframe(
        output_dir
        / "actual_audit_exact_paired_ratios.csv",
        exact_ratio_df,
    )
    atomic_write_dataframe(
        output_dir
        / "actual_audit_series_clustered_comparisons_paired.csv",
        comparison_df,
    )

    (
        no_action_cell_df,
        no_action_assessment,
    ) = build_no_action_consistency(
        exact_ratio_df,
        actual_summary_df,
    )
    atomic_write_dataframe(
        output_dir
        / "actual_audit_no_action_consistency.csv",
        no_action_cell_df,
    )
    write_json(
        output_dir
        / "actual_audit_no_action_consistency.json",
        no_action_assessment,
    )

    model_actual_df = (
        build_repeated_model_actual_validation(
            grid_run_df,
            model_policy_df,
            actual_summary_df,
            selected_config_id,
        )
    )
    atomic_write_dataframe(
        output_dir
        / "model_actual_validation_repeated.csv",
        model_actual_df,
    )

    median_afhc_error = float(
        model_actual_df[
            "absolute_afhc_ratio_error"
        ].median()
    )
    median_repressdb_error = float(
        model_actual_df[
            "absolute_repressdb_ratio_error"
        ].median()
    )
    model_actual_pass = bool(
        median_afhc_error
        <= MODEL_ACTUAL_MEDIAN_ERROR_LIMIT
        and median_repressdb_error
        <= MODEL_ACTUAL_MEDIAN_ERROR_LIMIT
    )

    actual_key_results: Dict[str, Any] = {}
    for trace in core.TRACE_NAMES:
        for chunk_size in core.CHUNK_SIZES:
            actual_key_results[
                f"{trace}|{chunk_size}"
            ] = comparison_lookup(
                comparison_df,
                trace,
                int(chunk_size),
                "repressdb",
                "afhc_frozen",
            )

    development_checks = {
        "D1_exactly_72_grid_configurations": bool(
            grid_summary_df[
                "config_id"
            ].nunique()
            == 72
        ),
        "D2_at_least_one_configuration_passes_unchanged_safety": bool(
            safe_count > 0
        ),
        "D3_one_and_only_one_action_sensitive_configuration_selected": bool(
            grid_summary_df["selected"].sum()
            == 1
        ),
        "D4_action_objective_is_non_degenerate": bool(
            grid_summary_df[
                "action_geometric_mean_ratio"
            ].nunique()
            > 1
        ),
        "D5_three_repeat_audit_complete": bool(
            len(actual_summary_df)
            == total_runs
            and actual_summary_df[
                "repeat_index"
            ].nunique()
            == ACTUAL_REPEATS
        ),
        "D6_no_action_wallclock_consistency": bool(
            no_action_assessment["pass"]
        ),
        "D7_calibrated_replay_tracks_repeated_actual_ratios": bool(
            model_actual_pass
        ),
        "D8_all_actual_queries_passed_truth_checks": True,
        "D9_holdouts_remain_untouched": True,
    }

    a2_go_checks = [
        "D1_exactly_72_grid_configurations",
        "D2_at_least_one_configuration_passes_unchanged_safety",
        "D3_one_and_only_one_action_sensitive_configuration_selected",
        "D4_action_objective_is_non_degenerate",
        "D5_three_repeat_audit_complete",
        "D6_no_action_wallclock_consistency",
        "D8_all_actual_queries_passed_truth_checks",
        "D9_holdouts_remain_untouched",
    ]
    ready_for_a2 = bool(
        all(
            development_checks[key]
            for key in a2_go_checks
        )
    )
    wallclock_primary_allowed = bool(
        ready_for_a2
        and development_checks[
            "D7_calibrated_replay_tracks_repeated_actual_ratios"
        ]
    )

    assessment = {
        "version": VERSION,
        "target_journal": TARGET_JOURNAL,
        "status": (
            "A1 action-sensitive AF-HC-D selection and "
            "three-repeat paired actual audit complete"
        ),
        "study_role": (
            "PEMS04 development selection only. PEMS08 and "
            "BDG2 remain untouched holdouts."
        ),
        "parent_a0_protocol_sha256": (
            parent_protocol_hash
        ),
        "parent_deadband_protocol_sha256": (
            deadband_hash
        ),
        "v074_protocol_sha256": (
            v074_protocol_hash
        ),
        "experiment_freeze_sha256": (
            freeze_hash
        ),
        "selected_configuration": (
            selected_config
        ),
        "selected_action_score": {
            "geometric_mean_ratio": float(
                selected_summary[
                    "action_geometric_mean_ratio"
                ]
            ),
            "action_traces": list(
                ACTION_TRACES
            ),
            "number_of_action_cells": 12,
            "safe_configuration_count": (
                safe_count
            ),
            "configs_in_0_5_percent_tie_band": int(
                selected_summary[
                    "n_configs_in_tie_band"
                ]
            ),
        },
        "actual_audit": {
            "repeats": ACTUAL_REPEATS,
            "expected_policy_runs": (
                total_runs
            ),
            "completed_policy_runs": int(
                len(actual_summary_df)
            ),
            "pairing_level": (
                "repeat × seed × trace × chunk size × series"
            ),
            "repressdb_vs_afhc": (
                actual_key_results
            ),
        },
        "no_action_consistency": (
            no_action_assessment
        ),
        "calibrated_replay_validation": {
            "median_absolute_afhc_ratio_error": (
                median_afhc_error
            ),
            "median_absolute_repressdb_ratio_error": (
                median_repressdb_error
            ),
            "required_maximum": (
                MODEL_ACTUAL_MEDIAN_ERROR_LIMIT
            ),
            "pass": model_actual_pass,
        },
        "development_checks": {
            **development_checks,
            "A2_go_primary_calibrated_replay": (
                ready_for_a2
            ),
            "wallclock_primary_allowed": (
                wallclock_primary_allowed
            ),
        },
        "interpretation_rule": (
            "If A2_go_primary_calibrated_replay is true, "
            "the frozen AF-HC-D configuration may be evaluated "
            "on PEMS08 and BDG2. If wallclock_primary_allowed "
            "is false, calibrated replay remains the primary "
            "performance metric and repeated wall-clock results "
            "are retained as diagnostics."
        ),
        "next_gate": (
            "Run A2 only with this exact hotcold_frozen_config.json "
            "and experiment-freeze hash. Do not modify AF-HC-D or "
            "RePressDB parameters after inspecting holdout results."
            if ready_for_a2
            else (
                "Stop before A2 and inspect failed development "
                "measurement checks. Do not access holdouts."
            )
        ),
    }

    write_json(
        output_dir / "hotcold_dev_assessment_v074.json",
        assessment,
    )
    write_json(
        project_dir / SUMMARY_JSON_NAME,
        assessment,
    )

    run_config = {
        "version": VERSION,
        "target_journal": TARGET_JOURNAL,
        "source_v070_result_directory": str(
            v070_dir
        ),
        "source_v073_result_directory": str(
            v073_dir
        ),
        "parent_a0_protocol_sha256": (
            parent_protocol_hash
        ),
        "parent_deadband_protocol_sha256": (
            deadband_hash
        ),
        "v074_protocol_sha256": (
            v074_protocol_hash
        ),
        "experiment_freeze_sha256": (
            freeze_hash
        ),
        "selection": {
            "action_traces": list(
                ACTION_TRACES
            ),
            "score": (
                "geometric mean of total-cost ratios"
            ),
            "safety_filters_unchanged": True,
            "tie_tolerance": float(
                core.TIE_SCORE_TOLERANCE
            ),
        },
        "actual_audit": {
            "repeats": ACTUAL_REPEATS,
            "full_warmup_each_run": True,
            "random_policy_order_each_repeat": True,
            "checkpoint_resume": True,
            "expected_policy_runs": (
                total_runs
            ),
            "policy_pause_seconds": (
                POLICY_PAUSE_SECONDS
            ),
        },
        "selected_configuration": (
            selected_config
        ),
        "RePressDB_controller": {
            "long_window": core.LONG_WINDOW,
            "short_window": core.SHORT_WINDOW,
            "minimum_observations": (
                core.MIN_OBSERVATIONS
            ),
            "decision_interval": (
                core.DECISION_INTERVAL
            ),
            "confirmation_rounds": (
                core.CONFIRMATION_ROUNDS
            ),
            "cooldown_queries": (
                core.COOLDOWN_QUERIES
            ),
            "prediction_horizon": (
                core.PREDICTION_HORIZON
            ),
            "minimum_relative_improvement": (
                core.MIN_RELATIVE_IMPROVEMENT
            ),
            "rewrite_safety_factor": (
                core.REWRITE_SAFETY_FACTOR
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__(
                "scipy"
            ).__version__,
            "timer": "time.perf_counter_ns",
        },
        "holdout_access": {
            "PEMS08_values_read": False,
            "BDG2_values_read": False,
        },
    }
    write_json(
        output_dir / "run_config.json",
        run_config,
    )

    shutil.rmtree(
        migration_root,
        ignore_errors=True,
    )
    patched_core = (
        output_dir / "_corrected_core_v073.py"
    )
    patched_core.unlink(missing_ok=True)

    print("=" * 112)
    print("RePressDB A0+A1 v0.7.4 complete")
    print(json.dumps(
        assessment["development_checks"],
        ensure_ascii=False,
        indent=2,
    ))
    print(f"Results: {output_dir.resolve()}")
    print("Upload:")
    print("  hotcold_dev_assessment_v074.json")
    print("  hotcold_frozen_config.json")
    print("  hotcold_grid_ranked_v074.csv")
    print("  actual_audit_global_run_records_repeated.csv")
    print("  actual_audit_chunk_run_records_repeated.csv")
    print("  actual_audit_exact_paired_ratios.csv")
    print("  actual_audit_series_clustered_comparisons_paired.csv")
    print("  actual_audit_no_action_consistency.json")
    print("  model_actual_validation_repeated.csv")
    print("  v074_selection_audit_protocol.json")
    print("  v074_protocol_sha256.txt")
    print("  experiment_freeze_sha256.txt")
    print("  run_config.json")
    print("=" * 112)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
