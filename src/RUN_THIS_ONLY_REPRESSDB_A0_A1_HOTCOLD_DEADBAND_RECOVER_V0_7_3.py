#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
RePressDB A0+A1 v0.7.3
AF-HC-D deadband nearest-baseline development and recovery.

Target journal: Journal of Big Data.

DIRECT-RUN
----------
Place this file beside:
  RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_DEV_PEMS04_V0_7_0.py
  RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT_V0_2_0.py
  RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT.py
  RUN_THIS_ONLY_REPRESSDB_E0C_V0_2_0.py

Keep the partially completed directory:
  results_a0_a1_hotcold_pems04_v070/

Open this file in PyCharm and click Run.

Scientific role
---------------
v0.7.2 established that a naive percentile-only access-frequency classifier
is not a defensible nearest baseline: small round-boundary count differences
caused repeated hot/cold layout oscillation even under stable equal-frequency
workloads. No configuration passed the preregistered stable/burst safety gates.

v0.7.3 evaluates AF-HC-D, an access-frequency-only hot/cold baseline with a
fixed multiplicative deadband:

  hot:     score >= 1.5 * median_score
  cold:    score <= median_score / 1.5
  neutral: preserve the current representation

The 1.5 factor is fixed from trace structure, not selected from PEMS04 results.
With 32 equally accessed chunks, a 100-query interval can create at most a
4-versus-3 boundary-count ratio (1.333), which stays inside the deadband.
The frequency-only trace contains a 4:1 access-rate contrast, which remains
detectable.

The original hot_fraction parameter is retained as a maximum hot-set capacity:
among chunks satisfying the hot threshold, only the highest-scoring
ceil(hot_fraction * n_chunks) chunks can be hot. This keeps the original
72-configuration development grid meaningful.

AF-HC-D information boundary
-----------------------------
Allowed:
  * per-chunk access counts;
  * frequency EMA;
  * fixed reclassification period;
  * fixed deadband factor;
  * maximum hot-set fraction;
  * fixed hot and cold codecs.

Forbidden:
  * query classes or query shapes;
  * point/scan/aggregate proportions;
  * RePressDB cost model;
  * dual timescales;
  * predicted saving or break-even;
  * confirmation rounds or cooldown;
  * future phase labels.

Unchanged from the frozen A0 protocol
-------------------------------------
  * PEMS04 development entities;
  * PEMS08 and BDG2 holdout manifests;
  * chunk sizes 8192 and 16384;
  * five trace designs;
  * three trace seeds;
  * 72 configurations;
  * stable ratio <= 1.05;
  * brief-burst ratio <= 1.05;
  * brief-burst median migrations/chunk <= 2;
  * frozen RePressDB controller.

The program reuses the completed v0.7.0:
  calibration_records.csv
  codec_records.csv
  measured_migration_costs.csv
  _candidate_storage/

It does not repeat candidate encoding, query calibration, or directed
migration calibration.
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
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

VERSION = "0.7.3"
TARGET_JOURNAL = "Journal of Big Data"

SOURCE_SCRIPT_NAME = (
    "RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_DEV_PEMS04_V0_7_0.py"
)
SOURCE_RESULT_DIR_NAME = "results_a0_a1_hotcold_pems04_v070"
OUTPUT_DIR_NAME = "results_a0_a1_hotcold_deadband_pems04_v073"
SUMMARY_JSON_NAME = "A0_A1_HOTCOLD_DEADBAND_PEMS04_ASSESSMENT_v073.json"

REQUIRED_SOURCE_FILES = (
    "hotcold_protocol.json",
    "protocol_sha256.txt",
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
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
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
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_corrected_core(
    source_script: Path,
    output_dir: Path,
) -> Any:
    source = source_script.read_text(encoding="utf-8")

    source = source.replace(
        'VERSION = "0.7.0"',
        'VERSION = "0.7.3"',
        1,
    )

    constant_anchor = "FREQUENCY_EMA_ALPHA = 0.50\n"
    if constant_anchor not in source:
        raise RuntimeError("Cannot locate FREQUENCY_EMA_ALPHA")
    source = source.replace(
        constant_anchor,
        constant_anchor + "AFHC_DEADBAND_FACTOR = 1.50\n",
        1,
    )

    old_classifier_start = source.index("def classify_hot_chunks(")
    old_classifier_end = source.index("\ndef simulate_afhc(", old_classifier_start)
    new_classifier = '''def classify_hot_cold_deadband(
    scores: np.ndarray,
    hot_fraction: float,
    deadband_factor: float = AFHC_DEADBAND_FACTOR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Access-frequency-only classification with a fixed multiplicative
    # deadband. Neutral chunks preserve their current layouts.
    values = np.asarray(scores, dtype=float)
    n_chunks = int(values.size)
    hot = np.zeros(n_chunks, dtype=bool)
    cold = np.zeros(n_chunks, dtype=bool)

    if n_chunks == 0:
        return hot, cold, ~(hot | cold)
    if not np.isfinite(values).all():
        raise ValueError("AF-HC-D scores must be finite")
    if deadband_factor <= 1.0:
        raise ValueError("deadband_factor must be greater than 1")

    positive = values[values > 0.0]
    if positive.size == 0:
        return hot, cold, ~(hot | cold)

    median_score = float(np.median(values))
    if median_score <= 0.0:
        median_score = float(np.median(positive))
    if median_score <= 0.0 or not np.isfinite(median_score):
        return hot, cold, ~(hot | cold)

    hot_threshold = float(deadband_factor * median_score)
    cold_threshold = float(median_score / deadband_factor)

    hot_candidates = values >= hot_threshold
    cold = values <= cold_threshold

    maximum_hot = max(
        1,
        min(
            n_chunks,
            int(math.ceil(float(hot_fraction) * n_chunks)),
        ),
    )
    candidate_positions = np.flatnonzero(hot_candidates)

    if candidate_positions.size <= maximum_hot:
        hot[candidate_positions] = True
    elif candidate_positions.size > 0:
        candidate_scores = values[candidate_positions]
        # Deterministic capacity tie-break: higher score first, then the
        # frozen manifest position. The manifest order is fixed before all
        # development and holdout runs.
        order = np.lexsort((candidate_positions, -candidate_scores))
        selected_positions = candidate_positions[order[:maximum_hot]]
        hot[selected_positions] = True

    cold[hot] = False
    neutral = ~(hot | cold)
    return hot, cold, neutral
'''
    source = source[:old_classifier_start] + new_classifier + source[old_classifier_end:]

    old_model_active = "    active = np.full(n_chunks, cold_position, dtype=np.int16)\n"
    new_model_active = (
        "    initial_positions = np.asarray(\n"
        "        config['_initial_codec_positions'], dtype=np.int16\n"
        "    )\n"
        "    if initial_positions.shape != (n_chunks,):\n"
        "        raise ValueError('AF-HC-D initial codec positions have wrong shape')\n"
        "    active = initial_positions.copy()\n"
    )
    if old_model_active not in source:
        raise RuntimeError("Cannot patch AF-HC-D model initialization")
    source = source.replace(old_model_active, new_model_active, 1)

    old_model_classification = (
        "        hot_mask = classify_hot_chunks(scores, hot_fraction)\n"
        "        target = np.where(hot_mask, hot_position, cold_position)\n\n"
        "        for chunk_position in np.flatnonzero(target != active):\n"
    )
    new_model_classification = (
        "        hot_mask, cold_mask, neutral_mask = (\n"
        "            classify_hot_cold_deadband(\n"
        "                scores, hot_fraction, AFHC_DEADBAND_FACTOR\n"
        "            )\n"
        "        )\n"
        "        target = active.copy()\n"
        "        target[hot_mask] = hot_position\n"
        "        target[cold_mask] = cold_position\n"
        "        # Neutral chunks preserve their active layouts.\n\n"
        "        for chunk_position in np.flatnonzero(target != active):\n"
    )
    if old_model_classification not in source:
        raise RuntimeError("Cannot patch AF-HC-D model classification")
    source = source.replace(old_model_classification, new_model_classification, 1)

    old_actual_initialization = (
        '    elif policy == "afhc_frozen":\n'
        "        active = {\n"
        '            chunk_id: str(frozen_config["cold_codec"])\n'
        "            for chunk_id in chunk_ids\n"
        "        }\n"
    )
    new_actual_initialization = (
        '    elif policy == "afhc_frozen":\n'
        "        active = {\n"
        "            chunk_id: e1.choose_best_codec(\n"
        '                profiles[chunk_id], base.WORKLOADS["scan_heavy"]\n'
        "            )[0]\n"
        "            for chunk_id in chunk_ids\n"
        "        }\n"
    )
    if old_actual_initialization not in source:
        raise RuntimeError("Cannot patch AF-HC-D actual initialization")
    source = source.replace(old_actual_initialization, new_actual_initialization, 1)

    old_actual_classification = (
        "                    hot_mask = classify_hot_chunks(\n"
        "                        afhc_scores, hot_fraction\n"
        "                    )\n"
        "                    for position, candidate_chunk_id in enumerate(chunk_ids):\n"
        "                        target = hot_codec if hot_mask[position] else cold_codec\n"
    )
    new_actual_classification = (
        "                    hot_mask, cold_mask, neutral_mask = (\n"
        "                        classify_hot_cold_deadband(\n"
        "                            afhc_scores, hot_fraction, AFHC_DEADBAND_FACTOR\n"
        "                        )\n"
        "                    )\n"
        "                    for position, candidate_chunk_id in enumerate(chunk_ids):\n"
        "                        if hot_mask[position]:\n"
        "                            target = hot_codec\n"
        "                            classification_reason = 'frequency_deadband_hot'\n"
        "                        elif cold_mask[position]:\n"
        "                            target = cold_codec\n"
        "                            classification_reason = 'frequency_deadband_cold'\n"
        "                        else:\n"
        "                            continue\n"
    )
    if old_actual_classification not in source:
        raise RuntimeError("Cannot patch AF-HC-D actual classification")
    source = source.replace(old_actual_classification, new_actual_classification, 1)
    source = source.replace(
        '"reason": "frequency_percentile_reclassification",',
        '"reason": classification_reason,',
        1,
    )

    old_initial_codec_field = (
        '                "initial_codec": (\n'
        '                    str(frozen_config["cold_codec"])\n'
        '                    if policy == "afhc_frozen"\n'
        '                    else e1.choose_best_codec(\n'
        '                        profiles[chunk_id],\n'
        '                        base.WORKLOADS["scan_heavy"],\n'
        '                    )[0]\n'
        '                ),\n'
    )
    new_initial_codec_field = (
        '                "initial_codec": e1.choose_best_codec(\n'
        '                    profiles[chunk_id],\n'
        '                    base.WORKLOADS["scan_heavy"],\n'
        '                )[0],\n'
    )
    if old_initial_codec_field not in source:
        raise RuntimeError("Cannot patch AF-HC-D initial codec record")
    source = source.replace(old_initial_codec_field, new_initial_codec_field, 1)

    old_hot_rule = (
        '"score > percentile(score, 1-hot_fraction); "\n'
        '                                "ties at threshold remain cold"'
    )
    new_hot_rule = (
        '"hot if score >= 1.5*median, cold if score <= median/1.5; "\n'
        '                                "neutral preserves current layout; hot_fraction "\n'
        '                                "caps maximum hot-set size"'
    )
    if old_hot_rule not in source:
        raise RuntimeError("Cannot patch AF-HC-D configuration description")
    source = source.replace(old_hot_rule, new_hot_rule, 1)

    patched_path = output_dir / "_corrected_core_v073.py"
    patched_path.write_text(source, encoding="utf-8")
    compile(source, str(patched_path), "exec")
    return load_module(
        patched_path,
        "repressdb_a0_a1_v073_deadband_core",
    )

def verify_source_directory(source_dir: Path) -> None:
    missing = [
        name
        for name in REQUIRED_SOURCE_FILES
        if not (source_dir / name).exists()
    ]
    storage_dir = source_dir / "_candidate_storage"
    if not storage_dir.exists():
        missing.append("_candidate_storage/")
    if missing:
        raise FileNotFoundError(
            "The v0.7.0 partial result directory is incomplete.\n"
            f"Directory: {source_dir}\n"
            f"Missing: {missing}\n"
            "Do not delete the failed-run result directory."
        )


def recompute_protocol_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for filename in (
        "hotcold_protocol.json",
        "pems04_selected_chunks.csv",
        "pems08_selected_chunks.csv",
        "bdg2_selected_buildings.csv",
        "trace_assignment_manifest.csv",
    ):
        path = directory / filename
        digest.update(filename.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def load_profiles(
    calibration_df: pd.DataFrame,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    profiles: Dict[str, Dict[str, Dict[str, float]]] = {}
    for chunk_id, chunk_group in calibration_df.groupby("chunk_id"):
        local: Dict[str, Dict[str, float]] = {}
        for codec, codec_group in chunk_group.groupby("codec"):
            local[str(codec)] = {
                str(row.query_class): float(row.median_latency_ns)
                for row in codec_group.itertuples(index=False)
            }
        profiles[str(chunk_id)] = local
    return profiles


def load_encode_decode_maps(
    codec_df: pd.DataFrame,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
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


def load_migration_maps(
    migration_df: pd.DataFrame,
) -> Tuple[
    Dict[str, Dict[Tuple[str, str], int]],
    Dict[str, Dict[Tuple[str, str], int]],
]:
    costs: Dict[str, Dict[Tuple[str, str], int]] = {}
    rewrite: Dict[str, Dict[Tuple[str, str], int]] = {}
    for chunk_id, group in migration_df.groupby("chunk_id"):
        cost_local: Dict[Tuple[str, str], int] = {}
        byte_local: Dict[Tuple[str, str], int] = {}
        for row in group.itertuples(index=False):
            key = (str(row.current_codec), str(row.target_codec))
            cost_local[key] = int(row.median_migration_ns)
            byte_local[key] = int(row.rewrite_bytes)
        costs[str(chunk_id)] = cost_local
        rewrite[str(chunk_id)] = byte_local
    return costs, rewrite


def reconstruct_stored_maps(
    core: Any,
    base: Any,
    chunks_by_id: Mapping[str, Any],
    codec_df: pd.DataFrame,
    source_storage_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    stored_by_chunk: Dict[str, Dict[str, Any]] = {}

    for chunk_id, group in codec_df.groupby("chunk_id"):
        chunk_id = str(chunk_id)
        if chunk_id not in chunks_by_id:
            raise KeyError(f"Unknown cached chunk: {chunk_id}")
        chunk = chunks_by_id[chunk_id]
        local: Dict[str, Any] = {}

        for row in group.itertuples(index=False):
            codec = str(row.codec)
            stem = source_storage_dir / chunk_id / codec
            data_path = stem.with_suffix(".bin")
            metadata_path = stem.with_suffix(".meta.bin")

            if not data_path.exists():
                raise FileNotFoundError(
                    f"Cached candidate payload is missing: {data_path}"
                )

            if metadata_path.exists():
                metadata_raw = metadata_path.read_bytes()
                metadata = base.parse_block_metadata(metadata_raw)
                actual_metadata_path = metadata_path
                metadata_size = int(metadata_path.stat().st_size)
            else:
                metadata = {"level": 3} if codec == "whole_zstd" else {}
                actual_metadata_path = None
                metadata_size = 0

            stored = base.StoredChunk(
                codec=codec,
                n=int(chunk.values.size),
                data_path=data_path,
                metadata_path=actual_metadata_path,
                payload_size_bytes=int(data_path.stat().st_size),
                metadata_size_bytes=metadata_size,
                metadata=metadata,
            )
            expected_size = int(row.stored_size_bytes)
            if stored.stored_size_bytes != expected_size:
                raise AssertionError(
                    f"Cached file size mismatch for {chunk_id}/{codec}: "
                    f"{stored.stored_size_bytes} != {expected_size}"
                )
            local[codec] = stored

        stored_by_chunk[chunk_id] = local

    return stored_by_chunk


def summarize_grid_without_hidden_relaxation(
    core: Any,
    grid_run_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for config_id, group in grid_run_df.groupby("config_id", sort=True):
        first = group.iloc[0]
        stable = group[group["trace"] == "stable_scan"]
        burst = group[group["trace"] == "brief_burst"]

        stable_ratio = float(stable["ratio_to_initial_best"].median())
        burst_ratio = float(burst["ratio_to_initial_best"].median())
        burst_migrations = float(
            burst["median_migrations_per_chunk"].median()
        )
        composite = float(group["ratio_to_initial_best"].median())
        overall_migrations = float(
            group["median_migrations_per_chunk"].median()
        )
        rewrite_bytes = float(group["rewrite_bytes"].median())

        reasons: List[str] = []
        if stable_ratio > core.MAX_STABLE_RATIO:
            reasons.append("stable_ratio_above_1.05")
        if burst_ratio > core.MAX_BURST_RATIO:
            reasons.append("burst_ratio_above_1.05")
        if (
            burst_migrations
            > core.MAX_BURST_MEDIAN_MIGRATIONS_PER_CHUNK
        ):
            reasons.append("burst_migrations_per_chunk_above_2")

        row = {
            "config_id": str(config_id),
            "adaptation_period": int(first["adaptation_period"]),
            "hot_fraction": float(first["hot_fraction"]),
            "cold_codec": str(first["cold_codec"]),
            "hot_codec": str(first["hot_codec"]),
            "composite_median_ratio": composite,
            "stable_median_ratio": stable_ratio,
            "burst_median_ratio": burst_ratio,
            "burst_median_migrations_per_chunk": burst_migrations,
            "overall_median_migrations_per_chunk": overall_migrations,
            "overall_median_rewrite_bytes": rewrite_bytes,
            "safety_pass": len(reasons) == 0,
            "failure_reasons": "|".join(reasons),
            "n_safety_failures": int(len(reasons)),
            "stable_excess_over_limit": max(
                0.0, stable_ratio - core.MAX_STABLE_RATIO
            ),
            "burst_excess_over_limit": max(
                0.0, burst_ratio - core.MAX_BURST_RATIO
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
    safe = summary[summary["safety_pass"]].copy()

    if safe.empty:
        diagnostic = summary.sort_values(
            [
                "n_safety_failures",
                "stable_excess_over_limit",
                "burst_excess_over_limit",
                "burst_migration_excess_over_limit",
                "composite_median_ratio",
                "overall_median_migrations_per_chunk",
                "overall_median_rewrite_bytes",
                "adaptation_period",
                "config_id",
            ],
            ascending=[
                True, True, True, True, True, True, True, False, True
            ],
        ).reset_index(drop=True)
        selected = diagnostic.iloc[0].to_dict()
        selected["selection_status"] = "diagnostic_only_no_safe_config"
        return (
            summary.sort_values("config_id").reset_index(drop=True),
            pd.DataFrame(failures),
            diagnostic,
            selected,
        )

    safe = safe.sort_values(
        [
            "composite_median_ratio",
            "overall_median_migrations_per_chunk",
            "overall_median_rewrite_bytes",
            "adaptation_period",
            "config_id",
        ],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)

    best_score = float(safe.iloc[0]["composite_median_ratio"])
    equivalent = safe[
        safe["composite_median_ratio"]
        <= best_score * (1.0 + core.TIE_SCORE_TOLERANCE)
    ].copy()
    equivalent = equivalent.sort_values(
        [
            "overall_median_migrations_per_chunk",
            "overall_median_rewrite_bytes",
            "adaptation_period",
            "config_id",
        ],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)

    selected = equivalent.iloc[0].to_dict()
    selected["selection_status"] = "frozen_safe_configuration"

    rank_map = {
        str(row.config_id): int(index + 1)
        for index, row in enumerate(safe.itertuples(index=False))
    }
    summary["safe_rank_one_based"] = summary["config_id"].map(rank_map)
    summary["selected"] = (
        summary["config_id"] == str(selected["config_id"])
    )
    return (
        summary.sort_values("config_id").reset_index(drop=True),
        pd.DataFrame(failures),
        safe,
        selected,
    )


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    source_script = project_dir / SOURCE_SCRIPT_NAME
    source_dir = project_dir / SOURCE_RESULT_DIR_NAME
    output_dir = project_dir / OUTPUT_DIR_NAME

    if not source_script.exists():
        raise FileNotFoundError(
            f"Missing source program: {source_script}"
        )
    verify_source_directory(source_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 108)
    print("RePressDB A0+A1 v0.7.3 — AF-HC-D deadband development")
    print(f"Target journal: {TARGET_JOURNAL}")
    print("Reuse v0.7.0 calibration; fixed 1.5x deadband; no PEMS08/BDG2 access.")
    print("=" * 108)

    core = build_corrected_core(source_script, output_dir)

    # The generated corrected core physically lives inside the result
    # directory. Its inherited loader resolves dependencies relative to
    # its own __file__, which would incorrectly search the result directory.
    # Point the module back to the original project-level source file so all
    # sibling RePressDB dependency scripts are found in the project root.
    core.__file__ = str(source_script.resolve())

    e1v2, e1, base = core.load_repressdb_modules()

    stored_protocol_hash = (
        source_dir / "protocol_sha256.txt"
    ).read_text(encoding="utf-8").strip()
    recomputed_hash = recompute_protocol_hash(source_dir)
    if stored_protocol_hash != recomputed_hash:
        raise AssertionError(
            "The v0.7.0 protocol files changed after the run.\n"
            f"Stored:     {stored_protocol_hash}\n"
            f"Recomputed: {recomputed_hash}"
        )

    for filename in REQUIRED_SOURCE_FILES:
        shutil.copy2(source_dir / filename, output_dir / filename)

    correction_protocol = {
        "version": VERSION,
        "parent_protocol_sha256": stored_protocol_hash,
        "change_classification": (
            "nearest-baseline redesign after preregistered no-go diagnostic"
        ),
        "baseline_name": "AF-HC-D",
        "deadband_factor": 1.5,
        "deadband_derivation": (
            "With 32 equally accessed chunks and the shortest 100-query "
            "classification interval, deterministic round truncation creates "
            "at most a 4/3 count ratio. A fixed 1.5 factor rejects that "
            "artifact while retaining the protocol's 4:1 frequency contrast."
        ),
        "classification": {
            "hot": "score >= 1.5 * median_score",
            "cold": "score <= median_score / 1.5",
            "neutral": "preserve the current representation",
            "hot_fraction_role": (
                "maximum hot-set capacity among chunks passing the hot threshold; "
                "equal-score capacity ties use frozen manifest order"
            ),
            "frequency_ema_alpha": 0.5,
        },
        "fair_initialization": (
            "AF-HC-D, initial-best, and RePressDB all begin from each "
            "chunk's scan-optimal representation."
        ),
        "information_boundary": {
            "allowed": [
                "per-chunk access counts",
                "frequency EMA",
                "fixed reclassification period",
                "fixed 1.5 deadband",
                "maximum hot-set fraction",
                "fixed hot and cold codecs",
            ],
            "forbidden": [
                "query class or query shape",
                "point/scan/aggregate proportions",
                "dual-timescale estimates",
                "predicted saving",
                "break-even",
                "confirmation rounds",
                "cooldown",
                "future phase labels",
            ],
        },
        "unchanged": {
            "pems04_series_selection": True,
            "pems08_reserved_selection": True,
            "bdg2_reserved_selection": True,
            "chunk_sizes": list(core.CHUNK_SIZES),
            "trace_names": list(core.TRACE_NAMES),
            "trace_seeds": list(core.TRACE_SEEDS),
            "grid_size": 72,
            "adaptation_periods": list(core.ADAPTATION_PERIODS),
            "hot_fractions": list(core.HOT_FRACTIONS),
            "cold_codecs": list(core.COLD_CODECS),
            "hot_codecs": list(core.HOT_CODECS),
            "stable_ratio_limit": core.MAX_STABLE_RATIO,
            "burst_ratio_limit": core.MAX_BURST_RATIO,
            "burst_migration_limit": core.MAX_BURST_MEDIAN_MIGRATIONS_PER_CHUNK,
            "repressdb_controller": "unchanged and frozen",
        },
        "previous_diagnostic": {
            "version": "0.7.2",
            "status": "no safe naive-percentile AF-HC configuration",
            "scientific_use": "diagnostic only; not the nearest baseline",
        },
        "holdout_status": (
            "PEMS08 and BDG2 have not been evaluated under AF-HC-D."
        ),
    }
    write_json(
        output_dir / "v073_deadband_protocol.json",
        correction_protocol,
    )

    corrected_protocol_digest = hashlib.sha256()
    corrected_protocol_digest.update(
        stored_protocol_hash.encode("ascii")
    )
    corrected_protocol_digest.update(
        sha256_file(
            output_dir / "v073_deadband_protocol.json"
        ).encode("ascii")
    )
    deadband_protocol_sha256 = corrected_protocol_digest.hexdigest()
    (
        output_dir / "deadband_protocol_sha256.txt"
    ).write_text(
        deadband_protocol_sha256 + "\n",
        encoding="utf-8",
    )

    # Rebuild exactly the same PEMS04 chunks from the frozen A0 manifest.
    pems04_manifest = pd.read_csv(
        source_dir / "pems04_selected_chunks.csv"
    )
    assignments = pd.read_csv(
        source_dir / "trace_assignment_manifest.csv"
    )
    calibration_df = pd.read_csv(
        source_dir / "calibration_records.csv"
    )
    codec_df = pd.read_csv(
        source_dir / "codec_records.csv"
    )
    migration_df = pd.read_csv(
        source_dir / "measured_migration_costs.csv"
    )

    pems04_dir, _ = base.default_data_dirs()
    pems04_file = base.auto_detect_data_file(
        pems04_dir, "PEMS04_AFHC_RECOVERY"
    )
    matrix, metadata = base.load_dataset(
        pems04_file,
        feature_index=0,
        max_series=core.N_SERIES,
        seed=core.SEED,
        excluded_series=None,
    )
    chunks, rebuilt_manifest = core.deterministic_series_chunks(
        base,
        matrix,
        "PEMS04",
        metadata["selected_original_series_indices"],
        chunk_sizes=core.CHUNK_SIZES,
    )

    compare_columns = [
        "dataset",
        "chunk_id",
        "original_series_index",
        "chunk_start",
        "chunk_size",
    ]
    expected_manifest = pems04_manifest[
        compare_columns
    ].sort_values(compare_columns).reset_index(drop=True)
    actual_manifest = rebuilt_manifest[
        compare_columns
    ].sort_values(compare_columns).reset_index(drop=True)
    if not expected_manifest.equals(actual_manifest):
        raise AssertionError(
            "Rebuilt PEMS04 chunk manifest differs from the frozen A0 manifest."
        )

    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    chunks_by_size = {
        int(size): sorted(
            [
                chunk
                for chunk in chunks
                if int(chunk.chunk_size) == int(size)
            ],
            key=lambda item: item.chunk_id,
        )
        for size in core.CHUNK_SIZES
    }

    reps = base.representation_registry(3)
    codecs = tuple(sorted(reps))
    profiles = load_profiles(calibration_df)
    (
        encode_ns_by_chunk,
        decode_ns_by_chunk,
    ) = load_encode_decode_maps(codec_df)
    (
        migration_costs,
        migration_bytes,
    ) = load_migration_maps(migration_df)

    stored_by_chunk = reconstruct_stored_maps(
        core,
        base,
        chunks_by_id,
        codec_df,
        source_dir / "_candidate_storage",
    )

    query_pools: Dict[str, Dict[str, List[Any]]] = {}
    for chunk_index, chunk in enumerate(chunks, start=1):
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
                if query.query_class == query_class
            ]
            for query_class in core.QUERY_CLASSES
        }
        if any(not values for values in pool.values()):
            raise AssertionError(
                f"Incomplete query pool: {chunk.chunk_id}"
            )
        query_pools[chunk.chunk_id] = pool

        if (
            chunk_index == 1
            or chunk_index == len(chunks)
            or chunk_index % max(1, len(chunks) // 10) == 0
        ):
            print(
                f"[recovery] query pools {chunk_index}/{len(chunks)}"
            )

    print(
        "[recovery] calibration, candidate files, and migration costs loaded."
    )
    print(
        "[recovery] restarting only the corrected model grid and actual audit."
    )

    grid_configs = core.config_grid()
    grid_run_rows: List[Dict[str, Any]] = []
    model_policy_rows: List[Dict[str, Any]] = []
    trace_manifest_rows: List[Dict[str, Any]] = []

    for chunk_size in core.CHUNK_SIZES:
        size_chunks = chunks_by_size[int(chunk_size)]
        size_chunk_ids = tuple(
            chunk.chunk_id for chunk in size_chunks
        )
        profiles_array = core.profile_arrays(
            profiles,
            size_chunk_ids,
            codecs,
        )
        initial_positions = core.scan_best_codec_positions(
            e1,
            base,
            profiles,
            size_chunk_ids,
            codecs,
        )
        size_assignments = assignments[
            (assignments["dataset"] == "PEMS04")
            & (assignments["chunk_size"] == int(chunk_size))
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

                trace_digest = hashlib.sha256()
                for array in (
                    plan.chunk_positions,
                    plan.local_indices,
                    plan.query_class_positions,
                    plan.query_choice_positions,
                    plan.workload_positions,
                ):
                    trace_digest.update(array.tobytes())
                trace_manifest_rows.append(
                    {
                        "dataset": "PEMS04",
                        "trace": trace_name,
                        "trace_seed": int(trace_seed),
                        "chunk_size": int(chunk_size),
                        "n_chunks": int(len(size_chunks)),
                        "n_global_queries": int(plan.n_events),
                        "trace_sha256": trace_digest.hexdigest(),
                    }
                )

                initial_result = core.simulate_initial_best(
                    plan,
                    profiles_array,
                    initial_positions,
                )
                repress_result = core.simulate_repressdb(
                    e1,
                    e1v2,
                    base,
                    plan,
                    profiles,
                    encode_ns_by_chunk,
                    decode_ns_by_chunk,
                    migration_costs,
                    migration_bytes,
                )
                for policy, result in (
                    ("initial_best", initial_result),
                    ("repressdb", repress_result),
                ):
                    model_policy_rows.append(
                        {
                            "trace": trace_name,
                            "trace_seed": int(trace_seed),
                            "chunk_size": int(chunk_size),
                            "policy": policy,
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
                        }
                    )

                for config in grid_configs:
                    runtime_config = dict(config)
                    runtime_config[
                        "_initial_codec_positions"
                    ] = initial_positions
                    result = core.simulate_afhc(
                        plan,
                        profiles_array,
                        codecs,
                        migration_costs,
                        migration_bytes,
                        runtime_config,
                    )
                    ratio = float(
                        result["total_time_ns"]
                        / initial_result["total_time_ns"]
                    )
                    grid_run_rows.append(
                        {
                            **config,
                            "trace": trace_name,
                            "trace_seed": int(trace_seed),
                            "chunk_size": int(chunk_size),
                            "n_chunks": int(len(size_chunks)),
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
                            "initial_best_total_time_ns": float(
                                initial_result["total_time_ns"]
                            ),
                            "ratio_to_initial_best": ratio,
                            "migration_count": int(
                                result["migration_count"]
                            ),
                            "median_migrations_per_chunk": float(
                                np.median(
                                    result[
                                        "per_chunk_migration_count"
                                    ]
                                )
                            ),
                            "mean_migrations_per_chunk": float(
                                np.mean(
                                    result[
                                        "per_chunk_migration_count"
                                    ]
                                )
                            ),
                            "rewrite_bytes": int(
                                result["rewrite_bytes"]
                            ),
                        }
                    )

                print(
                    f"[A1 corrected model] size={chunk_size}, "
                    f"trace={trace_name}, seed={trace_seed}, "
                    f"queries={plan.n_events}"
                )

    grid_run_df = pd.DataFrame(grid_run_rows)
    model_policy_df = pd.DataFrame(model_policy_rows)
    trace_manifest_df = pd.DataFrame(trace_manifest_rows)

    # Write raw results before selection so another exception cannot erase them.
    grid_run_df.to_csv(
        output_dir / "hotcold_grid_all_results.csv",
        index=False,
    )
    model_policy_df.to_csv(
        output_dir / "model_repressdb_initial_results.csv",
        index=False,
    )
    trace_manifest_df.to_csv(
        output_dir / "generated_trace_manifest.csv",
        index=False,
    )

    (
        grid_summary_df,
        grid_failures_df,
        grid_ranked_df,
        selected_summary,
    ) = summarize_grid_without_hidden_relaxation(
        core,
        grid_run_df,
    )

    grid_summary_df.to_csv(
        output_dir / "hotcold_grid_config_summary.csv",
        index=False,
    )
    grid_failures_df.to_csv(
        output_dir / "hotcold_grid_safety_failures.csv",
        index=False,
    )
    grid_ranked_df.to_csv(
        output_dir / "hotcold_grid_ranked.csv",
        index=False,
    )

    safe_count = int(grid_summary_df["safety_pass"].sum())
    print("-" * 108)
    print(f"[A1 corrected] safe configurations: {safe_count}/72")

    if safe_count == 0:
        diagnostic = {
            "version": VERSION,
            "status": "no_safe_afhc_deadband_configuration",
            "deadband_protocol_sha256": deadband_protocol_sha256,
            "best_diagnostic_configuration": selected_summary,
            "scientific_decision": (
                "Stop AF-HC-D development cleanly. Do not relax the "
                "predeclared safety thresholds and do not run PEMS08 or BDG2."
            ),
        }
        write_json(
            output_dir / "hotcold_no_safe_config_diagnostic.json",
            diagnostic,
        )
        write_json(
            project_dir / SUMMARY_JSON_NAME,
            diagnostic,
        )
        print(json.dumps(diagnostic, ensure_ascii=False, indent=2))
        print(f"Results: {output_dir.resolve()}")
        return 2

    selected_config_id = str(selected_summary["config_id"])
    selected_config = next(
        dict(config)
        for config in grid_configs
        if str(config["config_id"]) == selected_config_id
    )
    selected_config.update(
        {
            "deadband_protocol_sha256": deadband_protocol_sha256,
            "parent_protocol_sha256": stored_protocol_hash,
            "selection_dataset": "PEMS04",
            "selection_version": VERSION,
            "selection_summary": {
                key: json_safe(value)
                for key, value in selected_summary.items()
            },
            "frozen_for_holdouts": True,
            "deadband_rule": (
                "hot >= 1.5*median; cold <= median/1.5; "
                "neutral preserves current layout"
            ),
            "deadband_factor": 1.5,
            "initialization": (
                "per-chunk scan-optimal representation, identical to "
                "initial-best and RePressDB"
            ),
        }
    )
    write_json(
        output_dir / "hotcold_frozen_config.json",
        selected_config,
    )

    freeze_digest = hashlib.sha256()
    freeze_digest.update(deadband_protocol_sha256.encode("ascii"))
    for path in (
        output_dir / "generated_trace_manifest.csv",
        output_dir / "hotcold_frozen_config.json",
        Path(__file__).resolve(),
    ):
        freeze_digest.update(path.name.encode("utf-8"))
        freeze_digest.update(sha256_file(path).encode("ascii"))
    freeze_sha256 = freeze_digest.hexdigest()
    (
        output_dir / "experiment_freeze_sha256.txt"
    ).write_text(freeze_sha256 + "\n", encoding="utf-8")
    selected_config["experiment_freeze_sha256"] = freeze_sha256
    write_json(
        output_dir / "hotcold_frozen_config.json",
        selected_config,
    )

    print("[A1 corrected] frozen AF-HC configuration:")
    print(json.dumps(selected_config, ensure_ascii=False, indent=2))
    print("-" * 108)

    # Actual file-backed audit.
    actual_migration_dir = output_dir / "_actual_audit_migrations"
    actual_migration_dir.mkdir(parents=True, exist_ok=True)

    actual_summary_rows: List[Dict[str, Any]] = []
    actual_chunk_rows: List[Dict[str, Any]] = []
    actual_event_rows: List[Dict[str, Any]] = []

    for chunk_size in core.CHUNK_SIZES:
        size_chunks = chunks_by_size[int(chunk_size)]
        size_assignments = assignments[
            (assignments["dataset"] == "PEMS04")
            & (assignments["chunk_size"] == int(chunk_size))
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
                policy_order = list(core.AUDIT_POLICIES)
                random.Random(
                    core.stable_seed(
                        trace_seed,
                        chunk_size,
                        trace_name,
                        "actual_policy_order_v071",
                    )
                ).shuffle(policy_order)

                for policy in policy_order:
                    print(
                        f"[A1 corrected actual] size={chunk_size}, "
                        f"trace={trace_name}, seed={trace_seed}, "
                        f"policy={policy}, queries={plan.n_events}"
                    )
                    summary, chunk_rows, event_rows = (
                        core.run_actual_policy(
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
                            actual_migration_dir,
                        )
                    )
                    actual_summary_rows.append(summary)
                    actual_chunk_rows.extend(chunk_rows)
                    actual_event_rows.extend(event_rows)

    actual_summary_df = pd.DataFrame(actual_summary_rows)
    actual_chunk_df = pd.DataFrame(actual_chunk_rows)
    actual_event_df = pd.DataFrame(actual_event_rows)
    actual_comparison_df = core.actual_clustered_comparisons(
        actual_chunk_df
    )
    model_actual_validation_df = (
        core.build_model_actual_validation(
            grid_run_df,
            model_policy_df,
            actual_summary_df,
            selected_config_id,
        )
    )

    actual_summary_df.to_csv(
        output_dir / "actual_audit_global_run_records.csv",
        index=False,
    )
    actual_chunk_df.to_csv(
        output_dir / "actual_audit_chunk_run_records.csv",
        index=False,
    )
    actual_event_df.to_csv(
        output_dir / "actual_audit_event_diagnostics.csv",
        index=False,
    )
    actual_comparison_df.to_csv(
        output_dir
        / "actual_audit_series_clustered_comparisons.csv",
        index=False,
    )
    model_actual_validation_df.to_csv(
        output_dir / "model_actual_validation.csv",
        index=False,
    )

    assessment = core.build_assessment(
        selected_config,
        grid_summary_df,
        grid_run_df,
        actual_summary_df,
        actual_comparison_df,
        model_actual_validation_df,
        deadband_protocol_sha256,
    )
    assessment["version"] = VERSION
    assessment["parent_protocol_sha256"] = stored_protocol_hash
    assessment["deadband_protocol_sha256"] = (
        deadband_protocol_sha256
    )
    assessment["implementation_repair"] = (
        correction_protocol
    )
    assessment["safe_configuration_count"] = safe_count

    write_json(
        output_dir / "hotcold_dev_assessment.json",
        assessment,
    )
    write_json(
        project_dir / SUMMARY_JSON_NAME,
        assessment,
    )

    run_config = {
        "version": VERSION,
        "target_journal": TARGET_JOURNAL,
        "parent_protocol_sha256": stored_protocol_hash,
        "deadband_protocol_sha256": deadband_protocol_sha256,
        "experiment_freeze_sha256": freeze_sha256,
        "resumed_from": str(source_dir),
        "reused_artifacts": [
            "calibration_records.csv",
            "codec_records.csv",
            "measured_migration_costs.csv",
            "_candidate_storage/",
        ],
        "afhc_grid": {
            "adaptation_periods": list(core.ADAPTATION_PERIODS),
            "hot_fractions": list(core.HOT_FRACTIONS),
            "cold_codecs": list(core.COLD_CODECS),
            "hot_codecs": list(core.HOT_CODECS),
            "frequency_ema_alpha": core.FREQUENCY_EMA_ALPHA,
            "number_of_configurations": 72,
        },
        "safety_rules": {
            "maximum_stable_ratio": core.MAX_STABLE_RATIO,
            "maximum_burst_ratio": core.MAX_BURST_RATIO,
            "maximum_burst_median_migrations_per_chunk": (
                core.MAX_BURST_MEDIAN_MIGRATIONS_PER_CHUNK
            ),
            "tie_score_tolerance": core.TIE_SCORE_TOLERANCE,
        },
        "afhc_deadband_design": {
            "same_scan_optimal_initialization": True,
            "deadband_factor": 1.5,
            "neutral_preserves_current_layout": True,
            "hot_fraction_is_capacity_cap": True,
            "capacity_tie_break": "frozen manifest order",
        },
        "selected_configuration": selected_config,
        "repressdb_frozen_controller": {
            "long_window": core.LONG_WINDOW,
            "short_window": core.SHORT_WINDOW,
            "minimum_observations": core.MIN_OBSERVATIONS,
            "decision_interval": core.DECISION_INTERVAL,
            "confirmation_rounds": core.CONFIRMATION_ROUNDS,
            "cooldown_queries": core.COOLDOWN_QUERIES,
            "prediction_horizon": core.PREDICTION_HORIZON,
            "minimum_relative_improvement": (
                core.MIN_RELATIVE_IMPROVEMENT
            ),
            "rewrite_safety_factor": core.REWRITE_SAFETY_FACTOR,
        },
        "trace_seeds": list(core.TRACE_SEEDS),
        "chunk_sizes": list(core.CHUNK_SIZES),
        "trace_names": list(core.TRACE_NAMES),
        "representations": list(codecs),
        "primary_execution_mode": "warm_slicedmeta",
        "actual_audit": (
            "Real file-backed replay with actual decode+encode+fsync "
            "migration and every query checked against NumPy truth."
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "timer": "time.perf_counter_ns",
        },
    }
    write_json(
        output_dir / "run_config.json",
        run_config,
    )

    shutil.rmtree(actual_migration_dir, ignore_errors=True)
    patched_core_path = output_dir / "_corrected_core_v073.py"
    patched_core_path.unlink(missing_ok=True)

    print("=" * 108)
    print("A0+A1 v0.7.3 AF-HC-D development complete")
    print(
        json.dumps(
            assessment.get("development_checks", {}),
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Results: {output_dir.resolve()}")
    print("Upload:")
    print("  hotcold_dev_assessment.json")
    print("  hotcold_frozen_config.json")
    print("  hotcold_grid_ranked.csv")
    print("  hotcold_grid_all_results.csv")
    print("  actual_audit_series_clustered_comparisons.csv")
    print("  actual_audit_global_run_records.csv")
    print("  actual_audit_chunk_run_records.csv")
    print("  model_actual_validation.csv")
    print("  deadband_protocol_sha256.txt")
    print("  experiment_freeze_sha256.txt")
    print("  v073_deadband_protocol.json")
    print("  run_config.json")
    print("=" * 108)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
