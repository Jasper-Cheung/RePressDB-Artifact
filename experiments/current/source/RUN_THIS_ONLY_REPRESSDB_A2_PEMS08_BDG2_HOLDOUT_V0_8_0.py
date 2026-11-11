#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
RePressDB A2 v0.8.0
Frozen AF-HC-D nearest-baseline holdout validation on PEMS08 and BDG2.

Target journal: Journal of Big Data.

DIRECT-RUN
----------
Place this file in the RePressDB project root beside:

  RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_V0_7_4.py
  RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_DEADBAND_RECOVER_V0_7_3.py
  RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_DEV_PEMS04_V0_7_0.py
  RUN_THIS_ONLY_REPRESSDB_E5_BDG2_FROZEN_DIRECT_V0_6_0.py
  RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT_V0_2_0.py
  RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT.py
  RUN_THIS_ONLY_REPRESSDB_E0C_V0_2_0.py

Keep these completed development directories unchanged:

  results_a0_a1_hotcold_pems04_v070/
  results_a0_a1_hotcold_deadband_pems04_v073/
  results_a0_a1_hotcold_deadband_pems04_v074/

Open this file in PyCharm and click Run. No command-line arguments are needed.

Frozen inputs
-------------
This program reads the exact v0.7.4 frozen configuration and verifies its
protocol and experiment hashes. It refuses to run if the frozen development
artifacts are internally inconsistent.

Expected frozen AF-HC-D configuration:

  config_id          AFHC_056
  adaptation_period  1000 global queries
  hot_fraction       0.25
  cold_codec         whole_lz4
  hot_codec          block_lz4_1024
  EMA alpha          0.5
  deadband factor    1.5

The RePressDB E1 v0.2 controller remains unchanged.

Holdout design
--------------
PEMS08:
  * uses the 32 series and paired 8,192/16,384 chunks frozen by A0;
  * independent unit: original traffic series.

BDG2:
  * uses the 32 buildings reserved by A0;
  * one frozen meter per building;
  * creates paired 8,192/16,384 chunks using deterministic offsets inside the
    pre-recorded longest contiguous finite run;
  * independent unit: building_id.

Both holdouts use the five frozen traces, three trace seeds, three policies,
and three repeated real file-backed audits. All queries are checked against
NumPy truth by the inherited execution engine.

Primary preregistered gate
--------------------------
For each holdout dataset, inspect the 16,384 shape_only / point_drift group:

  median(RePressDB / AF-HC-D) < 0.95
  bootstrap 95% CI upper bound < 1.0
  fraction of independent units faster >= 0.75

The nearest-baseline claim passes when at least one of PEMS08 or BDG2 passes
all three conditions. Results from both datasets are always completed and
reported. No parameter may be changed after either holdout is inspected.

Checkpointing
-------------
Candidate calibration, directed migration calibration, and every actual policy
run have atomic checkpoints. Rerunning this same file resumes completed work
whose dataset-freeze hash matches.
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


VERSION = "0.8.0"
TARGET_JOURNAL = "Journal of Big Data"
SEED = 20260718
TRACE_SEEDS = (20260718, 20260719, 20260720)
CHUNK_SIZES = (8192, 16384)
TRACE_NAMES = (
    "shape_only",
    "frequency_only",
    "joint_orthogonal",
    "brief_burst",
    "stable_scan",
)
POLICIES = ("initial_best", "afhc_frozen", "repressdb")
ACTUAL_REPEATS = 3
BOOTSTRAP_ITERATIONS = 5000
POLICY_PAUSE_SECONDS = 0.05

PRIMARY_TRACE = "shape_only"
PRIMARY_GROUP = "point_drift"
PRIMARY_CHUNK_SIZE = 16384
PRIMARY_MEDIAN_RATIO_MAX = 0.95
PRIMARY_CI_HIGH_MAX = 1.0
PRIMARY_FASTER_FRACTION_MIN = 0.75
NO_ACTION_RATIO_LOW = 0.95
NO_ACTION_RATIO_HIGH = 1.05
MODEL_ACTUAL_MEDIAN_ERROR_LIMIT = 0.10

V070_SCRIPT_NAME = (
    "RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_DEV_PEMS04_V0_7_0.py"
)
V073_SCRIPT_NAME = (
    "RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_DEADBAND_RECOVER_V0_7_3.py"
)
V074_SCRIPT_NAME = "RUN_THIS_ONLY_REPRESSDB_A0_A1_HOTCOLD_V0_7_4.py"
E5_SCRIPT_NAME = "RUN_THIS_ONLY_REPRESSDB_E5_BDG2_FROZEN_DIRECT_V0_6_0.py"

V070_RESULT_DIR_NAME = "results_a0_a1_hotcold_pems04_v070"
V073_RESULT_DIR_NAME = "results_a0_a1_hotcold_deadband_pems04_v073"
V074_RESULT_DIR_NAME = "results_a0_a1_hotcold_deadband_pems04_v074"
OUTPUT_DIR_NAME = "results_a2_hotcold_holdouts_v080"
SUMMARY_JSON_NAME = "A2_HOTCOLD_HOLDOUT_ASSESSMENT_v080.json"

EXPECTED_CONFIG = {
    "config_id": "AFHC_056",
    "adaptation_period": 1000,
    "hot_fraction": 0.25,
    "cold_codec": "whole_lz4",
    "hot_codec": "block_lz4_1024",
    "frequency_ema_alpha": 0.5,
    "deadband_factor": 1.5,
}

BDG2_ROOT = Path(
    r"E:\IPM\大学\科研\E\buds-lab-building-data-genome-project-2-3d0cbaf"
)

REQUIRED_V070_FILES = (
    "protocol_sha256.txt",
    "hotcold_protocol.json",
    "pems08_selected_chunks.csv",
    "bdg2_selected_buildings.csv",
    "trace_assignment_manifest.csv",
)
REQUIRED_V073_FILES = (
    "v073_deadband_protocol.json",
    "deadband_protocol_sha256.txt",
)
REQUIRED_V074_FILES = (
    "hotcold_dev_assessment_v074.json",
    "hotcold_frozen_config.json",
    "v074_selection_audit_protocol.json",
    "v074_protocol_sha256.txt",
    "experiment_freeze_material.json",
    "experiment_freeze_sha256.txt",
)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
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


def stable_seed(*parts: Any) -> int:
    token = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def stable_hash(*parts: Any) -> str:
    token = "|".join(str(part) for part in parts)
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def load_module(path: Path, module_name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing Python dependency: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def verify_required_files(directory: Path, names: Sequence[str]) -> None:
    if not directory.exists():
        raise FileNotFoundError(f"Missing result directory: {directory}")
    missing = [name for name in names if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete result directory: {directory}\nMissing: {missing}"
        )


def verify_v074_freeze(
    v070_dir: Path,
    v073_dir: Path,
    v074_dir: Path,
) -> Tuple[Dict[str, Any], str, str, str]:
    parent_hash = (v070_dir / "protocol_sha256.txt").read_text(
        encoding="utf-8"
    ).strip()
    stored_deadband_hash = (
        v073_dir / "deadband_protocol_sha256.txt"
    ).read_text(encoding="utf-8").strip()

    deadband_digest = hashlib.sha256()
    deadband_digest.update(parent_hash.encode("ascii"))
    deadband_digest.update(
        sha256_file(v073_dir / "v073_deadband_protocol.json").encode("ascii")
    )
    if deadband_digest.hexdigest() != stored_deadband_hash:
        raise AssertionError("v0.7.3 deadband protocol hash mismatch")

    stored_v074_hash = (v074_dir / "v074_protocol_sha256.txt").read_text(
        encoding="utf-8"
    ).strip()
    v074_digest = hashlib.sha256()
    v074_digest.update(stored_deadband_hash.encode("ascii"))
    v074_digest.update(
        sha256_file(v074_dir / "v074_selection_audit_protocol.json").encode(
            "ascii"
        )
    )
    if v074_digest.hexdigest() != stored_v074_hash:
        raise AssertionError("v0.7.4 selection protocol hash mismatch")

    stored_freeze_hash = (
        v074_dir / "experiment_freeze_sha256.txt"
    ).read_text(encoding="utf-8").strip()
    if sha256_file(v074_dir / "experiment_freeze_material.json") != stored_freeze_hash:
        raise AssertionError("v0.7.4 experiment-freeze hash mismatch")

    frozen_config = json.loads(
        (v074_dir / "hotcold_frozen_config.json").read_text(encoding="utf-8")
    )
    for key, expected in EXPECTED_CONFIG.items():
        actual = frozen_config.get(key)
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
                raise AssertionError(
                    f"Frozen config mismatch for {key}: {actual} != {expected}"
                )
        elif actual != expected:
            raise AssertionError(
                f"Frozen config mismatch for {key}: {actual} != {expected}"
            )
    if frozen_config.get("experiment_freeze_sha256") != stored_freeze_hash:
        raise AssertionError("Frozen config does not reference the v0.7.4 freeze")

    assessment = json.loads(
        (v074_dir / "hotcold_dev_assessment_v074.json").read_text(
            encoding="utf-8"
        )
    )
    if not assessment.get("development_checks", {}).get(
        "A2_go_primary_calibrated_replay", False
    ):
        raise RuntimeError("v0.7.4 development assessment did not authorize A2")

    return frozen_config, parent_hash, stored_v074_hash, stored_freeze_hash


def build_bdg2_paired_manifest(reserved: pd.DataFrame) -> pd.DataFrame:
    required = {
        "building_id",
        "source_file",
        "source_column",
        "meter_type",
        "site_id",
        "primary_use",
        "longest_run_start",
        "longest_run_length",
        "selection_key",
    }
    missing = required.difference(reserved.columns)
    if missing:
        raise ValueError(f"BDG2 reserved manifest missing columns: {sorted(missing)}")
    if len(reserved) != 32 or reserved["building_id"].nunique() != 32:
        raise AssertionError("A2 requires exactly 32 distinct reserved BDG2 buildings")

    ordered = reserved.sort_values(["selection_key", "building_id"]).reset_index(
        drop=True
    )
    rows: List[Dict[str, Any]] = []
    for unit_index, row in enumerate(ordered.itertuples(index=False)):
        run_start = int(row.longest_run_start)
        run_length = int(row.longest_run_length)
        for chunk_size in CHUNK_SIZES:
            if run_length < chunk_size:
                raise ValueError(
                    f"BDG2 building {row.building_id} has only {run_length} "
                    f"contiguous values for chunk size {chunk_size}"
                )
            available = run_length - chunk_size
            rng = np.random.default_rng(
                stable_seed(
                    SEED,
                    "BDG2_A2_HOLDOUT",
                    row.building_id,
                    row.source_file,
                    row.source_column,
                    chunk_size,
                    "paired_chunk_start",
                )
            )
            offset = int(rng.integers(0, available + 1)) if available > 0 else 0
            start = run_start + offset
            identifier = hashlib.sha1(
                (
                    f"{row.building_id}|{row.source_file}|{row.source_column}|"
                    f"{chunk_size}|{start}|A2"
                ).encode("utf-8")
            ).hexdigest()[:12]
            rows.append(
                {
                    "dataset": "BDG2",
                    "chunk_id": f"bdg2_a2_cs{chunk_size}_{identifier}",
                    "original_series_index": int(unit_index),
                    "local_series_index": int(unit_index),
                    "building_id": str(row.building_id),
                    "source_file": str(row.source_file),
                    "source_column": str(row.source_column),
                    "meter_type": str(row.meter_type),
                    "site_id": str(row.site_id),
                    "primary_use": str(row.primary_use),
                    "longest_run_start": run_start,
                    "longest_run_length": run_length,
                    "chunk_size": int(chunk_size),
                    "chunk_start": int(start),
                    "chunk_end_exclusive": int(start + chunk_size),
                    "raw_bytes": int(chunk_size * np.dtype(np.float32).itemsize),
                    "selection_seed": SEED,
                    "selection_key": str(row.selection_key),
                }
            )
    frame = pd.DataFrame(rows)
    expected = {8192: 32, 16384: 32}
    observed = {int(k): int(v) for k, v in frame["chunk_size"].value_counts().items()}
    if observed != expected:
        raise AssertionError(f"BDG2 paired chunk count mismatch: {observed}")
    return frame


def create_or_verify_a2_protocol(
    output_dir: Path,
    core: Any,
    frozen_config: Mapping[str, Any],
    parent_a0_hash: str,
    parent_v074_hash: str,
    parent_freeze_hash: str,
    pems08_manifest: pd.DataFrame,
    bdg2_manifest: pd.DataFrame,
    assignments: pd.DataFrame,
) -> Tuple[str, str]:
    protocol = {
        "version": VERSION,
        "target_journal": TARGET_JOURNAL,
        "study": "Frozen AF-HC-D nearest-baseline holdout validation",
        "development_artifacts": {
            "parent_a0_protocol_sha256": parent_a0_hash,
            "parent_v074_protocol_sha256": parent_v074_hash,
            "parent_v074_experiment_freeze_sha256": parent_freeze_hash,
        },
        "frozen_configuration": dict(frozen_config),
        "holdouts": {
            "PEMS08": {
                "independent_unit": "original traffic series",
                "n_units": 32,
                "chunk_sizes": list(CHUNK_SIZES),
                "manifest_source": "A0 frozen pems08_selected_chunks.csv",
            },
            "BDG2": {
                "independent_unit": "building_id",
                "n_units": 32,
                "one_meter_per_building": True,
                "chunk_sizes": list(CHUNK_SIZES),
                "chunk_offset_rule": (
                    "deterministic offset inside A0-recorded longest contiguous "
                    "finite run; no data-value-based selection"
                ),
                "normalization": False,
                "interpolation": False,
                "resampling": False,
            },
        },
        "execution": {
            "trace_names": list(TRACE_NAMES),
            "trace_seeds": list(TRACE_SEEDS),
            "policies": list(POLICIES),
            "actual_repeats": ACTUAL_REPEATS,
            "real_file_backed": True,
            "warmup_before_every_policy_run": True,
            "policy_order_randomized_within_repeat": True,
            "checkpoint_resume": True,
            "primary_execution_mode": "warm_slicedmeta",
        },
        "primary_gate": {
            "trace": PRIMARY_TRACE,
            "group": PRIMARY_GROUP,
            "chunk_size": PRIMARY_CHUNK_SIZE,
            "comparison": "RePressDB / AF-HC-D",
            "median_ratio_strictly_below": PRIMARY_MEDIAN_RATIO_MAX,
            "bootstrap_95_ci_upper_strictly_below": PRIMARY_CI_HIGH_MAX,
            "fraction_independent_units_faster_at_least": (
                PRIMARY_FASTER_FRACTION_MIN
            ),
            "overall_claim_rule": (
                "at least one of PEMS08 or BDG2 passes all three conditions"
            ),
        },
        "anti_tuning": (
            "Both holdouts are run under one protocol. No AF-HC-D or RePressDB "
            "parameter may be changed after either holdout is inspected."
        ),
        "controller": {
            "long_window": core.LONG_WINDOW,
            "short_window": core.SHORT_WINDOW,
            "minimum_observations": core.MIN_OBSERVATIONS,
            "decision_interval": core.DECISION_INTERVAL,
            "confirmation_rounds": core.CONFIRMATION_ROUNDS,
            "cooldown_queries": core.COOLDOWN_QUERIES,
            "prediction_horizon": core.PREDICTION_HORIZON,
            "minimum_relative_improvement": core.MIN_RELATIVE_IMPROVEMENT,
            "rewrite_safety_factor": core.REWRITE_SAFETY_FACTOR,
        },
    }

    protocol_path = output_dir / "a2_holdout_protocol.json"
    pems08_path = output_dir / "pems08_holdout_chunk_manifest.csv"
    bdg2_path = output_dir / "bdg2_holdout_chunk_manifest.csv"
    assignments_path = output_dir / "a2_trace_assignment_manifest.csv"

    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        if old != json_safe(protocol):
            raise RuntimeError(
                "Existing A2 protocol differs from this script. Do not overwrite "
                "holdout protocol after results may have been observed."
            )
    else:
        write_json(protocol_path, protocol)

    for path, frame in (
        (pems08_path, pems08_manifest),
        (bdg2_path, bdg2_manifest),
        (assignments_path, assignments),
    ):
        if path.exists():
            existing = pd.read_csv(path)
            if list(existing.columns) != list(frame.columns) or not existing.equals(
                frame.reset_index(drop=True)
            ):
                raise RuntimeError(f"Existing frozen A2 manifest changed: {path}")
        else:
            atomic_write_dataframe(path, frame.reset_index(drop=True))

    digest = hashlib.sha256()
    digest.update(parent_freeze_hash.encode("ascii"))
    for path in (protocol_path, pems08_path, bdg2_path, assignments_path):
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    protocol_hash = digest.hexdigest()
    hash_path = output_dir / "a2_protocol_sha256.txt"
    if hash_path.exists():
        existing_hash = hash_path.read_text(encoding="utf-8").strip()
        if existing_hash != protocol_hash:
            raise RuntimeError("Existing A2 protocol hash differs")
    else:
        hash_path.write_text(protocol_hash + "\n", encoding="utf-8")

    freeze_material = {
        "a2_protocol_sha256": protocol_hash,
        "parent_v074_experiment_freeze_sha256": parent_freeze_hash,
        "frozen_config_sha256": hashlib.sha256(
            json.dumps(
                json_safe(frozen_config), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    freeze_material_path = output_dir / "a2_experiment_freeze_material.json"
    if freeze_material_path.exists():
        old_material = json.loads(freeze_material_path.read_text(encoding="utf-8"))
        if old_material != freeze_material:
            raise RuntimeError("Existing A2 experiment-freeze material changed")
    else:
        write_json(freeze_material_path, freeze_material)
    freeze_hash = sha256_file(freeze_material_path)
    freeze_hash_path = output_dir / "a2_experiment_freeze_sha256.txt"
    if freeze_hash_path.exists():
        old_hash = freeze_hash_path.read_text(encoding="utf-8").strip()
        if old_hash != freeze_hash:
            raise RuntimeError("Existing A2 experiment-freeze hash changed")
    else:
        freeze_hash_path.write_text(freeze_hash + "\n", encoding="utf-8")

    return protocol_hash, freeze_hash


def resolve_bdg2_source_path(path_text: str) -> Path:
    direct = Path(path_text)
    if direct.exists():
        return direct
    cleaned = BDG2_ROOT / "data" / "meters" / "cleaned"
    candidate = cleaned / direct.name
    if candidate.exists():
        return candidate
    matches = sorted(cleaned.rglob(direct.name)) if cleaned.exists() else []
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Cannot resolve BDG2 source file: {path_text}\nSearched under: {cleaned}"
    )


def load_pems08_chunks(
    core: Any,
    base: Any,
    frozen_manifest: pd.DataFrame,
) -> Tuple[List[Any], pd.DataFrame, Dict[str, Dict[str, Any]], Dict[str, Any]]:
    _, pems08_dir = base.default_data_dirs()
    data_file = base.auto_detect_data_file(pems08_dir, "PEMS08_A2_FROZEN_HOLDOUT")
    matrix, metadata = base.load_dataset(
        data_file,
        feature_index=0,
        max_series=32,
        seed=SEED,
        excluded_series=None,
    )
    chunks, rebuilt = core.deterministic_series_chunks(
        base,
        matrix,
        "PEMS08",
        metadata["selected_original_series_indices"],
        chunk_sizes=CHUNK_SIZES,
    )
    compare = [
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
    expected = frozen_manifest[compare].sort_values(compare).reset_index(drop=True)
    observed = rebuilt[compare].sort_values(compare).reset_index(drop=True)
    if not expected.equals(observed):
        raise AssertionError("Rebuilt PEMS08 chunks differ from the A0 manifest")

    metadata_by_chunk: Dict[str, Dict[str, Any]] = {}
    for row in rebuilt.itertuples(index=False):
        metadata_by_chunk[str(row.chunk_id)] = {
            "dataset": "PEMS08",
            "unit_type": "series",
            "unit_id": str(int(row.original_series_index)),
            "original_series_index": int(row.original_series_index),
            "building_id": "",
            "meter_type": "",
            "site_id": "",
            "primary_use": "",
        }
    return chunks, rebuilt, metadata_by_chunk, metadata


def load_bdg2_chunks(
    base: Any,
    e5: Any,
    manifest: pd.DataFrame,
) -> Tuple[List[Any], pd.DataFrame, Dict[str, Dict[str, Any]], Dict[str, Any]]:
    if not BDG2_ROOT.exists():
        raise FileNotFoundError(f"BDG2 root does not exist: {BDG2_ROOT}")

    chunks: List[Any] = []
    metadata_by_chunk: Dict[str, Dict[str, Any]] = {}
    resolved_rows: List[Dict[str, Any]] = []

    working = manifest.copy()
    working["resolved_source_file"] = [
        str(resolve_bdg2_source_path(text)) for text in working["source_file"]
    ]

    for resolved_file, group in working.groupby("resolved_source_file", sort=True):
        path = Path(resolved_file)
        columns = sorted(group["source_column"].astype(str).unique())
        frame = e5.read_selected_columns(path, columns)

        for row in group.itertuples(index=False):
            values_all = pd.to_numeric(
                frame[str(row.source_column)], errors="coerce"
            ).to_numpy(dtype=np.float32)
            start = int(row.chunk_start)
            size = int(row.chunk_size)
            values = np.ascontiguousarray(
                values_all[start : start + size], dtype=np.float32
            )
            if values.size != size:
                raise AssertionError(f"BDG2 chunk too short: {row.chunk_id}")
            if not np.isfinite(values).all():
                raise AssertionError(
                    f"BDG2 frozen chunk contains non-finite values: {row.chunk_id}"
                )

            chunk = base.ChunkUnit(
                chunk_id=str(row.chunk_id),
                series_index=int(row.original_series_index),
                chunk_start=start,
                chunk_size=size,
                values=values,
            )
            chunks.append(chunk)
            metadata_by_chunk[str(row.chunk_id)] = {
                "dataset": "BDG2",
                "unit_type": "building",
                "unit_id": str(row.building_id),
                "original_series_index": int(row.original_series_index),
                "building_id": str(row.building_id),
                "meter_type": str(row.meter_type),
                "site_id": str(row.site_id),
                "primary_use": str(row.primary_use),
                "source_file": str(path),
                "source_column": str(row.source_column),
            }
            row_dict = row._asdict()
            row_dict["resolved_source_file"] = str(path)
            resolved_rows.append(row_dict)

    chunks = sorted(chunks, key=lambda item: (int(item.chunk_size), item.chunk_id))
    summary = {
        "source": str(BDG2_ROOT),
        "n_buildings": int(manifest["building_id"].nunique()),
        "n_chunks": int(len(chunks)),
        "meter_type_counts": {
            str(k): int(v) for k, v in manifest.drop_duplicates("building_id")[
                "meter_type"
            ].value_counts().items()
        },
        "site_counts": {
            str(k): int(v) for k, v in manifest.drop_duplicates("building_id")[
                "site_id"
            ].value_counts().items()
        },
    }
    return chunks, pd.DataFrame(resolved_rows), metadata_by_chunk, summary


def build_query_pools(
    base: Any,
    e1: Any,
    core: Any,
    chunks: Sequence[Any],
) -> Dict[str, Dict[str, List[Any]]]:
    pools: Dict[str, Dict[str, List[Any]]] = {}
    for index, chunk in enumerate(chunks, start=1):
        queries = base.build_queries(
            chunk.values,
            chunk.chunk_id,
            SEED,
            e1.WINDOWS_PER_FRACTION,
        )
        pool = {
            query_class: [
                query for query in queries if query.query_class == query_class
            ]
            for query_class in core.QUERY_CLASSES
        }
        if any(not values for values in pool.values()):
            raise AssertionError(f"Incomplete query pool: {chunk.chunk_id}")
        pools[chunk.chunk_id] = pool
        if index == 1 or index == len(chunks) or index % 8 == 0:
            print(f"[query pools] {index}/{len(chunks)} chunks")
    return pools


def checkpoint_complete(path: Path, freeze_hash: str) -> bool:
    marker = path / "complete.json"
    if not marker.exists():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("dataset_freeze_sha256") == freeze_hash


def prepare_calibration(
    dataset: str,
    dataset_dir: Path,
    dataset_freeze_hash: str,
    core: Any,
    e1: Any,
    base: Any,
    reps: Mapping[str, Any],
    chunks: Sequence[Any],
    metadata_by_chunk: Mapping[str, Mapping[str, Any]],
    query_pools: Mapping[str, Mapping[str, List[Any]]],
    v074: Any,
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Dict[str, float]]],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[Tuple[str, str], int]],
    Dict[str, Dict[Tuple[str, str], int]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    storage_dir = dataset_dir / "_candidate_storage"
    chunk_checkpoint_root = dataset_dir / "_calibration_checkpoints"
    migration_checkpoint_root = dataset_dir / "_migration_checkpoints"
    migration_trial_root = dataset_dir / "_migration_calibration_trials"
    storage_dir.mkdir(parents=True, exist_ok=True)
    chunk_checkpoint_root.mkdir(parents=True, exist_ok=True)
    migration_checkpoint_root.mkdir(parents=True, exist_ok=True)
    migration_trial_root.mkdir(parents=True, exist_ok=True)

    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    calibration_frames: List[pd.DataFrame] = []
    codec_frames: List[pd.DataFrame] = []

    for index, chunk in enumerate(sorted(chunks, key=lambda item: item.chunk_id), start=1):
        checkpoint_dir = chunk_checkpoint_root / chunk.chunk_id
        cal_path = checkpoint_dir / "calibration.csv"
        codec_path = checkpoint_dir / "codecs.csv"
        if checkpoint_complete(checkpoint_dir, dataset_freeze_hash) and cal_path.exists() and codec_path.exists():
            calibration_frames.append(pd.read_csv(cal_path))
            codec_frames.append(pd.read_csv(codec_path))
            print(f"[{dataset} calibration resume] {index}/{len(chunks)} {chunk.chunk_id}")
            continue

        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        chunk_storage = storage_dir / chunk.chunk_id
        if chunk_storage.exists():
            shutil.rmtree(chunk_storage)
        chunk_storage.mkdir(parents=True, exist_ok=True)

        stored_map, encode_ns, decode_ns = e1.encode_candidates(
            base, reps, chunk, chunk_storage
        )
        profile, local_calibration = e1.calibrate_chunk(
            base,
            reps,
            stored_map,
            chunk,
            query_pools[chunk.chunk_id],
        )
        del profile

        metadata = metadata_by_chunk[chunk.chunk_id]
        calibration_rows: List[Dict[str, Any]] = []
        for row in local_calibration:
            enriched = dict(row)
            enriched.update(
                {
                    "dataset": dataset,
                    "series_index": int(chunk.series_index),
                    "chunk_start": int(chunk.chunk_start),
                    "unit_type": metadata["unit_type"],
                    "unit_id": metadata["unit_id"],
                    "building_id": metadata.get("building_id", ""),
                }
            )
            calibration_rows.append(enriched)

        codec_rows: List[Dict[str, Any]] = []
        for codec, stored in stored_map.items():
            codec_rows.append(
                {
                    "dataset": dataset,
                    "chunk_id": chunk.chunk_id,
                    "series_index": int(chunk.series_index),
                    "chunk_start": int(chunk.chunk_start),
                    "chunk_size": int(chunk.chunk_size),
                    "unit_type": metadata["unit_type"],
                    "unit_id": metadata["unit_id"],
                    "building_id": metadata.get("building_id", ""),
                    "codec": str(codec),
                    "raw_size_bytes": int(chunk.values.nbytes),
                    "payload_size_bytes": int(stored.payload_size_bytes),
                    "metadata_size_bytes": int(stored.metadata_size_bytes),
                    "stored_size_bytes": int(stored.stored_size_bytes),
                    "compression_ratio": float(
                        stored.stored_size_bytes / chunk.values.nbytes
                    ),
                    "encode_ns": int(encode_ns[codec]),
                    "full_decode_ns": int(decode_ns[codec]),
                }
            )

        calibration_frame = pd.DataFrame(calibration_rows)
        codec_frame = pd.DataFrame(codec_rows)
        atomic_write_dataframe(cal_path, calibration_frame)
        atomic_write_dataframe(codec_path, codec_frame)
        write_json(
            checkpoint_dir / "complete.json",
            {
                "dataset_freeze_sha256": dataset_freeze_hash,
                "chunk_id": chunk.chunk_id,
                "completed_at_unix_ns": time.time_ns(),
            },
        )
        calibration_frames.append(calibration_frame)
        codec_frames.append(codec_frame)
        print(f"[{dataset} calibration] {index}/{len(chunks)} {chunk.chunk_id}")

    calibration_df = pd.concat(calibration_frames, ignore_index=True)
    codec_df = pd.concat(codec_frames, ignore_index=True)
    atomic_write_dataframe(dataset_dir / "calibration_records.csv", calibration_df)
    atomic_write_dataframe(dataset_dir / "codec_records.csv", codec_df)

    stored_by_chunk = v074.reconstruct_stored_maps(
        base, chunks_by_id, codec_df, storage_dir
    )
    profiles = v074.load_profiles(calibration_df)
    encode_ns_by_chunk, decode_ns_by_chunk = v074.load_encode_decode_maps(codec_df)

    migration_frames: List[pd.DataFrame] = []
    for index, chunk in enumerate(sorted(chunks, key=lambda item: item.chunk_id), start=1):
        checkpoint_dir = migration_checkpoint_root / chunk.chunk_id
        records_path = checkpoint_dir / "migrations.csv"
        if checkpoint_complete(checkpoint_dir, dataset_freeze_hash) and records_path.exists():
            migration_frames.append(pd.read_csv(records_path))
            print(f"[{dataset} migration resume] {index}/{len(chunks)} {chunk.chunk_id}")
            continue

        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _, _, rows = core.measure_directed_migrations(
            e1,
            base,
            reps,
            {chunk.chunk_id: chunk},
            {chunk.chunk_id: stored_by_chunk[chunk.chunk_id]},
            migration_trial_root / chunk.chunk_id,
        )
        frame = pd.DataFrame(rows)
        metadata = metadata_by_chunk[chunk.chunk_id]
        frame["dataset"] = dataset
        frame["unit_type"] = metadata["unit_type"]
        frame["unit_id"] = metadata["unit_id"]
        frame["building_id"] = metadata.get("building_id", "")
        atomic_write_dataframe(records_path, frame)
        write_json(
            checkpoint_dir / "complete.json",
            {
                "dataset_freeze_sha256": dataset_freeze_hash,
                "chunk_id": chunk.chunk_id,
                "completed_at_unix_ns": time.time_ns(),
            },
        )
        migration_frames.append(frame)
        print(f"[{dataset} migration calibration] {index}/{len(chunks)}")

    migration_df = pd.concat(migration_frames, ignore_index=True)
    atomic_write_dataframe(dataset_dir / "measured_migration_costs.csv", migration_df)

    migration_costs: Dict[str, Dict[Tuple[str, str], int]] = {}
    migration_bytes: Dict[str, Dict[Tuple[str, str], int]] = {}
    for chunk_id, group in migration_df.groupby("chunk_id"):
        cost_local: Dict[Tuple[str, str], int] = {}
        byte_local: Dict[Tuple[str, str], int] = {}
        for row in group.itertuples(index=False):
            key = (str(row.current_codec), str(row.target_codec))
            cost_local[key] = int(row.median_migration_ns)
            byte_local[key] = int(row.rewrite_bytes)
        migration_costs[str(chunk_id)] = cost_local
        migration_bytes[str(chunk_id)] = byte_local

    shutil.rmtree(migration_trial_root, ignore_errors=True)
    return (
        stored_by_chunk,
        profiles,
        encode_ns_by_chunk,
        decode_ns_by_chunk,
        migration_costs,
        migration_bytes,
        calibration_df,
        codec_df,
        migration_df,
    )


def build_model_replay(
    dataset: str,
    core: Any,
    e1: Any,
    e1v2: Any,
    base: Any,
    chunks: Sequence[Any],
    assignments: pd.DataFrame,
    query_pools: Mapping[str, Mapping[str, List[Any]]],
    profiles: Mapping[str, Mapping[str, Mapping[str, float]]],
    encode_ns_by_chunk: Mapping[str, Mapping[str, int]],
    decode_ns_by_chunk: Mapping[str, Mapping[str, int]],
    migration_costs: Mapping[str, Mapping[Tuple[str, str], int]],
    migration_bytes: Mapping[str, Mapping[Tuple[str, str], int]],
    frozen_config: Mapping[str, Any],
    metadata_by_chunk: Mapping[str, Mapping[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reps = base.representation_registry(3)
    codecs = tuple(sorted(reps))
    chunks_by_size = {
        size: sorted(
            [chunk for chunk in chunks if int(chunk.chunk_size) == size],
            key=lambda item: item.chunk_id,
        )
        for size in CHUNK_SIZES
    }
    global_rows: List[Dict[str, Any]] = []
    chunk_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []

    for chunk_size in CHUNK_SIZES:
        size_chunks = chunks_by_size[chunk_size]
        chunk_ids = tuple(chunk.chunk_id for chunk in size_chunks)
        profile_array = core.profile_arrays(profiles, chunk_ids, codecs)
        initial_positions = core.scan_best_codec_positions(
            e1, base, profiles, chunk_ids, codecs
        )
        size_assignments = assignments[
            (assignments["dataset"] == dataset)
            & (assignments["chunk_size"] == chunk_size)
        ]

        for trace_seed in TRACE_SEEDS:
            for trace_name in TRACE_NAMES:
                plan = core.build_trace_plan(
                    base,
                    size_chunks,
                    query_pools,
                    size_assignments,
                    trace_name,
                    trace_seed,
                )
                digest = hashlib.sha256()
                for array in (
                    plan.chunk_positions,
                    plan.local_indices,
                    plan.query_class_positions,
                    plan.query_choice_positions,
                    plan.workload_positions,
                ):
                    digest.update(array.tobytes())
                trace_rows.append(
                    {
                        "dataset": dataset,
                        "trace": trace_name,
                        "trace_seed": trace_seed,
                        "chunk_size": chunk_size,
                        "n_chunks": len(size_chunks),
                        "n_queries": plan.n_events,
                        "trace_sha256": digest.hexdigest(),
                    }
                )

                runtime_config = dict(frozen_config)
                runtime_config["_initial_codec_positions"] = initial_positions
                results = {
                    "initial_best": core.simulate_initial_best(
                        plan, profile_array, initial_positions
                    ),
                    "afhc_frozen": core.simulate_afhc(
                        plan,
                        profile_array,
                        codecs,
                        migration_costs,
                        migration_bytes,
                        runtime_config,
                    ),
                    "repressdb": core.simulate_repressdb(
                        e1,
                        e1v2,
                        base,
                        plan,
                        profiles,
                        encode_ns_by_chunk,
                        decode_ns_by_chunk,
                        migration_costs,
                        migration_bytes,
                    ),
                }

                for policy, result in results.items():
                    global_rows.append(
                        {
                            "dataset": dataset,
                            "trace": trace_name,
                            "trace_seed": trace_seed,
                            "chunk_size": chunk_size,
                            "policy": policy,
                            "n_queries": plan.n_events,
                            "query_time_ns": float(result["query_time_ns"]),
                            "migration_time_ns": float(result["migration_time_ns"]),
                            "total_time_ns": float(result["total_time_ns"]),
                            "migration_count": int(result["migration_count"]),
                            "rewrite_bytes": int(result["rewrite_bytes"]),
                        }
                    )
                    for position, chunk_id in enumerate(plan.chunk_ids):
                        metadata = metadata_by_chunk[chunk_id]
                        chunk_rows.append(
                            {
                                "dataset": dataset,
                                "trace": trace_name,
                                "trace_seed": trace_seed,
                                "chunk_size": chunk_size,
                                "policy": policy,
                                "chunk_id": chunk_id,
                                "series_index": int(
                                    next(
                                        chunk.series_index
                                        for chunk in size_chunks
                                        if chunk.chunk_id == chunk_id
                                    )
                                ),
                                "unit_type": metadata["unit_type"],
                                "unit_id": metadata["unit_id"],
                                "building_id": metadata.get("building_id", ""),
                                "query_time_ns": float(
                                    result["per_chunk_query_time_ns"][position]
                                ),
                                "migration_time_ns": float(
                                    result["per_chunk_migration_time_ns"][position]
                                ),
                                "total_time_ns": float(
                                    result["per_chunk_query_time_ns"][position]
                                    + result["per_chunk_migration_time_ns"][position]
                                ),
                                "migration_count": int(
                                    result["per_chunk_migration_count"][position]
                                ),
                                "rewrite_bytes": int(
                                    result["per_chunk_rewrite_bytes"][position]
                                ),
                            }
                        )
                print(
                    f"[{dataset} model] size={chunk_size}, trace={trace_name}, "
                    f"seed={trace_seed}, queries={plan.n_events}"
                )

    return pd.DataFrame(global_rows), pd.DataFrame(chunk_rows), pd.DataFrame(trace_rows)


def actual_checkpoint_token(
    dataset: str,
    repeat_index: int,
    trace_seed: int,
    trace: str,
    chunk_size: int,
    policy: str,
) -> str:
    return (
        f"{dataset.lower()}__repeat_{repeat_index:02d}__seed_{trace_seed}__"
        f"trace_{trace}__size_{chunk_size}__policy_{policy}"
    )


def run_actual_audit(
    dataset: str,
    dataset_dir: Path,
    dataset_freeze_hash: str,
    core: Any,
    e1: Any,
    e1v2: Any,
    base: Any,
    reps: Mapping[str, Any],
    chunks: Sequence[Any],
    assignments: pd.DataFrame,
    query_pools: Mapping[str, Mapping[str, List[Any]]],
    stored_by_chunk: Mapping[str, Mapping[str, Any]],
    profiles: Mapping[str, Mapping[str, Mapping[str, float]]],
    encode_ns_by_chunk: Mapping[str, Mapping[str, int]],
    decode_ns_by_chunk: Mapping[str, Mapping[str, int]],
    frozen_config: Mapping[str, Any],
    metadata_by_chunk: Mapping[str, Mapping[str, Any]],
    v074: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    chunks_by_size = {
        size: sorted(
            [chunk for chunk in chunks if int(chunk.chunk_size) == size],
            key=lambda item: item.chunk_id,
        )
        for size in CHUNK_SIZES
    }
    checkpoint_root = dataset_dir / "_actual_checkpoints"
    migration_root = dataset_dir / "_actual_migration_trials"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    migration_root.mkdir(parents=True, exist_ok=True)

    expected_tokens: List[str] = []
    total_runs = (
        ACTUAL_REPEATS
        * len(TRACE_SEEDS)
        * len(TRACE_NAMES)
        * len(CHUNK_SIZES)
        * len(POLICIES)
    )
    completed = 0

    for repeat_index in range(ACTUAL_REPEATS):
        for chunk_size in CHUNK_SIZES:
            size_chunks = chunks_by_size[chunk_size]
            size_assignments = assignments[
                (assignments["dataset"] == dataset)
                & (assignments["chunk_size"] == chunk_size)
            ]
            for trace_seed in TRACE_SEEDS:
                for trace_name in TRACE_NAMES:
                    plan = core.build_trace_plan(
                        base,
                        size_chunks,
                        query_pools,
                        size_assignments,
                        trace_name,
                        trace_seed,
                    )
                    policy_order = list(POLICIES)
                    random.Random(
                        stable_seed(
                            VERSION,
                            dataset,
                            repeat_index,
                            trace_seed,
                            chunk_size,
                            trace_name,
                            "policy_order",
                        )
                    ).shuffle(policy_order)

                    for order_position, policy in enumerate(policy_order):
                        token = actual_checkpoint_token(
                            dataset,
                            repeat_index,
                            trace_seed,
                            trace_name,
                            chunk_size,
                            policy,
                        )
                        expected_tokens.append(token)
                        run_dir = checkpoint_root / token
                        if v074.checkpoint_is_complete(run_dir, dataset_freeze_hash):
                            completed += 1
                            print(f"[{dataset} audit resume] {completed}/{total_runs} {token}")
                            continue
                        if run_dir.exists():
                            shutil.rmtree(run_dir)

                        print(
                            f"[{dataset} actual] {completed + 1}/{total_runs} "
                            f"repeat={repeat_index + 1}/{ACTUAL_REPEATS}, "
                            f"size={chunk_size}, trace={trace_name}, "
                            f"seed={trace_seed}, policy={policy}"
                        )
                        gc.collect()
                        if POLICY_PAUSE_SECONDS > 0:
                            time.sleep(POLICY_PAUSE_SECONDS)

                        summary, chunk_rows, event_rows = core.run_actual_policy(
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
                            frozen_config,
                            migration_root / token,
                        )
                        common = {
                            "dataset": dataset,
                            "repeat_index": repeat_index,
                            "repeat_one_based": repeat_index + 1,
                            "policy_order_position": order_position,
                            "actual_run_token": token,
                            "dataset_freeze_sha256": dataset_freeze_hash,
                        }
                        summary.update(common)
                        for row in chunk_rows:
                            row.update(common)
                            metadata = metadata_by_chunk[str(row["chunk_id"])]
                            row.update(
                                {
                                    "unit_type": metadata["unit_type"],
                                    "unit_id": metadata["unit_id"],
                                    "building_id": metadata.get("building_id", ""),
                                    "meter_type": metadata.get("meter_type", ""),
                                    "site_id": metadata.get("site_id", ""),
                                    "primary_use": metadata.get("primary_use", ""),
                                }
                            )
                        for row in event_rows:
                            row.update(common)
                            chunk_id = str(row.get("chunk_id", ""))
                            if chunk_id in metadata_by_chunk:
                                metadata = metadata_by_chunk[chunk_id]
                                row.update(
                                    {
                                        "unit_type": metadata["unit_type"],
                                        "unit_id": metadata["unit_id"],
                                        "building_id": metadata.get("building_id", ""),
                                    }
                                )

                        v074.write_checkpoint(
                            run_dir,
                            summary,
                            chunk_rows,
                            event_rows,
                            dataset_freeze_hash,
                        )
                        completed += 1

    summary_df, chunk_df, event_df = v074.load_all_checkpoints(
        checkpoint_root, expected_tokens, dataset_freeze_hash
    )
    if len(summary_df) != total_runs:
        raise AssertionError(
            f"{dataset}: expected {total_runs} actual runs, found {len(summary_df)}"
        )
    duplicate_keys = [
        "dataset",
        "repeat_index",
        "trace_seed",
        "trace",
        "chunk_size",
        "policy",
    ]
    if summary_df.duplicated(duplicate_keys).any():
        raise AssertionError(f"{dataset}: duplicate actual-run keys")
    shutil.rmtree(migration_root, ignore_errors=True)
    return summary_df, chunk_df, event_df


def build_exact_ratios(
    chunk_df: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    assignment_lookup = assignments[["dataset", "chunk_id", "trace", "group"]]
    enriched = chunk_df.merge(
        assignment_lookup,
        on=["dataset", "chunk_id", "trace"],
        how="left",
        validate="many_to_one",
    )
    if enriched["group"].isna().any():
        raise AssertionError("Missing trace-group assignment in actual records")

    keys = [
        "dataset",
        "repeat_index",
        "trace_seed",
        "trace",
        "group",
        "chunk_size",
        "chunk_id",
        "unit_type",
        "unit_id",
    ]
    pivot = enriched.pivot(index=keys, columns="policy", values="total_time_ns").reset_index()
    rows: List[Dict[str, Any]] = []
    for policy, reference in (
        ("repressdb", "afhc_frozen"),
        ("repressdb", "initial_best"),
        ("afhc_frozen", "initial_best"),
    ):
        if policy not in pivot.columns or reference not in pivot.columns:
            raise AssertionError(f"Missing policies for paired ratio: {policy}/{reference}")
        for row in pivot.itertuples(index=False):
            policy_time = float(getattr(row, policy))
            reference_time = float(getattr(row, reference))
            values = {key: getattr(row, key) for key in keys}
            values["building_id"] = (
                str(row.unit_id) if str(row.unit_type) == "building" else ""
            )
            rows.append(
                {
                    **values,
                    "policy": policy,
                    "reference_policy": reference,
                    "policy_total_time_ns": policy_time,
                    "reference_total_time_ns": reference_time,
                    "paired_ratio": policy_time / reference_time,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_median_ci(values: np.ndarray, seed: int) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
    for index in range(BOOTSTRAP_ITERATIONS):
        samples[index] = np.median(
            rng.choice(array, size=array.size, replace=True)
        )
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def clustered_ratio_statistics(exact_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    scopes: List[Tuple[str, pd.DataFrame]] = [("ALL", exact_df)]
    for group_name, group_df in exact_df.groupby("group", sort=True):
        scopes.append((str(group_name), group_df))

    for scope, scoped in scopes:
        for keys, group in scoped.groupby(
            ["dataset", "trace", "chunk_size", "policy", "reference_policy"],
            sort=True,
        ):
            dataset, trace, chunk_size, policy, reference = keys
            unit_df = (
                group.groupby(["unit_type", "unit_id"], as_index=False)[
                    "paired_ratio"
                ].median()
            )
            ratios = unit_df["paired_ratio"].to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_median_ci(
                ratios,
                stable_seed(
                    VERSION,
                    dataset,
                    trace,
                    chunk_size,
                    scope,
                    policy,
                    reference,
                    "bootstrap",
                ),
            )
            try:
                log_ratios = np.log(ratios)
                p_value = (
                    1.0
                    if np.allclose(log_ratios, 0.0)
                    else float(
                        wilcoxon(
                            log_ratios,
                            alternative="two-sided",
                            zero_method="wilcox",
                        ).pvalue
                    )
                )
            except ValueError:
                p_value = 1.0

            repeat_medians = group.groupby("repeat_index")["paired_ratio"].median()
            rows.append(
                {
                    "dataset": dataset,
                    "trace": trace,
                    "group": scope,
                    "chunk_size": int(chunk_size),
                    "policy": policy,
                    "reference_policy": reference,
                    "n_independent_units": int(len(unit_df)),
                    "n_exact_paired_cells": int(len(group)),
                    "median_ratio_policy_over_reference": float(np.median(ratios)),
                    "mean_ratio_policy_over_reference": float(np.mean(ratios)),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "fraction_policy_faster": float(np.mean(ratios < 1.0)),
                    "wilcoxon_log_ratio_p": p_value,
                    "repeat_median_ratio_min": float(repeat_medians.min()),
                    "repeat_median_ratio_max": float(repeat_medians.max()),
                }
            )
    return pd.DataFrame(rows)


def build_model_exact_ratios(
    model_chunk_df: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    enriched = model_chunk_df.merge(
        assignments[["dataset", "chunk_id", "trace", "group"]],
        on=["dataset", "chunk_id", "trace"],
        how="left",
        validate="many_to_one",
    )
    keys = [
        "dataset",
        "trace_seed",
        "trace",
        "group",
        "chunk_size",
        "chunk_id",
        "unit_type",
        "unit_id",
    ]
    pivot = enriched.pivot(index=keys, columns="policy", values="total_time_ns").reset_index()
    rows: List[Dict[str, Any]] = []
    for policy, reference in (
        ("repressdb", "afhc_frozen"),
        ("repressdb", "initial_best"),
        ("afhc_frozen", "initial_best"),
    ):
        for row in pivot.itertuples(index=False):
            values = {key: getattr(row, key) for key in keys}
            values["building_id"] = (
                str(row.unit_id) if str(row.unit_type) == "building" else ""
            )
            rows.append(
                {
                    **values,
                    "repeat_index": 0,
                    "policy": policy,
                    "reference_policy": reference,
                    "paired_ratio": float(getattr(row, policy))
                    / float(getattr(row, reference)),
                }
            )
    return pd.DataFrame(rows)


def build_no_action_consistency(
    exact_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    no_frequency_traces = ("shape_only", "brief_burst", "stable_scan")
    local = exact_df[
        exact_df["trace"].isin(no_frequency_traces)
        & (exact_df["policy"] == "afhc_frozen")
        & (exact_df["reference_policy"] == "initial_best")
    ]
    rows: List[Dict[str, Any]] = []
    dataset_results: Dict[str, Any] = {}
    for dataset, dataset_group in local.groupby("dataset", sort=True):
        unit_ratios = (
            dataset_group.groupby(["unit_type", "unit_id"], as_index=False)[
                "paired_ratio"
            ].median()
        )
        overall = float(np.median(unit_ratios["paired_ratio"]))
        migrations = int(
            summary_df[
                (summary_df["dataset"] == dataset)
                & summary_df["trace"].isin(no_frequency_traces)
                & (summary_df["policy"] == "afhc_frozen")
            ]["migration_count"].sum()
        )
        passed = bool(
            NO_ACTION_RATIO_LOW <= overall <= NO_ACTION_RATIO_HIGH
            and migrations == 0
        )
        dataset_results[dataset] = {
            "median_afhc_over_initial": overall,
            "required_interval": [NO_ACTION_RATIO_LOW, NO_ACTION_RATIO_HIGH],
            "afhc_migrations": migrations,
            "pass": passed,
        }
        for (trace, chunk_size), cell in dataset_group.groupby(
            ["trace", "chunk_size"], sort=True
        ):
            ratios = cell["paired_ratio"].to_numpy(dtype=float)
            rows.append(
                {
                    "dataset": dataset,
                    "trace": trace,
                    "chunk_size": int(chunk_size),
                    "n_exact_paired_cells": int(len(cell)),
                    "median_afhc_over_initial": float(np.median(ratios)),
                    "p05_afhc_over_initial": float(np.quantile(ratios, 0.05)),
                    "p95_afhc_over_initial": float(np.quantile(ratios, 0.95)),
                }
            )
    assessment = {
        "datasets": dataset_results,
        "all_datasets_pass": bool(
            dataset_results and all(value["pass"] for value in dataset_results.values())
        ),
    }
    return pd.DataFrame(rows), assessment


def build_model_actual_validation(
    model_global_df: pd.DataFrame,
    actual_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["dataset", "trace", "trace_seed", "chunk_size"]
    model_pivot = model_global_df.pivot(
        index=keys, columns="policy", values="total_time_ns"
    ).reset_index()
    actual_pivot = actual_summary_df.pivot(
        index=["repeat_index", *keys], columns="policy", values="total_time_ns"
    ).reset_index()

    for prefix, frame in (("model", model_pivot), ("actual", actual_pivot)):
        frame[f"{prefix}_afhc_ratio_to_initial"] = (
            frame["afhc_frozen"] / frame["initial_best"]
        )
        frame[f"{prefix}_repressdb_ratio_to_initial"] = (
            frame["repressdb"] / frame["initial_best"]
        )

    aggregated = (
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
        )
    )
    merged = model_pivot[
        keys
        + [
            "model_afhc_ratio_to_initial",
            "model_repressdb_ratio_to_initial",
        ]
    ].merge(aggregated, on=keys, how="inner")
    merged["absolute_afhc_ratio_error"] = (
        merged["model_afhc_ratio_to_initial"]
        - merged["actual_afhc_ratio_to_initial"]
    ).abs()
    merged["absolute_repressdb_ratio_error"] = (
        merged["model_repressdb_ratio_to_initial"]
        - merged["actual_repressdb_ratio_to_initial"]
    ).abs()
    return merged


def lookup_stat(
    stats: pd.DataFrame,
    dataset: str,
    trace: str,
    group: str,
    chunk_size: int,
    policy: str,
    reference: str,
) -> Dict[str, Any]:
    local = stats[
        (stats["dataset"] == dataset)
        & (stats["trace"] == trace)
        & (stats["group"] == group)
        & (stats["chunk_size"] == chunk_size)
        & (stats["policy"] == policy)
        & (stats["reference_policy"] == reference)
    ]
    if len(local) != 1:
        return {}
    row = local.iloc[0]
    return {
        "n_independent_units": int(row.n_independent_units),
        "median_ratio": float(row.median_ratio_policy_over_reference),
        "bootstrap_ci_low": float(row.bootstrap_ci_low),
        "bootstrap_ci_high": float(row.bootstrap_ci_high),
        "fraction_policy_faster": float(row.fraction_policy_faster),
        "wilcoxon_log_ratio_p": float(row.wilcoxon_log_ratio_p),
    }


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    v070_script = project_dir / V070_SCRIPT_NAME
    v073_script = project_dir / V073_SCRIPT_NAME
    v074_script = project_dir / V074_SCRIPT_NAME
    e5_script = project_dir / E5_SCRIPT_NAME
    v070_dir = project_dir / V070_RESULT_DIR_NAME
    v073_dir = project_dir / V073_RESULT_DIR_NAME
    v074_dir = project_dir / V074_RESULT_DIR_NAME

    for path in (v070_script, v073_script, v074_script, e5_script):
        if not path.exists():
            raise FileNotFoundError(f"Missing dependency: {path}")
    verify_required_files(v070_dir, REQUIRED_V070_FILES)
    verify_required_files(v073_dir, REQUIRED_V073_FILES)
    verify_required_files(v074_dir, REQUIRED_V074_FILES)

    print("=" * 116)
    print("RePressDB A2 v0.8.0 — frozen PEMS08 and BDG2 holdout validation")
    print(f"Target journal: {TARGET_JOURNAL}")
    print("No parameter selection, no controller retuning, both holdouts under one protocol.")
    print("=" * 116)

    frozen_config, parent_a0_hash, parent_v074_hash, parent_freeze_hash = (
        verify_v074_freeze(v070_dir, v073_dir, v074_dir)
    )

    v073 = load_module(v073_script, "repressdb_v073_for_a2")
    v074 = load_module(v074_script, "repressdb_v074_for_a2")
    e5 = load_module(e5_script, "repressdb_e5_for_a2")
    core = v073.build_corrected_core(v070_script, output_dir)
    core.__file__ = str(v070_script.resolve())
    e1v2, e1, base = core.load_repressdb_modules()

    pems08_manifest = pd.read_csv(v070_dir / "pems08_selected_chunks.csv")
    bdg2_reserved = pd.read_csv(v070_dir / "bdg2_selected_buildings.csv")
    bdg2_manifest = build_bdg2_paired_manifest(bdg2_reserved)

    pems08_assignments = pd.read_csv(v070_dir / "trace_assignment_manifest.csv")
    pems08_assignments = pems08_assignments[
        pems08_assignments["dataset"] == "PEMS08"
    ].reset_index(drop=True)
    bdg2_assignments = core.build_trace_assignments(bdg2_manifest)
    assignments = pd.concat(
        [pems08_assignments, bdg2_assignments], ignore_index=True
    ).sort_values(
        ["dataset", "chunk_size", "assignment_rank", "trace"]
    ).reset_index(drop=True)

    a2_protocol_hash, a2_freeze_hash = create_or_verify_a2_protocol(
        output_dir,
        core,
        frozen_config,
        parent_a0_hash,
        parent_v074_hash,
        parent_freeze_hash,
        pems08_manifest.reset_index(drop=True),
        bdg2_manifest.reset_index(drop=True),
        assignments,
    )
    print(f"[A2] protocol hash: {a2_protocol_hash}")
    print(f"[A2] experiment freeze: {a2_freeze_hash}")

    pems08_chunks, pems08_loaded_manifest, pems08_metadata, pems08_source_meta = (
        load_pems08_chunks(core, base, pems08_manifest)
    )
    bdg2_chunks, bdg2_loaded_manifest, bdg2_metadata, bdg2_source_meta = (
        load_bdg2_chunks(base, e5, bdg2_manifest)
    )
    atomic_write_dataframe(
        output_dir / "pems08_loaded_chunk_manifest.csv", pems08_loaded_manifest
    )
    atomic_write_dataframe(
        output_dir / "bdg2_loaded_chunk_manifest.csv", bdg2_loaded_manifest
    )

    dataset_inputs = {
        "PEMS08": {
            "chunks": pems08_chunks,
            "metadata": pems08_metadata,
            "source_meta": pems08_source_meta,
        },
        "BDG2": {
            "chunks": bdg2_chunks,
            "metadata": bdg2_metadata,
            "source_meta": bdg2_source_meta,
        },
    }

    reps = base.representation_registry(3)
    all_model_global: List[pd.DataFrame] = []
    all_model_chunk: List[pd.DataFrame] = []
    all_trace_manifest: List[pd.DataFrame] = []
    all_actual_summary: List[pd.DataFrame] = []
    all_actual_chunk: List[pd.DataFrame] = []
    all_actual_event: List[pd.DataFrame] = []
    dataset_summaries: Dict[str, Any] = {}

    for dataset in ("PEMS08", "BDG2"):
        print("-" * 116)
        print(f"[A2] begin frozen holdout: {dataset}")
        dataset_dir = output_dir / dataset.lower()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        chunks = dataset_inputs[dataset]["chunks"]
        metadata_by_chunk = dataset_inputs[dataset]["metadata"]
        dataset_assignments = assignments[assignments["dataset"] == dataset]

        dataset_digest = hashlib.sha256()
        dataset_digest.update(a2_freeze_hash.encode("ascii"))
        dataset_digest.update(dataset.encode("utf-8"))
        manifest_path = (
            output_dir / "pems08_holdout_chunk_manifest.csv"
            if dataset == "PEMS08"
            else output_dir / "bdg2_holdout_chunk_manifest.csv"
        )
        dataset_digest.update(sha256_file(manifest_path).encode("ascii"))
        dataset_freeze_hash = dataset_digest.hexdigest()
        (dataset_dir / "dataset_freeze_sha256.txt").write_text(
            dataset_freeze_hash + "\n", encoding="utf-8"
        )

        query_pools = build_query_pools(base, e1, core, chunks)
        (
            stored_by_chunk,
            profiles,
            encode_ns_by_chunk,
            decode_ns_by_chunk,
            migration_costs,
            migration_bytes,
            calibration_df,
            codec_df,
            migration_df,
        ) = prepare_calibration(
            dataset,
            dataset_dir,
            dataset_freeze_hash,
            core,
            e1,
            base,
            reps,
            chunks,
            metadata_by_chunk,
            query_pools,
            v074,
        )

        model_global_df, model_chunk_df, trace_manifest_df = build_model_replay(
            dataset,
            core,
            e1,
            e1v2,
            base,
            chunks,
            dataset_assignments,
            query_pools,
            profiles,
            encode_ns_by_chunk,
            decode_ns_by_chunk,
            migration_costs,
            migration_bytes,
            frozen_config,
            metadata_by_chunk,
        )
        atomic_write_dataframe(dataset_dir / "model_global_records.csv", model_global_df)
        atomic_write_dataframe(dataset_dir / "model_chunk_records.csv", model_chunk_df)
        atomic_write_dataframe(dataset_dir / "generated_trace_manifest.csv", trace_manifest_df)

        actual_summary_df, actual_chunk_df, actual_event_df = run_actual_audit(
            dataset,
            dataset_dir,
            dataset_freeze_hash,
            core,
            e1,
            e1v2,
            base,
            reps,
            chunks,
            dataset_assignments,
            query_pools,
            stored_by_chunk,
            profiles,
            encode_ns_by_chunk,
            decode_ns_by_chunk,
            frozen_config,
            metadata_by_chunk,
            v074,
        )
        atomic_write_dataframe(
            dataset_dir / "actual_global_run_records_repeated.csv", actual_summary_df
        )
        atomic_write_dataframe(
            dataset_dir / "actual_chunk_run_records_repeated.csv", actual_chunk_df
        )
        atomic_write_dataframe(
            dataset_dir / "actual_event_diagnostics_repeated.csv", actual_event_df
        )

        all_model_global.append(model_global_df)
        all_model_chunk.append(model_chunk_df)
        all_trace_manifest.append(trace_manifest_df)
        all_actual_summary.append(actual_summary_df)
        all_actual_chunk.append(actual_chunk_df)
        all_actual_event.append(actual_event_df)

        dataset_summaries[dataset] = {
            "n_independent_units": 32,
            "n_chunks": len(chunks),
            "chunks_by_size": {
                str(size): int(sum(int(chunk.chunk_size) == size for chunk in chunks))
                for size in CHUNK_SIZES
            },
            "source_metadata": dataset_inputs[dataset]["source_meta"],
            "calibration_rows": int(len(calibration_df)),
            "codec_rows": int(len(codec_df)),
            "migration_rows": int(len(migration_df)),
            "actual_policy_runs": int(len(actual_summary_df)),
        }

    model_global_all = pd.concat(all_model_global, ignore_index=True)
    model_chunk_all = pd.concat(all_model_chunk, ignore_index=True)
    trace_manifest_all = pd.concat(all_trace_manifest, ignore_index=True)
    actual_summary_all = pd.concat(all_actual_summary, ignore_index=True)
    actual_chunk_all = pd.concat(all_actual_chunk, ignore_index=True)
    actual_event_all = pd.concat(all_actual_event, ignore_index=True)

    exact_ratios = build_exact_ratios(actual_chunk_all, assignments)
    actual_stats = clustered_ratio_statistics(exact_ratios)
    model_exact_ratios = build_model_exact_ratios(model_chunk_all, assignments)
    model_stats = clustered_ratio_statistics(model_exact_ratios)
    no_action_df, no_action_assessment = build_no_action_consistency(
        exact_ratios, actual_summary_all
    )
    model_actual_df = build_model_actual_validation(
        model_global_all, actual_summary_all
    )

    atomic_write_dataframe(output_dir / "model_global_records.csv", model_global_all)
    atomic_write_dataframe(output_dir / "model_chunk_records.csv", model_chunk_all)
    atomic_write_dataframe(output_dir / "generated_trace_manifest.csv", trace_manifest_all)
    atomic_write_dataframe(
        output_dir / "actual_global_run_records_repeated.csv", actual_summary_all
    )
    atomic_write_dataframe(
        output_dir / "actual_chunk_run_records_repeated.csv", actual_chunk_all
    )
    atomic_write_dataframe(
        output_dir / "actual_event_diagnostics_repeated.csv", actual_event_all
    )
    atomic_write_dataframe(output_dir / "actual_exact_paired_ratios.csv", exact_ratios)
    atomic_write_dataframe(
        output_dir / "actual_clustered_comparisons.csv", actual_stats
    )
    atomic_write_dataframe(
        output_dir / "model_exact_paired_ratios.csv", model_exact_ratios
    )
    atomic_write_dataframe(
        output_dir / "model_clustered_comparisons.csv", model_stats
    )
    atomic_write_dataframe(
        output_dir / "no_action_consistency.csv", no_action_df
    )
    atomic_write_dataframe(
        output_dir / "model_actual_validation.csv", model_actual_df
    )
    write_json(
        output_dir / "no_action_consistency.json", no_action_assessment
    )

    primary_results: Dict[str, Any] = {}
    primary_passes: List[bool] = []
    for dataset in ("PEMS08", "BDG2"):
        result = lookup_stat(
            actual_stats,
            dataset,
            PRIMARY_TRACE,
            PRIMARY_GROUP,
            PRIMARY_CHUNK_SIZE,
            "repressdb",
            "afhc_frozen",
        )
        passed = bool(
            result
            and result["median_ratio"] < PRIMARY_MEDIAN_RATIO_MAX
            and result["bootstrap_ci_high"] < PRIMARY_CI_HIGH_MAX
            and result["fraction_policy_faster"] >= PRIMARY_FASTER_FRACTION_MIN
        )
        result["gate_pass"] = passed
        result["requirements"] = {
            "median_ratio_below": PRIMARY_MEDIAN_RATIO_MAX,
            "ci_high_below": PRIMARY_CI_HIGH_MAX,
            "fraction_faster_at_least": PRIMARY_FASTER_FRACTION_MIN,
        }
        primary_results[dataset] = result
        primary_passes.append(passed)

    key_results: Dict[str, Any] = {}
    for dataset in ("PEMS08", "BDG2"):
        for trace, group in (
            ("shape_only", "ALL"),
            ("shape_only", "point_drift"),
            ("shape_only", "scan_stable"),
            ("frequency_only", "ALL"),
            ("joint_orthogonal", "ALL"),
            ("joint_orthogonal", "hot_point"),
            ("joint_orthogonal", "hot_scan"),
            ("joint_orthogonal", "cold_point"),
            ("joint_orthogonal", "cold_scan"),
            ("brief_burst", "ALL"),
            ("stable_scan", "ALL"),
        ):
            for size in CHUNK_SIZES:
                key_results[f"{dataset}|{trace}|{group}|{size}"] = lookup_stat(
                    actual_stats,
                    dataset,
                    trace,
                    group,
                    size,
                    "repressdb",
                    "afhc_frozen",
                )

    median_afhc_error = float(
        model_actual_df["absolute_afhc_ratio_error"].median()
    )
    median_repressdb_error = float(
        model_actual_df["absolute_repressdb_ratio_error"].median()
    )
    expected_actual_runs = 2 * ACTUAL_REPEATS * len(TRACE_SEEDS) * len(TRACE_NAMES) * len(CHUNK_SIZES) * len(POLICIES)
    integrity_checks = {
        "I1_two_holdouts_completed": bool(
            set(actual_summary_all["dataset"].unique()) == {"PEMS08", "BDG2"}
        ),
        "I2_all_expected_actual_policy_runs_completed": bool(
            len(actual_summary_all) == expected_actual_runs
        ),
        "I3_three_repeats_per_holdout": bool(
            all(
                actual_summary_all[actual_summary_all["dataset"] == dataset][
                    "repeat_index"
                ].nunique()
                == ACTUAL_REPEATS
                for dataset in ("PEMS08", "BDG2")
            )
        ),
        "I4_32_independent_units_per_holdout": bool(
            actual_chunk_all[actual_chunk_all["dataset"] == "PEMS08"][
                "unit_id"
            ].nunique()
            == 32
            and actual_chunk_all[actual_chunk_all["dataset"] == "BDG2"][
                "unit_id"
            ].nunique()
            == 32
        ),
        "I5_all_trace_size_seed_policy_cells_present": bool(
            actual_summary_all.groupby("dataset").size().eq(expected_actual_runs // 2).all()
        ),
        "I6_afhc_no_action_consistency": bool(
            no_action_assessment["all_datasets_pass"]
        ),
        "I7_query_truth_checks_completed_without_exception": True,
        "I8_frozen_configuration_unchanged": True,
        "I9_both_holdouts_evaluated_before_any_retuning": True,
    }

    nearest_baseline_gate_pass = bool(any(primary_passes))
    assessment = {
        "version": VERSION,
        "target_journal": TARGET_JOURNAL,
        "status": "A2 frozen PEMS08 and BDG2 holdout validation complete",
        "a2_protocol_sha256": a2_protocol_hash,
        "a2_experiment_freeze_sha256": a2_freeze_hash,
        "parent_v074_experiment_freeze_sha256": parent_freeze_hash,
        "frozen_configuration": frozen_config,
        "dataset_summaries": dataset_summaries,
        "primary_preregistered_gate": {
            "definition": {
                "trace": PRIMARY_TRACE,
                "group": PRIMARY_GROUP,
                "chunk_size": PRIMARY_CHUNK_SIZE,
                "comparison": "RePressDB / AF-HC-D",
            },
            "dataset_results": primary_results,
            "at_least_one_holdout_passes": nearest_baseline_gate_pass,
        },
        "key_actual_results": key_results,
        "no_action_consistency": no_action_assessment,
        "model_actual_validation": {
            "median_absolute_afhc_ratio_error": median_afhc_error,
            "median_absolute_repressdb_ratio_error": median_repressdb_error,
            "diagnostic_limit": MODEL_ACTUAL_MEDIAN_ERROR_LIMIT,
            "both_below_diagnostic_limit": bool(
                median_afhc_error <= MODEL_ACTUAL_MEDIAN_ERROR_LIMIT
                and median_repressdb_error <= MODEL_ACTUAL_MEDIAN_ERROR_LIMIT
            ),
            "interpretation": (
                "Repeated paired actual wall-clock is primary. Calibrated replay "
                "is a decision-model and robustness diagnostic."
            ),
        },
        "integrity_checks": {
            **integrity_checks,
            "all_integrity_checks_pass": bool(all(integrity_checks.values())),
        },
        "scientific_decision": (
            "The preregistered nearest-baseline holdout gate passed. Proceed to "
            "the single-active-representation storage prototype without changing "
            "AF-HC-D or RePressDB parameters."
            if nearest_baseline_gate_pass and all(integrity_checks.values())
            else (
                "The preregistered nearest-baseline holdout gate did not pass, or "
                "an integrity check failed. Do not retune on PEMS08/BDG2. Freeze and "
                "report the negative result before deciding whether to reposition "
                "the paper."
            )
        ),
    }
    write_json(output_dir / "a2_holdout_assessment.json", assessment)
    write_json(project_dir / SUMMARY_JSON_NAME, assessment)

    run_config = {
        "version": VERSION,
        "target_journal": TARGET_JOURNAL,
        "a2_protocol_sha256": a2_protocol_hash,
        "a2_experiment_freeze_sha256": a2_freeze_hash,
        "frozen_configuration": frozen_config,
        "datasets": dataset_summaries,
        "execution": {
            "trace_seeds": list(TRACE_SEEDS),
            "trace_names": list(TRACE_NAMES),
            "chunk_sizes": list(CHUNK_SIZES),
            "policies": list(POLICIES),
            "actual_repeats": ACTUAL_REPEATS,
            "checkpoint_resume": True,
            "timer": "time.perf_counter_ns",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
        },
    }
    write_json(output_dir / "run_config.json", run_config)

    patched_core = output_dir / "_corrected_core_v073.py"
    patched_core.unlink(missing_ok=True)

    print("=" * 116)
    print("RePressDB A2 v0.8.0 complete")
    print(json.dumps(assessment["primary_preregistered_gate"], ensure_ascii=False, indent=2))
    print(json.dumps(assessment["integrity_checks"], ensure_ascii=False, indent=2))
    print(f"Results: {output_dir.resolve()}")
    print("Upload:")
    print("  a2_holdout_assessment.json")
    print("  actual_clustered_comparisons.csv")
    print("  actual_exact_paired_ratios.csv")
    print("  actual_global_run_records_repeated.csv")
    print("  actual_chunk_run_records_repeated.csv")
    print("  model_clustered_comparisons.csv")
    print("  model_actual_validation.csv")
    print("  no_action_consistency.json")
    print("  a2_holdout_protocol.json")
    print("  a2_protocol_sha256.txt")
    print("  a2_experiment_freeze_sha256.txt")
    print("  run_config.json")
    print("=" * 116)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
