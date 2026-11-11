#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
RePressDB E5 v0.6.0
Frozen-controller cross-domain validation on Building Data Genome Project 2.

Target venue: EDBT 2027 (CCF-B).

DIRECT-RUN VERSION
------------------
Open this file in PyCharm and click the green Run button.
No command-line arguments and no source-code edits are required.

Required dependency files in the same directory:
  1) RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT_V0_2_0.py
  2) RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT.py
  3) RUN_THIS_ONLY_REPRESSDB_E0C_V0_2_0.py

BDG2 path is already fixed below:
  E:\IPM\大学\科研\E\buds-lab-building-data-genome-project-2-3d0cbaf

Protocol
--------
* Uses only data/meters/cleaned.
* No normalization, log transform, interpolation, or resampling.
* A meter is eligible only if it contains a contiguous hourly segment with:
    - at least 16,384 finite values;
    - non-zero variance;
    - at least 16 distinct float32 values;
    - at least 32 non-zero observations.
* At most one meter is selected from each building.
* 48 buildings are selected:
    - 16 chunks of length 2,048;
    - 16 chunks of length 8,192;
    - 16 chunks of length 16,384.
* Selection is deterministic and approximately stratified by meter type and site.
* The E1 v0.2 controller is frozen:
    - long window: 300 queries;
    - short window: 100 queries;
    - decision interval: 100 queries;
    - minimum observations: 200;
    - confirmation rounds: 3;
    - cooldown: 300 queries;
    - prediction horizon: 1,200 queries;
    - minimum relative improvement: 8%;
    - rewrite safety factor: 1.25.
* Three trace seeds and all four E1 traces are replayed.
* Source building is the independent statistical unit.

Important
---------
The first run scans BDG2 cleaned-meter files and creates a deterministic
selection manifest. Later runs reuse that manifest when it is still valid.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

VERSION = "0.6.0"
TARGET_VENUE = "EDBT 2027"
SEED = 20260718

BDG2_ROOT = Path(
    r"E:\IPM\大学\科研\E\buds-lab-building-data-genome-project-2-3d0cbaf"
)
BDG2_CLEANED_DIR = BDG2_ROOT / "data" / "meters" / "cleaned"
BDG2_METADATA_DIR = BDG2_ROOT / "data" / "metadata"

OUTPUT_DIR_NAME = "results_e5_bdg2_frozen_crossdomain_v060"
SUMMARY_JSON_NAME = "E5_BDG2_FROZEN_CROSSDOMAIN_ASSESSMENT_v060.json"

TRACE_SEEDS = (20260718, 20260719, 20260720)
CHUNK_SIZES = (2048, 8192, 16384)
CHUNKS_PER_SIZE = 16
TOTAL_SELECTED_BUILDINGS = CHUNKS_PER_SIZE * len(CHUNK_SIZES)

MIN_CONTIGUOUS_LENGTH = max(CHUNK_SIZES)
MIN_DISTINCT_VALUES = 16
MIN_NONZERO_COUNT = 32
VARIANCE_EPSILON = 1e-12

CATALOG_BATCH_SIZE = 192
MIN_ELIGIBLE_POOL = 96
MAX_SITE_COUNT_INITIAL = 4
MAX_SITE_COUNT_RELAXED = 8

SUPPORTED_SUFFIXES = (
    ".csv",
    ".csv.gz",
    ".parquet",
)

_TIMESTAMP_BREAK_CACHE: Dict[
    Tuple[str, str, int], Optional[np.ndarray]
] = {}

QUICK_TEST = os.environ.get("REPRESSDB_E5_QUICK", "0") == "1"


def stable_seed(*parts: Any) -> int:
    token = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def stable_order_key(*parts: Any) -> str:
    token = "|".join(str(part) for part in parts)
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


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


def load_module(filename: str, module_name: str) -> Any:
    path = Path(__file__).resolve().parent / filename
    if not path.exists():
        raise FileNotFoundError(
            f"找不到依赖文件：{filename}\n"
            "请下载完整包，并保证所有 Python 文件位于同一目录。"
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_modules() -> Tuple[Any, Any, Any]:
    e1v2 = load_module(
        "RUN_THIS_ONLY_REPRESSDB_E1_ONLINE_PEMS04_DIRECT_V0_2_0.py",
        "repressdb_e1_v020_for_e5",
    )
    e1 = e1v2.load_e1_module()
    base = e1.load_base_module()

    if getattr(e1v2, "VERSION", None) != "0.2.0":
        raise RuntimeError(
            "E5 需要冻结的 E1 v0.2.0 控制器。"
        )
    if getattr(e1, "VERSION", None) != "0.1.0":
        raise RuntimeError(
            "E5 需要 E1 v0.1.0 基础执行器。"
        )
    return e1v2, e1, base


def is_supported_meter_file(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)


def discover_meter_files() -> List[Path]:
    if not BDG2_ROOT.exists():
        raise FileNotFoundError(
            f"BDG2 根目录不存在：{BDG2_ROOT}"
        )
    if not BDG2_CLEANED_DIR.exists():
        raise FileNotFoundError(
            "找不到 BDG2 cleaned meters 目录：\n"
            f"{BDG2_CLEANED_DIR}\n"
            "请确认解压后的目录结构中存在 data/meters/cleaned。"
        )

    files = sorted(
        path
        for path in BDG2_CLEANED_DIR.rglob("*")
        if path.is_file() and is_supported_meter_file(path)
    )
    if not files:
        raise FileNotFoundError(
            f"{BDG2_CLEANED_DIR} 中没有找到 CSV、CSV.GZ 或 Parquet 文件。"
        )
    return files


def infer_meter_type(path: Path) -> str:
    name = path.name.lower()
    name = re.sub(r"\.csv(\.gz)?$|\.parquet$", "", name)
    mappings = (
        (("electricity", "electric"), "electricity"),
        (("chilledwater", "chilled_water", "chilled-water"), "chilledwater"),
        (("hotwater", "hot_water", "hot-water"), "hotwater"),
        (("steam",), "steam"),
        (("irrigation",), "irrigation"),
        (("solar",), "solar"),
        (("gas",), "gas"),
        (("water",), "water"),
    )
    for keywords, label in mappings:
        if any(keyword in name for keyword in keywords):
            return label

    cleaned = re.sub(
        r"(^|[_-])(cleaned|meter|meters|data|2016|2017|all)([_-]|$)",
        "_",
        name,
    )
    cleaned = re.sub(r"[_-]+", "_", cleaned).strip("_")
    return cleaned or "unknown"


def read_columns(path: Path) -> List[str]:
    lower = path.name.lower()
    if lower.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                f"读取 Parquet 需要 pyarrow：{path}"
            ) from exc
        return list(pq.ParquetFile(path).schema.names)

    return list(pd.read_csv(path, nrows=0).columns)


def detect_time_column(columns: Sequence[str]) -> Optional[str]:
    if not columns:
        return None

    preferred = (
        "timestamp",
        "datetime",
        "date_time",
        "time",
        "date",
    )
    normalized = {
        str(column).strip().lower(): str(column)
        for column in columns
    }
    for candidate in preferred:
        if candidate in normalized:
            return normalized[candidate]

    first = str(columns[0])
    first_lower = first.strip().lower()
    if (
        first_lower.startswith("unnamed")
        or first_lower in {"index", "level_0"}
        or "time" in first_lower
        or "date" in first_lower
    ):
        return first
    return None


def read_selected_columns(
    path: Path,
    columns: Sequence[str],
) -> pd.DataFrame:
    selected = list(dict.fromkeys(str(column) for column in columns))
    lower = path.name.lower()

    if lower.endswith(".parquet"):
        frame = pd.read_parquet(path, columns=selected)
    else:
        frame = pd.read_csv(
            path,
            usecols=selected,
            low_memory=False,
        )
    return frame


def derive_site_id(building_id: str) -> str:
    value = str(building_id)
    if "_" in value:
        return value.split("_", 1)[0]
    if "-" in value:
        return value.split("-", 1)[0]
    return "unknown"


def discover_metadata(
    candidate_building_ids: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    if not BDG2_METADATA_DIR.exists():
        return {}

    candidates = set(str(value) for value in candidate_building_ids)
    best_overlap = 0
    best_mapping: Dict[str, Dict[str, Any]] = {}

    metadata_files = sorted(
        path
        for path in BDG2_METADATA_DIR.rglob("*")
        if path.is_file()
        and (
            path.name.lower().endswith(".csv")
            or path.name.lower().endswith(".csv.gz")
        )
    )

    for path in metadata_files:
        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if frame.empty:
            continue

        id_column = None
        overlap = 0
        for column in frame.columns:
            values = set(frame[column].astype(str))
            local_overlap = len(values.intersection(candidates))
            if local_overlap > overlap:
                overlap = local_overlap
                id_column = column

        if id_column is None or overlap <= best_overlap:
            continue

        normalized_columns = {
            str(column).lower(): column
            for column in frame.columns
        }

        def find_column(names: Sequence[str]) -> Optional[str]:
            for name in names:
                if name in normalized_columns:
                    return normalized_columns[name]
            return None

        site_column = find_column(
            ("site_id", "site", "site_name", "siteid")
        )
        primary_use_column = find_column(
            (
                "primaryspaceusage",
                "primary_use",
                "primaryuse",
                "primary_space_usage",
            )
        )
        timezone_column = find_column(
            ("timezone", "time_zone")
        )

        mapping: Dict[str, Dict[str, Any]] = {}
        for _, row_series in frame.iterrows():
            building_id = str(row_series[id_column])
            if building_id not in candidates:
                continue
            mapping[building_id] = {
                "site_id": (
                    str(row_series[site_column])
                    if site_column is not None
                    else derive_site_id(building_id)
                ),
                "primary_use": (
                    str(row_series[primary_use_column])
                    if primary_use_column is not None
                    else "unknown"
                ),
                "timezone": (
                    str(row_series[timezone_column])
                    if timezone_column is not None
                    else "unknown"
                ),
                "metadata_file": str(path),
                "metadata_id_column": str(id_column),
            }

        best_overlap = overlap
        best_mapping = mapping

    return best_mapping


def build_candidate_catalog(
    meter_files: Sequence[Path],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    all_building_ids: List[str] = []

    for path in meter_files:
        columns = read_columns(path)
        time_column = detect_time_column(columns)
        series_columns = [
            str(column)
            for column in columns
            if time_column is None or str(column) != str(time_column)
        ]
        meter_type = infer_meter_type(path)
        for column in series_columns:
            building_id = str(column)
            all_building_ids.append(building_id)
            rows.append(
                {
                    "source_file": str(path),
                    "source_column": column,
                    "time_column": (
                        str(time_column)
                        if time_column is not None
                        else ""
                    ),
                    "meter_type": meter_type,
                    "building_id": building_id,
                }
            )

    if not rows:
        raise RuntimeError("BDG2 cleaned meter 文件中没有发现时序列。")

    metadata = discover_metadata(all_building_ids)
    for row in rows:
        info = metadata.get(str(row["building_id"]), {})
        row["site_id"] = str(
            info.get(
                "site_id",
                derive_site_id(str(row["building_id"])),
            )
        )
        row["primary_use"] = str(
            info.get("primary_use", "unknown")
        )
        row["timezone"] = str(
            info.get("timezone", "unknown")
        )
        row["metadata_file"] = str(
            info.get("metadata_file", "")
        )

    catalog = pd.DataFrame(rows)
    catalog["order_key"] = [
        stable_order_key(
            SEED,
            row.source_file,
            row.source_column,
            row.meter_type,
        )
        for row in catalog.itertuples(index=False)
    ]
    catalog = catalog.sort_values(
        ["order_key", "meter_type", "building_id"]
    ).reset_index(drop=True)
    return catalog


def timestamp_break_mask(
    path: Path,
    time_column: str,
    expected_length: int,
) -> Optional[np.ndarray]:
    if not time_column:
        return None

    cache_key = (
        str(path.resolve()),
        str(time_column),
        int(expected_length),
    )
    if cache_key in _TIMESTAMP_BREAK_CACHE:
        cached = _TIMESTAMP_BREAK_CACHE[cache_key]
        return (
            None
            if cached is None
            else cached.copy()
        )

    try:
        frame = read_selected_columns(path, [time_column])
        parsed = pd.to_datetime(
            frame[time_column],
            errors="coerce",
            utc=False,
        )
    except Exception:
        _TIMESTAMP_BREAK_CACHE[cache_key] = None
        return None

    if len(parsed) != expected_length:
        _TIMESTAMP_BREAK_CACHE[cache_key] = None
        return None
    valid_fraction = float(parsed.notna().mean())
    if valid_fraction < 0.95:
        _TIMESTAMP_BREAK_CACHE[cache_key] = None
        return None

    values = parsed.to_numpy()
    break_mask = np.zeros(expected_length, dtype=bool)
    if expected_length > 1:
        diffs = pd.Series(parsed).diff()
        gap = diffs != pd.Timedelta(hours=1)
        gap.iloc[0] = False
        break_mask = gap.fillna(True).to_numpy(dtype=bool)

    _TIMESTAMP_BREAK_CACHE[cache_key] = break_mask.copy()
    return break_mask


def longest_true_run(mask: np.ndarray) -> Tuple[int, int]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return 0, 0
    padded = np.concatenate(
        (
            np.asarray([False]),
            mask,
            np.asarray([False]),
        )
    )
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    lengths = ends - starts
    best = int(np.argmax(lengths))
    return int(starts[best]), int(lengths[best])


def evaluate_candidate_batch(
    batch: pd.DataFrame,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for source_file, group in batch.groupby(
        "source_file",
        sort=False,
    ):
        path = Path(str(source_file))
        columns = list(group["source_column"].astype(str))
        time_column = str(group.iloc[0]["time_column"])
        requested = columns + (
            [time_column] if time_column else []
        )

        try:
            frame = read_selected_columns(path, requested)
        except Exception as exc:
            for row in group.itertuples(index=False):
                results.append(
                    {
                        **row._asdict(),
                        "eligible": False,
                        "failure_reason": (
                            f"read_error:{type(exc).__name__}"
                        ),
                    }
                )
            continue

        n_rows = len(frame)
        break_mask = timestamp_break_mask(
            path,
            time_column,
            n_rows,
        )

        for row in group.itertuples(index=False):
            column = str(row.source_column)
            try:
                values = pd.to_numeric(
                    frame[column],
                    errors="coerce",
                ).to_numpy(dtype=np.float32)
            except Exception as exc:
                results.append(
                    {
                        **row._asdict(),
                        "eligible": False,
                        "failure_reason": (
                            f"numeric_error:{type(exc).__name__}"
                        ),
                    }
                )
                continue

            valid = np.isfinite(values)
            if break_mask is not None:
                valid = valid & (~break_mask)

            run_start, run_length = longest_true_run(valid)
            if run_length <= 0:
                results.append(
                    {
                        **row._asdict(),
                        "eligible": False,
                        "failure_reason": "no_finite_run",
                        "longest_run_start": 0,
                        "longest_run_length": 0,
                    }
                )
                continue

            segment = values[
                run_start : run_start + run_length
            ]
            variance = float(
                np.var(segment.astype(np.float64))
            )
            distinct_count = int(
                np.unique(segment).size
            )
            nonzero_count = int(
                np.count_nonzero(
                    np.abs(segment) > 1e-12
                )
            )

            reasons: List[str] = []
            if run_length < MIN_CONTIGUOUS_LENGTH:
                reasons.append("run_too_short")
            if not np.isfinite(variance) or variance <= VARIANCE_EPSILON:
                reasons.append("near_constant")
            if distinct_count < MIN_DISTINCT_VALUES:
                reasons.append("too_few_distinct_values")
            if nonzero_count < MIN_NONZERO_COUNT:
                reasons.append("too_few_nonzero_values")

            results.append(
                {
                    **row._asdict(),
                    "eligible": len(reasons) == 0,
                    "failure_reason": "|".join(reasons),
                    "n_rows": int(n_rows),
                    "longest_run_start": int(run_start),
                    "longest_run_length": int(run_length),
                    "variance": variance,
                    "distinct_count": distinct_count,
                    "nonzero_count": nonzero_count,
                    "nonzero_fraction": float(
                        nonzero_count / run_length
                    ),
                    "minimum": float(np.min(segment)),
                    "maximum": float(np.max(segment)),
                    "mean": float(
                        np.mean(segment.astype(np.float64))
                    ),
                    "standard_deviation": float(
                        np.std(segment.astype(np.float64))
                    ),
                    "timestamp_continuity_checked": bool(
                        break_mask is not None
                    ),
                }
            )
    return results


def can_select_enough(
    eligible_df: pd.DataFrame,
) -> bool:
    return (
        eligible_df["building_id"].nunique()
        >= TOTAL_SELECTED_BUILDINGS
    )


def build_or_reuse_eligible_catalog(
    output_dir: Path,
) -> pd.DataFrame:
    eligible_path = (
        output_dir / "bdg2_eligible_series_catalog.csv"
    )
    if eligible_path.exists():
        try:
            cached = pd.read_csv(eligible_path)
            required = {
                "source_file",
                "source_column",
                "building_id",
                "eligible",
                "longest_run_start",
                "longest_run_length",
            }
            if required.issubset(cached.columns):
                source_files_exist = all(
                    Path(str(path)).exists()
                    for path in cached["source_file"].unique()
                )
                if source_files_exist:
                    print(
                        "[BDG2] reuse existing eligible-series catalog"
                    )
                    return cached
        except Exception:
            pass

    meter_files = discover_meter_files()
    catalog = build_candidate_catalog(meter_files)

    print(
        f"[BDG2] cleaned meter files={len(meter_files)}, "
        f"candidate columns={len(catalog)}"
    )

    evaluated_rows: List[Dict[str, Any]] = []
    cursor = 0

    while cursor < len(catalog):
        batch = catalog.iloc[
            cursor : cursor + CATALOG_BATCH_SIZE
        ].copy()
        evaluated_rows.extend(
            evaluate_candidate_batch(batch)
        )
        cursor += len(batch)

        evaluated_df = pd.DataFrame(evaluated_rows)
        eligible_df = evaluated_df[
            evaluated_df["eligible"].astype(bool)
        ]
        print(
            f"[BDG2 qualification] "
            f"evaluated={len(evaluated_df)}/{len(catalog)}, "
            f"eligible meters={len(eligible_df)}, "
            f"unique buildings={eligible_df['building_id'].nunique()}"
        )

        if (
            len(eligible_df) >= MIN_ELIGIBLE_POOL
            and can_select_enough(eligible_df)
        ):
            break

    evaluated_df = pd.DataFrame(evaluated_rows)
    evaluated_df.to_csv(eligible_path, index=False)

    eligible_df = evaluated_df[
        evaluated_df["eligible"].astype(bool)
    ].copy()
    if not can_select_enough(eligible_df):
        raise RuntimeError(
            "符合连续有效长度要求的独立建筑不足 48 个。\n"
            f"当前合格建筑数：{eligible_df['building_id'].nunique()}\n"
            f"资格目录已保存：{eligible_path.resolve()}"
        )
    return evaluated_df


def select_buildings_round_robin(
    eligible_df: pd.DataFrame,
    site_cap: int,
) -> pd.DataFrame:
    eligible = eligible_df[
        eligible_df["eligible"].astype(bool)
    ].copy()
    eligible = eligible.sort_values(
        ["meter_type", "order_key", "building_id"]
    )

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for meter_type, group in eligible.groupby(
        "meter_type",
        sort=True,
    ):
        groups[str(meter_type)] = [
            row._asdict()
            for row in group.itertuples(index=False)
        ]

    selected: List[Dict[str, Any]] = []
    used_buildings: set[str] = set()
    site_counts: Counter[str] = Counter()
    meter_types = sorted(groups)
    positions = {meter_type: 0 for meter_type in meter_types}

    progress = True
    while (
        len(selected) < TOTAL_SELECTED_BUILDINGS
        and progress
    ):
        progress = False
        for meter_type in meter_types:
            candidates = groups[meter_type]
            position = positions[meter_type]

            while position < len(candidates):
                candidate = candidates[position]
                position += 1
                building_id = str(
                    candidate["building_id"]
                )
                site_id = str(candidate["site_id"])
                if building_id in used_buildings:
                    continue
                if site_counts[site_id] >= site_cap:
                    continue

                selected.append(candidate)
                used_buildings.add(building_id)
                site_counts[site_id] += 1
                progress = True
                break

            positions[meter_type] = position
            if len(selected) >= TOTAL_SELECTED_BUILDINGS:
                break

    return pd.DataFrame(selected)


def build_or_reuse_selection(
    output_dir: Path,
    evaluated_df: pd.DataFrame,
) -> pd.DataFrame:
    selection_path = (
        output_dir / "bdg2_selected_series.csv"
    )
    if selection_path.exists():
        try:
            selected = pd.read_csv(selection_path)
            required = {
                "source_file",
                "source_column",
                "building_id",
                "chunk_size",
                "selected_chunk_start",
            }
            if (
                required.issubset(selected.columns)
                and len(selected) == TOTAL_SELECTED_BUILDINGS
                and selected["building_id"].nunique()
                == TOTAL_SELECTED_BUILDINGS
                and all(
                    Path(str(path)).exists()
                    for path in selected["source_file"].unique()
                )
            ):
                print(
                    "[BDG2] reuse existing deterministic selection manifest"
                )
                return selected
        except Exception:
            pass

    selected = select_buildings_round_robin(
        evaluated_df,
        MAX_SITE_COUNT_INITIAL,
    )
    if len(selected) < TOTAL_SELECTED_BUILDINGS:
        selected = select_buildings_round_robin(
            evaluated_df,
            MAX_SITE_COUNT_RELAXED,
        )

    if len(selected) < TOTAL_SELECTED_BUILDINGS:
        eligible = evaluated_df[
            evaluated_df["eligible"].astype(bool)
        ].sort_values(
            ["order_key", "meter_type", "building_id"]
        )
        already = set(
            selected["building_id"].astype(str)
        )
        additions: List[Dict[str, Any]] = []
        for row in eligible.itertuples(index=False):
            building_id = str(row.building_id)
            if building_id in already:
                continue
            additions.append(row._asdict())
            already.add(building_id)
            if (
                len(selected) + len(additions)
                >= TOTAL_SELECTED_BUILDINGS
            ):
                break
        if additions:
            selected = pd.concat(
                [selected, pd.DataFrame(additions)],
                ignore_index=True,
            )

    if len(selected) < TOTAL_SELECTED_BUILDINGS:
        raise RuntimeError(
            "无法选择 48 个不同 building_id 的合格 meters。"
        )

    selected = selected.iloc[
        :TOTAL_SELECTED_BUILDINGS
    ].copy()

    # The round-robin selection order is also used to balance meter types
    # across the three chunk sizes.
    assigned_sizes = [
        CHUNK_SIZES[index % len(CHUNK_SIZES)]
        for index in range(len(selected))
    ]
    selected["chunk_size"] = assigned_sizes

    selected_starts: List[int] = []
    chunk_ids: List[str] = []
    for row in selected.itertuples(index=False):
        run_start = int(row.longest_run_start)
        run_length = int(row.longest_run_length)
        chunk_size = int(row.chunk_size)
        available = run_length - chunk_size
        rng = np.random.default_rng(
            stable_seed(
                SEED,
                row.source_file,
                row.source_column,
                chunk_size,
                "chunk_offset",
            )
        )
        offset = (
            int(rng.integers(0, available + 1))
            if available > 0
            else 0
        )
        selected_start = run_start + offset
        selected_starts.append(selected_start)

        identifier = hashlib.sha1(
            (
                f"{row.building_id}|{row.meter_type}|"
                f"{row.source_file}|{row.source_column}|"
                f"{chunk_size}|{selected_start}"
            ).encode("utf-8")
        ).hexdigest()[:12]
        chunk_ids.append(
            f"bdg2_cs{chunk_size}_{identifier}"
        )

    selected["selected_chunk_start"] = (
        selected_starts
    )
    selected["chunk_id"] = chunk_ids

    counts = (
        selected["chunk_size"].value_counts().to_dict()
    )
    expected = {
        size: CHUNKS_PER_SIZE for size in CHUNK_SIZES
    }
    if {
        int(key): int(value)
        for key, value in counts.items()
    } != expected:
        raise AssertionError(
            f"chunk size assignment incorrect: {counts}"
        )

    selected.to_csv(selection_path, index=False)
    return selected


def load_selected_chunks(
    base: Any,
    selected_df: pd.DataFrame,
) -> Tuple[List[Any], Dict[str, Dict[str, Any]]]:
    chunks: List[Any] = []
    metadata_by_chunk: Dict[str, Dict[str, Any]] = {}

    series_index = 0
    for source_file, group in selected_df.groupby(
        "source_file",
        sort=False,
    ):
        path = Path(str(source_file))
        columns = list(group["source_column"].astype(str))
        frame = read_selected_columns(path, columns)

        for row in group.itertuples(index=False):
            values_all = pd.to_numeric(
                frame[str(row.source_column)],
                errors="coerce",
            ).to_numpy(dtype=np.float32)

            start = int(row.selected_chunk_start)
            size = int(row.chunk_size)
            values = np.ascontiguousarray(
                values_all[start : start + size],
                dtype=np.float32,
            )
            if len(values) != size:
                raise AssertionError(
                    f"selected chunk is too short: {row.chunk_id}"
                )
            if not np.isfinite(values).all():
                raise AssertionError(
                    f"selected chunk contains non-finite values: {row.chunk_id}"
                )

            chunk = base.ChunkUnit(
                chunk_id=str(row.chunk_id),
                series_index=int(series_index),
                chunk_start=int(start),
                chunk_size=int(size),
                values=values,
            )
            chunks.append(chunk)
            metadata_by_chunk[str(row.chunk_id)] = {
                "building_id": str(row.building_id),
                "meter_type": str(row.meter_type),
                "site_id": str(row.site_id),
                "primary_use": str(row.primary_use),
                "timezone": str(row.timezone),
                "source_file": str(row.source_file),
                "source_column": str(row.source_column),
                "longest_run_start": int(
                    row.longest_run_start
                ),
                "longest_run_length": int(
                    row.longest_run_length
                ),
                "selected_chunk_start": int(start),
                "chunk_size": int(size),
            }
            series_index += 1

    chunks = sorted(
        chunks,
        key=lambda chunk: (
            int(chunk.chunk_size),
            str(chunk.chunk_id),
        ),
    )
    return chunks, metadata_by_chunk


def bootstrap_median_ci(
    values: np.ndarray,
    seed: int,
    iterations: int = 5000,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=float)
    for index in range(iterations):
        samples[index] = np.median(
            rng.choice(
                values,
                size=values.size,
                replace=True,
            )
        )
    return (
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def building_clustered_statistics(
    run_df: pd.DataFrame,
) -> pd.DataFrame:
    references = (
        "fixed_whole_lz4",
        "initial_best",
        "best_static",
        "periodic_greedy",
        "oracle",
    )
    rows: List[Dict[str, Any]] = []

    building_df = (
        run_df.groupby(
            [
                "trace",
                "chunk_size",
                "building_id",
                "policy",
            ],
            as_index=False,
        )["total_time_ns"]
        .median()
    )

    for (trace, chunk_size), group in building_df.groupby(
        ["trace", "chunk_size"],
        sort=True,
    ):
        pivot = group.pivot(
            index="building_id",
            columns="policy",
            values="total_time_ns",
        ).dropna()
        if "repressdb" not in pivot:
            continue

        for reference in references:
            if reference not in pivot:
                continue

            repress = pivot["repressdb"].to_numpy(
                dtype=float
            )
            baseline = pivot[reference].to_numpy(
                dtype=float
            )
            ratios = repress / baseline

            try:
                p_value = float(
                    wilcoxon(
                        repress,
                        baseline,
                        alternative="two-sided",
                        zero_method="wilcox",
                    ).pvalue
                )
            except ValueError:
                p_value = 1.0

            ci_low, ci_high = bootstrap_median_ci(
                ratios,
                stable_seed(
                    SEED,
                    trace,
                    chunk_size,
                    reference,
                    "building_clustered_bootstrap",
                ),
            )
            rows.append(
                {
                    "trace": str(trace),
                    "chunk_size": int(chunk_size),
                    "reference_policy": reference,
                    "n_independent_buildings": int(
                        len(pivot)
                    ),
                    "median_ratio_repressdb_over_reference": float(
                        np.median(ratios)
                    ),
                    "mean_ratio_repressdb_over_reference": float(
                        np.mean(ratios)
                    ),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "fraction_repressdb_faster": float(
                        np.mean(ratios < 1.0)
                    ),
                    "wilcoxon_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def static_layout_records(
    e1: Any,
    base: Any,
    chunk: Any,
    profile: Mapping[str, Mapping[str, float]],
    metadata: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for workload_name in (
        "scan_heavy",
        "point_heavy",
        "aggregate_heavy",
    ):
        best_codec, best_cost = e1.choose_best_codec(
            profile,
            base.WORKLOADS[workload_name],
        )
        whole_cost = e1.workload_cost(
            profile,
            "whole_lz4",
            base.WORKLOADS[workload_name],
        )
        rows.append(
            {
                "chunk_id": chunk.chunk_id,
                "building_id": metadata["building_id"],
                "meter_type": metadata["meter_type"],
                "site_id": metadata["site_id"],
                "primary_use": metadata["primary_use"],
                "chunk_size": int(chunk.chunk_size),
                "workload": workload_name,
                "best_codec": best_codec,
                "best_expected_query_ns": float(
                    best_cost
                ),
                "whole_lz4_expected_query_ns": float(
                    whole_cost
                ),
                "whole_lz4_regret_ratio": float(
                    whole_cost / best_cost
                    if best_cost > 0
                    else math.nan
                ),
            }
        )
    return rows


def static_layout_summary(
    static_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (
        chunk_size,
        workload,
    ), group in static_df.groupby(
        ["chunk_size", "workload"],
        sort=True,
    ):
        counts = group["best_codec"].value_counts()
        modal_codec = str(counts.index[0])
        rows.append(
            {
                "chunk_size": int(chunk_size),
                "workload": str(workload),
                "n_buildings": int(
                    group["building_id"].nunique()
                ),
                "n_distinct_best_codecs": int(
                    group["best_codec"].nunique()
                ),
                "modal_best_codec": modal_codec,
                "modal_fraction": float(
                    counts.iloc[0] / len(group)
                ),
                "median_whole_lz4_regret_ratio": float(
                    group[
                        "whole_lz4_regret_ratio"
                    ].median()
                ),
                "p90_whole_lz4_regret_ratio": float(
                    group[
                        "whole_lz4_regret_ratio"
                    ].quantile(0.90)
                ),
            }
        )
    return pd.DataFrame(rows)


def switch_near_optimality(
    e1: Any,
    base: Any,
    decisions_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
) -> Dict[str, Any]:
    profiles: Dict[
        str, Dict[str, Dict[str, float]]
    ] = {}
    for chunk_id, group in calibration_df.groupby(
        "chunk_id"
    ):
        profile: Dict[
            str, Dict[str, float]
        ] = {}
        for codec, local in group.groupby("codec"):
            profile[str(codec)] = {
                str(row.query_class): float(
                    row.median_latency_ns
                )
                for row in local.itertuples(index=False)
            }
        profiles[str(chunk_id)] = profile

    switches = decisions_df[
        (decisions_df["policy"] == "repressdb")
        & (decisions_df["decision"] == "switch")
    ]
    ratios: List[float] = []
    for row in switches.itertuples(index=False):
        profile = profiles[str(row.chunk_id)]
        weights = base.WORKLOADS[str(row.workload)]
        best_codec, best_cost = e1.choose_best_codec(
            profile, weights
        )
        candidate = str(row.candidate_codec)
        candidate_cost = e1.workload_cost(
            profile,
            candidate,
            weights,
        )
        ratios.append(
            float(
                candidate_cost / best_cost
                if best_cost > 0
                else math.nan
            )
        )

    ratio_array = np.asarray(ratios, dtype=float)
    ratio_array = ratio_array[
        np.isfinite(ratio_array)
    ]
    if ratio_array.size == 0:
        return {
            "n_switches": 0,
            "exact_best_fraction": 1.0,
            "within_1_percent_fraction": 1.0,
            "within_2_percent_fraction": 1.0,
            "within_5_percent_fraction": 1.0,
            "maximum_candidate_cost_ratio_to_phase_best": 1.0,
        }

    return {
        "n_switches": int(ratio_array.size),
        "exact_best_fraction": float(
            np.mean(
                np.isclose(
                    ratio_array,
                    1.0,
                    rtol=1e-12,
                    atol=1e-12,
                )
            )
        ),
        "within_1_percent_fraction": float(
            np.mean(ratio_array <= 1.01)
        ),
        "within_2_percent_fraction": float(
            np.mean(ratio_array <= 1.02)
        ),
        "within_5_percent_fraction": float(
            np.mean(ratio_array <= 1.05)
        ),
        "maximum_candidate_cost_ratio_to_phase_best": float(
            np.max(ratio_array)
        ),
    }


def get_comparison(
    comparisons_df: pd.DataFrame,
    trace: str,
    chunk_size: int,
    reference: str,
) -> Optional[pd.Series]:
    local = comparisons_df[
        (comparisons_df["trace"] == trace)
        & (
            comparisons_df["chunk_size"]
            == chunk_size
        )
        & (
            comparisons_df["reference_policy"]
            == reference
        )
    ]
    return local.iloc[0] if len(local) == 1 else None


def make_assessment(
    e1: Any,
    run_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    comparisons_df: pd.DataFrame,
    decisions_df: pd.DataFrame,
    static_df: pd.DataFrame,
    static_summary_df: pd.DataFrame,
    near_optimality: Mapping[str, Any],
    dataset_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    primary: Dict[str, Any] = {}
    primary_pass = True

    for chunk_size in (8192, 16384):
        comparison = get_comparison(
            comparisons_df,
            "scan_to_point",
            chunk_size,
            "initial_best",
        )
        if comparison is None:
            primary_pass = False
            continue

        record = {
            "median_ratio_vs_initial_best": float(
                comparison[
                    "median_ratio_repressdb_over_reference"
                ]
            ),
            "bootstrap_ci_low": float(
                comparison["bootstrap_ci_low"]
            ),
            "bootstrap_ci_high": float(
                comparison["bootstrap_ci_high"]
            ),
            "fraction_repressdb_faster": float(
                comparison[
                    "fraction_repressdb_faster"
                ]
            ),
            "wilcoxon_p": float(
                comparison["wilcoxon_p"]
            ),
            "n_independent_buildings": int(
                comparison[
                    "n_independent_buildings"
                ]
            ),
        }
        primary[str(chunk_size)] = record
        primary_pass = (
            primary_pass
            and record[
                "median_ratio_vs_initial_best"
            ]
            < 1.0
            and record["bootstrap_ci_high"] < 1.0
            and record["wilcoxon_p"] <= 0.05
            and record[
                "fraction_repressdb_faster"
            ]
            >= 0.75
        )

    point_2048 = get_comparison(
        comparisons_df,
        "scan_to_point",
        2048,
        "initial_best",
    )
    point_2048_nonregression = bool(
        point_2048 is not None
        and float(
            point_2048[
                "median_ratio_repressdb_over_reference"
            ]
        )
        <= 1.05
    )

    burst_summary: Dict[str, Any] = {}
    burst_pass = True
    for chunk_size in CHUNK_SIZES:
        local = aggregate_df[
            (
                aggregate_df["trace"]
                == "brief_point_burst"
            )
            & (
                aggregate_df["chunk_size"]
                == chunk_size
            )
        ].set_index("policy")
        if (
            "repressdb" not in local.index
            or "periodic_greedy"
            not in local.index
        ):
            burst_pass = False
            continue

        repress_migrations = float(
            local.loc[
                "repressdb",
                "median_migration_count",
            ]
        )
        periodic_migrations = float(
            local.loc[
                "periodic_greedy",
                "median_migration_count",
            ]
        )
        burst_summary[str(chunk_size)] = {
            "repressdb_median_migrations": (
                repress_migrations
            ),
            "periodic_median_migrations": (
                periodic_migrations
            ),
        }
        burst_pass = (
            burst_pass
            and repress_migrations
            <= periodic_migrations
        )

    repress_switches = decisions_df[
        (decisions_df["policy"] == "repressdb")
        & (decisions_df["decision"] == "switch")
    ]
    stale_count = int(
        repress_switches[
            "stale_switch_after_brief_burst"
        ].sum()
    )
    burst_pass = burst_pass and stale_count == 0

    per_seed: Dict[str, Dict[str, float]] = {}
    seed_pass = True
    pivot = run_df.pivot(
        index=[
            "trace_seed",
            "building_id",
            "trace",
            "chunk_size",
        ],
        columns="policy",
        values="total_time_ns",
    ).dropna()

    for trace_seed in TRACE_SEEDS:
        local_result: Dict[str, float] = {}
        for chunk_size in (8192, 16384):
            index = pivot.index
            mask = (
                (
                    index.get_level_values(
                        "trace_seed"
                    )
                    == trace_seed
                )
                & (
                    index.get_level_values("trace")
                    == "scan_to_point"
                )
                & (
                    index.get_level_values(
                        "chunk_size"
                    )
                    == chunk_size
                )
            )
            local = pivot.loc[mask]
            ratio = float(
                np.median(
                    local["repressdb"]
                    / local["initial_best"]
                )
            )
            local_result[str(chunk_size)] = ratio
            seed_pass = seed_pass and ratio < 1.0
        per_seed[str(trace_seed)] = local_result

    modal_by_workload = (
        static_summary_df.groupby("workload")[
            "modal_best_codec"
        ].nunique()
    )
    distinct_modal_codecs = int(
        static_summary_df[
            "modal_best_codec"
        ].nunique()
    )
    differing_scan_point_fraction = float(
        (
            static_df.pivot(
                index="chunk_id",
                columns="workload",
                values="best_codec",
            )["scan_heavy"]
            != static_df.pivot(
                index="chunk_id",
                columns="workload",
                values="best_codec",
            )["point_heavy"]
        ).mean()
    )
    static_pass = bool(
        distinct_modal_codecs >= 2
        and differing_scan_point_fraction
        >= 0.50
    )

    near_optimal_pass = bool(
        float(
            near_optimality[
                "within_2_percent_fraction"
            ]
        )
        >= 0.95
    )

    gates = {
        "X1_static_layout_depends_on_workload": static_pass,
        "X2_scan_to_point_significant_at_8192_and_16384": bool(
            primary_pass
        ),
        "X3_scan_to_point_2048_nonregression": (
            point_2048_nonregression
        ),
        "X4_brief_burst_is_stable": bool(
            burst_pass
        ),
        "X5_point_drift_gain_repeats_in_all_trace_seeds": bool(
            seed_pass
        ),
        "X6_at_least_95_percent_switches_within_2_percent_of_phase_best": bool(
            near_optimal_pass
        ),
    }

    aggregate_exploratory: Dict[str, Any] = {}
    for chunk_size in CHUNK_SIZES:
        comparison = get_comparison(
            comparisons_df,
            "scan_to_aggregate",
            chunk_size,
            "initial_best",
        )
        if comparison is not None:
            aggregate_exploratory[str(chunk_size)] = {
                "median_ratio_vs_initial_best": float(
                    comparison[
                        "median_ratio_repressdb_over_reference"
                    ]
                ),
                "bootstrap_ci_low": float(
                    comparison["bootstrap_ci_low"]
                ),
                "bootstrap_ci_high": float(
                    comparison["bootstrap_ci_high"]
                ),
                "wilcoxon_p": float(
                    comparison["wilcoxon_p"]
                ),
            }

    return {
        "version": VERSION,
        "target_venue": TARGET_VENUE,
        "dataset_label": dataset_meta[
            "dataset_label"
        ],
        "status": (
            "BDG2 frozen-controller cross-domain validation"
        ),
        "study_role": (
            "Cross-domain validation after all E1 controller constants "
            "were frozen on PEMS04. No BDG2-specific retuning is performed."
        ),
        "primary_execution_mode": "warm_slicedmeta",
        "validation_gates": {
            **gates,
            "all_six": bool(all(gates.values())),
        },
        "primary_scan_to_point": primary,
        "scan_to_point_2048": (
            {
                "median_ratio_vs_initial_best": float(
                    point_2048[
                        "median_ratio_repressdb_over_reference"
                    ]
                ),
                "bootstrap_ci_low": float(
                    point_2048[
                        "bootstrap_ci_low"
                    ]
                ),
                "bootstrap_ci_high": float(
                    point_2048[
                        "bootstrap_ci_high"
                    ]
                ),
                "wilcoxon_p": float(
                    point_2048["wilcoxon_p"]
                ),
            }
            if point_2048 is not None
            else {}
        ),
        "brief_burst": {
            "by_chunk_size": burst_summary,
            "stale_post_burst_switch_count": (
                stale_count
            ),
        },
        "static_layout": {
            "n_distinct_modal_codecs": (
                distinct_modal_codecs
            ),
            "fraction_chunks_with_different_scan_and_point_best_codec": (
                differing_scan_point_fraction
            ),
        },
        "switch_near_optimality": dict(
            near_optimality
        ),
        "per_trace_seed_point_ratios_vs_initial_best": (
            per_seed
        ),
        "aggregate_drift_exploratory": (
            aggregate_exploratory
        ),
        "important_limitations": [
            "BDG2 uses hourly building-meter data and therefore represents a different domain from traffic sensing.",
            "Only cleaned meter files are used; missing values are not interpolated.",
            "Each selected building contributes exactly one meter and one chunk.",
            "The experiment preserves original float32 values without normalization or log transformation.",
            "Queries execute in warm_slicedmeta mode; this is not a cold-SSD claim.",
            "BDG2 is a cross-domain validation set, not a new controller-development dataset.",
        ],
        "next_gate": (
            "If the primary point-drift endpoints pass, freeze E5 and "
            "move to final figures, tables, and manuscript writing. "
            "Do not tune the controller on BDG2."
        ),
    }


def clean_previous_benchmark_outputs(
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    preserve = {
        "bdg2_eligible_series_catalog.csv",
        "bdg2_selected_series.csv",
    }
    for path in output_dir.iterdir():
        if path.name in preserve:
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def main() -> int:
    e1v2, e1, base = load_modules()

    e1v2.decide_repressdb_two_timescale.e1 = e1
    e1.decide_repressdb = (
        e1v2.decide_repressdb_two_timescale
    )

    print("=" * 100)
    print(
        f"RePressDB E5 BDG2 frozen cross-domain validation v{VERSION}"
    )
    print(f"Target: {TARGET_VENUE}")
    print(f"BDG2 root: {BDG2_ROOT}")
    print(
        "Frozen E1 v0.2 controller; no BDG2-specific tuning."
    )
    print("=" * 100)

    output_dir = Path(
        "results_e5_bdg2_quick_smoke"
        if QUICK_TEST
        else OUTPUT_DIR_NAME
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if QUICK_TEST:
        # Quick mode verifies the E5 execution path without scanning BDG2.
        matrix, synthetic_meta = base.synthetic_dataset(
            SEED, 6
        )
        chunks = base.extract_chunks(
            matrix,
            [2048],
            3,
            SEED,
        )
        metadata_by_chunk = {
            chunk.chunk_id: {
                "building_id": f"synthetic_{index}",
                "meter_type": "synthetic",
                "site_id": "synthetic",
                "primary_use": "synthetic",
                "timezone": "synthetic",
                "source_file": "synthetic",
                "source_column": f"s{index}",
                "longest_run_start": 0,
                "longest_run_length": len(
                    chunk.values
                ),
                "selected_chunk_start": 0,
                "chunk_size": int(
                    chunk.chunk_size
                ),
            }
            for index, chunk in enumerate(chunks)
        }
        trace_seeds = (SEED,)
        dataset_base_meta = synthetic_meta
        clean_previous_benchmark_outputs(
            output_dir
        )
    else:
        evaluated_df = (
            build_or_reuse_eligible_catalog(
                output_dir
            )
        )
        selected_df = build_or_reuse_selection(
            output_dir,
            evaluated_df,
        )
        clean_previous_benchmark_outputs(
            output_dir
        )
        # Restore the preserved manifests after cleanup.
        evaluated_df.to_csv(
            output_dir
            / "bdg2_eligible_series_catalog.csv",
            index=False,
        )
        selected_df.to_csv(
            output_dir
            / "bdg2_selected_series.csv",
            index=False,
        )

        chunks, metadata_by_chunk = (
            load_selected_chunks(
                base,
                selected_df,
            )
        )
        trace_seeds = TRACE_SEEDS
        dataset_base_meta = {
            "source": str(BDG2_ROOT),
            "cleaned_meter_directory": str(
                BDG2_CLEANED_DIR
            ),
            "metadata_directory": str(
                BDG2_METADATA_DIR
            ),
        }

    storage_dir = output_dir / "_storage"
    migration_dir = (
        output_dir / "_migration_trials"
    )
    storage_dir.mkdir(
        parents=True, exist_ok=True
    )
    migration_dir.mkdir(
        parents=True, exist_ok=True
    )

    reps = base.representation_registry(3)

    print(
        f"[BDG2_E5] chunks={len(chunks)}, "
        f"trace_seeds={len(trace_seeds)}, "
        f"traces={len(e1.TRACE_SPECS)}, "
        f"policies={len(e1.POLICIES)}"
    )

    calibration_rows: List[
        Dict[str, Any]
    ] = []
    codec_rows: List[Dict[str, Any]] = []
    static_rows: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []
    timeline_rows: List[Dict[str, Any]] = []
    phase_rows: List[Dict[str, Any]] = []

    for chunk_index, chunk in enumerate(
        chunks,
        start=1,
    ):
        metadata = metadata_by_chunk[
            chunk.chunk_id
        ]
        chunk_dir = storage_dir / chunk.chunk_id

        stored_map, encode_ns, decode_ns = (
            e1.encode_candidates(
                base,
                reps,
                chunk,
                chunk_dir,
            )
        )

        for codec, stored in stored_map.items():
            codec_rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "building_id": metadata[
                        "building_id"
                    ],
                    "meter_type": metadata[
                        "meter_type"
                    ],
                    "site_id": metadata["site_id"],
                    "primary_use": metadata[
                        "primary_use"
                    ],
                    "chunk_size": int(
                        chunk.chunk_size
                    ),
                    "codec": codec,
                    "raw_size_bytes": int(
                        chunk.values.nbytes
                    ),
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
                    "encode_ns": int(
                        encode_ns[codec]
                    ),
                    "full_decode_ns": int(
                        decode_ns[codec]
                    ),
                }
            )

        queries = base.build_queries(
            chunk.values,
            chunk.chunk_id,
            SEED,
            1
            if QUICK_TEST
            else e1.WINDOWS_PER_FRACTION,
        )
        query_pool = {
            query_class: [
                query
                for query in queries
                if query.query_class
                == query_class
            ]
            for query_class in e1.QUERY_CLASSES
        }
        if any(
            not values
            for values in query_pool.values()
        ):
            raise AssertionError(
                f"Incomplete query pool: "
                f"{chunk.chunk_id}"
            )

        profile, local_calibration = (
            e1.calibrate_chunk(
                base,
                reps,
                stored_map,
                chunk,
                query_pool,
            )
        )

        for row in local_calibration:
            row.update(
                {
                    "building_id": metadata[
                        "building_id"
                    ],
                    "meter_type": metadata[
                        "meter_type"
                    ],
                    "site_id": metadata["site_id"],
                    "primary_use": metadata[
                        "primary_use"
                    ],
                }
            )
        calibration_rows.extend(
            local_calibration
        )

        static_rows.extend(
            static_layout_records(
                e1,
                base,
                chunk,
                profile,
                metadata,
            )
        )

        for trace_seed in trace_seeds:
            for trace_name in e1.TRACE_SPECS:
                trace = e1v2.build_trace(
                    e1,
                    base,
                    chunk,
                    trace_name,
                    query_pool,
                    int(trace_seed),
                )

                policies = list(e1.POLICIES)
                random.Random(
                    stable_seed(
                        trace_seed,
                        chunk.chunk_id,
                        trace_name,
                        "policy_order",
                    )
                ).shuffle(policies)

                for policy in policies:
                    (
                        summary,
                        decisions,
                        timeline,
                        phases,
                    ) = e1.run_policy(
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
                        migration_dir
                        / chunk.chunk_id,
                    )

                    common = {
                        "trace_seed": int(
                            trace_seed
                        ),
                        "building_id": metadata[
                            "building_id"
                        ],
                        "meter_type": metadata[
                            "meter_type"
                        ],
                        "site_id": metadata[
                            "site_id"
                        ],
                        "primary_use": metadata[
                            "primary_use"
                        ],
                    }
                    summary.update(common)
                    for row in decisions:
                        row.update(common)
                    for row in timeline:
                        row.update(common)
                    for row in phases:
                        row.update(common)

                    run_rows.append(summary)
                    decision_rows.extend(
                        decisions
                    )
                    phase_rows.extend(phases)
                    if policy in {
                        "repressdb",
                        "periodic_greedy",
                        "oracle",
                    }:
                        timeline_rows.extend(
                            timeline
                        )

        if (
            chunk_index == 1
            or chunk_index == len(chunks)
            or chunk_index
            % max(1, len(chunks) // 12)
            == 0
        ):
            print(
                f"[BDG2_E5] "
                f"{chunk_index}/{len(chunks)} "
                f"chunks complete"
            )

    calibration_df = pd.DataFrame(
        calibration_rows
    )
    codec_df = pd.DataFrame(codec_rows)
    static_df = pd.DataFrame(static_rows)
    run_df = pd.DataFrame(run_rows)
    decisions_df = pd.DataFrame(
        decision_rows
    )
    timeline_df = pd.DataFrame(
        timeline_rows
    )
    phase_df = pd.DataFrame(phase_rows)

    decisions_df = e1v2.add_switch_diagnostics(
        e1,
        base,
        decisions_df,
        calibration_df,
    )
    aggregate_df = e1.aggregate_results(
        run_df
    )

    for reference_policy, column_name in (
        ("repressdb", "ratio_to_repressdb"),
        ("oracle", "ratio_to_oracle"),
    ):
        reference = aggregate_df[
            aggregate_df["policy"]
            == reference_policy
        ][
            [
                "trace",
                "chunk_size",
                "median_total_time_ns",
            ]
        ].rename(
            columns={
                "median_total_time_ns": (
                    "reference_time_ns"
                )
            }
        )
        aggregate_df = aggregate_df.merge(
            reference,
            on=["trace", "chunk_size"],
            how="left",
        )
        aggregate_df[column_name] = (
            aggregate_df[
                "median_total_time_ns"
            ]
            / aggregate_df[
                "reference_time_ns"
            ]
        )
        aggregate_df = aggregate_df.drop(
            columns=["reference_time_ns"]
        )

    comparisons_df = (
        building_clustered_statistics(
            run_df
        )
    )
    static_summary_df = (
        static_layout_summary(static_df)
    )
    near_optimality = (
        switch_near_optimality(
            e1,
            base,
            decisions_df,
            calibration_df,
        )
    )

    dataset_meta = dict(dataset_base_meta)
    dataset_meta.update(
        {
            "dataset_label": (
                "BDG2_E5_FROZEN_CROSSDOMAIN"
                if not QUICK_TEST
                else "SYNTHETIC_E5_SMOKE"
            ),
            "bdg2_root": str(BDG2_ROOT),
            "data_version_role": (
                "repository cleaned meter data"
            ),
            "selection_seed": SEED,
            "num_selected_buildings": int(
                len(
                    {
                        metadata[
                            "building_id"
                        ]
                        for metadata in (
                            metadata_by_chunk.values()
                        )
                    }
                )
            ),
            "num_chunks": int(len(chunks)),
            "chunk_sizes": [
                int(size)
                for size in sorted(
                    {
                        chunk.chunk_size
                        for chunk in chunks
                    }
                )
            ],
            "chunks_per_size": {
                str(size): int(
                    sum(
                        chunk.chunk_size == size
                        for chunk in chunks
                    )
                )
                for size in CHUNK_SIZES
                if any(
                    chunk.chunk_size == size
                    for chunk in chunks
                )
            },
            "trace_seeds": [
                int(seed)
                for seed in trace_seeds
            ],
            "trace_names": list(
                e1.TRACE_SPECS
            ),
            "policies": list(e1.POLICIES),
            "meter_type_counts": dict(
                Counter(
                    metadata[
                        "meter_type"
                    ]
                    for metadata in (
                        metadata_by_chunk.values()
                    )
                )
            ),
            "site_counts": dict(
                Counter(
                    metadata["site_id"]
                    for metadata in (
                        metadata_by_chunk.values()
                    )
                )
            ),
            "selection_rules": {
                "only_cleaned_meter_files": True,
                "minimum_contiguous_finite_length": (
                    MIN_CONTIGUOUS_LENGTH
                ),
                "minimum_distinct_values": (
                    MIN_DISTINCT_VALUES
                ),
                "minimum_nonzero_count": (
                    MIN_NONZERO_COUNT
                ),
                "variance_epsilon": (
                    VARIANCE_EPSILON
                ),
                "one_meter_per_building": True,
                "normalization": False,
                "log_transform": False,
                "interpolation": False,
                "resampling": False,
            },
            "independent_statistical_unit": (
                "building_id"
            ),
            "primary_execution_mode": (
                "warm_slicedmeta"
            ),
            "target_venue": TARGET_VENUE,
        }
    )

    assessment = make_assessment(
        e1,
        run_df,
        aggregate_df,
        comparisons_df,
        decisions_df,
        static_df,
        static_summary_df,
        near_optimality,
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
    static_df.to_csv(
        output_dir
        / "static_workload_layout_records.csv",
        index=False,
    )
    static_summary_df.to_csv(
        output_dir
        / "static_workload_layout_summary.csv",
        index=False,
    )
    run_df.to_csv(
        output_dir / "policy_run_records.csv",
        index=False,
    )
    aggregate_df.to_csv(
        output_dir
        / "policy_aggregate_summary.csv",
        index=False,
    )
    comparisons_df.to_csv(
        output_dir
        / "policy_paired_comparisons_building_clustered.csv",
        index=False,
    )
    decisions_df.to_csv(
        output_dir / "decision_records.csv",
        index=False,
    )
    timeline_df.to_csv(
        output_dir / "timeline_sample.csv",
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
        output_dir / "e5_assessment.json",
        assessment,
    )

    config = {
        "version": VERSION,
        "target_venue": TARGET_VENUE,
        "controller_version": "E1 v0.2.0 frozen",
        "base_protocol": e1.BASE_PROTOCOL,
        "seed": SEED,
        "trace_seeds": list(
            trace_seeds
        ),
        "chunk_sizes": [
            int(size)
            for size in CHUNK_SIZES
        ],
        "chunks_per_size": (
            CHUNKS_PER_SIZE
            if not QUICK_TEST
            else None
        ),
        "windows_per_fraction": (
            1
            if QUICK_TEST
            else e1.WINDOWS_PER_FRACTION
        ),
        "calibration_queries_per_class": (
            1
            if QUICK_TEST
            else e1.CALIBRATION_QUERIES_PER_CLASS
        ),
        "calibration_repeats": (
            1
            if QUICK_TEST
            else e1.CALIBRATION_REPEATS
        ),
        "decision_interval": (
            20
            if QUICK_TEST
            else e1.DECISION_INTERVAL
        ),
        "long_observation_window": (
            e1.OBSERVATION_WINDOW
        ),
        "short_observation_window": (
            e1v2.SHORT_OBSERVATION_WINDOW
        ),
        "min_observations": (
            30
            if QUICK_TEST
            else e1.MIN_OBSERVATIONS
        ),
        "confirmation_rounds": (
            e1.CONFIRMATION_ROUNDS
        ),
        "cooldown_queries": (
            e1.COOLDOWN_QUERIES
        ),
        "prediction_horizon": (
            e1.PREDICTION_HORIZON
        ),
        "min_relative_improvement": (
            e1.MIN_RELATIVE_IMPROVEMENT
        ),
        "rewrite_safety_factor": (
            e1.REWRITE_SAFETY_FACTOR
        ),
        "trace_specs": e1.TRACE_SPECS,
        "policies": e1.POLICIES,
        "representations": list(reps),
        "workloads": base.WORKLOADS,
        "controller_rule": (
            "Long-window and short-window candidate codecs must agree. "
            "Relative improvement and predicted saving use the "
            "conservative minimum across both windows."
        ),
        "dataset_processing": {
            "bdg2_root": str(BDG2_ROOT),
            "cleaned_meter_directory": str(
                BDG2_CLEANED_DIR
            ),
            "normalization": False,
            "log_transform": False,
            "interpolation": False,
            "resampling": False,
            "dtype": "float32",
        },
        "statistical_unit": "building_id",
        "study_role": (
            "Frozen-controller cross-domain validation; "
            "no BDG2-specific retuning."
        ),
        "measurement_scope": (
            "Time-ordered file-backed replay in warm_slicedmeta mode. "
            "Each migration measures full decode plus target encode and fsync."
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "scipy": __import__(
                "scipy"
            ).__version__,
            "timer": "time.perf_counter_ns",
        },
    }
    write_json(
        output_dir / "run_config.json",
        config,
    )
    write_json(
        Path(
            "E5_QUICK_ASSESSMENT.json"
            if QUICK_TEST
            else SUMMARY_JSON_NAME
        ),
        assessment,
    )

    shutil.rmtree(
        storage_dir,
        ignore_errors=True,
    )
    shutil.rmtree(
        migration_dir,
        ignore_errors=True,
    )

    print("-" * 100)
    print(
        "Validation gates:",
        assessment["validation_gates"],
    )
    print(
        "Primary scan-to-point:",
        assessment["primary_scan_to_point"],
    )
    print(
        f"Results: {output_dir.resolve()}"
    )
    print(
        "Done. Upload e5_assessment.json, "
        "dataset_summary.json, bdg2_selected_series.csv, "
        "static_workload_layout_summary.csv, "
        "policy_paired_comparisons_building_clustered.csv, "
        "policy_aggregate_summary.csv, policy_run_records.csv, "
        "decision_records.csv, phase_records.csv, and run_config.json."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
