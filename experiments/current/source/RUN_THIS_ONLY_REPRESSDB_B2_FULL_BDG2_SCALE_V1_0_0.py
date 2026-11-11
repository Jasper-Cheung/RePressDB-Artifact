#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
RePressDB B2 v1.0.0
Full-BDG2 single-active storage and scale characterization.

Target journal: Journal of Big Data.

DIRECT RUN
----------
Place this file beside:

  RUN_THIS_ONLY_REPRESSDB_B1_SINGLE_ACTIVE_STORE_V0_9_0.py
  RUN_THIS_ONLY_REPRESSDB_E5_BDG2_FROZEN_DIRECT_V0_6_0.py
  RUN_THIS_ONLY_REPRESSDB_E0C_V0_2_0.py

Open this file in PyCharm and click Run.

No command-line arguments are required.

B2 PURPOSE
----------
B1 established correctness, physical migration, and crash recovery on 32
PEMS04 chunks. B2 evaluates the same single-active storage substrate on all
eligible cleaned BDG2 meters.

Two non-overlapping chunk views are constructed from every eligible meter:

  data_scale_8192:
      chunk_size = 8192
      intended to characterize data volume, storage, and query throughput.

  catalog_scale_2048:
      chunk_size = 2048
      intended to characterize larger chunk counts and catalog overhead.

Only full, non-overlapping chunks from the longest contiguous finite hourly
run are used. No interpolation, normalization, resampling, or synthetic
replication is performed.

PRODUCTION / TELEMETRY SEPARATION
---------------------------------
Each view has its own production-only SQLite-WAL catalog:

  stores/<view>/catalog.sqlite3

All benchmark progress and records are stored separately:

  telemetry.sqlite3

Catalog bytes/chunk therefore excludes query, migration, and experiment logs.

FORMAL B2 WORKLOAD
------------------
Across the two views:

  * all eligible BDG2 meter series;
  * all complete non-overlapping chunks;
  * 500 deterministic physical migrations per view;
  * 500,000 real file-backed queries per view;
  * 1,000 total migrations;
  * 1,000,000 total queries.

Every query is checked against a separate float32 truth memmap that is
excluded from production-store size measurements.

RESUME
------
The following stages are resumable:

  * full BDG2 qualification scan;
  * truth-store construction;
  * production chunk ingestion;
  * deterministic migrations;
  * query benchmark.

Rerun the same file after interruption. Do not delete:

  results_b2_full_bdg2_scale_v100/

Do not modify this script after the run begins. The script hash is included
in the experiment freeze.
"""

from __future__ import annotations

import collections
import contextlib
import csv
import gzip
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


VERSION = "1.0.0"
TARGET_JOURNAL = "Journal of Big Data"

B1_SCRIPT_NAME = "RUN_THIS_ONLY_REPRESSDB_B1_SINGLE_ACTIVE_STORE_V0_9_0.py"
E5_SCRIPT_NAME = "RUN_THIS_ONLY_REPRESSDB_E5_BDG2_FROZEN_DIRECT_V0_6_0.py"
E0C_SCRIPT_NAME = "RUN_THIS_ONLY_REPRESSDB_E0C_V0_2_0.py"

B1_RESULT_DIR_NAME = "results_b1_single_active_store_v090"
OUTPUT_DIR_NAME = "results_b2_full_bdg2_scale_v100"
SUMMARY_JSON_NAME = "B2_FULL_BDG2_SCALE_ASSESSMENT_v100.json"

SEED = 20260725
ZSTD_LEVEL = 3

VIEW_CONFIGS: Dict[str, Dict[str, Any]] = {
    "data_scale_8192": {
        "chunk_size": 8192,
        "query_count": 500_000,
        "migration_count": 500,
    },
    "catalog_scale_2048": {
        "chunk_size": 2048,
        "query_count": 500_000,
        "migration_count": 500,
    },
}

CODECS = (
    "whole_lz4",
    "whole_zstd",
    "block_lz4_64",
    "block_lz4_256",
    "block_lz4_1024",
)
INITIAL_CODEC = "whole_lz4"
MAX_OPEN_HANDLES = 256

QUALIFICATION_BATCH_SIZE = 192
INGEST_COLUMN_BATCH_SIZE = 64
QUERY_INSERT_BATCH = 1000
QUERY_PROGRESS_INTERVAL = 10_000
MIGRATION_PROGRESS_INTERVAL = 25
STARTUP_REPEATS = 5

QUERY_TYPES = (
    "point",
    "range_scan",
    "range_sum",
    "range_avg",
    "range_min",
    "range_max",
    "threshold_count",
)

MIN_CONTIGUOUS_LENGTH = max(
    int(config["chunk_size"])
    for config in VIEW_CONFIGS.values()
)
MIN_DISTINCT_VALUES = 16
MIN_NONZERO_COUNT = 32
VARIANCE_EPSILON = 1e-12


def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json_atomic(path: Path, value: Any) -> None:
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


def write_dataframe_atomic(path: Path, frame: pd.DataFrame) -> None:
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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def raw_array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float32)
    return sha256_bytes(array.tobytes(order="C"))


def stable_seed(*parts: Any) -> int:
    token = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def stable_order_key(*parts: Any) -> str:
    token = "|".join(str(part) for part in parts)
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def load_module(path: Path, name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing dependency: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def sqlite_file_bytes(path: Path) -> Dict[str, int]:
    paths = {
        "database_bytes": path,
        "wal_bytes": Path(str(path) + "-wal"),
        "shm_bytes": Path(str(path) + "-shm"),
    }
    result: Dict[str, int] = {}
    for key, candidate in paths.items():
        result[key] = (
            int(candidate.stat().st_size)
            if candidate.exists()
            else 0
        )
    result["total_catalog_files_bytes"] = int(
        sum(result.values())
    )
    return result


class MemoryTracker:
    def __init__(self) -> None:
        self.method = "unavailable"
        self.peak_rss_bytes = 0
        self.current_rss_bytes = 0
        self._process: Any = None
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
            self.method = "psutil_rss"
        except Exception:
            self._process = None
            if os.name != "nt":
                try:
                    import resource

                    self._resource = resource
                    self.method = "resource_ru_maxrss"
                except Exception:
                    self._resource = None
            else:
                self._resource = None

    def sample(self) -> int:
        value = 0
        if self._process is not None:
            value = int(self._process.memory_info().rss)
        elif getattr(self, "_resource", None) is not None:
            raw = int(
                self._resource.getrusage(
                    self._resource.RUSAGE_SELF
                ).ru_maxrss
            )
            value = (
                raw
                if platform.system() == "Darwin"
                else raw * 1024
            )
        self.current_rss_bytes = value
        self.peak_rss_bytes = max(
            self.peak_rss_bytes,
            value,
        )
        return value


def build_production_store_class(b1: Any) -> Any:
    class ProductionChunkStore(b1.SingleActiveChunkStore):
        """Production-only B1 store with an LRU file-handle cache."""

        def __init__(
            self,
            root: Path,
            base: Any,
            create: bool = True,
            recover_on_open: bool = True,
            max_open_handles: int = MAX_OPEN_HANDLES,
        ):
            self._max_open_handles = int(max_open_handles)
            super().__init__(
                root,
                base,
                create=create,
                recover_on_open=recover_on_open,
            )
            self._handle_cache = collections.OrderedDict(
                self._handle_cache
            )

        def _create_schema(self) -> None:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS store_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    series_index INTEGER NOT NULL,
                    n INTEGER NOT NULL,
                    raw_checksum TEXT NOT NULL,

                    active_version INTEGER NOT NULL,
                    active_codec TEXT NOT NULL,
                    active_data_path TEXT NOT NULL,
                    active_metadata_path TEXT,
                    active_payload_bytes INTEGER NOT NULL,
                    active_metadata_bytes INTEGER NOT NULL,

                    previous_version INTEGER,
                    previous_codec TEXT,
                    previous_data_path TEXT,
                    previous_metadata_path TEXT,
                    previous_payload_bytes INTEGER,
                    previous_metadata_bytes INTEGER,

                    migration_state TEXT NOT NULL DEFAULT 'idle',
                    created_at_ns INTEGER NOT NULL,
                    updated_at_ns INTEGER NOT NULL,

                    view_name TEXT,
                    series_key TEXT,
                    building_id TEXT,
                    meter_type TEXT,
                    site_id TEXT,
                    source_file TEXT,
                    source_column TEXT,
                    chunk_start INTEGER,
                    chunk_index_within_series INTEGER
                );

                CREATE TABLE IF NOT EXISTS migration_log (
                    migration_id TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    from_version INTEGER NOT NULL,
                    to_version INTEGER NOT NULL,
                    from_codec TEXT NOT NULL,
                    target_codec TEXT NOT NULL,
                    state TEXT NOT NULL,
                    temporary_data_path TEXT,
                    temporary_metadata_path TEXT,
                    final_data_path TEXT,
                    final_metadata_path TEXT,
                    target_payload_bytes INTEGER,
                    target_metadata_bytes INTEGER,
                    started_at_ns INTEGER NOT NULL,
                    updated_at_ns INTEGER NOT NULL,
                    completed_at_ns INTEGER,
                    recovery_action TEXT,
                    error_text TEXT,
                    FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_series_key
                ON chunks(series_key);

                CREATE INDEX IF NOT EXISTS idx_chunks_building
                ON chunks(building_id);

                CREATE INDEX IF NOT EXISTS idx_migration_chunk
                ON migration_log(chunk_id);
                """
            )

        def _open_cached_handles(
            self,
            chunk_id: str,
            row: sqlite3.Row,
            stored: Any,
        ) -> Any:
            active_version = int(row["active_version"])
            cached = self._handle_cache.get(chunk_id)
            if (
                cached is not None
                and cached.active_version == active_version
            ):
                self._handle_cache.move_to_end(chunk_id)
                return cached

            self.invalidate_handle(chunk_id)
            payload = open(stored.data_path, "rb")
            metadata = (
                open(stored.metadata_path, "rb")
                if stored.metadata_path is not None
                else None
            )
            cached = b1.CachedHandles(
                active_version,
                payload,
                metadata,
            )
            self._handle_cache[chunk_id] = cached
            self._handle_cache.move_to_end(chunk_id)

            while len(self._handle_cache) > self._max_open_handles:
                _, evicted = self._handle_cache.popitem(last=False)
                for handle in (evicted.payload, evicted.metadata):
                    if handle is not None:
                        try:
                            handle.close()
                        except Exception:
                            pass
            return cached

        def create_bdg2_chunk(
            self,
            row: Mapping[str, Any],
            values: np.ndarray,
        ) -> None:
            chunk_id = str(row["chunk_id"])
            self.create_chunk(
                chunk_id=chunk_id,
                series_index=int(row["series_ordinal"]),
                values=values,
                codec=INITIAL_CODEC,
            )
            self.connection.execute(
                """
                UPDATE chunks
                SET
                    view_name=?,
                    series_key=?,
                    building_id=?,
                    meter_type=?,
                    site_id=?,
                    source_file=?,
                    source_column=?,
                    chunk_start=?,
                    chunk_index_within_series=?
                WHERE chunk_id=?
                """,
                (
                    str(row["view_name"]),
                    str(row["series_key"]),
                    str(row["building_id"]),
                    str(row["meter_type"]),
                    str(row["site_id"]),
                    str(row["source_file"]),
                    str(row["source_column"]),
                    int(row["chunk_start"]),
                    int(row["chunk_index_within_series"]),
                    chunk_id,
                ),
            )

    return ProductionChunkStore


def open_production_store(
    store_class: Any,
    root: Path,
    base: Any,
    create: bool = True,
    recover_on_open: bool = True,
) -> Any:
    return store_class(
        root=root,
        base=base,
        create=create,
        recover_on_open=recover_on_open,
        max_open_handles=MAX_OPEN_HANDLES,
    )

def telemetry_connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        path,
        timeout=60.0,
        isolation_level=None,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA busy_timeout=60000")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS experiment_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS ingest_records (
            view_name TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            truth_index INTEGER NOT NULL,
            raw_checksum TEXT NOT NULL,
            stored_bytes INTEGER NOT NULL,
            elapsed_ns INTEGER NOT NULL,
            completed_at_ns INTEGER NOT NULL,
            PRIMARY KEY(view_name, chunk_id)
        );

        CREATE TABLE IF NOT EXISTS migration_plans (
            view_name TEXT NOT NULL,
            migration_index INTEGER NOT NULL,
            chunk_id TEXT NOT NULL,
            before_codec TEXT NOT NULL,
            before_version INTEGER NOT NULL,
            target_codec TEXT NOT NULL,
            state TEXT NOT NULL,
            migration_time_ns INTEGER,
            rewrite_bytes INTEGER,
            after_codec TEXT,
            after_version INTEGER,
            checksum_ok INTEGER,
            recovered_completion INTEGER NOT NULL DEFAULT 0,
            completed_at_ns INTEGER,
            PRIMARY KEY(view_name, migration_index)
        );

        CREATE TABLE IF NOT EXISTS query_records (
            view_name TEXT NOT NULL,
            query_index INTEGER NOT NULL,
            chunk_id TEXT NOT NULL,
            truth_index INTEGER NOT NULL,
            active_codec TEXT NOT NULL,
            active_version INTEGER NOT NULL,
            query_type TEXT NOT NULL,
            query_class TEXT NOT NULL,
            start_index INTEGER NOT NULL,
            end_index INTEGER NOT NULL,
            threshold REAL,
            latency_ns INTEGER NOT NULL,
            logical_bytes INTEGER NOT NULL,
            result_ok INTEGER NOT NULL,
            PRIMARY KEY(view_name, query_index)
        );

        CREATE TABLE IF NOT EXISTS stage_timings (
            view_name TEXT NOT NULL,
            stage_name TEXT NOT NULL,
            elapsed_ns INTEGER NOT NULL,
            record_count INTEGER,
            byte_count INTEGER,
            recorded_at_ns INTEGER NOT NULL,
            PRIMARY KEY(view_name, stage_name)
        );

        CREATE TABLE IF NOT EXISTS startup_records (
            view_name TEXT NOT NULL,
            repeat_index INTEGER NOT NULL,
            startup_ns INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL,
            PRIMARY KEY(view_name, repeat_index)
        );
        """
    )
    return connection


@contextlib.contextmanager
def telemetry_transaction(
    connection: sqlite3.Connection,
) -> Iterator[sqlite3.Connection]:
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield connection
    except Exception:
        connection.execute("ROLLBACK")
        raise
    else:
        connection.execute("COMMIT")


def telemetry_set(
    connection: sqlite3.Connection,
    key: str,
    value: Any,
) -> None:
    serialized = json.dumps(
        json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
    )
    connection.execute(
        """
        INSERT INTO experiment_metadata(key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (key, serialized),
    )


def telemetry_get(
    connection: sqlite3.Connection,
    key: str,
) -> Optional[Any]:
    row = connection.execute(
        "SELECT value FROM experiment_metadata WHERE key=?",
        (key,),
    ).fetchone()
    if row is None:
        return None
    return json.loads(row["value"])


def build_full_qualification(
    output_dir: Path,
    e5: Any,
    memory: MemoryTracker,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    candidate_path = output_dir / "bdg2_full_candidate_catalog.csv"
    qualification_path = (
        output_dir / "bdg2_full_qualification_catalog.csv"
    )

    if candidate_path.exists():
        candidates = pd.read_csv(candidate_path)
    else:
        meter_files = e5.discover_meter_files()
        candidates = e5.build_candidate_catalog(meter_files)
        candidates["series_key"] = [
            stable_order_key(
                "B2",
                row.source_file,
                row.source_column,
                row.meter_type,
            )
            for row in candidates.itertuples(index=False)
        ]
        write_dataframe_atomic(candidate_path, candidates)

    if "series_key" not in candidates.columns:
        candidates["series_key"] = [
            stable_order_key(
                "B2",
                row.source_file,
                row.source_column,
                row.meter_type,
            )
            for row in candidates.itertuples(index=False)
        ]
        write_dataframe_atomic(candidate_path, candidates)

    if qualification_path.exists():
        qualified = pd.read_csv(qualification_path)
    else:
        qualified = pd.DataFrame()

    completed_keys = (
        set(qualified["series_key"].astype(str))
        if not qualified.empty
        and "series_key" in qualified.columns
        else set()
    )

    remaining = candidates[
        ~candidates["series_key"].astype(str).isin(
            completed_keys
        )
    ].reset_index(drop=True)

    total = len(candidates)
    completed = len(completed_keys)
    for start in range(0, len(remaining), QUALIFICATION_BATCH_SIZE):
        batch = remaining.iloc[
            start : start + QUALIFICATION_BATCH_SIZE
        ].copy()
        rows = e5.evaluate_candidate_batch(batch)
        local = pd.DataFrame(rows)
        if not local.empty and "series_key" not in local.columns:
            local = local.merge(
                batch[
                    [
                        "source_file",
                        "source_column",
                        "series_key",
                    ]
                ],
                on=["source_file", "source_column"],
                how="left",
            )

        qualified = pd.concat(
            [qualified, local],
            ignore_index=True,
        )
        qualified = qualified.drop_duplicates(
            "series_key",
            keep="last",
        ).sort_values("series_key").reset_index(drop=True)
        write_dataframe_atomic(
            qualification_path,
            qualified,
        )
        completed += len(batch)
        eligible_count = int(
            parse_bool_series(
                qualified.get(
                    "eligible",
                    pd.Series(dtype=bool),
                )
            ).sum()
        )
        memory.sample()
        print(
            f"[full BDG2 qualification] "
            f"{completed}/{total}, eligible={eligible_count}"
        )

    if len(qualified) != len(candidates):
        raise AssertionError(
            "Full qualification did not cover every candidate series"
        )

    eligible_mask = parse_bool_series(qualified["eligible"])
    eligible = qualified[eligible_mask].copy()
    eligible = eligible[
        pd.to_numeric(
            eligible["longest_run_length"],
            errors="coerce",
        )
        >= MIN_CONTIGUOUS_LENGTH
    ].copy()

    if eligible.empty:
        raise RuntimeError("No eligible full-BDG2 meter series")
    eligible["series_ordinal"] = np.arange(
        len(eligible),
        dtype=np.int64,
    )
    eligible = eligible.sort_values(
        ["series_key"]
    ).reset_index(drop=True)
    eligible["series_ordinal"] = np.arange(
        len(eligible),
        dtype=np.int64,
    )
    write_dataframe_atomic(
        output_dir / "bdg2_full_eligible_series.csv",
        eligible,
    )
    return candidates, eligible


def build_manifests(
    output_dir: Path,
    eligible: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    manifests: Dict[str, pd.DataFrame] = {}

    for view_name, config in VIEW_CONFIGS.items():
        path = output_dir / f"{view_name}_chunk_manifest.csv"
        if path.exists():
            manifest = pd.read_csv(path)
            manifests[view_name] = manifest
            continue

        chunk_size = int(config["chunk_size"])
        rows: List[Dict[str, Any]] = []
        truth_index = 0

        for series in eligible.itertuples(index=False):
            run_start = int(series.longest_run_start)
            run_length = int(series.longest_run_length)
            n_chunks = int(run_length // chunk_size)
            for chunk_index in range(n_chunks):
                chunk_start = (
                    run_start + chunk_index * chunk_size
                )
                chunk_id = (
                    f"BDG2_B2_{chunk_size}_"
                    f"{str(series.series_key)[:16]}_"
                    f"{chunk_index:04d}"
                )
                rows.append(
                    {
                        "view_name": view_name,
                        "chunk_id": chunk_id,
                        "truth_index": truth_index,
                        "chunk_size": chunk_size,
                        "chunk_start": chunk_start,
                        "chunk_index_within_series": (
                            chunk_index
                        ),
                        "series_key": str(series.series_key),
                        "series_ordinal": int(
                            series.series_ordinal
                        ),
                        "building_id": str(
                            series.building_id
                        ),
                        "meter_type": str(
                            series.meter_type
                        ),
                        "site_id": str(series.site_id),
                        "primary_use": str(
                            getattr(
                                series,
                                "primary_use",
                                "unknown",
                            )
                        ),
                        "source_file": str(
                            series.source_file
                        ),
                        "source_column": str(
                            series.source_column
                        ),
                        "time_column": str(
                            getattr(
                                series,
                                "time_column",
                                "",
                            )
                        ),
                        "raw_bytes": chunk_size * 4,
                    }
                )
                truth_index += 1

        manifest = pd.DataFrame(rows)
        if manifest.empty:
            raise RuntimeError(
                f"No chunks generated for {view_name}"
            )
        if manifest["chunk_id"].duplicated().any():
            raise AssertionError(
                f"Duplicate chunk ids in {view_name}"
            )
        write_dataframe_atomic(path, manifest)
        manifests[view_name] = manifest

    return manifests


def ensure_truth_memmap(
    path: Path,
    n_chunks: int,
    chunk_size: int,
) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = int(n_chunks * chunk_size * 4)
    if not path.exists():
        with open(path, "wb") as file:
            file.truncate(expected_bytes)
            file.flush()
            os.fsync(file.fileno())
    elif int(path.stat().st_size) != expected_bytes:
        raise RuntimeError(
            f"Truth file size mismatch: {path}"
        )
    return np.memmap(
        path,
        mode="r+",
        dtype=np.float32,
        shape=(n_chunks, chunk_size),
    )


def record_stage_timing(
    telemetry: sqlite3.Connection,
    view_name: str,
    stage_name: str,
    elapsed_ns: int,
    record_count: Optional[int] = None,
    byte_count: Optional[int] = None,
) -> None:
    telemetry.execute(
        """
        INSERT INTO stage_timings(
            view_name, stage_name, elapsed_ns,
            record_count, byte_count, recorded_at_ns
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(view_name, stage_name)
        DO UPDATE SET
            elapsed_ns=excluded.elapsed_ns,
            record_count=excluded.record_count,
            byte_count=excluded.byte_count,
            recorded_at_ns=excluded.recorded_at_ns
        """,
        (
            view_name,
            stage_name,
            int(elapsed_ns),
            record_count,
            byte_count,
            time.time_ns(),
        ),
    )


def ingest_view(
    view_name: str,
    manifest: pd.DataFrame,
    store: Any,
    telemetry: sqlite3.Connection,
    truth_path: Path,
    e5: Any,
    memory: MemoryTracker,
) -> Dict[str, Any]:
    start_ns = time.perf_counter_ns()
    chunk_size = int(
        manifest["chunk_size"].iloc[0]
    )
    truth = ensure_truth_memmap(
        truth_path,
        len(manifest),
        chunk_size,
    )

    existing_store = set(store.list_chunk_ids())
    completed_rows = telemetry.execute(
        """
        SELECT chunk_id
        FROM ingest_records
        WHERE view_name=?
        """,
        (view_name,),
    ).fetchall()
    completed = {
        str(row["chunk_id"])
        for row in completed_rows
    }

    incomplete = manifest[
        ~manifest["chunk_id"].astype(str).isin(
            completed.intersection(existing_store)
        )
    ].copy()

    total_chunks = len(manifest)
    done_before = total_chunks - len(incomplete)

    for source_file, file_group in incomplete.groupby(
        "source_file",
        sort=True,
    ):
        source_path = Path(str(source_file))
        columns = sorted(
            file_group["source_column"]
            .astype(str)
            .unique()
            .tolist()
        )

        for column_start in range(
            0,
            len(columns),
            INGEST_COLUMN_BATCH_SIZE,
        ):
            batch_columns = columns[
                column_start :
                column_start + INGEST_COLUMN_BATCH_SIZE
            ]
            frame = e5.read_selected_columns(
                source_path,
                batch_columns,
            )

            insert_rows: List[Tuple[Any, ...]] = []
            batch_chunk_count = 0
            for column in batch_columns:
                values = pd.to_numeric(
                    frame[column],
                    errors="coerce",
                ).to_numpy(dtype=np.float32)
                column_rows = file_group[
                    file_group["source_column"].astype(str)
                    == str(column)
                ].sort_values(
                    "chunk_index_within_series"
                )

                for row in column_rows.to_dict(
                    orient="records"
                ):
                    chunk_id = str(row["chunk_id"])
                    start = int(row["chunk_start"])
                    end = start + chunk_size
                    chunk_values = np.ascontiguousarray(
                        values[start:end],
                        dtype=np.float32,
                    )
                    if (
                        chunk_values.size != chunk_size
                        or not np.isfinite(
                            chunk_values
                        ).all()
                    ):
                        raise AssertionError(
                            f"Invalid manifest chunk: {chunk_id}"
                        )

                    truth_index = int(row["truth_index"])
                    truth[truth_index, :] = chunk_values
                    checksum = raw_array_sha256(
                        chunk_values
                    )

                    operation_start = (
                        time.perf_counter_ns()
                    )
                    store.create_bdg2_chunk(
                        row,
                        chunk_values,
                    )
                    stored = store.active_stored(chunk_id)
                    elapsed_ns = (
                        time.perf_counter_ns()
                        - operation_start
                    )
                    if (
                        str(
                            store._row(chunk_id)[
                                "raw_checksum"
                            ]
                        )
                        != checksum
                    ):
                        raise AssertionError(
                            f"Catalog checksum mismatch: "
                            f"{chunk_id}"
                        )

                    insert_rows.append(
                        (
                            view_name,
                            chunk_id,
                            truth_index,
                            checksum,
                            int(stored.stored_size_bytes),
                            int(elapsed_ns),
                            time.time_ns(),
                        )
                    )
                    batch_chunk_count += 1

            truth.flush()
            with telemetry_transaction(telemetry):
                telemetry.executemany(
                    """
                    INSERT OR REPLACE INTO ingest_records(
                        view_name, chunk_id, truth_index,
                        raw_checksum, stored_bytes,
                        elapsed_ns, completed_at_ns
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    insert_rows,
                )

            done_before += batch_chunk_count
            memory.sample()
            print(
                f"[{view_name} ingest] "
                f"{done_before}/{total_chunks} chunks"
            )

    truth.flush()
    del truth

    elapsed_ns = time.perf_counter_ns() - start_ns
    ingest_aggregate = telemetry.execute(
        """
        SELECT
            COUNT(*) AS n,
            COALESCE(SUM(elapsed_ns),0) AS operation_ns,
            COALESCE(SUM(stored_bytes),0) AS stored_bytes
        FROM ingest_records
        WHERE view_name=?
        """,
        (view_name,),
    ).fetchone()
    active_bytes = int(
        store.connection.execute(
            """
            SELECT COALESCE(
                SUM(
                    active_payload_bytes
                    + active_metadata_bytes
                ),
                0
            ) AS total_bytes
            FROM chunks
            """
        ).fetchone()["total_bytes"]
    )
    record_stage_timing(
        telemetry,
        view_name,
        "ingestion",
        elapsed_ns,
        record_count=total_chunks,
        byte_count=active_bytes,
    )
    return {
        "wall_elapsed_ns_this_invocation": int(elapsed_ns),
        "chunk_count": int(total_chunks),
        "completed_ingest_records": int(ingest_aggregate["n"]),
        "measured_encode_store_ns": int(ingest_aggregate["operation_ns"]),
        "measured_ingest_throughput_mib_per_s": float(
            (int(ingest_aggregate["stored_bytes"]) / (1024.0 * 1024.0))
            / max(int(ingest_aggregate["operation_ns"]) / 1e9, 1e-12)
        ),
        "active_stored_bytes": active_bytes,
    }


def checkpoint_catalog(
    store: Any,
) -> Dict[str, int]:
    store.connection.execute(
        "PRAGMA wal_checkpoint(TRUNCATE)"
    )
    page_size = int(
        store.connection.execute(
            "PRAGMA page_size"
        ).fetchone()[0]
    )
    page_count = int(
        store.connection.execute(
            "PRAGMA page_count"
        ).fetchone()[0]
    )
    file_sizes = sqlite_file_bytes(store.db_path)
    file_sizes["page_size"] = page_size
    file_sizes["page_count"] = page_count
    file_sizes["allocated_database_bytes"] = (
        page_size * page_count
    )
    return file_sizes


def deterministic_target_codec(
    current_codec: str,
    view_name: str,
    migration_index: int,
) -> str:
    choices = [
        codec
        for codec in CODECS
        if codec != current_codec
    ]
    return choices[
        stable_seed(
            SEED,
            "migration-target",
            view_name,
            migration_index,
            current_codec,
        )
        % len(choices)
    ]


def run_migrations(
    view_name: str,
    manifest: pd.DataFrame,
    store: Any,
    telemetry: sqlite3.Connection,
    truth_path: Path,
    migration_count: int,
    memory: MemoryTracker,
) -> Dict[str, Any]:
    start_ns = time.perf_counter_ns()
    chunk_size = int(
        manifest["chunk_size"].iloc[0]
    )
    truth = np.memmap(
        truth_path,
        mode="r",
        dtype=np.float32,
        shape=(len(manifest), chunk_size),
    )
    manifest_by_chunk = manifest.set_index(
        "chunk_id",
        drop=False,
    )
    chunk_ids = manifest["chunk_id"].astype(
        str
    ).tolist()

    total_rewrite_bytes = 0
    measured_migration_ns = 0
    recovered_completions = 0

    for migration_index in range(migration_count):
        existing = telemetry.execute(
            """
            SELECT *
            FROM migration_plans
            WHERE view_name=? AND migration_index=?
            """,
            (view_name, migration_index),
        ).fetchone()
        if (
            existing is not None
            and str(existing["state"]) == "completed"
        ):
            total_rewrite_bytes += int(
                existing["rewrite_bytes"] or 0
            )
            measured_migration_ns += int(
                existing["migration_time_ns"] or 0
            )
            recovered_completions += int(
                existing["recovered_completion"] or 0
            )
            continue

        if existing is None:
            chunk_id = chunk_ids[
                stable_seed(
                    SEED,
                    "migration-chunk",
                    view_name,
                    migration_index,
                )
                % len(chunk_ids)
            ]
            before = store._row(chunk_id)
            target_codec = deterministic_target_codec(
                str(before["active_codec"]),
                view_name,
                migration_index,
            )
            with telemetry_transaction(telemetry):
                telemetry.execute(
                    """
                    INSERT INTO migration_plans(
                        view_name, migration_index,
                        chunk_id, before_codec,
                        before_version, target_codec,
                        state
                    ) VALUES (?, ?, ?, ?, ?, ?, 'planned')
                    """,
                    (
                        view_name,
                        migration_index,
                        chunk_id,
                        str(before["active_codec"]),
                        int(before["active_version"]),
                        target_codec,
                    ),
                )
            plan = telemetry.execute(
                """
                SELECT *
                FROM migration_plans
                WHERE view_name=? AND migration_index=?
                """,
                (view_name, migration_index),
            ).fetchone()
        else:
            plan = existing

        chunk_id = str(plan["chunk_id"])
        target_codec = str(plan["target_codec"])
        before_version = int(plan["before_version"])
        current = store._row(chunk_id)

        recovered_completion = 0
        if (
            str(current["active_codec"]) == target_codec
            and int(current["active_version"])
            == before_version + 1
        ):
            stored = store.active_stored(chunk_id)
            migration_time_ns = 0
            rewrite_bytes = int(
                stored.stored_size_bytes
            )
            recovered_completion = 1
        elif (
            int(current["active_version"])
            == before_version
        ):
            result = store.migrate(
                chunk_id,
                target_codec,
            )
            migration_time_ns = int(
                result["migration_time_ns"]
            )
            rewrite_bytes = int(
                result["rewrite_bytes"]
            )
        else:
            raise RuntimeError(
                "Cannot safely resume migration plan "
                f"{view_name}/{migration_index}: "
                f"planned version={before_version}, "
                f"current={current['active_version']}"
            )

        row = manifest_by_chunk.loc[chunk_id]
        expected = np.asarray(
            truth[int(row["truth_index"])],
            dtype=np.float32,
        )
        decoded = store.read_all(chunk_id)
        checksum_ok = bool(
            np.array_equal(decoded, expected)
            and raw_array_sha256(decoded)
            == str(store._row(chunk_id)["raw_checksum"])
        )
        if not checksum_ok:
            raise AssertionError(
                f"Post-migration truth mismatch: {chunk_id}"
            )

        after = store._row(chunk_id)
        with telemetry_transaction(telemetry):
            telemetry.execute(
                """
                UPDATE migration_plans
                SET
                    state='completed',
                    migration_time_ns=?,
                    rewrite_bytes=?,
                    after_codec=?,
                    after_version=?,
                    checksum_ok=1,
                    recovered_completion=?,
                    completed_at_ns=?
                WHERE view_name=? AND migration_index=?
                """,
                (
                    migration_time_ns,
                    rewrite_bytes,
                    str(after["active_codec"]),
                    int(after["active_version"]),
                    recovered_completion,
                    time.time_ns(),
                    view_name,
                    migration_index,
                ),
            )

        total_rewrite_bytes += rewrite_bytes
        measured_migration_ns += migration_time_ns
        recovered_completions += recovered_completion

        completed = migration_index + 1
        if (
            completed % MIGRATION_PROGRESS_INTERVAL
            == 0
        ):
            memory.sample()
            print(
                f"[{view_name} migrations] "
                f"{completed}/{migration_count}"
            )

    del truth
    elapsed_ns = time.perf_counter_ns() - start_ns
    logical_raw_migrated = (
        migration_count * chunk_size * 4
    )
    record_stage_timing(
        telemetry,
        view_name,
        "migrations",
        elapsed_ns,
        record_count=migration_count,
        byte_count=total_rewrite_bytes,
    )
    return {
        "elapsed_ns": int(elapsed_ns),
        "migration_count": int(migration_count),
        "rewrite_bytes": int(total_rewrite_bytes),
        "logical_raw_migrated_bytes": int(
            logical_raw_migrated
        ),
        "write_amplification": float(
            total_rewrite_bytes
            / logical_raw_migrated
        ),
        "measured_migration_ns": int(
            measured_migration_ns
        ),
        "migration_throughput_mib_per_s": float(
            (
                total_rewrite_bytes
                / (1024.0 * 1024.0)
            )
            / max(
                measured_migration_ns / 1e9,
                1e-12,
            )
        ),
        "recovered_completions": int(
            recovered_completions
        ),
    }


def build_query(
    base: Any,
    chunk_size: int,
    view_name: str,
    query_index: int,
    truth_values: np.ndarray,
) -> Any:
    selector = stable_seed(
        SEED,
        "query-type",
        view_name,
        query_index,
    ) % 100

    if selector < 35:
        query_type = "point"
    elif selector < 50:
        query_type = "range_scan"
    elif selector < 62:
        query_type = "range_sum"
    elif selector < 74:
        query_type = "range_avg"
    elif selector < 84:
        query_type = "range_min"
    elif selector < 94:
        query_type = "range_max"
    else:
        query_type = "threshold_count"

    rng = np.random.default_rng(
        stable_seed(
            SEED,
            "query-geometry",
            view_name,
            query_index,
        )
    )

    if query_type == "point":
        start = int(
            rng.integers(0, chunk_size)
        )
        end = start + 1
        query_class = "point"
        threshold = None
    else:
        fractions = (
            0.001,
            0.01,
            0.10,
            0.50,
            1.00,
        )
        fraction = fractions[
            stable_seed(
                SEED,
                "query-fraction",
                view_name,
                query_index,
            )
            % len(fractions)
        ]
        length = max(
            1,
            min(
                chunk_size,
                int(round(chunk_size * fraction)),
            ),
        )
        start = (
            0
            if length == chunk_size
            else int(
                rng.integers(
                    0,
                    chunk_size - length + 1,
                )
            )
        )
        end = start + length

        if query_type == "threshold_count":
            quantile = (0.50, 0.90, 0.99)[
                stable_seed(
                    SEED,
                    "query-threshold",
                    view_name,
                    query_index,
                )
                % 3
            ]
            threshold = float(
                np.quantile(
                    truth_values,
                    quantile,
                )
            )
            query_class = "threshold"
        else:
            threshold = None
            query_class = base.query_class(
                query_type,
                (end - start) / chunk_size,
            )

    return base.QuerySpec(
        query_id=f"B2:{view_name}:{query_index}",
        query_type=query_type,
        query_class=query_class,
        start=start,
        end=end,
        range_fraction=(
            (end - start) / chunk_size
        ),
        target_fraction=(
            (end - start) / chunk_size
        ),
        threshold=threshold,
        threshold_quantile=None,
    )


def run_queries(
    view_name: str,
    manifest: pd.DataFrame,
    store: Any,
    telemetry: sqlite3.Connection,
    truth_path: Path,
    base: Any,
    query_count: int,
    memory: MemoryTracker,
) -> Dict[str, Any]:
    start_ns = time.perf_counter_ns()
    chunk_size = int(
        manifest["chunk_size"].iloc[0]
    )
    truth = np.memmap(
        truth_path,
        mode="r",
        dtype=np.float32,
        shape=(len(manifest), chunk_size),
    )
    chunk_ids = manifest["chunk_id"].astype(
        str
    ).tolist()
    truth_indices = manifest["truth_index"].to_numpy(
        dtype=np.int64
    )

    next_row = telemetry.execute(
        """
        SELECT COALESCE(MAX(query_index), -1) + 1
        AS next_index
        FROM query_records
        WHERE view_name=?
        """,
        (view_name,),
    ).fetchone()
    start_index = int(next_row["next_index"])
    if start_index >= query_count:
        print(
            f"[{view_name} query resume] already complete"
        )

    pending: List[Tuple[Any, ...]] = []
    local_latency: List[int] = []
    stage_start = time.perf_counter_ns()

    for query_index in range(
        start_index,
        query_count,
    ):
        position = stable_seed(
            SEED,
            "query-chunk",
            view_name,
            query_index,
        ) % len(chunk_ids)
        chunk_id = chunk_ids[position]
        truth_index = int(
            truth_indices[position]
        )
        truth_values = np.asarray(
            truth[truth_index],
            dtype=np.float32,
        )
        query = build_query(
            base,
            chunk_size,
            view_name,
            query_index,
            truth_values,
        )
        expected = base.execute_on_array(
            truth_values,
            query,
        )
        actual, latency_ns, stats, row = (
            store.execute_query(
                chunk_id,
                query,
            )
        )
        result_ok = bool(
            base.results_equal(actual, expected)
        )
        if not result_ok:
            raise AssertionError(
                f"Query truth mismatch: "
                f"{view_name}/{query_index}/"
                f"{chunk_id}/{query.query_type}"
            )

        pending.append(
            (
                view_name,
                query_index,
                chunk_id,
                truth_index,
                str(row["active_codec"]),
                int(row["active_version"]),
                str(query.query_type),
                str(query.query_class),
                int(query.start),
                int(query.end),
                query.threshold,
                int(latency_ns),
                int(stats.total_logical_bytes),
                1,
            )
        )
        local_latency.append(int(latency_ns))

        if len(pending) >= QUERY_INSERT_BATCH:
            with telemetry_transaction(telemetry):
                telemetry.executemany(
                    """
                    INSERT OR REPLACE INTO query_records(
                        view_name, query_index,
                        chunk_id, truth_index,
                        active_codec, active_version,
                        query_type, query_class,
                        start_index, end_index,
                        threshold, latency_ns,
                        logical_bytes, result_ok
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?
                    )
                    """,
                    pending,
                )
            pending.clear()

        completed = query_index + 1
        if (
            completed % QUERY_PROGRESS_INTERVAL
            == 0
        ):
            memory.sample()
            print(
                f"[{view_name} queries] "
                f"{completed}/{query_count}"
            )

    if pending:
        with telemetry_transaction(telemetry):
            telemetry.executemany(
                """
                INSERT OR REPLACE INTO query_records(
                    view_name, query_index,
                    chunk_id, truth_index,
                    active_codec, active_version,
                    query_type, query_class,
                    start_index, end_index,
                    threshold, latency_ns,
                    logical_bytes, result_ok
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
                )
                """,
                pending,
            )

    del truth
    elapsed_ns = time.perf_counter_ns() - start_ns
    total_records = int(
        telemetry.execute(
            """
            SELECT COUNT(*) AS n
            FROM query_records
            WHERE view_name=?
            """,
            (view_name,),
        ).fetchone()["n"]
    )
    if total_records != query_count:
        raise AssertionError(
            f"Query count mismatch for {view_name}: "
            f"{total_records}/{query_count}"
        )

    record_stage_timing(
        telemetry,
        view_name,
        "queries",
        elapsed_ns,
        record_count=query_count,
    )

    query_frame = pd.read_sql_query(
        """
        SELECT latency_ns
        FROM query_records
        WHERE view_name=?
        ORDER BY query_index
        """,
        telemetry,
        params=(view_name,),
    )
    latencies = query_frame[
        "latency_ns"
    ].to_numpy(dtype=np.float64)
    measured_query_ns = int(np.sum(latencies, dtype=np.float64))
    return {
        "wall_elapsed_ns_this_invocation": int(elapsed_ns),
        "query_count": int(query_count),
        "measured_query_execution_ns": measured_query_ns,
        "storage_throughput_queries_per_second": float(
            query_count
            / max(measured_query_ns / 1e9, 1e-12)
        ),
        "wall_throughput_this_invocation_if_fresh": (
            float(query_count / max(elapsed_ns / 1e9, 1e-12))
            if start_index == 0
            else None
        ),
        "median_latency_ns": float(
            np.median(latencies)
        ),
        "p95_latency_ns": float(
            np.quantile(latencies, 0.95)
        ),
        "p99_latency_ns": float(
            np.quantile(latencies, 0.99)
        ),
        "maximum_latency_ns": float(
            np.max(latencies)
        ),
        "all_correct": True,
    }


def active_storage_metrics(
    store: Any,
    manifest: pd.DataFrame,
) -> Dict[str, Any]:
    row = store.connection.execute(
        """
        SELECT
            COUNT(*) AS n,
            COALESCE(
                SUM(active_payload_bytes),
                0
            ) AS payload_bytes,
            COALESCE(
                SUM(active_metadata_bytes),
                0
            ) AS metadata_bytes
        FROM chunks
        """
    ).fetchone()
    raw_bytes = int(
        manifest["raw_bytes"].sum()
    )
    payload = int(row["payload_bytes"])
    metadata = int(row["metadata_bytes"])
    stored = payload + metadata

    codec_rows = pd.read_sql_query(
        """
        SELECT
            active_codec,
            COUNT(*) AS chunk_count,
            SUM(
                active_payload_bytes
                + active_metadata_bytes
            ) AS stored_bytes
        FROM chunks
        GROUP BY active_codec
        ORDER BY active_codec
        """,
        store.connection,
    )

    return {
        "chunk_count": int(row["n"]),
        "raw_bytes": raw_bytes,
        "payload_bytes": payload,
        "metadata_bytes": metadata,
        "active_stored_bytes": stored,
        "stored_over_raw": float(
            stored / raw_bytes
        ),
        "compression_factor_raw_over_stored": float(
            raw_bytes / stored
        ),
        "codec_distribution": codec_rows.to_dict(
            orient="records"
        ),
    }


def benchmark_startup(
    view_name: str,
    store_class: Any,
    store_root: Path,
    base: Any,
    telemetry: sqlite3.Connection,
) -> Dict[str, Any]:
    existing = pd.read_sql_query(
        """
        SELECT *
        FROM startup_records
        WHERE view_name=?
        ORDER BY repeat_index
        """,
        telemetry,
        params=(view_name,),
    )
    completed = set(
        existing["repeat_index"].astype(int)
    ) if not existing.empty else set()

    for repeat_index in range(STARTUP_REPEATS):
        if repeat_index in completed:
            continue
        start_ns = time.perf_counter_ns()
        store = open_production_store(
            store_class,
            store_root,
            base,
            create=True,
            recover_on_open=True,
        )
        try:
            count = store.chunk_count()
            _ = store.list_chunk_ids()
        finally:
            store.close()
        elapsed_ns = time.perf_counter_ns() - start_ns
        telemetry.execute(
            """
            INSERT OR REPLACE INTO startup_records(
                view_name, repeat_index,
                startup_ns, chunk_count
            ) VALUES (?, ?, ?, ?)
            """,
            (
                view_name,
                repeat_index,
                int(elapsed_ns),
                int(count),
            ),
        )

    frame = pd.read_sql_query(
        """
        SELECT *
        FROM startup_records
        WHERE view_name=?
        ORDER BY repeat_index
        """,
        telemetry,
        params=(view_name,),
    )
    values = frame[
        "startup_ns"
    ].to_numpy(dtype=np.float64)
    return {
        "repeats": int(len(frame)),
        "median_startup_ns": float(
            np.median(values)
        ),
        "p95_startup_ns": float(
            np.quantile(values, 0.95)
        ),
        "minimum_startup_ns": float(
            np.min(values)
        ),
        "maximum_startup_ns": float(
            np.max(values)
        ),
    }


def export_telemetry(
    output_dir: Path,
    telemetry: sqlite3.Connection,
) -> None:
    exports = {
        "ingest_records.csv": (
            "SELECT * FROM ingest_records "
            "ORDER BY view_name, chunk_id"
        ),
        "migration_records.csv": (
            "SELECT * FROM migration_plans "
            "ORDER BY view_name, migration_index"
        ),
        "query_latency_summary.csv": """
            SELECT
                view_name,
                active_codec,
                query_type,
                COUNT(*) AS n_queries,
                AVG(latency_ns) AS mean_latency_ns,
                MIN(latency_ns) AS minimum_latency_ns,
                MAX(latency_ns) AS maximum_latency_ns,
                AVG(logical_bytes) AS mean_logical_bytes
            FROM query_records
            GROUP BY
                view_name,
                active_codec,
                query_type
            ORDER BY
                view_name,
                active_codec,
                query_type
        """,
        "stage_timings.csv": (
            "SELECT * FROM stage_timings "
            "ORDER BY view_name, stage_name"
        ),
        "startup_records.csv": (
            "SELECT * FROM startup_records "
            "ORDER BY view_name, repeat_index"
        ),
    }
    for filename, query in exports.items():
        frame = pd.read_sql_query(
            query,
            telemetry,
        )
        write_dataframe_atomic(
            output_dir / filename,
            frame,
        )

    # Detailed quantiles are calculated group-by-group to keep peak memory
    # bounded even with one million query records.
    groups = pd.read_sql_query(
        """
        SELECT DISTINCT view_name, active_codec, query_type
        FROM query_records
        ORDER BY view_name, active_codec, query_type
        """,
        telemetry,
    )
    detailed_rows: List[Dict[str, Any]] = []
    for group in groups.itertuples(index=False):
        local = pd.read_sql_query(
            """
            SELECT latency_ns, logical_bytes
            FROM query_records
            WHERE view_name=? AND active_codec=? AND query_type=?
            """,
            telemetry,
            params=(group.view_name, group.active_codec, group.query_type),
        )
        latency = local["latency_ns"].to_numpy(dtype=np.float64)
        logical = local["logical_bytes"].to_numpy(dtype=np.float64)
        detailed_rows.append(
            {
                "view_name": str(group.view_name),
                "active_codec": str(group.active_codec),
                "query_type": str(group.query_type),
                "n_queries": int(len(local)),
                "median_latency_ns": float(np.median(latency)),
                "p95_latency_ns": float(np.quantile(latency, 0.95)),
                "p99_latency_ns": float(np.quantile(latency, 0.99)),
                "mean_latency_ns": float(np.mean(latency)),
                "mean_logical_bytes": float(np.mean(logical)),
            }
        )
    write_dataframe_atomic(
        output_dir / "query_latency_detailed_summary.csv",
        pd.DataFrame(detailed_rows),
    )

    # Stream the full query record table directly into gzip rather than
    # materializing one million rows in a pandas DataFrame.
    query_export_path = output_dir / "query_records.csv.gz"
    temporary = query_export_path.with_name(query_export_path.name + ".tmp")
    cursor = telemetry.execute(
        """
        SELECT *
        FROM query_records
        ORDER BY view_name, query_index
        """
    )
    columns = [description[0] for description in cursor.description]
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        while True:
            rows = cursor.fetchmany(5000)
            if not rows:
                break
            writer.writerows([tuple(row) for row in rows])
    os.replace(temporary, query_export_path)


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    b1_path = project_dir / B1_SCRIPT_NAME
    e5_path = project_dir / E5_SCRIPT_NAME
    e0c_path = project_dir / E0C_SCRIPT_NAME
    b1_result_dir = (
        project_dir / B1_RESULT_DIR_NAME
    )
    output_dir = project_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 112)
    print(
        "RePressDB B2 v1.0.0 — full BDG2 "
        "single-active scale experiment"
    )
    print(f"Target journal: {TARGET_JOURNAL}")
    print("=" * 112)

    b1 = load_module(
        b1_path,
        "repressdb_b1_for_b2_v100",
    )
    e5 = load_module(
        e5_path,
        "repressdb_e5_for_b2_v100",
    )
    base = load_module(
        e0c_path,
        "repressdb_e0c_for_b2_v100",
    )
    ProductionStore = build_production_store_class(b1)
    # B2 eligibility is defined by the largest B2 chunk size (8192),
    # not by E5's former 16384-only cross-domain selection threshold.
    e5.MIN_CONTIGUOUS_LENGTH = MIN_CONTIGUOUS_LENGTH
    e5.MIN_DISTINCT_VALUES = MIN_DISTINCT_VALUES
    e5.MIN_NONZERO_COUNT = MIN_NONZERO_COUNT
    e5.VARIANCE_EPSILON = VARIANCE_EPSILON

    if not b1_result_dir.exists():
        raise FileNotFoundError(
            f"Missing frozen B1 result directory: "
            f"{b1_result_dir}"
        )
    b1_assessment_path = (
        b1_result_dir / "b1_assessment.json"
    )
    b1_freeze_path = (
        b1_result_dir
        / "b1_experiment_freeze_sha256.txt"
    )
    if (
        not b1_assessment_path.exists()
        or not b1_freeze_path.exists()
    ):
        raise FileNotFoundError(
            "B1 assessment or freeze hash is missing"
        )

    b1_assessment = json.loads(
        b1_assessment_path.read_text(
            encoding="utf-8"
        )
    )
    if not bool(
        b1_assessment.get(
            "acceptance_checks",
            {},
        ).get("all_nine", False)
    ):
        raise RuntimeError(
            "B1 is not frozen as fully passing"
        )
    b1_freeze_hash = b1_freeze_path.read_text(
        encoding="utf-8"
    ).strip()

    memory = MemoryTracker()
    memory.sample()
    telemetry_path = output_dir / "telemetry.sqlite3"
    telemetry = telemetry_connect(telemetry_path)

    protocol = {
        "version": VERSION,
        "target_journal": TARGET_JOURNAL,
        "study": (
            "Full-BDG2 single-active storage "
            "and scale characterization"
        ),
        "parent_b1_experiment_freeze_sha256": (
            b1_freeze_hash
        ),
        "bdg2_root": str(e5.BDG2_ROOT),
        "eligibility": {
            "only_cleaned_meter_files": True,
            "minimum_contiguous_length": (
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
            "all_candidates_scanned": True,
            "interpolation": False,
            "normalization": False,
            "resampling": False,
        },
        "views": VIEW_CONFIGS,
        "store": {
            "single_active_representation": True,
            "production_catalog": (
                "stores/<view>/catalog.sqlite3"
            ),
            "telemetry_database": (
                "telemetry.sqlite3"
            ),
            "catalog_telemetry_separated": True,
            "initial_codec": INITIAL_CODEC,
            "codecs": list(CODECS),
            "maximum_open_file_handle_cache": (
                MAX_OPEN_HANDLES
            ),
        },
        "formal_workload": {
            "total_queries": int(
                sum(
                    int(config["query_count"])
                    for config in VIEW_CONFIGS.values()
                )
            ),
            "total_migrations": int(
                sum(
                    int(config["migration_count"])
                    for config in VIEW_CONFIGS.values()
                )
            ),
            "startup_repeats_per_view": (
                STARTUP_REPEATS
            ),
        },
    }
    write_json_atomic(
        output_dir / "b2_protocol.json",
        protocol,
    )

    protocol_digest = hashlib.sha256()
    protocol_digest.update(
        b1_freeze_hash.encode("ascii")
    )
    protocol_digest.update(
        sha256_file(
            output_dir / "b2_protocol.json"
        ).encode("ascii")
    )
    b2_protocol_hash = protocol_digest.hexdigest()
    (
        output_dir / "b2_protocol_sha256.txt"
    ).write_text(
        b2_protocol_hash + "\n",
        encoding="utf-8",
    )

    freeze_material = {
        "b2_protocol_sha256": (
            b2_protocol_hash
        ),
        "script_sha256": sha256_file(
            Path(__file__).resolve()
        ),
        "b1_script_sha256": sha256_file(
            b1_path
        ),
        "e5_script_sha256": sha256_file(
            e5_path
        ),
        "e0c_script_sha256": sha256_file(
            e0c_path
        ),
    }
    write_json_atomic(
        output_dir
        / "b2_experiment_freeze_material.json",
        freeze_material,
    )
    freeze_hash = sha256_file(
        output_dir
        / "b2_experiment_freeze_material.json"
    )
    (
        output_dir
        / "b2_experiment_freeze_sha256.txt"
    ).write_text(
        freeze_hash + "\n",
        encoding="utf-8",
    )

    existing_freeze = telemetry_get(
        telemetry,
        "experiment_freeze_sha256",
    )
    if (
        existing_freeze is not None
        and existing_freeze != freeze_hash
    ):
        raise RuntimeError(
            "Existing B2 telemetry belongs to another "
            "experiment freeze.\n"
            f"Existing: {existing_freeze}\n"
            f"Current:  {freeze_hash}"
        )
    telemetry_set(
        telemetry,
        "experiment_freeze_sha256",
        freeze_hash,
    )
    telemetry_set(
        telemetry,
        "b2_protocol_sha256",
        b2_protocol_hash,
    )

    total_start_ns = time.perf_counter_ns()

    qualification_start = time.perf_counter_ns()
    candidates, eligible = build_full_qualification(
        output_dir,
        e5,
        memory,
    )
    qualification_elapsed_ns = (
        time.perf_counter_ns()
        - qualification_start
    )
    record_stage_timing(
        telemetry,
        "__global__",
        "full_qualification",
        qualification_elapsed_ns,
        record_count=len(candidates),
    )

    manifests = build_manifests(
        output_dir,
        eligible,
    )

    dataset_summary = {
        "candidate_meter_series": int(
            len(candidates)
        ),
        "eligible_meter_series": int(
            len(eligible)
        ),
        "eligible_unique_buildings": int(
            eligible["building_id"].nunique()
        ),
        "eligible_meter_types": {
            str(key): int(value)
            for key, value
            in eligible["meter_type"]
            .value_counts()
            .sort_index()
            .items()
        },
        "eligible_sites": int(
            eligible["site_id"].nunique()
        ),
        "views": {
            view_name: {
                "chunk_size": int(
                    manifest[
                        "chunk_size"
                    ].iloc[0]
                ),
                "chunk_count": int(
                    len(manifest)
                ),
                "raw_bytes": int(
                    manifest["raw_bytes"].sum()
                ),
                "unique_series": int(
                    manifest[
                        "series_key"
                    ].nunique()
                ),
                "unique_buildings": int(
                    manifest[
                        "building_id"
                    ].nunique()
                ),
            }
            for view_name, manifest
            in manifests.items()
        },
    }
    write_json_atomic(
        output_dir / "bdg2_full_dataset_summary.json",
        dataset_summary,
    )

    view_results: Dict[str, Any] = {}

    for view_name, config in VIEW_CONFIGS.items():
        print("-" * 112)
        print(f"[B2] processing {view_name}")
        manifest = manifests[view_name]
        view_root = output_dir / "stores" / view_name
        truth_path = (
            output_dir
            / "truth"
            / f"{view_name}.float32.bin"
        )

        store = open_production_store(
            ProductionStore,
            view_root,
            base,
            create=True,
            recover_on_open=True,
        )
        try:
            existing_store_freeze = store.get_metadata(
                "experiment_freeze_sha256"
            )
            if (
                existing_store_freeze is not None
                and existing_store_freeze
                != freeze_hash
            ):
                raise RuntimeError(
                    f"{view_name} store belongs to "
                    "another freeze"
                )
            store.set_metadata(
                "experiment_freeze_sha256",
                freeze_hash,
            )
            store.set_metadata(
                "view_name",
                view_name,
            )
            store.set_metadata(
                "chunk_size",
                int(config["chunk_size"]),
            )

            ingest_result = ingest_view(
                view_name,
                manifest,
                store,
                telemetry,
                truth_path,
                e5,
                memory,
            )

            if store.chunk_count() != len(manifest):
                raise AssertionError(
                    f"{view_name} catalog count mismatch"
                )

            initial_catalog_path = (
                output_dir / f"{view_name}_initial_catalog_metrics.json"
            )
            initial_storage_path = (
                output_dir / f"{view_name}_initial_storage_metrics.json"
            )
            if initial_catalog_path.exists() and initial_storage_path.exists():
                initial_catalog = json.loads(
                    initial_catalog_path.read_text(encoding="utf-8")
                )
                initial_storage = json.loads(
                    initial_storage_path.read_text(encoding="utf-8")
                )
            else:
                initial_storage = active_storage_metrics(
                    store,
                    manifest,
                )
                initial_catalog = checkpoint_catalog(store)
                initial_catalog["bytes_per_chunk_allocated"] = float(
                    initial_catalog["allocated_database_bytes"] / len(manifest)
                )
                initial_catalog["files_bytes_per_chunk"] = float(
                    initial_catalog["total_catalog_files_bytes"] / len(manifest)
                )
                write_json_atomic(initial_catalog_path, initial_catalog)
                write_json_atomic(initial_storage_path, initial_storage)

            migration_result = run_migrations(
                view_name,
                manifest,
                store,
                telemetry,
                truth_path,
                int(config["migration_count"]),
                memory,
            )

            query_result = run_queries(
                view_name,
                manifest,
                store,
                telemetry,
                truth_path,
                base,
                int(config["query_count"]),
                memory,
            )

            final_integrity = store.integrity_check()
            if not final_integrity["pass"]:
                raise AssertionError(
                    f"{view_name} final integrity failed: "
                    f"{final_integrity}"
                )
            final_storage = active_storage_metrics(
                store,
                manifest,
            )
            final_catalog = checkpoint_catalog(
                store
            )
            final_catalog[
                "bytes_per_chunk_allocated"
            ] = float(
                final_catalog[
                    "allocated_database_bytes"
                ]
                / len(manifest)
            )
            final_catalog[
                "files_bytes_per_chunk"
            ] = float(
                final_catalog[
                    "total_catalog_files_bytes"
                ]
                / len(manifest)
            )

            write_json_atomic(
                output_dir
                / f"{view_name}_final_integrity.json",
                final_integrity,
            )
            write_json_atomic(
                output_dir
                / f"{view_name}_final_catalog_metrics.json",
                final_catalog,
            )
            write_json_atomic(
                output_dir
                / f"{view_name}_final_storage_metrics.json",
                final_storage,
            )

            catalog_frame = pd.read_sql_query(
                """
                SELECT *
                FROM chunks
                ORDER BY chunk_id
                """,
                store.connection,
            )
            write_dataframe_atomic(
                output_dir
                / f"{view_name}_catalog_final.csv",
                catalog_frame,
            )
        finally:
            store.close()

        startup_result = benchmark_startup(
            view_name,
            ProductionStore,
            view_root,
            base,
            telemetry,
        )

        view_results[view_name] = {
            "manifest": (
                dataset_summary["views"][
                    view_name
                ]
            ),
            "ingestion": ingest_result,
            "initial_storage": (
                initial_storage
            ),
            "initial_catalog": (
                initial_catalog
            ),
            "migrations": migration_result,
            "queries": query_result,
            "final_integrity": final_integrity,
            "final_storage": final_storage,
            "final_catalog": final_catalog,
            "startup": startup_result,
        }

    export_telemetry(
        output_dir,
        telemetry,
    )

    memory.sample()
    total_elapsed_ns = (
        time.perf_counter_ns()
        - total_start_ns
    )

    total_queries = int(
        telemetry.execute(
            "SELECT COUNT(*) AS n "
            "FROM query_records"
        ).fetchone()["n"]
    )
    correct_queries = int(
        telemetry.execute(
            "SELECT COALESCE(SUM(result_ok),0) AS n "
            "FROM query_records"
        ).fetchone()["n"]
    )
    total_migrations = int(
        telemetry.execute(
            """
            SELECT COUNT(*) AS n
            FROM migration_plans
            WHERE state='completed'
            """
        ).fetchone()["n"]
    )
    migration_checksums = int(
        telemetry.execute(
            """
            SELECT COALESCE(SUM(checksum_ok),0) AS n
            FROM migration_plans
            WHERE state='completed'
            """
        ).fetchone()["n"]
    )

    expected_queries = int(
        sum(
            int(config["query_count"])
            for config in VIEW_CONFIGS.values()
        )
    )
    expected_migrations = int(
        sum(
            int(config["migration_count"])
            for config in VIEW_CONFIGS.values()
        )
    )

    checks = {
        "B2_1_all_candidate_meter_series_qualified": bool(
            len(candidates) > 0
            and len(candidates)
            == len(
                pd.read_csv(
                    output_dir
                    / "bdg2_full_qualification_catalog.csv"
                )
            )
        ),
        "B2_2_at_least_one_eligible_meter": bool(
            len(eligible) > 0
        ),
        "B2_3_both_full_manifests_nonempty": bool(
            all(
                len(manifest) > 0
                for manifest in manifests.values()
            )
        ),
        "B2_4_production_catalog_and_telemetry_separated": bool(
            telemetry_path.exists()
            and all(
                (
                    output_dir
                    / "stores"
                    / view_name
                    / "catalog.sqlite3"
                ).exists()
                for view_name in VIEW_CONFIGS
            )
        ),
        "B2_5_all_1000000_queries_correct": bool(
            total_queries == expected_queries
            and correct_queries == expected_queries
        ),
        "B2_6_all_1000_migrations_checksum_correct": bool(
            total_migrations == expected_migrations
            and migration_checksums
            == expected_migrations
        ),
        "B2_7_both_stores_have_one_active_payload_per_chunk": bool(
            all(
                bool(
                    view_results[view_name][
                        "final_integrity"
                    ]["pass"]
                )
                for view_name in VIEW_CONFIGS
            )
        ),
        "B2_8_no_staging_files": bool(
            all(
                int(
                    view_results[view_name][
                        "final_integrity"
                    ][
                        "temporary_file_count"
                    ]
                )
                == 0
                for view_name in VIEW_CONFIGS
            )
        ),
        "B2_9_catalog_bytes_per_chunk_reported": bool(
            all(
                float(
                    view_results[view_name][
                        "initial_catalog"
                    ][
                        "bytes_per_chunk_allocated"
                    ]
                )
                > 0
                for view_name in VIEW_CONFIGS
            )
        ),
        "B2_10_startup_measured_five_times_per_view": bool(
            all(
                int(
                    view_results[view_name][
                        "startup"
                    ]["repeats"]
                )
                == STARTUP_REPEATS
                for view_name in VIEW_CONFIGS
            )
        ),
    }
    checks["all_ten"] = bool(
        all(checks.values())
    )

    assessment = {
        "version": VERSION,
        "target_journal": TARGET_JOURNAL,
        "status": (
            "B2 complete"
            if checks["all_ten"]
            else (
                "B2 completed with failed "
                "acceptance checks"
            )
        ),
        "parent_b1_experiment_freeze_sha256": (
            b1_freeze_hash
        ),
        "b2_protocol_sha256": (
            b2_protocol_hash
        ),
        "b2_experiment_freeze_sha256": (
            freeze_hash
        ),
        "dataset_summary": dataset_summary,
        "view_results": view_results,
        "memory": {
            "method": memory.method,
            "peak_rss_bytes": int(
                memory.peak_rss_bytes
            ),
            "final_sample_rss_bytes": int(
                memory.current_rss_bytes
            ),
        },
        "total_elapsed_ns_this_invocation": int(
            total_elapsed_ns
        ),
        "acceptance_checks": checks,
        "interpretation": (
            "A passing B2 establishes full cleaned-BDG2 "
            "storage-scale characterization with production "
            "catalog and telemetry separation. It reports "
            "real-data storage, catalog, startup, query, "
            "migration, write-amplification, and memory "
            "measurements. Synthetic replicated-tenant "
            "1/5/10-GB scale-out and concurrent foreground/"
            "background interference remain B3."
        ),
        "next_gate": (
            "Proceed to B3 replicated-tenant 1/5/10-GB "
            "scale-out and concurrent-query experiments."
            if checks["all_ten"]
            else (
                "Stop and diagnose failed B2 checks before "
                "scale-out or concurrency experiments."
            )
        ),
    }
    write_json_atomic(
        output_dir / "b2_assessment.json",
        assessment,
    )
    write_json_atomic(
        project_dir / SUMMARY_JSON_NAME,
        assessment,
    )

    run_config = {
        "version": VERSION,
        "target_journal": TARGET_JOURNAL,
        "b2_protocol_sha256": (
            b2_protocol_hash
        ),
        "b2_experiment_freeze_sha256": (
            freeze_hash
        ),
        "parent_b1_experiment_freeze_sha256": (
            b1_freeze_hash
        ),
        "views": VIEW_CONFIGS,
        "qualification_batch_size": (
            QUALIFICATION_BATCH_SIZE
        ),
        "ingest_column_batch_size": (
            INGEST_COLUMN_BATCH_SIZE
        ),
        "query_insert_batch": (
            QUERY_INSERT_BATCH
        ),
        "maximum_open_handles": (
            MAX_OPEN_HANDLES
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sqlite_version": (
                sqlite3.sqlite_version
            ),
            "memory_measurement": (
                memory.method
            ),
            "timer": "time.perf_counter_ns",
        },
    }
    write_json_atomic(
        output_dir / "run_config.json",
        run_config,
    )

    telemetry.execute(
        "PRAGMA wal_checkpoint(TRUNCATE)"
    )
    telemetry.close()

    print("=" * 112)
    print("RePressDB B2 v1.0.0 finished")
    print(
        json.dumps(
            checks,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        f"Elapsed this invocation: "
        f"{total_elapsed_ns / 1e9 / 60:.2f} minutes"
    )
    print(f"Results: {output_dir.resolve()}")
    print("Upload:")
    print("  b2_assessment.json")
    print("  bdg2_full_dataset_summary.json")
    print("  data_scale_8192_initial_catalog_metrics.json")
    print("  catalog_scale_2048_initial_catalog_metrics.json")
    print("  data_scale_8192_final_storage_metrics.json")
    print("  catalog_scale_2048_final_storage_metrics.json")
    print("  query_latency_summary.csv")
    print("  migration_records.csv")
    print("  stage_timings.csv")
    print("  startup_records.csv")
    print("  b2_protocol.json")
    print("  b2_protocol_sha256.txt")
    print("  b2_experiment_freeze_sha256.txt")
    print("  run_config.json")
    print("=" * 112)
    return 0 if checks["all_ten"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
