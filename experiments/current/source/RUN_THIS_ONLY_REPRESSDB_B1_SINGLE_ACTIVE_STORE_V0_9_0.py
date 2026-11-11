#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
RePressDB B1 v0.9.0
Single-active-representation chunk store with SQLite-WAL catalog,
atomic version migration, checksum validation, crash recovery, and
resumable correctness tests.

Target journal: Journal of Big Data.

DIRECT RUN
----------
Place this file beside:

  RUN_THIS_ONLY_REPRESSDB_E0C_V0_2_0.py

Keep the frozen A0 directory:

  results_a0_a1_hotcold_pems04_v070/

Open this file in PyCharm and click Run.

No command-line arguments are required.

FORMAL B1 ACCEPTANCE WORKLOAD
-----------------------------
  * 32 frozen PEMS04 series
  * one 16384-value chunk per series
  * one active physical representation per chunk
  * 100,000 randomized file-backed queries
  * 1,000 successful randomized physical migrations
  * 6 fault-injection stages × 100 trials = 600 recovery trials

The store normally retains only the active representation. During migration,
the previous active version remains available only until the catalog commit
has been validated and cleanup finishes.

CHECKPOINTING
-------------
The store catalog and all test records are in SQLite. Completed query,
migration, and fault-injection records are committed transactionally.
Rerunning this file resumes from the first incomplete test index.

Do not delete:
  results_b1_single_active_store_v090/

Do not modify this script after the run begins. The script SHA-256 is part of
the experiment freeze.
"""

from __future__ import annotations

import contextlib
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
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


VERSION = "0.9.0"
TARGET_JOURNAL = "Journal of Big Data"

E0C_SCRIPT_NAME = "RUN_THIS_ONLY_REPRESSDB_E0C_V0_2_0.py"
A0_RESULT_DIR_NAME = "results_a0_a1_hotcold_pems04_v070"
OUTPUT_DIR_NAME = "results_b1_single_active_store_v090"

SOURCE_MANIFEST_NAME = "pems04_selected_chunks.csv"
SOURCE_PROTOCOL_HASH_NAME = "protocol_sha256.txt"

CHUNK_SIZE = 16384
EXPECTED_CHUNKS = 32
INITIAL_CODEC = "whole_lz4"
CODECS = (
    "whole_lz4",
    "whole_zstd",
    "block_lz4_64",
    "block_lz4_256",
    "block_lz4_1024",
)

QUERY_TEST_COUNT = 100_000
MIGRATION_TEST_COUNT = 1_000
FAULT_TRIALS_PER_STAGE = 100
FAULT_STAGES = (
    "after_intent",
    "after_encode",
    "after_validate",
    "after_payload_publish_before_metadata",
    "after_full_publish_before_catalog_commit",
    "after_catalog_commit_corrupt_active",
)

QUERY_BATCH_COMMIT = 500
MIGRATION_PROGRESS_PRINT = 25
FAULT_PROGRESS_PRINT = 10
QUERY_PROGRESS_PRINT = 5_000

SEED = 20260724
ZSTD_LEVEL = 3

QUERY_TYPES = (
    "point",
    "range_scan",
    "range_sum",
    "range_avg",
    "range_min",
    "range_max",
    "threshold_count",
)

NO_RESULT = object()


class InjectedMigrationFault(RuntimeError):
    def __init__(self, stage: str):
        super().__init__(f"Injected migration fault at stage: {stage}")
        self.stage = stage


def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def raw_array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float32)
    return sha256_bytes(array.tobytes(order="C"))


def stable_seed(*parts: Any) -> int:
    token = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def load_module(path: Path, name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing dependency: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def safe_unlink(path: Optional[Path]) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        time.sleep(0.05)
        path.unlink(missing_ok=True)


def relative_path(root: Path, path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    return str(path.resolve().relative_to(root.resolve()))


def absolute_path(root: Path, value: Optional[str]) -> Optional[Path]:
    if value is None or value == "":
        return None
    return root / value


def sqlite_connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        path,
        timeout=60.0,
        isolation_level=None,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=60000")
    return connection


@contextlib.contextmanager
def immediate_transaction(
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


@dataclasses.dataclass(frozen=True)
class FrozenChunk:
    chunk_id: str
    series_index: int
    chunk_start: int
    chunk_size: int
    values: np.ndarray
    raw_checksum: str


@dataclasses.dataclass
class CachedHandles:
    active_version: int
    payload: Any
    metadata: Any


class SingleActiveChunkStore:
    def __init__(
        self,
        root: Path,
        base: Any,
        create: bool = True,
        recover_on_open: bool = True,
    ):
        self.root = root.resolve()
        self.base = base
        self.root.mkdir(parents=True, exist_ok=True)
        self.chunk_root = self.root / "chunks"
        self.staging_root = self.root / "staging"
        self.chunk_root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "catalog.sqlite3"
        self.connection = sqlite_connect(self.db_path)
        self.representations = base.representation_registry(ZSTD_LEVEL)
        if set(self.representations) != set(CODECS):
            raise AssertionError(
                f"Unexpected representation registry: {sorted(self.representations)}"
            )
        self.mode = base.CACHE_MODES["warm_slicedmeta"]
        self._handle_cache: Dict[str, CachedHandles] = {}
        self._lock = threading.RLock()
        if create:
            self._create_schema()
        if recover_on_open:
            self.recover()

    def close(self) -> None:
        with self._lock:
            self.invalidate_all_handles()
            self.connection.close()

    def __enter__(self) -> "SingleActiveChunkStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

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
                updated_at_ns INTEGER NOT NULL
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

            CREATE INDEX IF NOT EXISTS idx_migration_chunk
            ON migration_log(chunk_id);

            CREATE TABLE IF NOT EXISTS query_test_records (
                query_index INTEGER PRIMARY KEY,
                chunk_id TEXT NOT NULL,
                active_codec TEXT NOT NULL,
                active_version INTEGER NOT NULL,
                query_type TEXT NOT NULL,
                query_class TEXT NOT NULL,
                start_index INTEGER NOT NULL,
                end_index INTEGER NOT NULL,
                threshold REAL,
                latency_ns INTEGER NOT NULL,
                logical_bytes INTEGER NOT NULL,
                result_ok INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS migration_test_records (
                migration_index INTEGER PRIMARY KEY,
                chunk_id TEXT NOT NULL,
                from_codec TEXT NOT NULL,
                target_codec TEXT NOT NULL,
                from_version INTEGER NOT NULL,
                to_version INTEGER NOT NULL,
                migration_time_ns INTEGER NOT NULL,
                rewrite_bytes INTEGER NOT NULL,
                checksum_ok INTEGER NOT NULL,
                query_spotcheck_ok INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS fault_test_records (
                stage TEXT NOT NULL,
                trial_index INTEGER NOT NULL,
                chunk_id TEXT NOT NULL,
                target_codec TEXT NOT NULL,
                injected_fault_observed INTEGER NOT NULL,
                recovered_checksum_ok INTEGER NOT NULL,
                recovered_query_ok INTEGER NOT NULL,
                catalog_idle INTEGER NOT NULL,
                one_active_payload INTEGER NOT NULL,
                no_temporary_files INTEGER NOT NULL,
                expected_outcome TEXT NOT NULL,
                observed_active_codec TEXT NOT NULL,
                elapsed_ns INTEGER NOT NULL,
                PRIMARY KEY(stage, trial_index)
            );
            """
        )

    def set_metadata(self, key: str, value: Any) -> None:
        serialized = json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True)
        self.connection.execute(
            """
            INSERT INTO store_metadata(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (key, serialized),
        )

    def get_metadata(self, key: str) -> Optional[Any]:
        row = self.connection.execute(
            "SELECT value FROM store_metadata WHERE key=?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["value"])

    def invalidate_handle(self, chunk_id: str) -> None:
        cached = self._handle_cache.pop(chunk_id, None)
        if cached is None:
            return
        for handle in (cached.payload, cached.metadata):
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass

    def invalidate_all_handles(self) -> None:
        for chunk_id in list(self._handle_cache):
            self.invalidate_handle(chunk_id)

    def _row(self, chunk_id: str) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM chunks WHERE chunk_id=?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown chunk: {chunk_id}")
        return row

    def list_chunk_ids(self) -> List[str]:
        rows = self.connection.execute(
            "SELECT chunk_id FROM chunks ORDER BY chunk_id"
        ).fetchall()
        return [str(row["chunk_id"]) for row in rows]

    def chunk_count(self) -> int:
        return int(
            self.connection.execute(
                "SELECT COUNT(*) AS n FROM chunks"
            ).fetchone()["n"]
        )

    def _version_stem(
        self,
        chunk_id: str,
        version: int,
        codec: str,
        temporary_token: Optional[str] = None,
    ) -> Path:
        if temporary_token is None:
            directory = self.chunk_root / chunk_id
            directory.mkdir(parents=True, exist_ok=True)
            return directory / f"v{version:08d}__{codec}"
        directory = self.staging_root / chunk_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory / (
            f"v{version:08d}__{codec}__{temporary_token}.tmp"
        )

    def _stored_from_fields(
        self,
        codec: str,
        n: int,
        data_path_text: str,
        metadata_path_text: Optional[str],
        payload_bytes: int,
        metadata_bytes: int,
    ) -> Any:
        data_path = absolute_path(self.root, data_path_text)
        metadata_path = absolute_path(self.root, metadata_path_text)
        if data_path is None:
            raise ValueError("Missing data path")
        if metadata_path is not None:
            metadata = self.base.parse_block_metadata(
                metadata_path.read_bytes()
            )
        elif codec == "whole_zstd":
            metadata = {"level": ZSTD_LEVEL}
        else:
            metadata = {}
        return self.base.StoredChunk(
            codec=codec,
            n=int(n),
            data_path=data_path,
            metadata_path=metadata_path,
            payload_size_bytes=int(payload_bytes),
            metadata_size_bytes=int(metadata_bytes),
            metadata=metadata,
        )

    def active_stored(self, chunk_id: str) -> Any:
        row = self._row(chunk_id)
        return self._stored_from_fields(
            codec=str(row["active_codec"]),
            n=int(row["n"]),
            data_path_text=str(row["active_data_path"]),
            metadata_path_text=row["active_metadata_path"],
            payload_bytes=int(row["active_payload_bytes"]),
            metadata_bytes=int(row["active_metadata_bytes"]),
        )

    def previous_stored(self, chunk_id: str) -> Optional[Any]:
        row = self._row(chunk_id)
        if row["previous_version"] is None:
            return None
        return self._stored_from_fields(
            codec=str(row["previous_codec"]),
            n=int(row["n"]),
            data_path_text=str(row["previous_data_path"]),
            metadata_path_text=row["previous_metadata_path"],
            payload_bytes=int(row["previous_payload_bytes"]),
            metadata_bytes=int(row["previous_metadata_bytes"]),
        )

    def _open_cached_handles(
        self,
        chunk_id: str,
        row: sqlite3.Row,
        stored: Any,
    ) -> CachedHandles:
        active_version = int(row["active_version"])
        cached = self._handle_cache.get(chunk_id)
        if cached is not None and cached.active_version == active_version:
            return cached
        self.invalidate_handle(chunk_id)
        payload = open(stored.data_path, "rb")
        metadata = (
            open(stored.metadata_path, "rb")
            if stored.metadata_path is not None
            else None
        )
        cached = CachedHandles(active_version, payload, metadata)
        self._handle_cache[chunk_id] = cached
        return cached

    def create_chunk(
        self,
        chunk_id: str,
        series_index: int,
        values: np.ndarray,
        codec: str = INITIAL_CODEC,
    ) -> None:
        with self._lock:
            if codec not in self.representations:
                raise ValueError(f"Unknown codec: {codec}")
            if self.connection.execute(
                "SELECT 1 FROM chunks WHERE chunk_id=?",
                (chunk_id,),
            ).fetchone() is not None:
                return

            array = np.ascontiguousarray(values, dtype=np.float32)
            checksum = raw_array_sha256(array)
            stem = self._version_stem(chunk_id, 0, codec)
            representation = self.representations[codec]
            stored = representation.encode_to_files(array, stem)

            decoded = self._decode_stored(stored)
            if not np.array_equal(decoded, array):
                self._delete_stored_files(stored)
                raise AssertionError(
                    f"Initial encoding mismatch: {chunk_id}/{codec}"
                )
            if raw_array_sha256(decoded) != checksum:
                self._delete_stored_files(stored)
                raise AssertionError(
                    f"Initial checksum mismatch: {chunk_id}/{codec}"
                )

            now = time.time_ns()
            with immediate_transaction(self.connection):
                self.connection.execute(
                    """
                    INSERT INTO chunks(
                        chunk_id, series_index, n, raw_checksum,
                        active_version, active_codec,
                        active_data_path, active_metadata_path,
                        active_payload_bytes, active_metadata_bytes,
                        migration_state, created_at_ns, updated_at_ns
                    ) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?, 'idle', ?, ?)
                    """,
                    (
                        chunk_id,
                        int(series_index),
                        int(array.size),
                        checksum,
                        codec,
                        relative_path(self.root, stored.data_path),
                        relative_path(self.root, stored.metadata_path),
                        int(stored.payload_size_bytes),
                        int(stored.metadata_size_bytes),
                        now,
                        now,
                    ),
                )

    def _decode_stored(self, stored: Any) -> np.ndarray:
        representation = self.representations[stored.codec]
        stats = self.base.IOStats()
        with open(stored.data_path, "rb") as payload:
            return representation.decode_all(
                stored,
                stored.metadata,
                payload,
                stats,
            )

    def read_all(self, chunk_id: str) -> np.ndarray:
        with self._lock:
            return self._decode_stored(self.active_stored(chunk_id))

    def execute_query(
        self,
        chunk_id: str,
        query: Any,
    ) -> Tuple[Any, int, Any, sqlite3.Row]:
        with self._lock:
            row = self._row(chunk_id)
            stored = self.active_stored(chunk_id)
            representation = self.representations[stored.codec]
            cached = self._open_cached_handles(chunk_id, row, stored)
            start_ns = time.perf_counter_ns()
            result, stats = self.base.run_query_once(
                representation,
                stored,
                query,
                self.mode,
                cached.payload,
                cached.metadata,
            )
            latency_ns = time.perf_counter_ns() - start_ns
            return result, int(latency_ns), stats, row

    def _delete_stored_files(self, stored: Optional[Any]) -> None:
        if stored is None:
            return
        safe_unlink(stored.data_path)
        safe_unlink(stored.metadata_path)

    def _validate_stored(
        self,
        stored: Any,
        expected_checksum: str,
        expected_values: Optional[np.ndarray] = None,
    ) -> bool:
        try:
            if not stored.data_path.exists():
                return False
            if (
                stored.metadata_path is not None
                and not stored.metadata_path.exists()
            ):
                return False
            decoded = self._decode_stored(stored)
            if raw_array_sha256(decoded) != expected_checksum:
                return False
            if (
                expected_values is not None
                and not np.array_equal(
                    decoded,
                    np.ascontiguousarray(expected_values, dtype=np.float32),
                )
            ):
                return False
            return True
        except Exception:
            return False

    def migrate(
        self,
        chunk_id: str,
        target_codec: str,
        fault_stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            if target_codec not in self.representations:
                raise ValueError(f"Unknown target codec: {target_codec}")
            row = self._row(chunk_id)
            if str(row["migration_state"]) != "idle":
                raise RuntimeError(
                    f"Chunk is not idle before migration: {chunk_id}"
                )
            current_codec = str(row["active_codec"])
            current_version = int(row["active_version"])
            if current_codec == target_codec:
                return {
                    "chunk_id": chunk_id,
                    "from_codec": current_codec,
                    "target_codec": target_codec,
                    "from_version": current_version,
                    "to_version": current_version,
                    "migration_time_ns": 0,
                    "rewrite_bytes": 0,
                    "no_op": True,
                }

            start_ns = time.perf_counter_ns()
            to_version = current_version + 1
            migration_id = uuid.uuid4().hex
            temporary_token = uuid.uuid4().hex
            temp_stem = self._version_stem(
                chunk_id,
                to_version,
                target_codec,
                temporary_token=temporary_token,
            )
            final_stem = self._version_stem(
                chunk_id,
                to_version,
                target_codec,
            )
            temp_data = temp_stem.with_suffix(".bin")
            temp_meta = temp_stem.with_suffix(".meta.bin")
            final_data = final_stem.with_suffix(".bin")
            final_meta = final_stem.with_suffix(".meta.bin")
            now = time.time_ns()

            with immediate_transaction(self.connection):
                self.connection.execute(
                    """
                    INSERT INTO migration_log(
                        migration_id, chunk_id,
                        from_version, to_version,
                        from_codec, target_codec,
                        state,
                        temporary_data_path,
                        temporary_metadata_path,
                        final_data_path,
                        final_metadata_path,
                        started_at_ns, updated_at_ns
                    ) VALUES (?, ?, ?, ?, ?, ?, 'intent', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        migration_id,
                        chunk_id,
                        current_version,
                        to_version,
                        current_codec,
                        target_codec,
                        relative_path(self.root, temp_data),
                        relative_path(self.root, temp_meta),
                        relative_path(self.root, final_data),
                        relative_path(self.root, final_meta),
                        now,
                        now,
                    ),
                )
                self.connection.execute(
                    """
                    UPDATE chunks
                    SET migration_state='intent', updated_at_ns=?
                    WHERE chunk_id=?
                    """,
                    (now, chunk_id),
                )

            if fault_stage == "after_intent":
                raise InjectedMigrationFault(fault_stage)

            current_stored = self.active_stored(chunk_id)
            values = self._decode_stored(current_stored)
            raw_checksum = str(row["raw_checksum"])
            if raw_array_sha256(values) != raw_checksum:
                raise AssertionError(
                    f"Current active checksum invalid before migration: {chunk_id}"
                )

            representation = self.representations[target_codec]
            encoded = representation.encode_to_files(values, temp_stem)
            with immediate_transaction(self.connection):
                self.connection.execute(
                    """
                    UPDATE migration_log
                    SET state='encoded',
                        temporary_data_path=?,
                        temporary_metadata_path=?,
                        target_payload_bytes=?,
                        target_metadata_bytes=?,
                        updated_at_ns=?
                    WHERE migration_id=?
                    """,
                    (
                        relative_path(self.root, encoded.data_path),
                        relative_path(self.root, encoded.metadata_path),
                        int(encoded.payload_size_bytes),
                        int(encoded.metadata_size_bytes),
                        time.time_ns(),
                        migration_id,
                    ),
                )
                self.connection.execute(
                    """
                    UPDATE chunks
                    SET migration_state='encoded', updated_at_ns=?
                    WHERE chunk_id=?
                    """,
                    (time.time_ns(), chunk_id),
                )

            if fault_stage == "after_encode":
                raise InjectedMigrationFault(fault_stage)

            if not self._validate_stored(
                encoded,
                raw_checksum,
                expected_values=values,
            ):
                raise AssertionError(
                    f"Target validation failed: {chunk_id}/{target_codec}"
                )
            with immediate_transaction(self.connection):
                self.connection.execute(
                    """
                    UPDATE migration_log
                    SET state='validated', updated_at_ns=?
                    WHERE migration_id=?
                    """,
                    (time.time_ns(), migration_id),
                )
                self.connection.execute(
                    """
                    UPDATE chunks
                    SET migration_state='validated', updated_at_ns=?
                    WHERE chunk_id=?
                    """,
                    (time.time_ns(), chunk_id),
                )

            if fault_stage == "after_validate":
                raise InjectedMigrationFault(fault_stage)

            final_data.parent.mkdir(parents=True, exist_ok=True)
            safe_unlink(final_data)
            safe_unlink(final_meta)
            os.replace(encoded.data_path, final_data)
            fsync_directory(final_data.parent)

            if fault_stage == "after_payload_publish_before_metadata":
                with immediate_transaction(self.connection):
                    self.connection.execute(
                        """
                        UPDATE migration_log
                        SET state='payload_published', updated_at_ns=?
                        WHERE migration_id=?
                        """,
                        (time.time_ns(), migration_id),
                    )
                    self.connection.execute(
                        """
                        UPDATE chunks
                        SET migration_state='payload_published', updated_at_ns=?
                        WHERE chunk_id=?
                        """,
                        (time.time_ns(), chunk_id),
                    )
                raise InjectedMigrationFault(fault_stage)

            if encoded.metadata_path is not None:
                os.replace(encoded.metadata_path, final_meta)
                final_metadata_path: Optional[Path] = final_meta
            else:
                safe_unlink(temp_meta)
                final_metadata_path = None
            fsync_directory(final_data.parent)

            published = self.base.StoredChunk(
                codec=target_codec,
                n=int(row["n"]),
                data_path=final_data,
                metadata_path=final_metadata_path,
                payload_size_bytes=int(final_data.stat().st_size),
                metadata_size_bytes=(
                    int(final_metadata_path.stat().st_size)
                    if final_metadata_path is not None
                    else 0
                ),
                metadata=(
                    self.base.parse_block_metadata(
                        final_metadata_path.read_bytes()
                    )
                    if final_metadata_path is not None
                    else (
                        {"level": ZSTD_LEVEL}
                        if target_codec == "whole_zstd"
                        else {}
                    )
                ),
            )

            with immediate_transaction(self.connection):
                self.connection.execute(
                    """
                    UPDATE migration_log
                    SET state='published',
                        final_data_path=?,
                        final_metadata_path=?,
                        target_payload_bytes=?,
                        target_metadata_bytes=?,
                        updated_at_ns=?
                    WHERE migration_id=?
                    """,
                    (
                        relative_path(self.root, published.data_path),
                        relative_path(self.root, published.metadata_path),
                        int(published.payload_size_bytes),
                        int(published.metadata_size_bytes),
                        time.time_ns(),
                        migration_id,
                    ),
                )
                self.connection.execute(
                    """
                    UPDATE chunks
                    SET migration_state='published', updated_at_ns=?
                    WHERE chunk_id=?
                    """,
                    (time.time_ns(), chunk_id),
                )

            if fault_stage == "after_full_publish_before_catalog_commit":
                raise InjectedMigrationFault(fault_stage)

            with immediate_transaction(self.connection):
                self.connection.execute(
                    """
                    UPDATE chunks
                    SET
                        previous_version=active_version,
                        previous_codec=active_codec,
                        previous_data_path=active_data_path,
                        previous_metadata_path=active_metadata_path,
                        previous_payload_bytes=active_payload_bytes,
                        previous_metadata_bytes=active_metadata_bytes,

                        active_version=?,
                        active_codec=?,
                        active_data_path=?,
                        active_metadata_path=?,
                        active_payload_bytes=?,
                        active_metadata_bytes=?,
                        migration_state='committed_pending_cleanup',
                        updated_at_ns=?
                    WHERE chunk_id=?
                    """,
                    (
                        to_version,
                        target_codec,
                        relative_path(self.root, published.data_path),
                        relative_path(self.root, published.metadata_path),
                        int(published.payload_size_bytes),
                        int(published.metadata_size_bytes),
                        time.time_ns(),
                        chunk_id,
                    ),
                )
                self.connection.execute(
                    """
                    UPDATE migration_log
                    SET state='catalog_committed', updated_at_ns=?
                    WHERE migration_id=?
                    """,
                    (time.time_ns(), migration_id),
                )

            self.invalidate_handle(chunk_id)

            if fault_stage == "after_catalog_commit_corrupt_active":
                with open(final_data, "r+b") as file:
                    file.truncate(max(0, final_data.stat().st_size // 3))
                    file.flush()
                    os.fsync(file.fileno())
                raise InjectedMigrationFault(fault_stage)

            refreshed = self._row(chunk_id)
            active_after_commit = self.active_stored(chunk_id)
            if not self._validate_stored(
                active_after_commit,
                str(refreshed["raw_checksum"]),
                expected_values=values,
            ):
                raise AssertionError(
                    f"Post-commit active validation failed: {chunk_id}"
                )

            previous = self.previous_stored(chunk_id)
            self._delete_stored_files(previous)
            with immediate_transaction(self.connection):
                self.connection.execute(
                    """
                    UPDATE chunks
                    SET
                        previous_version=NULL,
                        previous_codec=NULL,
                        previous_data_path=NULL,
                        previous_metadata_path=NULL,
                        previous_payload_bytes=NULL,
                        previous_metadata_bytes=NULL,
                        migration_state='idle',
                        updated_at_ns=?
                    WHERE chunk_id=?
                    """,
                    (time.time_ns(), chunk_id),
                )
                self.connection.execute(
                    """
                    UPDATE migration_log
                    SET state='completed',
                        completed_at_ns=?,
                        updated_at_ns=?
                    WHERE migration_id=?
                    """,
                    (
                        time.time_ns(),
                        time.time_ns(),
                        migration_id,
                    ),
                )

            elapsed_ns = time.perf_counter_ns() - start_ns
            return {
                "chunk_id": chunk_id,
                "migration_id": migration_id,
                "from_codec": current_codec,
                "target_codec": target_codec,
                "from_version": current_version,
                "to_version": to_version,
                "migration_time_ns": int(elapsed_ns),
                "rewrite_bytes": int(published.stored_size_bytes),
                "no_op": False,
            }

    def _mark_log_recovered(
        self,
        migration_id: str,
        state: str,
        action: str,
        error_text: Optional[str] = None,
    ) -> None:
        self.connection.execute(
            """
            UPDATE migration_log
            SET state=?,
                recovery_action=?,
                error_text=?,
                completed_at_ns=?,
                updated_at_ns=?
            WHERE migration_id=?
            """,
            (
                state,
                action,
                error_text,
                time.time_ns(),
                time.time_ns(),
                migration_id,
            ),
        )

    def recover(self) -> Dict[str, int]:
        with self._lock:
            self.invalidate_all_handles()
            actions = {
                "completed_cleanup": 0,
                "rolled_back": 0,
                "aborted_uncommitted": 0,
                "orphan_files_removed": 0,
            }

            rows = self.connection.execute(
                "SELECT * FROM chunks ORDER BY chunk_id"
            ).fetchall()
            for row in rows:
                chunk_id = str(row["chunk_id"])
                state = str(row["migration_state"])
                raw_checksum = str(row["raw_checksum"])

                if state == "committed_pending_cleanup":
                    active = self.active_stored(chunk_id)
                    active_ok = self._validate_stored(active, raw_checksum)
                    if active_ok:
                        previous = self.previous_stored(chunk_id)
                        self._delete_stored_files(previous)
                        with immediate_transaction(self.connection):
                            self.connection.execute(
                                """
                                UPDATE chunks
                                SET
                                    previous_version=NULL,
                                    previous_codec=NULL,
                                    previous_data_path=NULL,
                                    previous_metadata_path=NULL,
                                    previous_payload_bytes=NULL,
                                    previous_metadata_bytes=NULL,
                                    migration_state='idle',
                                    updated_at_ns=?
                                WHERE chunk_id=?
                                """,
                                (time.time_ns(), chunk_id),
                            )
                            pending = self.connection.execute(
                                """
                                SELECT migration_id
                                FROM migration_log
                                WHERE chunk_id=?
                                  AND state='catalog_committed'
                                ORDER BY started_at_ns DESC
                                LIMIT 1
                                """,
                                (chunk_id,),
                            ).fetchone()
                            if pending is not None:
                                self._mark_log_recovered(
                                    str(pending["migration_id"]),
                                    "completed",
                                    "recovery_completed_cleanup",
                                )
                        actions["completed_cleanup"] += 1
                    else:
                        previous = self.previous_stored(chunk_id)
                        previous_ok = (
                            previous is not None
                            and self._validate_stored(previous, raw_checksum)
                        )
                        if not previous_ok:
                            raise RuntimeError(
                                f"Neither active nor previous version is valid: {chunk_id}"
                            )

                        broken_active = active
                        with immediate_transaction(self.connection):
                            self.connection.execute(
                                """
                                UPDATE chunks
                                SET
                                    active_version=previous_version,
                                    active_codec=previous_codec,
                                    active_data_path=previous_data_path,
                                    active_metadata_path=previous_metadata_path,
                                    active_payload_bytes=previous_payload_bytes,
                                    active_metadata_bytes=previous_metadata_bytes,

                                    previous_version=NULL,
                                    previous_codec=NULL,
                                    previous_data_path=NULL,
                                    previous_metadata_path=NULL,
                                    previous_payload_bytes=NULL,
                                    previous_metadata_bytes=NULL,
                                    migration_state='idle',
                                    updated_at_ns=?
                                WHERE chunk_id=?
                                """,
                                (time.time_ns(), chunk_id),
                            )
                            pending = self.connection.execute(
                                """
                                SELECT migration_id
                                FROM migration_log
                                WHERE chunk_id=?
                                  AND state='catalog_committed'
                                ORDER BY started_at_ns DESC
                                LIMIT 1
                                """,
                                (chunk_id,),
                            ).fetchone()
                            if pending is not None:
                                self._mark_log_recovered(
                                    str(pending["migration_id"]),
                                    "rolled_back",
                                    "recovery_rolled_back_corrupt_active",
                                )
                        self._delete_stored_files(broken_active)
                        actions["rolled_back"] += 1

                elif state != "idle":
                    pending_rows = self.connection.execute(
                        """
                        SELECT *
                        FROM migration_log
                        WHERE chunk_id=?
                          AND state NOT IN ('completed','rolled_back','aborted')
                        ORDER BY started_at_ns
                        """,
                        (chunk_id,),
                    ).fetchall()
                    for migration in pending_rows:
                        for column in (
                            "temporary_data_path",
                            "temporary_metadata_path",
                            "final_data_path",
                            "final_metadata_path",
                        ):
                            path = absolute_path(self.root, migration[column])
                            if path is not None:
                                safe_unlink(path)
                        with immediate_transaction(self.connection):
                            self._mark_log_recovered(
                                str(migration["migration_id"]),
                                "aborted",
                                "recovery_removed_uncommitted_target",
                            )
                    self.connection.execute(
                        """
                        UPDATE chunks
                        SET migration_state='idle', updated_at_ns=?
                        WHERE chunk_id=?
                        """,
                        (time.time_ns(), chunk_id),
                    )
                    actions["aborted_uncommitted"] += 1

            referenced: set[Path] = set()
            for row in self.connection.execute(
                """
                SELECT
                    active_data_path, active_metadata_path,
                    previous_data_path, previous_metadata_path
                FROM chunks
                """
            ).fetchall():
                for value in row:
                    path = absolute_path(self.root, value)
                    if path is not None:
                        referenced.add(path.resolve())

            for directory in (self.chunk_root, self.staging_root):
                if not directory.exists():
                    continue
                for path in directory.rglob("*"):
                    if not path.is_file():
                        continue
                    if path.resolve() not in referenced:
                        safe_unlink(path)
                        actions["orphan_files_removed"] += 1

            return actions

    def integrity_check(self) -> Dict[str, Any]:
        with self._lock:
            self.invalidate_all_handles()
            rows = self.connection.execute(
                "SELECT * FROM chunks ORDER BY chunk_id"
            ).fetchall()
            invalid: List[str] = []
            non_idle: List[str] = []
            multiple_payload_chunks: List[str] = []
            active_bytes = 0

            for row in rows:
                chunk_id = str(row["chunk_id"])
                if str(row["migration_state"]) != "idle":
                    non_idle.append(chunk_id)
                active = self.active_stored(chunk_id)
                if not self._validate_stored(
                    active,
                    str(row["raw_checksum"]),
                ):
                    invalid.append(chunk_id)
                active_bytes += int(active.stored_size_bytes)

                directory = self.chunk_root / chunk_id
                payloads = list(directory.glob("*.bin"))
                metadata_files = list(directory.glob("*.meta.bin"))
                actual_payloads = [
                    path
                    for path in payloads
                    if not path.name.endswith(".meta.bin")
                ]
                if len(actual_payloads) != 1:
                    multiple_payload_chunks.append(chunk_id)

            temporary_files = [
                str(path)
                for path in self.staging_root.rglob("*")
                if path.is_file()
            ]

            return {
                "chunk_count": len(rows),
                "invalid_chunks": invalid,
                "non_idle_chunks": non_idle,
                "chunks_without_exactly_one_payload": multiple_payload_chunks,
                "temporary_file_count": len(temporary_files),
                "active_stored_bytes": int(active_bytes),
                "pass": bool(
                    len(rows) > 0
                    and not invalid
                    and not non_idle
                    and not multiple_payload_chunks
                    and not temporary_files
                ),
            }

    def catalog_dataframe(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM chunks ORDER BY chunk_id",
            self.connection,
        )

    def migration_log_dataframe(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM migration_log ORDER BY started_at_ns",
            self.connection,
        )


def load_frozen_chunks(
    project_dir: Path,
    base: Any,
) -> Tuple[List[FrozenChunk], Dict[str, Any]]:
    a0_dir = project_dir / A0_RESULT_DIR_NAME
    manifest_path = a0_dir / SOURCE_MANIFEST_NAME
    protocol_hash_path = a0_dir / SOURCE_PROTOCOL_HASH_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing frozen A0 manifest: {manifest_path}"
        )
    if not protocol_hash_path.exists():
        raise FileNotFoundError(
            f"Missing frozen A0 protocol hash: {protocol_hash_path}"
        )

    manifest = pd.read_csv(manifest_path)
    selected = manifest[
        manifest["chunk_size"] == CHUNK_SIZE
    ].copy()
    selected = selected.sort_values(
        ["original_series_index", "chunk_id"]
    ).reset_index(drop=True)
    if len(selected) != EXPECTED_CHUNKS:
        raise AssertionError(
            f"Expected {EXPECTED_CHUNKS} frozen {CHUNK_SIZE}-value chunks, "
            f"found {len(selected)}"
        )
    if selected["original_series_index"].nunique() != EXPECTED_CHUNKS:
        raise AssertionError("Frozen chunks are not one-per-series")

    pems04_dir, _ = base.default_data_dirs()
    data_file = base.auto_detect_data_file(
        pems04_dir,
        "PEMS04_B1_SINGLE_ACTIVE",
    )
    with np.load(data_file, allow_pickle=False) as archive:
        if "data" in archive.files:
            raw = np.asarray(archive["data"])
        else:
            candidates = [
                np.asarray(archive[key])
                for key in archive.files
                if np.asarray(archive[key]).ndim >= 2
            ]
            if not candidates:
                raise ValueError("No usable PEMS04 array")
            raw = candidates[0]

    if raw.ndim == 3:
        matrix = np.asarray(raw[:, :, 0], dtype=np.float32)
    elif raw.ndim == 2:
        matrix = np.asarray(raw, dtype=np.float32)
    else:
        raise ValueError(f"Unexpected PEMS04 shape: {raw.shape}")

    chunks: List[FrozenChunk] = []
    for row in selected.itertuples(index=False):
        series_index = int(row.original_series_index)
        start = int(row.chunk_start)
        size = int(row.chunk_size)
        values = np.ascontiguousarray(
            matrix[start : start + size, series_index],
            dtype=np.float32,
        )
        if values.size != CHUNK_SIZE:
            raise AssertionError(
                f"Incomplete frozen chunk: series={series_index}, start={start}"
            )
        if not np.isfinite(values).all():
            raise AssertionError(
                f"Non-finite values in frozen chunk: series={series_index}"
            )
        chunks.append(
            FrozenChunk(
                chunk_id=str(row.chunk_id),
                series_index=series_index,
                chunk_start=start,
                chunk_size=size,
                values=values,
                raw_checksum=raw_array_sha256(values),
            )
        )

    metadata = {
        "data_file": str(data_file),
        "data_file_sha256": sha256_file(data_file),
        "a0_protocol_sha256": protocol_hash_path.read_text(
            encoding="utf-8"
        ).strip(),
        "manifest_sha256": sha256_file(manifest_path),
        "chunk_count": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "series_indices": [chunk.series_index for chunk in chunks],
    }
    return chunks, metadata


def build_query(
    base: Any,
    chunk: FrozenChunk,
    query_index: int,
) -> Any:
    rng = np.random.default_rng(
        stable_seed(SEED, "query", query_index, chunk.chunk_id)
    )
    query_type = QUERY_TYPES[
        stable_seed(SEED, "query-type", query_index) % len(QUERY_TYPES)
    ]
    n = int(chunk.values.size)

    if query_type == "point":
        start = int(rng.integers(0, n))
        end = start + 1
        query_class = "point"
        threshold = None
    else:
        fraction_choices = (0.001, 0.01, 0.10, 0.50, 1.00)
        fraction = fraction_choices[
            stable_seed(SEED, "fraction", query_index)
            % len(fraction_choices)
        ]
        length = max(1, min(n, int(round(n * fraction))))
        start = (
            0
            if length == n
            else int(rng.integers(0, n - length + 1))
        )
        end = start + length
        if query_type == "threshold_count":
            quantile = (0.50, 0.90, 0.99)[
                stable_seed(SEED, "quantile", query_index) % 3
            ]
            threshold = float(np.quantile(chunk.values, quantile))
            query_class = "threshold"
        else:
            threshold = None
            query_class = base.query_class(query_type, fraction)

    return base.QuerySpec(
        query_id=f"B1:{query_index}",
        query_type=query_type,
        query_class=query_class,
        start=start,
        end=end,
        range_fraction=(end - start) / n,
        target_fraction=(end - start) / n,
        threshold=threshold,
        threshold_quantile=None,
    )


def expected_query_result(
    base: Any,
    chunk: FrozenChunk,
    query: Any,
) -> Any:
    return base.execute_on_array(chunk.values, query)


def query_result_equal(base: Any, actual: Any, expected: Any) -> bool:
    return bool(base.results_equal(actual, expected))


def deterministic_target_codec(
    current_codec: str,
    index: int,
    token: str,
) -> str:
    choices = [codec for codec in CODECS if codec != current_codec]
    return choices[
        stable_seed(SEED, token, index, current_codec) % len(choices)
    ]


def initialize_main_store(
    store: SingleActiveChunkStore,
    chunks: Sequence[FrozenChunk],
    freeze_hash: str,
    source_metadata: Mapping[str, Any],
) -> None:
    existing_hash = store.get_metadata("experiment_freeze_sha256")
    if existing_hash is not None and existing_hash != freeze_hash:
        raise RuntimeError(
            "The existing B1 store belongs to another experiment freeze.\n"
            f"Existing: {existing_hash}\n"
            f"Current:  {freeze_hash}"
        )

    store.set_metadata("experiment_freeze_sha256", freeze_hash)
    store.set_metadata("version", VERSION)
    store.set_metadata("source_metadata", dict(source_metadata))
    store.set_metadata("codec_names", list(CODECS))

    for index, chunk in enumerate(chunks, start=1):
        store.create_chunk(
            chunk.chunk_id,
            chunk.series_index,
            chunk.values,
            codec=INITIAL_CODEC,
        )
        if index == 1 or index == len(chunks) or index % 4 == 0:
            print(f"[store initialization] {index}/{len(chunks)} chunks")

    if store.chunk_count() != len(chunks):
        raise AssertionError("Main store chunk count mismatch")


def run_query_correctness_test(
    store: SingleActiveChunkStore,
    base: Any,
    chunks: Sequence[FrozenChunk],
) -> None:
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    start_index = int(
        store.connection.execute(
            "SELECT COALESCE(MAX(query_index), -1) + 1 AS next_index "
            "FROM query_test_records"
        ).fetchone()["next_index"]
    )
    if start_index >= QUERY_TEST_COUNT:
        print("[query test resume] already complete")
        return

    pending: List[Tuple[Any, ...]] = []
    for query_index in range(start_index, QUERY_TEST_COUNT):
        chunk = chunks[
            stable_seed(SEED, "query-chunk", query_index) % len(chunks)
        ]
        query = build_query(base, chunk, query_index)
        expected = expected_query_result(base, chunk, query)
        actual, latency_ns, stats, row = store.execute_query(
            chunk.chunk_id,
            query,
        )
        result_ok = query_result_equal(base, actual, expected)
        if not result_ok:
            raise AssertionError(
                f"Query mismatch at index {query_index}: "
                f"chunk={chunk.chunk_id}, type={query.query_type}, "
                f"actual={actual}, expected={expected}"
            )
        pending.append(
            (
                query_index,
                chunk.chunk_id,
                str(row["active_codec"]),
                int(row["active_version"]),
                query.query_type,
                query.query_class,
                int(query.start),
                int(query.end),
                query.threshold,
                int(latency_ns),
                int(stats.total_logical_bytes),
                1,
            )
        )
        if len(pending) >= QUERY_BATCH_COMMIT:
            with immediate_transaction(store.connection):
                store.connection.executemany(
                    """
                    INSERT OR REPLACE INTO query_test_records(
                        query_index, chunk_id,
                        active_codec, active_version,
                        query_type, query_class,
                        start_index, end_index, threshold,
                        latency_ns, logical_bytes, result_ok
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    pending,
                )
            pending.clear()

        completed = query_index + 1
        if completed % QUERY_PROGRESS_PRINT == 0:
            print(
                f"[query correctness] {completed}/{QUERY_TEST_COUNT}"
            )

    if pending:
        with immediate_transaction(store.connection):
            store.connection.executemany(
                """
                INSERT OR REPLACE INTO query_test_records(
                    query_index, chunk_id,
                    active_codec, active_version,
                    query_type, query_class,
                    start_index, end_index, threshold,
                    latency_ns, logical_bytes, result_ok
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                pending,
            )


def run_migration_correctness_test(
    store: SingleActiveChunkStore,
    base: Any,
    chunks: Sequence[FrozenChunk],
) -> None:
    start_index = int(
        store.connection.execute(
            "SELECT COALESCE(MAX(migration_index), -1) + 1 AS next_index "
            "FROM migration_test_records"
        ).fetchone()["next_index"]
    )
    if start_index >= MIGRATION_TEST_COUNT:
        print("[migration test resume] already complete")
        return

    for migration_index in range(start_index, MIGRATION_TEST_COUNT):
        chunk = chunks[
            stable_seed(SEED, "migration-chunk", migration_index)
            % len(chunks)
        ]
        before = store._row(chunk.chunk_id)
        target_codec = deterministic_target_codec(
            str(before["active_codec"]),
            migration_index,
            "migration-target",
        )

        result = store.migrate(chunk.chunk_id, target_codec)
        after = store._row(chunk.chunk_id)
        checksum_ok = (
            raw_array_sha256(store.read_all(chunk.chunk_id))
            == chunk.raw_checksum
        )

        query = build_query(
            base,
            chunk,
            1_000_000 + migration_index,
        )
        expected = expected_query_result(base, chunk, query)
        actual, _, _, _ = store.execute_query(chunk.chunk_id, query)
        query_ok = query_result_equal(base, actual, expected)

        if not checksum_ok or not query_ok:
            raise AssertionError(
                f"Post-migration validation failed: "
                f"index={migration_index}, chunk={chunk.chunk_id}"
            )
        if str(after["active_codec"]) != target_codec:
            raise AssertionError("Target codec was not activated")
        if str(after["migration_state"]) != "idle":
            raise AssertionError("Migration did not return chunk to idle")

        with immediate_transaction(store.connection):
            store.connection.execute(
                """
                INSERT OR REPLACE INTO migration_test_records(
                    migration_index, chunk_id,
                    from_codec, target_codec,
                    from_version, to_version,
                    migration_time_ns, rewrite_bytes,
                    checksum_ok, query_spotcheck_ok
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    migration_index,
                    chunk.chunk_id,
                    str(result["from_codec"]),
                    str(result["target_codec"]),
                    int(result["from_version"]),
                    int(result["to_version"]),
                    int(result["migration_time_ns"]),
                    int(result["rewrite_bytes"]),
                    int(checksum_ok),
                    int(query_ok),
                ),
            )

        completed = migration_index + 1
        if completed % MIGRATION_PROGRESS_PRINT == 0:
            print(
                f"[migration correctness] "
                f"{completed}/{MIGRATION_TEST_COUNT}"
            )


def count_active_payload_files(store_root: Path, chunk_id: str) -> int:
    directory = store_root / "chunks" / chunk_id
    if not directory.exists():
        return 0
    return len(
        [
            path
            for path in directory.glob("*.bin")
            if not path.name.endswith(".meta.bin")
        ]
    )


def temporary_file_count(store_root: Path) -> int:
    staging = store_root / "staging"
    if not staging.exists():
        return 0
    return sum(1 for path in staging.rglob("*") if path.is_file())


def run_one_fault_trial(
    base: Any,
    trial_root: Path,
    chunk: FrozenChunk,
    stage: str,
    trial_index: int,
) -> Dict[str, Any]:
    if trial_root.exists():
        shutil.rmtree(trial_root)
    trial_root.mkdir(parents=True, exist_ok=True)

    start_ns = time.perf_counter_ns()
    target_codec = (
        "block_lz4_256"
        if stage == "after_payload_publish_before_metadata"
        else deterministic_target_codec(
            INITIAL_CODEC,
            trial_index,
            f"fault-target:{stage}",
        )
    )
    injected_observed = False

    store = SingleActiveChunkStore(
        trial_root,
        base,
        create=True,
        recover_on_open=True,
    )
    try:
        store.create_chunk(
            chunk.chunk_id,
            chunk.series_index,
            chunk.values,
            codec=INITIAL_CODEC,
        )
        try:
            store.migrate(
                chunk.chunk_id,
                target_codec,
                fault_stage=stage,
            )
        except InjectedMigrationFault as error:
            if error.stage != stage:
                raise
            injected_observed = True
        else:
            raise AssertionError(
                f"Fault was not injected at stage {stage}"
            )
    finally:
        store.close()

    recovered = SingleActiveChunkStore(
        trial_root,
        base,
        create=True,
        recover_on_open=True,
    )
    try:
        row = recovered._row(chunk.chunk_id)
        values = recovered.read_all(chunk.chunk_id)
        checksum_ok = raw_array_sha256(values) == chunk.raw_checksum
        query = build_query(
            base,
            chunk,
            2_000_000
            + FAULT_STAGES.index(stage) * FAULT_TRIALS_PER_STAGE
            + trial_index,
        )
        expected = expected_query_result(base, chunk, query)
        actual, _, _, _ = recovered.execute_query(chunk.chunk_id, query)
        query_ok = query_result_equal(base, actual, expected)
        catalog_idle = str(row["migration_state"]) == "idle"
        payload_count = count_active_payload_files(
            trial_root,
            chunk.chunk_id,
        )
        no_temp = temporary_file_count(trial_root) == 0

        if stage == "after_catalog_commit_corrupt_active":
            expected_codec = INITIAL_CODEC
        else:
            expected_codec = INITIAL_CODEC

        observed_codec = str(row["active_codec"])
        outcome_ok = observed_codec == expected_codec

        result = {
            "stage": stage,
            "trial_index": trial_index,
            "chunk_id": chunk.chunk_id,
            "target_codec": target_codec,
            "injected_fault_observed": int(injected_observed),
            "recovered_checksum_ok": int(checksum_ok),
            "recovered_query_ok": int(query_ok),
            "catalog_idle": int(catalog_idle),
            "one_active_payload": int(payload_count == 1),
            "no_temporary_files": int(no_temp),
            "expected_outcome": (
                "rollback_to_previous_active"
                if stage == "after_catalog_commit_corrupt_active"
                else "retain_original_active"
            ),
            "observed_active_codec": observed_codec,
            "elapsed_ns": int(time.perf_counter_ns() - start_ns),
            "outcome_ok": bool(
                injected_observed
                and checksum_ok
                and query_ok
                and catalog_idle
                and payload_count == 1
                and no_temp
                and outcome_ok
            ),
        }
        if not result["outcome_ok"]:
            raise AssertionError(
                f"Fault recovery failed: {json.dumps(result, ensure_ascii=False)}"
            )
        return result
    finally:
        recovered.close()


def run_fault_injection_test(
    metrics_connection: sqlite3.Connection,
    base: Any,
    chunks: Sequence[FrozenChunk],
    output_dir: Path,
) -> None:
    fault_root = output_dir / "_fault_trial_current"
    for stage_index, stage in enumerate(FAULT_STAGES):
        completed_rows = metrics_connection.execute(
            "SELECT COUNT(*) AS n FROM fault_test_records WHERE stage=?",
            (stage,),
        ).fetchone()
        start_trial = int(completed_rows["n"])
        if start_trial >= FAULT_TRIALS_PER_STAGE:
            print(f"[fault resume] {stage} already complete")
            continue

        for trial_index in range(start_trial, FAULT_TRIALS_PER_STAGE):
            chunk = chunks[
                stable_seed(
                    SEED,
                    "fault-chunk",
                    stage,
                    trial_index,
                )
                % len(chunks)
            ]
            result = run_one_fault_trial(
                base,
                fault_root,
                chunk,
                stage,
                trial_index,
            )
            with immediate_transaction(metrics_connection):
                metrics_connection.execute(
                    """
                    INSERT OR REPLACE INTO fault_test_records(
                        stage, trial_index, chunk_id, target_codec,
                        injected_fault_observed,
                        recovered_checksum_ok,
                        recovered_query_ok,
                        catalog_idle,
                        one_active_payload,
                        no_temporary_files,
                        expected_outcome,
                        observed_active_codec,
                        elapsed_ns
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result["stage"],
                        result["trial_index"],
                        result["chunk_id"],
                        result["target_codec"],
                        result["injected_fault_observed"],
                        result["recovered_checksum_ok"],
                        result["recovered_query_ok"],
                        result["catalog_idle"],
                        result["one_active_payload"],
                        result["no_temporary_files"],
                        result["expected_outcome"],
                        result["observed_active_codec"],
                        result["elapsed_ns"],
                    ),
                )

            completed = trial_index + 1
            if completed % FAULT_PROGRESS_PRINT == 0:
                print(
                    f"[fault injection] {stage}: "
                    f"{completed}/{FAULT_TRIALS_PER_STAGE}"
                )

    shutil.rmtree(fault_root, ignore_errors=True)


def export_results(
    store: SingleActiveChunkStore,
    output_dir: Path,
) -> Dict[str, Any]:
    query_df = pd.read_sql_query(
        "SELECT * FROM query_test_records ORDER BY query_index",
        store.connection,
    )
    migration_df = pd.read_sql_query(
        "SELECT * FROM migration_test_records ORDER BY migration_index",
        store.connection,
    )
    fault_df = pd.read_sql_query(
        "SELECT * FROM fault_test_records ORDER BY stage, trial_index",
        store.connection,
    )
    catalog_df = store.catalog_dataframe()
    migration_log_df = store.migration_log_dataframe()

    write_dataframe_atomic(output_dir / "query_test_records.csv", query_df)
    write_dataframe_atomic(
        output_dir / "migration_test_records.csv",
        migration_df,
    )
    write_dataframe_atomic(
        output_dir / "fault_injection_records.csv",
        fault_df,
    )
    write_dataframe_atomic(output_dir / "catalog_final.csv", catalog_df)
    write_dataframe_atomic(
        output_dir / "migration_log.csv",
        migration_log_df,
    )

    query_summary = (
        query_df.groupby(["query_type", "active_codec"], as_index=False)
        .agg(
            n_queries=("latency_ns", "size"),
            median_latency_ns=("latency_ns", "median"),
            mean_latency_ns=("latency_ns", "mean"),
            p95_latency_ns=(
                "latency_ns",
                lambda values: float(np.quantile(values, 0.95)),
            ),
            p99_latency_ns=(
                "latency_ns",
                lambda values: float(np.quantile(values, 0.99)),
            ),
            median_logical_bytes=("logical_bytes", "median"),
        )
    )
    write_dataframe_atomic(
        output_dir / "query_latency_summary.csv",
        query_summary,
    )

    migration_summary = (
        migration_df.groupby(["from_codec", "target_codec"], as_index=False)
        .agg(
            n_migrations=("migration_time_ns", "size"),
            median_migration_ns=("migration_time_ns", "median"),
            p95_migration_ns=(
                "migration_time_ns",
                lambda values: float(np.quantile(values, 0.95)),
            ),
            median_rewrite_bytes=("rewrite_bytes", "median"),
        )
    )
    write_dataframe_atomic(
        output_dir / "migration_latency_summary.csv",
        migration_summary,
    )

    fault_summary = (
        fault_df.groupby("stage", as_index=False)
        .agg(
            n_trials=("trial_index", "size"),
            injected_faults=("injected_fault_observed", "sum"),
            checksum_successes=("recovered_checksum_ok", "sum"),
            query_successes=("recovered_query_ok", "sum"),
            idle_successes=("catalog_idle", "sum"),
            one_payload_successes=("one_active_payload", "sum"),
            no_temp_successes=("no_temporary_files", "sum"),
            median_recovery_trial_ns=("elapsed_ns", "median"),
        )
    )
    write_dataframe_atomic(
        output_dir / "fault_injection_summary.csv",
        fault_summary,
    )

    integrity = store.integrity_check()
    counts = {
        "query_records": int(len(query_df)),
        "query_correct": int(query_df["result_ok"].sum()),
        "migration_records": int(len(migration_df)),
        "migration_checksum_successes": int(
            migration_df["checksum_ok"].sum()
        ),
        "migration_query_successes": int(
            migration_df["query_spotcheck_ok"].sum()
        ),
        "fault_records": int(len(fault_df)),
        "fault_all_columns_success": int(
            (
                (fault_df["injected_fault_observed"] == 1)
                & (fault_df["recovered_checksum_ok"] == 1)
                & (fault_df["recovered_query_ok"] == 1)
                & (fault_df["catalog_idle"] == 1)
                & (fault_df["one_active_payload"] == 1)
                & (fault_df["no_temporary_files"] == 1)
            ).sum()
        ),
    }
    return {
        "integrity": integrity,
        "counts": counts,
        "query_summary_rows": int(len(query_summary)),
        "migration_summary_rows": int(len(migration_summary)),
        "fault_summary_rows": int(len(fault_summary)),
    }


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    e0c_path = project_dir / E0C_SCRIPT_NAME
    output_dir = project_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 112)
    print("RePressDB B1 v0.9.0 — single-active store and crash recovery")
    print(f"Target journal: {TARGET_JOURNAL}")
    print("=" * 112)

    base = load_module(
        e0c_path,
        "repressdb_e0c_for_b1_v090",
    )
    chunks, source_metadata = load_frozen_chunks(project_dir, base)

    protocol = {
        "version": VERSION,
        "target_journal": TARGET_JOURNAL,
        "study": "B1 single-active recoverable chunk-store prototype",
        "source": source_metadata,
        "store_design": {
            "catalog": "SQLite WAL",
            "sqlite_synchronous": "FULL",
            "normal_active_representations_per_chunk": 1,
            "migration_protocol": [
                "record migration intent",
                "decode active representation",
                "encode target into staging files",
                "fsync encoded files",
                "decode and checksum-validate target",
                "publish versioned target files",
                "atomically switch active catalog pointer",
                "validate committed active version",
                "delete previous version",
                "return catalog state to idle",
            ],
            "checksum": "SHA-256 of canonical contiguous float32 bytes",
            "query_mode": "warm_slicedmeta with version-aware handle cache",
        },
        "acceptance_tests": {
            "chunks": EXPECTED_CHUNKS,
            "chunk_size": CHUNK_SIZE,
            "random_queries": QUERY_TEST_COUNT,
            "successful_migrations": MIGRATION_TEST_COUNT,
            "fault_stages": list(FAULT_STAGES),
            "fault_trials_per_stage": FAULT_TRIALS_PER_STAGE,
            "total_fault_trials": (
                len(FAULT_STAGES) * FAULT_TRIALS_PER_STAGE
            ),
        },
        "codecs": list(CODECS),
        "initial_codec": INITIAL_CODEC,
        "resume": (
            "SQLite test records and persistent store state; "
            "rerun starts from first incomplete test index"
        ),
    }
    write_json_atomic(output_dir / "b1_protocol.json", protocol)

    protocol_digest = hashlib.sha256()
    protocol_digest.update(
        source_metadata["a0_protocol_sha256"].encode("ascii")
    )
    protocol_digest.update(
        sha256_file(output_dir / "b1_protocol.json").encode("ascii")
    )
    b1_protocol_hash = protocol_digest.hexdigest()
    (output_dir / "b1_protocol_sha256.txt").write_text(
        b1_protocol_hash + "\n",
        encoding="utf-8",
    )

    freeze_material = {
        "b1_protocol_sha256": b1_protocol_hash,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "e0c_script_sha256": sha256_file(e0c_path),
        "source_manifest_sha256": source_metadata["manifest_sha256"],
        "source_data_sha256": source_metadata["data_file_sha256"],
    }
    write_json_atomic(
        output_dir / "b1_experiment_freeze_material.json",
        freeze_material,
    )
    freeze_hash = sha256_file(
        output_dir / "b1_experiment_freeze_material.json"
    )
    (output_dir / "b1_experiment_freeze_sha256.txt").write_text(
        freeze_hash + "\n",
        encoding="utf-8",
    )

    store_root = output_dir / "store"
    start_wall_ns = time.perf_counter_ns()

    with SingleActiveChunkStore(
        store_root,
        base,
        create=True,
        recover_on_open=True,
    ) as store:
        initialize_main_store(
            store,
            chunks,
            freeze_hash,
            source_metadata,
        )

        initial_integrity = store.integrity_check()
        if not initial_integrity["pass"]:
            raise AssertionError(
                f"Initial store integrity failed: {initial_integrity}"
            )
        write_json_atomic(
            output_dir / "initial_store_integrity.json",
            initial_integrity,
        )

        print("-" * 112)
        print(
            f"[B1] query correctness: {QUERY_TEST_COUNT:,} operations"
        )
        run_query_correctness_test(store, base, chunks)

        print("-" * 112)
        print(
            f"[B1] migration correctness: "
            f"{MIGRATION_TEST_COUNT:,} operations"
        )
        run_migration_correctness_test(store, base, chunks)

        print("-" * 112)
        print(
            f"[B1] crash recovery: "
            f"{len(FAULT_STAGES)} stages × "
            f"{FAULT_TRIALS_PER_STAGE} trials"
        )
        run_fault_injection_test(
            store.connection,
            base,
            chunks,
            output_dir,
        )

        final_recovery_actions = store.recover()
        result_summary = export_results(store, output_dir)
        result_summary["final_recovery_actions"] = final_recovery_actions

        checks = {
            "B1_1_exactly_32_frozen_chunks": bool(
                result_summary["integrity"]["chunk_count"]
                == EXPECTED_CHUNKS
            ),
            "B1_2_exactly_one_active_payload_per_chunk": bool(
                not result_summary["integrity"][
                    "chunks_without_exactly_one_payload"
                ]
            ),
            "B1_3_no_invalid_active_chunks": bool(
                not result_summary["integrity"]["invalid_chunks"]
            ),
            "B1_4_catalog_all_idle": bool(
                not result_summary["integrity"]["non_idle_chunks"]
            ),
            "B1_5_no_staging_files": bool(
                result_summary["integrity"]["temporary_file_count"] == 0
            ),
            "B1_6_all_100000_queries_correct": bool(
                result_summary["counts"]["query_records"]
                == QUERY_TEST_COUNT
                and result_summary["counts"]["query_correct"]
                == QUERY_TEST_COUNT
            ),
            "B1_7_all_1000_migrations_checksum_correct": bool(
                result_summary["counts"]["migration_records"]
                == MIGRATION_TEST_COUNT
                and result_summary["counts"][
                    "migration_checksum_successes"
                ]
                == MIGRATION_TEST_COUNT
            ),
            "B1_8_all_1000_migration_query_spotchecks_correct": bool(
                result_summary["counts"]["migration_records"]
                == MIGRATION_TEST_COUNT
                and result_summary["counts"][
                    "migration_query_successes"
                ]
                == MIGRATION_TEST_COUNT
            ),
            "B1_9_all_600_fault_recoveries_correct": bool(
                result_summary["counts"]["fault_records"]
                == len(FAULT_STAGES) * FAULT_TRIALS_PER_STAGE
                and result_summary["counts"][
                    "fault_all_columns_success"
                ]
                == len(FAULT_STAGES) * FAULT_TRIALS_PER_STAGE
            ),
        }
        checks["all_nine"] = bool(all(checks.values()))

        elapsed_seconds = (
            time.perf_counter_ns() - start_wall_ns
        ) / 1e9
        assessment = {
            "version": VERSION,
            "target_journal": TARGET_JOURNAL,
            "status": (
                "B1 complete"
                if checks["all_nine"]
                else "B1 completed with failed acceptance checks"
            ),
            "b1_protocol_sha256": b1_protocol_hash,
            "b1_experiment_freeze_sha256": freeze_hash,
            "source_metadata": source_metadata,
            "acceptance_checks": checks,
            "result_summary": result_summary,
            "elapsed_seconds_this_invocation": elapsed_seconds,
            "interpretation": (
                "A passing B1 establishes a standalone file-backed "
                "single-active-representation prototype with transactional "
                "catalog switching and tested crash recovery. It does not "
                "yet establish GB-scale throughput or concurrent-query "
                "performance; those belong to B2/B3."
            ),
            "next_gate": (
                "Proceed to B2 full-BDG2 and GB-scale experiments."
                if checks["all_nine"]
                else (
                    "Stop and diagnose failed B1 checks before any "
                    "scale or concurrency experiment."
                )
            ),
        }
        write_json_atomic(
            output_dir / "b1_assessment.json",
            assessment,
        )
        write_json_atomic(
            project_dir / "B1_SINGLE_ACTIVE_STORE_ASSESSMENT_v090.json",
            assessment,
        )

        run_config = {
            "version": VERSION,
            "target_journal": TARGET_JOURNAL,
            "b1_protocol_sha256": b1_protocol_hash,
            "b1_experiment_freeze_sha256": freeze_hash,
            "store_root": str(store_root),
            "catalog_path": str(store.db_path),
            "query_test_count": QUERY_TEST_COUNT,
            "migration_test_count": MIGRATION_TEST_COUNT,
            "fault_stages": list(FAULT_STAGES),
            "fault_trials_per_stage": FAULT_TRIALS_PER_STAGE,
            "source_metadata": source_metadata,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sqlite_version": sqlite3.sqlite_version,
                "timer": "time.perf_counter_ns",
            },
        }
        write_json_atomic(output_dir / "run_config.json", run_config)

    print("=" * 112)
    print("RePressDB B1 v0.9.0 finished")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(f"Elapsed this invocation: {elapsed_seconds / 60:.2f} minutes")
    print(f"Results: {output_dir.resolve()}")
    print("Upload:")
    print("  b1_assessment.json")
    print("  query_latency_summary.csv")
    print("  migration_latency_summary.csv")
    print("  fault_injection_summary.csv")
    print("  catalog_final.csv")
    print("  migration_log.csv")
    print("  b1_protocol.json")
    print("  b1_protocol_sha256.txt")
    print("  b1_experiment_freeze_sha256.txt")
    print("  run_config.json")
    print("=" * 112)
    return 0 if checks["all_nine"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
