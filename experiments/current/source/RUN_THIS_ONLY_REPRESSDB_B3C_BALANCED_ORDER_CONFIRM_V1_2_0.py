#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
RePressDB B3-C v1.2.0
Warm-up controlled, AB/BA order-balanced concurrency confirmation.

Target journal: Journal of Big Data.

DIRECT RUN
----------
Place this file beside:

  RUN_THIS_ONLY_REPRESSDB_B3_SCALE_CONCURRENCY_V1_1_0.py
  RUN_THIS_ONLY_REPRESSDB_B3_SCALE_CONCURRENCY_RECOVER_V1_1_1.py
  RUN_THIS_ONLY_REPRESSDB_B2_FULL_BDG2_SCALE_V1_0_0.py
  RUN_THIS_ONLY_REPRESSDB_B1_SINGLE_ACTIVE_STORE_V0_9_0.py
  RUN_THIS_ONLY_REPRESSDB_E0C_V0_2_0.py

Keep:

  results_b2_full_bdg2_scale_v100/
  results_b3_scale_concurrency_v110/

Open this file in PyCharm and click Run.

No command-line arguments are required.

PURPOSE
-------
B3 v1.1.1 established correctness and provided descriptive concurrency
measurements. However, every original pair ran foreground_only before
background_migration, so the second condition could inherit warmer operating
system file caches.

B3-C confirms the interference result under:

  * identical deterministic warm-up before every measured cell;
  * complete warm-up coverage of all 720 frozen real BDG2 templates;
  * independent copies of the same frozen base catalog for both modes;
  * six paired repeats per scale/thread combination;
  * three FG->BG pairs and three BG->FG pairs;
  * identical measured query workload within every pair;
  * independent NumPy truth validation for every warm-up and measured query.

UNCHANGED FROM B3
-----------------
  logical scales: 1, 5, 10 GiB
  query threads: 1, 4, 8
  measured queries per cell: 50,000
  background migrations per background cell: 50
  physical codecs
  query distribution
  copy-on-write migration
  bounded production migration log
  thread-local codec registry from v1.1.1

FORMAL WORKLOAD
---------------
  3 scales
  x 3 thread counts
  x 6 paired repeats
  x 2 modes
  = 108 measured cells

  measured queries:
      108 x 50,000 = 5,400,000

  warm-up queries:
      108 x 5,000 = 540,000

  total verified queries:
      5,940,000

  background migrations:
      54 background cells x 50 = 2,700

ORDER SCHEDULE
--------------
For every scale/thread combination:

  pair 0: FG -> BG
  pair 1: BG -> FG
  pair 2: FG -> BG
  pair 3: BG -> FG
  pair 4: FG -> BG
  pair 5: BG -> FG

The query and migration seeds do not depend on order position.

RESUME
------
Completed cells are stored transactionally in:

  results_b3c_balanced_order_v120/telemetry.sqlite3

An interrupted cell is rerun from its beginning. Completed cells are skipped.
Do not delete the output directory and do not modify this script after the
first run.
"""

from __future__ import annotations

import contextlib
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


VERSION = "1.2.0"
TARGET_JOURNAL = "Journal of Big Data"

B3_PARENT_SCRIPT_NAME = (
    "RUN_THIS_ONLY_REPRESSDB_B3_SCALE_CONCURRENCY_V1_1_0.py"
)
B3_REPAIR_SCRIPT_NAME = (
    "RUN_THIS_ONLY_REPRESSDB_B3_SCALE_CONCURRENCY_RECOVER_V1_1_1.py"
)
E0C_SCRIPT_NAME = "RUN_THIS_ONLY_REPRESSDB_E0C_V0_2_0.py"

B2_RESULT_DIR_NAME = "results_b2_full_bdg2_scale_v100"
B3_RESULT_DIR_NAME = "results_b3_scale_concurrency_v110"
OUTPUT_DIR_NAME = "results_b3c_balanced_order_v120"
SUMMARY_JSON_NAME = "B3C_BALANCED_ORDER_ASSESSMENT_v120.json"

CONFIRM_SEED = 20260727

SCALE_NAMES = (
    "logical_1GiB",
    "logical_5GiB",
    "logical_10GiB",
)
QUERY_THREADS = (1, 4, 8)
PAIR_REPEATS = 6
MODES = (
    "foreground_only",
    "background_migration",
)

WARMUP_QUERIES_PER_CELL = 5_000
MEASURED_QUERIES_PER_CELL = 50_000
BACKGROUND_MIGRATIONS_PER_CELL = 50
CELL_SETTLE_SECONDS = 0.50

BOOTSTRAP_REPLICATES = 20_000
COMPLETED_LOG_RETAIN = 32

RAW_CHUNK_VALUES = 8192

LOW_INTERFERENCE_THROUGHPUT_FLOOR = 0.90
LOW_INTERFERENCE_P99_CEILING = 1.25


def json_safe(value: Any) -> Any:
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


def stable_seed(*parts: Any) -> int:
    token = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(
        token.encode("utf-8")
    ).digest()
    return int.from_bytes(
        digest[:8],
        "little",
        signed=False,
    )


def load_module(path: Path, name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    specification = importlib.util.spec_from_file_location(
        name,
        path,
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(
        specification
    )
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


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
    result["total_files_bytes"] = int(
        sum(result.values())
    )
    return result


def checkpoint_database(
    connection: sqlite3.Connection,
) -> None:
    connection.execute(
        "PRAGMA wal_checkpoint(TRUNCATE)"
    )


def copy_catalog(
    source: Path,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    for candidate in (
        destination,
        Path(str(destination) + "-wal"),
        Path(str(destination) + "-shm"),
    ):
        candidate.unlink(missing_ok=True)
    shutil.copy2(source, destination)


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
            self._resource: Any = None
            if os.name != "nt":
                try:
                    import resource

                    self._resource = resource
                    self.method = "resource_ru_maxrss"
                except Exception:
                    self._resource = None

    def sample(self) -> int:
        value = 0
        if self._process is not None:
            value = int(
                self._process.memory_info().rss
            )
        elif self._resource is not None:
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


def connect_telemetry(path: Path) -> sqlite3.Connection:
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
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pair_schedule (
            scale_name TEXT NOT NULL,
            thread_count INTEGER NOT NULL,
            pair_index INTEGER NOT NULL,
            order_label TEXT NOT NULL,
            first_mode TEXT NOT NULL,
            second_mode TEXT NOT NULL,
            PRIMARY KEY(
                scale_name,
                thread_count,
                pair_index
            )
        );

        CREATE TABLE IF NOT EXISTS confirmation_cells (
            cell_id TEXT PRIMARY KEY,
            scale_name TEXT NOT NULL,
            thread_count INTEGER NOT NULL,
            pair_index INTEGER NOT NULL,
            order_label TEXT NOT NULL,
            order_position INTEGER NOT NULL,
            mode TEXT NOT NULL,

            warmup_query_count INTEGER NOT NULL,
            warmup_correct_queries INTEGER NOT NULL,
            warmup_unique_templates INTEGER NOT NULL,
            warmup_elapsed_ns INTEGER NOT NULL,

            measured_query_count INTEGER NOT NULL,
            measured_correct_queries INTEGER NOT NULL,
            measured_elapsed_ns INTEGER NOT NULL,
            throughput_qps REAL NOT NULL,

            p50_latency_ns REAL NOT NULL,
            p95_latency_ns REAL NOT NULL,
            p99_latency_ns REAL NOT NULL,
            maximum_latency_ns REAL NOT NULL,

            p50_lock_wait_ns REAL NOT NULL,
            p95_lock_wait_ns REAL NOT NULL,
            p99_lock_wait_ns REAL NOT NULL,

            p50_execution_ns REAL NOT NULL,
            p95_execution_ns REAL NOT NULL,
            p99_execution_ns REAL NOT NULL,

            background_migrations INTEGER NOT NULL,
            background_migration_failures INTEGER NOT NULL,
            background_rewrite_bytes INTEGER NOT NULL,
            background_migration_elapsed_ns INTEGER NOT NULL,

            log_rows_before_compaction INTEGER NOT NULL,
            log_rows_after_compaction INTEGER NOT NULL,
            archived_log_rows INTEGER NOT NULL,

            catalog_bytes_before_compaction INTEGER NOT NULL,
            catalog_bytes_after_compaction INTEGER NOT NULL,
            peak_rss_bytes INTEGER NOT NULL,

            workload_sha256 TEXT NOT NULL,
            latency_file TEXT NOT NULL,
            completed_at_ns INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS migration_archive (
            cell_id TEXT NOT NULL,
            migration_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            from_version INTEGER NOT NULL,
            to_version INTEGER NOT NULL,
            from_codec TEXT NOT NULL,
            target_codec TEXT NOT NULL,
            state TEXT NOT NULL,
            started_at_ns INTEGER NOT NULL,
            updated_at_ns INTEGER NOT NULL,
            completed_at_ns INTEGER,
            rewrite_bytes INTEGER,
            migration_time_ns INTEGER,
            PRIMARY KEY(cell_id, migration_id)
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
        INSERT INTO metadata(key, value)
        VALUES (?, ?)
        ON CONFLICT(key)
        DO UPDATE SET value=excluded.value
        """,
        (key, serialized),
    )


def telemetry_get(
    connection: sqlite3.Connection,
    key: str,
) -> Optional[Any]:
    row = connection.execute(
        "SELECT value FROM metadata WHERE key=?",
        (key,),
    ).fetchone()
    if row is None:
        return None
    return json.loads(row["value"])


def save_latency_file(
    path: Path,
    latency_ns: np.ndarray,
    lock_wait_ns: np.ndarray,
    execution_ns: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_name(
        path.name + ".tmp"
    )
    with open(temporary, "wb") as file:
        np.savez_compressed(
            file,
            latency_ns=latency_ns,
            lock_wait_ns=lock_wait_ns,
            execution_ns=execution_ns,
        )
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def archive_migration_rows(
    telemetry: sqlite3.Connection,
    cell_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        return
    telemetry.executemany(
        """
        INSERT OR REPLACE INTO migration_archive(
            cell_id,
            migration_id,
            chunk_id,
            from_version,
            to_version,
            from_codec,
            target_codec,
            state,
            started_at_ns,
            updated_at_ns,
            completed_at_ns,
            rewrite_bytes,
            migration_time_ns
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?
        )
        """,
        [
            (
                cell_id,
                str(row["migration_id"]),
                str(row["chunk_id"]),
                int(row["from_version"]),
                int(row["to_version"]),
                str(row["from_codec"]),
                str(row["target_codec"]),
                str(row["state"]),
                int(row["started_at_ns"]),
                int(row["updated_at_ns"]),
                (
                    int(row["completed_at_ns"])
                    if row["completed_at_ns"] is not None
                    else None
                ),
                (
                    int(row["rewrite_bytes"])
                    if row["rewrite_bytes"] is not None
                    else None
                ),
                (
                    int(row["migration_time_ns"])
                    if row["migration_time_ns"] is not None
                    else None
                ),
            )
            for row in rows
        ],
    )


def pair_order(
    pair_index: int,
) -> Tuple[str, Tuple[str, str]]:
    if pair_index % 2 == 0:
        return (
            "FG_BG",
            (
                "foreground_only",
                "background_migration",
            ),
        )
    return (
        "BG_FG",
        (
            "background_migration",
            "foreground_only",
        ),
    )


def build_pair_schedule(
    telemetry: sqlite3.Connection,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scale_name in SCALE_NAMES:
        for thread_count in QUERY_THREADS:
            for pair_index in range(PAIR_REPEATS):
                order_label, modes = pair_order(
                    pair_index
                )
                row = {
                    "scale_name": scale_name,
                    "thread_count": thread_count,
                    "pair_index": pair_index,
                    "order_label": order_label,
                    "first_mode": modes[0],
                    "second_mode": modes[1],
                }
                rows.append(row)
                telemetry.execute(
                    """
                    INSERT OR REPLACE INTO pair_schedule(
                        scale_name,
                        thread_count,
                        pair_index,
                        order_label,
                        first_mode,
                        second_mode
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scale_name,
                        thread_count,
                        pair_index,
                        order_label,
                        modes[0],
                        modes[1],
                    ),
                )
    return pd.DataFrame(rows)


def workload_sha256(
    scale_name: str,
    thread_count: int,
    pair_index: int,
) -> str:
    material = {
        "seed": CONFIRM_SEED,
        "scale_name": scale_name,
        "thread_count": thread_count,
        "pair_index": pair_index,
        "warmup_queries": WARMUP_QUERIES_PER_CELL,
        "measured_queries": MEASURED_QUERIES_PER_CELL,
        "query_generator": (
            "global-index deterministic; mode independent"
        ),
    }
    return hashlib.sha256(
        json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def query_for_global_index(
    core: Any,
    base: Any,
    store: Any,
    phase: str,
    scale_name: str,
    thread_count: int,
    pair_index: int,
    global_index: int,
) -> Tuple[str, Any, Any]:
    template_count = len(store.templates)

    if (
        phase == "warmup"
        and global_index < template_count
    ):
        # Base catalogs are ordered by tenant and then template. The first
        # template_count logical chunks therefore cover every real template.
        chunk_position = global_index
    else:
        chunk_position = stable_seed(
            CONFIRM_SEED,
            "chunk",
            phase,
            scale_name,
            thread_count,
            pair_index,
            global_index,
        ) % len(store.chunk_ids)

    chunk_id = store.chunk_ids[chunk_position]
    entry = store.snapshot[chunk_id]
    template = store.templates[
        entry.template_index
    ]
    rng = random.Random(
        stable_seed(
            CONFIRM_SEED,
            "query",
            phase,
            scale_name,
            thread_count,
            pair_index,
            global_index,
        )
    )
    query = core.query_from_rng(
        base,
        template,
        rng,
        (
            f"B3C:{phase}:{scale_name}:"
            f"{thread_count}:{pair_index}:"
            f"{global_index}"
        ),
    )
    return chunk_id, template, query


def run_query_phase(
    core: Any,
    base: Any,
    store: Any,
    truth: np.memmap,
    memory: MemoryTracker,
    phase: str,
    scale_name: str,
    thread_count: int,
    pair_index: int,
    total_queries: int,
    start_event: threading.Event,
    retain_latencies: bool,
) -> Dict[str, Any]:
    assignments = [
        list(
            range(
                worker_index,
                total_queries,
                thread_count,
            )
        )
        for worker_index in range(
            thread_count
        )
    ]

    def worker(
        worker_index: int,
        indices: Sequence[int],
    ) -> Dict[str, Any]:
        count = len(indices)
        latency = (
            np.empty(count, dtype=np.int64)
            if retain_latencies
            else None
        )
        lock_wait = (
            np.empty(count, dtype=np.int64)
            if retain_latencies
            else None
        )
        execution = (
            np.empty(count, dtype=np.int64)
            if retain_latencies
            else None
        )
        correct = 0
        templates_seen: set[int] = set()

        start_event.wait()
        for local_position, global_index in enumerate(
            indices
        ):
            (
                chunk_id,
                template,
                query,
            ) = query_for_global_index(
                core,
                base,
                store,
                phase,
                scale_name,
                thread_count,
                pair_index,
                global_index,
            )
            expected = base.execute_on_array(
                np.asarray(
                    truth[
                        template.truth_index
                    ],
                    dtype=np.float32,
                ),
                query,
            )
            (
                actual,
                total_ns,
                wait_ns,
                execution_ns,
                _,
                observed_entry,
            ) = store.execute_query(
                chunk_id,
                query,
            )
            if not base.results_equal(
                actual,
                expected,
            ):
                raise AssertionError(
                    "B3-C query mismatch: "
                    f"phase={phase}, "
                    f"scale={scale_name}, "
                    f"threads={thread_count}, "
                    f"pair={pair_index}, "
                    f"worker={worker_index}, "
                    f"global={global_index}, "
                    f"chunk={chunk_id}, "
                    f"codec={observed_entry.active_codec}, "
                    f"query={query.query_type}, "
                    f"actual={actual}, "
                    f"expected={expected}"
                )
            correct += 1
            templates_seen.add(
                int(template.template_index)
            )
            if retain_latencies:
                assert latency is not None
                assert lock_wait is not None
                assert execution is not None
                latency[local_position] = total_ns
                lock_wait[local_position] = wait_ns
                execution[local_position] = (
                    execution_ns
                )
            if local_position % 5000 == 0:
                memory.sample()

        return {
            "correct": correct,
            "templates_seen": templates_seen,
            "latency": latency,
            "lock_wait": lock_wait,
            "execution": execution,
        }

    futures = []
    phase_start_ns = time.perf_counter_ns()
    with ThreadPoolExecutor(
        max_workers=thread_count,
        thread_name_prefix=(
            f"B3C-{phase}-{thread_count}"
        ),
    ) as executor:
        for worker_index, indices in enumerate(
            assignments
        ):
            futures.append(
                executor.submit(
                    worker,
                    worker_index,
                    indices,
                )
            )
        start_event.set()
        results = [
            future.result()
            for future in as_completed(
                futures
            )
        ]
    elapsed_ns = (
        time.perf_counter_ns()
        - phase_start_ns
    )

    correct_queries = int(
        sum(
            int(result["correct"])
            for result in results
        )
    )
    templates_seen: set[int] = set()
    for result in results:
        templates_seen.update(
            result["templates_seen"]
        )

    output: Dict[str, Any] = {
        "query_count": total_queries,
        "correct_queries": correct_queries,
        "unique_templates": len(
            templates_seen
        ),
        "elapsed_ns": int(elapsed_ns),
    }

    if retain_latencies:
        output["latency_ns"] = np.concatenate(
            [
                result["latency"]
                for result in results
            ]
        )
        output["lock_wait_ns"] = np.concatenate(
            [
                result["lock_wait"]
                for result in results
            ]
        )
        output["execution_ns"] = np.concatenate(
            [
                result["execution"]
                for result in results
            ]
        )

    return output


def run_confirmation_cell(
    project_dir: Path,
    output_dir: Path,
    core: Any,
    base: Any,
    truth_path: Path,
    telemetry: sqlite3.Connection,
    memory: MemoryTracker,
    scale_name: str,
    thread_count: int,
    pair_index: int,
    order_label: str,
    order_position: int,
    mode: str,
) -> Dict[str, Any]:
    cell_id = (
        f"{scale_name}__threads_{thread_count}"
        f"__pair_{pair_index}__{mode}"
    )
    existing = telemetry.execute(
        """
        SELECT *
        FROM confirmation_cells
        WHERE cell_id=?
        """,
        (cell_id,),
    ).fetchone()
    if existing is not None:
        print(f"[B3-C cell resume] {cell_id}")
        return dict(existing)

    base_catalog = (
        project_dir
        / B3_RESULT_DIR_NAME
        / "base_scales"
        / scale_name
        / "catalog.sqlite3"
    )
    if not base_catalog.exists():
        raise FileNotFoundError(base_catalog)

    work_root = (
        output_dir
        / "working_cells"
        / cell_id
    )
    shutil.rmtree(
        work_root,
        ignore_errors=True,
    )
    work_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    copy_catalog(
        base_catalog,
        work_root / "catalog.sqlite3",
    )

    time.sleep(CELL_SETTLE_SECONDS)

    store = core.ScaleStore(
        work_root,
        base,
        perform_recovery=True,
        load_snapshot=True,
    )
    truth = np.memmap(
        truth_path,
        mode="r",
        dtype=np.float32,
        shape=(
            len(store.templates),
            RAW_CHUNK_VALUES,
        ),
    )

    try:
        # Identical warm-up for both modes. No migration occurs during warm-up.
        warmup_event = threading.Event()
        warmup = run_query_phase(
            core,
            base,
            store,
            truth,
            memory,
            phase="warmup",
            scale_name=scale_name,
            thread_count=thread_count,
            pair_index=pair_index,
            total_queries=WARMUP_QUERIES_PER_CELL,
            start_event=warmup_event,
            retain_latencies=False,
        )
        if (
            warmup["correct_queries"]
            != WARMUP_QUERIES_PER_CELL
        ):
            raise AssertionError(
                f"Warm-up correctness failure: {cell_id}"
            )
        if (
            warmup["unique_templates"]
            != len(store.templates)
        ):
            raise AssertionError(
                "Warm-up did not cover every template: "
                f"{cell_id}, "
                f"{warmup['unique_templates']}/"
                f"{len(store.templates)}"
            )

        migration_results: List[
            Dict[str, Any]
        ] = []
        migration_errors: List[str] = []
        migration_lock = threading.Lock()
        measured_event = threading.Event()

        def migration_worker() -> None:
            rng = random.Random(
                stable_seed(
                    CONFIRM_SEED,
                    "migration",
                    scale_name,
                    thread_count,
                    pair_index,
                )
            )
            measured_event.wait()
            selected = rng.sample(
                store.chunk_ids,
                min(
                    BACKGROUND_MIGRATIONS_PER_CELL,
                    len(store.chunk_ids),
                ),
            )
            for chunk_id in selected:
                try:
                    entry = store.snapshot[
                        chunk_id
                    ]
                    choices = [
                        codec
                        for codec in core.CODECS
                        if codec
                        != entry.active_codec
                    ]
                    target_codec = choices[
                        rng.randrange(
                            len(choices)
                        )
                    ]
                    result = store.migrate(
                        chunk_id,
                        target_codec,
                    )
                    after = store.snapshot[
                        chunk_id
                    ]
                    decoded = store._decode_entry(
                        after
                    )
                    template = store.templates[
                        after.template_index
                    ]
                    expected = np.asarray(
                        truth[
                            template.truth_index
                        ],
                        dtype=np.float32,
                    )
                    if not np.array_equal(
                        decoded,
                        expected,
                    ):
                        raise AssertionError(
                            "B3-C background migration "
                            "truth mismatch"
                        )
                    with migration_lock:
                        migration_results.append(
                            result
                        )
                except Exception:
                    with migration_lock:
                        migration_errors.append(
                            traceback.format_exc()
                        )

        migration_thread: Optional[
            threading.Thread
        ] = None
        if mode == "background_migration":
            migration_thread = threading.Thread(
                target=migration_worker,
                name=f"B3C-migration-{cell_id}",
                daemon=False,
            )
            migration_thread.start()

        measured = run_query_phase(
            core,
            base,
            store,
            truth,
            memory,
            phase="measured",
            scale_name=scale_name,
            thread_count=thread_count,
            pair_index=pair_index,
            total_queries=(
                MEASURED_QUERIES_PER_CELL
            ),
            start_event=measured_event,
            retain_latencies=True,
        )

        if migration_thread is not None:
            migration_thread.join()

        if migration_errors:
            raise RuntimeError(
                "B3-C background migration failed:\n"
                + "\n".join(migration_errors)
            )
        if (
            measured["correct_queries"]
            != MEASURED_QUERIES_PER_CELL
        ):
            raise AssertionError(
                f"Measured correctness failure: {cell_id}"
            )

        latency_ns = measured["latency_ns"]
        lock_wait_ns = measured["lock_wait_ns"]
        execution_ns = measured["execution_ns"]

        latency_path = (
            output_dir
            / "latency_samples"
            / f"{cell_id}.npz"
        )
        save_latency_file(
            latency_path,
            latency_ns,
            lock_wait_ns,
            execution_ns,
        )

        catalog_before = sqlite_file_bytes(
            store.db_path
        )
        log_rows_before = int(
            store.connection.execute(
                """
                SELECT COUNT(*) AS n
                FROM migration_log
                """
            ).fetchone()["n"]
        )
        archived_rows, compaction = (
            store.compact_completed_logs()
        )
        archive_migration_rows(
            telemetry,
            cell_id,
            archived_rows,
        )
        catalog_after = sqlite_file_bytes(
            store.db_path
        )
        memory.sample()

        completed_migrations = [
            result
            for result in migration_results
            if not result.get(
                "no_op",
                False,
            )
        ]
        migration_count = len(
            completed_migrations
        )
        migration_elapsed_ns = int(
            sum(
                int(
                    result.get(
                        "migration_time_ns",
                        0,
                    )
                )
                for result in completed_migrations
            )
        )
        rewrite_bytes = int(
            sum(
                int(
                    result.get(
                        "rewrite_bytes",
                        0,
                    )
                )
                for result in completed_migrations
            )
        )

        result = {
            "cell_id": cell_id,
            "scale_name": scale_name,
            "thread_count": thread_count,
            "pair_index": pair_index,
            "order_label": order_label,
            "order_position": order_position,
            "mode": mode,

            "warmup_query_count": (
                WARMUP_QUERIES_PER_CELL
            ),
            "warmup_correct_queries": int(
                warmup["correct_queries"]
            ),
            "warmup_unique_templates": int(
                warmup["unique_templates"]
            ),
            "warmup_elapsed_ns": int(
                warmup["elapsed_ns"]
            ),

            "measured_query_count": (
                MEASURED_QUERIES_PER_CELL
            ),
            "measured_correct_queries": int(
                measured["correct_queries"]
            ),
            "measured_elapsed_ns": int(
                measured["elapsed_ns"]
            ),
            "throughput_qps": float(
                MEASURED_QUERIES_PER_CELL
                / max(
                    measured["elapsed_ns"]
                    / 1e9,
                    1e-12,
                )
            ),

            "p50_latency_ns": float(
                np.quantile(
                    latency_ns,
                    0.50,
                )
            ),
            "p95_latency_ns": float(
                np.quantile(
                    latency_ns,
                    0.95,
                )
            ),
            "p99_latency_ns": float(
                np.quantile(
                    latency_ns,
                    0.99,
                )
            ),
            "maximum_latency_ns": float(
                np.max(latency_ns)
            ),

            "p50_lock_wait_ns": float(
                np.quantile(
                    lock_wait_ns,
                    0.50,
                )
            ),
            "p95_lock_wait_ns": float(
                np.quantile(
                    lock_wait_ns,
                    0.95,
                )
            ),
            "p99_lock_wait_ns": float(
                np.quantile(
                    lock_wait_ns,
                    0.99,
                )
            ),

            "p50_execution_ns": float(
                np.quantile(
                    execution_ns,
                    0.50,
                )
            ),
            "p95_execution_ns": float(
                np.quantile(
                    execution_ns,
                    0.95,
                )
            ),
            "p99_execution_ns": float(
                np.quantile(
                    execution_ns,
                    0.99,
                )
            ),

            "background_migrations": int(
                migration_count
            ),
            "background_migration_failures": int(
                len(migration_errors)
            ),
            "background_rewrite_bytes": int(
                rewrite_bytes
            ),
            "background_migration_elapsed_ns": int(
                migration_elapsed_ns
            ),

            "log_rows_before_compaction": int(
                log_rows_before
            ),
            "log_rows_after_compaction": int(
                compaction["after"]
            ),
            "archived_log_rows": int(
                compaction["archived"]
            ),

            "catalog_bytes_before_compaction": int(
                catalog_before[
                    "total_files_bytes"
                ]
            ),
            "catalog_bytes_after_compaction": int(
                catalog_after[
                    "total_files_bytes"
                ]
            ),
            "peak_rss_bytes": int(
                memory.peak_rss_bytes
            ),

            "workload_sha256": workload_sha256(
                scale_name,
                thread_count,
                pair_index,
            ),
            "latency_file": str(
                latency_path
            ),
            "completed_at_ns": time.time_ns(),
        }

        columns = list(result.keys())
        placeholders = ",".join(
            "?"
            for _ in columns
        )
        with telemetry_transaction(
            telemetry
        ):
            telemetry.execute(
                f"""
                INSERT INTO confirmation_cells(
                    {",".join(columns)}
                ) VALUES ({placeholders})
                """,
                tuple(
                    result[column]
                    for column in columns
                ),
            )

        print(
            f"[B3-C cell complete] {cell_id}: "
            f"{result['throughput_qps']:.1f} q/s, "
            f"p99={result['p99_latency_ns'] / 1e6:.3f} ms, "
            f"warm templates="
            f"{result['warmup_unique_templates']}"
        )
        return result
    finally:
        store.close()
        del truth
        shutil.rmtree(
            work_root,
            ignore_errors=True,
        )


def geometric_mean(values: np.ndarray) -> float:
    array = np.asarray(
        values,
        dtype=np.float64,
    )
    if (
        array.size == 0
        or np.any(array <= 0)
    ):
        return float("nan")
    return float(
        np.exp(
            np.mean(
                np.log(array)
            )
        )
    )


def bootstrap_geometric_mean(
    values: np.ndarray,
    seed_parts: Sequence[Any],
) -> Tuple[float, float, float]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )
    if (
        array.size == 0
        or np.any(array <= 0)
    ):
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )
    rng = np.random.default_rng(
        stable_seed(
            CONFIRM_SEED,
            "bootstrap",
            *seed_parts,
        )
    )
    indices = rng.integers(
        0,
        array.size,
        size=(
            BOOTSTRAP_REPLICATES,
            array.size,
        ),
    )
    estimates = np.exp(
        np.mean(
            np.log(
                array[indices]
            ),
            axis=1,
        )
    )
    return (
        geometric_mean(array),
        float(
            np.quantile(
                estimates,
                0.025,
            )
        ),
        float(
            np.quantile(
                estimates,
                0.975,
            )
        ),
    )


def stratified_bootstrap_geometric_mean(
    frame: pd.DataFrame,
    value_column: str,
) -> Tuple[float, float, float]:
    groups = [
        group[
            value_column
        ].to_numpy(
            dtype=np.float64
        )
        for _, group in frame.groupby(
            [
                "scale_name",
                "thread_count",
            ],
            sort=True,
        )
    ]
    if (
        not groups
        or any(
            len(group) == 0
            or np.any(group <= 0)
            for group in groups
        )
    ):
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    point = float(
        np.exp(
            np.mean(
                np.concatenate(
                    [
                        np.log(group)
                        for group in groups
                    ]
                )
            )
        )
    )
    rng = np.random.default_rng(
        stable_seed(
            CONFIRM_SEED,
            "stratified-bootstrap",
            value_column,
        )
    )
    stratum_log_means = []
    for group in groups:
        indices = rng.integers(
            0,
            len(group),
            size=(
                BOOTSTRAP_REPLICATES,
                len(group),
            ),
        )
        stratum_log_means.append(
            np.mean(
                np.log(
                    group[indices]
                ),
                axis=1,
            )
        )
    bootstrap = np.exp(
        np.mean(
            np.vstack(
                stratum_log_means
            ),
            axis=0,
        )
    )
    return (
        point,
        float(
            np.quantile(
                bootstrap,
                0.025,
            )
        ),
        float(
            np.quantile(
                bootstrap,
                0.975,
            )
        ),
    )


def bootstrap_order_effect(
    paired: pd.DataFrame,
    value_column: str,
) -> Dict[str, float]:
    fg_bg = paired[
        paired["order_label"]
        == "FG_BG"
    ][value_column].to_numpy(
        dtype=np.float64
    )
    bg_fg = paired[
        paired["order_label"]
        == "BG_FG"
    ][value_column].to_numpy(
        dtype=np.float64
    )
    if (
        len(fg_bg) == 0
        or len(bg_fg) == 0
        or np.any(fg_bg <= 0)
        or np.any(bg_fg <= 0)
    ):
        return {
            "fg_bg_geometric_mean": float("nan"),
            "bg_fg_geometric_mean": float("nan"),
            "order_ratio_fg_bg_over_bg_fg": float(
                "nan"
            ),
            "order_ratio_ci_low": float("nan"),
            "order_ratio_ci_high": float("nan"),
        }

    point = float(
        np.exp(
            np.mean(np.log(fg_bg))
            - np.mean(np.log(bg_fg))
        )
    )
    rng = np.random.default_rng(
        stable_seed(
            CONFIRM_SEED,
            "order-bootstrap",
            value_column,
        )
    )
    fg_indices = rng.integers(
        0,
        len(fg_bg),
        size=(
            BOOTSTRAP_REPLICATES,
            len(fg_bg),
        ),
    )
    bg_indices = rng.integers(
        0,
        len(bg_fg),
        size=(
            BOOTSTRAP_REPLICATES,
            len(bg_fg),
        ),
    )
    estimates = np.exp(
        np.mean(
            np.log(
                fg_bg[fg_indices]
            ),
            axis=1,
        )
        - np.mean(
            np.log(
                bg_fg[bg_indices]
            ),
            axis=1,
        )
    )
    return {
        "fg_bg_geometric_mean": geometric_mean(
            fg_bg
        ),
        "bg_fg_geometric_mean": geometric_mean(
            bg_fg
        ),
        "order_ratio_fg_bg_over_bg_fg": point,
        "order_ratio_ci_low": float(
            np.quantile(
                estimates,
                0.025,
            )
        ),
        "order_ratio_ci_high": float(
            np.quantile(
                estimates,
                0.975,
            )
        ),
    }


def build_confirmation_summaries(
    output_dir: Path,
    telemetry: sqlite3.Connection,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Any],
]:
    cells = pd.read_sql_query(
        """
        SELECT *
        FROM confirmation_cells
        ORDER BY
            scale_name,
            thread_count,
            pair_index,
            order_position
        """,
        telemetry,
    )
    foreground = cells[
        cells["mode"]
        == "foreground_only"
    ].copy()
    background = cells[
        cells["mode"]
        == "background_migration"
    ].copy()

    keys = [
        "scale_name",
        "thread_count",
        "pair_index",
        "order_label",
    ]
    paired = foreground.merge(
        background,
        on=keys,
        suffixes=(
            "_foreground",
            "_background",
        ),
        validate="one_to_one",
    )
    paired[
        "throughput_ratio_background_over_foreground"
    ] = (
        paired[
            "throughput_qps_background"
        ]
        / paired[
            "throughput_qps_foreground"
        ]
    )
    for percentile in (
        "p50",
        "p95",
        "p99",
    ):
        paired[
            f"{percentile}_latency_amplification"
        ] = (
            paired[
                f"{percentile}_latency_ns_background"
            ]
            / paired[
                f"{percentile}_latency_ns_foreground"
            ]
        )
    paired[
        "p99_lock_wait_amplification"
    ] = (
        paired[
            "p99_lock_wait_ns_background"
        ]
        / np.maximum(
            paired[
                "p99_lock_wait_ns_foreground"
            ],
            1.0,
        )
    )
    paired[
        "workload_hash_match"
    ] = (
        paired[
            "workload_sha256_foreground"
        ]
        == paired[
            "workload_sha256_background"
        ]
    )

    write_dataframe_atomic(
        output_dir
        / "confirmation_cells.csv",
        cells,
    )
    write_dataframe_atomic(
        output_dir
        / "paired_balanced_results.csv",
        paired,
    )

    summary_rows: List[
        Dict[str, Any]
    ] = []
    for (
        scale_name,
        thread_count,
    ), group in paired.groupby(
        [
            "scale_name",
            "thread_count",
        ],
        sort=True,
    ):
        throughput = (
            group[
                "throughput_ratio_background_over_foreground"
            ].to_numpy(
                dtype=np.float64
            )
        )
        p99 = (
            group[
                "p99_latency_amplification"
            ].to_numpy(
                dtype=np.float64
            )
        )
        (
            throughput_point,
            throughput_low,
            throughput_high,
        ) = bootstrap_geometric_mean(
            throughput,
            (
                scale_name,
                thread_count,
                "throughput",
            ),
        )
        (
            p99_point,
            p99_low,
            p99_high,
        ) = bootstrap_geometric_mean(
            p99,
            (
                scale_name,
                thread_count,
                "p99",
            ),
        )
        row = {
            "scale_name": scale_name,
            "thread_count": int(
                thread_count
            ),
            "paired_repeats": int(
                len(group)
            ),
            "fg_bg_pairs": int(
                (
                    group[
                        "order_label"
                    ]
                    == "FG_BG"
                ).sum()
            ),
            "bg_fg_pairs": int(
                (
                    group[
                        "order_label"
                    ]
                    == "BG_FG"
                ).sum()
            ),
            "median_foreground_qps": float(
                group[
                    "throughput_qps_foreground"
                ].median()
            ),
            "median_background_qps": float(
                group[
                    "throughput_qps_background"
                ].median()
            ),
            "throughput_ratio_geometric_mean": (
                throughput_point
            ),
            "throughput_ratio_ci_low": (
                throughput_low
            ),
            "throughput_ratio_ci_high": (
                throughput_high
            ),
            "throughput_ratio_median": float(
                np.median(throughput)
            ),
            "throughput_ratio_minimum": float(
                np.min(throughput)
            ),
            "throughput_ratio_maximum": float(
                np.max(throughput)
            ),
            "p99_amplification_geometric_mean": (
                p99_point
            ),
            "p99_amplification_ci_low": (
                p99_low
            ),
            "p99_amplification_ci_high": (
                p99_high
            ),
            "p99_amplification_median": float(
                np.median(p99)
            ),
            "p99_amplification_minimum": float(
                np.min(p99)
            ),
            "p99_amplification_maximum": float(
                np.max(p99)
            ),
            "all_workload_hashes_match": bool(
                group[
                    "workload_hash_match"
                ].all()
            ),
            "all_warmups_cover_all_templates": bool(
                (
                    group[
                        "warmup_unique_templates_foreground"
                    ]
                    == group[
                        "warmup_unique_templates_background"
                    ]
                ).all()
                and (
                    group[
                        "warmup_unique_templates_foreground"
                    ]
                    == 720
                ).all()
            ),
        }
        summary_rows.append(row)

    summary = pd.DataFrame(
        summary_rows
    )
    write_dataframe_atomic(
        output_dir
        / "balanced_interference_summary.csv",
        summary,
    )

    (
        overall_throughput,
        overall_throughput_low,
        overall_throughput_high,
    ) = stratified_bootstrap_geometric_mean(
        paired,
        (
            "throughput_ratio_"
            "background_over_foreground"
        ),
    )
    (
        overall_p99,
        overall_p99_low,
        overall_p99_high,
    ) = stratified_bootstrap_geometric_mean(
        paired,
        "p99_latency_amplification",
    )

    throughput_order = bootstrap_order_effect(
        paired,
        (
            "throughput_ratio_"
            "background_over_foreground"
        ),
    )
    p99_order = bootstrap_order_effect(
        paired,
        "p99_latency_amplification",
    )

    order_rows = [
        {
            "metric": "throughput_ratio",
            **throughput_order,
        },
        {
            "metric": "p99_latency_amplification",
            **p99_order,
        },
    ]
    write_dataframe_atomic(
        output_dir
        / "order_effect_summary.csv",
        pd.DataFrame(order_rows),
    )

    overall = {
        "paired_observations": int(
            len(paired)
        ),
        "strata": int(
            paired.groupby(
                [
                    "scale_name",
                    "thread_count",
                ]
            ).ngroups
        ),
        "throughput_ratio": {
            "stratified_geometric_mean": (
                overall_throughput
            ),
            "ci_low": (
                overall_throughput_low
            ),
            "ci_high": (
                overall_throughput_high
            ),
            "throughput_loss_percent": float(
                (
                    1.0
                    - overall_throughput
                )
                * 100.0
            ),
        },
        "p99_latency_amplification": {
            "stratified_geometric_mean": (
                overall_p99
            ),
            "ci_low": overall_p99_low,
            "ci_high": overall_p99_high,
            "increase_percent": float(
                (
                    overall_p99
                    - 1.0
                )
                * 100.0
            ),
        },
        "order_effect": {
            "throughput": (
                throughput_order
            ),
            "p99_latency": p99_order,
        },
        "low_interference_confirmation": {
            "throughput_lower_ci_at_least_0_90": bool(
                overall_throughput_low
                >= LOW_INTERFERENCE_THROUGHPUT_FLOOR
            ),
            "p99_upper_ci_at_most_1_25": bool(
                overall_p99_high
                <= LOW_INTERFERENCE_P99_CEILING
            ),
            "confirmed": bool(
                overall_throughput_low
                >= LOW_INTERFERENCE_THROUGHPUT_FLOOR
                and overall_p99_high
                <= LOW_INTERFERENCE_P99_CEILING
            ),
        },
    }
    write_json_atomic(
        output_dir
        / "overall_balanced_effect.json",
        overall,
    )
    return cells, paired, overall


def validate_latency_files(
    cells: pd.DataFrame,
) -> Dict[str, Any]:
    missing: List[str] = []
    wrong_count: List[str] = []
    mismatched_summary: List[str] = []

    total_samples = 0
    for row in cells.itertuples(
        index=False
    ):
        path = Path(str(row.latency_file))
        if not path.exists():
            missing.append(str(path))
            continue
        with np.load(
            path,
            allow_pickle=False,
        ) as archive:
            latency = np.asarray(
                archive["latency_ns"],
                dtype=np.int64,
            )
            lock_wait = np.asarray(
                archive["lock_wait_ns"],
                dtype=np.int64,
            )
            execution = np.asarray(
                archive["execution_ns"],
                dtype=np.int64,
            )
        total_samples += int(
            latency.size
        )
        if not (
            latency.size
            == lock_wait.size
            == execution.size
            == MEASURED_QUERIES_PER_CELL
        ):
            wrong_count.append(
                str(row.cell_id)
            )
            continue

        recomputed = (
            float(
                np.quantile(
                    latency,
                    0.50,
                )
            ),
            float(
                np.quantile(
                    latency,
                    0.95,
                )
            ),
            float(
                np.quantile(
                    latency,
                    0.99,
                )
            ),
            float(np.max(latency)),
        )
        stored = (
            float(row.p50_latency_ns),
            float(row.p95_latency_ns),
            float(row.p99_latency_ns),
            float(row.maximum_latency_ns),
        )
        if not np.allclose(
            recomputed,
            stored,
            rtol=0.0,
            atol=0.5,
        ):
            mismatched_summary.append(
                str(row.cell_id)
            )

    return {
        "cell_count": int(len(cells)),
        "total_latency_samples": int(
            total_samples
        ),
        "missing_files": missing,
        "wrong_sample_count_cells": (
            wrong_count
        ),
        "summary_mismatch_cells": (
            mismatched_summary
        ),
        "pass": bool(
            not missing
            and not wrong_count
            and not mismatched_summary
        ),
    }


def main() -> int:
    project_dir = Path(
        __file__
    ).resolve().parent
    output_dir = (
        project_dir
        / OUTPUT_DIR_NAME
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    parent_script_path = (
        project_dir
        / B3_PARENT_SCRIPT_NAME
    )
    repair_script_path = (
        project_dir
        / B3_REPAIR_SCRIPT_NAME
    )
    e0c_script_path = (
        project_dir
        / E0C_SCRIPT_NAME
    )
    b2_dir = (
        project_dir
        / B2_RESULT_DIR_NAME
    )
    b3_dir = (
        project_dir
        / B3_RESULT_DIR_NAME
    )

    print("=" * 112)
    print(
        "RePressDB B3-C v1.2.0 — "
        "warm-up controlled AB/BA confirmation"
    )
    print(f"Target journal: {TARGET_JOURNAL}")
    print("=" * 112)

    required = [
        parent_script_path,
        repair_script_path,
        e0c_script_path,
        b2_dir
        / "truth"
        / "data_scale_8192.float32.bin",
        b3_dir
        / "b3_assessment.json",
        b3_dir
        / "b3_experiment_freeze_sha256.txt",
        b3_dir
        / "v111_thread_safety_repair_sha256.txt",
    ]
    for scale_name in SCALE_NAMES:
        required.append(
            b3_dir
            / "base_scales"
            / scale_name
            / "catalog.sqlite3"
        )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    b3_assessment = json.loads(
        (
            b3_dir
            / "b3_assessment.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    if not bool(
        b3_assessment.get(
            "acceptance_checks",
            {},
        ).get(
            "all_ten",
            False,
        )
    ):
        raise RuntimeError(
            "B3 v1.1.1 has not passed all "
            "acceptance checks"
        )

    parent_freeze = (
        b3_dir
        / "b3_experiment_freeze_sha256.txt"
    ).read_text(
        encoding="utf-8"
    ).strip()
    repair_hash = (
        b3_dir
        / "v111_thread_safety_repair_sha256.txt"
    ).read_text(
        encoding="utf-8"
    ).strip()

    core = load_module(
        parent_script_path,
        "repressdb_b3_parent_for_b3c_v120",
    )
    repair = load_module(
        repair_script_path,
        "repressdb_b3_repair_for_b3c_v120",
    )
    repair.install_thread_safe_codec_patch(
        core
    )
    base = load_module(
        e0c_script_path,
        "repressdb_e0c_for_b3c_v120",
    )

    protocol = {
        "version": VERSION,
        "target_journal": TARGET_JOURNAL,
        "parent_b3_experiment_freeze_sha256": (
            parent_freeze
        ),
        "parent_v111_repair_sha256": (
            repair_hash
        ),
        "purpose": (
            "Confirm foreground/background "
            "interference after controlling "
            "warm-up and execution order."
        ),
        "scales": list(SCALE_NAMES),
        "query_threads": list(
            QUERY_THREADS
        ),
        "paired_repeats_per_stratum": (
            PAIR_REPEATS
        ),
        "order_schedule": {
            "even_pairs": "FG_BG",
            "odd_pairs": "BG_FG",
            "pairs_per_order_per_stratum": 3,
        },
        "warmup": {
            "queries_per_cell": (
                WARMUP_QUERIES_PER_CELL
            ),
            "covers_all_720_templates": True,
            "migration_during_warmup": False,
            "mode_independent_workload": True,
        },
        "measurement": {
            "queries_per_cell": (
                MEASURED_QUERIES_PER_CELL
            ),
            "background_migrations_per_background_cell": (
                BACKGROUND_MIGRATIONS_PER_CELL
            ),
            "identical_query_workload_within_pair": True,
            "independent_base_catalog_copy_per_cell": True,
            "truth_validation_per_query": True,
        },
        "total": {
            "measured_cells": (
                len(SCALE_NAMES)
                * len(QUERY_THREADS)
                * PAIR_REPEATS
                * len(MODES)
            ),
            "measured_queries": (
                len(SCALE_NAMES)
                * len(QUERY_THREADS)
                * PAIR_REPEATS
                * len(MODES)
                * MEASURED_QUERIES_PER_CELL
            ),
            "warmup_queries": (
                len(SCALE_NAMES)
                * len(QUERY_THREADS)
                * PAIR_REPEATS
                * len(MODES)
                * WARMUP_QUERIES_PER_CELL
            ),
            "background_migrations": (
                len(SCALE_NAMES)
                * len(QUERY_THREADS)
                * PAIR_REPEATS
                * BACKGROUND_MIGRATIONS_PER_CELL
            ),
        },
        "statistical_summary": {
            "pair_effect": (
                "background / foreground"
            ),
            "primary_aggregation": (
                "stratified geometric mean "
                "across scale-thread strata"
            ),
            "confidence_interval": (
                f"{BOOTSTRAP_REPLICATES} "
                "deterministic bootstrap replicates"
            ),
            "order_effect": (
                "FG_BG pair ratio divided by "
                "BG_FG pair ratio"
            ),
            "low_interference_gate": {
                "throughput_ratio_lower_ci": (
                    LOW_INTERFERENCE_THROUGHPUT_FLOOR
                ),
                "p99_amplification_upper_ci": (
                    LOW_INTERFERENCE_P99_CEILING
                ),
            },
        },
        "unchanged": [
            "B3 scale catalogs",
            "physical codecs",
            "thread-local codec repair",
            "query distribution",
            "copy-on-write migration",
            "50 migrations per background cell",
            "bounded production migration log",
        ],
    }
    write_json_atomic(
        output_dir / "b3c_protocol.json",
        protocol,
    )

    protocol_digest = hashlib.sha256()
    protocol_digest.update(
        parent_freeze.encode("ascii")
    )
    protocol_digest.update(
        repair_hash.encode("ascii")
    )
    protocol_digest.update(
        sha256_file(
            output_dir
            / "b3c_protocol.json"
        ).encode("ascii")
    )
    protocol_hash = (
        protocol_digest.hexdigest()
    )
    (
        output_dir
        / "b3c_protocol_sha256.txt"
    ).write_text(
        protocol_hash + "\n",
        encoding="utf-8",
    )

    freeze_material = {
        "b3c_protocol_sha256": (
            protocol_hash
        ),
        "script_sha256": sha256_file(
            Path(__file__).resolve()
        ),
        "parent_b3_script_sha256": (
            sha256_file(
                parent_script_path
            )
        ),
        "parent_v111_repair_script_sha256": (
            sha256_file(
                repair_script_path
            )
        ),
        "e0c_script_sha256": (
            sha256_file(
                e0c_script_path
            )
        ),
        "parent_b3_experiment_freeze_sha256": (
            parent_freeze
        ),
        "parent_v111_repair_sha256": (
            repair_hash
        ),
    }
    write_json_atomic(
        output_dir
        / "b3c_experiment_freeze_material.json",
        freeze_material,
    )
    freeze_hash = sha256_file(
        output_dir
        / "b3c_experiment_freeze_material.json"
    )
    (
        output_dir
        / "b3c_experiment_freeze_sha256.txt"
    ).write_text(
        freeze_hash + "\n",
        encoding="utf-8",
    )

    telemetry_path = (
        output_dir
        / "telemetry.sqlite3"
    )
    telemetry = connect_telemetry(
        telemetry_path
    )
    existing_freeze = telemetry_get(
        telemetry,
        "experiment_freeze_sha256",
    )
    if (
        existing_freeze is not None
        and existing_freeze
        != freeze_hash
    ):
        telemetry.close()
        raise RuntimeError(
            "Existing B3-C output belongs "
            "to another experiment freeze"
        )
    telemetry_set(
        telemetry,
        "experiment_freeze_sha256",
        freeze_hash,
    )
    telemetry_set(
        telemetry,
        "b3c_protocol_sha256",
        protocol_hash,
    )

    schedule = build_pair_schedule(
        telemetry
    )
    write_dataframe_atomic(
        output_dir
        / "pair_schedule.csv",
        schedule,
    )

    truth_path = (
        b2_dir
        / "truth"
        / "data_scale_8192.float32.bin"
    )
    memory = MemoryTracker()
    memory.sample()
    total_start_ns = (
        time.perf_counter_ns()
    )

    for scale_name in SCALE_NAMES:
        for thread_count in QUERY_THREADS:
            for pair_index in range(
                PAIR_REPEATS
            ):
                order_label, modes = pair_order(
                    pair_index
                )
                for order_position, mode in enumerate(
                    modes,
                    start=1,
                ):
                    run_confirmation_cell(
                        project_dir,
                        output_dir,
                        core,
                        base,
                        truth_path,
                        telemetry,
                        memory,
                        scale_name,
                        thread_count,
                        pair_index,
                        order_label,
                        order_position,
                        mode,
                    )

    (
        cells,
        paired,
        overall_effect,
    ) = build_confirmation_summaries(
        output_dir,
        telemetry,
    )
    latency_validation = (
        validate_latency_files(
            cells
        )
    )
    write_json_atomic(
        output_dir
        / "latency_file_validation.json",
        latency_validation,
    )

    expected_cells = (
        len(SCALE_NAMES)
        * len(QUERY_THREADS)
        * PAIR_REPEATS
        * len(MODES)
    )
    expected_pairs = (
        len(SCALE_NAMES)
        * len(QUERY_THREADS)
        * PAIR_REPEATS
    )
    expected_measured_queries = (
        expected_cells
        * MEASURED_QUERIES_PER_CELL
    )
    expected_warmup_queries = (
        expected_cells
        * WARMUP_QUERIES_PER_CELL
    )
    expected_migrations = (
        len(SCALE_NAMES)
        * len(QUERY_THREADS)
        * PAIR_REPEATS
        * BACKGROUND_MIGRATIONS_PER_CELL
    )

    measured_queries = int(
        cells[
            "measured_query_count"
        ].sum()
    )
    measured_correct = int(
        cells[
            "measured_correct_queries"
        ].sum()
    )
    warmup_queries = int(
        cells[
            "warmup_query_count"
        ].sum()
    )
    warmup_correct = int(
        cells[
            "warmup_correct_queries"
        ].sum()
    )
    migrations = int(
        cells[
            "background_migrations"
        ].sum()
    )
    migration_failures = int(
        cells[
            "background_migration_failures"
        ].sum()
    )

    schedule_counts = (
        schedule.groupby(
            [
                "scale_name",
                "thread_count",
                "order_label",
            ]
        )
        .size()
        .reset_index(
            name="n"
        )
    )
    schedule_balanced = bool(
        (
            schedule_counts["n"]
            == 3
        ).all()
        and len(schedule_counts)
        == (
            len(SCALE_NAMES)
            * len(QUERY_THREADS)
            * 2
        )
    )

    checks = {
        "B3C_1_ab_ba_schedule_balanced": (
            schedule_balanced
        ),
        "B3C_2_all_108_cells_complete": bool(
            len(cells) == expected_cells
        ),
        "B3C_3_all_5400000_measured_queries_correct": bool(
            measured_queries
            == expected_measured_queries
            and measured_correct
            == expected_measured_queries
        ),
        "B3C_4_all_540000_warmup_queries_correct": bool(
            warmup_queries
            == expected_warmup_queries
            and warmup_correct
            == expected_warmup_queries
        ),
        "B3C_5_every_cell_warmup_covers_all_720_templates": bool(
            (
                cells[
                    "warmup_unique_templates"
                ]
                == 720
            ).all()
        ),
        "B3C_6_all_2700_background_migrations_complete": bool(
            migrations
            == expected_migrations
            and migration_failures == 0
        ),
        "B3C_7_identical_workload_hash_within_every_pair": bool(
            len(paired) == expected_pairs
            and bool(
                paired[
                    "workload_hash_match"
                ].all()
            )
        ),
        "B3C_8_completed_migration_log_bounded": bool(
            (
                cells[
                    "log_rows_after_compaction"
                ]
                <= COMPLETED_LOG_RETAIN
            ).all()
        ),
        "B3C_9_all_latency_files_recompute_exactly": bool(
            latency_validation["pass"]
            and latency_validation[
                "total_latency_samples"
            ]
            == expected_measured_queries
        ),
        "B3C_10_balanced_effect_and_order_effect_reported": bool(
            overall_effect[
                "paired_observations"
            ]
            == expected_pairs
            and "order_effect"
            in overall_effect
            and "low_interference_confirmation"
            in overall_effect
        ),
    }
    checks["all_ten"] = bool(
        all(checks.values())
    )

    total_elapsed_ns = (
        time.perf_counter_ns()
        - total_start_ns
    )
    telemetry_sizes = sqlite_file_bytes(
        telemetry_path
    )
    assessment = {
        "version": VERSION,
        "target_journal": TARGET_JOURNAL,
        "status": (
            "B3-C complete"
            if checks["all_ten"]
            else (
                "B3-C completed with failed "
                "acceptance checks"
            )
        ),
        "parent_b3_experiment_freeze_sha256": (
            parent_freeze
        ),
        "parent_v111_repair_sha256": (
            repair_hash
        ),
        "b3c_protocol_sha256": (
            protocol_hash
        ),
        "b3c_experiment_freeze_sha256": (
            freeze_hash
        ),
        "workload": {
            "measured_cells": int(
                len(cells)
            ),
            "paired_observations": int(
                len(paired)
            ),
            "measured_queries": (
                measured_queries
            ),
            "measured_correct_queries": (
                measured_correct
            ),
            "warmup_queries": warmup_queries,
            "warmup_correct_queries": (
                warmup_correct
            ),
            "background_migrations": (
                migrations
            ),
            "background_migration_failures": (
                migration_failures
            ),
        },
        "overall_balanced_effect": (
            overall_effect
        ),
        "latency_validation": (
            latency_validation
        ),
        "memory": {
            "method": memory.method,
            "peak_rss_bytes": int(
                memory.peak_rss_bytes
            ),
            "final_rss_bytes": int(
                memory.sample()
            ),
        },
        "telemetry_storage": (
            telemetry_sizes
        ),
        "total_elapsed_ns_this_invocation": int(
            total_elapsed_ns
        ),
        "acceptance_checks": checks,
        "claim_rule": {
            "low_interference_claim_allowed": bool(
                checks["all_ten"]
                and overall_effect[
                    "low_interference_confirmation"
                ]["confirmed"]
            ),
            "required_for_claim": (
                "overall stratified throughput "
                "ratio lower 95% CI >= 0.90 "
                "and p99 amplification upper "
                "95% CI <= 1.25"
            ),
            "if_not_confirmed": (
                "Report balanced descriptive "
                "effects and uncertainty, but "
                "do not claim low interference."
            ),
        },
        "next_gate": (
            "Stop adding experiments and proceed "
            "to final figures, tables, and paper."
            if checks["all_ten"]
            else (
                "Diagnose B3-C acceptance failures "
                "before final concurrency claims."
            )
        ),
    }
    write_json_atomic(
        output_dir
        / "b3c_assessment.json",
        assessment,
    )
    write_json_atomic(
        project_dir
        / SUMMARY_JSON_NAME,
        assessment,
    )

    run_config = {
        "version": VERSION,
        "target_journal": TARGET_JOURNAL,
        "parent_b3_experiment_freeze_sha256": (
            parent_freeze
        ),
        "parent_v111_repair_sha256": (
            repair_hash
        ),
        "b3c_protocol_sha256": (
            protocol_hash
        ),
        "b3c_experiment_freeze_sha256": (
            freeze_hash
        ),
        "scales": list(SCALE_NAMES),
        "query_threads": list(
            QUERY_THREADS
        ),
        "pair_repeats": PAIR_REPEATS,
        "warmup_queries_per_cell": (
            WARMUP_QUERIES_PER_CELL
        ),
        "measured_queries_per_cell": (
            MEASURED_QUERIES_PER_CELL
        ),
        "background_migrations_per_cell": (
            BACKGROUND_MIGRATIONS_PER_CELL
        ),
        "bootstrap_replicates": (
            BOOTSTRAP_REPLICATES
        ),
        "cell_settle_seconds": (
            CELL_SETTLE_SECONDS
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sqlite_version": (
                sqlite3.sqlite_version
            ),
            "memory_method": memory.method,
            "timer": (
                "time.perf_counter_ns"
            ),
        },
    }
    write_json_atomic(
        output_dir
        / "run_config.json",
        run_config,
    )

    checkpoint_database(
        telemetry
    )
    telemetry.close()

    print("=" * 112)
    print(
        "RePressDB B3-C v1.2.0 finished"
    )
    print(
        json.dumps(
            checks,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        "Low-interference confirmation:",
        overall_effect[
            "low_interference_confirmation"
        ]["confirmed"],
    )
    print(
        f"Elapsed this invocation: "
        f"{total_elapsed_ns / 1e9 / 60:.2f} minutes"
    )
    print(
        f"Results: {output_dir.resolve()}"
    )
    print("Upload:")
    print("  b3c_assessment.json")
    print("  overall_balanced_effect.json")
    print("  balanced_interference_summary.csv")
    print("  paired_balanced_results.csv")
    print("  order_effect_summary.csv")
    print("  confirmation_cells.csv")
    print("  pair_schedule.csv")
    print("  latency_file_validation.json")
    print("  b3c_protocol.json")
    print("  b3c_protocol_sha256.txt")
    print("  b3c_experiment_freeze_sha256.txt")
    print("  run_config.json")
    print("=" * 112)

    return (
        0
        if checks["all_ten"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
