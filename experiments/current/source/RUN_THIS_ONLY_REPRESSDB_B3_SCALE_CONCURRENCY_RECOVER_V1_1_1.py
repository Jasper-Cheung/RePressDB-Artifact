#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
RePressDB B3 v1.1.1
Thread-safe codec-registry recovery wrapper for B3 v1.1.0.

DIRECT RUN
----------
Place this file beside:

  RUN_THIS_ONLY_REPRESSDB_B3_SCALE_CONCURRENCY_V1_1_0.py
  RUN_THIS_ONLY_REPRESSDB_B2_FULL_BDG2_SCALE_V1_0_0.py
  RUN_THIS_ONLY_REPRESSDB_B1_SINGLE_ACTIVE_STORE_V0_9_0.py
  RUN_THIS_ONLY_REPRESSDB_E0C_V0_2_0.py

Keep the partially completed directory:

  results_b3_scale_concurrency_v110/

Open this file in PyCharm and click Run.

Do not run the v1.1.0 main file again.

REPAIR
------
B3 v1.1.0 already used thread-local persistent file handles, but the physical
representation registry was shared by all query workers. WholeZstd contains a
stateful decompressor object. Concurrent query workers could therefore share
codec state even though their file handles were independent.

v1.1.1 gives every worker thread its own complete representation registry:

  whole_lz4
  whole_zstd
  block_lz4_64
  block_lz4_256
  block_lz4_1024

The background migration thread retains a separate writer registry. No
algorithm, scale, workload, query, migration, startup, or acceptance parameter
changes.

CHECKPOINT REUSE
----------------
Completed v1.1.0 cells are retained only because every recorded query in those
cells was checked against independent NumPy truth before the cell was committed.
The failed 4-thread cell was never committed and is rerun from its beginning.

The parent v1.1.0 experiment freeze remains the checkpoint identity. A chained
v1.1.1 implementation-repair hash is written separately into the result
directory and final assessment.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


VERSION = "1.1.1"
PARENT_SCRIPT_NAME = (
    "RUN_THIS_ONLY_REPRESSDB_B3_SCALE_CONCURRENCY_V1_1_0.py"
)
E0C_SCRIPT_NAME = "RUN_THIS_ONLY_REPRESSDB_E0C_V0_2_0.py"
OUTPUT_DIR_NAME = "results_b3_scale_concurrency_v110"
B2_RESULT_DIR_NAME = "results_b2_full_bdg2_scale_v100"

PREFLIGHT_THREADS = 4
PREFLIGHT_QUERIES_PER_THREAD = 500


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        )
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def load_module(path: Path, name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_thread_safe_codec_patch(core: Any) -> None:
    def thread_representations(self: Any) -> Dict[str, Any]:
        registry = getattr(
            self._thread_local,
            "representation_registry",
            None,
        )
        if registry is None:
            registry = self.base.representation_registry(
                core.ZSTD_LEVEL
            )
            self._thread_local.representation_registry = registry
        return registry

    def execute_query_thread_safe(
        self: Any,
        chunk_id: str,
        query: Any,
    ) -> Tuple[Any, int, int, int, Any, Any]:
        total_start = time.perf_counter_ns()
        lock = self._stripe(chunk_id)
        wait_start = time.perf_counter_ns()
        lock.acquire()
        wait_ns = time.perf_counter_ns() - wait_start
        try:
            entry = self.snapshot[chunk_id]
            stored = self._stored(entry)
            handles = self._cached_handles(stored)
            representations = self._thread_representations()
            execution_start = time.perf_counter_ns()
            result, stats = self.base.run_query_once(
                representations[stored.codec],
                stored,
                query,
                self.mode,
                handles.payload,
                handles.metadata,
            )
            execution_ns = (
                time.perf_counter_ns()
                - execution_start
            )
        finally:
            lock.release()

        total_ns = (
            time.perf_counter_ns()
            - total_start
        )
        return (
            result,
            int(total_ns),
            int(wait_ns),
            int(execution_ns),
            stats,
            entry,
        )

    core.ScaleStore._thread_representations = (
        thread_representations
    )
    core.ScaleStore.execute_query = (
        execute_query_thread_safe
    )


def run_thread_safety_preflight(
    project_dir: Path,
    core: Any,
    base: Any,
) -> Dict[str, Any]:
    scale_root = (
        project_dir
        / OUTPUT_DIR_NAME
        / "base_scales"
        / "logical_1GiB"
    )
    catalog_path = scale_root / "catalog.sqlite3"
    truth_path = (
        project_dir
        / B2_RESULT_DIR_NAME
        / "truth"
        / "data_scale_8192.float32.bin"
    )

    if not catalog_path.exists() or not truth_path.exists():
        return {
            "status": "skipped",
            "reason": (
                "The 1-GiB base catalog or B2 truth file "
                "does not exist yet. The full run will build it."
            ),
        }

    store = core.ScaleStore(
        scale_root,
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
            core.RAW_CHUNK_VALUES,
        ),
    )

    zstd_chunks = [
        chunk_id
        for chunk_id, entry in store.snapshot.items()
        if entry.active_codec == "whole_zstd"
    ]
    other_chunks = [
        chunk_id
        for chunk_id, entry in store.snapshot.items()
        if entry.active_codec != "whole_zstd"
    ]
    if not zstd_chunks:
        zstd_chunks = list(store.chunk_ids)

    start_event = threading.Event()

    def worker(
        worker_index: int,
    ) -> Dict[str, Any]:
        rng = random.Random(
            core.stable_seed(
                "v111-preflight",
                worker_index,
            )
        )
        start_event.wait()
        codec_counts: Dict[str, int] = {}
        for local_index in range(
            PREFLIGHT_QUERIES_PER_THREAD
        ):
            # Three quarters of the preflight deliberately stress WholeZstd,
            # the representation that contained shared mutable codec state.
            if local_index % 4 != 3:
                chunk_id = zstd_chunks[
                    rng.randrange(len(zstd_chunks))
                ]
            else:
                source = (
                    other_chunks
                    if other_chunks
                    else store.chunk_ids
                )
                chunk_id = source[
                    rng.randrange(len(source))
                ]

            entry = store.snapshot[chunk_id]
            template = store.templates[
                entry.template_index
            ]
            query = core.query_from_rng(
                base,
                template,
                rng,
                (
                    f"B3-v111-preflight:"
                    f"{worker_index}:"
                    f"{local_index}"
                ),
            )
            expected = base.execute_on_array(
                np.asarray(
                    truth[template.truth_index],
                    dtype=np.float32,
                ),
                query,
            )
            (
                actual,
                _,
                _,
                _,
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
                    "Thread-safe preflight mismatch: "
                    f"worker={worker_index}, "
                    f"local={local_index}, "
                    f"chunk={chunk_id}, "
                    f"codec={observed_entry.active_codec}, "
                    f"query={query.query_type}, "
                    f"actual={actual}, "
                    f"expected={expected}"
                )
            codec = str(
                observed_entry.active_codec
            )
            codec_counts[codec] = (
                codec_counts.get(codec, 0) + 1
            )
        return {
            "worker": worker_index,
            "queries": PREFLIGHT_QUERIES_PER_THREAD,
            "codec_counts": codec_counts,
        }

    started_ns = time.perf_counter_ns()
    futures = []
    try:
        with ThreadPoolExecutor(
            max_workers=PREFLIGHT_THREADS,
            thread_name_prefix="B3-v111-preflight",
        ) as executor:
            for worker_index in range(
                PREFLIGHT_THREADS
            ):
                futures.append(
                    executor.submit(
                        worker,
                        worker_index,
                    )
                )
            start_event.set()
            results = [
                future.result()
                for future in as_completed(
                    futures
                )
            ]
    finally:
        store.close()
        del truth

    combined: Dict[str, int] = {}
    for result in results:
        for codec, count in result[
            "codec_counts"
        ].items():
            combined[codec] = (
                combined.get(codec, 0)
                + int(count)
            )

    return {
        "status": "passed",
        "threads": PREFLIGHT_THREADS,
        "queries_per_thread": (
            PREFLIGHT_QUERIES_PER_THREAD
        ),
        "total_queries": (
            PREFLIGHT_THREADS
            * PREFLIGHT_QUERIES_PER_THREAD
        ),
        "codec_counts": combined,
        "whole_zstd_queries": int(
            combined.get("whole_zstd", 0)
        ),
        "elapsed_ns": int(
            time.perf_counter_ns()
            - started_ns
        ),
    }


def append_repair_metadata(
    project_dir: Path,
    output_dir: Path,
    core: Any,
    parent_script_path: Path,
    preflight: Dict[str, Any],
    return_code: int,
) -> None:
    parent_freeze_path = (
        output_dir
        / "b3_experiment_freeze_sha256.txt"
    )
    parent_freeze = (
        parent_freeze_path.read_text(
            encoding="utf-8"
        ).strip()
        if parent_freeze_path.exists()
        else None
    )

    repair_protocol = {
        "version": VERSION,
        "repair_class": (
            "implementation-only concurrent codec isolation repair"
        ),
        "parent_version": "1.1.0",
        "parent_script_sha256": sha256_file(
            parent_script_path
        ),
        "parent_experiment_freeze_sha256": (
            parent_freeze
        ),
        "wrapper_script_sha256": sha256_file(
            Path(__file__).resolve()
        ),
        "observed_failure": {
            "cell": (
                "logical_1GiB, 4 query threads, "
                "foreground_only, repeat 0"
            ),
            "symptom": (
                "a real file-backed query disagreed with "
                "independent NumPy truth"
            ),
            "completed_parent_cells_before_failure": 6,
        },
        "root_cause": (
            "File handles were thread-local, but all workers shared one "
            "physical-representation registry. WholeZstd retains a codec "
            "decompressor instance, so concurrent workers could share mutable "
            "codec state."
        ),
        "repair": {
            "query_worker_registry": (
                "one complete physical-representation registry per thread"
            ),
            "persistent_file_handles": (
                "remain thread-local and unchanged"
            ),
            "migration_registry": (
                "separate writer registry retained"
            ),
        },
        "unchanged": [
            "B2 real-data template corpus",
            "logical 1/5/10-GiB definitions",
            "catalogs already built",
            "startup protocol",
            "query generation",
            "truth validation",
            "thread counts",
            "modes",
            "repeats",
            "queries per cell",
            "background migrations per cell",
            "migration-log retention",
            "all acceptance thresholds",
        ],
        "checkpoint_reuse": {
            "rule": (
                "reuse only cells already committed after every query "
                "passed independent truth validation"
            ),
            "failed_cell": (
                "not committed and rerun from its beginning"
            ),
            "parent_checkpoint_identity_preserved": True,
        },
        "preflight": preflight,
        "parent_main_return_code": int(
            return_code
        ),
    }
    repair_path = (
        output_dir
        / "v111_thread_safety_repair.json"
    )
    write_json_atomic(
        repair_path,
        repair_protocol,
    )

    digest = hashlib.sha256()
    if parent_freeze is not None:
        digest.update(
            parent_freeze.encode("ascii")
        )
    digest.update(
        sha256_file(repair_path).encode("ascii")
    )
    repair_hash = digest.hexdigest()
    (
        output_dir
        / "v111_thread_safety_repair_sha256.txt"
    ).write_text(
        repair_hash + "\n",
        encoding="utf-8",
    )

    assessment_path = (
        output_dir
        / "b3_assessment.json"
    )
    if assessment_path.exists():
        assessment = json.loads(
            assessment_path.read_text(
                encoding="utf-8"
            )
        )
        assessment[
            "effective_implementation_version"
        ] = VERSION
        assessment[
            "parent_experiment_freeze_sha256"
        ] = parent_freeze
        assessment[
            "v111_thread_safety_repair_sha256"
        ] = repair_hash
        assessment[
            "implementation_repair"
        ] = repair_protocol
        write_json_atomic(
            assessment_path,
            assessment,
        )
        write_json_atomic(
            project_dir
            / "B3_SCALE_CONCURRENCY_ASSESSMENT_v111.json",
            assessment,
        )

    run_config_path = (
        output_dir / "run_config.json"
    )
    if run_config_path.exists():
        run_config = json.loads(
            run_config_path.read_text(
                encoding="utf-8"
            )
        )
        run_config[
            "effective_implementation_version"
        ] = VERSION
        run_config[
            "v111_thread_safety_repair_sha256"
        ] = repair_hash
        run_config[
            "thread_local_codec_registry"
        ] = True
        write_json_atomic(
            run_config_path,
            run_config,
        )


def main() -> int:
    project_dir = Path(
        __file__
    ).resolve().parent
    parent_script_path = (
        project_dir / PARENT_SCRIPT_NAME
    )
    e0c_script_path = (
        project_dir / E0C_SCRIPT_NAME
    )
    output_dir = (
        project_dir / OUTPUT_DIR_NAME
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 112)
    print(
        "RePressDB B3 v1.1.1 — "
        "thread-safe codec registry recovery"
    )
    print(
        "Completed truth-verified v1.1.0 cells "
        "will be reused."
    )
    print("=" * 112)

    core = load_module(
        parent_script_path,
        "repressdb_b3_v110_parent_for_v111",
    )
    install_thread_safe_codec_patch(
        core
    )
    base = load_module(
        e0c_script_path,
        "repressdb_e0c_for_b3_v111_preflight",
    )

    preflight = run_thread_safety_preflight(
        project_dir,
        core,
        base,
    )
    write_json_atomic(
        output_dir
        / "v111_thread_safety_preflight.json",
        preflight,
    )
    print(
        "[v1.1.1 preflight] "
        + json.dumps(
            preflight,
            ensure_ascii=False,
        )
    )
    if preflight.get("status") == "passed":
        print(
            "[v1.1.1] Four-thread codec-isolation "
            "preflight passed."
        )

    return_code = 1
    try:
        return_code = int(
            core.main()
        )
        return return_code
    finally:
        append_repair_metadata(
            project_dir,
            output_dir,
            core,
            parent_script_path,
            preflight,
            return_code,
        )


if __name__ == "__main__":
    raise SystemExit(main())
