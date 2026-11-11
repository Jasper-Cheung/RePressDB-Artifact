#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RePressDB J1-R Low-Memory Paired Launcher v1.7.0
================================================

USER ENTRY POINT
----------------
Run this file only.

This launcher imports Python standard-library modules only. It never imports
NumPy, Pandas, LZ4, Zstandard, or the RePressDB experiment modules.

For every experimental cell it starts one isolated heavy Python process,
waits for it to exit, and only then starts the next process. The launcher
therefore does not retain a second heavy copy of the experiment in memory.

The experiment reruns both implementations under the same lightweight process
structure:

    legacy B1 read path
    active-view cached read path

Each pair uses the same trace, policy, repeat, calibration, base-store snapshot,
and query realization. Implementation order is balanced across the 54 pairs.

Output:
    results_jss_j1r_low_memory_paired_v170/

Resume:
    Rerun this same file after interruption. Completed workers are skipped.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


VERSION = "1.7.0"
HEAVY_WORKER_NAME = "REPRESSDB_J1R_LOW_MEMORY_HEAVY_WORKER_V1_7_0.py"
OUTPUT_DIR_NAME = "results_jss_j1r_low_memory_paired_v170"

SETUP_MIN_AVAILABLE_PHYSICAL_BYTES = int(1.5 * 1024**3)
CELL_MIN_AVAILABLE_PHYSICAL_BYTES = int(1.0 * 1024**3)
MIN_AVAILABLE_PAGEFILE_BYTES = int(2.0 * 1024**3)
SETUP_TIMEOUT_SECONDS = 10 * 60
CELL_TIMEOUT_SECONDS = 5 * 60
SAMPLE_INTERVAL_SECONDS = 3
REQUIRED_STABLE_SAMPLES = 2
MIN_SYSTEM_DRIVE_FREE_BYTES = 12 * 1024**3
POST_WORKER_COOLDOWN_SECONDS = 3


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def memory_status() -> Dict[str, int]:
    if os.name == "nt":
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(
            ctypes.byref(status)
        ):
            raise OSError("GlobalMemoryStatusEx failed")
        return {
            "memory_load_percent": int(status.dwMemoryLoad),
            "total_physical_bytes": int(status.ullTotalPhys),
            "available_physical_bytes": int(status.ullAvailPhys),
            "total_pagefile_bytes": int(status.ullTotalPageFile),
            "available_pagefile_bytes": int(status.ullAvailPageFile),
            "total_virtual_bytes": int(status.ullTotalVirtual),
            "available_virtual_bytes": int(status.ullAvailVirtual),
        }

    # Non-Windows fallback for local validation.
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    total_pages = int(os.sysconf("SC_PHYS_PAGES"))
    available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    total = page_size * total_pages
    available = page_size * available_pages
    return {
        "memory_load_percent": int(
            round(100.0 * (1.0 - available / max(total, 1)))
        ),
        "total_physical_bytes": total,
        "available_physical_bytes": available,
        "total_pagefile_bytes": 1 << 60,
        "available_pagefile_bytes": 1 << 60,
        "total_virtual_bytes": 1 << 60,
        "available_virtual_bytes": 1 << 60,
    }


def system_drive_root() -> Path:
    if os.name == "nt":
        drive = os.environ.get("SystemDrive", "C:")
        return Path(drive + "\\")
    return Path("/")


def preflight(project: Path, output: Path) -> Dict[str, Any]:
    required = [
        project / HEAVY_WORKER_NAME,
        project
        / "RUN_THIS_ONLY_REPRESSDB_J1R_ACTIVE_VIEW_CACHE_CONFIRM_V1_6_2.py",
        project
        / "RUN_THIS_ONLY_REPRESSDB_JSS_MAJOR_REVISION_EXPERIMENTS_V1_5_1.py",
        project / "results_jss_major_revision_v151",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required files are missing:\n" + "\n".join(missing)
        )

    drive = system_drive_root()
    usage = shutil.disk_usage(drive)
    memory = memory_status()
    report = {
        "version": VERSION,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "project": str(project),
        "output": str(output),
        "system_drive": str(drive),
        "system_drive_free_bytes": int(usage.free),
        "memory": memory,
        "launcher_sha256": sha256_file(Path(__file__).resolve()),
        "heavy_worker_sha256": sha256_file(
            project / HEAVY_WORKER_NAME
        ),
    }
    write_json_atomic(output / "launcher_preflight.json", report)

    if usage.free < MIN_SYSTEM_DRIVE_FREE_BYTES:
        raise RuntimeError(
            f"System drive free space is only {usage.free / 1024**3:.2f} GiB. "
            f"At least {MIN_SYSTEM_DRIVE_FREE_BYTES / 1024**3:.0f} GiB is "
            "required for pagefile headroom and result files."
        )
    if memory["available_pagefile_bytes"] < MIN_AVAILABLE_PAGEFILE_BYTES:
        raise RuntimeError(
            "Windows commit/pagefile headroom is below 2 GiB. Set virtual "
            "memory to 'System managed size', restart Windows, and rerun."
        )
    return report


def wait_for_stable_memory(
    stage: str,
    minimum_physical_bytes: int,
    timeout_seconds: int,
) -> Dict[str, Any]:
    start = time.monotonic()
    stable = 0
    observations: List[Dict[str, Any]] = []
    while True:
        status = memory_status()
        elapsed = time.monotonic() - start
        physical_ok = (
            status["available_physical_bytes"] >= minimum_physical_bytes
        )
        pagefile_ok = (
            status["available_pagefile_bytes"]
            >= MIN_AVAILABLE_PAGEFILE_BYTES
        )
        if physical_ok and pagefile_ok:
            stable += 1
        else:
            stable = 0

        row = {
            "elapsed_seconds": float(elapsed),
            **status,
            "physical_ok": bool(physical_ok),
            "pagefile_ok": bool(pagefile_ok),
            "stable_samples": int(stable),
        }
        observations.append(row)
        print(
            f"[memory] {stage}: "
            f"phys={status['available_physical_bytes'] / 1024**3:.2f} GiB "
            f"(need {minimum_physical_bytes / 1024**3:.2f}), "
            f"commit={status['available_pagefile_bytes'] / 1024**3:.2f} GiB "
            f"(need {MIN_AVAILABLE_PAGEFILE_BYTES / 1024**3:.2f}), "
            f"stable={stable}/{REQUIRED_STABLE_SAMPLES}",
            flush=True,
        )

        if stable >= REQUIRED_STABLE_SAMPLES:
            return {
                "passed": True,
                "stage": stage,
                "minimum_physical_bytes": int(minimum_physical_bytes),
                "minimum_pagefile_bytes": int(
                    MIN_AVAILABLE_PAGEFILE_BYTES
                ),
                "waited_seconds": float(elapsed),
                "observations": observations,
            }
        if elapsed >= timeout_seconds:
            return {
                "passed": False,
                "stage": stage,
                "minimum_physical_bytes": int(minimum_physical_bytes),
                "minimum_pagefile_bytes": int(
                    MIN_AVAILABLE_PAGEFILE_BYTES
                ),
                "waited_seconds": float(elapsed),
                "observations": observations,
            }
        time.sleep(SAMPLE_INTERVAL_SECONDS)


def tee_subprocess(
    arguments: Sequence[str],
    log_path: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(f'"{arg}"' for arg in arguments), flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(arguments),
            cwd=str(Path(__file__).resolve().parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return int(process.wait())


def complete_marker_valid(
    output_dir: Path,
    freeze_hash: str,
) -> bool:
    marker_path = output_dir / "complete.json"
    summary_path = output_dir / "summary.json"
    if not marker_path.exists() or not summary_path.exists():
        return False
    try:
        marker = read_json(marker_path)
    except Exception:
        return False
    return bool(
        marker.get("complete", False)
        and marker.get("freeze_sha256") == freeze_hash
        and marker.get("summary_sha256") == sha256_file(summary_path)
    )


def repair_interrupted_atomic_marker(
    output_dir: Path,
    freeze_hash: str,
) -> bool:
    """
    Salvage a worker that finished all measured work but the OS stopped during
    the final complete.json.tmp -> complete.json rename.
    """
    marker_path = output_dir / "complete.json"
    temporary_path = output_dir / "complete.json.tmp"
    summary_path = output_dir / "summary.json"
    if marker_path.exists():
        return complete_marker_valid(output_dir, freeze_hash)
    if not (temporary_path.exists() and summary_path.exists()):
        return False
    try:
        marker = read_json(temporary_path)
    except Exception:
        return False
    if (
        marker.get("complete", False)
        and marker.get("freeze_sha256") == freeze_hash
        and marker.get("summary_sha256") == sha256_file(summary_path)
    ):
        os.replace(temporary_path, marker_path)
        print(f"[repair] recovered final marker: {marker_path}", flush=True)
        return True
    return False


def run_heavy(
    project: Path,
    mode: str,
    log_path: Path,
    config_path: Optional[Path] = None,
) -> int:
    arguments = [
        sys.executable,
        str(project / HEAVY_WORKER_NAME),
        "--mode",
        mode,
        "--project",
        str(project),
    ]
    if config_path is not None:
        arguments.extend(["--config", str(config_path)])
    return tee_subprocess(arguments, log_path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-preflight-memory-wait",
        action="store_true",
        help=(
            "Use only after confirming system-managed virtual memory and "
            "stable available memory manually."
        ),
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    project = Path(__file__).resolve().parent
    output = project / OUTPUT_DIR_NAME
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print("RePressDB J1-R Low-Memory Paired Launcher v1.7.0")
    print("Only the lightweight launcher remains resident between workers.")
    print(f"Output: {output}")
    print("=" * 110)

    preflight(project, output)

    if not arguments.skip_preflight_memory_wait:
        setup_gate = wait_for_stable_memory(
            "setup",
            SETUP_MIN_AVAILABLE_PHYSICAL_BYTES,
            SETUP_TIMEOUT_SECONDS,
        )
        write_json_atomic(output / "setup_memory_gate.json", setup_gate)
        if not setup_gate["passed"]:
            raise RuntimeError(
                "Stable setup headroom was not available within 10 minutes. "
                "Restart Windows, close browsers/WSL/other Python processes, "
                "and rerun this same launcher."
            )

    plan_path = output / "experiment_plan.json"
    setup_marker = output / "setup_complete.json"
    if not (plan_path.exists() and setup_marker.exists()):
        code = run_heavy(
            project,
            "setup",
            output / "setup.log",
        )
        if code != 0:
            raise RuntimeError(
                f"Setup worker failed with exit code {code}. "
                f"See {output / 'setup.log'}"
            )

    plan = read_json(plan_path)
    freeze_hash = str(plan["freeze_sha256"])
    run_sequence = list(plan["run_sequence"])
    total = len(run_sequence)
    progress_path = output / "launcher_progress.json"

    completed = 0
    for index, cell in enumerate(run_sequence, start=1):
        cell_output = Path(str(cell["output_dir"]))
        if complete_marker_valid(cell_output, freeze_hash):
            completed += 1
            print(
                f"[resume {index}/{total}] {cell['pair_id']} "
                f"{cell['implementation']} complete",
                flush=True,
            )
            continue
        if repair_interrupted_atomic_marker(
            cell_output,
            freeze_hash,
        ):
            completed += 1
            continue

        if not arguments.skip_preflight_memory_wait:
            gate = wait_for_stable_memory(
                (
                    f"worker {index}/{total} "
                    f"{cell['implementation']} {cell['policy']} "
                    f"{cell['trace']} size={cell['chunk_size']}"
                ),
                CELL_MIN_AVAILABLE_PHYSICAL_BYTES,
                CELL_TIMEOUT_SECONDS,
            )
            write_json_atomic(cell_output / "launcher_memory_gate.json", gate)
            if not gate["passed"]:
                write_json_atomic(
                    progress_path,
                    {
                        "version": VERSION,
                        "status": "paused_memory_gate",
                        "completed_workers": completed,
                        "total_workers": total,
                        "next_worker_index": index,
                        "next_cell": cell,
                    },
                )
                raise RuntimeError(
                    "The next cell did not obtain stable 1.0-GiB physical "
                    "and 2-GiB commit headroom within five minutes. The run "
                    "is safely checkpointed; close memory-intensive programs "
                    "and rerun this launcher."
                )

        code = run_heavy(
            project,
            "cell",
            cell_output / "worker.log",
            Path(str(cell["config_path"])),
        )
        if code != 0:
            write_json_atomic(
                progress_path,
                {
                    "version": VERSION,
                    "status": "worker_failed",
                    "completed_workers": completed,
                    "total_workers": total,
                    "failed_worker_index": index,
                    "failed_cell": cell,
                    "worker_exit_code": code,
                    "worker_log": str(cell_output / "worker.log"),
                },
            )
            raise RuntimeError(
                f"Worker {index}/{total} failed with exit code {code}. "
                f"See {cell_output / 'worker.log'}"
            )

        if not complete_marker_valid(cell_output, freeze_hash):
            raise RuntimeError(
                f"Worker returned 0 but its final marker is invalid: "
                f"{cell_output}"
            )
        completed += 1
        write_json_atomic(
            progress_path,
            {
                "version": VERSION,
                "status": "running",
                "completed_workers": completed,
                "total_workers": total,
                "last_worker_index": index,
                "last_cell": cell,
            },
        )
        print(
            f"[complete {index}/{total}] pair={cell['pair_id']} "
            f"implementation={cell['implementation']}",
            flush=True,
        )
        time.sleep(POST_WORKER_COOLDOWN_SECONDS)

    write_json_atomic(
        progress_path,
        {
            "version": VERSION,
            "status": "aggregating",
            "completed_workers": completed,
            "total_workers": total,
        },
    )
    aggregate_code = run_heavy(
        project,
        "aggregate",
        output / "aggregate.log",
    )
    if aggregate_code not in (0, 2):
        raise RuntimeError(
            f"Aggregate worker failed with exit code {aggregate_code}. "
            f"See {output / 'aggregate.log'}"
        )

    assessment_path = output / "v170_assessment.json"
    assessment = read_json(assessment_path)
    write_json_atomic(
        progress_path,
        {
            "version": VERSION,
            "status": (
                "complete"
                if assessment.get("acceptance_checks", {}).get(
                    "all_ten", False
                )
                else "complete_with_failed_checks"
            ),
            "completed_workers": completed,
            "total_workers": total,
            "assessment": str(assessment_path),
            "result_archives": assessment.get("result_archives", {}),
        },
    )

    print("=" * 110)
    print("Low-memory paired experiment finished.")
    print(f"Assessment: {assessment_path}")
    for key, value in assessment.get("result_archives", {}).items():
        if key.endswith("_zip"):
            print(f"{key}: {value}")
    print("=" * 110)
    return 0 if aggregate_code == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
