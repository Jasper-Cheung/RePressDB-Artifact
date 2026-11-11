#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RePressDB E0-B v0.2.1

File-backed confirmatory pilot for workload-dependent historical time-series
physical representations.

Default behavior:
  1) auto-detect and run PEMS04 from the fixed directory below;
  2) auto-detect and run a holdout subset of PEMS08;
  3) write separate result directories.

The PEMS08 series used in E0-A are excluded from the E0-B holdout run.
This script reports warm-cache query latency and exact logical bytes read.
It does not claim uncached SSD latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import lz4.frame
import numpy as np
import pandas as pd
import scipy
import zstandard as zstd
from scipy.stats import wilcoxon

VERSION = "0.2.2"

PEMS04_DIR = Path(r"E:\IPM\大学\科研\鲁棒性预测\data\Traffic-main\PEMS04")
PEMS08_DIR = Path(r"E:\IPM\大学\科研\鲁棒性预测\data\Traffic-main\PEMS08")

# These 32 PEMS08 series were used in the uploaded E0-A pilot and are excluded
# from the confirmatory holdout subset.
PEMS08_E0A_SERIES = {
    3, 4, 17, 21, 22, 26, 34, 51, 52, 56, 60, 66, 67, 70, 72, 80,
    81, 92, 93, 95, 101, 102, 104, 122, 131, 133, 137, 140, 150, 156,
    158, 162,
}

QUERY_TYPES = (
    "point",
    "range_scan",
    "range_sum",
    "range_avg",
    "range_min",
    "range_max",
    "threshold_count",
)

# Workloads are fixed before the E0-B run. They intentionally distinguish
# access granularity rather than changing weights after observing E0-B data.
WORKLOADS: Dict[str, Dict[str, float]] = {
    "scan_heavy": {
        "point": 0.02,
        "short_scan": 0.03,
        "medium_scan": 0.10,
        "long_scan": 0.70,
        "short_agg": 0.02,
        "medium_agg": 0.03,
        "long_agg": 0.05,
        "threshold": 0.05,
    },
    "point_heavy": {
        "point": 0.80,
        "short_scan": 0.12,
        "medium_scan": 0.02,
        "long_scan": 0.01,
        "short_agg": 0.03,
        "medium_agg": 0.01,
        "long_agg": 0.00,
        "threshold": 0.01,
    },
    "aggregate_heavy": {
        "point": 0.02,
        "short_scan": 0.01,
        "medium_scan": 0.02,
        "long_scan": 0.02,
        "short_agg": 0.10,
        "medium_agg": 0.20,
        "long_agg": 0.55,
        "threshold": 0.08,
    },
}

for _name, _weights in WORKLOADS.items():
    if not math.isclose(sum(_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"Workload {_name} weights do not sum to 1")


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    query_type: str
    query_class: str
    start: int
    end: int
    range_fraction: float
    threshold: Optional[float] = None
    threshold_quantile: Optional[float] = None


@dataclass(frozen=True)
class ChunkUnit:
    chunk_id: str
    series_index: int
    chunk_start: int
    chunk_size: int
    values: np.ndarray


@dataclass
class StoredChunk:
    codec: str
    n: int
    data_path: Path
    metadata_path: Optional[Path]
    payload_size_bytes: int
    metadata_size_bytes: int
    metadata: Dict[str, Any]
    file_handle: Any = None

    @property
    def stored_size_bytes(self) -> int:
        return self.payload_size_bytes + self.metadata_size_bytes


class PhysicalRepresentation:
    name: str

    def encode_to_files(self, x: np.ndarray, stem: Path) -> StoredChunk:
        raise NotImplementedError

    def decode_all(self, stored: StoredChunk) -> Tuple[np.ndarray, int]:
        raise NotImplementedError

    def execute(self, stored: StoredChunk, query: QuerySpec) -> Tuple[Any, int]:
        raise NotImplementedError


class WholeLZ4(PhysicalRepresentation):
    name = "whole_lz4"

    def encode_to_files(self, x: np.ndarray, stem: Path) -> StoredChunk:
        arr = np.ascontiguousarray(x, dtype=np.float32)
        payload = lz4.frame.compress(arr.tobytes(order="C"), compression_level=0)
        data_path = stem.with_suffix(".bin")
        with open(data_path, "wb") as f:
            f.write(payload)
            f.flush()
        return StoredChunk(self.name, arr.size, data_path, None, data_path.stat().st_size, 0, {})

    def decode_all(self, stored: StoredChunk) -> Tuple[np.ndarray, int]:
        if stored.file_handle is not None:
            f = stored.file_handle
            f.seek(0)
            payload = f.read()
        else:
            with open(stored.data_path, "rb") as f:
                payload = f.read()
        raw = lz4.frame.decompress(payload)
        return np.frombuffer(raw, dtype=np.float32, count=stored.n).copy(), len(payload)

    def execute(self, stored: StoredChunk, query: QuerySpec) -> Tuple[Any, int]:
        x, read_bytes = self.decode_all(stored)
        return execute_on_array(x, query), read_bytes


class WholeZstd(PhysicalRepresentation):
    name = "whole_zstd"

    def __init__(self, level: int = 3):
        self.level = level
        self.compressor = zstd.ZstdCompressor(level=level)
        self.decompressor = zstd.ZstdDecompressor()

    def encode_to_files(self, x: np.ndarray, stem: Path) -> StoredChunk:
        arr = np.ascontiguousarray(x, dtype=np.float32)
        payload = self.compressor.compress(arr.tobytes(order="C"))
        data_path = stem.with_suffix(".bin")
        with open(data_path, "wb") as f:
            f.write(payload)
            f.flush()
        return StoredChunk(self.name, arr.size, data_path, None, data_path.stat().st_size, 0, {"level": self.level})

    def decode_all(self, stored: StoredChunk) -> Tuple[np.ndarray, int]:
        if stored.file_handle is not None:
            f = stored.file_handle
            f.seek(0)
            payload = f.read()
        else:
            with open(stored.data_path, "rb") as f:
                payload = f.read()
        raw = self.decompressor.decompress(payload, max_output_size=stored.n * 4)
        return np.frombuffer(raw, dtype=np.float32, count=stored.n).copy(), len(payload)

    def execute(self, stored: StoredChunk, query: QuerySpec) -> Tuple[Any, int]:
        x, read_bytes = self.decode_all(stored)
        return execute_on_array(x, query), read_bytes


class BlockLZ4(PhysicalRepresentation):
    """LZ4-compressed microblocks with exact per-block summaries."""

    def __init__(self, block_size: int):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = int(block_size)
        self.name = f"block_lz4_{self.block_size}"

    def encode_to_files(self, x: np.ndarray, stem: Path) -> StoredChunk:
        arr = np.ascontiguousarray(x, dtype=np.float32)
        offsets: List[int] = []
        lengths: List[int] = []
        counts: List[int] = []
        sums: List[float] = []
        mins: List[float] = []
        maxs: List[float] = []
        payloads: List[bytes] = []
        offset = 0
        for start in range(0, arr.size, self.block_size):
            block = arr[start : start + self.block_size]
            payload = lz4.frame.compress(block.tobytes(order="C"), compression_level=0)
            offsets.append(offset)
            lengths.append(len(payload))
            counts.append(int(block.size))
            sums.append(float(np.sum(block, dtype=np.float64)))
            mins.append(float(np.min(block)))
            maxs.append(float(np.max(block)))
            payloads.append(payload)
            offset += len(payload)

        data_path = stem.with_suffix(".bin")
        with open(data_path, "wb") as f:
            for payload in payloads:
                f.write(payload)
            f.flush()

        offsets_arr = np.asarray(offsets, dtype=np.int64)
        lengths_arr = np.asarray(lengths, dtype=np.int32)
        counts_arr = np.asarray(counts, dtype=np.int32)
        sums_arr = np.asarray(sums, dtype=np.float64)
        mins_arr = np.asarray(mins, dtype=np.float32)
        maxs_arr = np.asarray(maxs, dtype=np.float32)
        metadata_path = stem.with_suffix(".meta.bin")
        with open(metadata_path, "wb") as f:
            f.write(np.asarray([self.block_size, len(offsets)], dtype=np.int32).tobytes())
            f.write(offsets_arr.tobytes())
            f.write(lengths_arr.tobytes())
            f.write(counts_arr.tobytes())
            f.write(sums_arr.tobytes())
            f.write(mins_arr.tobytes())
            f.write(maxs_arr.tobytes())
            f.flush()
        metadata = {
            "offsets": offsets_arr,
            "lengths": lengths_arr,
            "counts": counts_arr,
            "sums": sums_arr,
            "mins": mins_arr,
            "maxs": maxs_arr,
            "block_size": self.block_size,
        }
        return StoredChunk(
            self.name,
            arr.size,
            data_path,
            metadata_path,
            data_path.stat().st_size,
            metadata_path.stat().st_size,
            metadata,
        )

    def _read_block(self, f: Any, stored: StoredChunk, block_idx: int) -> Tuple[np.ndarray, int]:
        m = stored.metadata
        offset = int(m["offsets"][block_idx])
        length = int(m["lengths"][block_idx])
        count = int(m["counts"][block_idx])
        f.seek(offset)
        payload = f.read(length)
        if len(payload) != length:
            raise IOError(f"Short read for {stored.codec} block {block_idx}")
        raw = lz4.frame.decompress(payload)
        return np.frombuffer(raw, dtype=np.float32, count=count), length

    def decode_all(self, stored: StoredChunk) -> Tuple[np.ndarray, int]:
        out = np.empty(stored.n, dtype=np.float32)
        bs = int(stored.metadata["block_size"])
        bytes_read = 0
        owns_handle = stored.file_handle is None
        f = open(stored.data_path, "rb") if owns_handle else stored.file_handle
        try:
            for block_idx in range(len(stored.metadata["counts"])):
                block, nbytes = self._read_block(f, stored, block_idx)
                start = block_idx * bs
                out[start : start + block.size] = block
                bytes_read += nbytes
        finally:
            if owns_handle:
                f.close()
        return out, bytes_read

    def _touched_blocks(self, stored: StoredChunk, start: int, end: int) -> Iterable[Tuple[int, int, int, bool]]:
        bs = int(stored.metadata["block_size"])
        first = start // bs
        last = (end - 1) // bs
        for block_idx in range(first, last + 1):
            block_start = block_idx * bs
            block_len = int(stored.metadata["counts"][block_idx])
            block_end = block_start + block_len
            local_start = max(start, block_start) - block_start
            local_end = min(end, block_end) - block_start
            full = local_start == 0 and local_end == block_len
            yield block_idx, local_start, local_end, full

    def execute(self, stored: StoredChunk, query: QuerySpec) -> Tuple[Any, int]:
        m = stored.metadata
        bs = int(m["block_size"])
        bytes_read = 0
        owns_handle = stored.file_handle is None
        f = open(stored.data_path, "rb") if owns_handle else stored.file_handle
        try:
            if query.query_type == "point":
                block_idx = query.start // bs
                block, nbytes = self._read_block(f, stored, block_idx)
                return float(block[query.start - block_idx * bs]), nbytes

            if query.query_type == "range_scan":
                total = 0.0
                for block_idx, local_start, local_end, _ in self._touched_blocks(stored, query.start, query.end):
                    block, nbytes = self._read_block(f, stored, block_idx)
                    total += float(np.sum(block[local_start:local_end], dtype=np.float64))
                    bytes_read += nbytes
                return total, bytes_read

            start_block = query.start // bs
            end_block = (query.end - 1) // bs
            if start_block == end_block:
                block, nbytes = self._read_block(f, stored, start_block)
                local_start = query.start - start_block * bs
                local_end = query.end - start_block * bs
                local_query = QuerySpec(
                    query.query_id,
                    query.query_type,
                    query.query_class,
                    local_start,
                    local_end,
                    query.range_fraction,
                    query.threshold,
                    query.threshold_quantile,
                )
                return execute_on_array(block, local_query), nbytes

            partials: List[np.ndarray] = []
            full_first = start_block
            full_last_exclusive = end_block + 1
            if query.start % bs != 0:
                block, nbytes = self._read_block(f, stored, start_block)
                partials.append(block[query.start - start_block * bs :])
                bytes_read += nbytes
                full_first += 1
            if query.end % bs != 0:
                block, nbytes = self._read_block(f, stored, end_block)
                partials.append(block[: query.end - end_block * bs])
                bytes_read += nbytes
                full_last_exclusive -= 1

            if query.query_type == "threshold_count":
                if query.threshold is None:
                    raise ValueError("threshold_count missing threshold")
                threshold = float(query.threshold)
                result = sum(int(np.count_nonzero(part > threshold)) for part in partials)
                if full_first < full_last_exclusive:
                    mins = m["mins"][full_first:full_last_exclusive]
                    maxs = m["maxs"][full_first:full_last_exclusive]
                    counts = m["counts"][full_first:full_last_exclusive]
                    result += int(np.sum(counts[mins > threshold], dtype=np.int64))
                    ambiguous = np.flatnonzero((mins <= threshold) & (maxs > threshold)) + full_first
                    for block_idx in ambiguous:
                        block, nbytes = self._read_block(f, stored, int(block_idx))
                        result += int(np.count_nonzero(block > threshold))
                        bytes_read += nbytes
                return result, bytes_read

            total = sum(float(np.sum(part, dtype=np.float64)) for part in partials)
            count = sum(int(part.size) for part in partials)
            current_min = min((float(np.min(part)) for part in partials), default=math.inf)
            current_max = max((float(np.max(part)) for part in partials), default=-math.inf)
            if full_first < full_last_exclusive:
                total += float(np.sum(m["sums"][full_first:full_last_exclusive], dtype=np.float64))
                count += int(np.sum(m["counts"][full_first:full_last_exclusive], dtype=np.int64))
                current_min = min(current_min, float(np.min(m["mins"][full_first:full_last_exclusive])))
                current_max = max(current_max, float(np.max(m["maxs"][full_first:full_last_exclusive])))
            if query.query_type == "range_sum":
                return total, bytes_read
            if query.query_type == "range_avg":
                return total / count, bytes_read
            if query.query_type == "range_min":
                return current_min, bytes_read
            if query.query_type == "range_max":
                return current_max, bytes_read
            raise ValueError(f"Unsupported query type: {query.query_type}")
        finally:
            if owns_handle:
                f.close()


def representation_registry(zstd_level: int) -> Dict[str, PhysicalRepresentation]:
    reps: List[PhysicalRepresentation] = [
        WholeLZ4(),
        WholeZstd(level=zstd_level),
        BlockLZ4(64),
        BlockLZ4(256),
        BlockLZ4(1024),
    ]
    return {r.name: r for r in reps}


def execute_on_array(x: np.ndarray, query: QuerySpec) -> Any:
    if query.query_type == "point":
        return float(x[query.start])
    selected = x[query.start:query.end]
    if selected.size == 0:
        raise ValueError("Empty query range")
    if query.query_type in {"range_scan", "range_sum"}:
        return float(np.sum(selected, dtype=np.float64))
    if query.query_type == "range_avg":
        return float(np.mean(selected, dtype=np.float64))
    if query.query_type == "range_min":
        return float(np.min(selected))
    if query.query_type == "range_max":
        return float(np.max(selected))
    if query.query_type == "threshold_count":
        if query.threshold is None:
            raise ValueError("threshold_count missing threshold")
        return int(np.count_nonzero(selected > query.threshold))
    raise ValueError(f"Unsupported query type: {query.query_type}")


def results_equal(a: Any, b: Any) -> bool:
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(a) == int(b)
    return math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=1e-5)


def stable_rng(seed: int, token: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{token}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def query_class(query_type: str, fraction: float) -> str:
    if query_type == "point":
        return "point"
    if query_type == "threshold_count":
        return "threshold"
    if query_type == "range_scan":
        if fraction <= 0.01 + 1e-12:
            return "short_scan"
        if fraction <= 0.10 + 1e-12:
            return "medium_scan"
        return "long_scan"
    if fraction <= 0.01 + 1e-12:
        return "short_agg"
    if fraction <= 0.10 + 1e-12:
        return "medium_agg"
    return "long_agg"


def build_queries(x: np.ndarray, chunk_id: str, seed: int, windows_per_fraction: int) -> List[QuerySpec]:
    rng = stable_rng(seed, chunk_id)
    n = int(x.size)
    queries: List[QuerySpec] = []
    point_count = max(6, windows_per_fraction * 3)
    for i, p in enumerate(rng.integers(0, n, size=point_count)):
        queries.append(QuerySpec(f"{chunk_id}:point:{i}", "point", "point", int(p), int(p + 1), 1.0 / n))

    fractions = (0.001, 0.01, 0.10, 0.50, 1.00)
    for fraction in fractions:
        length = max(1, min(n, int(round(n * fraction))))
        actual_fraction = length / n
        starts = [0] if length == n else rng.integers(0, n - length + 1, size=windows_per_fraction).tolist()
        for j, start in enumerate(starts):
            start_i = int(start)
            end_i = start_i + length
            for qtype in ("range_scan", "range_sum", "range_avg", "range_min", "range_max"):
                # Classify by the preregistered target fraction rather than the rounded
                # realized fraction. For example, round(2048 * 0.10) / 2048 is
                # slightly above 0.10 and must still remain a medium query.
                qclass = query_class(qtype, fraction)
                queries.append(QuerySpec(
                    f"{chunk_id}:{qtype}:f{actual_fraction:.6f}:{j}", qtype, qclass,
                    start_i, end_i, actual_fraction,
                ))

    for quantile in (0.50, 0.90, 0.99):
        threshold = float(np.quantile(x, quantile))
        for fraction in (0.01, 0.10, 1.00):
            length = max(1, min(n, int(round(n * fraction))))
            actual_fraction = length / n
            starts = [0] if length == n else rng.integers(0, n - length + 1, size=windows_per_fraction).tolist()
            for j, start in enumerate(starts):
                start_i = int(start)
                queries.append(QuerySpec(
                    f"{chunk_id}:threshold:q{quantile:.2f}:f{actual_fraction:.6f}:{j}",
                    "threshold_count", "threshold", start_i, start_i + length,
                    actual_fraction, threshold, quantile,
                ))
    return queries


def auto_detect_data_file(directory: Path, dataset_name: str) -> Path:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    candidates = list(directory.rglob("*.npz")) + list(directory.rglob("*.npy"))
    if not candidates:
        raise FileNotFoundError(f"No .npz or .npy found under {directory}")

    name_token = dataset_name.lower().replace("-", "").replace("_", "")
    bad_tokens = ("adj", "distance", "dist", "graph", "edge", "index", "mean", "std")

    def score(path: Path) -> Tuple[int, int]:
        normalized = path.stem.lower().replace("-", "").replace("_", "")
        s = 0
        if name_token in normalized:
            s += 100
        if path.suffix.lower() == ".npz":
            s += 10
        if any(t in normalized for t in bad_tokens):
            s -= 100
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        return s, size

    ranked = sorted(candidates, key=score, reverse=True)
    selected = ranked[0]
    if score(selected)[0] < 0:
        raise FileNotFoundError(f"Cannot reliably identify main data file under {directory}")
    print(f"[{dataset_name}] data file: {selected}")
    return selected


def interpolate_nonfinite(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    out = np.asarray(matrix, dtype=np.float32).copy()
    mask = ~np.isfinite(out)
    fraction = float(mask.mean())
    if not mask.any():
        return out, fraction
    for j in range(out.shape[1]):
        col = out[:, j]
        valid = np.isfinite(col)
        if not valid.any():
            col[:] = 0.0
        elif not valid.all():
            idx = np.arange(col.size)
            col[~valid] = np.interp(idx[~valid], idx[valid], col[valid]).astype(np.float32)
    return out, fraction


def load_dataset(path: Path, feature_index: int, max_series: int, seed: int,
                 excluded_series: Optional[set[int]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        keys = list(archive.keys())
        key = "data" if "data" in keys else keys[0]
        arr = np.asarray(archive[key])
        meta: Dict[str, Any] = {"source": str(path), "array_key": key, "available_keys": keys, "original_shape": list(arr.shape)}
    elif suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        meta = {"source": str(path), "original_shape": list(arr.shape)}
    else:
        raise ValueError(f"Unsupported file: {path}")

    if arr.ndim == 1:
        matrix = arr[:, None]
    elif arr.ndim == 2:
        matrix = arr
    elif arr.ndim == 3:
        if not 0 <= feature_index < arr.shape[2]:
            raise IndexError("feature_index out of bounds")
        matrix = arr[:, :, feature_index]
        meta["feature_index"] = feature_index
    else:
        matrix = arr.reshape(arr.shape[0], -1)
        meta["flattened_trailing_dimensions"] = True

    matrix, fraction = interpolate_nonfinite(np.asarray(matrix, dtype=np.float32))
    all_indices = np.arange(matrix.shape[1], dtype=int)
    if excluded_series:
        allowed = np.asarray([i for i in all_indices if int(i) not in excluded_series], dtype=int)
    else:
        allowed = all_indices
    if allowed.size == 0:
        raise ValueError("No series left after exclusion")
    rng = np.random.default_rng(seed)
    if allowed.size > max_series:
        chosen = np.sort(rng.choice(allowed, size=max_series, replace=False))
    else:
        chosen = allowed
    matrix = matrix[:, chosen]
    meta["selected_original_series_indices"] = chosen.tolist()
    meta["excluded_series_indices"] = sorted(excluded_series) if excluded_series else []
    meta["loaded_shape"] = list(matrix.shape)
    meta["imputed_nonfinite_fraction"] = fraction
    meta["dtype"] = str(matrix.dtype)
    return matrix, meta


def synthetic_dataset(seed: int, max_series: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    t = 20000
    s = max(8, max_series)
    ts = np.arange(t, dtype=np.float32)
    matrix = np.empty((t, s), dtype=np.float32)
    for j in range(s):
        matrix[:, j] = (
            20 * np.sin(2 * np.pi * ts / (288 + 11 * (j % 5)))
            + 5 * np.sin(2 * np.pi * ts / (48 + 3 * (j % 4)))
            + (ts // (1000 + 50 * (j % 3))) * (j % 3)
            + rng.normal(0, 0.5 + 0.1 * (j % 3), size=t)
        ).astype(np.float32)
    return matrix[:, :max_series], {"source": "synthetic", "original_shape": [t, s], "loaded_shape": [t, max_series]}


def extract_chunks(matrix: np.ndarray, chunk_sizes: Sequence[int], max_chunks_per_size: int,
                   seed: int) -> List[ChunkUnit]:
    rng = np.random.default_rng(seed)
    t, s = matrix.shape
    chunks: List[ChunkUnit] = []
    for chunk_size in chunk_sizes:
        if chunk_size > t:
            print(f"[WARN] skip chunk_size={chunk_size}; length={t}")
            continue
        candidates = [(series_idx, start) for series_idx in range(s)
                      for start in range(0, t - chunk_size + 1, chunk_size)]
        if len(candidates) > max_chunks_per_size:
            ids = np.sort(rng.choice(len(candidates), size=max_chunks_per_size, replace=False))
            candidates = [candidates[int(i)] for i in ids]
        for series_idx, start in candidates:
            values = np.ascontiguousarray(matrix[start:start + chunk_size, series_idx], dtype=np.float32)
            chunks.append(ChunkUnit(f"cs{chunk_size}_s{series_idx}_t{start}", series_idx, start, chunk_size, values))
    if not chunks:
        raise ValueError("No chunks generated")
    return chunks


def median_timed(callable_fn, warmups: int, repeats: int) -> Tuple[int, Any, int]:
    for _ in range(warmups):
        callable_fn()
    times: List[int] = []
    last_result: Any = None
    last_bytes = 0
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        last_result, last_bytes = callable_fn()
        times.append(time.perf_counter_ns() - t0)
    return int(np.median(np.asarray(times, dtype=np.int64))), last_result, int(last_bytes)


def encode_one(rep: PhysicalRepresentation, x: np.ndarray, stem: Path, repeats: int) -> Tuple[StoredChunk, int]:
    # First write creates the retained representation.
    if stem.with_suffix(".bin").exists():
        stem.with_suffix(".bin").unlink()
    if stem.with_suffix(".meta.bin").exists():
        stem.with_suffix(".meta.bin").unlink()
    t0 = time.perf_counter_ns()
    stored = rep.encode_to_files(x, stem)
    first_ns = time.perf_counter_ns() - t0
    if repeats <= 1:
        return stored, int(first_ns)

    extra: List[int] = [int(first_ns)]
    for r in range(1, repeats):
        trial_stem = stem.parent / f"{stem.name}__trial{r}"
        t0 = time.perf_counter_ns()
        trial = rep.encode_to_files(x, trial_stem)
        extra.append(time.perf_counter_ns() - t0)
        trial.data_path.unlink(missing_ok=True)
        if trial.metadata_path:
            trial.metadata_path.unlink(missing_ok=True)

    if not stored.data_path.is_file():
        raise FileNotFoundError(
            f"Retained encoded payload disappeared after repeat timing: {stored.data_path}"
        )
    if stored.metadata_path is not None and not stored.metadata_path.is_file():
        raise FileNotFoundError(
            f"Retained encoded metadata disappeared after repeat timing: {stored.metadata_path}"
        )
    return stored, int(np.median(np.asarray(extra, dtype=np.int64)))


def benchmark_dataset(matrix: np.ndarray, dataset_meta: Dict[str, Any], dataset_label: str,
                      output_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_dir = output_dir / "_storage"
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    storage_dir.mkdir(parents=True)

    reps = representation_registry(args.zstd_level)
    chunks = extract_chunks(matrix, args.chunk_sizes, args.max_chunks_per_size, args.seed)
    print(f"[{dataset_label}] chunks={len(chunks)}, representations={len(reps)}")

    codec_rows: List[Dict[str, Any]] = []
    query_rows: List[Dict[str, Any]] = []

    for chunk_no, chunk in enumerate(chunks, 1):
        chunk_dir = storage_dir / chunk.chunk_id
        chunk_dir.mkdir(parents=True)
        stored_map: Dict[str, StoredChunk] = {}
        for rep_name, rep in reps.items():
            stem = chunk_dir / rep_name
            stored, encode_write_ns = encode_one(rep, chunk.values, stem, args.encode_repeats)
            stored.file_handle = open(stored.data_path, "rb")
            decoded, read_bytes = rep.decode_all(stored)
            if not np.array_equal(decoded.view(np.uint32), chunk.values.view(np.uint32)):
                raise AssertionError(f"Lossless round trip failed: {chunk.chunk_id}, {rep_name}")
            full_read_decode_ns, _, full_read_bytes = median_timed(
                lambda rep=rep, stored=stored: rep.decode_all(stored),
                args.warmups,
                args.decode_repeats,
            )
            stored_map[rep_name] = stored
            codec_rows.append({
                "dataset": dataset_label,
                "chunk_id": chunk.chunk_id,
                "series_index": chunk.series_index,
                "chunk_start": chunk.chunk_start,
                "chunk_size": chunk.chunk_size,
                "codec": rep_name,
                "raw_size_bytes": chunk.values.nbytes,
                "payload_size_bytes": stored.payload_size_bytes,
                "metadata_size_bytes": stored.metadata_size_bytes,
                "stored_size_bytes": stored.stored_size_bytes,
                "compression_ratio": stored.stored_size_bytes / chunk.values.nbytes,
                "encode_write_ns": encode_write_ns,
                "full_read_decode_ns": full_read_decode_ns,
                "full_read_bytes": full_read_bytes,
            })

        queries = build_queries(chunk.values, chunk.chunk_id, args.seed, args.windows_per_fraction)
        expected_query_classes = set(next(iter(WORKLOADS.values())).keys())
        observed_query_classes = {query.query_class for query in queries}
        missing_query_classes = expected_query_classes - observed_query_classes
        if missing_query_classes:
            raise RuntimeError(
                f"Missing workload query classes for {chunk.chunk_id}: "
                f"{sorted(missing_query_classes)}"
            )
        for query in queries:
            truth = execute_on_array(chunk.values, query)
            for rep_name, rep in reps.items():
                stored = stored_map[rep_name]
                latency_ns, result, bytes_read = median_timed(
                    lambda rep=rep, stored=stored, query=query: rep.execute(stored, query),
                    args.warmups,
                    args.query_repeats,
                )
                if not results_equal(result, truth):
                    raise AssertionError(
                        f"Incorrect query result: {chunk.chunk_id}, {rep_name}, {query.query_id}, "
                        f"result={result}, truth={truth}"
                    )
                query_rows.append({
                    "dataset": dataset_label,
                    "query_id": query.query_id,
                    "chunk_id": chunk.chunk_id,
                    "series_index": chunk.series_index,
                    "chunk_size": chunk.chunk_size,
                    "codec": rep_name,
                    "query_type": query.query_type,
                    "query_class": query.query_class,
                    "start": query.start,
                    "end": query.end,
                    "range_length": query.end - query.start,
                    "range_fraction": query.range_fraction,
                    "threshold": query.threshold,
                    "threshold_quantile": query.threshold_quantile,
                    "latency_ns": latency_ns,
                    "bytes_read": bytes_read,
                    "result": result,
                    "ground_truth": truth,
                    "correct": True,
                })

        for stored in stored_map.values():
            if stored.file_handle is not None:
                stored.file_handle.close()
                stored.file_handle = None

        if chunk_no == 1 or chunk_no % max(1, len(chunks) // 20) == 0 or chunk_no == len(chunks):
            print(f"[{dataset_label}] {chunk_no}/{len(chunks)} chunks")

    codec_df = pd.DataFrame(codec_rows)
    query_df = pd.DataFrame(query_rows)
    codec_df.to_csv(output_dir / "codec_records.csv", index=False)
    query_df.to_csv(output_dir / "query_records.csv", index=False)

    codec_summary = summarize_codecs(codec_df)
    query_summary = summarize_queries(query_df)
    tests = paired_tests(query_df)
    workload_costs, best_layouts, drift = workload_analysis(codec_df, query_df)
    decision = make_decision(best_layouts, drift)

    codec_summary.to_csv(output_dir / "codec_summary.csv", index=False)
    query_summary.to_csv(output_dir / "query_summary.csv", index=False)
    tests.to_csv(output_dir / "statistical_tests.csv", index=False)
    workload_costs.to_csv(output_dir / "workload_costs.csv", index=False)
    best_layouts.to_csv(output_dir / "workload_best_layouts.csv", index=False)
    drift.to_csv(output_dir / "workload_drift_analysis.csv", index=False)

    dataset_meta = dict(dataset_meta)
    dataset_meta["dataset_label"] = dataset_label
    dataset_meta["num_chunks"] = len(chunks)
    dataset_meta["chunk_sizes"] = args.chunk_sizes
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(dataset_meta, f, ensure_ascii=False, indent=2)
    with open(output_dir / "e0b_decision.json", "w", encoding="utf-8") as f:
        json.dump(decision, f, ensure_ascii=False, indent=2, allow_nan=True)

    config = vars(args).copy()
    config.update({
        "version": VERSION,
        "dataset_label": dataset_label,
        "measurement_scope": "file-backed warm-cache latency plus exact logical bytes read; metadata cached in memory",
        "representations": list(reps),
        "workloads": WORKLOADS,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "lz4": getattr(lz4.frame, "__version__", "unknown"),
            "zstandard": zstd.__version__,
            "timer": "time.perf_counter_ns",
        },
    })
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    if not args.keep_storage:
        shutil.rmtree(storage_dir)

    print(f"[{dataset_label}] E0-B pass={decision['overall_e0b_pass']}")
    return decision


def summarize_codecs(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["compression_ratio", "encode_write_ns", "full_read_decode_ns", "stored_size_bytes"]
    rows: List[Dict[str, Any]] = []
    for (chunk_size, codec), group in df.groupby(["chunk_size", "codec"]):
        for metric in metrics:
            vals = group[metric].to_numpy(dtype=float)
            rows.append({
                "chunk_size": int(chunk_size),
                "codec": codec,
                "metric": metric,
                "n_chunks": len(group),
                "median": float(np.median(vals)),
                "mean": float(np.mean(vals)),
            })
    return pd.DataFrame(rows)


def summarize_queries(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, group in df.groupby(["chunk_size", "query_class", "codec"]):
        chunk_size, qclass, codec = keys
        rows.append({
            "chunk_size": int(chunk_size),
            "query_class": qclass,
            "codec": codec,
            "n_query_instances": len(group),
            "n_chunks": group["chunk_id"].nunique(),
            "median_latency_ns": float(group["latency_ns"].median()),
            "mean_latency_ns": float(group["latency_ns"].mean()),
            "median_bytes_read": float(group["bytes_read"].median()),
            "mean_bytes_read": float(group["bytes_read"].mean()),
        })
    return pd.DataFrame(rows)


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (n - rank) * float(p_values[idx])
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    codecs = sorted(df["codec"].unique())
    rows: List[Dict[str, Any]] = []
    for (chunk_size, qclass), group in df.groupby(["chunk_size", "query_class"]):
        pivot = group.pivot(index="query_id", columns="codec", values="latency_ns").dropna()
        local: List[Dict[str, Any]] = []
        ps: List[float] = []
        for i, a in enumerate(codecs):
            for b in codecs[i + 1:]:
                if a not in pivot or b not in pivot:
                    continue
                try:
                    p = float(wilcoxon(pivot[a], pivot[b], alternative="two-sided", zero_method="wilcox").pvalue)
                except ValueError:
                    p = 1.0
                ratio = pivot[a].to_numpy(dtype=float) / pivot[b].to_numpy(dtype=float)
                row = {
                    "chunk_size": int(chunk_size),
                    "query_class": qclass,
                    "codec_a": a,
                    "codec_b": b,
                    "n_pairs": len(pivot),
                    "median_ratio_a_over_b": float(np.median(ratio)),
                    "wilcoxon_p": p,
                }
                local.append(row)
                ps.append(p)
        adjusted = holm_adjust(ps) if ps else []
        for row, p_adj in zip(local, adjusted):
            row["wilcoxon_p_holm"] = p_adj
            rows.append(row)
    return pd.DataFrame(rows)


def workload_analysis(codec_df: pd.DataFrame, query_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    class_medians = (
        query_df.groupby(["chunk_id", "chunk_size", "codec", "query_class"], as_index=False)["latency_ns"]
        .median()
    )
    workload_rows: List[Dict[str, Any]] = []
    for workload, weights in WORKLOADS.items():
        temp = class_medians.copy()
        temp["weight"] = temp["query_class"].map(weights).fillna(0.0)
        temp["weighted_latency_ns"] = temp["latency_ns"] * temp["weight"]
        costs = temp.groupby(["chunk_id", "chunk_size", "codec"], as_index=False)["weighted_latency_ns"].sum()
        costs["workload"] = workload
        workload_rows.extend(costs.to_dict("records"))
    workload_df = pd.DataFrame(workload_rows)

    idx = workload_df.groupby(["chunk_id", "chunk_size", "workload"])["weighted_latency_ns"].idxmin()
    best = workload_df.loc[idx, ["chunk_id", "chunk_size", "workload", "codec", "weighted_latency_ns"]].copy()
    best = best.rename(columns={"codec": "best_codec", "weighted_latency_ns": "best_expected_latency_ns"})

    codec_costs = codec_df.set_index(["chunk_id", "codec"])[["encode_write_ns", "full_read_decode_ns", "stored_size_bytes"]]
    old = best[best["workload"] == "scan_heavy"].set_index("chunk_id")
    drift_rows: List[Dict[str, Any]] = []
    for new_workload in ("point_heavy", "aggregate_heavy"):
        new = best[best["workload"] == new_workload].set_index("chunk_id")
        new_costs = workload_df[workload_df["workload"] == new_workload].set_index(["chunk_id", "codec"])
        for chunk_id, old_row in old.iterrows():
            old_codec = str(old_row["best_codec"])
            new_codec = str(new.loc[chunk_id, "best_codec"])
            old_cost = float(new_costs.loc[(chunk_id, old_codec), "weighted_latency_ns"])
            new_cost = float(new_costs.loc[(chunk_id, new_codec), "weighted_latency_ns"])
            saving = old_cost - new_cost
            rewrite_ns = float(codec_costs.loc[(chunk_id, old_codec), "full_read_decode_ns"] +
                               codec_costs.loc[(chunk_id, new_codec), "encode_write_ns"])
            rewrite_bytes = int(codec_costs.loc[(chunk_id, old_codec), "stored_size_bytes"] +
                                codec_costs.loc[(chunk_id, new_codec), "stored_size_bytes"])
            drift_rows.append({
                "chunk_id": chunk_id,
                "chunk_size": int(old_row["chunk_size"]),
                "old_workload": "scan_heavy",
                "new_workload": new_workload,
                "old_codec": old_codec,
                "new_codec": new_codec,
                "codec_switch": old_codec != new_codec,
                "old_layout_expected_latency_ns": old_cost,
                "new_opt_expected_latency_ns": new_cost,
                "regret_ratio": old_cost / new_cost if new_cost > 0 else math.inf,
                "saving_per_query_ns": saving,
                "rewrite_cost_ns": rewrite_ns,
                "rewrite_io_bytes": rewrite_bytes,
                "break_even_future_queries": rewrite_ns / saving if saving > 0 else math.inf,
            })
    return workload_df, best, pd.DataFrame(drift_rows)


def bootstrap_median_ci(values: np.ndarray, seed: int = 20260718, iterations: int = 4000) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=float)
    for i in range(iterations):
        samples[i] = np.median(rng.choice(values, size=values.size, replace=True))
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def make_decision(best: pd.DataFrame, drift: pd.DataFrame) -> Dict[str, Any]:
    strata: Dict[str, Any] = {}
    qualifying_chunk_sizes: set[int] = set()
    c2 = False
    c3 = False
    for (new_workload, chunk_size), group in drift.groupby(["new_workload", "chunk_size"]):
        finite = group[np.isfinite(group["regret_ratio"])].copy()
        switch_fraction = float(finite["codec_switch"].mean())
        median_regret = float(finite["regret_ratio"].median())
        lo, hi = bootstrap_median_ci(finite["regret_ratio"].to_numpy())
        switched = finite[(finite["codec_switch"]) & np.isfinite(finite["break_even_future_queries"])]
        fraction_break_even_5000 = float((switched["break_even_future_queries"] <= 5000).mean()) if not switched.empty else 0.0
        if switch_fraction >= 0.50:
            qualifying_chunk_sizes.add(int(chunk_size))
        if switch_fraction >= 0.50 and median_regret >= 1.20 and lo >= 1.05:
            c2 = True
        if switch_fraction >= 0.50 and fraction_break_even_5000 >= 0.50:
            c3 = True
        strata[f"{new_workload}|{int(chunk_size)}"] = {
            "n_chunks": int(len(finite)),
            "switch_fraction": switch_fraction,
            "median_regret_ratio": median_regret,
            "bootstrap_95ci_median_regret": [lo, hi],
            "fraction_switched_break_even_le_5000": fraction_break_even_5000,
        }

    # C1 requires workload-specific layouts at two distinct chunk-size strata.
    c1 = len(qualifying_chunk_sizes) >= 2
    modal = (
        best.groupby(["workload", "chunk_size"])["best_codec"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
    )
    return {
        "version": VERSION,
        "preregistered_criteria": {
            "C1_workload_specific_layouts": {
                "pass": bool(c1),
                "definition": "At least two distinct chunk sizes have a drift scenario where >=50% of chunks switch best representation.",
            },
            "C2_material_regret": {
                "pass": bool(c2),
                "definition": "At least one stratum has switch fraction >=50%, median regret >=1.20, and bootstrap 95% lower bound >=1.05.",
            },
            "C3_file_backed_amortization": {
                "pass": bool(c3),
                "definition": "At least one stratum with switch fraction >=50% has >=50% of switched chunks break even within 5000 future queries.",
            },
        },
        "modal_best_by_workload_and_chunk_size": modal.to_dict("records"),
        "strata": strata,
        "overall_e0b_pass": bool(c1 and c2 and c3),
        "measurement_warning": (
            "Query latency is warm-cache and metadata is cached in memory. Exact logical bytes read are recorded separately. "
            "This is not an uncached SSD or full DBMS claim."
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=None, help="Optional single .npz/.npy input; default runs fixed PEMS04 and PEMS08 directories")
    p.add_argument("--dataset-label", type=str, default="CUSTOM")
    p.add_argument("--output", type=str, default="results_e0b_custom")
    p.add_argument("--feature-index", type=int, default=0)
    p.add_argument("--chunk-sizes", type=str, default="2048,8192,16384")
    p.add_argument("--max-series", type=int, default=32)
    p.add_argument("--max-chunks-per-size", type=int, default=96)
    p.add_argument("--zstd-level", type=int, default=3)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--encode-repeats", type=int, default=3)
    p.add_argument("--decode-repeats", type=int, default=5)
    p.add_argument("--query-repeats", type=int, default=5)
    p.add_argument("--windows-per-fraction", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260718)
    p.add_argument("--keep-storage", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    args.chunk_sizes = [int(x.strip()) for x in args.chunk_sizes.split(",") if x.strip()]
    if args.quick:
        args.chunk_sizes = [2048, 8192]
        args.max_series = min(args.max_series, 4)
        args.max_chunks_per_size = 6
        args.encode_repeats = 1
        args.decode_repeats = 2
        args.query_repeats = 2
        args.windows_per_fraction = 1
    return args


def main() -> int:
    args = parse_args()
    print("=" * 80)
    print(f"RePressDB E0-B v{VERSION}")
    print("File-backed warm-cache benchmark; exact logical bytes read are recorded.")
    print("=" * 80)

    if args.quick and args.input is None:
        matrix, meta = synthetic_dataset(args.seed, args.max_series)
        benchmark_dataset(matrix, meta, "SYNTHETIC_QUICK", Path("results_e0b_quick"), args)
        return 0

    if args.input:
        path = Path(args.input)
        matrix, meta = load_dataset(path, args.feature_index, args.max_series, args.seed)
        benchmark_dataset(matrix, meta, args.dataset_label, Path(args.output), args)
        return 0

    jobs = [
        ("PEMS04_CONFIRMATORY", PEMS04_DIR, Path("results_e0b_pems04_v022"), None),
        ("PEMS08_HOLDOUT_REPLICATION", PEMS08_DIR, Path("results_e0b_pems08_holdout_v022"), PEMS08_E0A_SERIES),
    ]
    decisions: Dict[str, Any] = {}
    for label, directory, output, excluded in jobs:
        data_file = auto_detect_data_file(directory, label)
        matrix, meta = load_dataset(data_file, args.feature_index, args.max_series, args.seed, excluded)
        decisions[label] = benchmark_dataset(matrix, meta, label, output, args)

    with open("E0B_COMBINED_DECISIONS_v022.json", "w", encoding="utf-8") as f:
        json.dump(decisions, f, ensure_ascii=False, indent=2, allow_nan=True)
    print("Done. Upload both v0.2.2 result directories and E0B_COMBINED_DECISIONS_v022.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
