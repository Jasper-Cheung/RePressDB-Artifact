#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RePressDB E0-A in-memory feasibility benchmark.

Purpose
-------
This program tests three preregistered feasibility questions:
1) Do different query workloads prefer different lossless physical representations?
2) Does a layout selected for an old workload incur measurable regret after workload drift?
3) Can selective recompression be amortized by future query savings?

The benchmark is intentionally in-memory. It isolates representation/query trade-offs
from storage-device and operating-system page-cache effects. Disk-level experiments
belong to the next stage only if E0 passes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import lz4.frame
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import friedmanchisquare, wilcoxon
import zstandard as zstd


VERSION = "0.1.2"

# 用户本机的固定数据目录。直接运行本文件时，会自动依次运行 PEMS04 和 PEMS08。
DEFAULT_DATASET_DIRS: Dict[str, Path] = {
    "PEMS04": Path(r"E:\IPM\大学\科研\鲁棒性预测\data\Traffic-main\PEMS04"),
    "PEMS08": Path(r"E:\IPM\大学\科研\鲁棒性预测\data\Traffic-main\PEMS08"),
}
SUPPORTED_DATA_SUFFIXES = {".npz", ".npy", ".csv", ".txt"}
CODEC_ORDER = ["raw", "lz4", "zstd", "xor_zstd", "indexed_zstd"]
DEPLOYABLE_CODECS = ["lz4", "zstd", "xor_zstd", "indexed_zstd"]
QUERY_ORDER = ["point", "range_scan", "range_sum", "range_avg", "range_min", "range_max", "threshold_count"]


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    query_type: str
    start: int
    end: int
    threshold: Optional[float] = None
    threshold_quantile: Optional[float] = None


@dataclass
class EncodedChunk:
    codec: str
    n: int
    payload: Any
    size_bytes: int
    metadata: Dict[str, Any]


@dataclass
class ChunkUnit:
    chunk_id: str
    series_index: int
    start: int
    chunk_size: int
    values: np.ndarray


@dataclass
class Workload:
    name: str
    weights: Dict[str, float]


WORKLOADS: Tuple[Workload, ...] = (
    Workload(
        "scan_heavy",
        {
            "point": 0.03,
            "range_scan": 0.70,
            "range_sum": 0.07,
            "range_avg": 0.05,
            "range_min": 0.03,
            "range_max": 0.02,
            "threshold_count": 0.10,
        },
    ),
    Workload(
        "aggregate_heavy",
        {
            "point": 0.03,
            "range_scan": 0.02,
            "range_sum": 0.27,
            "range_avg": 0.23,
            "range_min": 0.18,
            "range_max": 0.17,
            "threshold_count": 0.10,
        },
    ),
    Workload(
        "sparse_lookup_heavy",
        {
            "point": 0.85,
            "range_scan": 0.03,
            "range_sum": 0.02,
            "range_avg": 0.02,
            "range_min": 0.02,
            "range_max": 0.02,
            "threshold_count": 0.04,
        },
    ),
)


class Codec:
    name: str

    def encode(self, x: np.ndarray) -> EncodedChunk:
        raise NotImplementedError

    def decode(self, encoded: EncodedChunk) -> np.ndarray:
        raise NotImplementedError

    def execute(self, encoded: EncodedChunk, query: QuerySpec) -> Any:
        x = self.decode(encoded)
        return execute_on_array(x, query)


class RawCodec(Codec):
    name = "raw"

    def encode(self, x: np.ndarray) -> EncodedChunk:
        arr = np.ascontiguousarray(x, dtype=np.float32)
        payload = arr.tobytes(order="C")
        return EncodedChunk(self.name, arr.size, payload, len(payload), {})

    def decode(self, encoded: EncodedChunk) -> np.ndarray:
        return np.frombuffer(encoded.payload, dtype=np.float32, count=encoded.n).copy()

    def execute(self, encoded: EncodedChunk, query: QuerySpec) -> Any:
        view = np.frombuffer(encoded.payload, dtype=np.float32, count=encoded.n)
        return execute_on_array(view, query)


class LZ4Codec(Codec):
    name = "lz4"

    def __init__(self, compression_level: int = 0):
        self.compression_level = compression_level

    def encode(self, x: np.ndarray) -> EncodedChunk:
        arr = np.ascontiguousarray(x, dtype=np.float32)
        payload = lz4.frame.compress(
            arr.tobytes(order="C"),
            compression_level=self.compression_level,
            block_linked=True,
            store_size=True,
        )
        return EncodedChunk(self.name, arr.size, payload, len(payload) + 8, {"level": self.compression_level})

    def decode(self, encoded: EncodedChunk) -> np.ndarray:
        raw = lz4.frame.decompress(encoded.payload)
        return np.frombuffer(raw, dtype=np.float32, count=encoded.n).copy()


class ZstdCodec(Codec):
    name = "zstd"

    def __init__(self, level: int = 3):
        self.level = level
        self.compressor = zstd.ZstdCompressor(level=level)
        self.decompressor = zstd.ZstdDecompressor()

    def encode(self, x: np.ndarray) -> EncodedChunk:
        arr = np.ascontiguousarray(x, dtype=np.float32)
        payload = self.compressor.compress(arr.tobytes(order="C"))
        return EncodedChunk(self.name, arr.size, payload, len(payload) + 8, {"level": self.level})

    def decode(self, encoded: EncodedChunk) -> np.ndarray:
        raw = self.decompressor.decompress(encoded.payload, max_output_size=encoded.n * 4)
        return np.frombuffer(raw, dtype=np.float32, count=encoded.n).copy()


class XorZstdCodec(Codec):
    """Bitwise-lossless XOR transform followed by Zstandard."""

    name = "xor_zstd"

    def __init__(self, level: int = 3):
        self.level = level
        self.compressor = zstd.ZstdCompressor(level=level)
        self.decompressor = zstd.ZstdDecompressor()

    def encode(self, x: np.ndarray) -> EncodedChunk:
        arr = np.ascontiguousarray(x, dtype=np.float32)
        bits = arr.view(np.uint32)
        transformed = np.empty_like(bits)
        if bits.size:
            transformed[0] = bits[0]
            transformed[1:] = np.bitwise_xor(bits[1:], bits[:-1])
        payload = self.compressor.compress(transformed.tobytes(order="C"))
        return EncodedChunk(self.name, arr.size, payload, len(payload) + 8, {"level": self.level})

    def decode(self, encoded: EncodedChunk) -> np.ndarray:
        raw = self.decompressor.decompress(encoded.payload, max_output_size=encoded.n * 4)
        transformed = np.frombuffer(raw, dtype=np.uint32, count=encoded.n)
        bits = np.bitwise_xor.accumulate(transformed, dtype=np.uint32)
        return bits.view(np.float32).copy()


class IndexedZstdCodec(Codec):
    """
    Exact query-friendly representation.

    The chunk is divided into microblocks. Each microblock is compressed separately.
    Exact min/max/sum/count metadata permits direct evaluation of aligned aggregate
    portions and exact pruning for threshold_count. Partial microblocks are decoded.
    """

    name = "indexed_zstd"

    def __init__(self, level: int = 3, microblock_size: int = 128):
        if microblock_size <= 0:
            raise ValueError("microblock_size must be positive")
        self.level = level
        self.microblock_size = microblock_size
        self.compressor = zstd.ZstdCompressor(level=level)
        self.decompressor = zstd.ZstdDecompressor()

    def encode(self, x: np.ndarray) -> EncodedChunk:
        arr = np.ascontiguousarray(x, dtype=np.float32)
        blocks: List[bytes] = []
        sums: List[float] = []
        mins: List[float] = []
        maxs: List[float] = []
        counts: List[int] = []
        for start in range(0, arr.size, self.microblock_size):
            block = arr[start : start + self.microblock_size]
            blocks.append(self.compressor.compress(block.tobytes(order="C")))
            sums.append(float(np.sum(block, dtype=np.float64)))
            mins.append(float(np.min(block)))
            maxs.append(float(np.max(block)))
            counts.append(int(block.size))
        # Serializable-size estimate: compressed payload + per-block offset/length/count,
        # float64 sum, float32 min/max, plus a small global header.
        payload_bytes = sum(len(b) for b in blocks)
        metadata_bytes = len(blocks) * (4 + 4 + 4 + 8 + 4 + 4) + 24
        payload = {
            "blocks": blocks,
            "sums": np.asarray(sums, dtype=np.float64),
            "mins": np.asarray(mins, dtype=np.float32),
            "maxs": np.asarray(maxs, dtype=np.float32),
            "counts": np.asarray(counts, dtype=np.int32),
        }
        return EncodedChunk(
            self.name,
            arr.size,
            payload,
            payload_bytes + metadata_bytes,
            {"level": self.level, "microblock_size": self.microblock_size, "num_blocks": len(blocks)},
        )

    def _decode_block(self, encoded: EncodedChunk, block_idx: int) -> np.ndarray:
        payload = encoded.payload
        count = int(payload["counts"][block_idx])
        raw = self.decompressor.decompress(payload["blocks"][block_idx], max_output_size=count * 4)
        return np.frombuffer(raw, dtype=np.float32, count=count)

    def decode(self, encoded: EncodedChunk) -> np.ndarray:
        out = np.empty(encoded.n, dtype=np.float32)
        bs = int(encoded.metadata["microblock_size"])
        for block_idx in range(len(encoded.payload["blocks"])):
            block = self._decode_block(encoded, block_idx)
            start = block_idx * bs
            out[start : start + block.size] = block
        return out

    def _iter_touched_blocks(self, encoded: EncodedChunk, start: int, end: int) -> Iterable[Tuple[int, int, int, bool]]:
        bs = int(encoded.metadata["microblock_size"])
        first = start // bs
        last = (end - 1) // bs
        for block_idx in range(first, last + 1):
            block_start = block_idx * bs
            block_end = min(block_start + bs, encoded.n)
            local_start = max(start, block_start) - block_start
            local_end = min(end, block_end) - block_start
            full = local_start == 0 and local_end == block_end - block_start
            yield block_idx, local_start, local_end, full

    def execute(self, encoded: EncodedChunk, query: QuerySpec) -> Any:
        payload = encoded.payload
        if query.query_type == "point":
            bs = int(encoded.metadata["microblock_size"])
            block_idx = query.start // bs
            block = self._decode_block(encoded, block_idx)
            return float(block[query.start - block_idx * bs])

        if query.query_type == "range_scan":
            pieces: List[np.ndarray] = []
            for block_idx, local_start, local_end, _ in self._iter_touched_blocks(encoded, query.start, query.end):
                block = self._decode_block(encoded, block_idx)
                pieces.append(block[local_start:local_end])
            selected = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)
            return float(np.sum(selected, dtype=np.float64))

        if query.query_type in {"range_sum", "range_avg", "range_min", "range_max", "threshold_count"}:
            bs = int(encoded.metadata["microblock_size"])
            start_block = query.start // bs
            end_block = (query.end - 1) // bs

            # A range contained in one microblock needs exactly one decompression.
            if start_block == end_block:
                block = self._decode_block(encoded, start_block)
                local_start = query.start - start_block * bs
                local_end = query.end - start_block * bs
                return execute_on_array(
                    block,
                    QuerySpec(
                        query.query_id,
                        query.query_type,
                        local_start,
                        local_end,
                        threshold=query.threshold,
                        threshold_quantile=query.threshold_quantile,
                    ),
                )

            partial_values: List[np.ndarray] = []
            full_first = start_block
            full_last_exclusive = end_block + 1

            if query.start % bs != 0:
                first = self._decode_block(encoded, start_block)
                partial_values.append(first[query.start - start_block * bs :])
                full_first += 1
            if query.end % bs != 0:
                last = self._decode_block(encoded, end_block)
                partial_values.append(last[: query.end - end_block * bs])
                full_last_exclusive -= 1

            if query.query_type == "threshold_count":
                assert query.threshold is not None
                threshold = float(query.threshold)
                result = sum(int(np.count_nonzero(part > threshold)) for part in partial_values)
                if full_first < full_last_exclusive:
                    mins = payload["mins"][full_first:full_last_exclusive]
                    maxs = payload["maxs"][full_first:full_last_exclusive]
                    counts = payload["counts"][full_first:full_last_exclusive]
                    result += int(np.sum(counts[mins > threshold], dtype=np.int64))
                    ambiguous = np.flatnonzero((mins <= threshold) & (maxs > threshold)) + full_first
                    for block_idx in ambiguous:
                        block = self._decode_block(encoded, int(block_idx))
                        result += int(np.count_nonzero(block > threshold))
                return result

            total = sum(float(np.sum(part, dtype=np.float64)) for part in partial_values)
            count = sum(int(part.size) for part in partial_values)
            current_min = min((float(np.min(part)) for part in partial_values), default=math.inf)
            current_max = max((float(np.max(part)) for part in partial_values), default=-math.inf)
            if full_first < full_last_exclusive:
                total += float(np.sum(payload["sums"][full_first:full_last_exclusive], dtype=np.float64))
                count += int(np.sum(payload["counts"][full_first:full_last_exclusive], dtype=np.int64))
                current_min = min(current_min, float(np.min(payload["mins"][full_first:full_last_exclusive])))
                current_max = max(current_max, float(np.max(payload["maxs"][full_first:full_last_exclusive])))
            if query.query_type == "range_sum":
                return total
            if query.query_type == "range_avg":
                return total / count
            if query.query_type == "range_min":
                return current_min
            return current_max

        raise ValueError(f"Unsupported query type: {query.query_type}")


def codec_registry(zstd_level: int, lz4_level: int, microblock_size: int) -> Dict[str, Codec]:
    return {
        "raw": RawCodec(),
        "lz4": LZ4Codec(compression_level=lz4_level),
        "zstd": ZstdCodec(level=zstd_level),
        "xor_zstd": XorZstdCodec(level=zstd_level),
        "indexed_zstd": IndexedZstdCodec(level=zstd_level, microblock_size=microblock_size),
    }


def execute_on_array(x: np.ndarray, query: QuerySpec) -> Any:
    if query.query_type == "point":
        return float(x[query.start])
    selected = x[query.start : query.end]
    if selected.size == 0:
        raise ValueError("Empty query range")
    if query.query_type == "range_scan":
        return float(np.sum(selected, dtype=np.float64))
    if query.query_type == "range_sum":
        return float(np.sum(selected, dtype=np.float64))
    if query.query_type == "range_avg":
        return float(np.mean(selected, dtype=np.float64))
    if query.query_type == "range_min":
        return float(np.min(selected))
    if query.query_type == "range_max":
        return float(np.max(selected))
    if query.query_type == "threshold_count":
        if query.threshold is None:
            raise ValueError("threshold_count requires threshold")
        return int(np.count_nonzero(selected > query.threshold))
    raise ValueError(f"Unsupported query type: {query.query_type}")


def values_equal(actual: Any, expected: Any, query_type: str) -> bool:
    if query_type == "threshold_count":
        return int(actual) == int(expected)
    return bool(np.isclose(float(actual), float(expected), rtol=1e-7, atol=1e-7 * max(1.0, abs(float(expected)))))


def median_timed_ns(func, warmups: int, repeats: int) -> Tuple[int, Any]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    result = None
    for _ in range(max(0, warmups)):
        result = func()
    timings: List[int] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            result = func()
            timings.append(time.perf_counter_ns() - t0)
    finally:
        if gc_was_enabled:
            gc.enable()
    return int(statistics.median(timings)), result


def interpolate_nonfinite(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    out = np.asarray(matrix, dtype=np.float32).copy()
    nonfinite = ~np.isfinite(out)
    fraction = float(nonfinite.mean()) if out.size else 0.0
    if not nonfinite.any():
        return out, fraction
    for col in range(out.shape[1]):
        y = out[:, col]
        good = np.isfinite(y)
        if not good.any():
            raise ValueError(f"Series column {col} contains no finite values")
        if good.all():
            continue
        idx = np.arange(y.size)
        y[~good] = np.interp(idx[~good], idx[good], y[good]).astype(np.float32)
    return out, fraction


def resolve_dataset_file(dataset_dir: Path, dataset_name: str) -> Path:
    """Resolve the primary time-series array file inside a dataset directory.

    The resolver prefers NPZ/NPY files whose names contain the dataset name or
    ``data`` and penalizes common auxiliary files such as adjacency matrices,
    distance tables, graph files, and index files.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"固定数据目录不存在: {dataset_dir}")
    if dataset_dir.is_file():
        if dataset_dir.suffix.lower() not in SUPPORTED_DATA_SUFFIXES:
            raise ValueError(f"不支持的数据文件格式: {dataset_dir}")
        return dataset_dir

    candidates = [
        path for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_DATA_SUFFIXES
    ]
    if not candidates:
        raise FileNotFoundError(
            f"目录中没有找到 .npz/.npy/.csv/.txt 数据文件: {dataset_dir}"
        )

    target = dataset_name.lower()
    negative_tokens = (
        "distance", "dist", "adj", "adjacency", "graph", "edge",
        "index", "idx", "node", "sensor", "location", "label",
    )

    def score(path: Path) -> Tuple[int, int]:
        name = path.name.lower()
        stem = path.stem.lower()
        suffix_score = {".npz": 400, ".npy": 300, ".csv": 100, ".txt": 50}.get(path.suffix.lower(), 0)
        value = suffix_score
        if stem == target:
            value += 1000
        elif target in stem:
            value += 700
        if stem in {"data", "dataset"}:
            value += 350
        elif "data" in stem:
            value += 180
        if any(token in stem for token in negative_tokens):
            value -= 900
        # Prefer the larger file when names receive the same semantic score.
        return value, path.stat().st_size

    ranked = sorted(candidates, key=score, reverse=True)
    selected = ranked[0]
    best_score = score(selected)[0]
    if best_score < 0:
        readable = "\n".join(f"  - {x}" for x in ranked[:10])
        raise FileNotFoundError(
            f"无法可靠识别 {dataset_name} 的主数据文件。候选文件为:\n{readable}"
        )

    print(f"[{dataset_name}] 自动识别数据文件: {selected}")
    if len(ranked) > 1:
        print(f"[{dataset_name}] 其余候选文件数: {len(ranked) - 1}")
    return selected


def load_input(
    input_path: str,
    array_key: Optional[str],
    feature_index: int,
    csv_columns: Optional[List[str]],
    max_series: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    if input_path.lower() == "synthetic":
        t = 32768
        s = max(max_series, 16)
        time_axis = np.arange(t, dtype=np.float32)
        matrix = np.empty((t, s), dtype=np.float32)
        for j in range(s):
            seasonal = 10.0 * np.sin(2.0 * np.pi * time_axis / (96 + 7 * (j % 5)))
            trend = (0.0005 + j * 0.00002) * time_axis
            steps = ((time_axis // (256 + 16 * (j % 3))) % 4) * (j % 4)
            noise = rng.normal(0.0, 0.3 + 0.05 * (j % 3), size=t)
            matrix[:, j] = seasonal + trend + steps + noise
        meta = {"source": "synthetic", "original_shape": [t, s]}
    else:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".npz":
            archive = np.load(path, allow_pickle=False)
            keys = list(archive.keys())
            if not keys:
                raise ValueError("NPZ contains no arrays")
            key = array_key or ("data" if "data" in keys else keys[0])
            if key not in archive:
                raise KeyError(f"array_key={key!r} not found. Available keys: {keys}")
            arr = np.asarray(archive[key])
            meta = {"source": str(path.resolve()), "array_key": key, "available_keys": keys, "original_shape": list(arr.shape)}
        elif suffix == ".npy":
            arr = np.load(path, allow_pickle=False)
            meta = {"source": str(path.resolve()), "original_shape": list(arr.shape)}
        elif suffix in {".csv", ".txt"}:
            df = pd.read_csv(path)
            if csv_columns:
                missing = [c for c in csv_columns if c not in df.columns]
                if missing:
                    raise KeyError(f"CSV columns not found: {missing}")
                numeric = df[csv_columns]
            else:
                numeric = df.select_dtypes(include=[np.number])
            if numeric.shape[1] == 0:
                raise ValueError("CSV has no numeric columns after filtering")
            arr = numeric.to_numpy()
            meta = {
                "source": str(path.resolve()),
                "original_shape": list(arr.shape),
                "columns": list(numeric.columns),
            }
        else:
            raise ValueError("Supported input formats: .npz, .npy, .csv, .txt, or the literal 'synthetic'")

        if arr.ndim == 1:
            matrix = arr[:, None]
        elif arr.ndim == 2:
            matrix = arr
        elif arr.ndim == 3:
            if feature_index < 0 or feature_index >= arr.shape[2]:
                raise IndexError(f"feature_index={feature_index} is outside last dimension size {arr.shape[2]}")
            matrix = arr[:, :, feature_index]
            meta["feature_index"] = feature_index
        else:
            matrix = arr.reshape(arr.shape[0], -1)
            meta["flattened_trailing_dimensions"] = True

    matrix = np.asarray(matrix, dtype=np.float32)
    matrix, imputed_fraction = interpolate_nonfinite(matrix)
    if matrix.shape[1] > max_series:
        chosen = np.sort(rng.choice(matrix.shape[1], size=max_series, replace=False))
        matrix = matrix[:, chosen]
        meta["selected_original_series_indices"] = chosen.tolist()
    else:
        meta["selected_original_series_indices"] = list(range(matrix.shape[1]))
    meta["loaded_shape"] = list(matrix.shape)
    meta["imputed_nonfinite_fraction"] = imputed_fraction
    meta["dtype"] = str(matrix.dtype)
    return matrix, meta


def extract_chunk_units(
    matrix: np.ndarray,
    chunk_sizes: Sequence[int],
    max_chunks_per_size: int,
    seed: int,
) -> List[ChunkUnit]:
    rng = np.random.default_rng(seed)
    units: List[ChunkUnit] = []
    t, s = matrix.shape
    for chunk_size in chunk_sizes:
        if chunk_size > t:
            print(f"[WARN] Skip chunk_size={chunk_size}; series length is {t}", file=sys.stderr)
            continue
        candidates: List[Tuple[int, int]] = []
        for series_idx in range(s):
            for start in range(0, t - chunk_size + 1, chunk_size):
                candidates.append((series_idx, start))
        if len(candidates) > max_chunks_per_size:
            selected_ids = rng.choice(len(candidates), size=max_chunks_per_size, replace=False)
            selected = [candidates[int(i)] for i in sorted(selected_ids)]
        else:
            selected = candidates
        for series_idx, start in selected:
            x = np.ascontiguousarray(matrix[start : start + chunk_size, series_idx], dtype=np.float32)
            chunk_id = f"cs{chunk_size}_s{series_idx}_t{start}"
            units.append(ChunkUnit(chunk_id, series_idx, start, chunk_size, x))
    if not units:
        raise ValueError("No chunk units were generated. Check chunk sizes and data length.")
    return units


def build_queries(x: np.ndarray, chunk_id: str, seed: int, windows_per_fraction: int) -> List[QuerySpec]:
    digest = hashlib.sha256(f"{seed}:{chunk_id}".encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(local_seed)
    n = x.size
    queries: List[QuerySpec] = []

    point_count = max(4, windows_per_fraction * 2)
    for i, point in enumerate(rng.integers(0, n, size=point_count)):
        queries.append(QuerySpec(f"{chunk_id}:point:{i}", "point", int(point), int(point + 1)))

    fractions = (0.01, 0.10, 0.50, 1.00)
    range_types = ("range_scan", "range_sum", "range_avg", "range_min", "range_max")
    for fraction in fractions:
        length = max(1, min(n, int(round(n * fraction))))
        starts = [0] if length == n else rng.integers(0, n - length + 1, size=windows_per_fraction).tolist()
        for j, start in enumerate(starts):
            start_i = int(start)
            end_i = start_i + length
            for query_type in range_types:
                queries.append(
                    QuerySpec(
                        f"{chunk_id}:{query_type}:f{fraction:.2f}:{j}",
                        query_type,
                        start_i,
                        end_i,
                    )
                )

    for quantile in (0.50, 0.90, 0.99):
        threshold = float(np.quantile(x, quantile))
        for fraction in (0.10, 0.50, 1.00):
            length = max(1, min(n, int(round(n * fraction))))
            starts = [0] if length == n else rng.integers(0, n - length + 1, size=windows_per_fraction).tolist()
            for j, start in enumerate(starts):
                start_i = int(start)
                queries.append(
                    QuerySpec(
                        f"{chunk_id}:threshold:q{quantile:.2f}:f{fraction:.2f}:{j}",
                        "threshold_count",
                        start_i,
                        start_i + length,
                        threshold=threshold,
                        threshold_quantile=quantile,
                    )
                )
    return queries


def bootstrap_median_ci(values: np.ndarray, clusters: np.ndarray, seed: int, iterations: int = 2000) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    if values.size == 0:
        return math.nan, math.nan
    if unique.size == 1:
        median = float(np.median(values))
        return median, median
    rng = np.random.default_rng(seed)
    grouped = {cluster: values[clusters == cluster] for cluster in unique}
    samples = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        selected_clusters = rng.choice(unique, size=unique.size, replace=True)
        sampled = np.concatenate([grouped[c] for c in selected_clusters])
        samples[i] = np.median(sampled)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * float(p_values[int(idx)])
        running = max(running, candidate)
        adjusted[int(idx)] = min(1.0, running)
    return adjusted.tolist()


def benchmark(
    units: Sequence[ChunkUnit],
    codecs: Mapping[str, Codec],
    output_dir: Path,
    warmups: int,
    encode_repeats: int,
    query_repeats: int,
    windows_per_fraction: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    codec_records: List[Dict[str, Any]] = []
    query_records: List[Dict[str, Any]] = []

    total = len(units)
    for unit_idx, unit in enumerate(units, 1):
        if unit_idx == 1 or unit_idx % max(1, total // 20) == 0 or unit_idx == total:
            print(f"[{unit_idx}/{total}] {unit.chunk_id}")
        x = unit.values
        queries = build_queries(x, unit.chunk_id, seed, windows_per_fraction)
        ground_truth = {q.query_id: execute_on_array(x, q) for q in queries}

        for codec_name in CODEC_ORDER:
            codec = codecs[codec_name]
            encode_ns, encoded = median_timed_ns(lambda c=codec, arr=x: c.encode(arr), warmups, encode_repeats)
            decoded = codec.decode(encoded)
            if not np.array_equal(decoded.view(np.uint32), x.view(np.uint32)):
                mismatch = int(np.count_nonzero(decoded.view(np.uint32) != x.view(np.uint32)))
                raise AssertionError(f"Codec {codec_name} is not bitwise lossless on {unit.chunk_id}; mismatches={mismatch}")
            decode_ns, _ = median_timed_ns(lambda c=codec, e=encoded: c.decode(e), warmups, query_repeats)
            codec_records.append(
                {
                    "chunk_id": unit.chunk_id,
                    "series_index": unit.series_index,
                    "chunk_start": unit.start,
                    "chunk_size": unit.chunk_size,
                    "codec": codec_name,
                    "raw_size_bytes": int(x.nbytes),
                    "encoded_size_bytes": int(encoded.size_bytes),
                    "compression_ratio": float(encoded.size_bytes / x.nbytes),
                    "encode_ns": encode_ns,
                    "decode_ns": decode_ns,
                    "metadata_json": json.dumps(encoded.metadata, sort_keys=True),
                }
            )

            for query in queries:
                latency_ns, actual = median_timed_ns(
                    lambda c=codec, e=encoded, q=query: c.execute(e, q),
                    warmups,
                    query_repeats,
                )
                expected = ground_truth[query.query_id]
                correct = values_equal(actual, expected, query.query_type)
                if not correct:
                    raise AssertionError(
                        f"Incorrect result: codec={codec_name}, chunk={unit.chunk_id}, query={query.query_id}, "
                        f"actual={actual}, expected={expected}"
                    )
                query_records.append(
                    {
                        "query_id": query.query_id,
                        "chunk_id": unit.chunk_id,
                        "series_index": unit.series_index,
                        "chunk_size": unit.chunk_size,
                        "codec": codec_name,
                        "query_type": query.query_type,
                        "start": query.start,
                        "end": query.end,
                        "range_length": query.end - query.start,
                        "range_fraction": (query.end - query.start) / unit.chunk_size,
                        "threshold": query.threshold,
                        "threshold_quantile": query.threshold_quantile,
                        "latency_ns": latency_ns,
                        "result": float(actual),
                        "ground_truth": float(expected),
                        "correct": correct,
                    }
                )

    codec_df = pd.DataFrame(codec_records)
    query_df = pd.DataFrame(query_records)
    codec_df.to_csv(output_dir / "codec_records.csv", index=False)
    query_df.to_csv(output_dir / "query_records.csv", index=False)
    return codec_df, query_df


def summarize_codecs(codec_df: pd.DataFrame, output_dir: Path, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (chunk_size, codec), group in codec_df.groupby(["chunk_size", "codec"], sort=False):
        for metric in ("compression_ratio", "encode_ns", "decode_ns"):
            values = group[metric].to_numpy(dtype=np.float64)
            ci_low, ci_high = bootstrap_median_ci(values, group["chunk_id"].to_numpy(), seed + int(chunk_size))
            rows.append(
                {
                    "chunk_size": int(chunk_size),
                    "codec": codec,
                    "metric": metric,
                    "n_chunks": int(group.shape[0]),
                    "median": float(np.median(values)),
                    "mean": float(np.mean(values)),
                    "ci95_low_cluster_bootstrap": ci_low,
                    "ci95_high_cluster_bootstrap": ci_high,
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "codec_summary.csv", index=False)
    return summary


def summarize_queries(query_df: pd.DataFrame, output_dir: Path, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (chunk_size, query_type, codec), group in query_df.groupby(["chunk_size", "query_type", "codec"], sort=False):
        values = group["latency_ns"].to_numpy(dtype=np.float64)
        ci_low, ci_high = bootstrap_median_ci(values, group["chunk_id"].to_numpy(), seed + int(chunk_size) + QUERY_ORDER.index(query_type))
        rows.append(
            {
                "chunk_size": int(chunk_size),
                "query_type": query_type,
                "codec": codec,
                "n_query_instances": int(group.shape[0]),
                "n_chunks": int(group["chunk_id"].nunique()),
                "median_latency_ns": float(np.median(values)),
                "mean_latency_ns": float(np.mean(values)),
                "ci95_low_cluster_bootstrap_ns": ci_low,
                "ci95_high_cluster_bootstrap_ns": ci_high,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "query_summary.csv", index=False)
    return summary


def paired_statistical_tests(query_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (chunk_size, query_type), group in query_df.groupby(["chunk_size", "query_type"], sort=False):
        pivot = group.pivot_table(index="query_id", columns="codec", values="latency_ns", aggfunc="first").dropna()
        available = [c for c in CODEC_ORDER if c in pivot.columns]
        if len(available) < 2 or pivot.shape[0] < 5:
            continue
        friedman_stat, friedman_p = friedmanchisquare(*[pivot[c].to_numpy() for c in available])
        medians = pivot[available].median(axis=0)
        best_codec = str(medians.idxmin())
        local_rows: List[Dict[str, Any]] = []
        local_p: List[float] = []
        for codec in available:
            if codec == best_codec:
                continue
            best_vals = pivot[best_codec].to_numpy(dtype=np.float64)
            other_vals = pivot[codec].to_numpy(dtype=np.float64)
            log_ratio = np.log(other_vals / best_vals)
            try:
                stat, p = wilcoxon(log_ratio, alternative="greater", zero_method="wilcox", method="auto")
            except ValueError:
                stat, p = 0.0, 1.0
            local_p.append(float(p))
            local_rows.append(
                {
                    "chunk_size": int(chunk_size),
                    "query_type": query_type,
                    "n_paired": int(pivot.shape[0]),
                    "friedman_stat": float(friedman_stat),
                    "friedman_p": float(friedman_p),
                    "best_codec_by_median": best_codec,
                    "compared_codec": codec,
                    "median_speedup_best_over_compared": float(np.median(other_vals / best_vals)),
                    "wilcoxon_stat_log_ratio": float(stat),
                    "wilcoxon_p_one_sided": float(p),
                }
            )
        adjusted = holm_adjust(local_p)
        for row, adjusted_p in zip(local_rows, adjusted):
            row["wilcoxon_p_holm"] = adjusted_p
            rows.append(row)
    tests = pd.DataFrame(rows)
    tests.to_csv(output_dir / "statistical_tests.csv", index=False)
    return tests


def workload_analysis(
    codec_df: pd.DataFrame,
    query_df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Raw is an uncompressed control, not a candidate historical compressed layout.
    candidate_queries = query_df[query_df["codec"].isin(DEPLOYABLE_CODECS)].copy()
    query_means = (
        candidate_queries.groupby(["chunk_id", "chunk_size", "codec", "query_type"], as_index=False)["latency_ns"]
        .median()
    )
    codec_costs = codec_df.set_index(["chunk_id", "codec"])[["encode_ns", "decode_ns", "encoded_size_bytes"]]

    workload_rows: List[Dict[str, Any]] = []
    for workload in WORKLOADS:
        weighted = query_means.copy()
        weighted["weight"] = weighted["query_type"].map(workload.weights).fillna(0.0)
        weighted["weighted_latency_ns"] = weighted["latency_ns"] * weighted["weight"]
        expected = weighted.groupby(["chunk_id", "chunk_size", "codec"], as_index=False)["weighted_latency_ns"].sum()
        expected["workload"] = workload.name
        workload_rows.extend(expected.to_dict("records"))
    workload_df = pd.DataFrame(workload_rows)

    best_rows: List[Dict[str, Any]] = []
    for (chunk_id, chunk_size, workload), group in workload_df.groupby(["chunk_id", "chunk_size", "workload"]):
        idx = group["weighted_latency_ns"].idxmin()
        row = group.loc[idx]
        best_rows.append(
            {
                "chunk_id": chunk_id,
                "chunk_size": int(chunk_size),
                "workload": workload,
                "best_codec": row["codec"],
                "best_expected_latency_ns": float(row["weighted_latency_ns"]),
            }
        )
    best_df = pd.DataFrame(best_rows)

    drift_rows: List[Dict[str, Any]] = []
    old_name = WORKLOADS[0].name
    old_best = best_df[best_df["workload"] == old_name].set_index("chunk_id")
    for new_workload in [w.name for w in WORKLOADS[1:]]:
        new_best = best_df[best_df["workload"] == new_workload].set_index("chunk_id")
        new_costs = workload_df[workload_df["workload"] == new_workload].set_index(["chunk_id", "codec"])
        for chunk_id, old_row in old_best.iterrows():
            old_codec = str(old_row["best_codec"])
            new_codec = str(new_best.loc[chunk_id, "best_codec"])
            old_layout_cost = float(new_costs.loc[(chunk_id, old_codec), "weighted_latency_ns"])
            new_opt_cost = float(new_costs.loc[(chunk_id, new_codec), "weighted_latency_ns"])
            regret_ratio = old_layout_cost / new_opt_cost if new_opt_cost > 0 else math.inf
            old_decode = float(codec_costs.loc[(chunk_id, old_codec), "decode_ns"])
            new_encode = float(codec_costs.loc[(chunk_id, new_codec), "encode_ns"])
            rewrite_cost = old_decode + new_encode
            saving = old_layout_cost - new_opt_cost
            break_even = rewrite_cost / saving if saving > 0 else math.inf
            drift_rows.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_size": int(old_row["chunk_size"]),
                    "old_workload": old_name,
                    "new_workload": new_workload,
                    "old_codec": old_codec,
                    "new_codec": new_codec,
                    "codec_switch": old_codec != new_codec,
                    "old_layout_expected_latency_ns": old_layout_cost,
                    "new_opt_expected_latency_ns": new_opt_cost,
                    "regret_ratio": regret_ratio,
                    "rewrite_cost_ns_in_memory": rewrite_cost,
                    "saving_per_query_ns": saving,
                    "break_even_future_queries": break_even,
                }
            )
    drift_df = pd.DataFrame(drift_rows)

    workload_df.to_csv(output_dir / "workload_costs.csv", index=False)
    best_df.to_csv(output_dir / "workload_best_layouts.csv", index=False)
    drift_df.to_csv(output_dir / "workload_drift_analysis.csv", index=False)
    return workload_df, best_df, drift_df


def e0_decision(query_summary: pd.DataFrame, best_df: pd.DataFrame, drift_df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    # Preregistered evidence thresholds; they are intentionally conservative but not claims of universality.
    # C1: at least two workload scenarios have different modal best codecs.
    modal_best: Dict[str, str] = {}
    for workload, group in best_df.groupby("workload"):
        modal_best[workload] = str(group["best_codec"].value_counts().idxmax())
    c1 = len(set(modal_best.values())) >= 2

    finite_drift = drift_df[np.isfinite(drift_df["regret_ratio"])].copy()
    drift_summary: Dict[str, Any] = {}
    c2 = False
    c3 = False
    for workload, group in finite_drift.groupby("new_workload"):
        median_regret = float(group["regret_ratio"].median())
        p75_regret = float(group["regret_ratio"].quantile(0.75))
        switch_fraction = float(group["codec_switch"].mean())
        finite_be = group[np.isfinite(group["break_even_future_queries"])]
        median_be = float(finite_be["break_even_future_queries"].median()) if not finite_be.empty else math.inf
        fraction_be_le_1000 = float((finite_be["break_even_future_queries"] <= 1000).mean()) if not finite_be.empty else 0.0
        drift_summary[workload] = {
            "median_regret_ratio": median_regret,
            "p75_regret_ratio": p75_regret,
            "codec_switch_fraction": switch_fraction,
            "median_break_even_future_queries": median_be,
            "fraction_positive_switches_break_even_le_1000": fraction_be_le_1000,
        }
        c2 = c2 or (median_regret >= 1.15 and switch_fraction >= 0.20)
        c3 = c3 or (fraction_be_le_1000 >= 0.25)

    decision = {
        "version": VERSION,
        "criteria": {
            "C1_workload_specific_modal_best": {
                "pass": bool(c1),
                "definition": "At least two preregistered workloads have different modal best representations across chunks.",
                "modal_best_by_workload": modal_best,
            },
            "C2_material_static_layout_regret": {
                "pass": bool(c2),
                "definition": "For at least one drift, median regret >= 1.15 and at least 20% of chunks change their best representation.",
            },
            "C3_plausible_in_memory_amortization": {
                "pass": bool(c3),
                "definition": "For at least one drift, at least 25% of positive-saving switches break even within 1000 future queries under in-memory rewrite cost.",
                "warning": "This is not a disk-level claim. E1 must add actual storage I/O before publication.",
            },
        },
        "drift_summary": drift_summary,
        "overall_e0a_pass": bool(c1 and c2 and c3),
        "interpretation": (
            "PASS means the in-memory premise is worth taking to the required file-backed E0-B prototype. "
            "FAIL means the current representation/query set does not yet justify E0-B without revising the representations or workloads."
        ),
    }
    with open(output_dir / "e0_decision.json", "w", encoding="utf-8") as f:
        json.dump(decision, f, ensure_ascii=False, indent=2, allow_nan=True)
    return decision


def make_plots(codec_df: pd.DataFrame, query_summary: pd.DataFrame, drift_df: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    for chunk_size, group in codec_df.groupby("chunk_size"):
        data = [group[group["codec"] == c]["compression_ratio"].to_numpy() for c in CODEC_ORDER]
        plt.figure(figsize=(9, 5))
        plt.boxplot(data, tick_labels=CODEC_ORDER, showfliers=False)
        plt.ylabel("Encoded size / raw size")
        plt.title(f"Compression ratio, chunk size={chunk_size}")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(plot_dir / f"compression_ratio_cs{chunk_size}.png", dpi=180)
        plt.close()

    for chunk_size, group in query_summary.groupby("chunk_size"):
        for query_type, qgroup in group.groupby("query_type"):
            qgroup = qgroup.set_index("codec").reindex(CODEC_ORDER).dropna().reset_index()
            plt.figure(figsize=(9, 5))
            plt.bar(qgroup["codec"], qgroup["median_latency_ns"] / 1000.0)
            plt.ylabel("Median latency (microseconds)")
            plt.title(f"{query_type}, chunk size={chunk_size}")
            plt.xticks(rotation=20)
            plt.tight_layout()
            plt.savefig(plot_dir / f"latency_{query_type}_cs{chunk_size}.png", dpi=180)
            plt.close()

    finite = drift_df[np.isfinite(drift_df["break_even_future_queries"])].copy()
    if not finite.empty:
        for workload, group in finite.groupby("new_workload"):
            plt.figure(figsize=(8, 5))
            clipped = np.minimum(group["break_even_future_queries"].to_numpy(), 10000)
            plt.hist(clipped, bins=40)
            plt.xlabel("Future queries to break even (clipped at 10,000)")
            plt.ylabel("Chunk count")
            plt.title(f"In-memory break-even after drift to {workload}")
            plt.tight_layout()
            plt.savefig(plot_dir / f"break_even_{workload}.png", dpi=180)
            plt.close()


def environment_info() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "lz4": getattr(lz4, "__version__", "unknown"),
        "zstandard": zstd.__version__,
        "timer": "time.perf_counter_ns",
    }


def parse_int_list(value: str) -> List[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(v <= 0 for v in result):
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers")
    return result


def parse_str_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RePressDB E0-A in-memory feasibility benchmark")
    parser.add_argument("--input", default=None, help="Optional path to .npz/.npy/.csv or synthetic; omitted = auto-run fixed PEMS04 and PEMS08 directories")
    parser.add_argument("--array-key", default=None, help="Array key for NPZ; default: data or first key")
    parser.add_argument("--feature-index", type=int, default=0, help="Feature index for 3-D arrays [time, series, feature]")
    parser.add_argument("--csv-columns", default=None, help="Comma-separated numeric CSV columns; default: all numeric columns")
    parser.add_argument("--output", default=None, help="Output root directory; auto mode defaults to results_pems04/results_pems08")
    parser.add_argument("--chunk-sizes", type=parse_int_list, default=[512, 2048, 8192])
    parser.add_argument("--max-series", type=int, default=32)
    parser.add_argument("--max-chunks-per-size", type=int, default=120)
    parser.add_argument("--microblock-size", type=int, default=128)
    parser.add_argument("--zstd-level", type=int, default=3)
    parser.add_argument("--lz4-level", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--encode-repeats", type=int, default=5)
    parser.add_argument("--query-repeats", type=int, default=7)
    parser.add_argument("--windows-per-fraction", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--quick", action="store_true", help="Small smoke test, not valid for scientific conclusions")
    return parser


def run_single_dataset(args: argparse.Namespace, input_path: str, output_root: str, dataset_label: Optional[str] = None) -> Dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    source_name = dataset_label.lower() if dataset_label else ("synthetic" if input_path.lower() == "synthetic" else Path(input_path).stem)
    output_dir = Path(output_root) / f"{source_name}_{run_stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    config = vars(args).copy()
    config["input"] = input_path
    config["output"] = output_root
    config["dataset_label"] = dataset_label
    config["version"] = VERSION
    config["environment"] = environment_info()
    config["measurement_scope"] = "E0-A: in-memory encoding, decoding, and query execution; excludes file I/O"
    config["deployable_codecs_for_workload_selection"] = DEPLOYABLE_CODECS
    config["control_codec_excluded_from_layout_selection"] = "raw"
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    matrix, dataset_meta = load_input(
        input_path,
        args.array_key,
        args.feature_index,
        parse_str_list(args.csv_columns),
        args.max_series,
        args.seed,
    )
    dataset_meta["dataset_label"] = dataset_label
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(dataset_meta, f, ensure_ascii=False, indent=2)

    units = extract_chunk_units(matrix, args.chunk_sizes, args.max_chunks_per_size, args.seed)
    codecs = codec_registry(args.zstd_level, args.lz4_level, args.microblock_size)

    print("=" * 78)
    print(f"RePressDB E0 v{VERSION}")
    if dataset_label:
        print(f"Dataset: {dataset_label}")
    print(f"Input file: {input_path}")
    print(f"Input shape: {matrix.shape}; chunk units: {len(units)}")
    print(f"Output: {output_dir.resolve()}")
    print("Measurement scope: in-memory; file I/O intentionally excluded")
    if args.quick:
        print("WARNING: --quick is a smoke test only and cannot support scientific conclusions.")
    print("=" * 78)

    codec_df, query_df = benchmark(
        units,
        codecs,
        output_dir,
        args.warmups,
        args.encode_repeats,
        args.query_repeats,
        args.windows_per_fraction,
        args.seed,
    )
    summarize_codecs(codec_df, output_dir, args.seed)
    query_summary = summarize_queries(query_df, output_dir, args.seed)
    paired_statistical_tests(query_df, output_dir)
    _, best_df, drift_df = workload_analysis(codec_df, query_df, output_dir)
    decision = e0_decision(query_summary, best_df, drift_df, output_dir)
    make_plots(codec_df, query_summary, drift_df, output_dir)

    print("\nE0-A decision:")
    print(json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=True))
    print(f"\nCompleted. Results: {output_dir.resolve()}")
    return {
        "dataset": dataset_label or source_name,
        "input": input_path,
        "output": str(output_dir.resolve()),
        "overall_e0a_pass": bool(decision["overall_e0a_pass"]),
    }


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.quick:
        args.chunk_sizes = [256, 1024]
        args.max_series = min(args.max_series, 4)
        args.max_chunks_per_size = min(args.max_chunks_per_size, 8)
        args.warmups = 1
        args.encode_repeats = 2
        args.query_repeats = 2
        args.windows_per_fraction = 1

    if args.input is not None:
        output_root = args.output or "results_e0"
        run_single_dataset(args, args.input, output_root)
        return 0

    print("未提供 --input，将自动依次运行以下固定数据目录：")
    for label, directory in DEFAULT_DATASET_DIRS.items():
        print(f"  {label}: {directory}")

    resolved: List[Tuple[str, Path]] = []
    for label, directory in DEFAULT_DATASET_DIRS.items():
        resolved.append((label, resolve_dataset_file(directory, label)))

    summaries: List[Dict[str, Any]] = []
    for label, data_file in resolved:
        output_root = str(Path(args.output) / label.lower()) if args.output else f"results_{label.lower()}"
        summaries.append(run_single_dataset(args, str(data_file), output_root, dataset_label=label))

    print("\n" + "=" * 78)
    print("PEMS04 / PEMS08 自动批量实验完成")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
