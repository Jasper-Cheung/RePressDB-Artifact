#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RePressDB E0-C v0.2.0

Cache and metadata boundary experiment for RePressDB / RePress.
Target paper: EDBT 2027 (CCF-B), not a USENIX-scale systems submission.

This experiment extends the frozen E0-B v0.2.2 protocol and answers a narrower
question: do workload-dependent representation choices survive when we include
file-open overhead and block-metadata file I/O, and (on Linux/WSL2) when we use
best-effort per-file page-cache eviction before every measured query?

Supported cache modes
---------------------
1. warm_cached
   Persistent payload handle; block metadata already parsed in Python memory.
2. warm_slicedmeta
   Persistent payload and metadata handles; only the metadata fields and block
   interval needed by the current query are read from the metadata file.
3. reopen_cached
   Payload file is opened per query; block metadata remains in Python memory.
4. reopen_slicedmeta
   Payload and metadata files are opened per query; only query-relevant metadata
   slices are read. This is the primary portable file-backed boundary mode.
5. warm_fullreloadmeta / reopen_fullreloadmeta
   Diagnostic worst-case modes that read and parse the complete metadata file
   for every query. They are excluded from the default protocol.
6. linux_cold_cached / linux_cold_slicedmeta
   Linux/WSL2-only advisory page-cache eviction modes.

Important limitation
--------------------
POSIX_FADV_DONTNEED is an advisory cache-eviction request, not proof that every
read reached physical SSD media. The script records the exact method and never
labels Windows close/reopen measurements as "cold cache".

Default behavior runs only the PEMS04 pilot. After inspecting that output, run
PEMS08 with --datasets pems08 using the unchanged protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import lz4.frame
import numpy as np
import pandas as pd
import scipy
import zstandard as zstd
from scipy.stats import wilcoxon

VERSION = "0.2.0"
E0B_PROTOCOL_VERSION = "0.2.2"
TARGET_VENUE = "EDBT 2027"

WINDOWS_PEMS04_DIR = Path(r"E:\IPM\大学\科研\鲁棒性预测\data\Traffic-main\PEMS04")
WINDOWS_PEMS08_DIR = Path(r"E:\IPM\大学\科研\鲁棒性预测\data\Traffic-main\PEMS08")
LINUX_PEMS04_DIR = Path("/mnt/e/IPM/大学/科研/鲁棒性预测/data/Traffic-main/PEMS04")
LINUX_PEMS08_DIR = Path("/mnt/e/IPM/大学/科研/鲁棒性预测/data/Traffic-main/PEMS08")

PEMS08_E0A_SERIES = {
    3, 4, 17, 21, 22, 26, 34, 51, 52, 56, 60, 66, 67, 70, 72, 80,
    81, 92, 93, 95, 101, 102, 104, 122, 131, 133, 137, 140, 150, 156,
    158, 162,
}

EXPECTED_QUERY_CLASSES = {
    "point",
    "short_scan",
    "medium_scan",
    "long_scan",
    "short_agg",
    "medium_agg",
    "long_agg",
    "threshold",
}

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

for _workload_name, _weights in WORKLOADS.items():
    if not math.isclose(sum(_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"Workload {_workload_name} weights do not sum to 1")


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    query_type: str
    query_class: str
    start: int
    end: int
    range_fraction: float
    target_fraction: float
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

    @property
    def stored_size_bytes(self) -> int:
        return self.payload_size_bytes + self.metadata_size_bytes


@dataclass
class IOStats:
    payload_bytes: int = 0
    metadata_bytes: int = 0
    payload_open_calls: int = 0
    metadata_open_calls: int = 0
    payload_read_calls: int = 0
    metadata_read_calls: int = 0
    payload_seek_calls: int = 0
    metadata_seek_calls: int = 0

    @property
    def total_logical_bytes(self) -> int:
        return self.payload_bytes + self.metadata_bytes

    @property
    def total_open_calls(self) -> int:
        return self.payload_open_calls + self.metadata_open_calls

    @property
    def total_read_calls(self) -> int:
        return self.payload_read_calls + self.metadata_read_calls

    @property
    def total_seek_calls(self) -> int:
        return self.payload_seek_calls + self.metadata_seek_calls


@dataclass(frozen=True)
class CacheMode:
    name: str
    persistent_payload_handle: bool
    metadata_from_file: bool
    evict_payload_before_repeat: bool
    evict_metadata_before_repeat: bool
    description: str
    metadata_sliced: bool = False
    persistent_metadata_handle: bool = False


CACHE_MODES: Dict[str, CacheMode] = {
    "warm_cached": CacheMode(
        "warm_cached", True, False, False, False,
        "Persistent payload handle; block metadata cached in Python memory.",
    ),
    "warm_slicedmeta": CacheMode(
        "warm_slicedmeta", True, False, False, False,
        "Persistent payload and metadata handles; query-relevant metadata slices are read.",
        metadata_sliced=True,
        persistent_metadata_handle=True,
    ),
    "reopen_cached": CacheMode(
        "reopen_cached", False, False, False, False,
        "Payload file opened per query; block metadata cached in Python memory.",
    ),
    "reopen_slicedmeta": CacheMode(
        "reopen_slicedmeta", False, False, False, False,
        "Payload and metadata files opened per query; query-relevant metadata slices are read.",
        metadata_sliced=True,
        persistent_metadata_handle=False,
    ),
    "warm_fullreloadmeta": CacheMode(
        "warm_fullreloadmeta", True, True, False, False,
        "Diagnostic worst case: complete metadata file opened, read and parsed per query.",
    ),
    "reopen_fullreloadmeta": CacheMode(
        "reopen_fullreloadmeta", False, True, False, False,
        "Diagnostic worst case: payload reopened and complete metadata file reloaded per query.",
    ),
    "linux_cold_cached": CacheMode(
        "linux_cold_cached", False, False, True, False,
        "Linux/WSL2 advisory payload eviction before each repeat; metadata cached.",
    ),
    "linux_cold_slicedmeta": CacheMode(
        "linux_cold_slicedmeta", False, False, True, True,
        "Linux/WSL2 advisory payload and metadata eviction; query-relevant metadata slices read.",
        metadata_sliced=True,
        persistent_metadata_handle=False,
    ),
}


class PhysicalRepresentation:
    name: str

    def encode_to_files(self, x: np.ndarray, stem: Path) -> StoredChunk:
        raise NotImplementedError

    def decode_all(
        self,
        stored: StoredChunk,
        metadata: Dict[str, Any],
        payload_handle: Any,
        stats: IOStats,
    ) -> np.ndarray:
        raise NotImplementedError

    def execute(
        self,
        stored: StoredChunk,
        query: QuerySpec,
        metadata: Dict[str, Any],
        payload_handle: Any,
        stats: IOStats,
    ) -> Any:
        raise NotImplementedError


class WholeLZ4(PhysicalRepresentation):
    name = "whole_lz4"

    def encode_to_files(self, x: np.ndarray, stem: Path) -> StoredChunk:
        arr = np.ascontiguousarray(x, dtype=np.float32)
        payload = lz4.frame.compress(arr.tobytes(order="C"), compression_level=0)
        data_path = stem.with_suffix(".bin")
        write_bytes_fsync(data_path, payload)
        return StoredChunk(self.name, arr.size, data_path, None, data_path.stat().st_size, 0, {})

    def decode_all(
        self,
        stored: StoredChunk,
        metadata: Dict[str, Any],
        payload_handle: Any,
        stats: IOStats,
    ) -> np.ndarray:
        payload = read_payload_all(stored, payload_handle, stats)
        raw = lz4.frame.decompress(payload)
        return np.frombuffer(raw, dtype=np.float32, count=stored.n).copy()

    def execute(
        self,
        stored: StoredChunk,
        query: QuerySpec,
        metadata: Dict[str, Any],
        payload_handle: Any,
        stats: IOStats,
    ) -> Any:
        x = self.decode_all(stored, metadata, payload_handle, stats)
        return execute_on_array(x, query)


class WholeZstd(PhysicalRepresentation):
    name = "whole_zstd"

    def __init__(self, level: int = 3):
        self.level = int(level)
        self.compressor = zstd.ZstdCompressor(level=self.level)
        self.decompressor = zstd.ZstdDecompressor()

    def encode_to_files(self, x: np.ndarray, stem: Path) -> StoredChunk:
        arr = np.ascontiguousarray(x, dtype=np.float32)
        payload = self.compressor.compress(arr.tobytes(order="C"))
        data_path = stem.with_suffix(".bin")
        write_bytes_fsync(data_path, payload)
        return StoredChunk(
            self.name,
            arr.size,
            data_path,
            None,
            data_path.stat().st_size,
            0,
            {"level": self.level},
        )

    def decode_all(
        self,
        stored: StoredChunk,
        metadata: Dict[str, Any],
        payload_handle: Any,
        stats: IOStats,
    ) -> np.ndarray:
        payload = read_payload_all(stored, payload_handle, stats)
        raw = self.decompressor.decompress(payload, max_output_size=stored.n * 4)
        return np.frombuffer(raw, dtype=np.float32, count=stored.n).copy()

    def execute(
        self,
        stored: StoredChunk,
        query: QuerySpec,
        metadata: Dict[str, Any],
        payload_handle: Any,
        stats: IOStats,
    ) -> Any:
        x = self.decode_all(stored, metadata, payload_handle, stats)
        return execute_on_array(x, query)


class BlockLZ4(PhysicalRepresentation):
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
        data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(data_path, "wb") as f:
            for payload in payloads:
                f.write(payload)
            f.flush()
            os.fsync(f.fileno())

        metadata = {
            "offsets": np.asarray(offsets, dtype=np.int64),
            "lengths": np.asarray(lengths, dtype=np.int32),
            "counts": np.asarray(counts, dtype=np.int32),
            "sums": np.asarray(sums, dtype=np.float64),
            "mins": np.asarray(mins, dtype=np.float32),
            "maxs": np.asarray(maxs, dtype=np.float32),
            "block_size": self.block_size,
        }
        metadata_path = stem.with_suffix(".meta.bin")
        write_block_metadata(metadata_path, metadata)

        return StoredChunk(
            self.name,
            arr.size,
            data_path,
            metadata_path,
            data_path.stat().st_size,
            metadata_path.stat().st_size,
            metadata,
        )

    def _read_block(
        self,
        stored: StoredChunk,
        metadata: Dict[str, Any],
        payload_handle: Any,
        block_idx: int,
        stats: IOStats,
    ) -> np.ndarray:
        offset = int(metadata["offsets"][block_idx])
        length = int(metadata["lengths"][block_idx])
        count = int(metadata["counts"][block_idx])
        payload = read_payload_slice(stored, payload_handle, offset, length, stats)
        raw = lz4.frame.decompress(payload)
        return np.frombuffer(raw, dtype=np.float32, count=count)

    @staticmethod
    def _touched_blocks(
        metadata: Dict[str, Any], start: int, end: int
    ) -> Iterable[Tuple[int, int, int, bool]]:
        bs = int(metadata["block_size"])
        first = start // bs
        last = (end - 1) // bs
        for block_idx in range(first, last + 1):
            block_start = block_idx * bs
            block_len = int(metadata["counts"][block_idx])
            block_end = block_start + block_len
            local_start = max(start, block_start) - block_start
            local_end = min(end, block_end) - block_start
            full = local_start == 0 and local_end == block_len
            yield block_idx, local_start, local_end, full

    @staticmethod
    def _block_count(stored: StoredChunk, block_size: int, block_idx: int) -> int:
        return int(min(block_size, stored.n - block_idx * block_size))

    def _read_block_from_index(
        self,
        stored: StoredChunk,
        payload_handle: Any,
        block_idx: int,
        offset: int,
        length: int,
        stats: IOStats,
    ) -> np.ndarray:
        count = self._block_count(stored, self.block_size, block_idx)
        payload = read_payload_slice(stored, payload_handle, int(offset), int(length), stats)
        raw = lz4.frame.decompress(payload)
        return np.frombuffer(raw, dtype=np.float32, count=count)

    def execute_sliced_metadata(
        self,
        stored: StoredChunk,
        query: QuerySpec,
        metadata_handle: Any,
        payload_handle: Any,
        stats: IOStats,
    ) -> Any:
        if metadata_handle is None:
            raise RuntimeError("Sliced metadata mode requires an open metadata handle")

        bs = self.block_size

        def read_index_range(first: int, last_exclusive: int) -> Tuple[np.ndarray, np.ndarray]:
            offsets = read_metadata_field_slice(
                stored, metadata_handle, "offsets", first, last_exclusive, stats
            )
            lengths = read_metadata_field_slice(
                stored, metadata_handle, "lengths", first, last_exclusive, stats
            )
            return offsets, lengths

        def read_one_block(block_idx: int) -> np.ndarray:
            offsets, lengths = read_index_range(block_idx, block_idx + 1)
            return self._read_block_from_index(
                stored,
                payload_handle,
                block_idx,
                int(offsets[0]),
                int(lengths[0]),
                stats,
            )

        if query.query_type == "point":
            block_idx = query.start // bs
            block = read_one_block(block_idx)
            return float(block[query.start - block_idx * bs])

        if query.query_type == "range_scan":
            first = query.start // bs
            last = (query.end - 1) // bs
            offsets, lengths = read_index_range(first, last + 1)
            total = 0.0
            for local_idx, block_idx in enumerate(range(first, last + 1)):
                block = self._read_block_from_index(
                    stored,
                    payload_handle,
                    block_idx,
                    int(offsets[local_idx]),
                    int(lengths[local_idx]),
                    stats,
                )
                block_start = block_idx * bs
                local_start = max(query.start, block_start) - block_start
                local_end = min(query.end, block_start + block.size) - block_start
                total += float(np.sum(block[local_start:local_end], dtype=np.float64))
            return total

        start_block = query.start // bs
        end_block = (query.end - 1) // bs
        if start_block == end_block:
            block = read_one_block(start_block)
            local_start = query.start - start_block * bs
            local_end = query.end - start_block * bs
            return execute_on_array(
                block,
                replace(query, start=local_start, end=local_end),
            )

        partials: List[np.ndarray] = []
        full_first = start_block
        full_last_exclusive = end_block + 1

        if query.start % bs != 0:
            block = read_one_block(start_block)
            partials.append(block[query.start - start_block * bs :])
            full_first += 1
        if query.end % bs != 0:
            block = read_one_block(end_block)
            partials.append(block[: query.end - end_block * bs])
            full_last_exclusive -= 1

        if query.query_type == "threshold_count":
            if query.threshold is None:
                raise ValueError("threshold_count missing threshold")
            threshold = float(query.threshold)
            result = sum(int(np.count_nonzero(part > threshold)) for part in partials)
            if full_first < full_last_exclusive:
                mins = read_metadata_field_slice(
                    stored, metadata_handle, "mins", full_first, full_last_exclusive, stats
                )
                maxs = read_metadata_field_slice(
                    stored, metadata_handle, "maxs", full_first, full_last_exclusive, stats
                )
                block_indices = np.arange(full_first, full_last_exclusive, dtype=np.int64)
                counts = np.minimum(bs, stored.n - block_indices * bs).astype(np.int64)
                result += int(np.sum(counts[mins > threshold], dtype=np.int64))
                ambiguous_local = np.flatnonzero((mins <= threshold) & (maxs > threshold))
                if ambiguous_local.size:
                    offsets, lengths = read_index_range(full_first, full_last_exclusive)
                    for local_idx in ambiguous_local:
                        block_idx = full_first + int(local_idx)
                        block = self._read_block_from_index(
                            stored,
                            payload_handle,
                            block_idx,
                            int(offsets[int(local_idx)]),
                            int(lengths[int(local_idx)]),
                            stats,
                        )
                        result += int(np.count_nonzero(block > threshold))
            return result

        if query.query_type in {"range_sum", "range_avg"}:
            total = sum(float(np.sum(part, dtype=np.float64)) for part in partials)
            if full_first < full_last_exclusive:
                sums = read_metadata_field_slice(
                    stored, metadata_handle, "sums", full_first, full_last_exclusive, stats
                )
                total += float(np.sum(sums, dtype=np.float64))
            if query.query_type == "range_sum":
                return total
            return total / (query.end - query.start)

        if query.query_type == "range_min":
            current = min((float(np.min(part)) for part in partials), default=math.inf)
            if full_first < full_last_exclusive:
                mins = read_metadata_field_slice(
                    stored, metadata_handle, "mins", full_first, full_last_exclusive, stats
                )
                current = min(current, float(np.min(mins)))
            return current

        if query.query_type == "range_max":
            current = max((float(np.max(part)) for part in partials), default=-math.inf)
            if full_first < full_last_exclusive:
                maxs = read_metadata_field_slice(
                    stored, metadata_handle, "maxs", full_first, full_last_exclusive, stats
                )
                current = max(current, float(np.max(maxs)))
            return current

        raise ValueError(f"Unsupported query type: {query.query_type}")

    def decode_all(
        self,
        stored: StoredChunk,
        metadata: Dict[str, Any],
        payload_handle: Any,
        stats: IOStats,
    ) -> np.ndarray:
        out = np.empty(stored.n, dtype=np.float32)
        bs = int(metadata["block_size"])
        for block_idx in range(len(metadata["counts"])):
            block = self._read_block(stored, metadata, payload_handle, block_idx, stats)
            start = block_idx * bs
            out[start : start + block.size] = block
        return out

    def execute(
        self,
        stored: StoredChunk,
        query: QuerySpec,
        metadata: Dict[str, Any],
        payload_handle: Any,
        stats: IOStats,
    ) -> Any:
        bs = int(metadata["block_size"])

        if query.query_type == "point":
            block_idx = query.start // bs
            block = self._read_block(stored, metadata, payload_handle, block_idx, stats)
            return float(block[query.start - block_idx * bs])

        if query.query_type == "range_scan":
            total = 0.0
            for block_idx, local_start, local_end, _ in self._touched_blocks(
                metadata, query.start, query.end
            ):
                block = self._read_block(stored, metadata, payload_handle, block_idx, stats)
                total += float(np.sum(block[local_start:local_end], dtype=np.float64))
            return total

        start_block = query.start // bs
        end_block = (query.end - 1) // bs
        if start_block == end_block:
            block = self._read_block(stored, metadata, payload_handle, start_block, stats)
            local_start = query.start - start_block * bs
            local_end = query.end - start_block * bs
            local_query = replace(query, start=local_start, end=local_end)
            return execute_on_array(block, local_query)

        partials: List[np.ndarray] = []
        full_first = start_block
        full_last_exclusive = end_block + 1

        if query.start % bs != 0:
            block = self._read_block(stored, metadata, payload_handle, start_block, stats)
            partials.append(block[query.start - start_block * bs :])
            full_first += 1
        if query.end % bs != 0:
            block = self._read_block(stored, metadata, payload_handle, end_block, stats)
            partials.append(block[: query.end - end_block * bs])
            full_last_exclusive -= 1

        if query.query_type == "threshold_count":
            if query.threshold is None:
                raise ValueError("threshold_count missing threshold")
            threshold = float(query.threshold)
            result = sum(int(np.count_nonzero(part > threshold)) for part in partials)
            if full_first < full_last_exclusive:
                mins = metadata["mins"][full_first:full_last_exclusive]
                maxs = metadata["maxs"][full_first:full_last_exclusive]
                counts = metadata["counts"][full_first:full_last_exclusive]
                result += int(np.sum(counts[mins > threshold], dtype=np.int64))
                ambiguous = np.flatnonzero((mins <= threshold) & (maxs > threshold)) + full_first
                for block_idx in ambiguous:
                    block = self._read_block(
                        stored, metadata, payload_handle, int(block_idx), stats
                    )
                    result += int(np.count_nonzero(block > threshold))
            return result

        total = sum(float(np.sum(part, dtype=np.float64)) for part in partials)
        count = sum(int(part.size) for part in partials)
        current_min = min((float(np.min(part)) for part in partials), default=math.inf)
        current_max = max((float(np.max(part)) for part in partials), default=-math.inf)

        if full_first < full_last_exclusive:
            total += float(np.sum(metadata["sums"][full_first:full_last_exclusive], dtype=np.float64))
            count += int(np.sum(metadata["counts"][full_first:full_last_exclusive], dtype=np.int64))
            current_min = min(
                current_min,
                float(np.min(metadata["mins"][full_first:full_last_exclusive])),
            )
            current_max = max(
                current_max,
                float(np.max(metadata["maxs"][full_first:full_last_exclusive])),
            )

        if query.query_type == "range_sum":
            return total
        if query.query_type == "range_avg":
            return total / count
        if query.query_type == "range_min":
            return current_min
        if query.query_type == "range_max":
            return current_max
        raise ValueError(f"Unsupported query type: {query.query_type}")


def write_bytes_fsync(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def write_block_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    block_size = int(metadata["block_size"])
    num_blocks = int(len(metadata["offsets"]))
    with open(path, "wb") as f:
        f.write(np.asarray([block_size, num_blocks], dtype=np.int32).tobytes())
        f.write(np.asarray(metadata["offsets"], dtype=np.int64).tobytes())
        f.write(np.asarray(metadata["lengths"], dtype=np.int32).tobytes())
        f.write(np.asarray(metadata["counts"], dtype=np.int32).tobytes())
        f.write(np.asarray(metadata["sums"], dtype=np.float64).tobytes())
        f.write(np.asarray(metadata["mins"], dtype=np.float32).tobytes())
        f.write(np.asarray(metadata["maxs"], dtype=np.float32).tobytes())
        f.flush()
        os.fsync(f.fileno())


def parse_block_metadata(raw: bytes) -> Dict[str, Any]:
    if len(raw) < 8:
        raise IOError("Block metadata file is shorter than its header")
    header = np.frombuffer(raw[:8], dtype=np.int32, count=2)
    block_size = int(header[0])
    num_blocks = int(header[1])
    if block_size <= 0 or num_blocks <= 0:
        raise IOError("Invalid block metadata header")

    cursor = 8

    def take(dtype: Any, count: int) -> np.ndarray:
        nonlocal cursor
        nbytes = np.dtype(dtype).itemsize * count
        end = cursor + nbytes
        if end > len(raw):
            raise IOError("Truncated block metadata file")
        arr = np.frombuffer(raw[cursor:end], dtype=dtype, count=count).copy()
        cursor = end
        return arr

    metadata = {
        "offsets": take(np.int64, num_blocks),
        "lengths": take(np.int32, num_blocks),
        "counts": take(np.int32, num_blocks),
        "sums": take(np.float64, num_blocks),
        "mins": take(np.float32, num_blocks),
        "maxs": take(np.float32, num_blocks),
        "block_size": block_size,
    }
    if cursor != len(raw):
        raise IOError(f"Unexpected trailing bytes in metadata file: {len(raw) - cursor}")
    return metadata


METADATA_FIELD_DTYPES: Dict[str, Any] = {
    "offsets": np.int64,
    "lengths": np.int32,
    "counts": np.int32,
    "sums": np.float64,
    "mins": np.float32,
    "maxs": np.float32,
}


def metadata_field_base(stored: StoredChunk, field: str) -> int:
    if field not in METADATA_FIELD_DTYPES:
        raise KeyError(f"Unknown metadata field: {field}")
    num_blocks = int(len(stored.metadata["offsets"]))
    bases = {
        "offsets": 8,
        "lengths": 8 + 8 * num_blocks,
        "counts": 8 + 8 * num_blocks + 4 * num_blocks,
        "sums": 8 + 8 * num_blocks + 4 * num_blocks + 4 * num_blocks,
        "mins": 8 + 8 * num_blocks + 4 * num_blocks + 4 * num_blocks + 8 * num_blocks,
        "maxs": 8 + 8 * num_blocks + 4 * num_blocks + 4 * num_blocks + 8 * num_blocks + 4 * num_blocks,
    }
    return int(bases[field])


def read_metadata_field_slice(
    stored: StoredChunk,
    metadata_handle: Any,
    field: str,
    first: int,
    last_exclusive: int,
    stats: IOStats,
) -> np.ndarray:
    if stored.metadata_path is None:
        raise RuntimeError("Block metadata path is unavailable")
    if metadata_handle is None:
        raise RuntimeError("Metadata handle must be open at query scope")
    num_blocks = int(len(stored.metadata["offsets"]))
    if not (0 <= first <= last_exclusive <= num_blocks):
        raise IndexError(
            f"Invalid metadata slice {field}[{first}:{last_exclusive}] for {num_blocks} blocks"
        )
    dtype = np.dtype(METADATA_FIELD_DTYPES[field])
    count = int(last_exclusive - first)
    if count == 0:
        return np.empty(0, dtype=dtype)
    offset = metadata_field_base(stored, field) + first * dtype.itemsize
    nbytes = count * dtype.itemsize
    metadata_handle.seek(offset)
    stats.metadata_seek_calls += 1
    raw = metadata_handle.read(nbytes)
    stats.metadata_read_calls += 1
    stats.metadata_bytes += len(raw)
    if len(raw) != nbytes:
        raise IOError(
            f"Short metadata slice read for {stored.codec} field={field}: {len(raw)} != {nbytes}"
        )
    return np.frombuffer(raw, dtype=dtype, count=count).copy()


def load_metadata_for_query(stored: StoredChunk, mode: CacheMode, stats: IOStats) -> Dict[str, Any]:
    if stored.metadata_path is None:
        return stored.metadata
    if not mode.metadata_from_file:
        return stored.metadata

    stats.metadata_open_calls += 1
    with open(stored.metadata_path, "rb") as f:
        raw = f.read()
        stats.metadata_read_calls += 1
        stats.metadata_bytes += len(raw)
    if len(raw) != stored.metadata_size_bytes:
        raise IOError(
            f"Short metadata read for {stored.codec}: {len(raw)} != {stored.metadata_size_bytes}"
        )
    return parse_block_metadata(raw)


def read_payload_all(stored: StoredChunk, payload_handle: Any, stats: IOStats) -> bytes:
    if payload_handle is None:
        raise RuntimeError("Payload handle must be opened once at query scope")
    payload_handle.seek(0)
    stats.payload_seek_calls += 1
    payload = payload_handle.read()
    stats.payload_read_calls += 1
    stats.payload_bytes += len(payload)
    if len(payload) != stored.payload_size_bytes:
        raise IOError(
            f"Short payload read for {stored.codec}: {len(payload)} != {stored.payload_size_bytes}"
        )
    return payload


def read_payload_slice(
    stored: StoredChunk,
    payload_handle: Any,
    offset: int,
    length: int,
    stats: IOStats,
) -> bytes:
    if payload_handle is None:
        raise RuntimeError("Payload handle must be opened once at query scope")
    payload_handle.seek(offset)
    stats.payload_seek_calls += 1
    payload = payload_handle.read(length)
    stats.payload_read_calls += 1
    stats.payload_bytes += len(payload)
    if len(payload) != length:
        raise IOError(
            f"Short payload slice for {stored.codec}: {len(payload)} != {length}"
        )
    return payload


def representation_registry(zstd_level: int) -> Dict[str, PhysicalRepresentation]:
    reps: List[PhysicalRepresentation] = [
        WholeLZ4(),
        WholeZstd(level=zstd_level),
        BlockLZ4(64),
        BlockLZ4(256),
        BlockLZ4(1024),
    ]
    return {rep.name: rep for rep in reps}


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


def query_class(query_type: str, target_fraction: float) -> str:
    if query_type == "point":
        return "point"
    if query_type == "threshold_count":
        return "threshold"
    if query_type == "range_scan":
        if target_fraction <= 0.01 + 1e-12:
            return "short_scan"
        if target_fraction <= 0.10 + 1e-12:
            return "medium_scan"
        return "long_scan"
    if target_fraction <= 0.01 + 1e-12:
        return "short_agg"
    if target_fraction <= 0.10 + 1e-12:
        return "medium_agg"
    return "long_agg"


def build_queries(
    x: np.ndarray,
    chunk_id: str,
    seed: int,
    windows_per_fraction: int,
) -> List[QuerySpec]:
    rng = stable_rng(seed, chunk_id)
    n = int(x.size)
    queries: List[QuerySpec] = []

    point_count = max(6, windows_per_fraction * 3)
    for i, p in enumerate(rng.integers(0, n, size=point_count)):
        queries.append(
            QuerySpec(
                f"{chunk_id}:point:{i}",
                "point",
                "point",
                int(p),
                int(p + 1),
                1.0 / n,
                0.0,
            )
        )

    fractions = (0.001, 0.01, 0.10, 0.50, 1.00)
    for target_fraction in fractions:
        length = max(1, min(n, int(round(n * target_fraction))))
        actual_fraction = length / n
        starts = (
            [0]
            if length == n
            else rng.integers(0, n - length + 1, size=windows_per_fraction).tolist()
        )
        for j, start in enumerate(starts):
            start_i = int(start)
            end_i = start_i + length
            for query_type in (
                "range_scan",
                "range_sum",
                "range_avg",
                "range_min",
                "range_max",
            ):
                queries.append(
                    QuerySpec(
                        f"{chunk_id}:{query_type}:f{target_fraction:.3f}:{j}",
                        query_type,
                        query_class(query_type, target_fraction),
                        start_i,
                        end_i,
                        actual_fraction,
                        target_fraction,
                    )
                )

    for quantile in (0.50, 0.90, 0.99):
        threshold = float(np.quantile(x, quantile))
        for target_fraction in (0.01, 0.10, 1.00):
            length = max(1, min(n, int(round(n * target_fraction))))
            actual_fraction = length / n
            starts = (
                [0]
                if length == n
                else rng.integers(0, n - length + 1, size=windows_per_fraction).tolist()
            )
            for j, start in enumerate(starts):
                start_i = int(start)
                queries.append(
                    QuerySpec(
                        f"{chunk_id}:threshold:q{quantile:.2f}:f{target_fraction:.3f}:{j}",
                        "threshold_count",
                        "threshold",
                        start_i,
                        start_i + length,
                        actual_fraction,
                        target_fraction,
                        threshold,
                        quantile,
                    )
                )

    present = {query.query_class for query in queries}
    if present != EXPECTED_QUERY_CLASSES:
        raise AssertionError(
            f"Incomplete query classes for {chunk_id}: present={sorted(present)}, "
            f"expected={sorted(EXPECTED_QUERY_CLASSES)}"
        )
    return queries


def default_data_dirs() -> Tuple[Path, Path]:
    if os.name == "nt":
        return WINDOWS_PEMS04_DIR, WINDOWS_PEMS08_DIR
    return LINUX_PEMS04_DIR, LINUX_PEMS08_DIR


def auto_detect_data_file(directory: Path, dataset_name: str) -> Path:
    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}\n"
            "Use --pems04-dir/--pems08-dir or --input to provide the correct path."
        )
    candidates = list(directory.rglob("*.npz")) + list(directory.rglob("*.npy"))
    if not candidates:
        raise FileNotFoundError(f"No .npz or .npy found under {directory}")

    name_token = dataset_name.lower().replace("-", "").replace("_", "")
    bad_tokens = ("adj", "distance", "dist", "graph", "edge", "index", "mean", "std")

    def score(path: Path) -> Tuple[int, int]:
        normalized = path.stem.lower().replace("-", "").replace("_", "")
        value = 0
        if "pems04" in name_token and "pems04" in normalized:
            value += 100
        if "pems08" in name_token and "pems08" in normalized:
            value += 100
        if path.suffix.lower() == ".npz":
            value += 10
        if any(token in normalized for token in bad_tokens):
            value -= 100
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        return value, size

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
    for column_index in range(out.shape[1]):
        column = out[:, column_index]
        valid = np.isfinite(column)
        if not valid.any():
            column[:] = 0.0
        elif not valid.all():
            positions = np.arange(column.size)
            column[~valid] = np.interp(
                positions[~valid], positions[valid], column[valid]
            ).astype(np.float32)
    return out, fraction


def load_dataset(
    path: Path,
    feature_index: int,
    max_series: int,
    seed: int,
    excluded_series: Optional[set[int]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            keys = list(archive.keys())
            key = "data" if "data" in keys else keys[0]
            arr = np.asarray(archive[key])
        meta: Dict[str, Any] = {
            "source": str(path),
            "array_key": key,
            "available_keys": keys,
            "original_shape": list(arr.shape),
        }
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

    matrix, nonfinite_fraction = interpolate_nonfinite(np.asarray(matrix, dtype=np.float32))
    all_indices = np.arange(matrix.shape[1], dtype=int)
    if excluded_series:
        allowed = np.asarray(
            [index for index in all_indices if int(index) not in excluded_series],
            dtype=int,
        )
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
    meta.update(
        {
            "selected_original_series_indices": chosen.tolist(),
            "excluded_series_indices": sorted(excluded_series) if excluded_series else [],
            "loaded_shape": list(matrix.shape),
            "imputed_nonfinite_fraction": nonfinite_fraction,
            "dtype": str(matrix.dtype),
        }
    )
    return matrix, meta


def synthetic_dataset(seed: int, max_series: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    length = 20000
    num_series = max(8, max_series)
    timestamps = np.arange(length, dtype=np.float32)
    matrix = np.empty((length, num_series), dtype=np.float32)
    for series_index in range(num_series):
        matrix[:, series_index] = (
            20 * np.sin(2 * np.pi * timestamps / (288 + 11 * (series_index % 5)))
            + 5 * np.sin(2 * np.pi * timestamps / (48 + 3 * (series_index % 4)))
            + (timestamps // (1000 + 50 * (series_index % 3))) * (series_index % 3)
            + rng.normal(0, 0.5 + 0.1 * (series_index % 3), size=length)
        ).astype(np.float32)
    return matrix[:, :max_series], {
        "source": "synthetic",
        "original_shape": [length, num_series],
        "loaded_shape": [length, max_series],
    }


def extract_chunks(
    matrix: np.ndarray,
    chunk_sizes: Sequence[int],
    max_chunks_per_size: int,
    seed: int,
) -> List[ChunkUnit]:
    rng = np.random.default_rng(seed)
    time_length, num_series = matrix.shape
    chunks: List[ChunkUnit] = []

    for chunk_size in chunk_sizes:
        if chunk_size > time_length:
            print(f"[WARN] skip chunk_size={chunk_size}; time_length={time_length}")
            continue
        candidates = [
            (series_index, start)
            for series_index in range(num_series)
            for start in range(0, time_length - chunk_size + 1, chunk_size)
        ]
        if len(candidates) > max_chunks_per_size:
            selected = np.sort(
                rng.choice(len(candidates), size=max_chunks_per_size, replace=False)
            )
            candidates = [candidates[int(index)] for index in selected]
        for series_index, start in candidates:
            values = np.ascontiguousarray(
                matrix[start : start + chunk_size, series_index], dtype=np.float32
            )
            chunks.append(
                ChunkUnit(
                    f"cs{chunk_size}_s{series_index}_t{start}",
                    series_index,
                    start,
                    chunk_size,
                    values,
                )
            )

    if not chunks:
        raise ValueError("No chunks generated")
    return chunks


def linux_cache_eviction_supported() -> bool:
    return (
        os.name == "posix"
        and hasattr(os, "posix_fadvise")
        and hasattr(os, "POSIX_FADV_DONTNEED")
    )


def evict_file_cache_advisory(path: Optional[Path]) -> bool:
    if path is None:
        return True
    if not linux_cache_eviction_supported():
        return False
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError as exc:
            raise RuntimeError(
                f"POSIX_FADV_DONTNEED failed for {path}: {exc}. "
                "For WSL2 cold-mode runs, copy the data and output directory to the "
                "native Linux filesystem (for example under ~/repressdb_data), not /mnt/c or /mnt/e."
            ) from exc
    finally:
        os.close(fd)
    return True


def cache_control_label(mode: CacheMode) -> str:
    if mode.evict_payload_before_repeat or mode.evict_metadata_before_repeat:
        return "posix_fadvise_dontneed_advisory"
    if mode.persistent_payload_handle and mode.persistent_metadata_handle:
        return "warm_os_cache_persistent_payload_and_metadata_handles"
    if mode.persistent_payload_handle:
        return "warm_os_cache_persistent_payload_handle"
    return "warm_os_cache_reopen_per_query"


def resolve_modes(mode_argument: str) -> List[CacheMode]:
    if mode_argument.strip().lower() == "auto":
        # The default pilot is deliberately portable and never calls a close/reopen
        # measurement "cold". Advisory Linux cold modes must be selected explicitly
        # with --modes all or by name after placing data on a native Linux/WSL2 ext4 path.
        names = [
            "warm_cached",
            "warm_slicedmeta",
            "reopen_cached",
            "reopen_slicedmeta",
        ]
    elif mode_argument.strip().lower() == "all":
        names = list(CACHE_MODES)
    else:
        names = [name.strip() for name in mode_argument.split(",") if name.strip()]

    unknown = [name for name in names if name not in CACHE_MODES]
    if unknown:
        raise ValueError(f"Unknown cache modes: {unknown}; valid={list(CACHE_MODES)}")

    resolved: List[CacheMode] = []
    for name in names:
        mode = CACHE_MODES[name]
        requires_linux = mode.evict_payload_before_repeat or mode.evict_metadata_before_repeat
        if requires_linux and not linux_cache_eviction_supported():
            print(
                f"[WARN] skip {name}: POSIX_FADV_DONTNEED is unavailable on "
                f"{platform.system()}. This mode is not relabeled as cold."
            )
            continue
        resolved.append(mode)
    if not resolved:
        raise RuntimeError("No supported cache modes remain")
    return resolved


def encode_one(
    rep: PhysicalRepresentation,
    x: np.ndarray,
    stem: Path,
    repeats: int,
) -> Tuple[StoredChunk, int]:
    stem.with_suffix(".bin").unlink(missing_ok=True)
    stem.with_suffix(".meta.bin").unlink(missing_ok=True)

    start_ns = time.perf_counter_ns()
    stored = rep.encode_to_files(x, stem)
    times = [time.perf_counter_ns() - start_ns]

    if not stored.data_path.exists():
        raise FileNotFoundError(f"Encoded payload disappeared: {stored.data_path}")
    if stored.metadata_path is not None and not stored.metadata_path.exists():
        raise FileNotFoundError(f"Encoded metadata disappeared: {stored.metadata_path}")

    for repeat_index in range(1, repeats):
        trial_stem = stem.parent / f"{stem.name}__trial{repeat_index}"
        start_ns = time.perf_counter_ns()
        trial = rep.encode_to_files(x, trial_stem)
        times.append(time.perf_counter_ns() - start_ns)
        trial.data_path.unlink(missing_ok=True)
        if trial.metadata_path is not None:
            trial.metadata_path.unlink(missing_ok=True)

        if not stored.data_path.exists():
            raise FileNotFoundError(
                f"Retained payload was overwritten by trial encoding: {stored.data_path}"
            )
        if stored.metadata_path is not None and not stored.metadata_path.exists():
            raise FileNotFoundError(
                f"Retained metadata was overwritten by trial encoding: {stored.metadata_path}"
            )

    return stored, int(np.median(np.asarray(times, dtype=np.int64)))


def measure_full_decode_warm(
    rep: PhysicalRepresentation,
    stored: StoredChunk,
    warmups: int,
    repeats: int,
) -> Tuple[int, int]:
    handle = open(stored.data_path, "rb")
    try:
        for _ in range(warmups):
            stats = IOStats()
            rep.decode_all(stored, stored.metadata, handle, stats)
        times: List[int] = []
        bytes_read = 0
        for _ in range(repeats):
            stats = IOStats()
            start_ns = time.perf_counter_ns()
            rep.decode_all(stored, stored.metadata, handle, stats)
            times.append(time.perf_counter_ns() - start_ns)
            bytes_read = stats.payload_bytes
        return int(np.median(np.asarray(times, dtype=np.int64))), int(bytes_read)
    finally:
        handle.close()


def run_query_once(
    rep: PhysicalRepresentation,
    stored: StoredChunk,
    query: QuerySpec,
    mode: CacheMode,
    persistent_handle: Any,
    persistent_metadata_handle: Any,
) -> Tuple[Any, IOStats]:
    stats = IOStats()

    owns_payload_handle = persistent_handle is None
    if owns_payload_handle:
        stats.payload_open_calls += 1
        payload_handle = open(stored.data_path, "rb")
    else:
        payload_handle = persistent_handle

    metadata_handle = persistent_metadata_handle
    owns_metadata_handle = False
    if mode.metadata_sliced and stored.metadata_path is not None:
        if metadata_handle is None:
            stats.metadata_open_calls += 1
            metadata_handle = open(stored.metadata_path, "rb")
            owns_metadata_handle = True

    try:
        if mode.metadata_sliced and isinstance(rep, BlockLZ4):
            result = rep.execute_sliced_metadata(
                stored, query, metadata_handle, payload_handle, stats
            )
        else:
            metadata = load_metadata_for_query(stored, mode, stats)
            result = rep.execute(stored, query, metadata, payload_handle, stats)
        return result, stats
    finally:
        if owns_metadata_handle and metadata_handle is not None:
            metadata_handle.close()
        if owns_payload_handle:
            payload_handle.close()


def measure_query(
    rep: PhysicalRepresentation,
    stored: StoredChunk,
    query: QuerySpec,
    mode: CacheMode,
    persistent_handle: Any,
    persistent_metadata_handle: Any,
    warmups: int,
    repeats: int,
) -> Tuple[int, float, Any, IOStats]:
    cold_mode = mode.evict_payload_before_repeat or mode.evict_metadata_before_repeat

    if cold_mode:
        run_query_once(
            rep, stored, query, mode, persistent_handle, persistent_metadata_handle
        )
    else:
        for _ in range(warmups):
            run_query_once(
                rep, stored, query, mode, persistent_handle, persistent_metadata_handle
            )

    wall_times: List[int] = []
    cpu_times: List[int] = []
    last_result: Any = None
    last_stats = IOStats()

    for _ in range(repeats):
        if mode.evict_payload_before_repeat:
            if not evict_file_cache_advisory(stored.data_path):
                raise RuntimeError("Payload cache eviction requested but unsupported")
        if mode.evict_metadata_before_repeat and stored.metadata_path is not None:
            if not evict_file_cache_advisory(stored.metadata_path):
                raise RuntimeError("Metadata cache eviction requested but unsupported")

        cpu_start = time.process_time_ns()
        wall_start = time.perf_counter_ns()
        result, stats = run_query_once(
            rep, stored, query, mode, persistent_handle, persistent_metadata_handle
        )
        wall_times.append(time.perf_counter_ns() - wall_start)
        cpu_times.append(time.process_time_ns() - cpu_start)
        last_result = result
        last_stats = stats

    cpu_median = int(np.median(np.asarray(cpu_times, dtype=np.int64)))
    cpu_value = float(cpu_median) if cpu_median > 0 else math.nan
    return (
        int(np.median(np.asarray(wall_times, dtype=np.int64))),
        cpu_value,
        last_result,
        last_stats,
    )


def benchmark_dataset(
    matrix: np.ndarray,
    dataset_meta: Dict[str, Any],
    dataset_label: str,
    output_dir: Path,
    modes: Sequence[CacheMode],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_dir = output_dir / "_storage"
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    storage_dir.mkdir(parents=True)

    reps = representation_registry(args.zstd_level)
    chunks = extract_chunks(
        matrix,
        args.chunk_sizes,
        args.max_chunks_per_size,
        args.seed,
    )
    print(
        f"[{dataset_label}] chunks={len(chunks)}, representations={len(reps)}, "
        f"modes={[mode.name for mode in modes]}"
    )

    codec_rows: List[Dict[str, Any]] = []
    query_rows: List[Dict[str, Any]] = []

    for chunk_number, chunk in enumerate(chunks, 1):
        chunk_dir = storage_dir / chunk.chunk_id
        chunk_dir.mkdir(parents=True)
        stored_map: Dict[str, StoredChunk] = {}

        for rep_name, rep in reps.items():
            stem = chunk_dir / rep_name
            stored, encode_write_ns = encode_one(
                rep, chunk.values, stem, args.encode_repeats
            )

            validation_stats = IOStats()
            with open(stored.data_path, "rb") as handle:
                decoded = rep.decode_all(stored, stored.metadata, handle, validation_stats)
            if not np.array_equal(decoded.view(np.uint32), chunk.values.view(np.uint32)):
                raise AssertionError(
                    f"Lossless round trip failed: {chunk.chunk_id}, {rep_name}"
                )

            full_decode_ns, full_decode_bytes = measure_full_decode_warm(
                rep,
                stored,
                args.warmups,
                args.decode_repeats,
            )
            stored_map[rep_name] = stored
            codec_rows.append(
                {
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
                    "full_read_decode_warm_ns": full_decode_ns,
                    "full_read_decode_payload_bytes": full_decode_bytes,
                }
            )

        queries = build_queries(
            chunk.values,
            chunk.chunk_id,
            args.seed,
            args.windows_per_fraction,
        )

        for mode in modes:
            persistent_handles: Dict[str, Any] = {}
            persistent_metadata_handles: Dict[str, Any] = {}
            if mode.persistent_payload_handle:
                persistent_handles = {
                    rep_name: open(stored.data_path, "rb")
                    for rep_name, stored in stored_map.items()
                }
            if mode.metadata_sliced and mode.persistent_metadata_handle:
                persistent_metadata_handles = {
                    rep_name: open(stored.metadata_path, "rb")
                    for rep_name, stored in stored_map.items()
                    if stored.metadata_path is not None
                }

            try:
                tasks = [
                    (query, rep_name)
                    for query in queries
                    for rep_name in reps
                ]
                shuffle_rng = random.Random(
                    int.from_bytes(
                        hashlib.sha256(
                            f"{args.seed}:{chunk.chunk_id}:{mode.name}".encode("utf-8")
                        ).digest()[:8],
                        "little",
                    )
                )
                shuffle_rng.shuffle(tasks)

                for query, rep_name in tasks:
                    rep = reps[rep_name]
                    stored = stored_map[rep_name]
                    truth = execute_on_array(chunk.values, query)
                    handle = persistent_handles.get(rep_name)
                    metadata_handle = persistent_metadata_handles.get(rep_name)
                    latency_ns, cpu_ns, result, stats = measure_query(
                        rep,
                        stored,
                        query,
                        mode,
                        handle,
                        metadata_handle,
                        args.warmups,
                        args.query_repeats,
                    )
                    if not results_equal(result, truth):
                        raise AssertionError(
                            f"Incorrect query result: chunk={chunk.chunk_id}, mode={mode.name}, "
                            f"codec={rep_name}, query={query.query_id}, result={result}, truth={truth}"
                        )

                    query_rows.append(
                        {
                            "dataset": dataset_label,
                            "cache_mode": mode.name,
                            "cache_control": cache_control_label(mode),
                            "metadata_policy": (
                                "read_query_relevant_slices_from_file"
                                if mode.metadata_sliced
                                else (
                                    "read_full_metadata_file_per_query"
                                    if mode.metadata_from_file
                                    else "cached_in_python_memory"
                                )
                            ),
                            "metadata_handle_policy": (
                                "persistent"
                                if mode.metadata_sliced and mode.persistent_metadata_handle
                                else (
                                    "reopen_per_query" if mode.metadata_sliced else "not_applicable"
                                )
                            ),
                            "payload_handle_policy": (
                                "persistent"
                                if mode.persistent_payload_handle
                                else "reopen_per_query"
                            ),
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
                            "target_fraction": query.target_fraction,
                            "threshold": query.threshold,
                            "threshold_quantile": query.threshold_quantile,
                            "latency_ns": latency_ns,
                            "cpu_time_ns": cpu_ns,
                            "payload_bytes_read": stats.payload_bytes,
                            "metadata_bytes_read": stats.metadata_bytes,
                            "total_logical_bytes_read": stats.total_logical_bytes,
                            "payload_open_calls": stats.payload_open_calls,
                            "metadata_open_calls": stats.metadata_open_calls,
                            "total_open_calls": stats.total_open_calls,
                            "payload_read_calls": stats.payload_read_calls,
                            "metadata_read_calls": stats.metadata_read_calls,
                            "total_read_calls": stats.total_read_calls,
                            "payload_seek_calls": stats.payload_seek_calls,
                            "metadata_seek_calls": stats.metadata_seek_calls,
                            "total_seek_calls": stats.total_seek_calls,
                            "result": result,
                            "ground_truth": truth,
                            "correct": True,
                        }
                    )
            finally:
                for handle in persistent_metadata_handles.values():
                    handle.close()
                for handle in persistent_handles.values():
                    handle.close()

        if (
            chunk_number == 1
            or chunk_number % max(1, len(chunks) // 20) == 0
            or chunk_number == len(chunks)
        ):
            print(f"[{dataset_label}] {chunk_number}/{len(chunks)} chunks")

    codec_df = pd.DataFrame(codec_rows)
    query_df = pd.DataFrame(query_rows)
    validate_output_completeness(codec_df, query_df, chunks, reps, modes)

    codec_summary = summarize_codecs(codec_df)
    query_summary = summarize_queries(query_df)
    tests = paired_tests(query_df)
    workload_costs, best_layouts, drift = workload_analysis(codec_df, query_df)
    mode_comparison = compare_modes(best_layouts, query_summary)
    assessment = make_e0c_assessment(best_layouts, drift, mode_comparison, modes)

    codec_df.to_csv(output_dir / "codec_records.csv", index=False)
    query_df.to_csv(output_dir / "query_records.csv", index=False)
    codec_summary.to_csv(output_dir / "codec_summary.csv", index=False)
    query_summary.to_csv(output_dir / "query_summary.csv", index=False)
    tests.to_csv(output_dir / "statistical_tests.csv", index=False)
    workload_costs.to_csv(output_dir / "workload_costs.csv", index=False)
    best_layouts.to_csv(output_dir / "workload_best_layouts.csv", index=False)
    drift.to_csv(output_dir / "workload_drift_analysis.csv", index=False)
    mode_comparison.to_csv(output_dir / "cache_mode_comparison.csv", index=False)

    dataset_meta = dict(dataset_meta)
    dataset_meta.update(
        {
            "dataset_label": dataset_label,
            "num_chunks": len(chunks),
            "chunk_sizes": args.chunk_sizes,
            "cache_modes": [mode.name for mode in modes],
            "target_venue": TARGET_VENUE,
        }
    )
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(dataset_meta, f, ensure_ascii=False, indent=2)
    with open(output_dir / "e0c_assessment.json", "w", encoding="utf-8") as f:
        json.dump(assessment, f, ensure_ascii=False, indent=2, allow_nan=True)

    config = vars(args).copy()
    config.update(
        {
            "version": VERSION,
            "e0b_protocol_version": E0B_PROTOCOL_VERSION,
            "target_venue": TARGET_VENUE,
            "dataset_label": dataset_label,
            "measurement_scope": (
                "File-backed query latency with explicit payload-open and metadata policies; "
                "sliced modes read only query-relevant metadata fields and block intervals. "
                "Exact logical payload and metadata bytes are counted. Linux cold modes use "
                "advisory POSIX_FADV_DONTNEED and are not claimed as guaranteed physical SSD reads."
            ),
            "representations": list(reps),
            "workloads": WORKLOADS,
            "cache_modes_resolved": [asdict(mode) for mode in modes],
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
                "cpu_timer": "time.process_time_ns",
                "cpu_timer_note": "Per-query CPU time is written as NaN when platform timer resolution yields zero.",
                "posix_fadvise_supported": linux_cache_eviction_supported(),
            },
        }
    )
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    if not args.keep_storage:
        shutil.rmtree(storage_dir)

    print(f"[{dataset_label}] assessment_status={assessment['status']}")
    return assessment


def validate_output_completeness(
    codec_df: pd.DataFrame,
    query_df: pd.DataFrame,
    chunks: Sequence[ChunkUnit],
    reps: Dict[str, PhysicalRepresentation],
    modes: Sequence[CacheMode],
) -> None:
    expected_codec_rows = len(chunks) * len(reps)
    if len(codec_df) != expected_codec_rows:
        raise AssertionError(
            f"codec_records incomplete: {len(codec_df)} != {expected_codec_rows}"
        )
    if codec_df.duplicated(["chunk_id", "codec"]).any():
        raise AssertionError("Duplicate chunk_id-codec rows in codec_records")

    expected_query_classes = EXPECTED_QUERY_CLASSES
    for (mode_name, chunk_size), group in query_df.groupby(["cache_mode", "chunk_size"]):
        present = set(group["query_class"].unique())
        if present != expected_query_classes:
            raise AssertionError(
                f"Incomplete query classes for mode={mode_name}, chunk_size={chunk_size}: "
                f"{sorted(present)}"
            )

    expected_modes = {mode.name for mode in modes}
    present_modes = set(query_df["cache_mode"].unique())
    if present_modes != expected_modes:
        raise AssertionError(
            f"Cache modes incomplete: present={sorted(present_modes)}, expected={sorted(expected_modes)}"
        )
    if not bool(query_df["correct"].all()):
        raise AssertionError("At least one query result is incorrect")


def summarize_codecs(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "compression_ratio",
        "encode_write_ns",
        "full_read_decode_warm_ns",
        "stored_size_bytes",
        "payload_size_bytes",
        "metadata_size_bytes",
    ]
    rows: List[Dict[str, Any]] = []
    for (chunk_size, codec), group in df.groupby(["chunk_size", "codec"]):
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            rows.append(
                {
                    "chunk_size": int(chunk_size),
                    "codec": codec,
                    "metric": metric,
                    "n_chunks": len(group),
                    "median": float(np.median(values)),
                    "mean": float(np.mean(values)),
                }
            )
    return pd.DataFrame(rows)


def summarize_queries(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_columns = ["cache_mode", "chunk_size", "query_class", "codec"]
    for keys, group in df.groupby(group_columns):
        cache_mode, chunk_size, query_class_name, codec = keys
        cpu_values = group["cpu_time_ns"].dropna()
        median_cpu_time_ns = (
            float(cpu_values.median()) if not cpu_values.empty else math.nan
        )
        rows.append(
            {
                "cache_mode": cache_mode,
                "chunk_size": int(chunk_size),
                "query_class": query_class_name,
                "codec": codec,
                "n_query_instances": len(group),
                "n_chunks": int(group["chunk_id"].nunique()),
                "median_latency_ns": float(group["latency_ns"].median()),
                "mean_latency_ns": float(group["latency_ns"].mean()),
                "p95_latency_ns": float(group["latency_ns"].quantile(0.95)),
                "p99_latency_ns": float(group["latency_ns"].quantile(0.99)),
                "median_cpu_time_ns": median_cpu_time_ns,
                "median_payload_bytes_read": float(group["payload_bytes_read"].median()),
                "median_metadata_bytes_read": float(group["metadata_bytes_read"].median()),
                "median_total_logical_bytes_read": float(
                    group["total_logical_bytes_read"].median()
                ),
                "median_total_open_calls": float(group["total_open_calls"].median()),
                "median_total_read_calls": float(group["total_read_calls"].median()),
                "median_total_seek_calls": float(group["total_seek_calls"].median()),
            }
        )
    return pd.DataFrame(rows)


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (count - rank) * float(p_values[index])
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    codecs = sorted(df["codec"].unique())
    rows: List[Dict[str, Any]] = []

    for (cache_mode, chunk_size, query_class_name), group in df.groupby(
        ["cache_mode", "chunk_size", "query_class"]
    ):
        pivot = group.pivot(index="query_id", columns="codec", values="latency_ns").dropna()
        local_rows: List[Dict[str, Any]] = []
        p_values: List[float] = []

        for index, codec_a in enumerate(codecs):
            for codec_b in codecs[index + 1 :]:
                if codec_a not in pivot or codec_b not in pivot:
                    continue
                try:
                    p_value = float(
                        wilcoxon(
                            pivot[codec_a],
                            pivot[codec_b],
                            alternative="two-sided",
                            zero_method="wilcox",
                        ).pvalue
                    )
                except ValueError:
                    p_value = 1.0
                ratio = (
                    pivot[codec_a].to_numpy(dtype=float)
                    / pivot[codec_b].to_numpy(dtype=float)
                )
                local_rows.append(
                    {
                        "cache_mode": cache_mode,
                        "chunk_size": int(chunk_size),
                        "query_class": query_class_name,
                        "codec_a": codec_a,
                        "codec_b": codec_b,
                        "n_pairs": len(pivot),
                        "median_ratio_a_over_b": float(np.median(ratio)),
                        "wilcoxon_p": p_value,
                    }
                )
                p_values.append(p_value)

        adjusted = holm_adjust(p_values) if p_values else []
        for row, adjusted_p in zip(local_rows, adjusted):
            row["wilcoxon_p_holm"] = adjusted_p
            rows.append(row)

    return pd.DataFrame(rows)


def workload_analysis(
    codec_df: pd.DataFrame,
    query_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    class_medians = (
        query_df.groupby(
            ["cache_mode", "chunk_id", "chunk_size", "codec", "query_class"],
            as_index=False,
        )["latency_ns"]
        .median()
    )

    workload_rows: List[Dict[str, Any]] = []
    for workload_name, weights in WORKLOADS.items():
        temp = class_medians.copy()
        temp["weight"] = temp["query_class"].map(weights).fillna(0.0)
        temp["weighted_latency_ns"] = temp["latency_ns"] * temp["weight"]
        costs = temp.groupby(
            ["cache_mode", "chunk_id", "chunk_size", "codec"], as_index=False
        )["weighted_latency_ns"].sum()
        costs["workload"] = workload_name
        workload_rows.extend(costs.to_dict("records"))

    workload_df = pd.DataFrame(workload_rows)
    best_indices = workload_df.groupby(
        ["cache_mode", "chunk_id", "chunk_size", "workload"]
    )["weighted_latency_ns"].idxmin()
    best = workload_df.loc[
        best_indices,
        [
            "cache_mode",
            "chunk_id",
            "chunk_size",
            "workload",
            "codec",
            "weighted_latency_ns",
        ],
    ].copy()
    best = best.rename(
        columns={
            "codec": "best_codec",
            "weighted_latency_ns": "best_expected_latency_ns",
        }
    )

    codec_costs = codec_df.set_index(["chunk_id", "codec"])[
        ["encode_write_ns", "full_read_decode_warm_ns", "stored_size_bytes"]
    ]
    drift_rows: List[Dict[str, Any]] = []

    for cache_mode, mode_best in best.groupby("cache_mode"):
        old = mode_best[mode_best["workload"] == "scan_heavy"].set_index("chunk_id")
        mode_costs = workload_df[workload_df["cache_mode"] == cache_mode]

        for new_workload in ("point_heavy", "aggregate_heavy"):
            new = mode_best[mode_best["workload"] == new_workload].set_index("chunk_id")
            new_costs = mode_costs[mode_costs["workload"] == new_workload].set_index(
                ["chunk_id", "codec"]
            )

            for chunk_id, old_row in old.iterrows():
                old_codec = str(old_row["best_codec"])
                new_codec = str(new.loc[chunk_id, "best_codec"])
                old_cost = float(
                    new_costs.loc[(chunk_id, old_codec), "weighted_latency_ns"]
                )
                new_cost = float(
                    new_costs.loc[(chunk_id, new_codec), "weighted_latency_ns"]
                )
                saving = old_cost - new_cost
                rewrite_ns = float(
                    codec_costs.loc[(chunk_id, old_codec), "full_read_decode_warm_ns"]
                    + codec_costs.loc[(chunk_id, new_codec), "encode_write_ns"]
                )
                rewrite_bytes = int(
                    codec_costs.loc[(chunk_id, old_codec), "stored_size_bytes"]
                    + codec_costs.loc[(chunk_id, new_codec), "stored_size_bytes"]
                )
                drift_rows.append(
                    {
                        "cache_mode": cache_mode,
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
                        "break_even_future_queries": (
                            rewrite_ns / saving if saving > 0 else math.inf
                        ),
                        "rewrite_cost_scope": (
                            "warm full decode of old representation plus encode+fsync of new representation"
                        ),
                    }
                )

    return workload_df, best, pd.DataFrame(drift_rows)


def bootstrap_median_ci(
    values: np.ndarray,
    seed: int = 20260718,
    iterations: int = 4000,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=float)
    for index in range(iterations):
        samples[index] = np.median(
            rng.choice(values, size=values.size, replace=True)
        )
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def e0b_style_mode_decision(drift: pd.DataFrame) -> Dict[str, Any]:
    strata: Dict[str, Any] = {}
    qualifying_chunk_sizes: set[int] = set()
    criterion_2 = False
    criterion_3 = False

    for (new_workload, chunk_size), group in drift.groupby(
        ["new_workload", "chunk_size"]
    ):
        finite = group[np.isfinite(group["regret_ratio"])].copy()
        switch_fraction = float(finite["codec_switch"].mean())
        median_regret = float(finite["regret_ratio"].median())
        lower, upper = bootstrap_median_ci(finite["regret_ratio"].to_numpy())
        switched = finite[
            finite["codec_switch"]
            & np.isfinite(finite["break_even_future_queries"])
        ]
        fraction_break_even_5000 = (
            float((switched["break_even_future_queries"] <= 5000).mean())
            if not switched.empty
            else 0.0
        )

        if switch_fraction >= 0.50:
            qualifying_chunk_sizes.add(int(chunk_size))
        if switch_fraction >= 0.50 and median_regret >= 1.20 and lower >= 1.05:
            criterion_2 = True
        if switch_fraction >= 0.50 and fraction_break_even_5000 >= 0.50:
            criterion_3 = True

        strata[f"{new_workload}|{int(chunk_size)}"] = {
            "n_chunks": int(len(finite)),
            "switch_fraction": switch_fraction,
            "median_regret_ratio": median_regret,
            "bootstrap_95ci_median_regret": [lower, upper],
            "fraction_switched_break_even_le_5000": fraction_break_even_5000,
        }

    criterion_1 = len(qualifying_chunk_sizes) >= 2
    return {
        "C1_workload_specific_layouts": bool(criterion_1),
        "C2_material_regret": bool(criterion_2),
        "C3_file_backed_amortization": bool(criterion_3),
        "all_three": bool(criterion_1 and criterion_2 and criterion_3),
        "strata": strata,
    }


def compare_modes(best: pd.DataFrame, query_summary: pd.DataFrame) -> pd.DataFrame:
    if "warm_cached" not in set(best["cache_mode"]):
        return pd.DataFrame()

    reference_modal = (
        best[best["cache_mode"] == "warm_cached"]
        .groupby(["workload", "chunk_size"])["best_codec"]
        .agg(lambda values: values.value_counts().idxmax())
        .rename("warm_cached_modal_codec")
    )

    latency_reference = query_summary[
        query_summary["cache_mode"] == "warm_cached"
    ].set_index(["chunk_size", "query_class", "codec"])["median_latency_ns"]

    rows: List[Dict[str, Any]] = []
    for cache_mode, mode_best in best.groupby("cache_mode"):
        modal = (
            mode_best.groupby(["workload", "chunk_size"])["best_codec"]
            .agg(lambda values: values.value_counts().idxmax())
            .rename("mode_modal_codec")
        )
        joined = pd.concat([reference_modal, modal], axis=1).dropna()
        layout_agreement = float(
            (joined["warm_cached_modal_codec"] == joined["mode_modal_codec"]).mean()
        )

        mode_summary = query_summary[query_summary["cache_mode"] == cache_mode].set_index(
            ["chunk_size", "query_class", "codec"]
        )["median_latency_ns"]
        aligned = pd.concat(
            [latency_reference.rename("reference"), mode_summary.rename("mode")],
            axis=1,
        ).dropna()
        ratios = aligned["mode"] / aligned["reference"]
        rows.append(
            {
                "cache_mode": cache_mode,
                "n_workload_chunk_strata": int(len(joined)),
                "modal_layout_agreement_with_warm_cached": layout_agreement,
                "n_aligned_query_summary_cells": int(len(aligned)),
                "median_latency_ratio_vs_warm_cached": float(ratios.median()),
                "p95_latency_ratio_vs_warm_cached": float(ratios.quantile(0.95)),
                "max_latency_ratio_vs_warm_cached": float(ratios.max()),
            }
        )
    return pd.DataFrame(rows)


def make_e0c_assessment(
    best: pd.DataFrame,
    drift: pd.DataFrame,
    mode_comparison: pd.DataFrame,
    modes: Sequence[CacheMode],
) -> Dict[str, Any]:
    mode_results: Dict[str, Any] = {}
    modal_tables: Dict[str, Any] = {}

    for mode_name, mode_drift in drift.groupby("cache_mode"):
        mode_results[mode_name] = e0b_style_mode_decision(mode_drift)
        mode_best = best[best["cache_mode"] == mode_name]
        modal = (
            mode_best.groupby(["workload", "chunk_size"])["best_codec"]
            .agg(lambda values: values.value_counts().idxmax())
            .reset_index()
        )
        modal_tables[mode_name] = modal.to_dict("records")

    resolved_names = [mode.name for mode in modes]
    cold_names = [name for name in resolved_names if name.startswith("linux_cold")]
    filemeta_names = [name for name in resolved_names if "meta" in name]
    slicedmeta_names = [name for name in resolved_names if "slicedmeta" in name]
    fullreload_names = [name for name in resolved_names if "fullreloadmeta" in name]

    comparison_lookup = (
        mode_comparison.set_index("cache_mode").to_dict("index")
        if not mode_comparison.empty
        else {}
    )

    if cold_names:
        status = "complete_with_linux_advisory_cold_modes"
    else:
        status = "portable_pilot_without_cold_modes"

    return {
        "version": VERSION,
        "target_venue": TARGET_VENUE,
        "status": status,
        "resolved_cache_modes": resolved_names,
        "cold_modes_available": cold_names,
        "file_metadata_modes_available": filemeta_names,
        "sliced_metadata_modes_available": slicedmeta_names,
        "full_reload_stress_modes_available": fullreload_names,
        "e0b_style_criteria_by_mode": mode_results,
        "modal_best_by_mode": modal_tables,
        "comparison_with_warm_cached": comparison_lookup,
        "interpretation_rules": {
            "warm_cached": "Direct replication boundary relative to frozen E0-B.",
            "warm_slicedmeta": (
                "Persistent file handles with only query-relevant metadata fields and block intervals read."
            ),
            "reopen_cached": (
                "Includes payload open/close overhead but does not claim cold-cache physical I/O."
            ),
            "reopen_slicedmeta": (
                "Primary portable file-backed mode: payload and metadata reopen per query, with sliced metadata reads."
            ),
            "full_reload_stress_modes": (
                "Diagnostic worst cases that reload and parse all metadata per query; not used as the primary system model."
            ),
            "linux_cold_modes": (
                "Use advisory POSIX_FADV_DONTNEED before each repeat; report as controlled/advisory cold-cache evidence."
            ),
        },
        "next_gate": (
            "Use PEMS04 to validate sliced metadata correctness and stability. If valid, freeze v0.2.0 and run PEMS08 holdout without retuning."
        ),
    }


def parse_args() -> argparse.Namespace:
    default_pems04, default_pems08 = default_data_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default="pems04",
        help="Comma-separated: pems04,pems08. Default runs only the PEMS04 pilot.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional single .npz/.npy input; overrides --datasets.",
    )
    parser.add_argument("--dataset-label", type=str, default="CUSTOM")
    parser.add_argument("--output", type=str, default="results_e0c_custom")
    parser.add_argument("--pems04-dir", type=str, default=str(default_pems04))
    parser.add_argument("--pems08-dir", type=str, default=str(default_pems08))
    parser.add_argument("--feature-index", type=int, default=0)
    parser.add_argument("--chunk-sizes", type=str, default="2048,8192,16384")
    parser.add_argument("--max-series", type=int, default=32)
    parser.add_argument("--max-chunks-per-size", type=int, default=32)
    parser.add_argument("--zstd-level", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--encode-repeats", type=int, default=1)
    parser.add_argument("--decode-repeats", type=int, default=3)
    parser.add_argument("--query-repeats", type=int, default=3)
    parser.add_argument("--windows-per-fraction", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument(
        "--modes",
        type=str,
        default="auto",
        help=(
            "auto, all, or comma-separated cache modes. Auto runs cached and sliced-metadata "
            "warm/reopen modes. Select all explicitly to include diagnostic full-reload and Linux cold modes."
        ),
    )
    parser.add_argument("--keep-storage", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    args.chunk_sizes = [
        int(value.strip())
        for value in args.chunk_sizes.split(",")
        if value.strip()
    ]
    args.datasets = [
        value.strip().lower()
        for value in args.datasets.split(",")
        if value.strip()
    ]

    valid_datasets = {"pems04", "pems08"}
    invalid_datasets = [value for value in args.datasets if value not in valid_datasets]
    if invalid_datasets:
        raise ValueError(
            f"Invalid datasets: {invalid_datasets}; valid={sorted(valid_datasets)}"
        )

    if args.quick:
        args.chunk_sizes = [2048, 8192]
        args.max_series = min(args.max_series, 4)
        args.max_chunks_per_size = 3
        args.encode_repeats = 1
        args.decode_repeats = 1
        args.query_repeats = 1
        args.windows_per_fraction = 1
        if args.modes == "auto":
            args.modes = "warm_cached,reopen_slicedmeta"
    return args


def main() -> int:
    args = parse_args()
    modes = resolve_modes(args.modes)

    print("=" * 88)
    print(f"RePressDB E0-C v{VERSION}")
    print(f"Target venue: {TARGET_VENUE} (CCF-B)")
    print("Cache/metadata boundary experiment with query-relevant sliced metadata I/O.")
    print("Cold modes, when present, use advisory POSIX_FADV_DONTNEED and are labeled accordingly.")
    print("=" * 88)

    if args.quick and args.input is None:
        matrix, meta = synthetic_dataset(args.seed, args.max_series)
        assessment = benchmark_dataset(
            matrix,
            meta,
            "SYNTHETIC_QUICK",
            Path("results_e0c_quick"),
            modes,
            args,
        )
        with open("E0C_QUICK_ASSESSMENT.json", "w", encoding="utf-8") as f:
            json.dump(assessment, f, ensure_ascii=False, indent=2, allow_nan=True)
        return 0

    if args.input:
        input_path = Path(args.input)
        matrix, meta = load_dataset(
            input_path,
            args.feature_index,
            args.max_series,
            args.seed,
        )
        assessment = benchmark_dataset(
            matrix,
            meta,
            args.dataset_label,
            Path(args.output),
            modes,
            args,
        )
        with open(
            Path(args.output).parent / "E0C_CUSTOM_ASSESSMENT.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(assessment, f, ensure_ascii=False, indent=2, allow_nan=True)
        return 0

    jobs: List[Tuple[str, Path, Path, Optional[set[int]]]] = []
    if "pems04" in args.datasets:
        jobs.append(
            (
                "PEMS04_E0C_PILOT",
                Path(args.pems04_dir),
                Path("results_e0c_pems04_v020"),
                None,
            )
        )
    if "pems08" in args.datasets:
        jobs.append(
            (
                "PEMS08_E0C_HOLDOUT",
                Path(args.pems08_dir),
                Path("results_e0c_pems08_holdout_v020"),
                PEMS08_E0A_SERIES,
            )
        )

    assessments: Dict[str, Any] = {}
    for label, directory, output_dir, excluded in jobs:
        data_file = auto_detect_data_file(directory, label)
        matrix, meta = load_dataset(
            data_file,
            args.feature_index,
            args.max_series,
            args.seed,
            excluded,
        )
        assessments[label] = benchmark_dataset(
            matrix,
            meta,
            label,
            output_dir,
            modes,
            args,
        )

    with open("E0C_COMBINED_ASSESSMENTS_v020.json", "w", encoding="utf-8") as f:
        json.dump(assessments, f, ensure_ascii=False, indent=2, allow_nan=True)

    print("Done.")
    print("Upload the result directory/directories and E0C_COMBINED_ASSESSMENTS_v020.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
