#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RePressDB B3 v1.1.0
1/5/10-GiB replicated-tenant logical scale-out, clean/dirty startup,
bounded migration log, and foreground/background concurrency.

DIRECT RUN
----------
Place beside:
  RUN_THIS_ONLY_REPRESSDB_B2_FULL_BDG2_SCALE_V1_0_0.py
  RUN_THIS_ONLY_REPRESSDB_B1_SINGLE_ACTIVE_STORE_V0_9_0.py
  RUN_THIS_ONLY_REPRESSDB_E0C_V0_2_0.py
Keep:
  results_b2_full_bdg2_scale_v100/
Run this file in PyCharm without command-line arguments.

Scientific boundary: 1/5/10 GiB are deterministic replicated-tenant LOGICAL
raw sizes based on the frozen real B2 BDG2 template corpus. Immutable template
payloads are physically shared. Migrated logical chunks use copy-on-write.
Logical bytes, shared physical bytes, private bytes, catalog bytes, and
telemetry bytes are reported separately.
"""
from __future__ import annotations

import collections
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

VERSION = "1.1.0"
TARGET_JOURNAL = "Journal of Big Data"
B2_SCRIPT_NAME = "RUN_THIS_ONLY_REPRESSDB_B2_FULL_BDG2_SCALE_V1_0_0.py"
B1_SCRIPT_NAME = "RUN_THIS_ONLY_REPRESSDB_B1_SINGLE_ACTIVE_STORE_V0_9_0.py"
E0C_SCRIPT_NAME = "RUN_THIS_ONLY_REPRESSDB_E0C_V0_2_0.py"
B2_RESULT_DIR_NAME = "results_b2_full_bdg2_scale_v100"
OUTPUT_DIR_NAME = "results_b3_scale_concurrency_v110"
SUMMARY_JSON_NAME = "B3_SCALE_CONCURRENCY_ASSESSMENT_v110.json"
SEED = 20260726
ZSTD_LEVEL = 3
RAW_CHUNK_VALUES = 8192
RAW_CHUNK_BYTES = RAW_CHUNK_VALUES * 4
SCALE_TARGETS_GIB = (1, 5, 10)
QUERY_THREADS = (1, 4, 8)
MODES = ("foreground_only", "background_migration")
REPEATS = 3
QUERIES_PER_CELL = 50_000
BACKGROUND_MIGRATIONS_PER_CELL = 50
STARTUP_REPEATS = 5
DIRTY_START_REPEATS = 3
COMPLETED_LOG_RETAIN = 32
CATALOG_BUILD_BATCH = 5000
SNAPSHOT_FETCH_BATCH = 10000
MAX_HANDLES_PER_QUERY_THREAD = 64
LOCK_STRIPES = 8192
MEMORY_SAMPLE_INTERVAL_QUERIES = 5000
CODECS = ("whole_lz4", "whole_zstd", "block_lz4_64", "block_lz4_256", "block_lz4_1024")


def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(json_safe(value), f, ensure_ascii=False, indent=2, allow_nan=True)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def write_dataframe_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block: break
            digest.update(block)
    return digest.hexdigest()


def raw_array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float32).tobytes()).hexdigest()


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def load_module(path: Path, name: str) -> Any:
    if not path.exists(): raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sqlite_file_bytes(path: Path) -> Dict[str, int]:
    items = {"database_bytes": path, "wal_bytes": Path(str(path)+"-wal"), "shm_bytes": Path(str(path)+"-shm")}
    out = {k: (int(p.stat().st_size) if p.exists() else 0) for k, p in items.items()}
    out["total_catalog_files_bytes"] = int(sum(out.values()))
    return out


def safe_unlink(path: Optional[Path]) -> None:
    if path is None: return
    try: path.unlink(missing_ok=True)
    except PermissionError:
        time.sleep(0.05); path.unlink(missing_ok=True)


def fsync_directory(directory: Path) -> None:
    if os.name == "nt": return
    fd = os.open(str(directory), os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def checkpoint_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def copy_catalog(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for p in (destination, Path(str(destination)+"-wal"), Path(str(destination)+"-shm")): safe_unlink(p)
    shutil.copy2(source, destination)


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists(): return "existing"
    try:
        os.link(source, destination); return "hardlink"
    except Exception:
        shutil.copy2(source, destination); return "copy"


class MemoryTracker:
    def __init__(self) -> None:
        self.method = "unavailable"; self.peak_rss_bytes = 0; self.current_rss_bytes = 0
        self._process = None; self._resource = None
        try:
            import psutil
            self._process = psutil.Process(os.getpid()); self.method = "psutil_rss"
        except Exception:
            if os.name != "nt":
                try:
                    import resource
                    self._resource = resource; self.method = "resource_ru_maxrss"
                except Exception: pass
    def sample(self) -> int:
        value = 0
        if self._process is not None: value = int(self._process.memory_info().rss)
        elif self._resource is not None:
            raw = int(self._resource.getrusage(self._resource.RUSAGE_SELF).ru_maxrss)
            value = raw if platform.system() == "Darwin" else raw * 1024
        self.current_rss_bytes = value; self.peak_rss_bytes = max(self.peak_rss_bytes, value)
        return value


def connect_catalog(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=60.0, isolation_level=None, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL"); con.execute("PRAGMA synchronous=FULL")
    con.execute("PRAGMA foreign_keys=ON"); con.execute("PRAGMA busy_timeout=60000")
    return con


@contextlib.contextmanager
def immediate_transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    connection.execute("BEGIN IMMEDIATE")
    try: yield connection
    except Exception:
        connection.execute("ROLLBACK"); raise
    else: connection.execute("COMMIT")


def connect_telemetry(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=60.0, isolation_level=None, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL"); con.execute("PRAGMA synchronous=FULL"); con.execute("PRAGMA busy_timeout=60000")
    con.executescript("""
    CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS scale_build_records(
      scale_name TEXT PRIMARY KEY,target_gib INTEGER,target_raw_bytes INTEGER,
      logical_chunk_count INTEGER,logical_raw_bytes INTEGER,tenant_count INTEGER,
      base_catalog_bytes INTEGER,logical_active_stored_bytes INTEGER,
      shared_template_bytes INTEGER,completed_at_ns INTEGER);
    CREATE TABLE IF NOT EXISTS startup_records(
      scale_name TEXT,startup_type TEXT,repeat_index INTEGER,startup_ns INTEGER,
      recovery_ns INTEGER,snapshot_load_ns INTEGER,loaded_chunks INTEGER,
      inspected_pending_chunks INTEGER,full_directory_scan INTEGER,success INTEGER,
      PRIMARY KEY(scale_name,startup_type,repeat_index));
    CREATE TABLE IF NOT EXISTS concurrency_cells(
      cell_id TEXT PRIMARY KEY,scale_name TEXT,thread_count INTEGER,mode TEXT,
      repeat_index INTEGER,query_count INTEGER,correct_queries INTEGER,
      wall_elapsed_ns INTEGER,throughput_qps REAL,p50_latency_ns REAL,
      p95_latency_ns REAL,p99_latency_ns REAL,maximum_latency_ns REAL,
      p50_lock_wait_ns REAL,p95_lock_wait_ns REAL,p99_lock_wait_ns REAL,
      p50_execution_ns REAL,p95_execution_ns REAL,p99_execution_ns REAL,
      background_migrations INTEGER,background_migration_failures INTEGER,
      background_rewrite_bytes INTEGER,background_migration_elapsed_ns INTEGER,
      log_rows_before_compaction INTEGER,log_rows_after_compaction INTEGER,
      archived_log_rows INTEGER,work_catalog_bytes_before_compaction INTEGER,
      work_catalog_bytes_after_compaction INTEGER,peak_rss_bytes INTEGER,
      latency_file TEXT,completed_at_ns INTEGER);
    CREATE TABLE IF NOT EXISTS migration_archive(
      cell_id TEXT,migration_id TEXT,chunk_id TEXT,from_version INTEGER,to_version INTEGER,
      from_codec TEXT,target_codec TEXT,state TEXT,started_at_ns INTEGER,updated_at_ns INTEGER,
      completed_at_ns INTEGER,rewrite_bytes INTEGER,migration_time_ns INTEGER,
      PRIMARY KEY(cell_id,migration_id));
    CREATE TABLE IF NOT EXISTS fsck_records(
      scale_name TEXT,fsck_type TEXT,checked_chunks INTEGER,checksum_failures INTEGER,
      missing_file_failures INTEGER,elapsed_ns INTEGER,success INTEGER,
      PRIMARY KEY(scale_name,fsck_type));
    """)
    return con


def telemetry_set(con: sqlite3.Connection, key: str, value: Any) -> None:
    con.execute("INSERT INTO metadata(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True)))


def telemetry_get(con: sqlite3.Connection, key: str) -> Optional[Any]:
    row = con.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return None if row is None else json.loads(row["value"])


@dataclasses.dataclass(slots=True)
class TemplateRecord:
    template_index: int; template_chunk_id: str; truth_index: int; n: int; raw_checksum: str
    codec: str; data_path: str; metadata_path: Optional[str]; payload_bytes: int; metadata_bytes: int
    threshold_p50: float; threshold_p90: float; threshold_p99: float


@dataclasses.dataclass(slots=True)
class CatalogEntry:
    chunk_id: str; tenant_id: int; template_index: int; n: int; raw_checksum: str
    active_version: int; active_codec: str; active_shared: int
    active_data_path: Optional[str]; active_metadata_path: Optional[str]
    active_payload_bytes: int; active_metadata_bytes: int


@dataclasses.dataclass
class HandlePair:
    payload: Any; metadata: Any


class ScaleStore:
    def __init__(self, root: Path, base: Any, perform_recovery: bool=True, load_snapshot: bool=True,
                 maximum_handles_per_thread: int=MAX_HANDLES_PER_QUERY_THREAD):
        self.root = root.resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.private_root = self.root / "private"; self.staging_root = self.root / "staging"
        self.private_root.mkdir(parents=True, exist_ok=True); self.staging_root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "catalog.sqlite3"; self.base = base
        self.representations = base.representation_registry(ZSTD_LEVEL)
        self.mode = base.CACHE_MODES["warm_slicedmeta"]
        self.connection = connect_catalog(self.db_path); self._maximum_handles = int(maximum_handles_per_thread)
        self._thread_local = threading.local(); self._cache_registry_lock = threading.Lock(); self._cache_registry = []
        self._writer_lock = threading.RLock(); self._stripes = [threading.RLock() for _ in range(LOCK_STRIPES)]
        self.templates: Dict[int,TemplateRecord] = {}; self.snapshot: Dict[str,CatalogEntry] = {}; self.chunk_ids: List[str] = []
        self.recovery_stats = {"dirty_at_open":False,"inspected_pending_chunks":0,"recovery_ns":0,"full_directory_scan":False}
        self.snapshot_load_ns = 0
        if perform_recovery:
            # Templates are needed when a dirty committed private version must be validated.
            self.load_templates()
            start = time.perf_counter_ns(); dirty = int(self.get_meta("store_dirty",0)); self.recovery_stats["dirty_at_open"] = bool(dirty)
            if dirty: self.targeted_recover()
            self.recovery_stats["recovery_ns"] = time.perf_counter_ns()-start
        if load_snapshot:
            start = time.perf_counter_ns(); self.load_templates(); self.load_snapshot(); self.snapshot_load_ns = time.perf_counter_ns()-start

    def close(self, mark_clean: bool=True) -> None:
        if mark_clean:
            pending = int(self.connection.execute("SELECT COUNT(*) n FROM chunks WHERE migration_state!='idle'").fetchone()["n"])
            if pending == 0: self.set_meta("store_dirty",0)
        self.close_all_handles(); checkpoint_database(self.connection); self.connection.close()

    def get_meta(self,key:str,default:Any=None)->Any:
        row=self.connection.execute("SELECT value FROM store_metadata WHERE key=?",(key,)).fetchone()
        return default if row is None else json.loads(row["value"])
    def set_meta(self,key:str,value:Any)->None:
        self.connection.execute("INSERT INTO store_metadata(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                                (key,json.dumps(json_safe(value),ensure_ascii=False,sort_keys=True)))
    def _stripe(self,chunk_id:str)->threading.RLock: return self._stripes[stable_seed("stripe",chunk_id)%len(self._stripes)]

    def load_templates(self)->None:
        self.templates.clear()
        for r in self.connection.execute("SELECT * FROM templates ORDER BY template_index").fetchall():
            t=TemplateRecord(int(r["template_index"]),str(r["template_chunk_id"]),int(r["truth_index"]),int(r["n"]),str(r["raw_checksum"]),
                             str(r["codec"]),str(r["data_path"]),str(r["metadata_path"]) if r["metadata_path"] is not None else None,
                             int(r["payload_bytes"]),int(r["metadata_bytes"]),float(r["threshold_p50"]),float(r["threshold_p90"]),float(r["threshold_p99"]))
            self.templates[t.template_index]=t

    def load_snapshot(self)->None:
        self.snapshot.clear(); self.chunk_ids.clear()
        cur=self.connection.execute("""SELECT chunk_id,tenant_id,template_index,n,raw_checksum,active_version,active_codec,active_shared,
          active_data_path,active_metadata_path,active_payload_bytes,active_metadata_bytes FROM chunks ORDER BY chunk_ordinal""")
        while True:
            rows=cur.fetchmany(SNAPSHOT_FETCH_BATCH)
            if not rows: break
            for r in rows:
                e=CatalogEntry(str(r["chunk_id"]),int(r["tenant_id"]),int(r["template_index"]),int(r["n"]),str(r["raw_checksum"]),
                  int(r["active_version"]),str(r["active_codec"]),int(r["active_shared"]),
                  str(r["active_data_path"]) if r["active_data_path"] is not None else None,
                  str(r["active_metadata_path"]) if r["active_metadata_path"] is not None else None,
                  int(r["active_payload_bytes"]),int(r["active_metadata_bytes"]))
                self.snapshot[e.chunk_id]=e; self.chunk_ids.append(e.chunk_id)

    def _resolve_paths(self,e:CatalogEntry)->Tuple[str,Optional[str],int,int]:
        if e.active_shared:
            t=self.templates[e.template_index]; return t.data_path,t.metadata_path,t.payload_bytes,t.metadata_bytes
        if e.active_data_path is None: raise RuntimeError(f"Private entry missing path: {e.chunk_id}")
        return e.active_data_path,e.active_metadata_path,e.active_payload_bytes,e.active_metadata_bytes

    def _stored(self,e:CatalogEntry)->Any:
        data,meta,pbytes,mbytes=self._resolve_paths(e); dp=Path(data); mp=Path(meta) if meta else None
        if mp is not None: metadata=self.base.parse_block_metadata(mp.read_bytes())
        elif e.active_codec=="whole_zstd": metadata={"level":ZSTD_LEVEL}
        else: metadata={}
        return self.base.StoredChunk(codec=e.active_codec,n=e.n,data_path=dp,metadata_path=mp,
          payload_size_bytes=pbytes,metadata_size_bytes=mbytes,metadata=metadata)

    def _thread_cache(self)->collections.OrderedDict:
        cache=getattr(self._thread_local,"cache",None)
        if cache is None:
            cache=collections.OrderedDict(); self._thread_local.cache=cache
            with self._cache_registry_lock: self._cache_registry.append(cache)
        return cache

    def _cached_handles(self,stored:Any)->HandlePair:
        cache=self._thread_cache(); key=(str(stored.data_path),str(stored.metadata_path) if stored.metadata_path else "")
        pair=cache.get(key)
        if pair is not None: cache.move_to_end(key); return pair
        pair=HandlePair(open(stored.data_path,"rb"),open(stored.metadata_path,"rb") if stored.metadata_path else None)
        cache[key]=pair; cache.move_to_end(key)
        while len(cache)>self._maximum_handles:
            _,old=cache.popitem(last=False)
            for h in (old.payload,old.metadata):
                if h is not None:
                    try:h.close()
                    except Exception:pass
        return pair

    def invalidate_paths(self,paths:Iterable[Path])->None:
        normalized={str(p) for p in paths if p is not None}
        if not normalized:return
        with self._cache_registry_lock:
            for cache in self._cache_registry:
                for key in list(cache.keys()):
                    if key[0] in normalized or key[1] in normalized:
                        pair=cache.pop(key)
                        for h in (pair.payload,pair.metadata):
                            if h is not None:
                                try:h.close()
                                except Exception:pass

    def close_all_handles(self)->None:
        with self._cache_registry_lock:
            for cache in self._cache_registry:
                for pair in cache.values():
                    for h in (pair.payload,pair.metadata):
                        if h is not None:
                            try:h.close()
                            except Exception:pass
                cache.clear()
            self._cache_registry.clear()

    def execute_query(self,chunk_id:str,query:Any)->Tuple[Any,int,int,int,Any,CatalogEntry]:
        total_start=time.perf_counter_ns(); lock=self._stripe(chunk_id); wait_start=time.perf_counter_ns(); lock.acquire()
        wait_ns=time.perf_counter_ns()-wait_start
        try:
            e=self.snapshot[chunk_id]; stored=self._stored(e); handles=self._cached_handles(stored); exec_start=time.perf_counter_ns()
            result,stats=self.base.run_query_once(self.representations[stored.codec],stored,query,self.mode,handles.payload,handles.metadata)
            exec_ns=time.perf_counter_ns()-exec_start
        finally: lock.release()
        return result,int(time.perf_counter_ns()-total_start),int(wait_ns),int(exec_ns),stats,e

    def _decode_entry(self,e:CatalogEntry)->np.ndarray:
        stored=self._stored(e); stats=self.base.IOStats()
        with open(stored.data_path,"rb") as payload:
            return self.representations[stored.codec].decode_all(stored,stored.metadata,payload,stats)

    def _private_stem(self,chunk_id:str,version:int,codec:str,staging:bool,token:str)->Path:
        root=self.staging_root if staging else self.private_root; directory=root/chunk_id; directory.mkdir(parents=True,exist_ok=True)
        return directory/f"v{version:08d}__{codec}{'__'+token+'.tmp' if staging else ''}"

    def _delete_private(self,shared:int,data:Optional[str],meta:Optional[str])->None:
        if shared:return
        paths=[Path(data) if data else None,Path(meta) if meta else None]
        self.invalidate_paths([p for p in paths if p is not None])
        for p in paths:safe_unlink(p)

    def migrate(self,chunk_id:str,target_codec:str)->Dict[str,Any]:
        with self._stripe(chunk_id):
          with self._writer_lock:
            old=self.snapshot[chunk_id]
            if target_codec==old.active_codec:return {"no_op":True,"migration_time_ns":0,"rewrite_bytes":0,"chunk_id":chunk_id}
            start=time.perf_counter_ns(); to_version=old.active_version+1
            migration_id=hashlib.sha256(f"{chunk_id}|{old.active_version}|{target_codec}|{time.time_ns()}".encode()).hexdigest(); token=migration_id[:16]
            temp_stem=self._private_stem(chunk_id,to_version,target_codec,True,token); final_stem=self._private_stem(chunk_id,to_version,target_codec,False,"")
            temp_data=temp_stem.with_suffix(".bin"); temp_meta=temp_stem.with_suffix(".meta.bin")
            final_data=final_stem.with_suffix(".bin"); final_meta=final_stem.with_suffix(".meta.bin")
            self.set_meta("store_dirty",1); now=time.time_ns()
            with immediate_transaction(self.connection):
                self.connection.execute("""INSERT INTO migration_log(migration_id,chunk_id,from_version,to_version,from_codec,target_codec,state,
                 temporary_data_path,temporary_metadata_path,final_data_path,final_metadata_path,started_at_ns,updated_at_ns)
                 VALUES(?,?,?,?,?,?,'intent',?,?,?,?,?,?)""",
                 (migration_id,chunk_id,old.active_version,to_version,old.active_codec,target_codec,str(temp_data),str(temp_meta),str(final_data),str(final_meta),now,now))
                self.connection.execute("UPDATE chunks SET migration_state='intent',updated_at_ns=? WHERE chunk_id=?",(now,chunk_id))
            values=self._decode_entry(old)
            if raw_array_sha256(values)!=old.raw_checksum:raise AssertionError("Active checksum failed")
            encoded=self.representations[target_codec].encode_to_files(values,temp_stem)
            candidate=CatalogEntry(chunk_id,old.tenant_id,old.template_index,old.n,old.raw_checksum,to_version,target_codec,0,
                                   str(encoded.data_path),str(encoded.metadata_path) if encoded.metadata_path else None,
                                   int(encoded.payload_size_bytes),int(encoded.metadata_size_bytes))
            if raw_array_sha256(self._decode_entry(candidate))!=old.raw_checksum:raise AssertionError("Target checksum failed")
            final_data.parent.mkdir(parents=True,exist_ok=True); safe_unlink(final_data);safe_unlink(final_meta);os.replace(encoded.data_path,final_data)
            if encoded.metadata_path is not None:
                os.replace(encoded.metadata_path,final_meta); final_meta_text=str(final_meta); final_meta_bytes=int(final_meta.stat().st_size)
            else:
                safe_unlink(temp_meta); final_meta_text=None; final_meta_bytes=0
            fsync_directory(final_data.parent); final_payload_bytes=int(final_data.stat().st_size); commit=time.time_ns()
            with immediate_transaction(self.connection):
                self.connection.execute("""UPDATE chunks SET previous_version=active_version,previous_codec=active_codec,previous_shared=active_shared,
                 previous_template_index=active_template_index,previous_data_path=active_data_path,previous_metadata_path=active_metadata_path,
                 previous_payload_bytes=active_payload_bytes,previous_metadata_bytes=active_metadata_bytes,
                 active_version=?,active_codec=?,active_shared=0,active_template_index=NULL,active_data_path=?,active_metadata_path=?,
                 active_payload_bytes=?,active_metadata_bytes=?,migration_state='committed_pending_cleanup',updated_at_ns=? WHERE chunk_id=?""",
                 (to_version,target_codec,str(final_data),final_meta_text,final_payload_bytes,final_meta_bytes,commit,chunk_id))
                self.connection.execute("UPDATE migration_log SET state='catalog_committed',rewrite_bytes=?,updated_at_ns=? WHERE migration_id=?",
                 (final_payload_bytes+final_meta_bytes,commit,migration_id))
            new=CatalogEntry(chunk_id,old.tenant_id,old.template_index,old.n,old.raw_checksum,to_version,target_codec,0,str(final_data),final_meta_text,
                             final_payload_bytes,final_meta_bytes);self.snapshot[chunk_id]=new
            if raw_array_sha256(self._decode_entry(new))!=old.raw_checksum:raise AssertionError("Committed checksum failed")
            prev=self.connection.execute("SELECT previous_shared,previous_data_path,previous_metadata_path FROM chunks WHERE chunk_id=?",(chunk_id,)).fetchone()
            self._delete_private(int(prev["previous_shared"] or 0),prev["previous_data_path"],prev["previous_metadata_path"])
            elapsed=time.perf_counter_ns()-start;done=time.time_ns()
            with immediate_transaction(self.connection):
                self.connection.execute("""UPDATE chunks SET previous_version=NULL,previous_codec=NULL,previous_shared=NULL,previous_template_index=NULL,
                 previous_data_path=NULL,previous_metadata_path=NULL,previous_payload_bytes=NULL,previous_metadata_bytes=NULL,
                 migration_state='idle',updated_at_ns=? WHERE chunk_id=?""",(done,chunk_id))
                self.connection.execute("UPDATE migration_log SET state='completed',migration_time_ns=?,completed_at_ns=?,updated_at_ns=? WHERE migration_id=?",
                                        (int(elapsed),done,done,migration_id))
            self.set_meta("store_dirty",0)
            return {"no_op":False,"migration_id":migration_id,"chunk_id":chunk_id,"from_codec":old.active_codec,"target_codec":target_codec,
                    "from_version":old.active_version,"to_version":to_version,"migration_time_ns":int(elapsed),
                    "rewrite_bytes":int(final_payload_bytes+final_meta_bytes)}

    def targeted_recover(self)->None:
        start=time.perf_counter_ns();rows=self.connection.execute("SELECT * FROM chunks WHERE migration_state!='idle' ORDER BY chunk_ordinal").fetchall()
        self.recovery_stats["inspected_pending_chunks"]=len(rows)
        for r in rows:
            chunk_id=str(r["chunk_id"]);state=str(r["migration_state"])
            log=self.connection.execute("SELECT * FROM migration_log WHERE chunk_id=? AND state NOT IN('completed','aborted','rolled_back') ORDER BY started_at_ns DESC LIMIT 1",(chunk_id,)).fetchone()
            if state=="committed_pending_cleanup":
                # B3 dirty benchmark injects only pre-commit intents; this branch is retained for completeness.
                active_ok=True
                if not int(r["active_shared"]):
                    e=CatalogEntry(str(r["chunk_id"]),int(r["tenant_id"]),int(r["template_index"]),int(r["n"]),str(r["raw_checksum"]),
                      int(r["active_version"]),str(r["active_codec"]),0,str(r["active_data_path"]),
                      str(r["active_metadata_path"]) if r["active_metadata_path"] else None,int(r["active_payload_bytes"]),int(r["active_metadata_bytes"]))
                    try:active_ok=raw_array_sha256(self._decode_entry(e))==str(r["raw_checksum"])
                    except Exception:active_ok=False
                if active_ok:
                    self._delete_private(int(r["previous_shared"] or 0),r["previous_data_path"],r["previous_metadata_path"])
                    with immediate_transaction(self.connection):
                        self.connection.execute("""UPDATE chunks SET previous_version=NULL,previous_codec=NULL,previous_shared=NULL,previous_template_index=NULL,
                         previous_data_path=NULL,previous_metadata_path=NULL,previous_payload_bytes=NULL,previous_metadata_bytes=NULL,migration_state='idle',updated_at_ns=?
                         WHERE chunk_id=?""",(time.time_ns(),chunk_id))
                        if log is not None:self.connection.execute("UPDATE migration_log SET state='completed',recovery_action='targeted_cleanup',completed_at_ns=?,updated_at_ns=? WHERE migration_id=?",
                                                                  (time.time_ns(),time.time_ns(),str(log["migration_id"])))
                else:raise RuntimeError("Committed active invalid during targeted recovery")
            else:
                if log is not None:
                    for col in ("temporary_data_path","temporary_metadata_path","final_data_path","final_metadata_path"):
                        if log[col]:safe_unlink(Path(str(log[col])))
                with immediate_transaction(self.connection):
                    self.connection.execute("UPDATE chunks SET migration_state='idle',updated_at_ns=? WHERE chunk_id=?",(time.time_ns(),chunk_id))
                    if log is not None:self.connection.execute("UPDATE migration_log SET state='aborted',recovery_action='targeted_abort',completed_at_ns=?,updated_at_ns=? WHERE migration_id=?",
                                                              (time.time_ns(),time.time_ns(),str(log["migration_id"])))
        self.set_meta("store_dirty",0);self.recovery_stats["recovery_ns"]=time.perf_counter_ns()-start

    def compact_completed_logs(self)->Tuple[List[Dict[str,Any]],Dict[str,int]]:
        before=int(self.connection.execute("SELECT COUNT(*) n FROM migration_log").fetchone()["n"])
        keep={str(r["migration_id"]) for r in self.connection.execute("SELECT migration_id FROM migration_log WHERE state='completed' ORDER BY completed_at_ns DESC,updated_at_ns DESC LIMIT ?",(COMPLETED_LOG_RETAIN,)).fetchall()}
        completed=self.connection.execute("SELECT * FROM migration_log WHERE state='completed' ORDER BY completed_at_ns").fetchall()
        archive=[dict(r) for r in completed if str(r["migration_id"]) not in keep]
        if archive:
            with immediate_transaction(self.connection):
                self.connection.executemany("DELETE FROM migration_log WHERE migration_id=?",[(str(r["migration_id"]),) for r in archive])
        checkpoint_database(self.connection);self.connection.execute("PRAGMA incremental_vacuum")
        after=int(self.connection.execute("SELECT COUNT(*) n FROM migration_log").fetchone()["n"])
        return archive,{"before":before,"after":after,"archived":len(archive)}

    def offline_fsck(self,maximum_chunks:Optional[int])->Dict[str,Any]:
        start=time.perf_counter_ns()
        if maximum_chunks is None:selected=list(self.chunk_ids);fsck_type="full"
        else:
            count=min(maximum_chunks,len(self.chunk_ids));selected=random.Random(stable_seed(SEED,"fsck",len(self.chunk_ids),maximum_chunks)).sample(self.chunk_ids,count);fsck_type=f"sample_{count}"
        checksum_failures=0;missing=0
        for chunk_id in selected:
            e=self.snapshot[chunk_id];stored=self._stored(e)
            if not stored.data_path.exists() or (stored.metadata_path is not None and not stored.metadata_path.exists()):missing+=1;continue
            try:
                if raw_array_sha256(self._decode_entry(e))!=e.raw_checksum:checksum_failures+=1
            except Exception:checksum_failures+=1
        return {"fsck_type":fsck_type,"checked_chunks":len(selected),"checksum_failures":checksum_failures,
                "missing_file_failures":missing,"elapsed_ns":time.perf_counter_ns()-start,
                "success":checksum_failures==0 and missing==0}

def create_catalog_schema(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA auto_vacuum=INCREMENTAL")
    connection.executescript("""
    CREATE TABLE IF NOT EXISTS store_metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS templates(
      template_index INTEGER PRIMARY KEY,template_chunk_id TEXT UNIQUE,truth_index INTEGER,n INTEGER,
      raw_checksum TEXT,codec TEXT,data_path TEXT,metadata_path TEXT,payload_bytes INTEGER,metadata_bytes INTEGER,
      threshold_p50 REAL,threshold_p90 REAL,threshold_p99 REAL);
    CREATE TABLE IF NOT EXISTS chunks(
      chunk_id TEXT PRIMARY KEY,chunk_ordinal INTEGER UNIQUE,tenant_id INTEGER,template_index INTEGER,n INTEGER,raw_checksum TEXT,
      active_version INTEGER,active_codec TEXT,active_shared INTEGER,active_template_index INTEGER,
      active_data_path TEXT,active_metadata_path TEXT,active_payload_bytes INTEGER,active_metadata_bytes INTEGER,
      previous_version INTEGER,previous_codec TEXT,previous_shared INTEGER,previous_template_index INTEGER,
      previous_data_path TEXT,previous_metadata_path TEXT,previous_payload_bytes INTEGER,previous_metadata_bytes INTEGER,
      migration_state TEXT,created_at_ns INTEGER,updated_at_ns INTEGER,
      FOREIGN KEY(template_index) REFERENCES templates(template_index));
    CREATE INDEX IF NOT EXISTS idx_chunks_tenant ON chunks(tenant_id);
    CREATE INDEX IF NOT EXISTS idx_chunks_template ON chunks(template_index);
    CREATE INDEX IF NOT EXISTS idx_chunks_state ON chunks(migration_state);
    CREATE TABLE IF NOT EXISTS migration_log(
      migration_id TEXT PRIMARY KEY,chunk_id TEXT,from_version INTEGER,to_version INTEGER,from_codec TEXT,target_codec TEXT,state TEXT,
      temporary_data_path TEXT,temporary_metadata_path TEXT,final_data_path TEXT,final_metadata_path TEXT,
      rewrite_bytes INTEGER,migration_time_ns INTEGER,started_at_ns INTEGER,updated_at_ns INTEGER,completed_at_ns INTEGER,
      recovery_action TEXT,error_text TEXT,FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id));
    CREATE INDEX IF NOT EXISTS idx_migration_state ON migration_log(state);
    CREATE INDEX IF NOT EXISTS idx_migration_chunk ON migration_log(chunk_id);
    """)


def build_template_registry(output_dir: Path, b2_dir: Path) -> Tuple[pd.DataFrame, Path]:
    registry_path=output_dir/"template_registry.csv";truth_path=b2_dir/"truth"/"data_scale_8192.float32.bin"
    manifest_path=b2_dir/"data_scale_8192_chunk_manifest.csv";source_catalog=b2_dir/"stores"/"data_scale_8192"/"catalog.sqlite3"
    for p in (truth_path,manifest_path,source_catalog):
        if not p.exists():raise FileNotFoundError(p)
    if registry_path.exists():
        reg=pd.read_csv(registry_path);ok=True
        for r in reg.itertuples(index=False):
            if not Path(str(r.data_path)).exists():ok=False;break
            mp=getattr(r,"metadata_path",None)
            if pd.notna(mp) and str(mp) not in ("","None") and not Path(str(mp)).exists():ok=False;break
        if ok:return reg,truth_path
    manifest=pd.read_csv(manifest_path);con=sqlite3.connect(source_catalog);con.row_factory=sqlite3.Row
    catalog=pd.read_sql_query("SELECT chunk_id,n,raw_checksum,active_codec,active_data_path,active_metadata_path,active_payload_bytes,active_metadata_bytes FROM chunks ORDER BY chunk_id",con);con.close()
    merged=manifest.merge(catalog,on="chunk_id",how="inner",validate="one_to_one").sort_values("chunk_id").reset_index(drop=True)
    if len(merged)!=len(manifest):raise AssertionError("B2 manifest/catalog mismatch")
    truth=np.memmap(truth_path,mode="r",dtype=np.float32,shape=(len(manifest),RAW_CHUNK_VALUES));source_root=b2_dir/"stores"/"data_scale_8192";template_root=output_dir/"templates"
    rows=[]
    for index,row in enumerate(merged.to_dict("records")):
        src_data=source_root/Path(str(row["active_data_path"]).replace("\\",os.sep))
        src_meta=(source_root/Path(str(row["active_metadata_path"]).replace("\\",os.sep)) if pd.notna(row["active_metadata_path"]) and str(row["active_metadata_path"]) not in ("","None") else None)
        if not src_data.exists():raise FileNotFoundError(src_data)
        stem=template_root/f"template_{index:04d}__{row['active_codec']}";dst_data=stem.with_suffix(".bin");method=link_or_copy(src_data,dst_data)
        if src_meta is not None:dst_meta=stem.with_suffix(".meta.bin");link_or_copy(src_meta,dst_meta)
        else:dst_meta=None
        values=np.asarray(truth[int(row["truth_index"])],dtype=np.float32);checksum=raw_array_sha256(values)
        if checksum!=str(row["raw_checksum"]):raise AssertionError(f"Truth checksum mismatch: {row['chunk_id']}")
        rows.append({"template_index":index,"template_chunk_id":str(row["chunk_id"]),"truth_index":int(row["truth_index"]),"n":int(row["n"]),
          "raw_checksum":checksum,"codec":str(row["active_codec"]),"data_path":str(dst_data.resolve()),
          "metadata_path":str(dst_meta.resolve()) if dst_meta is not None else "","payload_bytes":int(dst_data.stat().st_size),
          "metadata_bytes":int(dst_meta.stat().st_size) if dst_meta is not None else 0,
          "threshold_p50":float(np.quantile(values,.5)),"threshold_p90":float(np.quantile(values,.9)),"threshold_p99":float(np.quantile(values,.99)),"link_method":method})
        if index==0 or index+1==len(merged) or (index+1)%100==0:print(f"[template registry] {index+1}/{len(merged)}")
    del truth;reg=pd.DataFrame(rows);write_dataframe_atomic(registry_path,reg);return reg,truth_path


def scale_name(target_gib:int)->str:return f"logical_{target_gib}GiB"


def build_scale_catalog(output_dir:Path,registry:pd.DataFrame,target_gib:int,freeze_hash:str,telemetry:sqlite3.Connection)->Dict[str,Any]:
    name=scale_name(target_gib);root=output_dir/"base_scales"/name;root.mkdir(parents=True,exist_ok=True);catalog=root/"catalog.sqlite3"
    target_bytes=target_gib*1024**3;logical_chunks=int(math.ceil(target_bytes/RAW_CHUNK_BYTES));logical_raw=logical_chunks*RAW_CHUNK_BYTES
    template_count=len(registry);tenant_count=int(math.ceil(logical_chunks/template_count));con=connect_catalog(catalog);create_catalog_schema(con)
    row=con.execute("SELECT value FROM store_metadata WHERE key='experiment_freeze_sha256'").fetchone()
    if row is not None and json.loads(row["value"])!=freeze_hash:con.close();raise RuntimeError(f"{name} belongs to another freeze")
    metadata={"experiment_freeze_sha256":freeze_hash,"scale_name":name,"target_gib":target_gib,"target_raw_bytes":target_bytes,
              "logical_chunk_count":logical_chunks,"logical_raw_bytes":logical_raw,"tenant_count":tenant_count,"store_dirty":0,
              "startup_policy":"clean catalog-only; dirty targeted pending-row recovery"}
    with immediate_transaction(con):
        for k,v in metadata.items():con.execute("INSERT INTO store_metadata(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(k,json.dumps(v)))
    existing_templates=int(con.execute("SELECT COUNT(*) n FROM templates").fetchone()["n"])
    if existing_templates==0:
        rows=[]
        for r in registry.to_dict("records"):
            mp=str(r["metadata_path"]) if pd.notna(r["metadata_path"]) and str(r["metadata_path"]) not in ("","None") else None
            rows.append((int(r["template_index"]),str(r["template_chunk_id"]),int(r["truth_index"]),int(r["n"]),str(r["raw_checksum"]),str(r["codec"]),
                         str(r["data_path"]),mp,int(r["payload_bytes"]),int(r["metadata_bytes"]),float(r["threshold_p50"]),float(r["threshold_p90"]),float(r["threshold_p99"])))
        with immediate_transaction(con):con.executemany("INSERT INTO templates VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",rows)
    elif existing_templates!=template_count:con.close();raise AssertionError(f"{name} template count mismatch")
    existing=int(con.execute("SELECT COUNT(*) n FROM chunks").fetchone()["n"]);regrows=registry.to_dict("records");now=time.time_ns()
    for start in range(existing,logical_chunks,CATALOG_BUILD_BATCH):
        end=min(logical_chunks,start+CATALOG_BUILD_BATCH);batch=[]
        for ordinal in range(start,end):
            ti=ordinal%template_count;tenant=ordinal//template_count;t=regrows[ti];cid=f"{name}__tenant_{tenant:06d}__template_{ti:04d}"
            batch.append((cid,ordinal,tenant,ti,int(t["n"]),str(t["raw_checksum"]),0,str(t["codec"]),1,ti,None,None,
                          int(t["payload_bytes"]),int(t["metadata_bytes"]),"idle",now,now))
        with immediate_transaction(con):con.executemany("""INSERT INTO chunks(chunk_id,chunk_ordinal,tenant_id,template_index,n,raw_checksum,
          active_version,active_codec,active_shared,active_template_index,active_data_path,active_metadata_path,active_payload_bytes,active_metadata_bytes,
          migration_state,created_at_ns,updated_at_ns) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",batch)
        print(f"[{name} catalog] {end}/{logical_chunks}")
    actual=int(con.execute("SELECT COUNT(*) n FROM chunks").fetchone()["n"])
    if actual!=logical_chunks:con.close();raise AssertionError(f"{name} count mismatch")
    checkpoint_database(con);page_size=int(con.execute("PRAGMA page_size").fetchone()[0]);page_count=int(con.execute("PRAGMA page_count").fetchone()[0])
    logical_stored=int(con.execute("SELECT COALESCE(SUM(active_payload_bytes+active_metadata_bytes),0) total FROM chunks").fetchone()["total"]);con.close()
    sizes=sqlite_file_bytes(catalog);shared=int((registry["payload_bytes"]+registry["metadata_bytes"]).sum())
    result={"scale_name":name,"target_gib":target_gib,"target_raw_bytes":target_bytes,"logical_chunk_count":logical_chunks,"logical_raw_bytes":logical_raw,
            "tenant_count":tenant_count,"template_count":template_count,"logical_active_stored_bytes":logical_stored,"shared_template_bytes":shared,"private_copy_on_write_bytes_base":0,
            "catalog_database_bytes":int(sizes["database_bytes"]),"catalog_total_files_bytes":int(sizes["total_catalog_files_bytes"]),
            "catalog_allocated_bytes":page_size*page_count,"catalog_bytes_per_logical_chunk":float(page_size*page_count/logical_chunks),
            "logical_compression_factor":float(logical_raw/logical_stored),"physical_shared_fraction_of_logical_stored":float(shared/logical_stored),
            "base_catalog_path":str(catalog)}
    telemetry.execute("INSERT OR REPLACE INTO scale_build_records VALUES(?,?,?,?,?,?,?,?,?,?)",
      (name,target_gib,target_bytes,logical_chunks,logical_raw,tenant_count,int(sizes["database_bytes"]),logical_stored,shared,time.time_ns()))
    write_json_atomic(output_dir/f"{name}_base_metrics.json",result);return result


def inject_dirty_intent(root:Path)->str:
    con=connect_catalog(root/"catalog.sqlite3");r=con.execute("SELECT chunk_id,active_version,active_codec FROM chunks ORDER BY chunk_ordinal LIMIT 1").fetchone()
    if r is None:con.close();raise RuntimeError("Empty catalog")
    cid=str(r["chunk_id"]);staging=root/"staging"/cid;staging.mkdir(parents=True,exist_ok=True);dummy=staging/"dirty.tmp.bin"
    with open(dummy,"wb") as f:f.write(b"dirty-recovery-test");f.flush();os.fsync(f.fileno())
    mid=hashlib.sha256(f"dirty|{cid}|{time.time_ns()}".encode()).hexdigest();now=time.time_ns();target="whole_zstd" if str(r["active_codec"])!="whole_zstd" else "whole_lz4"
    with immediate_transaction(con):
        con.execute("""INSERT INTO migration_log(migration_id,chunk_id,from_version,to_version,from_codec,target_codec,state,temporary_data_path,started_at_ns,updated_at_ns)
          VALUES(?,?,?,?,?,?,'intent',?,?,?)""",(mid,cid,int(r["active_version"]),int(r["active_version"])+1,str(r["active_codec"]),target,str(dummy),now,now))
        con.execute("UPDATE chunks SET migration_state='intent',updated_at_ns=? WHERE chunk_id=?",(now,cid))
        con.execute("INSERT INTO store_metadata(key,value) VALUES('store_dirty','1') ON CONFLICT(key) DO UPDATE SET value='1'")
    checkpoint_database(con);con.close();return cid


def benchmark_startup(output_dir:Path,metrics:Mapping[str,Any],base:Any,telemetry:sqlite3.Connection,memory:MemoryTracker)->Dict[str,Any]:
    name=str(metrics["scale_name"]);base_catalog=Path(str(metrics["base_catalog_path"]));startup_root=output_dir/"startup_work"/name;startup_root.mkdir(parents=True,exist_ok=True)
    for startup_type,repeats in (("clean",STARTUP_REPEATS),("dirty_targeted",DIRTY_START_REPEATS)):
        existing={int(r["repeat_index"]) for r in telemetry.execute("SELECT repeat_index FROM startup_records WHERE scale_name=? AND startup_type=?",(name,startup_type)).fetchall()}
        for rep in range(repeats):
            if rep in existing:continue
            work=startup_root/f"{startup_type}_{rep:02d}";shutil.rmtree(work,ignore_errors=True);work.mkdir(parents=True,exist_ok=True);copy_catalog(base_catalog,work/"catalog.sqlite3")
            injected=inject_dirty_intent(work) if startup_type=="dirty_targeted" else None
            start=time.perf_counter_ns();store=ScaleStore(work,base,True,True);elapsed=time.perf_counter_ns()-start;success=True
            if injected is not None:
                state=store.connection.execute("SELECT migration_state FROM chunks WHERE chunk_id=?",(injected,)).fetchone()["migration_state"]
                success=state=="idle" and int(store.get_meta("store_dirty",1))==0
            memory.sample();telemetry.execute("INSERT OR REPLACE INTO startup_records VALUES(?,?,?,?,?,?,?,?,?,?)",
              (name,startup_type,rep,int(elapsed),int(store.recovery_stats["recovery_ns"]),int(store.snapshot_load_ns),len(store.chunk_ids),
               int(store.recovery_stats["inspected_pending_chunks"]),int(store.recovery_stats["full_directory_scan"]),int(success)))
            store.close();shutil.rmtree(work,ignore_errors=True);print(f"[{name} startup] {startup_type} {rep+1}/{repeats}: {elapsed/1e9:.3f}s")
    frame=pd.read_sql_query("SELECT * FROM startup_records WHERE scale_name=? ORDER BY startup_type,repeat_index",telemetry,params=(name,));result={}
    for st,g in frame.groupby("startup_type"):
        vals=g["startup_ns"].to_numpy(float);result[str(st)]={"repeats":int(len(g)),"median_startup_ns":float(np.median(vals)),
          "p95_startup_ns":float(np.quantile(vals,.95)),"median_recovery_ns":float(g["recovery_ns"].median()),
          "median_snapshot_load_ns":float(g["snapshot_load_ns"].median()),"maximum_inspected_pending_chunks":int(g["inspected_pending_chunks"].max()),
          "any_full_directory_scan":bool(g["full_directory_scan"].astype(bool).any()),"all_success":bool(g["success"].astype(bool).all())}
    write_json_atomic(output_dir/f"{name}_startup_metrics.json",result);return result


def query_from_rng(base:Any,t:TemplateRecord,rng:random.Random,query_id:str)->Any:
    selector=rng.randrange(100)
    if selector<35:qtype="point"
    elif selector<50:qtype="range_scan"
    elif selector<62:qtype="range_sum"
    elif selector<74:qtype="range_avg"
    elif selector<84:qtype="range_min"
    elif selector<94:qtype="range_max"
    else:qtype="threshold_count"
    n=t.n
    if qtype=="point":start=rng.randrange(n);end=start+1;qclass="point";threshold=None
    else:
        fractions=(.001,.01,.1,.5,1.0);fraction=fractions[rng.randrange(len(fractions))];length=max(1,min(n,int(round(n*fraction))))
        start=0 if length==n else rng.randrange(0,n-length+1);end=start+length
        if qtype=="threshold_count":threshold=(t.threshold_p50,t.threshold_p90,t.threshold_p99)[rng.randrange(3)];qclass="threshold"
        else:threshold=None;qclass=base.query_class(qtype,(end-start)/n)
    return base.QuerySpec(query_id=query_id,query_type=qtype,query_class=qclass,start=start,end=end,range_fraction=(end-start)/n,
                          target_fraction=(end-start)/n,threshold=threshold,threshold_quantile=None)


def save_latency_file(path:Path,latency:np.ndarray,wait:np.ndarray,execution:np.ndarray)->None:
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+".tmp")
    with open(tmp,"wb") as f:np.savez_compressed(f,latency_ns=latency,lock_wait_ns=wait,execution_ns=execution);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def archive_logs(telemetry:sqlite3.Connection,cell_id:str,rows:Sequence[Mapping[str,Any]])->None:
    if not rows:return
    telemetry.executemany("""INSERT OR REPLACE INTO migration_archive(cell_id,migration_id,chunk_id,from_version,to_version,from_codec,target_codec,state,
      started_at_ns,updated_at_ns,completed_at_ns,rewrite_bytes,migration_time_ns) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
      [(cell_id,str(r["migration_id"]),str(r["chunk_id"]),int(r["from_version"]),int(r["to_version"]),str(r["from_codec"]),str(r["target_codec"]),str(r["state"]),
        int(r["started_at_ns"]),int(r["updated_at_ns"]),int(r["completed_at_ns"]) if r["completed_at_ns"] is not None else None,
        int(r["rewrite_bytes"]) if r["rewrite_bytes"] is not None else None,int(r["migration_time_ns"]) if r["migration_time_ns"] is not None else None) for r in rows])

def run_concurrency_cell(output_dir:Path,metrics:Mapping[str,Any],thread_count:int,mode:str,repeat_index:int,base:Any,
                         truth_path:Path,telemetry:sqlite3.Connection,memory:MemoryTracker)->Dict[str,Any]:
    name=str(metrics["scale_name"]);cell_id=f"{name}__threads_{thread_count}__{mode}__repeat_{repeat_index}"
    existing=telemetry.execute("SELECT * FROM concurrency_cells WHERE cell_id=?",(cell_id,)).fetchone()
    if existing is not None:print(f"[cell resume] {cell_id}");return dict(existing)
    work=output_dir/"working_cells"/cell_id;shutil.rmtree(work,ignore_errors=True);work.mkdir(parents=True,exist_ok=True)
    copy_catalog(Path(str(metrics["base_catalog_path"])),work/"catalog.sqlite3");store=ScaleStore(work,base,True,True)
    truth=np.memmap(truth_path,mode="r",dtype=np.float32,shape=(len(store.templates),RAW_CHUNK_VALUES))
    start_event=threading.Event();migration_results=[];migration_errors=[];migration_lock=threading.Lock()
    def migration_worker()->None:
        rng=random.Random(stable_seed(SEED,"background-migration",name,thread_count,repeat_index));start_event.wait()
        selected=rng.sample(store.chunk_ids,min(BACKGROUND_MIGRATIONS_PER_CELL,len(store.chunk_ids)))
        for cid in selected:
            try:
                e=store.snapshot[cid];choices=[c for c in CODECS if c!=e.active_codec];target=choices[rng.randrange(len(choices))]
                result=store.migrate(cid,target);decoded=store._decode_entry(store.snapshot[cid])
                template=store.templates[store.snapshot[cid].template_index];expected=np.asarray(truth[template.truth_index],dtype=np.float32)
                if not np.array_equal(decoded,expected):raise AssertionError("Background migration truth mismatch")
                with migration_lock:migration_results.append(result)
            except Exception:
                with migration_lock:migration_errors.append(traceback.format_exc())
    counts=[QUERIES_PER_CELL//thread_count for _ in range(thread_count)]
    for i in range(QUERIES_PER_CELL%thread_count):counts[i]+=1
    def query_worker(worker:int,count:int)->Tuple[np.ndarray,np.ndarray,np.ndarray,int]:
        rng=random.Random(stable_seed(SEED,"query-workload",name,thread_count,repeat_index,worker));lat=np.empty(count,np.int64);wait=np.empty(count,np.int64);exe=np.empty(count,np.int64);correct=0
        start_event.wait()
        for local in range(count):
            cid=store.chunk_ids[rng.randrange(len(store.chunk_ids))];entry=store.snapshot[cid];template=store.templates[entry.template_index]
            query=query_from_rng(base,template,rng,f"B3:{cell_id}:{worker}:{local}");expected=base.execute_on_array(np.asarray(truth[template.truth_index],dtype=np.float32),query)
            actual,total_ns,wait_ns,exec_ns,_,_=store.execute_query(cid,query)
            if not base.results_equal(actual,expected):raise AssertionError(f"Query mismatch {cell_id}/{worker}/{local}")
            correct+=1;lat[local]=total_ns;wait[local]=wait_ns;exe[local]=exec_ns
            if local%MEMORY_SAMPLE_INTERVAL_QUERIES==0:memory.sample()
        return lat,wait,exe,correct
    migration_thread=None
    if mode=="background_migration":
        migration_thread=threading.Thread(target=migration_worker,name=f"B3-migration-{cell_id}",daemon=False);migration_thread.start()
    start=time.perf_counter_ns();futures=[]
    with ThreadPoolExecutor(max_workers=thread_count,thread_name_prefix=f"B3-query-{thread_count}") as executor:
        for w,count in enumerate(counts):futures.append(executor.submit(query_worker,w,count))
        start_event.set();worker_results=[f.result() for f in as_completed(futures)]
    # Foreground throughput ends when all queries finish; any remaining background
    # migration time is reported separately and must not be charged as query time.
    elapsed=time.perf_counter_ns()-start
    if migration_thread is not None:migration_thread.join()
    lat=np.concatenate([r[0] for r in worker_results]);wait=np.concatenate([r[1] for r in worker_results]);exe=np.concatenate([r[2] for r in worker_results]);correct=sum(r[3] for r in worker_results)
    if len(lat)!=QUERIES_PER_CELL or correct!=QUERIES_PER_CELL:raise AssertionError("Cell query count/correctness mismatch")
    if migration_errors:raise RuntimeError("Background migration failed:\n"+"\n".join(migration_errors))
    latency_path=output_dir/"latency_samples"/f"{cell_id}.npz";save_latency_file(latency_path,lat,wait,exe)
    before=sqlite_file_bytes(store.db_path);log_before=int(store.connection.execute("SELECT COUNT(*) n FROM migration_log").fetchone()["n"])
    archive,compact=store.compact_completed_logs();archive_logs(telemetry,cell_id,archive);after=sqlite_file_bytes(store.db_path);memory.sample()
    migrations=[r for r in migration_results if not r.get("no_op",False)]
    result={"cell_id":cell_id,"scale_name":name,"thread_count":thread_count,"mode":mode,"repeat_index":repeat_index,
      "query_count":QUERIES_PER_CELL,"correct_queries":correct,"wall_elapsed_ns":int(elapsed),"throughput_qps":float(QUERIES_PER_CELL/max(elapsed/1e9,1e-12)),
      "p50_latency_ns":float(np.quantile(lat,.5)),"p95_latency_ns":float(np.quantile(lat,.95)),"p99_latency_ns":float(np.quantile(lat,.99)),"maximum_latency_ns":float(np.max(lat)),
      "p50_lock_wait_ns":float(np.quantile(wait,.5)),"p95_lock_wait_ns":float(np.quantile(wait,.95)),"p99_lock_wait_ns":float(np.quantile(wait,.99)),
      "p50_execution_ns":float(np.quantile(exe,.5)),"p95_execution_ns":float(np.quantile(exe,.95)),"p99_execution_ns":float(np.quantile(exe,.99)),
      "background_migrations":len(migrations),"background_migration_failures":len(migration_errors),
      "background_rewrite_bytes":int(sum(int(r.get("rewrite_bytes",0)) for r in migrations)),
      "background_migration_elapsed_ns":int(sum(int(r.get("migration_time_ns",0)) for r in migrations)),
      "log_rows_before_compaction":log_before,"log_rows_after_compaction":int(compact["after"]),"archived_log_rows":int(compact["archived"]),
      "work_catalog_bytes_before_compaction":int(before["total_catalog_files_bytes"]),"work_catalog_bytes_after_compaction":int(after["total_catalog_files_bytes"]),
      "peak_rss_bytes":int(memory.peak_rss_bytes),"latency_file":str(latency_path),"completed_at_ns":time.time_ns()}
    telemetry.execute("""INSERT INTO concurrency_cells VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",tuple(result.values()))
    store.close();del truth;shutil.rmtree(work,ignore_errors=True)
    print(f"[cell complete] {cell_id}: {result['throughput_qps']:.1f} q/s, p99={result['p99_latency_ns']/1e6:.3f} ms")
    return result


def build_interference_tables(output_dir:Path,telemetry:sqlite3.Connection)->Tuple[pd.DataFrame,pd.DataFrame]:
    cells=pd.read_sql_query("SELECT * FROM concurrency_cells ORDER BY scale_name,thread_count,repeat_index,mode",telemetry)
    fg=cells[cells["mode"]=="foreground_only"].copy();bg=cells[cells["mode"]=="background_migration"].copy();keys=["scale_name","thread_count","repeat_index"]
    paired=fg.merge(bg,on=keys,suffixes=("_foreground","_background"),validate="one_to_one")
    paired["throughput_ratio_background_over_foreground"]=paired["throughput_qps_background"]/paired["throughput_qps_foreground"]
    for p in ("p50","p95","p99"):paired[f"{p}_latency_amplification"]=paired[f"{p}_latency_ns_background"]/paired[f"{p}_latency_ns_foreground"]
    paired["p99_lock_wait_amplification"]=paired["p99_lock_wait_ns_background"]/np.maximum(paired["p99_lock_wait_ns_foreground"],1.0)
    write_dataframe_atomic(output_dir/"concurrency_cells.csv",cells);write_dataframe_atomic(output_dir/"paired_interference_results.csv",paired)
    summary=paired.groupby(["scale_name","thread_count"],as_index=False).agg(
      repeats=("repeat_index","size"),median_foreground_qps=("throughput_qps_foreground","median"),median_background_qps=("throughput_qps_background","median"),
      median_throughput_ratio=("throughput_ratio_background_over_foreground","median"),median_p50_amplification=("p50_latency_amplification","median"),
      median_p95_amplification=("p95_latency_amplification","median"),median_p99_amplification=("p99_latency_amplification","median"),
      maximum_p99_amplification=("p99_latency_amplification","max"),median_background_migrations=("background_migrations_background","median"),
      total_background_failures=("background_migration_failures_background","sum"))
    write_dataframe_atomic(output_dir/"interference_summary.csv",summary);return cells,summary


def run_fsck_gate(output_dir:Path,metrics:Mapping[str,Any],base:Any,telemetry:sqlite3.Connection)->Dict[str,Any]:
    name=str(metrics["scale_name"]);maximum=None if name=="logical_1GiB" else 2000;fsck_type="full" if maximum is None else "sample_2000"
    existing=telemetry.execute("SELECT * FROM fsck_records WHERE scale_name=? AND fsck_type=?",(name,fsck_type)).fetchone()
    if existing is not None:return dict(existing)
    store=ScaleStore(output_dir/"base_scales"/name,base,True,True);result=store.offline_fsck(maximum)
    telemetry.execute("INSERT INTO fsck_records VALUES(?,?,?,?,?,?,?)",(name,result["fsck_type"],result["checked_chunks"],result["checksum_failures"],result["missing_file_failures"],result["elapsed_ns"],int(result["success"])))
    store.close();return {"scale_name":name,**result}


def main()->int:
    project=Path(__file__).resolve().parent;output=project/OUTPUT_DIR_NAME;output.mkdir(parents=True,exist_ok=True);b2_dir=project/B2_RESULT_DIR_NAME
    b2_script=project/B2_SCRIPT_NAME;b1_script=project/B1_SCRIPT_NAME;e0c_script=project/E0C_SCRIPT_NAME
    print("="*112);print("RePressDB B3 v1.1.0 — 1/5/10-GiB logical scale and concurrency interference");print(f"Target journal: {TARGET_JOURNAL}");print("="*112)
    for p in (b2_script,b1_script,e0c_script,b2_dir/"b2_assessment.json",b2_dir/"b2_experiment_freeze_sha256.txt"):
        if not p.exists():raise FileNotFoundError(p)
    b2_assessment=json.loads((b2_dir/"b2_assessment.json").read_text(encoding="utf-8"))
    if not bool(b2_assessment.get("acceptance_checks",{}).get("all_ten",False)):raise RuntimeError("B2 has not passed all acceptance checks")
    b2_freeze=(b2_dir/"b2_experiment_freeze_sha256.txt").read_text(encoding="utf-8").strip();base=load_module(e0c_script,"repressdb_e0c_for_b3_v110")
    telemetry_path=output/"telemetry.sqlite3";telemetry=connect_telemetry(telemetry_path);memory=MemoryTracker();memory.sample()
    protocol={"version":VERSION,"target_journal":TARGET_JOURNAL,"parent_b2_experiment_freeze_sha256":b2_freeze,
      "study":"replicated-tenant logical scale-out and foreground/background interference",
      "scientific_boundary":{"independent_real_data":"frozen B2 full-BDG2 8192 template corpus",
       "scale_method":"deterministic replicated logical tenants with immutable template sharing and copy-on-write migration",
       "reported_separately":["logical raw bytes","logical active stored bytes","shared physical template bytes","private copy-on-write bytes","production catalog bytes","telemetry bytes"],
       "not_claimed":"additional independent measurements or physically duplicated 10-GiB data"},
      "scales_gib":list(SCALE_TARGETS_GIB),"raw_chunk_values":RAW_CHUNK_VALUES,"raw_chunk_bytes":RAW_CHUNK_BYTES,
      "concurrency":{"query_threads":list(QUERY_THREADS),"modes":list(MODES),"repeats":REPEATS,"queries_per_cell":QUERIES_PER_CELL,
       "background_migrations_per_cell":BACKGROUND_MIGRATIONS_PER_CELL,"total_cells":len(SCALE_TARGETS_GIB)*len(QUERY_THREADS)*len(MODES)*REPEATS,
       "total_queries":len(SCALE_TARGETS_GIB)*len(QUERY_THREADS)*len(MODES)*REPEATS*QUERIES_PER_CELL,
       "total_background_migrations":len(SCALE_TARGETS_GIB)*len(QUERY_THREADS)*REPEATS*BACKGROUND_MIGRATIONS_PER_CELL},
      "startup":{"clean_repeats":STARTUP_REPEATS,"dirty_repeats":DIRTY_START_REPEATS,"clean_path":"catalog load only; no recursive file scan",
       "dirty_path":"only migration_state != idle and incomplete logs","offline_fsck":"explicit, outside startup"},
      "migration_log":{"completed_rows_retained":COMPLETED_LOG_RETAIN,"older_completed_rows":"archived to telemetry"}}
    write_json_atomic(output/"b3_protocol.json",protocol);digest=hashlib.sha256();digest.update(b2_freeze.encode("ascii"));digest.update(sha256_file(output/"b3_protocol.json").encode("ascii"));protocol_hash=digest.hexdigest()
    (output/"b3_protocol_sha256.txt").write_text(protocol_hash+"\n",encoding="utf-8")
    freeze_material={"b3_protocol_sha256":protocol_hash,"script_sha256":sha256_file(Path(__file__).resolve()),"b2_script_sha256":sha256_file(b2_script),
                     "b1_script_sha256":sha256_file(b1_script),"e0c_script_sha256":sha256_file(e0c_script)}
    write_json_atomic(output/"b3_experiment_freeze_material.json",freeze_material);freeze_hash=sha256_file(output/"b3_experiment_freeze_material.json")
    (output/"b3_experiment_freeze_sha256.txt").write_text(freeze_hash+"\n",encoding="utf-8")
    existing=telemetry_get(telemetry,"experiment_freeze_sha256")
    if existing is not None and existing!=freeze_hash:telemetry.close();raise RuntimeError("Existing B3 belongs to another freeze")
    telemetry_set(telemetry,"experiment_freeze_sha256",freeze_hash);telemetry_set(telemetry,"b3_protocol_sha256",protocol_hash)
    start_total=time.perf_counter_ns();registry,truth_path=build_template_registry(output,b2_dir);scale_results={};startup_results={};fsck_results={}
    for target in SCALE_TARGETS_GIB:
        metrics=build_scale_catalog(output,registry,target,freeze_hash,telemetry);name=str(metrics["scale_name"]);scale_results[name]=metrics
        startup_results[name]=benchmark_startup(output,metrics,base,telemetry,memory);fsck_results[name]=run_fsck_gate(output,metrics,base,telemetry)
        for threads in QUERY_THREADS:
            for repeat in range(REPEATS):
                for mode in MODES:run_concurrency_cell(output,metrics,threads,mode,repeat,base,truth_path,telemetry,memory)
    cells,interference=build_interference_tables(output,telemetry)
    startup_frame=pd.read_sql_query("SELECT * FROM startup_records ORDER BY scale_name,startup_type,repeat_index",telemetry);write_dataframe_atomic(output/"startup_records.csv",startup_frame)
    fsck_frame=pd.read_sql_query("SELECT * FROM fsck_records ORDER BY scale_name,fsck_type",telemetry);write_dataframe_atomic(output/"fsck_records.csv",fsck_frame)
    scale_frame=pd.read_sql_query("SELECT * FROM scale_build_records ORDER BY target_gib",telemetry);write_dataframe_atomic(output/"scale_build_summary.csv",scale_frame)
    expected_cells=len(SCALE_TARGETS_GIB)*len(QUERY_THREADS)*len(MODES)*REPEATS;expected_queries=expected_cells*QUERIES_PER_CELL
    expected_migrations=len(SCALE_TARGETS_GIB)*len(QUERY_THREADS)*REPEATS*BACKGROUND_MIGRATIONS_PER_CELL
    actual_cells=len(cells);actual_queries=int(cells["query_count"].sum());correct_queries=int(cells["correct_queries"].sum())
    actual_migrations=int(cells["background_migrations"].sum());migration_failures=int(cells["background_migration_failures"].sum())
    checks={
      "B3_1_three_logical_scales_built":len(scale_results)==3 and all(int(scale_results[scale_name(t)]["logical_raw_bytes"])>=t*1024**3 for t in SCALE_TARGETS_GIB),
      "B3_2_scale_method_explicitly_reports_logical_and_physical_bytes":all("logical_active_stored_bytes" in r and "shared_template_bytes" in r and "catalog_database_bytes" in r for r in scale_results.values()),
      "B3_3_clean_start_never_scans_chunk_directories":all(not bool(startup_results[n]["clean"]["any_full_directory_scan"]) for n in startup_results),
      "B3_4_dirty_start_uses_targeted_recovery_and_passes":all(bool(startup_results[n]["dirty_targeted"]["all_success"]) and int(startup_results[n]["dirty_targeted"]["maximum_inspected_pending_chunks"])<=1 and not bool(startup_results[n]["dirty_targeted"]["any_full_directory_scan"]) for n in startup_results),
      "B3_5_all_54_concurrency_cells_complete":actual_cells==expected_cells,
      "B3_6_all_2700000_queries_correct":actual_queries==expected_queries and correct_queries==expected_queries,
      "B3_7_all_1350_background_migrations_complete":actual_migrations==expected_migrations and migration_failures==0,
      "B3_8_completed_migration_log_is_bounded":bool((cells["log_rows_after_compaction"]<=COMPLETED_LOG_RETAIN).all()),
      "B3_9_thread_and_interference_metrics_reported":len(interference)==len(SCALE_TARGETS_GIB)*len(QUERY_THREADS) and all(c in interference.columns for c in ("median_foreground_qps","median_background_qps","median_p99_amplification")),
      "B3_10_startup_catalog_memory_and_fsck_metrics_reported":len(startup_frame)==len(SCALE_TARGETS_GIB)*(STARTUP_REPEATS+DIRTY_START_REPEATS) and len(fsck_frame)==len(SCALE_TARGETS_GIB) and bool(fsck_frame["success"].astype(bool).all()) and memory.peak_rss_bytes>0}
    checks["all_ten"]=bool(all(checks.values()));elapsed=time.perf_counter_ns()-start_total;telemetry_sizes=sqlite_file_bytes(telemetry_path)
    assessment={"version":VERSION,"target_journal":TARGET_JOURNAL,"status":"B3 complete" if checks["all_ten"] else "B3 completed with failed acceptance checks",
      "parent_b2_experiment_freeze_sha256":b2_freeze,"b3_protocol_sha256":protocol_hash,"b3_experiment_freeze_sha256":freeze_hash,
      "scale_results":scale_results,"startup_results":startup_results,"fsck_results":fsck_results,
      "concurrency":{"expected_cells":expected_cells,"completed_cells":actual_cells,"expected_queries":expected_queries,"correct_queries":correct_queries,
       "expected_background_migrations":expected_migrations,"completed_background_migrations":actual_migrations,"background_migration_failures":migration_failures,
       "interference_summary":interference.to_dict("records")},
      "memory":{"method":memory.method,"peak_rss_bytes":int(memory.peak_rss_bytes),"final_rss_bytes":int(memory.sample())},
      "telemetry_storage":telemetry_sizes,"total_elapsed_ns_this_invocation":int(elapsed),"acceptance_checks":checks,
      "interpretation_boundary":"Passing B3 establishes catalog/concurrency scaling over deterministic replicated logical tenants backed by frozen real BDG2 templates. Replicas are not independent real data, and immutable template payloads are physically shared.",
      "next_gate":"Proceed to final paper figures, tables, ablation consolidation, and manuscript writing." if checks["all_ten"] else "Stop and diagnose failed B3 checks before scale/concurrency claims."}
    write_json_atomic(output/"b3_assessment.json",assessment);write_json_atomic(project/SUMMARY_JSON_NAME,assessment)
    run_config={"version":VERSION,"target_journal":TARGET_JOURNAL,"b3_protocol_sha256":protocol_hash,"b3_experiment_freeze_sha256":freeze_hash,
      "parent_b2_experiment_freeze_sha256":b2_freeze,"scales_gib":list(SCALE_TARGETS_GIB),"query_threads":list(QUERY_THREADS),"modes":list(MODES),
      "repeats":REPEATS,"queries_per_cell":QUERIES_PER_CELL,"background_migrations_per_cell":BACKGROUND_MIGRATIONS_PER_CELL,
      "completed_log_retain":COMPLETED_LOG_RETAIN,"maximum_handles_per_query_thread":MAX_HANDLES_PER_QUERY_THREAD,"lock_stripes":LOCK_STRIPES,
      "environment":{"python":sys.version,"platform":platform.platform(),"processor":platform.processor(),"cpu_count":os.cpu_count(),"numpy":np.__version__,
       "pandas":pd.__version__,"sqlite_version":sqlite3.sqlite_version,"memory_method":memory.method,"timer":"time.perf_counter_ns"}}
    write_json_atomic(output/"run_config.json",run_config);checkpoint_database(telemetry);telemetry.close()
    print("="*112);print("RePressDB B3 v1.1.0 finished");print(json.dumps(checks,ensure_ascii=False,indent=2));print(f"Elapsed: {elapsed/1e9/60:.2f} minutes");print(f"Results: {output.resolve()}")
    print("Upload: b3_assessment.json, scale_build_summary.csv, startup_records.csv, concurrency_cells.csv, paired_interference_results.csv, interference_summary.csv, fsck_records.csv, template_registry.csv, b3_protocol.json, hashes, run_config.json")
    print("="*112);return 0 if checks["all_ten"] else 2


if __name__=="__main__":raise SystemExit(main())
