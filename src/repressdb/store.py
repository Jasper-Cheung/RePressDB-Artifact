from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib, json, os, sqlite3, threading, uuid
import numpy as np
from .representations import encode, decode, metadata_bytes, metadata_from_bytes
from .queries import Query

@dataclass
class ActiveView:
    chunk_id: str
    version: int
    representation: str
    payload_path: Path
    metadata_path: Path
    values: np.ndarray

class SingleActiveStore:
    def __init__(self, root):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True); (self.root/'payloads').mkdir(exist_ok=True)
        self.db=sqlite3.connect(self.root/'catalog.sqlite3')
        self.db.row_factory=sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL'); self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('''CREATE TABLE IF NOT EXISTS chunks(
          chunk_id TEXT PRIMARY KEY, active_version INTEGER NOT NULL, active_representation TEXT NOT NULL,
          active_payload TEXT NOT NULL, active_metadata TEXT NOT NULL, active_checksum TEXT NOT NULL,
          previous_version INTEGER, previous_representation TEXT, previous_payload TEXT, previous_metadata TEXT, previous_checksum TEXT,
          state TEXT NOT NULL DEFAULT 'idle', pending_target TEXT, migration_id TEXT)''')
        self.db.commit(); self._cache={}; self._lock=threading.RLock()

    @staticmethod
    def _sha(data: bytes): return hashlib.sha256(data).hexdigest()
    def _write_fsync(self,path: Path,data: bytes):
        with open(path,'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    def _paths(self, chunk_id, version, rep, suffix=''):
        safe=str(chunk_id).replace('/','_').replace('\\','_'); base=f'{safe}.v{version}.{rep}{suffix}'
        return self.root/'payloads'/f'{base}.bin', self.root/'payloads'/f'{base}.json'
    def _invalidate(self,chunk_id): self._cache.pop(str(chunk_id),None)

    def create_chunk(self, chunk_id: str, values, representation='lz4_whole'):
        with self._lock:
            if self.db.execute('SELECT 1 FROM chunks WHERE chunk_id=?',(str(chunk_id),)).fetchone(): raise ValueError('chunk exists')
            enc=encode(values,representation); meta=metadata_bytes(enc.metadata); pp,mp=self._paths(chunk_id,1,representation)
            self._write_fsync(pp,enc.payload); self._write_fsync(mp,meta); checksum=self._sha(np.asarray(values,dtype='<f4').tobytes())
            self.db.execute('INSERT INTO chunks(chunk_id,active_version,active_representation,active_payload,active_metadata,active_checksum,state) VALUES(?,?,?,?,?,?,?)',(str(chunk_id),1,representation,str(pp),str(mp),checksum,'idle')); self.db.commit(); self._invalidate(chunk_id)

    def _row(self,chunk_id):
        r=self.db.execute('SELECT * FROM chunks WHERE chunk_id=?',(str(chunk_id),)).fetchone()
        if r is None: raise KeyError(chunk_id)
        return r
    def active_representation(self,chunk_id): return self._row(chunk_id)['active_representation']
    def _load_view(self,chunk_id):
        r=self._row(chunk_id); key=(str(chunk_id),int(r['active_version']))
        v=self._cache.get(str(chunk_id))
        if v is not None and (v.chunk_id,v.version)==key: return v
        payload=Path(r['active_payload']).read_bytes(); meta=metadata_from_bytes(Path(r['active_metadata']).read_bytes()); values=decode(payload,meta)
        if self._sha(values.astype('<f4',copy=False).tobytes()) != r['active_checksum']: raise IOError('active checksum mismatch')
        v=ActiveView(str(chunk_id),int(r['active_version']),r['active_representation'],Path(r['active_payload']),Path(r['active_metadata']),values); self._cache[str(chunk_id)]=v; return v
    def query(self,chunk_id,query: Query): return query.execute(self._load_view(chunk_id).values)

    def migrate(self,chunk_id,target_representation):
        with self._lock:
            r=self._row(chunk_id)
            if r['state']!='idle': raise RuntimeError(f"chunk state is {r['state']}")
            if r['active_representation']==target_representation: return
            mid=uuid.uuid4().hex; newver=int(r['active_version'])+1
            self.db.execute('UPDATE chunks SET state=?, pending_target=?, migration_id=? WHERE chunk_id=?',('intent',target_representation,mid,str(chunk_id))); self.db.commit()
            values=self._load_view(chunk_id).values.copy(); enc=encode(values,target_representation); meta=metadata_bytes(enc.metadata)
            stage_pp,stage_mp=self._paths(chunk_id,newver,target_representation,'.staging'); final_pp,final_mp=self._paths(chunk_id,newver,target_representation)
            self._write_fsync(stage_pp,enc.payload); self._write_fsync(stage_mp,meta); self.db.execute('UPDATE chunks SET state=? WHERE chunk_id=?',('encoded',str(chunk_id))); self.db.commit()
            check=decode(stage_pp.read_bytes(),metadata_from_bytes(stage_mp.read_bytes()))
            if not np.array_equal(check,values): raise IOError('target validation failed')
            self.db.execute('UPDATE chunks SET state=? WHERE chunk_id=?',('validated',str(chunk_id))); self.db.commit()
            os.replace(stage_pp,final_pp); os.replace(stage_mp,final_mp); self.db.execute('UPDATE chunks SET state=? WHERE chunk_id=?',('published',str(chunk_id))); self.db.commit()
            checksum=self._sha(values.astype('<f4',copy=False).tobytes())
            self.db.execute('''UPDATE chunks SET previous_version=active_version,previous_representation=active_representation,previous_payload=active_payload,previous_metadata=active_metadata,previous_checksum=active_checksum,
              active_version=?,active_representation=?,active_payload=?,active_metadata=?,active_checksum=?,state='committed_pending_cleanup' WHERE chunk_id=?''',(newver,target_representation,str(final_pp),str(final_mp),checksum,str(chunk_id))); self.db.commit(); self._invalidate(chunk_id)
            # validate committed active version before cleanup
            self._load_view(chunk_id)
            r=self._row(chunk_id)
            for k in ['previous_payload','previous_metadata']:
                p=r[k]
                if p:
                    try: Path(p).unlink()
                    except FileNotFoundError: pass
            self.db.execute('''UPDATE chunks SET previous_version=NULL,previous_representation=NULL,previous_payload=NULL,previous_metadata=NULL,previous_checksum=NULL,state='idle',pending_target=NULL,migration_id=NULL WHERE chunk_id=?''',(str(chunk_id),)); self.db.commit(); self._invalidate(chunk_id)

    def recover(self):
        with self._lock:
            rows=self.db.execute("SELECT * FROM chunks WHERE state!='idle'").fetchall()
            for r in rows:
                cid=r['chunk_id']; state=r['state']
                if state in {'intent','encoded','validated','published'}:
                    # old active version is still catalog-visible; remove staging/final candidate files not referenced by active fields
                    for p in (self.root/'payloads').glob(f"{cid.replace('/','_')}.v*.staging.*"):
                        try: p.unlink()
                        except FileNotFoundError: pass
                    self.db.execute("UPDATE chunks SET state='idle',pending_target=NULL,migration_id=NULL WHERE chunk_id=?",(cid,)); self.db.commit(); self._invalidate(cid); continue
                if state=='committed_pending_cleanup':
                    try:
                        self._invalidate(cid); self._load_view(cid)
                        # active is valid; cleanup retained previous version
                        for k in ['previous_payload','previous_metadata']:
                            p=r[k]
                            if p:
                                try: Path(p).unlink()
                                except FileNotFoundError: pass
                        self.db.execute("UPDATE chunks SET previous_version=NULL,previous_representation=NULL,previous_payload=NULL,previous_metadata=NULL,previous_checksum=NULL,state='idle',pending_target=NULL,migration_id=NULL WHERE chunk_id=?",(cid,)); self.db.commit(); self._invalidate(cid)
                    except Exception:
                        if r['previous_version'] is None: raise
                        self.db.execute('''UPDATE chunks SET active_version=previous_version,active_representation=previous_representation,active_payload=previous_payload,active_metadata=previous_metadata,active_checksum=previous_checksum,
                          previous_version=NULL,previous_representation=NULL,previous_payload=NULL,previous_metadata=NULL,previous_checksum=NULL,state='idle',pending_target=NULL,migration_id=NULL WHERE chunk_id=?''',(cid,)); self.db.commit(); self._invalidate(cid); self._load_view(cid)
                else: raise RuntimeError(f'unknown recovery state {state}')
    def close(self): self._cache.clear(); self.db.close()
    def __enter__(self): return self
    def __exit__(self,*exc): self.close()
