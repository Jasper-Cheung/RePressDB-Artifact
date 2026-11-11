"""Three-transaction durable replacement over the retained single-active store.

The intent transaction records every candidate pathname. Intermediate progress
is local: recovery either discards all candidate paths or validates the committed
active version. File synchronization and both validations remain in the path.
"""
import os
import time
import uuid


class EfficientMigration:
    def __init__(self, cached_store, b1):
        self.cached = cached_store
        self.s = cached_store.raw
        self.b1 = b1

    def migrate_profiled(self, chunk_id, target_codec, fault_hook=None):
        s, cached, b1 = self.s, self.cached, self.b1
        with s._lock:
            row = s._row(chunk_id)
            if row['migration_state'] != 'idle':
                raise RuntimeError('Replacement requires an idle chunk')
            if target_codec not in s.representations:
                raise ValueError(target_codec)
            source = row['active_codec']
            version = int(row['active_version'])
            if source == target_codec:
                return dict(chunk_id=chunk_id, from_codec=source,
                            target_codec=target_codec, from_version=version,
                            to_version=version, migration_time_ns=0,
                            rewrite_bytes=0, no_op=True, stage_times_ns={},
                            durable_transactions=0)
            begin = time.perf_counter_ns()
            stages = {}
            mid = uuid.uuid4().hex
            temporary = s._version_stem(chunk_id, version + 1, target_codec,
                                        temporary_token=uuid.uuid4().hex)
            final = s._version_stem(chunk_id, version + 1, target_codec)
            td, tm = temporary.with_suffix('.bin'), temporary.with_suffix('.meta.bin')
            fd, fm = final.with_suffix('.bin'), final.with_suffix('.meta.bin')
            context = dict(chunk_id=chunk_id, migration_id=mid, temp_data=td,
                           temp_meta=tm, final_data=fd, final_meta=fm)
            def hook(name):
                if fault_hook is not None:
                    fault_hook(name, context)
            def relative(path):
                return b1.relative_path(s.root, path)

            # T1: reserve the version and make all rollback paths durable.
            start = time.perf_counter_ns()
            now = time.time_ns()
            with b1.immediate_transaction(s.connection):
                s.connection.execute('''INSERT INTO migration_log
                    (migration_id,chunk_id,from_version,to_version,from_codec,
                     target_codec,state,temporary_data_path,temporary_metadata_path,
                     final_data_path,final_metadata_path,started_at_ns,updated_at_ns)
                    VALUES (?,?,?,?,?,?,'intent',?,?,?,?,?,?)''',
                    (mid,chunk_id,version,version+1,source,target_codec,
                     relative(td),relative(tm),relative(fd),relative(fm),now,now))
                s.connection.execute("UPDATE chunks SET migration_state='intent',updated_at_ns=? WHERE chunk_id=?",(now,chunk_id))
            stages['intent_ns'] = time.perf_counter_ns()-start
            hook('after_intent')

            start = time.perf_counter_ns()
            values = s._decode_stored(cached.active_stored(chunk_id))
            checksum = str(row['raw_checksum'])
            if b1.raw_array_sha256(values) != checksum:
                raise AssertionError('Source checksum mismatch')
            stages['decode_ns'] = time.perf_counter_ns()-start
            start = time.perf_counter_ns()
            encoded = s.representations[target_codec].encode_to_files(values,temporary)
            stages['encode_sync_ns'] = time.perf_counter_ns()-start
            hook('after_encode')
            start = time.perf_counter_ns()
            if not s._validate_stored(encoded,checksum,expected_values=values):
                raise AssertionError('Prepared candidate validation failed')
            stages['validate_ns'] = time.perf_counter_ns()-start
            hook('after_validate')

            start = time.perf_counter_ns()
            fd.parent.mkdir(parents=True,exist_ok=True)
            if fd.exists() or fm.exists():
                raise RuntimeError('Reserved candidate version already exists')
            os.replace(td,fd)
            b1.fsync_directory(fd.parent)
            hook('after_data_publish')
            if encoded.metadata_path is not None:
                os.replace(tm,fm)
                metadata_path = fm
            else:
                metadata_path = None
            b1.fsync_directory(fd.parent)
            stages['publish_files_ns'] = time.perf_counter_ns()-start
            hook('after_publish')

            # T2: install the validated version and retain the old version.
            start = time.perf_counter_ns()
            now = time.time_ns()
            with b1.immediate_transaction(s.connection):
                s.connection.execute('''UPDATE chunks SET
                    previous_version=active_version,previous_codec=active_codec,
                    previous_data_path=active_data_path,
                    previous_metadata_path=active_metadata_path,
                    previous_payload_bytes=active_payload_bytes,
                    previous_metadata_bytes=active_metadata_bytes,
                    active_version=?,active_codec=?,active_data_path=?,
                    active_metadata_path=?,active_payload_bytes=?,active_metadata_bytes=?,
                    migration_state='committed_pending_cleanup',updated_at_ns=?
                    WHERE chunk_id=?''',
                    (version+1,target_codec,relative(fd),relative(metadata_path),
                     int(encoded.payload_size_bytes),int(encoded.metadata_size_bytes),now,chunk_id))
                s.connection.execute('''UPDATE migration_log SET state='catalog_committed',
                    final_data_path=?,final_metadata_path=?,target_payload_bytes=?,
                    target_metadata_bytes=?,updated_at_ns=? WHERE migration_id=?''',
                    (relative(fd),relative(metadata_path),int(encoded.payload_size_bytes),
                     int(encoded.metadata_size_bytes),now,mid))
            cached.invalidate_handle(chunk_id)
            stages['active_commit_ns'] = time.perf_counter_ns()-start
            hook('after_commit')

            start = time.perf_counter_ns()
            if not s._validate_stored(cached.active_stored(chunk_id),checksum,expected_values=values):
                raise AssertionError('Committed candidate validation failed')
            stages['post_validate_ns'] = time.perf_counter_ns()-start
            hook('after_post_validate')
            start = time.perf_counter_ns()
            previous = s.previous_stored(chunk_id)
            # T3 retires the pointer BEFORE unlinking. A crash during unlink
            # leaves unreferenced files for startup sweeping, never a catalog
            # pointer whose block metadata has already disappeared.
            now = time.time_ns()
            with b1.immediate_transaction(s.connection):
                s.connection.execute('''UPDATE chunks SET previous_version=NULL,
                    previous_codec=NULL,previous_data_path=NULL,previous_metadata_path=NULL,
                    previous_payload_bytes=NULL,previous_metadata_bytes=NULL,
                    migration_state='idle',updated_at_ns=? WHERE chunk_id=?''',(now,chunk_id))
                s.connection.execute("UPDATE migration_log SET state='completed',completed_at_ns=?,updated_at_ns=? WHERE migration_id=?",(now,now,mid))
            cached.invalidate_handle(chunk_id)
            hook('after_retire')
            s._delete_stored_files(previous)
            hook('after_unlink_previous')
            stages['cleanup_ns'] = time.perf_counter_ns()-start
            hook('after_cleanup')
            return dict(chunk_id=chunk_id,migration_id=mid,from_codec=source,
                        target_codec=target_codec,from_version=version,to_version=version+1,
                        migration_time_ns=time.perf_counter_ns()-begin,
                        rewrite_bytes=int(encoded.stored_size_bytes),no_op=False,
                        stage_times_ns=stages,durable_transactions=3)
