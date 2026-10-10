import numpy as np
from repressdb.store import SingleActiveStore
from repressdb.queries import Query,QueryClass

def test_single_active_migration(tmp_path):
    x=np.arange(2048,dtype=np.float32)
    with SingleActiveStore(tmp_path) as s:
        s.create_chunk('x',x,'lz4_whole')
        a=s.query('x',Query(QueryClass.RANGE_SHORT,20,200))
        s.migrate('x','zstd3_whole')
        b=s.query('x',Query(QueryClass.RANGE_SHORT,20,200))
        assert np.array_equal(a,b)
        assert s.active_representation('x')=='zstd3_whole'
