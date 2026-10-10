#!/usr/bin/env python3
from pathlib import Path
import sys, tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from repressdb import SingleActiveStore, Query, QueryClass

def main():
    values=np.linspace(-2.0,3.0,4096,dtype=np.float32)
    with tempfile.TemporaryDirectory() as td:
        with SingleActiveStore(td) as store:
            store.create_chunk('demo',values,'lz4_whole')
            before=store.query('demo',Query(QueryClass.RANGE_MEDIUM,125,800))
            store.migrate('demo','lz4_block256')
            after=store.query('demo',Query(QueryClass.RANGE_MEDIUM,125,800))
            assert np.array_equal(before,after)
        with SingleActiveStore(td) as store:
            store.recover()
            point=store.query('demo',Query(QueryClass.POINT,777))
            assert point == float(values[777])
    print('smoke test: PASS')
if __name__=='__main__': main()
