"""Verify optimized counter decisions against the preserved NumPy implementation."""
from pathlib import Path
import importlib.util, sys, json
import numpy as np
from fgcs_benchmark import CreditController

p=Path(__file__).with_name('fgcs_benchmark_development_v11.py')
s=importlib.util.spec_from_file_location('development_reference',p);old=importlib.util.module_from_spec(s);sys.modules[s.name]=old;s.loader.exec_module(old)
rng=np.random.default_rng(20261031);events=0
for trial in range(40):
    profile=rng.uniform(1e4,2e5,(5,8));migration=rng.uniform(1e6,1e8,(5,5));np.fill_diagonal(migration,0.)
    a=old.CreditController(0,5);b=CreditController(0,5)
    for q in rng.integers(0,8,10000):
        x=a.step(int(q),profile,migration);y=b.step(int(q),profile,migration)
        assert x==y and np.array_equal(a.credit,b.credit)
        if x is not None:a.committed_switch(x);b.committed_switch(y)
        events+=1
report={'pass':True,'trials':40,'events':events,'comparison':'exact credit values and event proposal equality; identical committed switches','seed':20261031}
Path(__file__).with_name('implementation_equivalence.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(report)
