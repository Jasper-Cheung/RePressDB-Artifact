"""Behavioral checks for risk rejection, repair, and batched evidence."""
from types import SimpleNamespace
import numpy as np
from risk_controller import RiskController,feature_key

def query(point=True):
    return SimpleNamespace(query_type='point' if point else 'range_scan',start=0,end=1 if point else 100)
keys=[('point',0),('range_scan',4)]
mean=np.array([[100.,100.],[30.,500.]])
samples=np.repeat(mean[:,:,None],64,axis=2)
migration=np.array([[0.,1000.],[1000.,0.]])
ctl=RiskController(0,mean,samples,migration,keys,100)
proposal=None
for i in range(1280):
    x=ctl.observe(query())
    if x is not None:proposal=x
assert proposal==1,'Sustained safe improvement should be admitted'
ctl.committed_switch(1)
assert np.all(ctl.credit==0)
repairs=[]
for i in range(512):
    x=ctl.observe(query(False))
    if x is not None:repairs.append(x)
assert repairs and repairs[-1]==0,'Mixture change must initiate risk repair'
ctl=RiskController(0,mean,samples,migration,keys,100)
for i in range(4096):
    # Candidate has a lower mean but a worse p99 in this mixed regime.
    assert ctl.observe(query(i%256<250)) is None
assert any(not d['feasible'][1] for d in ctl.decisions)
assert ctl.credit[1]==0,'An infeasible epoch must discard its old credit'
mean2=np.array([[100.,100.],[90.,100.]])
ctl2=RiskController(0,mean2,np.repeat(mean2[:,:,None],64,axis=2),np.array([[0.,10000.],[10000.,0.]]),keys,100)
for i in range(11*256):assert ctl2.observe(query()) is None
last=None
for i in range(256):last=ctl2.observe(query())
assert last==1,'Forward plus return cost must be covered before admission'
rng=np.random.default_rng(2311)
ctl=RiskController(0,mean,samples,np.full((2,2),1e30),keys,100)
credit=np.zeros(2)
for i,q in enumerate(rng.integers(0,2,20000),1):
    credit=np.maximum(0,credit+mean[0,q]-mean[:,q])
    assert ctl.observe(query(q==0)) is None
    if i%256==0:assert np.all(ctl.credit<=credit+1e-7)
print('PASS: safe admission, mixed-tail rejection, risk repair, reset, conservative block evidence')
