"""Operation/length cost surface with empirical tail admission and block updates."""
from bisect import bisect_left
from collections import deque
import numpy as np

LENGTH_BOUNDS=(.002,.02,.2,.75,1.00001)
def feature_key(query, n):
    if query.query_type == 'point':
        return ('point',0)
    return (query.query_type,bisect_left(LENGTH_BOUNDS,(query.end-query.start)/n))

def mixture_quantile(samples,weights,quantile=.99):
    # samples has layout-feature-sample axes; all strata have equal counts.
    values=samples.ravel();mass=np.repeat(weights/samples.shape[-1],samples.shape[-1])
    order=np.argsort(values)
    return float(values[order[min(np.searchsorted(np.cumsum(mass[order]),quantile),len(order)-1)]])

class RiskController:
    """All per-request accounting is scalar; decisions occur each 100 requests.

    The empirical tail reference is the initial layout on the current mixture.
    A risk-repair proposal can return to that reference under a changed mixture.
    """
    def __init__(self, active, mean, samples, migration, feature_keys, n,
                 tail_factor=1.10, block_size=256):
        self.active_position=active;self.anchor=active
        self.mean=mean;self.samples=samples;self.migration=migration
        self.keys={k:i for i,k in enumerate(feature_keys)};self.n=n
        self.hist=[0]*len(feature_keys);self.blocks=deque(maxlen=3)
        self.count=0;self.block_size=block_size;self.credit=np.zeros(len(mean))
        self.tail_factor=tail_factor;self.confirm_target=None;self.confirm=0
        self.last_switch=-10**9;self.decisions=[];self.proposal_reason=None
        self.anchor_order=np.argsort(samples[active].ravel())
        self.anchor_values=samples[active].ravel()[self.anchor_order]

    def observe(self, query):
        k=self.keys[feature_key(query,self.n)]
        self.hist[k]+=1;self.count+=1
        if self.count%self.block_size:return None
        h=np.asarray(self.hist,dtype=float);self.hist[:]=[0]*len(self.hist)
        self.blocks.append(h)
        cost=self.mean@h
        self.credit=np.maximum(0,self.credit+cost[self.active_position]-cost)
        if self.count<3*self.block_size:return None
        long=sum(self.blocks);long/=long.sum();short=h/h.sum()
        feasible=np.ones(len(self.mean),dtype=bool);thresholds=[]
        for weights in (long,short):
            mass=np.repeat(weights/self.samples.shape[-1],self.samples.shape[-1])
            position=min(np.searchsorted(np.cumsum(mass[self.anchor_order]),.99),len(mass)-1)
            threshold=self.tail_factor*float(self.anchor_values[position])
            thresholds.append(threshold)
            exceed=((self.samples>threshold).mean(axis=2))@weights
            feasible &= exceed<=.01+1e-12
        feasible[self.anchor]=True
        lc=self.mean@long;sc=self.mean@short;current=self.active_position
        target=int(np.argmin(np.where(feasible,lc,np.inf)))
        reason='hold';proposal=None
        if not feasible[current]:
            # Risk repair uses the feasible initial layout; budget still applies.
            target=self.anchor;reason='risk_repair'
        elif target==current or target!=int(np.argmin(np.where(feasible,sc,np.inf))):
            self.confirm_target=None;self.confirm=0
            reason='stable_or_disagreement'
        elif min((lc[current]-lc[target])/lc[current],(sc[current]-sc[target])/sc[current])<.08:
            self.confirm_target=None;self.confirm=0;reason='gain'
        else:reason='candidate'
        if reason in ('candidate','risk_repair'):
            self.confirm=self.confirm+1 if target==self.confirm_target else 1
            self.confirm_target=target
            required=2 if reason=='risk_repair' else 3
            if self.confirm>=required and (reason=='risk_repair' or self.count-self.last_switch>=3*self.block_size):
                if reason=='risk_repair' or self.credit[target]>=1.25*self.migration[current,target]:
                    proposal=target
        # Decision diagnostics are inside the controller timer.
        self.decisions.append(dict(local_count=self.count,active=current,target=target,
                                   feasible=feasible.tolist(),reason=reason,
                                   credit=float(self.credit[target]),
                                   migration_cost=float(self.migration[current,target]),
                                   proposal=proposal,tail_thresholds=thresholds))
        self.proposal_reason=reason
        return proposal

    def committed_switch(self,target):
        self.active_position=target;self.credit.fill(0);self.last_switch=self.count
        self.confirm_target=None;self.confirm=0
