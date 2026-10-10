from repressdb.controller import RePressController, ControllerConfig

def test_controller_admits_persistent_target():
    qcost={}; mcost={}; bytes_={}
    for q in ['point']:
        qcost[('c','a',q)]=10.0; qcost[('c','b',q)]=5.0
    mcost[('c','a','b')]=50.0; bytes_[('c','a','b')]=100
    cfg=ControllerConfig(long_window=30,short_window=10,min_observations=20,decision_interval=10,confirmations=3,cooldown_queries=0,projection_horizon=1200,minimum_relative_gain=.08,migration_safety_factor=1.25,global_budget_interval=1000)
    c=RePressController(qcost,mcost,bytes_,cfg)
    proposal=None
    for i in range(60):
        c.observe('c','point')
        p=c.consider('c','a')
        if p: proposal=p; break
    assert proposal is not None and proposal.target=='b'
