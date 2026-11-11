"""Small semantic checks for the revised admission rule."""
import numpy as np
from fgcs_benchmark import CreditController, WorkFunctionController

def run():
    migration=np.array([[0.,10000.],[12000.,0.]])
    equal=np.ones((2,8))*100.
    ctl=CreditController(0,2)
    assert all(ctl.step(0,equal,migration) is None for _ in range(2000))
    profile=np.array([[100.]*8,[50.]*8])
    ctl=CreditController(0,2)
    proposal=None
    for _ in range(400):proposal=ctl.step(0,profile,migration)
    assert proposal==1 and ctl.credit[1]>=12500.
    ctl.committed_switch(1)
    assert ctl.active_position==1 and np.array_equal(ctl.credit,np.zeros(2))
    assert all(ctl.step(0,profile,migration) is None for _ in range(1000))
    mixed=np.array([[100.]*4+[50.]*4,[50.]*4+[100.]*4])
    ctl=CreditController(0,2)
    for _ in range(100):ctl.step(0,mixed,migration)
    for _ in range(100):ctl.step(4,mixed,migration)
    assert ctl.credit[1]==0.
    w=WorkFunctionController(0,2)
    assert all(w.step(0,equal,migration) is None for _ in range(1000))
    print('PASS: no-gain stability, causal evidence, commit reset, negative evidence and work-function stability')

if __name__=='__main__':run()
