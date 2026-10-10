from __future__ import annotations
from collections import Counter, deque
from dataclasses import dataclass
from typing import Mapping

@dataclass(frozen=True)
class ControllerConfig:
    long_window: int = 300
    short_window: int = 100
    min_observations: int = 200
    decision_interval: int = 100
    confirmations: int = 3
    cooldown_queries: int = 300
    projection_horizon: int = 1200
    minimum_relative_gain: float = 0.08
    migration_safety_factor: float = 1.25
    global_budget_interval: int = 1000

@dataclass(frozen=True)
class MigrationProposal:
    chunk_id: str
    source: str
    target: str
    projected_saving: float
    migration_cost: float
    rewrite_bytes: int

class _ChunkState:
    def __init__(self, config: ControllerConfig):
        self.events = deque(maxlen=config.long_window)
        self.local_count = 0
        self.last_decision = 0
        self.last_migration = -10**18
        self.confirm_target = None
        self.confirm_count = 0

class RePressController:
    def __init__(self, query_costs: Mapping, migration_costs: Mapping, rewrite_bytes: Mapping, config: ControllerConfig | None = None):
        self.query_costs = query_costs
        self.migration_costs = migration_costs
        self.rewrite_bytes = rewrite_bytes
        self.config = config or ControllerConfig()
        self.states = {}
        self.global_query_count = 0
        self.budget_interval_id = -1
        self.budget_used = 0
        all_bytes = [int(v) for v in rewrite_bytes.values()] if rewrite_bytes else [0]
        self.budget_limit = max(all_bytes)

    def _state(self, chunk_id):
        return self.states.setdefault(chunk_id, _ChunkState(self.config))

    def observe(self, chunk_id: str, query_class: str):
        st = self._state(chunk_id)
        st.events.append(str(query_class)); st.local_count += 1; self.global_query_count += 1

    def _weights(self, events):
        c = Counter(events); n = max(1, len(events)); return {q: v/n for q,v in c.items()}

    def _expected(self, chunk_id, representation, weights):
        return sum(weights.get(q,0.0) * float(self.query_costs[(chunk_id, representation, q)]) for q in weights)

    def consider(self, chunk_id: str, active_representation: str):
        cfg = self.config; st = self._state(chunk_id)
        if st.local_count < cfg.min_observations: return None
        if st.local_count - st.last_decision < cfg.decision_interval: return None
        st.last_decision = st.local_count
        if st.local_count - st.last_migration < cfg.cooldown_queries: return None
        events = list(st.events); sw = self._weights(events[-cfg.short_window:]); lw = self._weights(events[-cfg.long_window:])
        reps = sorted({r for (c,r,q) in self.query_costs if c == chunk_id})
        if not reps: return None
        short_target = min(reps, key=lambda r: self._expected(chunk_id,r,sw))
        long_target = min(reps, key=lambda r: self._expected(chunk_id,r,lw))
        if short_target != long_target or short_target == active_representation:
            st.confirm_target = None; st.confirm_count = 0; return None
        target = short_target
        active_cost = self._expected(chunk_id, active_representation, lw)
        target_cost = self._expected(chunk_id, target, lw)
        if active_cost <= 0: return None
        relative_gain = (active_cost-target_cost)/active_cost
        if relative_gain < cfg.minimum_relative_gain:
            st.confirm_target = None; st.confirm_count = 0; return None
        if st.confirm_target == target: st.confirm_count += 1
        else: st.confirm_target = target; st.confirm_count = 1
        if st.confirm_count < cfg.confirmations: return None
        mkey=(chunk_id,active_representation,target)
        migration_cost=float(self.migration_costs[mkey]); projected=cfg.projection_horizon*(active_cost-target_cost)
        if projected <= cfg.migration_safety_factor*migration_cost: return None
        interval = self.global_query_count // cfg.global_budget_interval
        if interval != self.budget_interval_id:
            self.budget_interval_id = interval; self.budget_used = 0
        b=int(self.rewrite_bytes[mkey])
        if self.budget_used + b > self.budget_limit: return None
        return MigrationProposal(chunk_id, active_representation, target, projected, migration_cost, b)

    def commit(self, proposal: MigrationProposal):
        st=self._state(proposal.chunk_id); st.last_migration=st.local_count; st.confirm_count=0; st.confirm_target=None; self.budget_used += proposal.rewrite_bytes
