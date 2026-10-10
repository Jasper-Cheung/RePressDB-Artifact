from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import numpy as np

class QueryClass(str, Enum):
    POINT = "point"
    RANGE_SHORT = "range_short"
    RANGE_MEDIUM = "range_medium"
    RANGE_LONG = "range_long"
    AGG_SHORT = "aggregate_short"
    AGG_MEDIUM = "aggregate_medium"
    AGG_LONG = "aggregate_long"
    THRESHOLD_COUNT = "threshold_count"

@dataclass(frozen=True)
class Query:
    kind: QueryClass
    start: int = 0
    stop: int | None = None
    threshold: float = 0.0

    def execute(self, values: np.ndarray):
        n = len(values)
        start = max(0, min(int(self.start), n - 1 if n else 0))
        stop = n if self.stop is None else max(start + 1, min(int(self.stop), n))
        if self.kind == QueryClass.POINT:
            return float(values[start])
        if self.kind in {QueryClass.RANGE_SHORT, QueryClass.RANGE_MEDIUM, QueryClass.RANGE_LONG}:
            return values[start:stop].copy()
        if self.kind in {QueryClass.AGG_SHORT, QueryClass.AGG_MEDIUM, QueryClass.AGG_LONG}:
            return float(np.sum(values[start:stop], dtype=np.float64))
        if self.kind == QueryClass.THRESHOLD_COUNT:
            return int(np.count_nonzero(values[start:stop] > self.threshold))
        raise ValueError(f"unsupported query class: {self.kind}")
