from typing import List, Tuple


class NoSolution(RuntimeError):
    def __init__(self):
        pass

    def __repr__(self):
        return f"NoSolution[]"


class NoFeature(NoSolution):
    def __init__(self, ext_edges: List[Tuple[int, Tuple[int, int]]]):
        self._ext_edges: List[Tuple[int, Tuple[int, int]]] = ext_edges

    def __repr__(self):
        return f"NoFeature[ext_edges={ext_edges}]"


class MaxRestarts(NoSolution):
    def __init__(self, max_restarts: int, cost_bound: int):
        self._max_restarts: int = max_restarts
        self._cost_bound: int = cost_bound

    def __repr__(self):
        return f"MaxRestarts[max_restarts={self._max_restarts}, cost_bound={self._cost_bound}]"

