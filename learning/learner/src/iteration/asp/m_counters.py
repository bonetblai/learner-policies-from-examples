import logging
import numpy as np

# Bitset-based unordered sets of unsigned integers
from intbitset import intbitset

from termcolor import colored
from typing import Set, Tuple, List, Union, Dict, Any, Optional, Union
from collections import defaultdict

from ..statistics import Statistics
from ...util import Timer


class MCounters:
    def __init__(self, ds: Dict[str, Any], monotone_only_by_dec: bool = False, timers: Optional[Statistics] = None):
        # Get relevant data structures
        self._relevant_features_idxs: intbitset = ds.get("relevant_features_idxs")
        self._numerical_features_idxs: intbitset = ds.get("numerical_features_idxs")
        """
        self._ext_states_by_bvalue_on_feature: Dict[Tuple[int, int], Set[Tuple[int, int]]] = ds.get("ext_states_by_bvalue_on_feature")
        self._ext_states_by_change_on_feature: Dict[Tuple[int, str], Set[Tuple[int, int]]] = ds.get("ext_states_by_change_on_feature")
        """
        self._features_by_bvalue_on_ext_state: Dict[Tuple[Tuple[int, int], int], intbitset] = ds.get("features_by_bvalue_on_ext_state")
        self._features_by_change_on_ext_state: Dict[Tuple[Tuple[int, int], str], intbitset] = ds.get("features_by_change_on_ext_state")
        """
        self._features_decreased_by_some_rule: intbitset = ds.get("features_decreased_by_some_rule")
        self._features_increased_by_some_rule: intbitset = ds.get("features_increased_by_some_rule")
        self._features_not_increased_by_any_rule: intbitset = ds.get("features_not_increased_by_any_rule")
        self._features_not_decreased_by_any_rule: intbitset = ds.get("features_not_decreased_by_any_rule")
        """
        self._ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = ds.get("ext_state_to_feature_valuations")
        self._timers = timers if timers is not None else Statistics()
        self._monotone_only_by_dec: bool = monotone_only_by_dec

        # Ext state to rule indices
        self._ext_state_to_rule_index: Dict[Tuple[int, int], int] = dict()
        self._rule_index_to_ext_state: List[Tuple[int, int]] = []

        # Counters
        self._change_to_f_idx_to_r_idxs: Dict[str, Dict[int, intbitset]] = None
        self._change_to_r_idx_to_f_idxs: Dict[str, Dict[int, intbitset]] = None

        # Dynamic counters and data structures to support solver
        self._dynamic_log: List[Any] = None
        self._dynamic_r_idxs: intbitset = None
        self._dynamic_change_to_f_idx_to_r_idxs: Dict[str, Dict[int, intbitset]] = None
        self._dynamic_change_to_r_idx_to_f_idxs: Dict[str, Dict[int, intbitset]] = None

        self._initialize()
        self._calculate_counters()

    def _check_invariant(self):
        if self._dynamic_change_to_f_idx_to_r_idxs is not None and self._dynamic_change_to_r_idx_to_f_idxs is not None:
            for change, f_counters in self._dynamic_change_to_f_idx_to_r_idxs.items():
                if change in ["dec", "inc"]:
                    for f_idx, r_idxs in f_counters.items():
                        for r_idx in r_idxs:
                            assert f_idx in self._dynamic_change_to_r_idx_to_f_idxs.get(change).get(r_idx)
            for change, r_counters in self._dynamic_change_to_r_idx_to_f_idxs.items():
                if change in ["dec", "inc"]:
                    for r_idx, f_idxs in r_counters.items():
                        for f_idx in f_idxs:
                            assert r_idx in self._dynamic_change_to_f_idx_to_r_idxs.get(change).get(f_idx)

    def _initialize(self):
        self._change_to_f_idx_to_r_idxs: Dict[str, Dict[int, intbitset]] = {"dec": defaultdict(intbitset), "inc": defaultdict(intbitset), "eqv": defaultdict(intbitset)}
        self._change_to_r_idx_to_f_idxs: Dict[str, Dict[int, intbitset]] = {"dec": defaultdict(intbitset), "inc": defaultdict(intbitset), "eqv": defaultdict(intbitset)}

    def get_r_idx(self, ext_state: Tuple[int, int], add_rule: bool = False) -> int:
        r_idx: int = self._ext_state_to_rule_index.get(ext_state)
        if r_idx is None and add_rule:
            r_idx = len(self._ext_state_to_rule_index)
            self._ext_state_to_rule_index[ext_state] = r_idx
            self._rule_index_to_ext_state.append(tuple(ext_state))
        return r_idx

    def get_ext_state(self, r_idx: int) -> Tuple[int, int]:
        return self._rule_index_to_ext_state[r_idx]

    # Each feature f has two sets: rules that may increase f, and rules that may decrease f
    def _calculate_counters(self):
        local_timer: Timer = Timer()
        features: intbitset = intbitset()
        for (ext_state, change), f_idxs in self._features_by_change_on_ext_state.items():
            r_idx: int = self.get_r_idx(ext_state, add_rule=True)
            assert change in ["dec", "inc", "eqv"], f"Uknown change '{change}'"
            for f_idx in f_idxs:
                self._change_to_f_idx_to_r_idxs[change][f_idx].add(r_idx)
                self._change_to_r_idx_to_f_idxs[change][r_idx].add(f_idx)
            features |= f_idxs
        logging.info(f"MCounters: counters for {len(features)} feature(s) calculated in {local_timer.get_elapsed_sec():0.2f} second(s)")

    def _execute(self, **kwargs):
        operation: str = kwargs.get("operation")
        if operation in ["add", "remove"]:
            #logging.info(f"[MCounters]: execute: {kwargs}")
            f_idx: int = kwargs.get("f_idx")
            change: int = kwargs.get("change")
            r_idx: int = kwargs.get("r_idx")
            if operation == "add":
                assert r_idx not in self._dynamic_change_to_f_idx_to_r_idxs.get(change).get(f_idx)
                self._dynamic_change_to_f_idx_to_r_idxs.get(change).get(f_idx).add(r_idx)
                assert f_idx not in self._dynamic_change_to_r_idx_to_f_idxs.get(change).get(r_idx)
                self._dynamic_change_to_r_idx_to_f_idxs.get(change).get(r_idx).add(f_idx)
            else:
                assert r_idx in self._dynamic_change_to_f_idx_to_r_idxs.get(change).get(f_idx)
                self._dynamic_change_to_f_idx_to_r_idxs.get(change).get(f_idx).remove(r_idx)
                assert f_idx in self._dynamic_change_to_r_idx_to_f_idxs.get(change).get(r_idx)
                self._dynamic_change_to_r_idx_to_f_idxs.get(change).get(r_idx).remove(f_idx)
        elif operation in ["add_rule", "remove_rule"]:
            #logging.info(f"[MCounters]: execute: {kwargs}")
            assert self._dynamic_r_idxs is not None
            r_idx: int = kwargs.get("r_idx")
            if operation == "add_rule":
                assert r_idx not in self._dynamic_r_idxs, f"r_idx={r_idx}"
                self._dynamic_r_idxs.add(r_idx)
            else:
                assert r_idx in self._dynamic_r_idxs, f"r_idx={r_idx}"
                self._dynamic_r_idxs.remove(r_idx)
        else:
            raise RuntimeError(f"Unexpected operation '{operation}'")
        if kwargs.get("log", False): self._dynamic_log.append(dict(kwargs))

    def reset(self):
        self._dynamic_log: List[Any] = []
        self._dynamic_r_idxs: intbitset = intbitset(range(len(self._rule_index_to_ext_state)))

        self._dynamic_change_to_f_idx_to_r_idxs: Dict[str, Dict[int, intbitset]] = dict()
        for change, f_counters in self._change_to_f_idx_to_r_idxs.items():
            if change in ["inc", "dec"]:
                new_f_counters: Dict[int, intbitset] = {f_idx: intbitset(r_idxs) for f_idx, r_idxs in f_counters.items()}
                self._dynamic_change_to_f_idx_to_r_idxs[change] = new_f_counters

        self._dynamic_change_to_r_idx_to_f_idxs: Dict[str, Dict[int, intbitset]] = dict()
        for change, r_counters in self._change_to_r_idx_to_f_idxs.items():
            if change in ["inc", "dec"]:
                new_r_counters: Dict[int, intbitset] = {r_idx: intbitset(f_idxs) for r_idx, f_idxs in r_counters.items()}
                self._dynamic_change_to_r_idx_to_f_idxs[change] = new_r_counters
        #self._check_invariant()

    def remove(self, f_idx: int, change: str, r_idx: int, log: bool = True):
        self._execute(operation="remove", f_idx=f_idx, change=change, r_idx=r_idx, log=log)

    def add(self, f_idx: int, change: str, r_idx: int, log: bool = True):
        self._execute(operation="add", f_idx=f_idx, change=change, r_idx=r_idx, log=log)

    def remove_rule(self, r_idx: int, log: bool = True):
        # Removing rule implies decreasing counters
        ext_state: Tuple[int, int] = self._rule_index_to_ext_state[r_idx]
        for change in ["dec", "inc"]:
            for f_idx in self._features_by_change_on_ext_state.get((ext_state, change), []):
                self.remove(f_idx, change, r_idx, log=False)
        self._execute(operation="remove_rule", r_idx=r_idx, log=log)
        #self._check_invariant()

    def add_rule(self, r_idx: int, log: bool = True):
        # Adding rule implies increasing counters
        ext_state: Tuple[int, int] = self._rule_index_to_ext_state[r_idx]
        for change in ["dec", "inc"]:
            for f_idx in self._features_by_change_on_ext_state.get((ext_state, change), []):
                self.add(f_idx, change, r_idx, log=False)
        self._execute(operation="add_rule", r_idx=r_idx, log=log)
        #self._check_invariant()

    def undo(self, n: int = 1):
        for _ in range(n):
            opposite: Dict[str, str] = {"add": "remove", "add_rule": "remove_rule", "remove": "add", "remove_rule": "add_rule"}
            kwargs = self._dynamic_log.pop()
            kwargs["operation"] = opposite[kwargs["operation"]]
            kwargs["log"] = False
            self._execute(**kwargs)

    def is_empty(self, f_idx, change) -> bool:
        return len(self._dynamic_change_to_f_idx_to_r_idxs.get(change, dict()).get(f_idx, [])) == 0

    def is_monotone(self, f_idx: int, monotone_only_by_dec: bool = False) -> bool:
        # f_idx is monotone iff:
        if self.is_empty(f_idx, "inc"):
            # 1) no rule may increase f_idx, OR
            return True
        elif self.is_empty(f_idx, "dec"):
            # 2) no rule may decrease it AND (NOT "monotone_only_by_dec" OR f_idx is Boolean)
            return not monotone_only_by_dec or f_idx not in self._numerical_features_idxs

    def r_idxs(self) -> intbitset:
        return self._dynamic_r_idxs

    def monotone_features(self, monotone_only_by_dec: bool = False) -> intbitset:
        return intbitset([f_idx for f_idx in self._relevant_features_idxs if self.is_monotone(f_idx, monotone_only_by_dec=monotone_only_by_dec)])

    def f_idxs_changed(self, r_idx: int) -> intbitset:
        return self._dynamic_change_to_r_idx_to_f_idxs.get("dec", dict()).get(r_idx, intbitset()) | self._dynamic_change_to_r_idx_to_f_idxs.get("inc", dict()).get(r_idx, intbitset())

    def project_condition(self, r_idx: int, f_idxs: Union[intbitset, int], single: bool = False) -> Tuple[Tuple[int, int]]:
        ext_state: Tuple[int, int] = self._rule_index_to_ext_state[r_idx]
        feature_values: np.ndarray = self._ext_state_to_feature_valuations.get(ext_state)
        if single:
            f_idx: int = f_idxs
            projection: Tuple[Tuple[int, int]] = ((f_idx, 1 if feature_values[f_idx] > 0 else 0),)
        else:
            projection: Tuple[Tuple[int, int]] = tuple([(f_idx, 1 if feature_values[f_idx] > 0 else 0) for f_idx in sorted(f_idxs)])
        return projection

