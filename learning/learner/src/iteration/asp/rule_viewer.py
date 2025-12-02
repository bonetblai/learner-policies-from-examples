import logging
import numpy as np

# Bitset-based unordered sets of unsigned integers
from intbitset import intbitset

from termcolor import colored
from typing import Set, Tuple, List, Union, Dict, Any, Optional, Union
from collections import defaultdict

from ..feature_pool import Feature
from ..statistics import Statistics
from ...util import Timer


# Detection of monotone features for "active set" by watching rules per each feature.
# 1. For each feature, two rules are watched, one that increments and anoother that decrements it.
# 2. A feature is monotone if one of the watched rules is None.
# 3. When a rule is "removed", features that watch the rule must be updated
# 4. When a rule is "restored" (ie., by returning from recursion), features with a 'None' watched rule may be updated
# Data structures needed:
# 1. For each rule, which features are incremented/decremented
# 2. For each feature, which rules increment/decrement the feature
# 3. For each feature, two watched rules
# 4. For each rule, which features watch the rule (stalkers)
# 5. Dangling features; those that previously were watching a rule but now watch 'None' because no active rule affect them

class RuleViewer:
    def __init__(self, ds: Dict[str, Any], monotone_only_by_dec: bool = False, timers: Optional[Statistics] = None):
        self._relevant_features: List[Any] = ds.get("relevant_features")
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
        self._min_complexity: int = None

        # Ext state to rule indices
        self._r_idx_to_ext_state: List[Tuple[int, int]] = []
        self._ext_state_to_r_idx: Dict[Tuple[int, int], int] = dict()

        # f_idx to feature_idx
        self._f_idx_to_feature_idx: Dict[int, int] = None
        self._feature_idx_to_f_idx: Tuple[int] = None

        # Cross reference of rules and features
        self._feature_idx_to_r_idxs: Tuple[Tuple[intbitset, intbitset]] = None
        self._r_idx_to_affected_features: Tuple[Tuple[intbitset, intbitset]] = None

        # Cross reference of affected features by rules
        self._change_to_f_idx_to_r_idxs: Dict[str, Dict[int, intbitset]] = None
        self._change_to_r_idx_to_f_idxs: Dict[str, Dict[int, intbitset]] = None

        # Watched rules
        self._active_r_idxs: intbitset = None
        self._feature_idx_to_watched_rules: Tuple[List[int, int]] = None
        self._r_idx_to_stalkers: Tuple[Tuple[intbitset, intbitset]] = None
        self._dangling_features: Tuple[intbitset, intbitset] = None

        # Initialization
        self._initialize()

    def _initialize(self):
        local_timer: Timer = Timer()

        # Create f_idx <-> feature_idx maps
        self._f_idx_to_feature_idx: Dict[int, int] = dict()
        self._feature_idx_to_f_idx: List[int] = []
        for feature_idx, (f_idx, feature) in enumerate(self._relevant_features):
            self._f_idx_to_feature_idx[f_idx] = feature_idx
            self._feature_idx_to_f_idx.append(f_idx)
            self._min_complexity = feature.complexity if self._min_complexity is None else min(self._min_complexity, feature.complexity)
        self._feature_idx_to_f_idx: Tuple[int] = tuple(self._feature_idx_to_f_idx)
        logging.info(f"RuleViewer: min_complexity={self._min_complexity}")

        # Cross reference rules and features
        self._feature_idx_to_r_idxs: Tuple[Tuple[intbitset, intbitset]] = tuple((intbitset(), intbitset()) for _ in self._feature_idx_to_f_idx)
        self._r_idx_to_affected_features: List[Tuple[intbitset, intbitset]] = []

        self._change_to_f_idx_to_r_idxs: Dict[str, Dict[int, intbitset]] = {"dec": defaultdict(intbitset), "inc": defaultdict(intbitset), "eqv": defaultdict(intbitset)}
        self._change_to_r_idx_to_f_idxs: Dict[str, Dict[int, intbitset]] = {"dec": defaultdict(intbitset), "inc": defaultdict(intbitset), "eqv": defaultdict(intbitset)}

        for (ext_state, change), f_idxs in self._features_by_change_on_ext_state.items():
            r_idx: int = self.get_r_idx(ext_state, add_rule=True)
            assert change in ["dec", "inc", "eqv"], f"Uknown change '{change}'"
            for f_idx in f_idxs:
                self._change_to_f_idx_to_r_idxs[change][f_idx].add(r_idx)
                self._change_to_r_idx_to_f_idxs[change][r_idx].add(f_idx)
                feature_idx: int = self._f_idx_to_feature_idx.get(f_idx)
                if change != "eqv":
                    i: int = 0 if change == "dec" else 1
                    self._feature_idx_to_r_idxs[feature_idx][i].add(r_idx)
                    self._r_idx_to_affected_features[r_idx][i].add(feature_idx)
        self._r_idx_to_affected_features: Tuple[Tuple[intbitset, intbitset]] = tuple(self._r_idx_to_affected_features)

        local_timer.stop()
        logging.info(f"RuleViewer: {local_timer.get_elapsed_sec():0.2f} second(s) for initialization: ")

    def _initialize_watched_rules(self):
        local_timer: Timer = Timer()
        self._active_r_idxs: intbitset = self.all_rules()
        self._feature_idx_to_watched_rules: List[List[int, int]] = []
        self._r_idx_to_stalkers: Tuple[Tuple[intbitset, intbitset]] = tuple((intbitset(), intbitset()) for _ in self._r_idx_to_ext_state)
        self._dangling_features: Tuple[intbitset, intbitset] = (intbitset(), intbitset())

        for feature_idx, r_idxs in enumerate(self._feature_idx_to_r_idxs):
            watched_rules: List[int, int] = [next(iter(r_idxs[i])) if len(r_idxs[i]) > 0 else None for i in [0, 1]]
            self._feature_idx_to_watched_rules.append(watched_rules)
            for i, r_idx in enumerate(watched_rules):
                if r_idx is not None:
                    self._r_idx_to_stalkers[r_idx][i].add(feature_idx)

        self._feature_idx_to_watched_rules: Tuple[List[int, int]] = tuple(self._feature_idx_to_watched_rules)

        local_timer.stop()
        logging.info(f"RuleViewer: {local_timer.get_elapsed_sec():0.2f} second(s) for initialization of watched rules for {len(self._feature_idx_to_r_idxs)} feature(s)")

    def f_idx_to_feature(self, f_idx) -> Feature:
        pair: Tuple[int, Feature] = self._relevant_features[self._f_idx_to_feature_idx.get(f_idx)]
        assert f_idx == pair[0], f"{f_idx} {pair[0]} {pair[1]}"
        return pair[1]

    def all_rules(self) -> intbitset:
        return intbitset(range(len(self._r_idx_to_affected_features)))

    def ext_states_that_change_no_feature(self) -> List[Tuple[int, int]]:
        features_changed_by_ext_state: Dict[Tuple[int, int], intbitset] = defaultdict(intbitset)
        for (ext_state, change), f_idxs in self._features_by_change_on_ext_state.items():
            if change != "eqv":
                features_changed_by_ext_state[ext_state] |= f_idxs
            elif ext_state not in features_changed_by_ext_state:
                features_changed_by_ext_state[ext_state] = intbitset()
        return [ext_state for ext_state, f_idxs in features_changed_by_ext_state.items() if len(f_idxs) == 0]

    def reset(self):
        self._initialize_watched_rules()

    def get_r_idx(self, ext_state: Tuple[int, int], add_rule: bool = False) -> int:
        r_idx: int = self._ext_state_to_r_idx.get(ext_state)
        if r_idx is None and add_rule:
            r_idx = len(self._ext_state_to_r_idx)
            self._ext_state_to_r_idx[ext_state] = r_idx
            self._r_idx_to_ext_state.append(tuple(ext_state))
            self._r_idx_to_affected_features.append((intbitset(), intbitset()))
        return r_idx

    def get_ext_state(self, r_idx: int) -> Tuple[int, int]:
        return self._r_idx_to_ext_state[r_idx]

    def is_monotone(self, f_idx: int, monotone_only_by_dec: bool = None) -> bool:
        # f_idx is monotone iff no rule main increase f_idx, or no rule may decrease it AND (NOT monotone_only_by_dec OR f_idx is Boolean)
        monotone_only_by_dec: bool = monotone_only_by_dec or self._monotone_only_by_dec
        feature_idx: int = self._f_idx_to_feature_idx.get(f_idx)
        watched_rules: List[int, int] = self._feature_idx_to_watched_rules[feature_idx]
        if watched_rules[1] is None:
            return True
        elif watched_rules[0] is None:
            return not monotone_only_by_dec or f_idx not in self._numerical_features_idxs
        else:
            return False

    def monotone_features(self, monotone_only_by_dec: bool = None) -> intbitset:
        monotone_only_by_dec: bool = monotone_only_by_dec or self._monotone_only_by_dec
        return intbitset([f_idx for f_idx in self._relevant_features_idxs if self.is_monotone(f_idx, monotone_only_by_dec=monotone_only_by_dec)])

    def f_idxs_changed_by(self, r_idx: int) -> intbitset:
        return self._change_to_r_idx_to_f_idxs.get("dec", dict()).get(r_idx, intbitset()) | self._change_to_r_idx_to_f_idxs.get("inc", dict()).get(r_idx, intbitset())

    def r_idxs_that_change(self, f_idx) -> intbitset:
        #return intbitset([r_idx for r_idx in self._active_r_idxs if f_idx in self.f_idxs_changed_by(r_idx)])
        return (self._change_to_f_idx_to_r_idxs.get("dec", dict()).get(f_idx, intbitset()) | self._change_to_f_idx_to_r_idxs.get("inc", dict()).get(f_idx, intbitset())) & self._active_r_idxs

    def r_idxs(self) -> intbitset:
        return self._active_r_idxs

    def remove_rule(self, r_idx: int, log: bool = True):
        # Removing a rule implies updated watched rules for features that watch the rule
        assert r_idx in self._active_r_idxs
        self._active_r_idxs.remove(r_idx)
        for i in [0, 1]:
            stalkers: List[int] = list(self._r_idx_to_stalkers[r_idx][i])
            for feature_idx in stalkers:
                assert self._feature_idx_to_watched_rules[feature_idx][i] == r_idx
                # Look for another active rule to watch
                good_r_idxs: intbitset = self._feature_idx_to_r_idxs[feature_idx][i] & self._active_r_idxs
                new_watched_r_idx: int = next(iter(good_r_idxs)) if len(good_r_idxs) > 0 else None
                self._feature_idx_to_watched_rules[feature_idx][i] = new_watched_r_idx
                self._r_idx_to_stalkers[r_idx][i].remove(feature_idx)
                assert feature_idx not in self._dangling_features[i]
                if new_watched_r_idx is not None:
                    assert feature_idx not in self._r_idx_to_stalkers[new_watched_r_idx][i]
                    self._r_idx_to_stalkers[new_watched_r_idx][i].add(feature_idx)
                else:
                    self._dangling_features[i].add(feature_idx)

    def restore_rule(self, r_idx: int, log: bool = True):
        assert r_idx not in self._active_r_idxs
        self._active_r_idxs.add(r_idx)
        for i, affected_features in enumerate(self._r_idx_to_affected_features[r_idx]):
            for feature_idx in affected_features & self._dangling_features[i]:
                assert self._feature_idx_to_watched_rules[feature_idx][i] is None
                self._feature_idx_to_watched_rules[feature_idx][i] = r_idx
                assert feature_idx not in self._r_idx_to_stalkers[r_idx][i]
                self._r_idx_to_stalkers[r_idx][i].add(feature_idx)
                self._dangling_features[i].remove(feature_idx)

    def project_condition(self, r_idx: int, f_idxs: Union[intbitset, int], boolean_projection: bool = True) -> Tuple[Tuple[int, int]]:
        ext_state: Tuple[int, int] = self._r_idx_to_ext_state[r_idx]
        feature_values: np.ndarray = self._ext_state_to_feature_valuations.get(ext_state)

        if type(f_idxs) == int:
            f_idxs: List[int] = [f_idxs]

        projection: List[Tuple[int, int]] = []
        for f_idx in f_idxs:
            if boolean_projection:
                projection.append((f_idx, 1 if feature_values[f_idx] > 0 else 0))
            else:
                projection.append((f_idx, feature_values[f_idx]))
        return tuple(projection)

