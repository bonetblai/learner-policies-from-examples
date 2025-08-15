import logging

# Bitset-based unordered sets of unsigned integers
from intbitset import intbitset

from termcolor import colored
from typing import Set, Tuple, List, Union, Dict, Any, Optional, Union
from collections import defaultdict

#from ..feature_pool import Feature
#from ..feature_pool_utils import prune_features_with_same_feature_change_AND_boolean_valuation_reduced_v1, prune_features_with_same_feature_change_AND_boolean_valuation_reduced_v2
#from ..state_pair_equivalence_utils import make_conditions, make_effects
#from ..iteration_data import IterationData
from ..statistics import Statistics
from ...util import Timer

#from .asp_solver import ASPSolver
#from .returncodes import ClingoExitCode
#from ...state_space import PDDLInstance, StateFactory, get_plan, get_plan_v2


class MPairs:
    def __init__(self, ds: Dict[str, Any], monotone_only_by_dec: bool = False, lazy: bool = False, timers: Optional[Statistics] = None):
        # Get relevant data structures
        self._relevant_features_idxs: intbitset = ds.get("relevant_features_idxs")
        self._numerical_features_idxs: intbitset = ds.get("numerical_features_idxs")
        self._ext_states_by_bvalue_on_feature: Dict[Tuple[int, int], Set[Tuple[int, int]]] = ds.get("ext_states_by_bvalue_on_feature")
        self._ext_states_by_change_on_feature: Dict[Tuple[int, str], Set[Tuple[int, int]]] = ds.get("ext_states_by_change_on_feature")
        self._features_by_bvalue_on_ext_state: Dict[Tuple[Tuple[int, int], int], intbitset] = ds.get("features_by_bvalue_on_ext_state")
        self._features_by_change_on_ext_state: Dict[Tuple[Tuple[int, int], str], intbitset] = ds.get("features_by_change_on_ext_state")
        self._features_decreased_by_some_rule: intbitset = ds.get("features_decreased_by_some_rule")
        self._features_increased_by_some_rule: intbitset = ds.get("features_increased_by_some_rule")
        self._features_not_increased_by_any_rule: intbitset = ds.get("features_not_increased_by_any_rule")
        self._features_not_decreased_by_any_rule: intbitset = ds.get("features_not_decreased_by_any_rule")
        self._timers = timers if timers is not None else Statistics()
        self._monotone_only_by_dec: bool = monotone_only_by_dec
        self._lazy: bool = lazy

        self._initialize(self._lazy)
        self._calculate_monotone_features()

        if not self._lazy:
            self._ds_m_pairs: Dict[str, Any] = self._calculate()

    @classmethod
    def _invariant_for_fg_and_gf_maps(cls, fg_companions: Dict[int, intbitset], gf_companions: Dict[int, intbitset]) -> bool:
        for f_idx, g_idxs in fg_companions.items():
            for g_idx in g_idxs:
                if f_idx not in gf_companions.get(g_idx, intbitset()):
                    return False
        for g_idx, f_idxs in gf_companions.items():
            for f_idx in f_idxs:
                if g_idx not in fg_companions.get(f_idx, intbitset()):
                    return False
        return True

    def _initialize(self, lazy: bool):
        # Data structures
        self._m_pair_to_m_pair_index: Dict[Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]], int] = dict()
        self._m_pair_index_to_m_pair: List[Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]] = []
        self._m_pair_index_to_g_idxs: List[intbitset] = []
        self._m_pair_index_to_f_idxs: List[intbitset] = []
        self._g_idx_to_m_pair_index: Dict[int, int] = dict()
        self._f_idx_to_m_pair_indices: Dict[int, intbitset] = defaultdict(intbitset)

        self._monotone_features: intbitset = None
        self._usable_features: intbitset = None
        self._fg_companions: Dict[int, intbitset] = defaultdict(intbitset)
        self._gf_companions: Dict[int, intbitset] = defaultdict(intbitset)

    # (A, B) is m-pair for g_idx iff A (resp. B) is set of ext-states where g_idx remains "eqv" and g_idx Boolean value is 0 (resp. 1)
    def _m_pair_index_for_g_idx(self, g_idx: int) -> int:
        m_pair_index: int = self._g_idx_to_m_pair_index.get(g_idx)
        if m_pair_index is None:
            ext_states_for_eqv_change: Set[Tuple[int, int]] = self._ext_states_by_change_on_feature.get((g_idx, "eqv"), frozenset())
            ext_states_for_bvalue_0: Set[Tuple[int, int]] = self._ext_states_by_bvalue_on_feature.get((g_idx, 0), frozenset())
            ext_states_for_bvalue_1: Set[Tuple[int, int]] = self._ext_states_by_bvalue_on_feature.get((g_idx, 1), frozenset())
            m_pair: Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]] = (ext_states_for_eqv_change & ext_states_for_bvalue_0, ext_states_for_eqv_change & ext_states_for_bvalue_1)
            m_pair_index: int = self._m_pair_to_m_pair_index.get(m_pair)
            if m_pair_index is None:
                m_pair_index: int = len(self._m_pair_index_to_m_pair)
                self._m_pair_to_m_pair_index[m_pair] = m_pair_index
                self._m_pair_index_to_m_pair.append(m_pair)
                #self._m_pair_index_to_g_idxs.append(set())
                self._m_pair_index_to_g_idxs.append(intbitset())
                self._m_pair_index_to_f_idxs.append(None)
            self._m_pair_index_to_g_idxs[m_pair_index].add(g_idx)
            self._g_idx_to_m_pair_index[g_idx] = m_pair_index
        return m_pair_index

    def _f_idxs_for_m_pair_index(self, m_pair_index: int) -> intbitset:
        # Feature F is accepted for a change C given feature G and value V if any of the following:
        # - (1a) At some xstate S where G may equal V and G remains constant, F changes according to C, and
        #   (1b) for each xstate S2: if G may equal V and G may remain constant, F cannot change in opposite direction.
        # - (2a) There is no xstate S where G may equal V and G remains constant.

        f_idxs: intbitset = self._m_pair_index_to_f_idxs[m_pair_index]
        if f_idxs is None:
            m_pair: Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]] = self._m_pair_index_to_m_pair[m_pair_index]
            # Note: New version
            #   - f_idx is accepted <=> f_idx is accepted_for_0 AND f_idx is accepted_for_1
            #   - f_idx is accepted_for_0 <=> f_idx is accepted_for_0_for_DEC OR f_idx is accepted_for_0_for_INC
            #       - f_idx is accepted_for_0_for_DEC <=> ...
            #       - f_idx is accepted_for_0_for_INC <=> ...
            #   - f_idx is accepted_for_1 <=> f_idx is accepted_for_1_for_DEC OR f_idx is accepted_for_1_for_INC
            #       - f_idx is accepted_for_1_for_DEC <=> ...
            #       - f_idx is accepted_for_1_for_INC <=> ...
            bvalue_to_f_idxs: List[intbitset] = []
            for bvalue, m in enumerate(m_pair):
                if len(m) == 0:
                    bvalue_to_f_idxs.append(intbitset(self._relevant_features_idxs))
                else:
                    bvalue_to_f_idxs.append(intbitset())
                    for change in ["dec", "inc"]:
                        features_for_m_by_case1a: intbitset = intbitset().union(*[self._features_by_change_on_ext_state.get((ext_state, change), intbitset()) for ext_state in m])
                        m_iter = iter(m)
                        ext_state = next(m_iter)
                        features_for_m_by_case1b: intbitset = self._features_by_change_on_ext_state.get((ext_state, change), intbitset()) | self._features_by_change_on_ext_state.get((ext_state, "eqv"), intbitset())
                        features_for_m_by_case1b.intersection_update(*[self._features_by_change_on_ext_state.get((ext_state, change), intbitset()) | self._features_by_change_on_ext_state.get((ext_state, "eqv"), intbitset()) for ext_state in m_iter])
                        features_for_m_by_case1: intbitset = features_for_m_by_case1a & features_for_m_by_case1b
                        if self._monotone_only_by_dec and change == "inc":
                            features_for_m_by_case1 = features_for_m_by_case1 - self._numerical_features_idxs
                        bvalue_to_f_idxs[bvalue] |= features_for_m_by_case1
            f_idxs: intbitset = bvalue_to_f_idxs[0] & bvalue_to_f_idxs[1]
            self._m_pair_index_to_f_idxs[m_pair_index] = f_idxs
            for f_idx in f_idxs:
                self._f_idx_to_m_pair_indices[f_idx].add(m_pair_index)
        return f_idxs

    def _calculate_companions(self):
        # Incomplete unless all m-pairs have been generated.
        # At any time, it returns companions given current set of m-pairs.
        for g_idxs, f_idxs in zip(self._m_pair_index_to_g_idxs, self._m_pair_index_to_f_idxs):
            for f_idx in f_idxs - self._monotone_features:
                filtered_g_idxs: intbitset = g_idxs - intbitset([f_idx])
                if len(filtered_g_idxs) > 0:
                    self._fg_companions[f_idx] |= filtered_g_idxs
            for g_idx in g_idxs:
                filtered_f_idxs: intbitset = f_idxs - self._monotone_features - intbitset([g_idx])
                if len(filtered_f_idxs) > 0:
                    self._gf_companions[g_idx] |= filtered_f_idxs
        assert MPairs._invariant_for_fg_and_gf_maps(self._fg_companions, self._gf_companions)

    def _calculate_monotone_features(self) -> intbitset:
        if self._monotone_features is None:
            local_timer: Timer = Timer()
            #self._monotone_features: intbitset = (self._features_decreased_by_some_rule & self._features_not_increased_by_any_rule) | (self._features_increased_by_some_rule & self._features_not_decreased_by_any_rule)
            self._monotone_features: intbitset = self._features_decreased_by_some_rule & self._features_not_increased_by_any_rule
            if not self._monotone_only_by_dec:
                self._monotone_features |= self._features_increased_by_some_rule & self._features_not_decreased_by_any_rule
            else:
                self._monotone_features |= (self._features_increased_by_some_rule & self._features_not_decreased_by_any_rule) - self._numerical_features_idxs
            logging.info(f"MPairsContext: {len(self._monotone_features)} monotone feature(s) calculated in {local_timer.get_elapsed_sec():.02f} second(s)")
        return self._monotone_features

    def _calculate_usable_features(self) -> intbitset:
        if self._usable_features is None:
            self._usable_features: intbitset = intbitset()
            new_usable_features: intbitset = self._monotone_features
            while len(new_usable_features) > 0:
                self._usable_features |= new_usable_features
                new_usable_features: intbitset = self.f_idxs_for_g_idxs(self._usable_features) - self._usable_features
        return self._usable_features

    def _calculate(self, calculate_fg_and_gf_companions: bool = False):
        # Start timer
        local_timer: Timer = Timer()
        self._timers.resume("m_pairs")

        # Calculate m-pairs (partitions of ext states) generated by (relevant) features
        for g_idx in self._relevant_features_idxs:
            self._m_pair_index_for_g_idx(g_idx)
        logging.info(f"MPairs: #features={len(self._relevant_features_idxs)}, #m-pairs={len(self._m_pair_index_to_m_pair)}")

        # Calculate f_idxs enabled by each m-pair
        for m_pair_index in range(len(self._m_pair_index_to_m_pair)):
            self._f_idxs_for_m_pair_index(m_pair_index)

        local_timer.stop()
        self._timers.stop("m_pairs")
        logging.info(f"MPairs: maps for f_idxs and g_idxs computed in {local_timer.get_elapsed_sec():.02f} second(s)")

        if calculate_fg_and_gf_companions:
            # Calculate fg and gf companions mediated by m-pairs
            local_timer.resume()
            self._timers.resume("m_pairs")

            self._calculate_companions()

            local_timer.stop()
            self._timers.stop("m_pairs")
            logging.info(f"MPairs: #fg={sum([len(g_idxs) for f_idx, g_idxs in self._fg_companions.items()])} and #gf={sum([len(f_idxs) for g_idx, f_idxs in self._gf_companions.items()])} companion pair(s) computed in {local_timer.get_elapsed_sec():.02f} second(s)")

        # Calculate usable (reachable) features
        self._calculate_usable_features()
        logging.info(f"{len(self._usable_features)} usable feature(s) from {len(self._relevant_features_idxs)} feature(s)")

    def monotone_features(self) -> intbitset:
        self._calculate_monotone_features()
        return self._monotone_features

    def usable_features(self) -> intbitset:
        self._calculate_usable_features()
        return self._usable_features

    def g_idxs_for_f_idx(self, f_idx: int) -> intbitset:
        # This may be incomplete when not all m-pairs have been generated.
        # At any time, it can only return g_idxs that have been used to generate m-pairs.
        m_pair_indices: intbitset = self._f_idx_to_m_pair_indices.get(f_idx, intbitset())
        g_idxs: intbitset = intbitset().union(*[self._m_pair_index_to_g_idxs[m_pair_index] for m_pair_index in m_pair_indices])
        if self._monotone_only_by_dec and f_idx in self._numerical_features_idxs:
            # DISABLED
            # Numerical g_idx enables numerical f_idx only if
            # 1. f_idx is monotone by dec given g_idx
            # 2. there is no ext_state such that g_idx DEC and f_idx > 0
            #print(f"g_idxs={g_idxs}, sz={len(g_idxs)}")
            #bad_g_idxs: intbitset = intbitset().union(*[self._features_by_change_on_ext_state.get((ext_state, "dec"), intbitset()) for ext_state in self._ext_states_by_bvalue_on_feature.get((f_idx, 1), [])])
            #bad_g_idxs &= self._numerical_features_idxs
            #print(f"bad_g_idxs={bad_g_idxs}")
            #g_idxs -= bad_g_idxs
            #print(f"g_idxs={g_idxs}, sz={len(g_idxs)}")
            pass
        return g_idxs

    def f_idxs_for_g_idx(self, g_idx: int) -> intbitset:
        f_idxs: intbitset = self._gf_companions.get(g_idx)
        if f_idxs is None:
            m_pair_index: int = self._m_pair_index_for_g_idx(g_idx)
            f_idxs: intbitset = self._f_idxs_for_m_pair_index(m_pair_index) - self._monotone_features - intbitset([g_idx])
            if self._monotone_only_by_dec and g_idx in self._numerical_features_idxs:
                # DISABLED
                # Numerical g_idx enables numerical f_idx only if
                # 1. f_idx is monotone by dec given g_idx
                # 2. there is no ext_state such that g_idx DEC and f_idx > 0
                #print(f"f_idxs={f_idxs}, sz={len(f_idxs)}")
                #bad_f_idxs: intbitset = intbitset().union(*[self._features_by_bvalue_on_ext_state.get((ext_state, 1), intbitset()) for ext_state in self._ext_states_by_change_on_feature.get((g_idx, "dec"), [])])
                #bad_f_idxs &= self._numerical_features_idxs
                #print(f"bad_f_idxs={bad_f_idxs}")
                #f_idxs -= bad_f_idxs
                #print(f"f_idxs={f_idxs}, sz={len(f_idxs)}")
                pass
            self._gf_companions[g_idx] = f_idxs
        return f_idxs

    def f_idxs_for_g_idxs(self, g_idxs: intbitset) -> intbitset:
        f_idxs: intbitset = intbitset(self._monotone_features)
        for g_idx in g_idxs:
            f_idxs |= self.f_idxs_for_g_idx(g_idx)
        return f_idxs

