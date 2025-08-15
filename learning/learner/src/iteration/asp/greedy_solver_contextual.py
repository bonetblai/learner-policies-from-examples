import logging
import random
import numpy as np

# Bitset-based unordered sets of unsigned integers
from intbitset import intbitset

from termcolor import colored
from typing import Set, Tuple, List, Union, Dict, Any, Optional, Union
from collections import OrderedDict, defaultdict, deque
from itertools import product

import dlplan.core as dlplan_core

from ..feature_pool import Feature
from ...util import Timer
from ...state_space import StateFactory

#from .m_pairs_contextual import MPairsContextual


class TransitiveClosure:
    def __init__(self):
        self._TC: Dict[str, Dict[int, intbitset]] = {"start-at": defaultdict(intbitset), "end-at": defaultdict(intbitset)}

    def __call__(self, edge) -> bool:
        src, dst = edge
        return dst in self._TC["start-at"].get(src, [])

    def _new_edges(self, src: int, dst: int) -> Set[Tuple[int, int]]:
        new_edges: Set[Tuple[int, int]] = set([(src, dst)])
        # Add (x,dst) if (x,src)
        new_edges |= set([(x, dst) for x in self.end_at(src)])
        # Add (src,y) if (dst,y)
        new_edges |= set([(src, y) for y in self.start_at(dst)])
        # Add (x,y) if (x,src) & (dst,y)
        new_edges |= set([(x, y) for x in self.end_at(src) for y in self.start_at(dst)])
        return new_edges

    def edge(self, src: int, dst: int) -> bool:
        return src in self.end_at(dst)

    def start_at(self, src: int) -> intbitset:
        return self._TC["start-at"].get(src, intbitset())

    def end_at(self, dst: int) -> intbitset:
        return self._TC["end-at"].get(dst, intbitset())

    def update(self, src: int, dst: int):
        for (src2, dst2) in self._new_edges(src, dst):
            self._TC["start-at"][src2].add(dst2)
            self._TC["end-at"][dst2].add(src2)

    def maintains_acyclicity(self, src: int, dst: int) -> bool:
        if len(self.start_at(dst)) == 0 or len(self.end_at(src)) == 0:
            return True
        else:
            for (src2, dst2) in self._new_edges(src, dst):
                if src2 == dst2:
                    return False
            return True

    def calculate_ranks(self, vertices: intbitset, unique: bool = False) -> Dict[int, int]:
        # Vertices of rank = 0
        ranks: Dict[int, int] = {i: 0 for i in vertices if len(self.end_at(i)) == 0}
        assigned: intbitset = intbitset(list(ranks.keys()))

        # Vertices of rank > 0
        if not unique:
            rank = 1
            while len(assigned) < len(vertices):
                new_assigned: intbitset = intbitset()
                for j in vertices - assigned:
                    # Vertex j has rank R iff there is i with rank R-1 such that (i, j) in TC AND there is no k with (i, k) AND (k, j) in TC
                    below: List[int] = [i for i in self.end_at(j) if ranks.get(i) == rank - 1 and len(self.start_at(i) & self.end_at(j)) == 0]
                    if len(below) > 0:
                        ranks[j] = rank
                        new_assigned.add(j)
                        assigned.add(j)
                assert len(new_assigned) > 0
                assigned |= new_assigned
                rank += 1
        else:
            rank = 1
            while len(assigned) < len(vertices):
                for j in vertices - assigned:
                    if len(self.end_at(j) - assigned) == 0:
                        ranks[j] = rank
                        assigned.add(j)
                        break
                rank += 1

        return ranks


class GreedySolverContextual:
    def __init__(self, preprocessing_data: Dict[str, Any], state_factory: StateFactory, **kwargs):
        self._preprocessing_data: Dict[str, Any] = preprocessing_data
        self._state_factory: StateFactory = state_factory
        self._simplify_policy: bool = kwargs.get("simplify_policy", False)
        self._simplify_only_conditions: bool = kwargs.get("simplify_only_conditions", False)
        self._uniform_costs: bool = kwargs.get("uniform_costs", False)
        self._monotone_only_by_dec: bool = kwargs.get("monotone_only_by_dec", False)
        assert self._preprocessing_data is not None

        # Extract relevant data from pre-processing
        self._relevant_features: List[Tuple[int, Feature]] = self._preprocessing_data.get("relevant_features")
        self._relevant_f_idxs: List[int] = [f_idx for f_idx, _ in self._relevant_features]
        self._f_idx_to_feature_index: Dict[int, int] = self._preprocessing_data.get("f_idx_to_feature_index")
        self._f_idx_to_feature: Dict[int, Tuple[int, Feature]] = {f_idx: (f_idx_index, feature) for f_idx_index, (f_idx, feature) in enumerate(self._relevant_features)}

        # Pairs and contexts
        self._nu_context_to_index: OrderedDict[Tuple[int, int], int] = self._preprocessing_data.get("nu_context_to_index")
        self._nu_contexts: List[Tuple[int, int]] = list(self._nu_context_to_index.keys())
        self._fnu_pair_to_index: List[Tuple[int, Tuple[int, int]]] = self._preprocessing_data.get("fnu_pair_to_index")
        self._fnu_pairs: List[Tuple[int, Tuple[int, int]]] = list(self._fnu_pair_to_index.keys())
        self._fnu_idx_to_direction: List[str] = self._preprocessing_data.get("fnu_idx_to_direction")

        self._fnu_pair_costs: List[int] = [self._f_idx_to_feature[f_idx][1].complexity for f_idx, _ in self._fnu_pairs]
        for fnu_idx, (_, nu_idx) in enumerate(self._fnu_pairs):
            nu_context: Tuple[int, int] = self._nu_contexts[nu_idx]
            if len(nu_context) > 0:
                g_idx: int = nu_context[0]
                self._fnu_pair_costs[fnu_idx] += self._f_idx_to_feature[g_idx][1].complexity

        self._feature_index_to_nu_idxs: List[List[int]] = [[] for _ in self._relevant_features]
        self._feature_index_to_fnu_idxs: List[List[int]] = [[] for _ in self._relevant_features]
        self._nu_idx_to_fnu_idxs: List[List[int]] = [[] for _ in self._nu_contexts]
        for nu_idx, nu_context in enumerate(self._nu_contexts):
            if len(nu_context) > 0:
                g_idx = nu_context[0]
                g_idx_index = self._f_idx_to_feature_index[g_idx]
                self._feature_index_to_nu_idxs[g_idx_index].append(nu_idx)
        for fnu_idx, (f_idx, nu_idx) in enumerate(self._fnu_pairs):
            f_idx_index = self._f_idx_to_feature_index[f_idx]
            self._feature_index_to_fnu_idxs[f_idx_index].append(fnu_idx)
            self._nu_idx_to_fnu_idxs[nu_idx].append(fnu_idx)

        assert self._relevant_features is not None
        assert self._f_idx_to_feature_index is not None
        assert self._nu_context_to_index is not None
        assert self._fnu_pair_to_index is not None
        for nu_idx, nu_context in enumerate(self._nu_contexts): assert nu_idx == self._nu_context_to_index.get(nu_context), (nu_idx, self._nu_context_to_index.get(nu_context))
        for fnu_idx, fnu_pair in enumerate(self._fnu_pairs): assert fnu_idx == self._fnu_pair_to_index.get(fnu_pair), (fnu_idx, self._fnu_pair_to_index.get(fnu_pair))
        logging.info(f"{len(self._nu_contexts)} nu-context(s), {len(self._fnu_pairs)} fnu-pair(s)")

        # Requirements
        self._requirements_for_good_transitions: Dict[Tuple[int, int], intbitset] = self._preprocessing_data.get("requirements_for_good_transitions")
        self._goal_ext_pair_to_separating_features: Dict[Tuple[int, Tuple[int, int]], intbitset] = self._preprocessing_data.get("goal_ext_pair_to_separating_features")
        #self._deadend_path_to_separating_features: Dict[Tuple[Tuple[int, int]], List[intbitset]] = self._preprocessing_data.get("deadend_path_to_separating_features")
        self._ext_state_to_separating_features_for_deadend_paths: Dict[Tuple[int, int], Dict[Tuple[int, Tuple[int, int]], intbitset]] = self._preprocessing_data.get("ext_state_to_separating_features_for_deadend_paths")
        #XXX self._ext_sibling_to_separating_features: Dict[Tuple[int, int, Tuple[int, int]], intbitset] = self._preprocessing_data.get("ext_sibling_to_separating_features")
        self._ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]] = self._preprocessing_data.get("ext_state_to_ext_edge")

        #XXX self._m_pairs: MPairsContextual = self._preprocessing_data.get("m_pairs")
        #XXX self._monotone_features: intbitset = self._m_pairs.monotone_features()

        # Calculate numerical features
        self._numerical_features: intbitset = intbitset([f_idx for f_idx, feature in self._relevant_features if isinstance(feature.dlplan_feature, dlplan_core.Numerical)])

        # Construct requirements, one per ex-edge and one per pair of goal and non-goal xstates
        self._annotated_requirements: List[Tuple[Dict[str, Any], intbitset]] = []
        self._annotated_requirements.extend([({"key": "Good", "ext_state": ext_state}, requirement) for ext_state, requirement in self._requirements_for_good_transitions.items()])
        self._annotated_requirements.extend([({"key": "Goal", "pair": pair}, separating_features) for pair, separating_features in self._goal_ext_pair_to_separating_features.items()])
        self._annotated_requirements.extend([({"key": "Deadend", "ext_state": ext_state, "path": path}, separating_features) for ext_state, separating_features_for_deadend_paths in self._ext_state_to_separating_features_for_deadend_paths.items() for path, separating_features in separating_features_for_deadend_paths.items()])
        #XXX self._annotated_requirements.extend([({"key": "Sibling", "ext_sibling": ext_sibling}, separating_features) for ext_sibling, separating_features in self._ext_sibling_to_separating_features.items()])
        self._requirements: List[Tuple[str, intbitset]] = [(d["key"], requirement) for d, requirement in self._annotated_requirements]
        self._other_requirements: List[intbitset] = [requirement for annotation, requirement in self._annotated_requirements if annotation.get("key") in ["XGoal", "Deadend"]]
        self._num_requirements: Dict[str, int] = {key: sum([1 if annotation.get("key") == key else 0 for annotation, _ in self._annotated_requirements]) for key in ["Good", "Goal", "Deadend", "Sibling"]}
        logging.info(f"{len(self._requirements)} requirement(s) split as {self._num_requirements}")

        # Support for simplification of policies
        #XXX self._ex_ext_states: Set[Tuple[int, int]] = self._preprocessing_data.get("ex_ext_states")
        self._ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = self._preprocessing_data.get("ext_state_to_feature_valuations")

    def _is_numerical(self, f_idx: int) -> bool:
        return f_idx in self._numerical_features

    def _f_idxs_for_fnu_idx(self, fnu_idx: int) -> Tuple[int, int]:
        f_idx, nu_idx = self._fnu_pairs[fnu_idx]
        nu_context: Tuple[int, int] = self._nu_contexts[nu_idx]
        g_idx: int = None if len(nu_context) == 0 else nu_context[0]
        return (f_idx, g_idx)

    def _fnu_idxs_affected_by_f_idxs(self, f_idxs: List[int]) -> intbitset:
        f_idx_indices: List[int] = [self._f_idx_to_feature_index[f_idx] for f_idx in f_idxs]
        fnu_idxs: intbitset = intbitset([fnu_idx for f_idx_index in f_idx_indices for fnu_idx in self._feature_index_to_fnu_idxs[f_idx_index]])
        fnu_idxs |= intbitset([fnu_idx for f_idx_index in f_idx_indices for nu_idx in self._feature_index_to_nu_idxs[f_idx_index] for fnu_idx in self._nu_idx_to_fnu_idxs[nu_idx]])
        return fnu_idxs

    def _fnu_idxs_affected_by_fnu_idx(self, fnu_idx: int) -> intbitset:
        f_idxs: List[int] = [f_idx for f_idx in self._f_idxs_for_fnu_idx(fnu_idx) if f_idx is not None]
        return self._fnu_idxs_affected_by_f_idxs(f_idxs)

    def _eligible_fnu_idx(self, fnu_idx: int, incumbent_f_idxs: intbitset, TC: TransitiveClosure) -> bool:
        f_idx, g_idx = self._f_idxs_for_fnu_idx(fnu_idx)
        return g_idx is None or (g_idx in incumbent_f_idxs and TC.maintains_acyclicity(g_idx, f_idx))

    def _score_fnu_idx(self, fnu_idx: int, requirements_good: List[int], requirements_other: List[int], incumbent_f_idxs: intbitset, fnu_pair_costs: List[int]) -> Tuple[float, int, int]:
        fnu_pair_cost: int = fnu_pair_costs[fnu_idx]
        f_idx, g_idx = self._f_idxs_for_fnu_idx(fnu_idx)
        supported: bool = g_idx is None or g_idx in incumbent_f_idxs
        f_idxs: intbitset = intbitset([f_idx] if g_idx is None else [f_idx, g_idx])

        count_fulfilled_requirements: int = sum([1 for i in requirements_good if fnu_idx in self._requirements[i][1]])
        count_fulfilled_requirements_other: int = sum([1 for i in requirements_other if len(f_idxs & self._requirements[i][1]) > 0])

        if fnu_pair_cost == 0:
            if count_fulfilled_requirements + count_fulfilled_requirements_other > 0:
                return (1e6, count_fulfilled_requirements, count_fulfilled_requirements_other)
            else:
                return (-1,)
        elif supported:
            return (1, count_fulfilled_requirements / fnu_pair_cost, count_fulfilled_requirements, count_fulfilled_requirements_other)
        else:
            return (0, count_fulfilled_requirements / fnu_pair_cost, count_fulfilled_requirements, count_fulfilled_requirements_other)

    def _score_f_idx(self, f_idx: int, pending_requirements: intbitset, feature_costs: List[int]) -> Tuple[float, int]:
        f_idx_cost: int = feature_costs[self._f_idx_to_feature_index[f_idx]]
        count_fulfilled_requirements: int = sum([1 for i in pending_requirements if self._requirements[i][0] != "Good" and f_idx in self._requirements[i][1]])
        assert f_idx_cost > 0
        return (count_fulfilled_requirements / f_idx_cost, count_fulfilled_requirements)

    def _calculate_decorations(self, fnu_idxs: intbitset, ext_states_to_fnu_idxs: Dict[Tuple[int, int], intbitset]) -> Dict[str, Dict[int, Dict[int, intbitset]]]:
        unknowns: Dict[int, Dict[int, intbitset]] = defaultdict(lambda: defaultdict(intbitset))
        dont_cares: Dict[int, Dict[int, intbitset]] = defaultdict(lambda: defaultdict(intbitset))

        fnu_pairs: List[Tuple[int, int]] = [self._fnu_pairs[fnu_idx] for fnu_idx in fnu_idxs]
        varrho: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
        for f_idx, nu_idx in fnu_pairs:
            varrho[f_idx].add(self._nu_contexts[nu_idx])
        f_idxs: intbitset = intbitset([f_idx for f_idx, _ in fnu_pairs]) | intbitset([nu_context[0] for nu_contexts in varrho.values() for nu_context in nu_contexts if len(nu_context) > 0])
        logging.debug(f"fnu_pairs: {fnu_pairs}")
        logging.debug(f"varrho: {varrho}")
        logging.debug(f"f_idxs: {sorted(f_idxs)}")

        # Simplify conditions
        if True: # CAUSES CYCLE IN CHILDSNACK
            for ext_state, fnu_idxs in ext_states_to_fnu_idxs.items():
                assert len(fnu_idxs) == 1
                fnu_idx: int = list(fnu_idxs)[0]
                f_idx, nu_idx = self._fnu_pairs[fnu_idx]
                nu_context: Tuple[int, int] = self._nu_contexts[nu_idx]
                g_idx: int = None if len(nu_context) == 0 else nu_context[0]
                g_value: int = None if len(nu_context) == 0 else nu_context[1]

                ext_edge: Tuple[int, Tuple[int, int]] = self._ext_state_to_ext_edge.get(ext_state)
                src_feature_values: np.ndarray = self._ext_state_to_feature_valuations.get(ext_state)
                dst_feature_values: np.ndarray = self._ext_state_to_feature_valuations.get((ext_edge[0], ext_edge[1][1]))
                conditions: Dict[int, str] = {h_idx: "EQ" if src_feature_values[h_idx] == 0 else "GT" for h_idx in f_idxs}
                effects: List[Tuple[int, int]] = [(h_idx, dst_feature_values[h_idx] - src_feature_values[h_idx]) for h_idx in f_idxs]
                effects: Dict[int, str] = {h_idx: "BOT" if d == 0 else "INC" if d > 0 else "DEC" for h_idx, d in effects}
                #logging.info(f"    conditions: {conditions}")
                #logging.info(f"       effects: {effects}")

                for h_idx, condition in conditions.items():
                    effect: str = effects.get(h_idx)
                    if condition == "EQ" and h_idx != g_idx: # Last condition holds when either g_idx is None or g_idx is not None
                        # Add r' = cond(r) + (h, GT) => eff(r)A
                        dont_cares[ext_state[0]][ext_state[1]].add(h_idx)
                    elif condition == "GT" and effect != "DEC" and h_idx != g_idx: # Last condition holds when either g_idx is None or g_idx is not None
                        # Add r' = cond(r) + (h, EQ) => eff(r)
                        dont_cares[ext_state[0]][ext_state[1]].add(h_idx)

        # Simplify effects
        if True: # CAUSES CYCLE IN GRIPPER
            for ext_state, fnu_idxs in ext_states_to_fnu_idxs.items():
                assert len(fnu_idxs) == 1
                fnu_idx: int = list(fnu_idxs)[0]
                f_idx, nu_idx = self._fnu_pairs[fnu_idx]
                nu_context: Tuple[int, int] = self._nu_contexts[nu_idx]
                g_idx: int = None if len(nu_context) == 0 else nu_context[0]
                g_value: int = None if len(nu_context) == 0 else nu_context[1]

                ext_edge: Tuple[int, Tuple[int, int]] = self._ext_state_to_ext_edge.get(ext_state)
                src_feature_values: np.ndarray = self._ext_state_to_feature_valuations.get(ext_state)
                dst_feature_values: np.ndarray = self._ext_state_to_feature_valuations.get((ext_edge[0], ext_edge[1][1]))
                conditions: Dict[int, str] = {h_idx: "EQ" if src_feature_values[h_idx] == 0 else "GT" for h_idx in f_idxs}
                effects: List[Tuple[int, int]] = [(h_idx, dst_feature_values[h_idx] - src_feature_values[h_idx]) for h_idx in f_idxs]
                effects: Dict[int, str] = {h_idx: "BOT" if d == 0 else "INC" if d > 0 else "DEC" for h_idx, d in effects}
                #logging.info(f"    conditions: {conditions}")
                #logging.info(f"       effects: {effects}")

                for h_idx, effect in effects.items():
                    # If g_idx is None, f_idx is unconditionally monotone and its effect cannot be simplified
                    if g_idx is not None and h_idx not in [f_idx, g_idx]:
                        # Let C = {\nu' : (h, \nu') is pair for h} be set of pairs for h
                        # If {\nu', cond(r)} is inconsistent for ALL \nu' in C:
                        #     Replace effect (h, e) by (h, UNK)
                        nu_contexts_for_h_idx: Set[Tuple[int, int]] = varrho.get(h_idx)
                        if any([len(nu_context) == 0 for nu_context in nu_contexts_for_h_idx]):
                            continue
                        if all([conditions[x_idx] != x_value for x_idx, x_value in nu_contexts_for_h_idx]):
                            unknowns[ext_state[0]][ext_state[1]].add(h_idx)

        # Merge rules
        some_change: bool = True
        ext_states: List[Tuple[int, int]] = list(ext_states_to_fnu_idxs.keys())
        ext_pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = [(ext_state0, ext_state1) for ext_state0, ext_state1 in product(ext_states, ext_states) if ext_state0[0] == ext_state1[0] and ext_state0[1] < ext_state1[1]]
        logging.warning("*** DISABLE MERGE OF RULES")
        while False and some_change:
            some_change = False
            for ext_state0, ext_state1 in ext_pairs:
                instance_idx: int = ext_state0[0]
                ext_edge0: Tuple[int, Tuple[int, int]] = self._ext_state_to_ext_edge.get(ext_state0)
                ext_edge1: Tuple[int, Tuple[int, int]] = self._ext_state_to_ext_edge.get(ext_state1)
                #state_idx0, state_idx1 = ext_state0[1], ext_state1[1]
                #dont_cares0: intbitset = dont_cares.get(instance_idx, dict()).get(state_idx0, intbitset())
                #dont_cares1: intbitset = dont_cares.get(instance_idx, dict()).get(state_idx1, intbitset())
                #dont_cares_both: intbitset = dont_cares0 | dont_cares1

                src_feature_values0: np.ndarray = self._ext_state_to_feature_valuations.get(ext_state0)
                src_feature_values1: np.ndarray = self._ext_state_to_feature_valuations.get(ext_state1)
                condition0: List[int] = [src_feature_values0[f_idx] for f_idx in f_idxs]
                condition1: List[int] = [src_feature_values1[f_idx] for f_idx in f_idxs]
                condition_diff: List[int] = [f_idx for i, f_idx in enumerate(f_idxs) if condition0[i] != condition1[i]]

                dst_feature_values0: np.ndarray = self._ext_state_to_feature_valuations.get((instance_idx, ext_edge0[1][1]))
                dst_feature_values1: np.ndarray = self._ext_state_to_feature_valuations.get((instance_idx, ext_edge1[1][1]))
                effect0: List[int] = [dst_feature_values0[f_idx] for f_idx in f_idxs]
                effect1: List[int] = [dst_feature_values1[f_idx] for f_idx in f_idxs]
                effect_diff: List[int] = [f_idx for i, f_idx in enumerate(f_idxs) if effect0[i] != effect1[i]]
                unknown_common: intbitset = unknowns.get(instance_idx, dict()).get(ext_state0[1], intbitset()) & unknowns.get(instance_idx, dict()).get(ext_state1[1], intbitset())
                effect_equivalent: bool = len([f_idx for f_idx in effect_diff if f_idx not in unknown_common]) == 0

                if effect_equivalent and len(condition_diff) == 1:
                    f_idx: int = condition_diff[0]
                    num_additions: int = 0
                    if f_idx not in dont_cares.get(instance_idx, dict()).get(ext_state0[1], []):
                        dont_cares[instance_idx][ext_state0[1]].add(f_idx)
                        num_additions += 1
                        logging.info(f"ADD don't care f{f_idx} to rule0 associated with {ext_state0}")
                    if f_idx not in dont_cares.get(instance_idx, dict()).get(ext_state1[1], []):
                        dont_cares[instance_idx][ext_state1[1]].add(f_idx)
                        logging.info(f"ADD don't care f{f_idx} to rule1 associated with {ext_state1}")
                        num_additions += 1
                    some_change = num_additions > 0
                    #if some_change: assert False
        logging.warning("*** FULL MERGE OF RULEs NOT YET IMPLEMENTED")

        # Construct and return decorations
        dont_cares: Dict[int, Dict[int, intbitset]] = {instance_idx: {state_idx: f_idxs for state_idx, f_idxs in subdict.items()} for instance_idx, subdict in dont_cares.items()}
        unknowns: Dict[int, Dict[int, intbitset]] = {instance_idx: {state_idx: f_idxs for state_idx, f_idxs in subdict.items()} for instance_idx, subdict in unknowns.items()}
        decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = {"unknown": unknowns, "dont_care": dont_cares}
        return decorations

    def _calculate_decorations2(self,
                                chosen_fnu_idxs: intbitset,
                                chosen_f_idxs: intbitset,
                                ext_states_to_fnu_idxs: Dict[Tuple[int, int], intbitset],
                                ranks: Dict[int, int],
                                sigma: Dict[Tuple[int, Tuple[int, int]], str],
                                inv_sigma: Dict[Tuple[int, int], List[int]],
                                TC: TransitiveClosure) -> Dict[str, Dict[int, Dict[int, intbitset]]]:
        # Calculate min-cost subset of f_idxs that solve other requirements
        f_idxs_for_other_requirements: intbitset = intbitset()
        pending_other_requirements: intbitset = intbitset(range(len(self._other_requirements)))
        while len(pending_other_requirements) > 0:
            f_idxs_with_scores: List[Tuple[int, int]] = [(f_idx, sum([1 if f_idx in self._other_requirements[i] else 0 for i in pending_other_requirements])) for f_idx in chosen_f_idxs - f_idxs_for_other_requirements]
            sorted_f_idxs_with_scores: List[Tuple[int, int]] = sorted(f_idxs_with_scores, key=lambda pair: pair[1], reverse=True)
            assert len(sorted_f_idxs_with_scores) > 0
            best_f_idx: int = sorted_f_idxs_with_scores[0][0]
            f_idxs_for_other_requirements.add(best_f_idx)
            pending_other_requirements: intbitset = pending_other_requirements - intbitset([i for i in pending_other_requirements if best_f_idx in self._other_requirements[i]])
        assert all([len(requirement & f_idxs_for_other_requirements) > 0 for requirement in self._other_requirements])
        logging.info(f"f_idxs: ALL={sorted(chosen_f_idxs)}, KEEP={sorted(f_idxs_for_other_requirements)}")

        # Unknown and don't care f_idxs for ext_states (rules)
        unknowns: Dict[int, Dict[int, intbitset]] = defaultdict(lambda: defaultdict(intbitset))
        dont_cares: Dict[int, Dict[int, intbitset]] = defaultdict(lambda: defaultdict(intbitset))

        # Rule set data
        #fnu_pairs: List[Tuple[int, int]] = [self._fnu_pairs[fnu_idx] for fnu_idx in chosen_fnu_idxs]
        #varrho: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
        #for f_idx, nu_idx in fnu_pairs:
        #    varrho[f_idx].add(self._nu_contexts[nu_idx])
        #logging.info(f"fnu_pairs: {fnu_pairs}")
        #logging.info(f"varrho: {varrho}")
        #logging.info(f"f_idxs={sorted(chosen_f_idxs)}, numerical={sorted(chosen_f_idxs & self._numerical_features)}")

        # Calculate decorations for each rule
        for ext_state, fnu_idxs_for_ext_state in ext_states_to_fnu_idxs.items():
            assert len(fnu_idxs_for_ext_state) == 1
            instance_idx, state_idx = ext_state
            fnu_idx: int = list(fnu_idxs_for_ext_state)[0]
            f_idx, nu_idx = self._fnu_pairs[fnu_idx]
            nu_context: Tuple[int, int] = self._nu_contexts[nu_idx]
            g_idx: int = None if len(nu_context) == 0 else nu_context[0]
            g_value: int = None if len(nu_context) == 0 else nu_context[1]
            logging.info(f"ext_state={ext_state}, fnu_idxs={sorted(fnu_idxs_for_ext_state)}, f_idx={f_idx}, nu_context={nu_context}")

            #preceding_features: intbitset = intbitset([h_idx for h_idx in chosen_f_idxs if ranks.get(h_idx) < f_idx_rank])
            #logging.info(f"    f_idx={f_idx}, preceding_features: {preceding_features}")

            ext_edge: Tuple[int, Tuple[int, int]] = self._ext_state_to_ext_edge.get(ext_state)
            src_feature_values: np.ndarray = self._ext_state_to_feature_valuations.get(ext_state)
            dst_feature_values: np.ndarray = self._ext_state_to_feature_valuations.get((ext_edge[0], ext_edge[1][1]))
            conditions: List[Tuple[int, int]] = [(h_idx, src_feature_values[h_idx]) for h_idx in chosen_f_idxs]
            effects: List[Tuple[int, int]] = [(h_idx, dst_feature_values[h_idx] - src_feature_values[h_idx]) for h_idx in chosen_f_idxs]
            conditions: Dict[int, str] = {h_idx: "EQ" if value == 0 else "GT" for h_idx, value in conditions}
            effects: Dict[int, str] = {h_idx: "BOT" if d == 0 else ("INC" if d > 0 else "DEC") for h_idx, d in effects}
            #changed: Set[int] = {h_idx for h_idx, effect in effects.items() if effect != "BOT"}
            logging.info(f"    conditions: {conditions}")
            logging.info(f"       effects: {effects}")
            #logging.info(f"       changed: {changed}")

            # Check invariants: f_idx must be changed by rule according to (registered) sigma, g_idx cannot changt and agree with sigma, g_idx must comes before f_idx
            assert effects.get(f_idx) in ["DEC", "INC"]
            assert effects.get(f_idx) == sigma.get((f_idx, nu_context))
            assert g_idx is None or (effects.get(g_idx) == "BOT" and conditions.get(g_idx) == ("EQ" if g_value == 0 else "GT"))
            assert g_idx is None or TC.edge(g_idx, f_idx)
            for x_idx, context in sigma.keys():
                if x_idx == f_idx and len(context) > 0:
                    assert TC.edge(context[0], f_idx)
                    assert not TC.edge(f_idx, context[0])

            # Situation: contexts (f, g1), (f, g2)

            for h_idx in chosen_f_idxs - f_idxs_for_other_requirements:
                # Decorate conditions and effects
                # Mark feature h as DC and UNK when feature f for rule comes *before* h
                if TC.edge(f_idx, h_idx):
                    assert h_idx != f_idx
                    logging.info(f"     MARK.0 [f_idx={f_idx}] DC(f{h_idx}) and UNK(f{h_idx}) from {conditions.get(h_idx)}(f{h_idx}) and {effects.get(h_idx)}(f{h_idx})")
                    dont_cares[instance_idx][state_idx].add(h_idx)
                    if not self._simplify_only_conditions:
                        unknowns[instance_idx][state_idx].add(h_idx)

                # Decorate conditions
                # Mark feature h with condition h=v as DC when:
                # 1. rule doesn't change h
                # 2. for all pairs (i,h=1-v) in \sigma: direction of (i,h=1-v) is *consistent* with change of i by r
                elif False and effects.get(h_idx) == "BOT":
                    condition_on_all_pairs: bool = True
                    for (i_idx, i_context), direction in sigma.items():
                        if len(i_context) == 0:
                            j_idx, j_value = None, None
                        else:
                            j_idx, j_value = i_context
                        j_condition: str = "EQ" if j_value == 0 else "GT"
                        logging.info(f"h_idx={h_idx}, i_idx={i_idx}, i_context={(j_idx, j_value)}, direction={direction}, effect={effects.get(i_idx)}(f{i_idx})")
                        if (j_idx is None or j_idx == h_idx or conditions.get(j_idx) == j_condition) and direction != effects.get(i_idx):
                            condition_on_all_pairs = False
                            break
                    logging.debug(f"h_idx={h_idx}, condition_on_all_pairs: {condition_on_all_pairs}")
                    if condition_on_all_pairs:
                        logging.info(f"     MARK.1 [f_idx={f_idx}] DC(f{h_idx}) from {conditions.get(h_idx)}(f{h_idx})")
                        dont_cares[instance_idx][state_idx].add(h_idx)

                # Decorate effects
                # Mark feature h as UNK when:
                # 1. Monotone-only-by-dec is TRUE, h is numerical, and INC(h) is effect
                elif False and (not self._simplify_only_conditions) and self._monotone_only_by_dec and self._is_numerical(h_idx) and effects.get(h_idx) == "INC":
                    logging.info(f"     MARK.2 [f_idx={f_idx}] UNK(f{h_idx}) from {effects.get(h_idx)}(f{h_idx})")
                    unknowns[instance_idx][state_idx].add(h_idx)

        # Return aggregated decorations
        dont_cares: Dict[int, Dict[int, intbitset]] = {instance_idx: {state_idx: f_idxs for state_idx, f_idxs in subdict.items()} for instance_idx, subdict in dont_cares.items()}
        unknowns: Dict[int, Dict[int, intbitset]] = {instance_idx: {state_idx: f_idxs for state_idx, f_idxs in subdict.items()} for instance_idx, subdict in unknowns.items()}
        decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = {"unknown": unknowns, "dont_care": dont_cares}
        return decorations

    # Solver that at each iteration solves a requiement. Number of iteration is thus bounded by number of requirements.
    def solve(self, **kwargs) -> Tuple[bool, intbitset, int, Dict[str, Dict[int, Dict[int, intbitset]]], int]:
        logging.info(f"Starting greedy solver...")
        local_timer: Timer = Timer()

        # Initialize costs and transitive closure
        feature_costs: List[int] = [feature.complexity for _, feature in self._relevant_features]
        fnu_pair_costs: List[int] = list(self._fnu_pair_costs)
        TC: TransitiveClosure = TransitiveClosure()

        # Grow incumbent set until all requirements are fulfilled
        incumbent_fnu_idxs: intbitset = intbitset()
        incumbent_f_idxs: intbitset = intbitset()
        pending_requirements_good: List[int] = [i for i, (label, _) in enumerate(self._requirements) if label == "Good"]
        pending_requirements_other: List[int] = [i for i, (label, _) in enumerate(self._requirements) if label != "Good"]
        pending_requirements: List[int] = pending_requirements_good + pending_requirements_other
        ext_states_to_fnu_idxs: Dict[Tuple[int, int], intbitset] = dict()
        while len(pending_requirements) > 0:
            logging.info(f"{len(pending_requirements)} pending requirement(s): good={sorted(pending_requirements_good)}, other={sorted(pending_requirements_other)}")
            choose_fnu_idx: bool = len(pending_requirements_good) > 0
            choose_fnu_idx: bool = True

            # Sort eligible items by score
            timer: Timer = Timer()
            if choose_fnu_idx:
                # Eligible items are those that do not create a dependency loop (i.e. there is ranking function for them + chosen)
                eligible_items: List[int] = [fnu_idx for fnu_idx in range(len(fnu_pair_costs)) if fnu_idx not in incumbent_fnu_idxs and self._eligible_fnu_idx(fnu_idx, incumbent_f_idxs, TC)]

                # Sort eligible items by score
                eligible_items_with_score: List[Tuple[int, Any]] = [(fnu_idx, self._score_fnu_idx(fnu_idx, pending_requirements_good, pending_requirements_other, incumbent_f_idxs, fnu_pair_costs)) for fnu_idx in eligible_items]
                eligible_items_with_score: List[Tuple[int, Any]] = [(fnu_idx, score) for fnu_idx, score in eligible_items_with_score if score != (0, 0, 0)]
                sorted_eligible_items: List[Tuple[int, Any]] = sorted(eligible_items_with_score, key=lambda item: item[1], reverse=True)
                sorted_eligible_items: List[Tuple[int, Any]] = [(item, score) for item, score in sorted_eligible_items if score[0] >= 0 and score[1] > 0]

            if False and (not choose_fnu_idx or len(sorted_eligible_items) == 0):
                # Eligible items are f_idxs not already chosen
                eligible_items: List[int] = [f_idx for f_idx in self._relevant_f_idxs if f_idx not in incumbent_f_idxs]

                # Sort eligible items by score
                eligible_items_with_score: List[Tuple[int, Any]] = [(f_idx, self._score_f_idx(f_idx, pending_requirements, feature_costs)) for f_idx in eligible_items]
                sorted_eligible_items: List[Tuple[int, Any]] = sorted(eligible_items_with_score, key=lambda pair: pair[1], reverse=True)
                sorted_eligible_items: List[Tuple[int, Any]] = [(item, score) for item, score in sorted_eligible_items if score[0] > 0]
            timer.stop()
            logging.info(f"{len(eligible_items)} eligible item(s) computed in {timer.get_elapsed_sec():0.2f} second(s)")

            # Check for early termination due to non-existence of solution
            if len(sorted_eligible_items) == 0:
                logging.warning(f"No eligible items: incumbent_fnu_idxs={sorted(incumbent_fnu_idxs)}")
                logging.warning(f"Analysis of pending requirements:")
                for r_idx in pending_requirements:
                    annotation, requirement = self._annotated_requirements[r_idx]
                    key: str = annotation.get("key")
                    if key == "Good":
                        ext_state: Tuple[int, int] = annotation.get("ext_state")
                        ext_edge: Tuple[int, Tuple[int, int]] = self._ext_state_to_ext_edge.get(ext_state)
                        src_dlplan_state: dlplan_core.State = self._state_factory.get_dlplan_state(ext_edge[0], ext_edge[1][0])
                        dst_dlplan_state: dlplan_core.State = self._state_factory.get_dlplan_state(ext_edge[0], ext_edge[1][1])
                        logging.warning(f"  r_idx={r_idx}, key={key}, ext_edge={ext_edge}, requirement={requirement}")
                        logging.warning(f"      src_state: {ext_edge[1][0]}.{src_dlplan_state}")
                        logging.warning(f"      dst_state: {ext_edge[1][1]}.{dst_dlplan_state}")
                    elif key == "Goal":
                        path: Tuple[Tuple[int, int]] = annotation.get("path")
                        pair: Tuple[int, Tuple[int, int]] = annotation.get("pair")
                        dlplan_state_0: dlplan_core.State = self._state_factory.get_dlplan_state(pair[0], pair[1][0])
                        dlplan_state_1: dlplan_core.State = self._state_factory.get_dlplan_state(pair[0], pair[1][1])
                        logging.warning(f"  r_idx={r_idx}, key={key}, pair={pair}, requirement={requirement}")
                        logging.warning(f"      state_0: {pair[1][0]}.{dlplan_state_0}")
                        logging.warning(f"      state_1: {pair[1][1]}.{dlplan_state_1}")
                    elif key == "Deadend":
                        ext_edge: Tuple[int, Tuple[int, int]] = annotation.get("path")
                        src_dlplan_state: dlplan_core.State = self._state_factory.get_dlplan_state(ext_edge[0], ext_edge[1][0])
                        dst_dlplan_state: dlplan_core.State = self._state_factory.get_dlplan_state(ext_edge[0], ext_edge[1][1])
                        logging.warning(f"  r_idx={r_idx}, key={key}, ext_edge={ext_edge}, requirement={requirement}")
                        logging.warning(f"      src_state: {ext_edge[1][0]}.{src_dlplan_state}")
                        logging.warning(f"      dst_state: {ext_edge[1][1]}.{dst_dlplan_state}")
                    elif key == "Sibling":
                        ext_sibling: Tuple[int, int, Tuple[int, int]] = annotation.get("ext_sibling")
                        src_dlplan_state: dlplan_core.State = self._state_factory.get_dlplan_state(ext_sibling[0], ext_sibling[1])
                        dst1_dlplan_state: dlplan_core.State = self._state_factory.get_dlplan_state(ext_sibling[0], ext_sibling[2][0])
                        dst2_dlplan_state: dlplan_core.State = self._state_factory.get_dlplan_state(ext_sibling[0], ext_sibling[2][1])
                        logging.warning(f"  r_idx={r_idx}, key={key}, ext_sibling={ext_sibling}, requirement={requirement}")
                        logging.warning(f"       src_state: {ext_sibling[1]}.{src_dlplan_state}")
                        logging.warning(f"      dst1_state: {ext_sibling[2][0]}.{dst1_dlplan_state}")
                        logging.warning(f"      dst2_state: {ext_sibling[2][1]}.{dst2_dlplan_state}")
                        assert False
                raise RuntimeError(f"No eligible features")

            # Choose a best items
            best_score: Any = sorted_eligible_items[0][1]
            best_items: List[int] = [item for item, score in sorted_eligible_items if score == best_score]

            if choose_fnu_idx:
                fnu_idx: int = random.choice(best_items)
                cost: int = fnu_pair_costs[fnu_idx]
                f_idx, nu_idx = self._fnu_pairs[fnu_idx]
                nu_context: Tuple[int, int] = self._nu_contexts[nu_idx]
                g_idx: int = None if len(nu_context) == 0 else nu_context[0]
                logging.info(f"#eligible={len(eligible_items)}, #best={len(best_items)}, score={best_score}, fnu={fnu_idx}.({f_idx}, {nu_context}), cost={cost}, f={f_idx}.{self._f_idx_to_feature[f_idx][1]._dlplan_feature}, g={g_idx}{'' if g_idx is None else '.' + str(self._f_idx_to_feature[g_idx][1]._dlplan_feature)}")

                # Extend incumbent sets and revise feature costs
                assert g_idx is None or g_idx in incumbent_f_idxs
                incumbent_fnu_idxs.add(fnu_idx)
                incumbent_f_idxs.add(f_idx)
                feature_costs[self._f_idx_to_feature_index[f_idx]] = 0
                if g_idx is not None:
                    incumbent_f_idxs.add(g_idx)
                    feature_costs[self._f_idx_to_feature_index[g_idx]] = 0
                    TC.update(g_idx, f_idx)

                # Register ext_states "solved" by fnu_idx
                solved_requirements: List[Tuple[int, intbitset]] = [(i, incumbent_fnu_idxs & self._requirements[i][1]) for i in pending_requirements_good if len(incumbent_fnu_idxs & self._requirements[i][1]) > 0]
                ext_states_to_fnu_idxs.update({self._annotated_requirements[i][0]["ext_state"]: fnu_idxs for i, fnu_idxs in solved_requirements})

                # Recompute pending requirements
                f_idxs: intbitset = intbitset([f_idx] if g_idx is None else [f_idx, g_idx])
                new_pending_requirements_good: List[int] = [i for i in pending_requirements_good if len(incumbent_fnu_idxs & self._requirements[i][1]) == 0]
                new_pending_requirements_other: List[int] = [i for i in pending_requirements_other if len(f_idxs & self._requirements[i][1]) == 0]
                new_pending_requirements: List[int] = new_pending_requirements_good + new_pending_requirements_other
                assert len(new_pending_requirements_good) < len(pending_requirements_good) #, sorted_eligible_items
                pending_requirements_good: List[int] = new_pending_requirements_good
                pending_requirements_other: List[int] = new_pending_requirements_other
                pending_requirements: List[int] = new_pending_requirements

                # Affected fnu pairs
                affected_fnu_idxs: intbitset = self._fnu_idxs_affected_by_fnu_idx(fnu_idx)
            else:
                f_idx: int = random.choice(best_items)
                f_idx_index: int = self._f_idx_to_feature_index[f_idx]
                cost: int = feature_costs[f_idx_index]
                logging.info(f"#eligible={len(eligible_items)}, #best={len(best_items)}, score={best_score}, cost={cost}, f={f_idx}.{self._f_idx_to_feature[f_idx][1]._dlplan_feature}")

                # Extend incumbent sets and revise feature costs
                incumbent_f_idxs.add(f_idx)
                feature_costs[f_idx_index] = 0

                # Recompute pending requirements
                new_pending_requirements_other: List[int] = [i for i in pending_requirements_other if f_idx not in self._requirements[i][1]]
                new_pending_requirements: List[int] = pending_requirements_good + new_pending_requirements_other
                assert len(new_pending_requirements_other) < len(pending_requirements_other)
                pending_requirements_other: List[int] = new_pending_requirements_other
                pending_requirements: List[int] = new_pending_requirements

                # Revise costs of affected fnu pairs
                affected_fnu_idxs: intbitset = self._fnu_idxs_affected_by_f_idxs([f_idx])

            # Revise costs of affected fnu pairs
            for fnu_idx in affected_fnu_idxs:
                f_idx, g_idx = self._f_idxs_for_fnu_idx(fnu_idx)
                f_idx_index: int = self._f_idx_to_feature_index[f_idx]
                g_idx_index: int = self._f_idx_to_feature_index.get(g_idx)
                f_idx_cost: int = feature_costs[f_idx_index]
                g_idx_cost: int = 0 if g_idx is None else feature_costs[g_idx_index]
                revised_cost: int = f_idx_cost + g_idx_cost
                if revised_cost < fnu_pair_costs[fnu_idx]:
                    fnu_pair_costs[fnu_idx] = revised_cost
        assert set(self._requirements_for_good_transitions.keys()).issubset(set(ext_states_to_fnu_idxs.keys()))

        chosen_fnu_idxs: intbitset = intbitset(incumbent_fnu_idxs)
        chosen_f_idxs: intbitset = intbitset(incumbent_f_idxs)
        fnu_idxs_dict: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for fnu_idx in chosen_fnu_idxs:
            f_idx, nu_idx = self._fnu_pairs[fnu_idx]
            fnu_idxs_dict[f_idx].append(self._nu_contexts[nu_idx])
        fnu_idxs_dict: Dict[int, List[Tuple[int, int]]] = {f_idx: sorted(contexts) for f_idx, contexts in fnu_idxs_dict.items()}

        cost = sum([self._f_idx_to_feature[f_idx][1].complexity for f_idx in chosen_f_idxs])
        logging.info(f"Solution: f_idxs={sorted(chosen_f_idxs)}, cost={cost}")

        # Calculate ranks for chosen features
        ranks: Dict[int, int] = TC.calculate_ranks(chosen_f_idxs, unique=True)
        sigma: Dict[Tuple[int, Tuple[int, int]], str] = dict()
        inv_sigma: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for f_idx in ranks.keys():
            for context in fnu_idxs_dict.get(f_idx, []):
                nu_idx: int = self._nu_context_to_index.get(context)
                fnu_idx: int = self._fnu_pair_to_index.get((f_idx, nu_idx))
                direction: str = self._fnu_idx_to_direction[fnu_idx].upper()
                assert direction in ["DEC", "INC"]
                assert not self._monotone_only_by_dec or not self._is_numerical(f_idx) or direction == "DEC"
                sigma[(f_idx, context)] = direction
                inv_sigma[context].append(f_idx)
        logging.info(f"Ranks: {sorted([(f_idx, rank) for f_idx, rank in ranks.items()], key=lambda item: item[1])}")
        logging.info(f"Sigma: {sorted([((f_idx, context), direction) for (f_idx, context), direction in sigma.items()], key=lambda item: ranks.get(item[0][0]))}")
        logging.info(f"Inv(sigma): {inv_sigma}")

        # Calculate decorations
        decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = dict()
        if self._simplify_policy:
            #decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = self._calculate_decorations(chosen_fnu_idxs, ext_states_to_fnu_idxs)
            decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = self._calculate_decorations2(chosen_fnu_idxs, chosen_f_idxs, ext_states_to_fnu_idxs, ranks, sigma, inv_sigma, TC)
        logging.info(f"Decorations: {decorations}")

        local_timer.stop()
        logging.info(f"Greedy solver finished in {local_timer.get_elapsed_sec():0.2f} second(s)")
        return True, chosen_f_idxs, [cost], decorations, ranks

