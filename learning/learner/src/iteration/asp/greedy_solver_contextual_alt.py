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
from .stratified_policy import StratifiedPolicy, StratifiedPolicyByFeaturesContextual
from .transitive_closure import TransitiveClosure


class GreedySolverContextualAlt:
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
        self._f_idx_to_feature_index: Dict[int, int] = self._preprocessing_data.get("f_idx_to_feature_index")
        self._f_idx_to_feature: Dict[int, Tuple[int, Feature]] = {f_idx: (f_idx_index, feature) for f_idx_index, (f_idx, feature) in enumerate(self._relevant_features)}
        self._feature_index_to_f_idx: List[int] = [f_idx for f_idx, _ in self._relevant_features]

        # Pairs and contexts
        self._nu_context_to_index: OrderedDict[Tuple[int, int], int] = self._preprocessing_data.get("nu_context_to_index")
        self._nu_contexts: List[Tuple[int, int]] = list(self._nu_context_to_index.keys())
        self._fnu_pair_to_index: Dict[Tuple[int, Tuple[int]], int] = self._preprocessing_data.get("fnu_pair_to_index")
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
        self._bad_ext_edges: Set[Tuple[int, Tuple[int, int]]] = self._preprocessing_data.get("bad_ext_edges")

        self._m_pairs: MPairsContextual = self._preprocessing_data.get("m_pairs")
        self._monotone_features: intbitset = self._m_pairs.monotone_features()
        #XXX self._ex_ext_states: Set[Tuple[int, int]] = self._preprocessing_data.get("ex_ext_states")

        # Calculate numerical features
        self._numerical_features: List[Tuple[int, Feature]] = [(f_idx, feature) for f_idx, feature in self._relevant_features if isinstance(feature.dlplan_feature, dlplan_core.Numerical)]
        self._numerical_f_idxs: intbitset = intbitset([f_idx for f_idx, _ in self._numerical_features])

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
        self._ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = self._preprocessing_data.get("ext_state_to_feature_valuations")

    def _is_numerical(self, f_idx: int) -> bool:
        return f_idx in self._numerical_f_idxs

    def _f_idxs_for_fnu_idx(self, fnu_idx: int) -> Tuple[int, int]:
        f_idx, nu_idx = self._fnu_pairs[fnu_idx]
        nu_context: Tuple[int, int] = self._nu_contexts[nu_idx]
        g_idx: int = None if len(nu_context) == 0 else nu_context[0]
        return (f_idx, g_idx)

    def _revise_costs_and_chains(self,
                                 f_idxs: intbitset,
                                 feature_costs: List[int],
                                 feature_chains: List[Tuple[Tuple[int, Tuple[int]]]]) -> int:
        num_revisions: int = 0
        q: deque = deque(f_idxs)
        while len(q) > 0:
            g_idx: int = q.popleft()
            g_idx_index: int = self._f_idx_to_feature_index[g_idx]
            g_idx_complexity: int = self._relevant_features[g_idx_index][1].complexity
            g_idx_cost: int = feature_costs[g_idx_index]
            g_idx_chain: Tuple[Tuple[int, Tuple[int]]] = feature_chains[g_idx_index]
            assert g_idx_cost < int(1e6) and g_idx_chain is not None
            logging.debug(f"Dequeued: g_idx={g_idx}{'*' if g_idx in self._monotone_features else ''}, complexity={g_idx_complexity}, cost={g_idx_cost}")

            f_idxs_01: Tuple[intbitset, intbitset] = self._m_pairs.f_idxs_for_g_idx(g_idx)
            for value, f_idxs in enumerate(f_idxs_01):
                nu_context: Tuple[int, int] = (g_idx, value)
                nu_context_idx: int = self._nu_context_to_index.get(nu_context)
                if nu_context_idx is not None:
                    for f_idx in f_idxs:
                        f_idx_index: int = self._f_idx_to_feature_index[f_idx]
                        f_idx_complexity: int = self._relevant_features[f_idx_index][1].complexity
                        f_idx_cost: int = feature_costs[f_idx_index]
                        new_f_idx_cost: int = g_idx_cost + f_idx_complexity
                        new_f_idx_chain: Tuple[Tuple[int, Tuple[int]]] = g_idx_chain + ((f_idx, nu_context),)
                        if new_f_idx_cost < f_idx_cost:
                            num_revisions += 1
                            feature_costs[f_idx_index] = new_f_idx_cost
                            feature_chains[f_idx_index] = new_f_idx_chain
                            q.append(f_idx)
                            logging.debug(f"  Revise cost of f_idx={f_idx}{'*' if f_idx in self._monotone_features else ''} from {f_idx_cost} to {new_f_idx_cost} ; new_chain={new_f_idx_chain}")
        return num_revisions

    def _eligible_feature(self,
                          f_idx_index: int,
                          feature_costs: List[int],
                          feature_chains: List[Tuple[Tuple[int, Tuple[int]]]],
                          TC: TransitiveClosure) -> bool:
        cost: int = feature_costs[f_idx_index]
        chain: Tuple[Tuple[int, Tuple[int]]] = feature_chains[f_idx_index]
        return cost > 0 and chain is not None and TC.acyclic_if_path_added([f_idx for f_idx, _ in chain])

    def _score_fn(self,
                  f_idx_index: int,
                  pending_requirements_fnu_idxs: List[int],
                  pending_requirements_f_idxs: List[int],
                  feature_costs: List[int],
                  feature_chains: List[Tuple[Tuple[int, Tuple[int]]]]) -> Tuple[float]:
        chain: Tuple[Tuple[int, Tuple[int]]] = feature_chains[f_idx_index]
        chain_cost: int = sum([feature_costs[self._f_idx_to_feature_index.get(g_idx)] for g_idx, _ in chain])
        fnu_idxs_in_chain: intbitset = intbitset([self._fnu_pair_to_index.get((g_idx, self._nu_context_to_index.get(context))) for g_idx, context in chain])
        g_idxs_in_chain: intbitset = intbitset([g_idx for g_idx, _ in chain])
        assert all([fnu_idx is not None for fnu_idx in fnu_idxs_in_chain])
        solved_requirements: List[int] = [i for i in pending_requirements_fnu_idxs if len(fnu_idxs_in_chain & self._requirements[i][1]) > 0] + [i for i in pending_requirements_f_idxs if len(g_idxs_in_chain & self._requirements[i][1]) > 0]
        if len(solved_requirements) == 0:
            return (0,)
        elif chain_cost == 0:
            return (1e6,)
        else:
            numerical_f_idx_index: int = 1 if self._feature_index_to_f_idx[f_idx_index] in self._numerical_f_idxs else 0
            return (len(solved_requirements) / chain_cost, numerical_f_idx_index)

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
        #logging.info(f"f_idxs={sorted(chosen_f_idxs)}, numerical={sorted(chosen_f_idxs & self._numerical_f_idxs)}")

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

    def _calculate_decorations3(self,
                                chosen_fnu_idxs: intbitset,
                                chosen_f_idxs: intbitset,
                                ext_states_to_fnu_idxs: Dict[Tuple[int, int], intbitset],
                                ranks: Dict[int, int],
                                sigma: Dict[Tuple[int, Tuple[int, int]], str],
                                inv_sigma: Dict[Tuple[int, int], List[int]],
                                TC: TransitiveClosure) -> Dict[str, Dict[int, Dict[int, intbitset]]]:
        policy: StratifiedPolicy = StratifiedPolicyByFeaturesContextual(chosen_f_idxs, self._numerical_f_idxs, sigma, self._ext_state_to_ext_edge, self._ext_state_to_feature_valuations, self._bad_ext_edges)
        decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = policy.calculate_decorations(self._simplify_only_conditions)
        return decorations

    # Solver that at each iteration solves a requiement. Number of iteration is thus bounded by number of requirements.
    def solve(self, **kwargs) -> Tuple[bool, intbitset, int, Dict[str, Dict[int, Dict[int, intbitset]]], int]:
        logging.info(f"Starting greedy solver...")
        local_timer: Timer = Timer()

        # Define costs, chains, and transitive closure
        feature_costs: List[int] = [int(1e6) for _ in self._relevant_features]
        feature_chains: List[Tuple[Tuple[int, Tuple[int]]]] = [None for _ in self._relevant_features]
        TC: TransitiveClosure = TransitiveClosure()

        # Initialize costs and chains for monotone features, and propagate
        logging.info(f"Constructing chains for {len(self._monotone_features)} monotone feature(s) using dynamic programming...")
        empty_context_idx: int = self._nu_context_to_index.get(())
        assert empty_context_idx is not None, "Index for empty context not found"
        for f_idx in self._monotone_features:
            f_idx_index: int = self._f_idx_to_feature_index[f_idx]
            f_idx_complexity = self._relevant_features[f_idx_index][1].complexity
            fnu_idx: int = self._fnu_pair_to_index.get((f_idx, empty_context_idx))
            assert fnu_idx is not None, f"Index for fnu pair {(f_idx, empty_context_idx)} not found"
            feature_costs[f_idx_index] = f_idx_complexity
            feature_chains[f_idx_index] = ((f_idx, ()),)
        num_revisions: int = self._revise_costs_and_chains(self._monotone_features, feature_costs, feature_chains)
        logging.info(f"{num_revisions} cost revision(s)")

        # Grow incumbent set until all requirements are fulfilled
        incumbent_fnu_idxs: intbitset = intbitset()
        incumbent_f_idxs: intbitset = intbitset()
        pending_requirements_fnu_idxs: List[int] = [i for i, (label, _) in enumerate(self._requirements) if label == "Good"]
        pending_requirements_f_idxs: List[int] = [i for i, (label, _) in enumerate(self._requirements) if label != "Good"]
        pending_requirements: List[int] = pending_requirements_fnu_idxs + pending_requirements_f_idxs
        ext_states_to_fnu_idxs: Dict[Tuple[int, int], intbitset] = dict()
        while len(pending_requirements) > 0:
            logging.info(f"{len(pending_requirements)} pending requirement(s): fnu_idxs={sorted(pending_requirements_fnu_idxs)}, f_idxs={sorted(pending_requirements_f_idxs)}")
            timer: Timer = Timer()

            # Eligible features are those whose chains do not create a dependency loop
            eligible_features: List[int] = [f_idx_index for f_idx_index in range(len(self._relevant_features)) if self._eligible_feature(f_idx_index, feature_costs, feature_chains, TC)]

            # Sort eligible features by score
            eligible_features_with_score: List[Tuple[int, Tuple[float]]] = [(f_idx_index, self._score_fn(f_idx_index, pending_requirements_fnu_idxs, pending_requirements_f_idxs, feature_costs, feature_chains)) for f_idx_index in eligible_features]
            eligible_features_with_score: List[Tuple[int, Tuple[float]]] = [(f_idx_index, score) for (f_idx_index, score) in eligible_features_with_score if score != (0,)]
            sorted_eligible_features: List[Tuple[int, Tuple[float]]] = sorted(eligible_features_with_score, key=lambda item: item[1], reverse=True)
            logging.info(f"{len(sorted_eligible_features)} eligible feature(s) computed in {timer.get_elapsed_sec():0.2f} second(s)")

            # Check for early termination due to non-existence of solution
            if len(sorted_eligible_features) == 0:
                logging.warning(f"No eligible features: incumbent_fnu_idxs={sorted(incumbent_fnu_idxs)}")
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
            best_score: Tuple[float] = sorted_eligible_features[0][1]
            best_features: List[int] = [f_idx_index for f_idx_index, score in sorted_eligible_features if score == best_score]

            f_idx_index: int = random.choice(best_features)
            f_idx: int = self._feature_index_to_f_idx[f_idx_index]
            f_idx_cost: int = feature_costs[f_idx_index]
            f_idx_chain: Tuple[Tuple[int, Tuple[int]]] = feature_chains[f_idx_index]
            logging.info(f"#eligible={len(sorted_eligible_features)}, #best={len(best_features)}, score={best_score}, f={f_idx}.{self._f_idx_to_feature[f_idx][1]._dlplan_feature}, cost={f_idx_cost}, chain={f_idx_chain}")

            # Extend incumbent sets
            fnu_idxs_in_chain: intbitset = intbitset([self._fnu_pair_to_index.get((f_idx, self._nu_context_to_index.get(context))) for f_idx, context in f_idx_chain])
            assert all([fnu_idx is not None for fnu_idx in fnu_idxs_in_chain])
            g_idxs_in_chain: intbitset = intbitset([g_idx for g_idx, _ in f_idx_chain])
            incumbent_fnu_idxs |= intbitset(fnu_idxs_in_chain)
            incumbent_f_idxs |= g_idxs_in_chain

            # Update TC
            assert f_idx_chain is not None
            f_idx_path: List[int] = [f_idx for f_idx, _ in f_idx_chain]
            TC.update_with_path(f_idx_path)
            logging.info(f"Edges in TC: {list(TC.edges())}")

            # Revise costs
            revised_g_idxs: intbitset = intbitset()
            for g_idx, _ in f_idx_chain:
                g_idx_index: int = self._f_idx_to_feature_index[g_idx]
                g_idx_cost = feature_costs[g_idx_index]
                assert g_idx_cost < int(1e6)
                if g_idx_cost > 0:
                    feature_costs[g_idx_index] = 0
                    revised_g_idxs.add(g_idx)
            num_revisions: int = self._revise_costs_and_chains(revised_g_idxs, feature_costs, feature_chains)
            logging.info(f"{num_revisions} cost revision(s)")

            # Register ext_states "solved" by fnu_idxs
            for i in pending_requirements_fnu_idxs:
                fnu_idxs: intbitset = fnu_idxs_in_chain & self._requirements[i][1]
                if len(fnu_idxs) > 0:
                    ext_state: Tuple[int, int] = self._annotated_requirements[i][0]["ext_state"]
                    assert ext_state not in ext_states_to_fnu_idxs
                    ext_states_to_fnu_idxs[ext_state] = fnu_idxs

            # Recompute pending requirements
            pending_requirements_fnu_idxs: List[int] = [i for i in pending_requirements_fnu_idxs if len(fnu_idxs_in_chain & self._requirements[i][1]) == 0]
            pending_requirements_f_idxs: List[int] = [i for i in pending_requirements_f_idxs if len(g_idxs_in_chain & self._requirements[i][1]) == 0]
            pending_requirements: List[int] = pending_requirements_fnu_idxs + pending_requirements_f_idxs

        # Check that every rule is covered by some fnu pair
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
            decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = self._calculate_decorations3(chosen_fnu_idxs, chosen_f_idxs, ext_states_to_fnu_idxs, ranks, sigma, inv_sigma, TC)
        logging.info(f"Decorations: {decorations}")

        local_timer.stop()
        logging.info(f"Greedy solver finished in {local_timer.get_elapsed_sec():0.2f} second(s)")
        return True, chosen_f_idxs, [cost], decorations, ranks

