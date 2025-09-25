import logging
import random
import heapq
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

from .m_counters import MCounters
from .watched_rules import WatchedRules
#from .stratified_policy import StratifiedPolicy, StratifiedPolicyByRules


class GreedySolverByRuleElimination:
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
        assert self._relevant_features is not None

        self._requirements_for_good_transitions: Dict[Tuple[int, int], intbitset] = self._preprocessing_data.get("requirements_for_good_transitions")
        self._goal_ext_pair_to_separating_features: Dict[Tuple[int, Tuple[int, int]], intbitset] = self._preprocessing_data.get("goal_ext_pair_to_separating_features")
        #self._deadend_path_to_separating_features: Dict[Tuple[Tuple[int, int]], List[intbitset]] = self._preprocessing_data.get("deadend_path_to_separating_features")
        self._ext_state_to_separating_features_for_deadend_paths: Dict[Tuple[int, int], Dict[Tuple[int, Tuple[int, int]], intbitset]] = self._preprocessing_data.get("ext_state_to_separating_features_for_deadend_paths")
        self._ext_sibling_to_separating_features: Dict[Tuple[int, int, Tuple[int, int]], intbitset] = self._preprocessing_data.get("ext_sibling_to_separating_features")
        self._ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]] = self._preprocessing_data.get("ext_state_to_ext_edge")
        self._bad_ext_edges: Set[Tuple[int, Tuple[int, int]]] = self._preprocessing_data.get("bad_ext_edges")

        # Monotone features calculated via watched rules
        #self._m_counters: MCounters = self._preprocessing_data.get("m_counters")
        self._watched_rules: MCounters = self._preprocessing_data.get("watched_rules")

        # Calculate numerical features
        self._numerical_features: List[Tuple[int, Feature]] = [(f_idx, feature) for f_idx, feature in self._relevant_features if isinstance(feature.dlplan_feature, dlplan_core.Numerical)]
        self._numerical_f_idxs: intbitset = intbitset([f_idx for f_idx, _ in self._numerical_features])

        # Construct requirements, one per ex-edge and one per pair of goal and non-goal xstates
        self._annotated_requirements: List[Tuple[Dict[str, Any], intbitset]] = []
        self._annotated_requirements.extend([({"key": "Edge", "ext_state": ext_state}, requirement) for ext_state, requirement in self._requirements_for_good_transitions.items()])
        self._annotated_requirements.extend([({"key": "Goal", "pair": pair}, separating_features) for pair, separating_features in self._goal_ext_pair_to_separating_features.items()])
        self._annotated_requirements.extend([({"key": "Deadend", "ext_state": ext_state, "path": path}, separating_features) for ext_state, separating_features_for_deadend_paths in self._ext_state_to_separating_features_for_deadend_paths.items() for path, separating_features in separating_features_for_deadend_paths.items()])
        self._annotated_requirements.extend([({"key": "Sibling", "ext_sibling": ext_sibling}, separating_features) for ext_sibling, separating_features in self._ext_sibling_to_separating_features.items()])
        self._requirements: List[intbitset] = [requirement for _, requirement in self._annotated_requirements]
        self._requirements_for_deadends: List[intbitset] = [requirement for annotation, requirement in self._annotated_requirements if annotation.get("key") == "Deadend"]
        self._num_requirements: Dict[str, int] = {key: sum([1 if annotation.get("key") == key else 0 for annotation, _ in self._annotated_requirements]) for key in ["Edge", "Goal", "Deadend", "Sibling"]}
        logging.info(f"{len(self._requirements)} requirement(s) split as {self._num_requirements}")

        """
        # Support for simplification of policies
        self._ex_ext_states: Set[Tuple[int, int]] = self._preprocessing_data.get("ex_ext_states")
        self._ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = self._preprocessing_data.get("ext_state_to_feature_valuations")
        """

    def _partition_r_idxs_with_f_idx(self, r_idxs: intbitset, f_idx: int) -> Dict[Tuple[Tuple[int, int]], intbitset]:
        partition: Dict[Tuple[Tuple[int, int]], intbitset] = defaultdict(intbitset)
        for r_idx in r_idxs:
            #projection: Tuple[Tuple[int, int]] = self._m_counters.project_condition(r_idx, f_idx, single=True)
            projection: Tuple[Tuple[int, int]] = self._watched_rules.project_condition(r_idx, f_idx, single=True)
            partition[projection].add(r_idx)
        return partition

    def _score_fn(self, f_idx: int, chosen: intbitset) -> Tuple[Union[int, float]]:
        feature_index: int = self._f_idx_to_feature_index[f_idx]
        feature_complexity: int = self._relevant_features[feature_index][1].complexity
        #r_idxs_to_remove: intbitset = intbitset([r_idx for r_idx in self._m_counters.r_idxs() if f_idx in self._m_counters.f_idxs_changed(r_idx)])
        #partition: Dict[Tuple[Tuple[int, int]], intbitset] = self._partition_r_idxs_with_f_idx(self._m_counters.r_idxs(), f_idx)
        r_idxs_to_remove: intbitset = intbitset([r_idx for r_idx in self._watched_rules.r_idxs() if f_idx in self._watched_rules.f_idxs_changed(r_idx)])
        partition: Dict[Tuple[Tuple[int, int]], intbitset] = self._partition_r_idxs_with_f_idx(self._watched_rules.r_idxs(), f_idx)
        block_sizes: List[int] = [len(block - r_idxs_to_remove) for _, block in partition.items() if len(block - r_idxs_to_remove) > 0]
        num_blocks: int = len(block_sizes)
        std_block_sizes: float = -np.std(block_sizes) if num_blocks > 0 else 0
        #monotone_by_dec: int = 1 if self._m_counters.is_monotone(f_idx, monotone_only_by_dec=True) else 0
        monotone_by_dec: int = 1 if self._watched_rules.is_monotone(f_idx, monotone_only_by_dec=True) else 0
        return (monotone_by_dec, len(r_idxs_to_remove) / feature_complexity, 1 if f_idx in chosen else 0, num_blocks, std_block_sizes)

    def _score_fn2(self, f_idx: int, pending_requirements: List[int]) -> Tuple[Union[int, float]]:
        feature_index: int = self._f_idx_to_feature_index[f_idx]
        feature_complexity: int = self._relevant_features[feature_index][1].complexity
        solved_pending_requirements: List[int] = [i for i in pending_requirements if f_idx in self._requirements[i]]
        return (len(solved_pending_requirements) / feature_complexity,)

    def _calculate_decorations(self, solution: intbitset, r_idx_to_info: Dict[int, Tuple[int, intbitset]]) -> Dict[str, Dict[int, Dict[int, intbitset]]]:
        # Necessary f_idxs
        r_idx_to_necessary_f_idxs: Dict[int, List[intbitset]] = defaultdict(list)
        for annotation, requirement in self._annotated_requirements:
            if annotation["key"] == "Deadend":
                ext_state: Tuple[int, int] = annotation["ext_state"]
                #r_idx: int = self._m_counters.get_r_idx(ext_state)
                r_idx: int = self._watched_rules.get_r_idx(ext_state)
                assert r_idx is not None
                necessary_f_idxs: intbitset = requirement & solution
                assert len(necessary_f_idxs) > 0
                r_idx_to_necessary_f_idxs[r_idx].append(necessary_f_idxs)

        if len(r_idx_to_necessary_f_idxs) > 0:
            logging.info(f"r_idx_to_necessary_f_idxs: {r_idx_to_necessary_f_idxs}")

        # Calculate decorations
        decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = {"dont_care": defaultdict(lambda: defaultdict(intbitset)), "unknown": defaultdict(lambda: defaultdict(intbitset))}
        for r_idx, (f_idx, above) in r_idx_to_info.items():
            singleton: intbitset = intbitset([f_idx])
            #instance_idx, state_idx = self._m_counters.get_ext_state(r_idx)
            instance_idx, state_idx = self._watched_rules.get_ext_state(r_idx)
            assert instance_idx not in decorations["unknown"] or state_idx not in decorations["unknown"][instance_idx]
            assert instance_idx not in decorations["dont_care"] or state_idx not in decorations["dont_care"][instance_idx]
            f_idxs_to_remove: intbitset = solution - above - singleton
            f_idxs_to_preserve: intbitset = intbitset()
            list_necessary_f_idxs: List[intbitset] = r_idx_to_necessary_f_idxs.get(r_idx, [])
            for necessary_f_idxs in list_necessary_f_idxs:
                if necessary_f_idxs.issubset(f_idxs_to_remove) and len(necessary_f_idxs & f_idxs_to_preserve) == 0:
                    # Add one f_idx to preserve list
                    f_idxs_to_preserve.add(list(necessary_f_idxs)[0])
            f_idxs_to_remove -= f_idxs_to_preserve
            decorations["dont_care"][instance_idx][state_idx] = f_idxs_to_remove
            if not self._simplify_only_conditions:
                decorations["unknown"][instance_idx][state_idx] = f_idxs_to_remove

        return decorations

    # Solver calls recursive solve
    def solve(self, **kwargs) -> Any:
        # Calculate features that make "eliminate" all rules
        #self._m_counters.reset()
        self._watched_rules.reset()
        r_idx_to_info: Dict[int, Tuple[int, intbitset]] = dict()
        terminating_set: intbitset = self._rec_solve(0, intbitset(), intbitset(), r_idx_to_info, **kwargs)

        # Solve pending requirements
        solution, _ = self._solve_pending_requirements(terminating_set)

        # Cost of solution
        cost = sum([self._relevant_features[self._f_idx_to_feature_index[f_idx]][1].complexity for f_idx in solution])

        # Decorations
        decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = dict()
        if self._simplify_policy:
            decorations = self._calculate_decorations(solution, r_idx_to_info)

        return True, solution, [cost], decorations, None

    def _rec_solve(self,
                   depth: int,
                   chosen: intbitset,
                   above: intbitset,
                   r_idx_to_info: Dict[int, Tuple[int, intbitset]],
                   **kwargs) -> intbitset:
        indent_str: str = '   ' * depth

        # Base case: if no remaining rules, empty set of features is solution
        #r_idxs: intbitset = self._m_counters.r_idxs()
        r_idxs: intbitset = self._watched_rules.r_idxs()
        if len(r_idxs) == 0: return intbitset()

        # Get eligible features sorted by score, removing features that do not eliminate any rule (i.e., with zero score in corresponding component)
        #eligible_features: intbitset = self._m_counters.monotone_features(kwargs.get("monotone_only_by_dec", False))
        eligible_features: intbitset = self._watched_rules.monotone_features(kwargs.get("monotone_only_by_dec", False))
        logging.info(f"[_rec_solve]:{indent_str} => chosen={list(chosen)}, r_idxs={list(r_idxs)}, {len(eligible_features)} mononote feature(s)")
        eligible_features_with_score: List[Tuple[int, Tuple[float]]] = [(f_idx, self._score_fn(f_idx, chosen)) for f_idx in eligible_features]
        eligible_features_with_non_zero_score: List[Tuple[int, Tuple[float]]] = [(f_idx, score) for f_idx, score in eligible_features_with_score if score[1] > 0]
        sorted_eligible_features: List[Tuple[int, Tuple[float]]] = sorted(eligible_features_with_non_zero_score, key=lambda item: item[1], reverse=True)

        # Check for early termination due to non-existence of solution
        if len(sorted_eligible_features) == 0:
            logging.warning(f"No monotone feature to eliminate rules: r_idxs={sorted(r_idxs)}")
            for r_idx in r_idxs:
                #ext_state: Tuple[int, int] = self._m_counters.get_ext_state(r_idx)
                ext_state: Tuple[int, int] = self._watched_rules.get_ext_state(r_idx)
                ext_edge: Tuple[int, Tuple[int, int]] = self._ext_state_to_ext_edge.get(ext_state)
                src_dlplan_state: dlplan_core.State = self._state_factory.get_dlplan_state(ext_edge[0], ext_edge[1][0])
                dst_dlplan_state: dlplan_core.State = self._state_factory.get_dlplan_state(ext_edge[0], ext_edge[1][1])
                logging.warning(f"  r_idx={r_idx}, ext_edge={ext_edge}")
                logging.warning(f"    src_state: {ext_edge[1][0]}.{src_dlplan_state}")
                logging.warning(f"    dst_state: {ext_edge[1][1]}.{dst_dlplan_state}")
                requirement_for_r_idx: intbitset = None
                for annotation, requirement in self._annotated_requirements:
                    if annotation["key"] == "Edge" and annotation["ext_state"] == ext_state:
                        requirement_for_f_idx = requirement
                        break
                logging.warning(f"    requirement: {requirement and sorted(requirement)}")
            raise RuntimeError(f"No eligible features")

        # Choose a best eligible feature
        best_f_idxs: List[Tuple[int]] = [f_idx for f_idx, score in sorted_eligible_features if score == sorted_eligible_features[0][1]]
        best_f_idx: int = random.choice(best_f_idxs)
        best_score: Tuple[float] = self._score_fn(best_f_idx, chosen)
        complexity: int = self._relevant_features[self._f_idx_to_feature_index[best_f_idx]][1].complexity
        #r_idxs_to_remove: intbitset = intbitset([r_idx for r_idx in r_idxs if best_f_idx in self._m_counters.f_idxs_changed(r_idx)])
        r_idxs_to_remove: intbitset = intbitset([r_idx for r_idx in r_idxs if best_f_idx in self._watched_rules.f_idxs_changed(r_idx)])
        partition: Dict[Tuple[Tuple[int, int]], intbitset] = self._partition_r_idxs_with_f_idx(r_idxs, best_f_idx)
        logging.info(f"[_rec_solve]:{indent_str}    f{best_f_idx}.{self._relevant_features[self._f_idx_to_feature_index[best_f_idx]][1]._dlplan_feature}/{complexity}, score={best_score}, r_idxs_to_remove={r_idxs_to_remove}, partition={[len(block - r_idxs_to_remove) for block in partition.values()]}")
        assert len(r_idxs_to_remove) > 0

        # Calculate info
        for r_idx in r_idxs_to_remove:
            r_idx_to_info[r_idx] = (best_f_idx, intbitset(above))

        # Remove rules that change chosen feature
        for r_idx in r_idxs_to_remove:
            #self._m_counters.remove_rule(r_idx)
            self._watched_rules.remove_rule(r_idx)

        # Recursion on each partition block
        singleton: intbitset = intbitset([best_f_idx])
        solution: intbitset = intbitset(singleton)
        for key, block in partition.items():
            # Remove r_idxs not in partition block before recursive call
            r_idxs_not_in_block: intbitset = r_idxs - block
            for r_idx in r_idxs_not_in_block:
                #self._m_counters.remove_rule(r_idx)
                self._watched_rules.remove_rule(r_idx)

            # Recursive call
            solution_for_block: intbitset = self._rec_solve(1 + depth, chosen | solution, above | singleton, r_idx_to_info, **kwargs)
            solution |= solution_for_block

            # Restore r_idxs not in partition block after recursive call
            for r_idx in r_idxs_not_in_block:
                #self._m_counters.add_rule(r_idx)
                self._watched_rules.restore_rule(r_idx)

        # Restore r_idxs that change best f_idx
        for r_idx in r_idxs_to_remove:
            #self._m_counters.add_rule(r_idx)
            self._watched_rules.restore_rule(r_idx)

        logging.info(f"[_rec_solve]:{indent_str} solution={list(solution)}")
        return solution

    def _solve_pending_requirements(self, f_idxs: intbitset) -> Tuple[intbitset, intbitset]:
        solution: intbitset = intbitset(f_idxs)
        pending_requirements: List[int] = [i for i, requirement in enumerate(self._requirements) if len(requirement & solution) == 0]
        while len(pending_requirements) > 0:
            logging.info(f"[_solve_pending_requirements] pending_requirements: {pending_requirements}")
            eligible_features_with_score: List[Tuple[int, Tuple[float]]] = [(f_idx, self._score_fn2(f_idx, pending_requirements)) for f_idx, _ in self._relevant_features if f_idx not in solution]
            eligible_features_with_non_zero_score: List[Tuple[int, Tuple[float]]] = [(f_idx, score) for f_idx, score in eligible_features_with_score if score[0] > 0]
            sorted_eligible_features: List[Tuple[int, Tuple[float]]] = sorted(eligible_features_with_non_zero_score, key=lambda item: item[1], reverse=True)

            if len(sorted_eligible_features) == 0:
                logging.info("No feature for solving pending requirements:")
                for annotation, requirement in [self._annotated_requirements[i] for i in pending_requirements]:
                    if annotation["key"] == "Edge":
                        ext_state: Tuple[int, int] = annotation["ext_state"]
                        #r_idx: int = self._m_counters.get_r_idx(ext_state)
                        r_idx: int = self._watched_rules.get_r_idx(ext_state)
                        logging.info(f"  Unexpected pending requirement: key='Edge', ext_state={ext_state}, r_idx={r_idx}")
                        logging.info(f"    This should have detected during rule elimination by _rec_solve")
                    elif annotation["key"] == "Goal":
                        pair: Any = annotation["pair"]
                        logging.info(f"  Goal requirement: pair={pair}, requirement={sorted(requirement)}")
                    elif annotation["key"] == "Deadend":
                        ext_state: Tuple[int, int] = annotation["ext_state"]
                        path: path = annotation["path"]
                        logging.info(f"  Deadend requirement: ext_state={ext_state}, path={path}, requirement={sorted(requirement)}")
                    elif annotation["key"] == "Sibling":
                        ext_sibling: Tuple[int, int] = annotation["ext_sibling"]
                        logging.info(f"  Sibling requirement: ext_sibling={ext_sibling}, requirement={sorted(requirement)}")
                    else:
                        raise RuntimeError(f"Unknown requirement key '{annotation['key']}'")
                raise RuntimeError("No feature for solving pending requirements")

            # Choose a best eligible feature
            best_f_idxs: List[Tuple[int]] = [f_idx for f_idx, score in sorted_eligible_features if score == sorted_eligible_features[0][1]]
            best_f_idx: int = random.choice(best_f_idxs)
            best_score: Tuple[float] = self._score_fn2(best_f_idx, pending_requirements)
            complexity: int = self._relevant_features[self._f_idx_to_feature_index[best_f_idx]][1].complexity
            logging.info(f"[_solve_pending_requirements] f{best_f_idx}.{self._relevant_features[self._f_idx_to_feature_index[best_f_idx]][1]._dlplan_feature}/{complexity}, score={best_score}")

            solution.add(best_f_idx)
            pending_requirements_reduced: List[int] = [i for i in pending_requirements if len(self._requirements[i] & solution) == 0]
            assert len(pending_requirements_reduced) < len(pending_requirements)
            pending_requirements = pending_requirements_reduced

        if len(solution - f_idxs) > 0:
            logging.info(f"[_solve_pending_requirements] added={sorted(solution - f_idxs)}")

        return solution, solution - f_idxs

