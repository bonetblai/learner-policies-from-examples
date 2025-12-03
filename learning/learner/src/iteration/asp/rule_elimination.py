import os
import logging
import random
import heapq
import numpy as np

# Bitset-based unordered sets of unsigned integers
from intbitset import intbitset

from pathlib import Path
from termcolor import colored
from typing import Set, Tuple, List, Union, Dict, Any, Optional, Union, Generator
from collections import OrderedDict, defaultdict, deque
from itertools import product

import dlplan.core as dlplan_core

from ..feature_pool import Feature
from ...util import Timer
from ...state_space import StateFactory

from .rule_viewer import RuleViewer
from .policy_finalizer import PolicyFinalizer
from .solution_generator import SolutionGenerator


def _partition_r_idxs_with_f_idx(r_idxs: intbitset, f_idx: int, viewer: RuleViewer) -> Dict[Tuple[Tuple[int, int]], intbitset]:
    partition: Dict[Tuple[Tuple[int, int]], intbitset] = defaultdict(intbitset)
    for r_idx in r_idxs:
        projection: Tuple[Tuple[int, int]] = viewer.project_condition(r_idx, f_idx)
        partition[projection].add(r_idx)
    return partition


class RuleElimination:
    def __init__(self, benchmark: Any, preprocessing_data: Dict[str, Any], non_sound_ext_states: Set[Tuple[int, int]], **kwargs):
        self._benchmark: Any = benchmark
        self._preprocessing_data: Dict[str, Any] = preprocessing_data
        self._non_sound_ext_states: Set[Tuple[int, int]] = non_sound_ext_states
        self._simplify_policy: bool = kwargs.get("simplify_policy", False)
        self._simplify_only_conditions: bool = kwargs.get("simplify_only_conditions", False)
        self._uniform_costs: bool = kwargs.get("uniform_costs", False)
        self._monotone_only_by_dec: bool = kwargs.get("monotone_only_by_dec", False)
        assert self._preprocessing_data is not None

        self._requirements_for_good_transitions: Dict[Tuple[int, int], intbitset] = self._preprocessing_data.get("requirements_for_good_transitions")
        self._goal_ext_pair_to_separating_features: Dict[Tuple[int, Tuple[int, int]], intbitset] = self._preprocessing_data.get("goal_ext_pair_to_separating_features")
        #self._deadend_path_to_separating_features: Dict[Tuple[Tuple[int, int]], List[intbitset]] = self._preprocessing_data.get("deadend_path_to_separating_features")
        self._ext_state_to_separating_features_for_deadend_paths: Dict[Tuple[int, int], Dict[Tuple[int, Tuple[int, int]], intbitset]] = self._preprocessing_data.get("ext_state_to_separating_features_for_deadend_paths")
        self._ext_sibling_to_separating_features: Dict[Tuple[int, int, Tuple[int, int]], intbitset] = self._preprocessing_data.get("ext_sibling_to_separating_features")
        self._ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]] = self._preprocessing_data.get("ext_state_to_ext_edge")

        # Features
        self._relevant_features: List[Tuple[int, Feature]] = self._preprocessing_data.get("relevant_features")
        self._f_idx_to_feature_index: Dict[int, int] = self._preprocessing_data.get("f_idx_to_feature_index")
        self._numerical_features: List[Tuple[int, Feature]] = [(f_idx, feature) for f_idx, feature in self._relevant_features if isinstance(feature.dlplan_feature, dlplan_core.Numerical)]
        self._numerical_f_idxs: intbitset = intbitset([f_idx for f_idx, _ in self._numerical_features])

        # Rule viewer
        self._viewer: RuleViewer = self._preprocessing_data.get("rule_viewer")

        # Construct requirements, one per ex-edge and one per pair of goal and non-goal xstates
        self._annotated_requirements: List[Tuple[Dict[str, Any], intbitset]] = []
        self._annotated_requirements.extend([({"key": "Edge", "ext_state": ext_state}, requirement) for ext_state, requirement in self._requirements_for_good_transitions.items()])
        self._annotated_requirements.extend([({"key": "Goal", "pair": pair}, separating_features) for pair, separating_features in self._goal_ext_pair_to_separating_features.items()])
        self._annotated_requirements.extend([({"key": "Deadend", "ext_state": ext_state, "path": path}, separating_features) for ext_state, separating_features_for_deadend_paths in self._ext_state_to_separating_features_for_deadend_paths.items() for path, separating_features in separating_features_for_deadend_paths.items()])
        self._annotated_requirements.extend([({"key": "Sibling", "ext_sibling": ext_sibling}, separating_features) for ext_sibling, separating_features in self._ext_sibling_to_separating_features.items()])
        self._requirements: List[intbitset] = [requirement for _, requirement in self._annotated_requirements]
        self._requirements_for_goals: List[intbitset] = [requirement for annotation, requirement in self._annotated_requirements if annotation.get("key") == "Goal"]
        self._requirements_for_deadends: List[intbitset] = [requirement for annotation, requirement in self._annotated_requirements if annotation.get("key") == "Deadend"]
        self._num_requirements: Dict[str, int] = {key: sum([1 if annotation.get("key") == key else 0 for annotation, _ in self._annotated_requirements]) for key in ["Edge", "Goal", "Deadend", "Sibling"]}
        logging.info(f"{len(self._requirements)} requirement(s) split as {self._num_requirements}")

        # Policy finalizer
        self._finalizer_options: Dict[str, Any] = {
            "simplify_policy": kwargs.get("simplify_policy"),
            "simplify_only_conditions": kwargs.get("simplify_only_conditions", False), # for naive simplification
            "threshold_for_asp_based_simplification": kwargs.get("threshold_for_asp_based_simplification"),
            "solve_pending_requirements": kwargs.get("solve_pending_requirements", False),
            "hard_constraints": kwargs.get("hard_constraints", False),
            "dump_asp_program": kwargs.get("dump_asp_program", False),
        }
        self._finalizer: PolicyFinalizer = PolicyFinalizer(self._benchmark,
                                                           self._preprocessing_data,
                                                           self._viewer._r_idx_to_ext_state,
                                                           self._viewer._ext_state_to_r_idx,
                                                           self._annotated_requirements,
                                                           self._non_sound_ext_states,
                                                           **self._finalizer_options)

    def _score_fn(self, f_idx: int, chosen: intbitset) -> Tuple[Union[int, float]]:
        feature_index: int = self._f_idx_to_feature_index[f_idx]
        feature_complexity: int = self._relevant_features[feature_index][1].complexity - 1
        r_idxs_to_remove: intbitset = intbitset([r_idx for r_idx in self._viewer.r_idxs() if f_idx in self._viewer.f_idxs_changed_by(r_idx)])
        partition: Dict[Tuple[Tuple[int, int]], intbitset] = _partition_r_idxs_with_f_idx(self._viewer.r_idxs(), f_idx, self._viewer)
        block_sizes: List[int] = [len(block - r_idxs_to_remove) for _, block in partition.items() if len(block - r_idxs_to_remove) > 0]
        num_blocks: int = len(block_sizes)
        std_block_sizes: float = -np.std(block_sizes) if num_blocks > 0 else 0
        monotone_by_dec: int = 1 if self._viewer.is_monotone(f_idx, monotone_only_by_dec=True) else 0
        return (monotone_by_dec, len(r_idxs_to_remove) / feature_complexity, 1 if f_idx in chosen else 0, num_blocks, std_block_sizes) #, -feature_complexity)

    def _rec_solve_v1(self,
                      branch: List[int],
                      chosen: intbitset,
                      above: intbitset,
                      r_idx_to_info: Dict[int, Tuple[int, intbitset]],
                      **kwargs) -> intbitset:
        indent_str: str = '   ' * len(branch)

        # Base case: if no remaining rules, empty set of features is solution
        r_idxs: intbitset = self._viewer.r_idxs()
        if len(r_idxs) == 0: return intbitset()

        # Sort eligible features by score, removing features that do not eliminate any rule (i.e., with zero score in corresponding component)
        eligible_features: intbitset = self._viewer.monotone_features(kwargs.get("monotone_only_by_dec", False))
        logging.info(f"[_rec_solve_v1]:{indent_str} => chosen={list(chosen)}, r_idxs={list(r_idxs)}, {len(eligible_features)} mononote feature(s)")
        eligible_features_with_score: List[Tuple[int, Tuple[float]]] = [(f_idx, self._score_fn(f_idx, chosen)) for f_idx in eligible_features]
        eligible_features_with_non_zero_score: List[Tuple[int, Tuple[float]]] = [(f_idx, score) for f_idx, score in eligible_features_with_score if score[1] > 0]
        sorted_eligible_features: List[Tuple[int, Tuple[float]]] = sorted(eligible_features_with_non_zero_score, key=lambda item: item[1], reverse=True)

        # Check for early termination due to non-existence of solution
        if len(sorted_eligible_features) == 0:
            logging.warning(f"No monotone feature to eliminate rules: r_idxs={sorted(r_idxs)}")
            for r_idx in r_idxs:
                ext_state: Tuple[int, int] = self._viewer.get_ext_state(r_idx)
                ext_edge: Tuple[int, Tuple[int, int]] = self._ext_state_to_ext_edge.get(ext_state)
                src_dlplan_state: dlplan_core.State = self._benchmark.get_dlplan_state(ext_edge[0], ext_edge[1][0])
                dst_dlplan_state: dlplan_core.State = self._benchmark.get_dlplan_state(ext_edge[0], ext_edge[1][1])
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
        complexity: int = self._relevant_features[self._f_idx_to_feature_index[best_f_idx]][1].complexity - 1
        r_idxs_to_remove: intbitset = intbitset([r_idx for r_idx in r_idxs if best_f_idx in self._viewer.f_idxs_changed_by(r_idx)])
        partition: Dict[Tuple[Tuple[int, int]], intbitset] = _partition_r_idxs_with_f_idx(r_idxs, best_f_idx, self._viewer)
        logging.info(f"[_rec_solve_v1]:{indent_str}    f{best_f_idx}.{self._relevant_features[self._f_idx_to_feature_index[best_f_idx]][1]._dlplan_feature}/{complexity}, score={best_score}, r_idxs_to_remove={r_idxs_to_remove}, partition={[len(block - r_idxs_to_remove) for block in partition.values()]}")
        assert len(r_idxs_to_remove) > 0

        # Calculate info
        for r_idx in r_idxs_to_remove:
            r_idx_to_info[r_idx] = (best_f_idx, intbitset(above))

        # Remove rules that change chosen feature
        for r_idx in r_idxs_to_remove:
            self._viewer.remove_rule(r_idx)
        alive_r_idxs: intbitset = r_idxs - r_idxs_to_remove

        # Recursion on each partition block
        singleton: intbitset = intbitset([best_f_idx])
        solution: intbitset = intbitset(singleton)
        for block in partition.values():
            # Remove r_idxs not in partition block before recursive call
            r_idxs_not_in_block: intbitset = alive_r_idxs - block
            for r_idx in r_idxs_not_in_block:
                self._viewer.remove_rule(r_idx)

            # Recursive call
            solution_for_block = self._rec_solve_v1(branch + [best_f_idx], chosen | solution, above | singleton, r_idx_to_info, **kwargs)
            solution |= solution_for_block

            # Restore r_idxs not in partition block after recursive call
            for r_idx in r_idxs_not_in_block:
                self._viewer.restore_rule(r_idx)

        # Restore r_idxs that change best f_idx
        for r_idx in r_idxs_to_remove:
            self._viewer.restore_rule(r_idx)

        logging.info(f"[_rec_solve_v1]:{indent_str} solution={list(solution)}")
        return solution

    def _rec_solve_v2(self,
                      branch: List[int],
                      chosen: intbitset,
                      above: intbitset,
                      r_idx_to_info: Dict[int, Tuple[int, intbitset]],
                      **kwargs) -> intbitset:
        indent_str: str = '   ' * len(branch)

        # Base case: if no remaining rules, empty set of features is solution
        r_idxs: intbitset = self._viewer.r_idxs()
        if len(r_idxs) == 0: return intbitset()

        # Sort eligible features by score, removing features that do not eliminate any rule (i.e., with zero score in corresponding component)
        eligible_features: intbitset = self._viewer.monotone_features(kwargs.get("monotone_only_by_dec", False))
        logging.info(f"[_rec_solve_v2]:{indent_str} => chosen={list(chosen)}, r_idxs={list(r_idxs)}, {len(eligible_features)} mononote feature(s)")
        eligible_features_with_score: List[Tuple[int, Tuple[float]]] = [(f_idx, self._score_fn(f_idx, chosen)) for f_idx in eligible_features]
        eligible_features_with_non_zero_score: List[Tuple[int, Tuple[float]]] = [(f_idx, score) for f_idx, score in eligible_features_with_score if score[1] > 0]
        random.shuffle(eligible_features_with_non_zero_score)
        sorted_eligible_features: List[Tuple[int, Tuple[float]]] = sorted(eligible_features_with_non_zero_score, key=lambda item: item[1], reverse=True)

        # Try eligible feature in order by score, returning first solution found
        for f_idx, score in sorted_eligible_features:
            complexity: int = self._relevant_features[self._f_idx_to_feature_index[f_idx]][1].complexity - 1
            r_idxs_to_remove: intbitset = intbitset([r_idx for r_idx in r_idxs if f_idx in self._viewer.f_idxs_changed_by(r_idx)])
            partition: Dict[Tuple[Tuple[int, int]], intbitset] = _partition_r_idxs_with_f_idx(r_idxs, f_idx, self._viewer)
            logging.info(f"[_rec_solve_v2]:{indent_str}    f{f_idx}.{self._relevant_features[self._f_idx_to_feature_index[f_idx]][1]._dlplan_feature}/{complexity}, score={score}, r_idxs_to_remove={r_idxs_to_remove}, partition={[len(block - r_idxs_to_remove) for block in partition.values()]}")
            assert len(r_idxs_to_remove) > 0

            # Register rules info
            for r_idx in r_idxs_to_remove:
                r_idx_to_info[r_idx] = (f_idx, intbitset(above))

            # Remove rules that change f_idx
            for r_idx in r_idxs_to_remove:
                self._viewer.remove_rule(r_idx)
            alive_r_idxs: intbitset = r_idxs - r_idxs_to_remove

            # Recursion on each partition block
            successful_recursion: bool = True
            singleton: intbitset = intbitset([f_idx])
            solution: intbitset = intbitset(singleton)
            for block in partition.values():
                # Remove r_idxs not in partition block before recursive call
                r_idxs_not_in_block: intbitset = alive_r_idxs - block
                for r_idx in r_idxs_not_in_block:
                    self._viewer.remove_rule(r_idx)

                # Recursive call
                solution_for_block = self._rec_solve_v2(branch + [f_idx], chosen | solution, above | singleton, r_idx_to_info, **kwargs)
                if solution_for_block is None:
                    successful_recursion = False
                else:
                    solution |= solution_for_block

                # Restore r_idxs not in partition block after recursive call
                for r_idx in r_idxs_not_in_block:
                    self._viewer.restore_rule(r_idx)

                # If failure, break
                if not successful_recursion: break

            # Restore rules that change f_idx
            for r_idx in r_idxs_to_remove:
                self._viewer.restore_rule(r_idx)

            if successful_recursion:
                logging.info(f"[_rec_solve_v2]:{indent_str} solution={list(solution)}")
                return solution

        # Backtrack because there is no solution
        logging.info(f"[_rec_solve_v2]:{indent_str} Backtrack")
        return None

    def solve(self, **kwargs) -> Dict[str, Any]:
        # This is the OLD recursive solver implemented above by _rec_solve_v2()

        # Reset rule viewer
        self._viewer.reset()

        # TODO: Fix error for zenotravel3
        # TODO: Prefer numerical over boolean features

        # Calculate features that "eliminate" all rules with greedy solver
        r_idx_to_info: Dict[int, Tuple[int, intbitset]] = dict()
        f_idxs = self._rec_solve_v2([], intbitset(), intbitset(), r_idx_to_info, **kwargs)
        if f_idxs is None: raise RuntimeError("No solution for given features")

        # Finalize policy
        result: Dict[str, Any] = self._finalizer(f_idxs, r_idx_to_info, **self._finalizer_options)
        result.update({"r_idxs": list(r_idx_to_info.keys())})
        return result

    def one_solution(self, other_ext_states: List[Tuple[int, int]], **kwargs) -> Dict[str, Any]:
        options: Dict[str, Any] = dict(kwargs)
        if "cost_bound" in options: options.pop("cost_bound")
        generator: SolutionGenerator = SolutionGenerator(self._preprocessing_data, self._benchmark._state_factory, self._finalizer, **options)
        solution: Dict[str, Any] = generator.one_solution(other_ext_states, cost_bound=None, **options)
        return solution

    def solutions(self, other_ext_states: List[Tuple[int, int]], **kwargs) -> Generator[Dict[str, Any], None, None]:
        generator: SolutionGenerator = SolutionGenerator(self._preprocessing_data, self._benchmark._state_factory, self._finalizer, **kwargs)
        yield from generator.solutions(other_ext_states, **kwargs)

    def repair(self, solution: Dict[str, Any], reasons: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._finalizer.repair(solution, reasons)

