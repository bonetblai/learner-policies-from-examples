import os
import logging
import random
import heapq
import numpy as np

# Bitset-based unordered sets of unsigned integers
from intbitset import intbitset

from pathlib import Path
from termcolor import colored
from typing import Set, Tuple, List, Union, Dict, Any, Optional, Union
from collections import OrderedDict, defaultdict, deque
from itertools import product

import dlplan.core as dlplan_core

from ..feature_pool import Feature
from ...util import Timer
from ...state_space import StateFactory

from .watched_rules import WatchedRules
from .asp_solver import ASPSolver
#from .stratified_policy import StratifiedPolicy, StratifiedPolicyByRules

LIST_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


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

        # Monotone features calculated via watched rules
        self._watched_rules: WatchedRules = self._preprocessing_data.get("watched_rules")

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
        self._requirements_for_goals: List[intbitset] = [requirement for annotation, requirement in self._annotated_requirements if annotation.get("key") == "Goal"]
        self._requirements_for_deadends: List[intbitset] = [requirement for annotation, requirement in self._annotated_requirements if annotation.get("key") == "Deadend"]
        self._num_requirements: Dict[str, int] = {key: sum([1 if annotation.get("key") == key else 0 for annotation, _ in self._annotated_requirements]) for key in ["Edge", "Goal", "Deadend", "Sibling"]}
        logging.info(f"{len(self._requirements)} requirement(s) split as {self._num_requirements}")

        # Support for simplification of policies
        #self._ex_ext_states: Set[Tuple[int, int]] = self._preprocessing_data.get("ex_ext_states")
        self._ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = self._preprocessing_data.get("ext_state_to_feature_valuations")
        self._bad_ext_edges: Set[Tuple[int, Tuple[int, int]]] = self._preprocessing_data.get("bad_ext_edges")
        self._ext_successors: Dict[Tuple[int, int], List[Tuple[str, int]]] = self._preprocessing_data.get("ext_successors")

    def _partition_r_idxs_with_f_idx(self, r_idxs: intbitset, f_idx: int) -> Dict[Tuple[Tuple[int, int]], intbitset]:
        partition: Dict[Tuple[Tuple[int, int]], intbitset] = defaultdict(intbitset)
        for r_idx in r_idxs:
            projection: Tuple[Tuple[int, int]] = self._watched_rules.project_condition(r_idx, f_idx, single=True)
            partition[projection].add(r_idx)
        return partition

    def _score_fn(self, f_idx: int, chosen: intbitset) -> Tuple[Union[int, float]]:
        feature_index: int = self._f_idx_to_feature_index[f_idx]
        feature_complexity: int = self._relevant_features[feature_index][1].complexity
        r_idxs_to_remove: intbitset = intbitset([r_idx for r_idx in self._watched_rules.r_idxs() if f_idx in self._watched_rules.f_idxs_changed(r_idx)])
        partition: Dict[Tuple[Tuple[int, int]], intbitset] = self._partition_r_idxs_with_f_idx(self._watched_rules.r_idxs(), f_idx)
        block_sizes: List[int] = [len(block - r_idxs_to_remove) for _, block in partition.items() if len(block - r_idxs_to_remove) > 0]
        num_blocks: int = len(block_sizes)
        std_block_sizes: float = -np.std(block_sizes) if num_blocks > 0 else 0
        monotone_by_dec: int = 1 if self._watched_rules.is_monotone(f_idx, monotone_only_by_dec=True) else 0
        return (monotone_by_dec, len(r_idxs_to_remove) / feature_complexity, 1 if f_idx in chosen else 0, num_blocks, std_block_sizes)

    def _score_fn2(self, f_idx: int, pending_requirements: List[int]) -> Tuple[Union[int, float]]:
        feature_index: int = self._f_idx_to_feature_index[f_idx]
        feature_complexity: int = self._relevant_features[feature_index][1].complexity
        solved_pending_requirements: List[int] = [i for i in pending_requirements if f_idx in self._requirements[i]]
        return (len(solved_pending_requirements) / feature_complexity,)

    def _construct_asp_solver_and_facts(self,
                                        solution: intbitset,
                                        r_idx_to_info: Dict[int, Tuple[int, intbitset]],
                                        branches: Dict[int, List[int]]) -> Tuple[ASPSolver, Dict[str, Any]]:
        # Features and transitions
        features: List[int] = sorted(solution)
        transitions: List[Tuple[int, Tuple[int, int]]] = []
        transitions_good: intbitset = intbitset()
        transitions_bad: intbitset = intbitset()
        transitions_siblings: Dict[int, intbitset] = dict()
        transitions_r: Dict[Tuple[int, Tuple[int, int]], int] = dict()

        # Good transitions
        tr_idx_to_r_idx: List[int] = []
        for tr_idx, r_idx in enumerate(r_idx_to_info.keys()):
            tr_idx_to_r_idx.append(r_idx)
            ext_state: Tuple[int, int] = self._watched_rules.get_ext_state(r_idx)
            ext_edge: Tuple[int, Tuple[int, int]] = self._ext_state_to_ext_edge.get(ext_state)
            assert ext_state is not None and ext_edge is not None
            assert ext_edge not in transitions_r
            transitions.append(ext_edge)
            transitions_r[ext_edge] = tr_idx
            transitions_good.add(tr_idx)
        num_transitions = len(transitions)
        logging.info(f"{len(transitions_good)} good transition(s)")

        # Bad transitions
        for tr_idx, ext_edge in enumerate(list(self._bad_ext_edges)):
            assert ext_edge not in transitions_r
            transitions.append(ext_edge)
            transitions_r[ext_edge] = num_transitions + tr_idx
            transitions_bad.add(num_transitions + tr_idx)
        num_transitions = len(transitions)
        logging.info(f"{len(transitions_bad)} bad transition(s)")

        # Siblings
        for tr_idx in transitions_good:
            ext_edge: Tuple[int, Tuple[int, int]] = transitions[tr_idx]
            ext_state: Tuple[int, int] = (ext_edge[0], ext_edge[1][0])
            successors: List[Tuple[str, int]] = self._ext_successors.get(ext_state)
            assert successors is not None
            transitions_siblings[tr_idx] = intbitset()
            for action, succ_state_idx in successors:
                if succ_state_idx != ext_edge[1][1]:
                    sibling: Tuple[int, Tuple[int, int]] = (ext_state[0], (ext_state[1], succ_state_idx))
                    assert sibling != ext_edge
                    if sibling not in transitions_r:
                        new_tr_idx = len(transitions)
                        transitions.append(sibling)
                        transitions_r[sibling] = new_tr_idx
                        transitions_siblings[tr_idx].add(new_tr_idx)
        num_transitions = len(transitions)

        # Valuations and changes
        ext_state_to_valuations: Dict[Tuple[int, int], np.ndarray] = dict()
        ext_state_to_valuations_boolean: Dict[Tuple[int, int], List[int]] = dict()
        tr_idx_to_changes: List[List[str]] = []
        for tr_idx, ext_edge in enumerate(transitions):
            ext_state_src: Tuple[int, int] = (ext_edge[0], ext_edge[1][0])
            ext_state_dst: Tuple[int, int] = (ext_edge[0], ext_edge[1][1])
            for ext_state in [ext_state_src, ext_state_dst]:
                valuations: np.ndarray = self._ext_state_to_feature_valuations.get(ext_state)[features]
                boolean_valuations: List[int] = [1 if value > 0 else 0 for value in valuations]
                ext_state_to_valuations[ext_state] = valuations
                ext_state_to_valuations_boolean[ext_state] = boolean_valuations
            changes: np.ndarray = ext_state_to_valuations.get(ext_state_dst) - ext_state_to_valuations.get(ext_state_src)
            changes: List[str] = ["inc" if d > 0 else ("dec" if d < 0 else "eqv") for d in changes]
            tr_idx_to_changes.append(changes)

        # Calculate valuations that separate goal from non-goal states
        goal_separating_features: List[int] = []
        pending_goal_requirements: List[int] = list(range(len(self._requirements_for_goals)))
        while len(pending_goal_requirements) > 0:
            candidates: List[Tuple[int, int]] = [(i, sum([1 if f_idx in self._requirements_for_goals[j] else 0 for j in pending_goal_requirements])) for i, f_idx in enumerate(features)]
            candidates_sorted: List[Tuple[int, int]] = sorted(candidates, key=lambda p: p[1], reverse=True)
            assert candidates_sorted[0][1] > 0
            goal_separating_features.append(candidates_sorted[0][0])
            pending_goal_requirements: List[int] = [i for i in pending_goal_requirements if features[candidates_sorted[0][0]] not in self._requirements_for_goals[i]]

        feature_valuations_for_goals: Set[Tuple[Tuple[int, int]]] = set()
        feature_valuations_for_non_goals: Set[Tuple[Tuple[int, int]]] = set()
        for pair in [annotation.get("pair") for annotation, _ in self._annotated_requirements if annotation.get("key") == "Goal"]:
            goal_ext_state: Tuple[int, int] = (pair[0], pair[1][0])
            non_goal_ext_state: Tuple[int, int] = (pair[0], pair[1][1])
            goal_boolean_valuation: Tuple[int] = ext_state_to_valuations_boolean.get(goal_ext_state)
            non_goal_boolean_valuation: Tuple[int] = ext_state_to_valuations_boolean.get(non_goal_ext_state)
            if non_goal_boolean_valuation is None:
                # This can happen because set of transitions above doesn't not contain all the relevant transitions
                valuations: np.ndarray = self._ext_state_to_feature_valuations.get(non_goal_ext_state)[features]
                boolean_valuations: List[int] = [1 if value > 0 else 0 for value in valuations]
                ext_state_to_valuations[non_goal_ext_state] = valuations
                ext_state_to_valuations_boolean[non_goal_ext_state] = boolean_valuations
                non_goal_boolean_valuation = boolean_valuations
            feature_valuations_for_goals.add(tuple([(i, goal_boolean_valuation[i]) for i in goal_separating_features]))
            feature_valuations_for_non_goals.add(tuple([(i, non_goal_boolean_valuation[i]) for i in goal_separating_features]))

        if len(feature_valuations_for_goals & feature_valuations_for_non_goals) > 0:
            logging.warning(f"Non-empty intersection of feature valuations for goal and non-goal states: {feature_valuations_for_goals & feature_valuations_for_non_goals}")
        logging.info(f"feature_valuations_for_goals: {sorted(feature_valuations_for_goals)}")

        # Construct solver
        fact_signatures: List[Tuple[Any]] = [
            ("feature", ("f",), "feature(f)."),
            ("boolean", ("f",), "boolean(f)."),
            ("good", ("t",), "good(t)."),
            ("bad", ("t",), "bad(t)."),
            ("other", ("t",), "other(t)."),
            ("yield", ("y",), "yield(y)."),
            ("goal", ("y",), "goal(y)."),
            ("fixed", ("t", "f"), "fixed(t,f)."),
            ("sibling", ("t", "s"), "sibling(t,s)."),
            ("source", ("t", "f", "v"), "source(t,f,v)."),
            ("change", ("t", "f", "c"), "change(t,f,c)."),
            ("value", ("y", "f", "v",), "value(y,f,v)."),
        ]
        arguments: List[str] = ["--parallel-mode=16", "-n", "0"]
        loads: List[str] = [str(LIST_DIR / f"relax_transitions.lp")]
        asp_solver: ASPSolver = ASPSolver(arguments=arguments, fact_signatures=fact_signatures, loads=loads)
        facts: Dict[str, List[Any]] = defaultdict(list)

        # Contruct facts for feature/1, transition/1, fixed/2, sibling/2, bad/1, source/3, change/3, yield/1, value/3, goal/1
        for f_idx in features:
            facts["feature/1"].append(asp_solver.make_fact("feature", f_idx))
            if f_idx not in self._numerical_f_idxs:
                facts["feature/1"].append(asp_solver.make_fact("boolean", f_idx))

        # good/1, source/3, and change/3
        for tr_idx in transitions_good:
            facts["good/1"].append(asp_solver.make_fact("good", tr_idx))
            ext_edge: Tuple[int, Tuple[int, int]] = transitions[tr_idx]
            ext_state_src: Tuple[int, int] = (ext_edge[0], ext_edge[1][0])
            boolean_valuations: List[int] = ext_state_to_valuations_boolean.get(ext_state_src)
            changes: List[str] = tr_idx_to_changes[tr_idx]
            for i, f_idx in enumerate(features):
                facts["source/3"].append(asp_solver.make_fact("source", tr_idx, features[i], boolean_valuations[i]))
            for i, change in enumerate(changes):
                facts["change/3"].append(asp_solver.make_fact("change", tr_idx, features[i], change))

        # fixed/2
        for tr_idx in transitions_good:
            r_idx: int = tr_idx_to_r_idx[tr_idx]
            assert r_idx in r_idx_to_info
            f_idx, above = r_idx_to_info.get(r_idx)
            facts["fixed/2"].append(asp_solver.make_fact("fixed", tr_idx, f_idx))
            for g_idx in above:
                facts["fixed/2"].append(asp_solver.make_fact("fixed", tr_idx, g_idx))

        # sibling/2
        for tr_idx in transitions_good:
            facts["sibling/2"].append(asp_solver.make_fact("sibling", tr_idx, tr_idx))
            for sibling_idx in transitions_siblings.get(tr_idx):
                facts["sibling/2"].append(asp_solver.make_fact("sibling", tr_idx, sibling_idx))
                sibling: Tuple[int, Tuple[int, int]] = transitions[sibling_idx]
                ext_state_src: Tuple[int, int] = (sibling[0], sibling[1][0])
                boolean_valuations: List[int] = ext_state_to_valuations_boolean.get(ext_state_src)
                changes: List[str] = tr_idx_to_changes[sibling_idx]
                for i, f_idx in enumerate(features):
                    facts["source/3"].append(asp_solver.make_fact("source", sibling_idx, f_idx, boolean_valuations[i]))
                for i, change in enumerate(changes):
                    facts["change/3"].append(asp_solver.make_fact("change", sibling_idx, features[i], change))

        # bad/1
        for tr_idx in transitions_bad:
            facts["bad/1"].append(asp_solver.make_fact("bad", tr_idx))
            ext_edge: Tuple[int, Tuple[int, int]] = transitions[tr_idx]
            ext_state_src: Tuple[int, int] = (ext_edge[0], ext_edge[1][0])
            boolean_valuations: List[int] = ext_state_to_valuations_boolean.get(ext_state_src)
            changes: List[str] = tr_idx_to_changes[tr_idx]
            for i, f_idx in enumerate(features):
                facts["source/3"].append(asp_solver.make_fact("source", tr_idx, f_idx, boolean_valuations[i]))
            for i, change in enumerate(changes):
                facts["change/3"].append(asp_solver.make_fact("change", tr_idx, features[i], change))

        # yield/1, value/3, and goal/1
        for i, _yield in enumerate(product(*[(0, 1) for _ in features])):
            facts["yield/1"].append(asp_solver.make_fact("yield", i))
            for j, f_idx in enumerate(features):
                facts["value/3"].append(asp_solver.make_fact("value", i, f_idx, _yield[j]))
            if any([all([_yield[j] == value for j, value in feature_valuation]) for feature_valuation in feature_valuations_for_goals]):
                facts["goal/1"].append(asp_solver.make_fact("goal", i))

        return asp_solver, facts, transitions

    def _calculate_decorations_with_asp_solver(self,
                                               solution: intbitset,
                                               r_idx_to_info: Dict[int, Tuple[int, intbitset]],
                                               branches: Dict[int, List[int]]) -> Dict[str, Dict[int, Dict[int, intbitset]]]:
        local_timer: Timer = Timer()
        asp_solver, facts, transitions = self._construct_asp_solver_and_facts(solution, r_idx_to_info, branches)
        asp_solver.ground(facts, dump_asp_program=False)
        symbols, cost, exit_code = asp_solver.optimize_model()
        local_timer.stop()
        logging.info(f"{local_timer.get_elapsed_sec():.02f} second(s) for ASP solver")

        # Read symbols
        eqclass: intbitset = intbitset()
        eq: Dict[int, intbitset] = defaultdict(intbitset)
        marks: Dict[str, Dict[int, intbitset]] = {"dont_care": defaultdict(intbitset), "unknown": defaultdict(intbitset)}
        for symbol in symbols:
            if symbol.name == "class":
                eqclass.add(symbol.arguments[0].number)
            elif symbol.name == "eq":
                eq[symbol.arguments[0].number].add(symbol.arguments[1].number)
            elif symbol.name in ["dont_care", "unknown"]:
                marks[symbol.name][symbol.arguments[0].number].add(symbol.arguments[1].number)
            elif symbol.name in ["nonsound", "nonclosed", "nonsafe"]:
                print(symbol)

        # Construct decorations
        decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = {"dont_care": defaultdict(lambda: defaultdict(intbitset)), "unknown": defaultdict(lambda: defaultdict(intbitset))}
        for mark in marks.keys():
            for tr_idx, f_idxs in marks[mark].items():
                for tr_idx_2 in eq.get(tr_idx):
                    ext_edge: Tuple[int, Tuple[int, int]] = transitions[tr_idx_2]
                    ext_state: Tuple[int, int] = (ext_edge[0], ext_edge[1][0])
                    decorations[mark][ext_state[0]][ext_state[1]] |= f_idxs

        return decorations

    def _calculate_decorations_naive(self,
                                     solution: intbitset,
                                     r_idx_to_info: Dict[int, Tuple[int, intbitset]]) -> Dict[str, Dict[int, Dict[int, intbitset]]]:
        # Necessary f_idxs
        r_idx_to_necessary_f_idxs: Dict[int, List[intbitset]] = defaultdict(list)
        for annotation, requirement in self._annotated_requirements:
            if annotation["key"] == "Deadend":
                ext_state: Tuple[int, int] = annotation["ext_state"]
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
        self._watched_rules.reset()
        r_idx_to_info: Dict[int, Tuple[int, intbitset]] = dict()
        terminating_set, branches = self._rec_solve([], intbitset(), intbitset(), r_idx_to_info, **kwargs)
        #print(f"branches={branches}")
        #print(f"r_idx_to_info={r_idx_to_info}")

        # Solve pending requirements
        solution, _ = self._solve_pending_requirements(terminating_set)
        logging.info(f"Solution: {sorted(solution)}")

        # Cost of solution
        cost = sum([self._relevant_features[self._f_idx_to_feature_index[f_idx]][1].complexity for f_idx in solution])

        # Decorations
        decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = dict()
        if self._simplify_policy:
            if len(solution) > 12:
                logging.info(f"Calculating decorations with NAIVE solver: {len(solution)} feature(s)")
                decorations = self._calculate_decorations_naive(solution, r_idx_to_info)
            else:
                decorations = self._calculate_decorations_with_asp_solver(solution, r_idx_to_info, branches)

        return True, solution, [cost], decorations, None

    def _rec_solve(self,
                   branch: List[int],
                   chosen: intbitset,
                   above: intbitset,
                   r_idx_to_info: Dict[int, Tuple[int, intbitset]],
                   **kwargs) -> Tuple[intbitset, Dict[int, List[int]]]:
        indent_str: str = '   ' * len(branch)

        # Base case: if no remaining rules, empty set of features is solution
        r_idxs: intbitset = self._watched_rules.r_idxs()
        if len(r_idxs) == 0: return intbitset(), dict()

        # Get eligible features sorted by score, removing features that do not eliminate any rule (i.e., with zero score in corresponding component)
        eligible_features: intbitset = self._watched_rules.monotone_features(kwargs.get("monotone_only_by_dec", False))
        logging.info(f"[_rec_solve]:{indent_str} => chosen={list(chosen)}, r_idxs={list(r_idxs)}, {len(eligible_features)} mononote feature(s)")
        eligible_features_with_score: List[Tuple[int, Tuple[float]]] = [(f_idx, self._score_fn(f_idx, chosen)) for f_idx in eligible_features]
        eligible_features_with_non_zero_score: List[Tuple[int, Tuple[float]]] = [(f_idx, score) for f_idx, score in eligible_features_with_score if score[1] > 0]
        sorted_eligible_features: List[Tuple[int, Tuple[float]]] = sorted(eligible_features_with_non_zero_score, key=lambda item: item[1], reverse=True)

        # Check for early termination due to non-existence of solution
        if len(sorted_eligible_features) == 0:
            logging.warning(f"No monotone feature to eliminate rules: r_idxs={sorted(r_idxs)}")
            for r_idx in r_idxs:
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
        r_idxs_to_remove: intbitset = intbitset([r_idx for r_idx in r_idxs if best_f_idx in self._watched_rules.f_idxs_changed(r_idx)])
        partition: Dict[Tuple[Tuple[int, int]], intbitset] = self._partition_r_idxs_with_f_idx(r_idxs, best_f_idx)
        logging.info(f"[_rec_solve]:{indent_str}    f{best_f_idx}.{self._relevant_features[self._f_idx_to_feature_index[best_f_idx]][1]._dlplan_feature}/{complexity}, score={best_score}, r_idxs_to_remove={r_idxs_to_remove}, partition={[len(block - r_idxs_to_remove) for block in partition.values()]}")
        assert len(r_idxs_to_remove) > 0

        # Calculate info
        branches: Dict[int, List[int]] = {best_f_idx: branch}
        for r_idx in r_idxs_to_remove:
            r_idx_to_info[r_idx] = (best_f_idx, intbitset(above))

        # Remove rules that change chosen feature
        for r_idx in r_idxs_to_remove:
            self._watched_rules.remove_rule(r_idx)

        # Recursion on each partition block
        singleton: intbitset = intbitset([best_f_idx])
        solution: intbitset = intbitset(singleton)
        for key, block in partition.items():
            # Remove r_idxs not in partition block before recursive call
            r_idxs_not_in_block: intbitset = r_idxs - block
            for r_idx in r_idxs_not_in_block:
                self._watched_rules.remove_rule(r_idx)

            # Recursive call
            solution_for_block, rec_branches = self._rec_solve(branch + [best_f_idx], chosen | solution, above | singleton, r_idx_to_info, **kwargs)
            solution |= solution_for_block
            branches.update(rec_branches)

            # Restore r_idxs not in partition block after recursive call
            for r_idx in r_idxs_not_in_block:
                self._watched_rules.restore_rule(r_idx)

        # Restore r_idxs that change best f_idx
        for r_idx in r_idxs_to_remove:
            self._watched_rules.restore_rule(r_idx)

        logging.info(f"[_rec_solve]:{indent_str} solution={list(solution)}, branches={branches}")
        return solution, branches

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

