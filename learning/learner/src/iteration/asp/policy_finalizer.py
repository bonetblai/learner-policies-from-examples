import os
import logging
import random
import numpy as np

# Bitset-based unordered sets of unsigned integers
from intbitset import intbitset

from pathlib import Path
from termcolor import colored
from typing import Set, Tuple, List, Union, Dict, Any, Optional, Union, Callable, Generator
from collections import defaultdict
from itertools import product

import dlplan.core as dlplan_core

from ..feature_pool import Feature
from ...util import Timer

from .asp_solver import ASPSolver

LIST_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


# Adds features to satisfy pending requirements and simplify resulting policy
class PolicyFinalizer:
    def __init__(self,
                 preprocessing_data: Dict[str, Any],
                 r_idx_to_ext_state: List[Tuple[int, int]],
                 ext_state_to_r_idx: Dict[Tuple[int, int], int],
                 annotated_requirements: List[Tuple[Dict[str, Any], intbitset]],
                 **kwargs):
        self._r_idx_to_ext_state: List[Tuple[int, int]] = r_idx_to_ext_state
        self._ext_state_to_r_idx: Dict[Tuple[int, int], int] = ext_state_to_r_idx
        self._annotated_requirements: List[Tuple[Dict[str, Any], intbitset]] = annotated_requirements

        self._ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]] = preprocessing_data.get("ext_state_to_ext_edge")
        self._ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = preprocessing_data.get("ext_state_to_feature_valuations")
        self._bad_ext_edges: Set[Tuple[int, Tuple[int, int]]] = preprocessing_data.get("bad_ext_edges")
        self._ext_successors: Dict[Tuple[int, int], List[Tuple[str, int]]] = preprocessing_data.get("ext_successors")

        # Features
        self._f_idx_to_feature_index: Dict[int, int] = preprocessing_data.get("f_idx_to_feature_index")
        self._relevant_features: List[Tuple[int, Feature]] = preprocessing_data.get("relevant_features")
        self._numerical_features: List[Tuple[int, Feature]] = [(f_idx, feature) for f_idx, feature in self._relevant_features if isinstance(feature.dlplan_feature, dlplan_core.Numerical)]
        self._numerical_f_idxs: intbitset = intbitset([f_idx for f_idx, _ in self._numerical_features])

        # Requirements
        self._requirements: List[intbitset] = [requirement for _, requirement in self._annotated_requirements]
        self._requirements_for_goals: List[intbitset] = [requirement for annotation, requirement in self._annotated_requirements if annotation.get("key") == "Goal"]

        # By product of asp-based calculation
        self._solution: List[int] = None
        self._feature_valuations_for_goals: Set[Tuple[Tuple[int, int]]] = None
        self._feature_valuations_for_non_goals: Set[Tuple[Tuple[int, int]]] = None

    def _get_ext_state(self, r_idx: int) -> Tuple[int, int]:
        return self._r_idx_to_ext_state[r_idx]

    def _get_r_idx(self, ext_state: Tuple[int, int]) -> int:
        return self._ext_state_to_r_idx.get(ext_state)

    def _score_fn(self, f_idx: int, pending_requirements: List[int]) -> Tuple[Union[int, float]]:
        feature_index: int = self._f_idx_to_feature_index[f_idx]
        feature_complexity: int = self._relevant_features[feature_index][1].complexity - 1
        solved_pending_requirements: List[int] = [i for i in pending_requirements if f_idx in self._requirements[i]]
        return (len(solved_pending_requirements) / feature_complexity,)

    def _construct_asp_solver_and_facts(self,
                                        solution: intbitset,
                                        r_idx_to_info: Dict[int, Tuple[int, intbitset]]) -> Tuple[ASPSolver, Dict[str, Any]]:
        # Features and transitions
        self._solution: List[int] = sorted(solution)
        transitions: List[Tuple[int, Tuple[int, int]]] = []
        transitions_good: intbitset = intbitset()
        transitions_bad: intbitset = intbitset()
        transitions_siblings: Dict[int, intbitset] = dict()
        transitions_r: Dict[Tuple[int, Tuple[int, int]], int] = dict()

        # Good transitions
        tr_idx_to_r_idx: List[int] = []
        for tr_idx, r_idx in enumerate(r_idx_to_info.keys()):
            tr_idx_to_r_idx.append(r_idx)
            ext_state: Tuple[int, int] = self._get_ext_state(r_idx)
            ext_edge: Tuple[int, Tuple[int, int]] = self._ext_state_to_ext_edge.get(ext_state)
            assert ext_state is not None and ext_edge is not None
            assert ext_edge not in transitions_r
            transitions.append(ext_edge)
            transitions_r[ext_edge] = tr_idx
            transitions_good.add(tr_idx)
        num_transitions = len(transitions)

        # Bad transitions
        for tr_idx, ext_edge in enumerate(list(self._bad_ext_edges)):
            assert ext_edge not in transitions_r
            transitions.append(ext_edge)
            transitions_r[ext_edge] = num_transitions + tr_idx
            transitions_bad.add(num_transitions + tr_idx)
        num_transitions = len(transitions)
        logging.info(f"{len(transitions_good)} good transition(s) and {len(transitions_bad)} bad transition(s)")

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
                valuations: np.ndarray = self._ext_state_to_feature_valuations.get(ext_state)[self._solution]
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
            candidates: List[Tuple[int, int]] = [(i, sum([1 if f_idx in self._requirements_for_goals[j] else 0 for j in pending_goal_requirements])) for i, f_idx in enumerate(self._solution)]
            candidates_sorted: List[Tuple[int, int]] = sorted(candidates, key=lambda p: p[1], reverse=True)
            assert candidates_sorted[0][1] > 0
            goal_separating_features.append(candidates_sorted[0][0])
            pending_goal_requirements: List[int] = [i for i in pending_goal_requirements if self._solution[candidates_sorted[0][0]] not in self._requirements_for_goals[i]]

        self._feature_valuations_for_goals: Set[Tuple[Tuple[int, int]]] = set()
        self._feature_valuations_for_non_goals: Set[Tuple[Tuple[int, int]]] = set()
        for pair in [annotation.get("pair") for annotation, _ in self._annotated_requirements if annotation.get("key") == "Goal"]:
            goal_ext_state: Tuple[int, int] = (pair[0], pair[1][0])
            non_goal_ext_state: Tuple[int, int] = (pair[0], pair[1][1])
            goal_boolean_valuation: Tuple[int] = ext_state_to_valuations_boolean.get(goal_ext_state)
            non_goal_boolean_valuation: Tuple[int] = ext_state_to_valuations_boolean.get(non_goal_ext_state)
            if non_goal_boolean_valuation is None:
                # This can happen because set of transitions above doesn't contain all the relevant transitions
                valuations: np.ndarray = self._ext_state_to_feature_valuations.get(non_goal_ext_state)[self._solution]
                boolean_valuations: List[int] = [1 if value > 0 else 0 for value in valuations]
                ext_state_to_valuations[non_goal_ext_state] = valuations
                ext_state_to_valuations_boolean[non_goal_ext_state] = boolean_valuations
                non_goal_boolean_valuation = boolean_valuations
            self._feature_valuations_for_goals.add(tuple([(i, goal_boolean_valuation[i]) for i in goal_separating_features]))
            self._feature_valuations_for_non_goals.add(tuple([(i, non_goal_boolean_valuation[i]) for i in goal_separating_features]))

        if len(self._feature_valuations_for_goals & self._feature_valuations_for_non_goals) > 0:
            logging.warning(f"Non-empty intersection of feature valuations for goal and non-goal states: {self._feature_valuations_for_goals & self._feature_valuations_for_non_goals}")
        logging.info(f"feature_valuations_for_goals: features={self._solution}, valuations={sorted(self._feature_valuations_for_goals)}")

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
        for f_idx in self._solution:
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
            for i, f_idx in enumerate(self._solution):
                facts["source/3"].append(asp_solver.make_fact("source", tr_idx, f_idx, boolean_valuations[i]))
            for i, change in enumerate(changes):
                facts["change/3"].append(asp_solver.make_fact("change", tr_idx, self._solution[i], change))

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
                for i, f_idx in enumerate(self._solution):
                    facts["source/3"].append(asp_solver.make_fact("source", sibling_idx, f_idx, boolean_valuations[i]))
                for i, change in enumerate(changes):
                    facts["change/3"].append(asp_solver.make_fact("change", sibling_idx, self._solution[i], change))

        # bad/1
        for tr_idx in transitions_bad:
            facts["bad/1"].append(asp_solver.make_fact("bad", tr_idx))
            ext_edge: Tuple[int, Tuple[int, int]] = transitions[tr_idx]
            ext_state_src: Tuple[int, int] = (ext_edge[0], ext_edge[1][0])
            boolean_valuations: List[int] = ext_state_to_valuations_boolean.get(ext_state_src)
            changes: List[str] = tr_idx_to_changes[tr_idx]
            for i, f_idx in enumerate(self._solution):
                facts["source/3"].append(asp_solver.make_fact("source", tr_idx, f_idx, boolean_valuations[i]))
            for i, change in enumerate(changes):
                facts["change/3"].append(asp_solver.make_fact("change", tr_idx, self._solution[i], change))

        # yield/1, value/3, and goal/1
        for i, _yield in enumerate(product(*[(0, 1) for _ in self._solution])):
            facts["yield/1"].append(asp_solver.make_fact("yield", i))
            for j, f_idx in enumerate(self._solution):
                facts["value/3"].append(asp_solver.make_fact("value", i, f_idx, _yield[j]))
            if any([all([_yield[j] == value for j, value in feature_valuation]) for feature_valuation in self._feature_valuations_for_goals]):
                facts["goal/1"].append(asp_solver.make_fact("goal", i))
                logging.info(f"GOAL: yield={_yield}")

        return asp_solver, facts, transitions

    def _calculate_decorations_with_asp_solver(self,
                                               solution: intbitset,
                                               r_idx_to_info: Dict[int, Tuple[int, intbitset]],
                                               **kwargs) -> Dict[str, Dict[int, Dict[int, intbitset]]]:
        local_timer: Timer = Timer()
        asp_solver, facts, transitions = self._construct_asp_solver_and_facts(solution, r_idx_to_info)
        asp_solver.ground(facts, dump_asp_program=False)
        symbols, cost, exit_code = asp_solver.optimize_model()
        local_timer.stop()
        logging.debug(f"{local_timer.get_elapsed_sec():.02f} second(s) for ASP solver")

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
                                     r_idx_to_info: Dict[int, Tuple[int, intbitset]],
                                     **kwargs) -> Dict[str, Dict[int, Dict[int, intbitset]]]:
        #assert False, "CHECK: need to revise naive simplification assuring that only 'active' rules are considered, not all of them"
        # Necessary f_idxs
        r_idx_to_necessary_f_idxs: Dict[int, List[intbitset]] = defaultdict(list)
        for annotation, requirement in self._annotated_requirements:
            if annotation["key"] == "Deadend":
                ext_state: Tuple[int, int] = annotation["ext_state"]
                r_idx: int = self._get_r_idx(ext_state)
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
            instance_idx, state_idx = self._get_ext_state(r_idx)
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
            if not kwargs.get("simplify_only_conditions", False):
                decorations["unknown"][instance_idx][state_idx] = f_idxs_to_remove

        return decorations

    def _solve_pending_requirements(self, f_idxs: intbitset) -> Tuple[intbitset, intbitset]:
        solution: intbitset = intbitset(f_idxs)
        pending_requirements: List[int] = [i for i, (annotation, requirement) in enumerate(self._annotated_requirements) if annotation["key"] != "Edge" and len(requirement & solution) == 0]
        while len(pending_requirements) > 0:
            logging.debug(f"[_solve_pending_requirements] pending_requirements: {pending_requirements}")
            eligible_features_with_score: List[Tuple[int, Tuple[float]]] = [(f_idx, self._score_fn(f_idx, pending_requirements)) for f_idx, _ in self._relevant_features if f_idx not in solution]
            eligible_features_with_non_zero_score: List[Tuple[int, Tuple[float]]] = [(f_idx, score) for f_idx, score in eligible_features_with_score if score[0] > 0]
            sorted_eligible_features: List[Tuple[int, Tuple[float]]] = sorted(eligible_features_with_non_zero_score, key=lambda item: item[1], reverse=True)

            if len(sorted_eligible_features) == 0:
                logging.info("No feature for solving pending requirements:")
                for annotation, requirement in [self._annotated_requirements[i] for i in pending_requirements]:
                    if annotation["key"] == "Edge":
                        ext_state: Tuple[int, int] = annotation["ext_state"]
                        r_idx: int = self._get_r_idx(ext_state)
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
                return None, None

            # Choose a best eligible feature
            best_f_idxs: List[Tuple[int]] = [f_idx for f_idx, score in sorted_eligible_features if score == sorted_eligible_features[0][1]]
            best_f_idx: int = random.choice(best_f_idxs)
            best_score: Tuple[float] = self._score_fn(best_f_idx, pending_requirements)
            complexity: int = self._relevant_features[self._f_idx_to_feature_index[best_f_idx]][1].complexity - 1
            logging.info(f"[_solve_pending_requirements] f{best_f_idx}.{self._relevant_features[self._f_idx_to_feature_index[best_f_idx]][1]._dlplan_feature}/{complexity}, score={best_score}")

            solution.add(best_f_idx)
            pending_requirements_reduced: List[int] = [i for i in pending_requirements if len(self._annotated_requirements[i][1] & solution) == 0]
            assert len(pending_requirements_reduced) < len(pending_requirements)
            pending_requirements = pending_requirements_reduced

        if len(solution - f_idxs) > 0:
            logging.info(f"[_solve_pending_requirements] added={sorted(solution - f_idxs)}")

        return solution, solution - f_idxs

    # This is the inteded "external access point" (API)
    def __call__(self, f_idxs: intbitset, r_idx_to_info: Dict[int, Tuple[int, intbitset]], **kwargs) -> Tuple[intbitset, Dict[str, Dict[int, Dict[int, intbitset]]]]:
        # Solve pending requirements
        if kwargs.get("solve_pending_requirements", False):
            solution, _ = self._solve_pending_requirements(f_idxs)
            if solution is None:
                return None, None
            elif solution != f_idxs:
                logging.info(f"{len(solution - f_idxs)} feature(s) added to solution: {sorted(solution - f_idxs)}")
        else:
            solution = f_idxs

        cost: int = sum([self._relevant_features[self._f_idx_to_feature_index[f_idx]][1].complexity - 1 for f_idx in solution])
        logging.info(f"Solution {sorted(solution)} has cost {cost}")

        # Decorations
        decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = dict()
        if kwargs.get("simplify_policy", False):
            if len(solution) >= kwargs.get("threshold_for_asp_based_simplification", 12):
                logging.info(f"Calculating decorations with NAIVE solver: {len(solution)} feature(s)")
                decorations = self._calculate_decorations_naive(solution, r_idx_to_info)
            else:
                decorations = self._calculate_decorations_with_asp_solver(solution, r_idx_to_info)

        return solution, cost, decorations

