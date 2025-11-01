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


class ASPFinalizer:
    def __init__(self,
                 benchmark: Any,
                 r_idx_to_ext_state: List[Tuple[int, int]],
                 ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]],
                 ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray],
                 goal_ext_states: List[Tuple[int, int]],
                 non_goal_ext_states: List[Tuple[int, int]],
                 requirements_for_goals, # REMOVE THIS
                 annotated_requirements, # REMOVE
                 numerical_f_idxs, # REMOVE
                 feature_pool: List[Feature], # REMOVE
                 width: int
                 ):

        self._benchmark: Any = benchmark
        self._r_idx_to_ext_state: List[Tuple[int, int]] = r_idx_to_ext_state
        self._ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]] = ext_state_to_ext_edge
        self._ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = ext_state_to_feature_valuations
        self._goal_ext_states: List[Tuple[int, int]] = goal_ext_states
        self._non_goal_ext_states: List[Tuple[int, int]] = non_goal_ext_states
        self._requirements_for_goals = requirements_for_goals
        self._annotated_requirements = annotated_requirements
        self._numerical_f_idxs = numerical_f_idxs
        self._feature_pool: List[Feature] = feature_pool
        self._width: int = width

        self._asp_solver: ASPSolver = None
        self._solution: List[int] = None
        self._r_idx_to_info:  Dict[int, Tuple[int, intbitset]] = None
        self._ext_pairs: List[Tuple[str, Tuple[int, Tuple[int, int]]]] = None
        self._ext_pairs_set: Set[Any] = None
        self._pair_idx_to_r_idx: List[int] = None
        self._pair_idx_to_changes: List[Tuple[str]] = None
        self._reachable: List[Tuple[Tuple[int, int], bool, bool]] = None
        self._reachable_set: Set[Tuple[int, int]] = None
        self._ext_states: List[Tuple[str, Tuple[int, int]]] = None
        self._ext_states_set: Set[Tuple[int, int]] = None
        self._ext_state_to_boolean_valuation: Dict[Tuple[int, int], List[int]] = None
        self._goal_separating_features: List[Any] = None
        self._facts: Dict[str, List[Any]] = None

    def _get_ext_state(self, r_idx: int) -> Tuple[int, int]:
        return self._r_idx_to_ext_state[r_idx]

    def _get_feature_valuations(self, ext_state: Tuple[int, int]) -> np.ndarray:
        feature_valuations: np.ndarray = self._ext_state_to_feature_valuations.get(ext_state)
        if feature_valuations is None:
            assert self._feature_pool is not None
            feature_valuations: np.ndarray = self._benchmark.get_feature_valuations(*ext_state, self._feature_pool)
        return feature_valuations

    def add_ext_pair(self, pair_type: str, ext_pair: Tuple[int, Tuple[int, int]], r_idx: int):
        assert pair_type in ["good", "bad"]
        assert ext_pair not in self._ext_pairs_set

        # Add ext_pair
        pair_idx: int = len(self._ext_pairs)
        self._ext_pairs.append((pair_type, ext_pair))
        self._ext_pairs_set.add(ext_pair)
        self._pair_idx_to_r_idx.append(r_idx)

        # Calculate boolean valuations and changes for ext_pair
        ext_state_src: Tuple[int, int] = (ext_pair[0], ext_pair[1][0])
        ext_state_dst: Tuple[int, int] = (ext_pair[0], ext_pair[1][1])
        ext_states: List[Tuple[int, int]] = [ext_state_src, ext_state_dst]
        valuations: List[np.ndarray] = [self._get_feature_valuations(ext_state)[self._solution] for ext_state in ext_states]
        boolean_valuations: List[List[int]] = [[1 if value > 0 else 0 for value in valuation] for valuation in valuations]
        for i in [0, 1]:
            if ext_states[i] not in self._ext_state_to_boolean_valuation:
                self._ext_state_to_boolean_valuation[ext_states[i]] = boolean_valuations[i]
        changes: np.ndarray = valuations[1] - valuations[0]
        changes: Tuple[str] = tuple("inc" if d > 0 else ("dec" if d < 0 else "eqv") for d in changes)
        self._pair_idx_to_changes.append(changes)

        # Create facts for source/3, change/3, and fixed/2
        assert pair_type in ["good", "bad"]
        self._facts[f"{pair_type}/1"].append(self._asp_solver.make_fact(pair_type, pair_idx))
        for i, f_idx in enumerate(self._solution):
            self._facts["source/3"].append(self._asp_solver.make_fact("source", pair_idx, f_idx, boolean_valuations[0][i]))
        for i, change in enumerate(changes):
            self._facts["change/3"].append(self._asp_solver.make_fact("change", pair_idx, self._solution[i], change))

        # Create facts for fixed/2
        if pair_type == "good":
            f_idx, above = self._r_idx_to_info.get(r_idx)
            self._facts["fixed/2"].append(self._asp_solver.make_fact("fixed", pair_idx, f_idx))
            for g_idx in above:
                self._facts["fixed/2"].append(self._asp_solver.make_fact("fixed", pair_idx, g_idx))

    def add_ext_state_reachable(self, ext_state: Tuple[int, int], force: bool, soft: bool):
        assert ext_state not in self._reachable_set

        reachable_idx: int = len(self._reachable)
        self._reachable.append((ext_state, force, soft))
        self._reachable_set.add(ext_state)

        # Calculate boolean valuation for ext_state
        valuation: np.ndarray = self._get_feature_valuations(ext_state)[self._solution]
        boolean_valuation: List[int] = [1 if value > 0 else 0 for value in valuation]
        if ext_state not in self._ext_state_to_boolean_valuation:
            self._ext_state_to_boolean_valuation[ext_state] = boolean_valuation

        # Create facts for reachable/1, force/1, and soft/1
        self._facts["reachable/1"].append(self._asp_solver.make_fact("reachable", reachable_idx))
        if force: self._facts["force/1"].append(self._asp_solver.make_fact("force", reachable_idx))
        if soft: self._facts["soft/1"].append(self._asp_solver.make_fact("soft", reachable_idx))

        # Create facts for value_r/3
        for i, f_idx in enumerate(self._solution):
            self._facts["value_r/3"].append(self._asp_solver.make_fact("value_r", reachable_idx, f_idx, boolean_valuation[i]))

        # Create facts for transition/2, and change/3
        # Make width exploration from ext-state S to obtain all reachable states S'.
        # Add pairs (S,S') as transition/2 atoms, and add change/3 atoms
        seen_changes: Set[Tuple[str]] = set()
        for state_idx in self._benchmark.exploration(*ext_state, self._width, caching=True):
            ext_state_dst: Tuple[int, int] = (ext_state[0], state_idx)
            ext_pair: Tuple[int, Tuple[int, int]] = (ext_state[0], (ext_state[1], state_idx))
            assert ext_pair not in self._ext_pairs_set

            # Calculate boolean valuation for (next) state_idx, and changes for ext_pair
            valuation_dst: np.ndarray = self._get_feature_valuations(ext_state_dst)[self._solution]
            boolean_valuation_dst: List[int] = [1 if value > 0 else 0 for value in valuation_dst]
            if ext_state_dst not in self._ext_state_to_boolean_valuation:
                self._ext_state_to_boolean_valuation[ext_state_dst] = boolean_valuation_dst
            changes: np.ndarray = valuation_dst - valuation
            changes: Tuple[str] = tuple("inc" if d > 0 else ("dec" if d < 0 else "eqv") for d in changes)

            # Create facts if this is a "new pair"
            if changes not in seen_changes:
                seen_changes.add(changes)

                # Add pair
                pair_idx: int = len(self._ext_pairs)
                self._ext_pairs.append(("reachable", ext_pair))
                self._pair_idx_to_r_idx.append(None)
                self._pair_idx_to_changes.append(changes)

                # Create facts for transition/2 and change/3
                self._facts["transition/2"].append(self._asp_solver.make_fact("transition", reachable_idx, pair_idx))
                for i, f_idx in enumerate(self._solution):
                    self._facts["source/3"].append(self._asp_solver.make_fact("source", pair_idx, f_idx, boolean_valuation[i]))
                for i, change in enumerate(changes):
                    self._facts["change/3"].append(self._asp_solver.make_fact("change", pair_idx, self._solution[i], change))
                logging.debug(f"Pair {ext_pair} added as atom 'transition({reachable_idx},{pair_idx})': changes={changes}")

    def add_bad_ext_edges(self, bad_ext_edges: List[Tuple[int, Tuple[int, int]]]):
        for ext_edge in bad_ext_edges:
            self.add_ext_pair("bad", ext_edge, None)

    def add_ext_state(self, state_type: str, ext_state: Tuple[int, int]):
        assert state_type in ["goal", "non-goal"]
        assert ext_state not in self._ext_states_set
        ext_state_idx: int = len(self._ext_states)
        self._ext_states.append((state_type, ext_state))

        valuation: np.ndarray = self._get_feature_valuations(ext_state)[self._solution]
        boolean_valuation: List[int] = [1 if value > 0 else 0 for value in valuation]
        if ext_state not in self._ext_state_to_boolean_valuation:
            self._ext_state_to_boolean_valuation[ext_state] = boolean_valuation

        self._facts["state/2"].append(self._asp_solver.make_fact("state", state_type, ext_state_idx))
        for i, f_idx in enumerate(self._solution):
            self._facts["value_s/3"].append(self._asp_solver.make_fact("value_s", ext_state_idx, f_idx, boolean_valuation[i]))

    def get_new_solver(self) -> ASPSolver:
        fact_signatures: List[Tuple[Any]] = [
            ("feature", ("f",), "feature(f)."),
            ("boolean", ("f",), "boolean(f)."),
            ("good", ("g",), "good(g)."),
            ("bad", ("b",), "bad(b)."),
            ("fixed", ("g", "f"), "fixed(g,f)."),
            ("source", ("t", "f", "v"), "source(t,f,v)."),
            ("change", ("t", "f", "c"), "change(t,f,c)."),
            ("yield", ("y",), "yield(y)."),
            #("yield", ("g", "y",), "yield(g,y)."),
            ("value", ("y", "f", "v",), "value(y,f,v)."),
            ("goal", ("y",), "goal(y)."),
            ("reachable", ("r",), "reachable(r)."),
            ("value_r", ("r", "f", "v"), "value_r(r,f,v)."),
            ("transition", ("r", "t"), "transition(r,t)."),
            ("force", ("r",), "force(r)."),
            ("soft", ("r",), "soft(r)."),
            ("state", ("t", "s",), "state(t,s)."),
            ("value_s", ("s", "f", "v"), "value_s(s,f,v)."),
            ("soft_constraints", (), "soft_constraints."),
            ("hard_constraints", (), "hard_constraints."),
        ]
        arguments: List[str] = ["--parallel-mode=16", "-n", "0"]
        #loads: List[str] = [str(LIST_DIR / f"relax_transitions.lp")]
        loads: List[str] = [str(LIST_DIR / f"relax_transitions_v2.lp")]
        asp_solver: ASPSolver = ASPSolver(arguments=arguments, fact_signatures=fact_signatures, loads=loads)
        return asp_solver

    def construct(self, solution: List[int], r_idx_to_info: Dict[int, Tuple[int, intbitset]], soft_constraints):
        # Base elements
        self._asp_solver: ASPSolver = self.get_new_solver()
        self._solution: List[int] = solution
        self._r_idx_to_info: Dict[int, Tuple[int, intbitset]] = r_idx_to_info
        self._ext_pairs: List[Tuple[str, Tuple[int, Tuple[int, int]]]] = []
        self._ext_pairs_set: Set[Any] = set()
        self._pair_idx_to_r_idx: List[int] = []
        self._pair_idx_to_changes: List[Tuple[str]] = []
        self._reachable: List[Tuple[Tuple[int, int], bool, bool]] = []
        self._reachable_set: Set[Tuple[int, int]] = set()
        self._ext_states: List[Tuple[str, Tuple[int, int]]] = []
        self._ext_states_set: Set[Tuple[int, int]] = set()
        self._ext_state_to_boolean_valuation: Dict[Tuple[int, int], List[int]] = dict()
        self._goal_separating_features: List[Any] = []
        self._facts: Dict[str, List[Any]] = defaultdict(list)

        # Good transitions
        for r_idx in r_idx_to_info.keys():
            ext_state: Tuple[int, int] = self._get_ext_state(r_idx)
            ext_edge: Tuple[int, Tuple[int, int]] = self._ext_state_to_ext_edge.get(ext_state)
            assert ext_state is not None and ext_edge is not None
            self.add_ext_pair("good", ext_edge, r_idx)

        # Goal and non-goal states
        for ext_state in  self._goal_ext_states:
            self.add_ext_state("goal", ext_state)

        for ext_state in  self._non_goal_ext_states:
            self.add_ext_state("non-goal", ext_state)

        """
        # Valuations that separate goal from non-goal states
        pending_goal_requirements: List[int] = list(range(len(self._requirements_for_goals)))
        while len(pending_goal_requirements) > 0:
            candidates: List[Tuple[int, int]] = [(i, sum([1 if f_idx in self._requirements_for_goals[j] else 0 for j in pending_goal_requirements])) for i, f_idx in enumerate(solution)]
            candidates_sorted: List[Tuple[int, int]] = sorted(candidates, key=lambda p: p[1], reverse=True)
            assert candidates_sorted[0][1] > 0
            self._goal_separating_features.append(candidates_sorted[0][0])
            pending_goal_requirements: List[int] = [i for i in pending_goal_requirements if solution[candidates_sorted[0][0]] not in self._requirements_for_goals[i]]

        for pair in [annotation.get("pair") for annotation, _ in self._annotated_requirements if annotation.get("key") == "Goal"]:
            goal_ext_state: Tuple[int, int] = (pair[0], pair[1][0])
            non_goal_ext_state: Tuple[int, int] = (pair[0], pair[1][1])
            goal_boolean_valuation: Tuple[int] = self._ext_state_to_boolean_valuation.get(goal_ext_state)
            non_goal_boolean_valuation: Tuple[int] = self._ext_state_to_boolean_valuation.get(non_goal_ext_state)
            if non_goal_boolean_valuation is None:
                # This can happen because set of transitions above doesn't contain all the relevant transitions
                valuation: np.ndarray = self._get_feature_valuations(non_goal_ext_state)[solution]
                non_goal_boolean_valuation: List[int] = [1 if value > 0 else 0 for value in valuation]
                self._ext_state_to_boolean_valuation[non_goal_ext_state] = non_goal_boolean_valuation
            self._conditions_for_goal.add(tuple([(i, goal_boolean_valuation[i]) for i in self._goal_separating_features]))
            self._conditions_for_non_goal.add(tuple([(i, non_goal_boolean_valuation[i]) for i in self._goal_separating_features]))

        if len(self._conditions_for_goal & self._conditions_for_non_goal) > 0:
            logging.warning(f"Non-empty intersection of feature valuations for goal and non-goal states: {self._conditions_for_goal & self._conditions_for_non_goal}")
        logging.info(f"conditions_for_goal: features={self._solution}, valuations={sorted(self._conditions_for_goal)}")
        """

        # Base facts
        self._facts["constraints"].append(self._asp_solver.make_fact("soft_constraints" if soft_constraints else "hard_constraints"))

        # feature/1 and boolean/1
        for f_idx in self._solution:
            self._facts["feature/1"].append(self._asp_solver.make_fact("feature", f_idx))
            if f_idx not in self._numerical_f_idxs:
                self._facts["feature/1"].append(self._asp_solver.make_fact("boolean", f_idx))

        # yield/1, value/3, and goal/1
        self._yields: List[Tuple[int]] = list(product(*[(0, 1) for _ in self._solution]))
        for i, _yield in enumerate(self._yields):
            self._facts["yield/1"].append(self._asp_solver.make_fact("yield", i))
            for j, f_idx in enumerate(self._solution):
                self._facts["value/3"].append(self._asp_solver.make_fact("value", i, f_idx, _yield[j]))
            """
            if any([all([_yield[j] == value for j, value in condition]) for condition in self._conditions_for_goal]):
                self._facts["goal/1"].append(self._asp_solver.make_fact("goal", i))
                logging.info(f"GOAL: yield={_yield}")
            """

    def solve(self) -> Dict[str, Any]:
        local_timer: Timer = Timer()
        self._asp_solver.ground(self._facts, dump_asp_program=False)
        symbols, cost, exit_code = self._asp_solver.optimize_model()
        local_timer.stop()
        logging.info(f"{local_timer.get_elapsed_sec():.02f} second(s) for ASP solver; exit_code={exit_code}")

        # If no solution, return None
        if symbols is None:
            return {
                "decorations": None,
                "conditions_for_goal": None,
            }

        # Read symbols
        eqclass: intbitset = intbitset()
        eq: Dict[int, intbitset] = defaultdict(intbitset)
        goal_yields: List[int] = []
        marks: Dict[str, Dict[int, intbitset]] = {"dont_care": defaultdict(intbitset), "unknown": defaultdict(intbitset)}
        for symbol in symbols:
            if symbol.name == "class":
                eqclass.add(symbol.arguments[0].number)
            elif symbol.name == "eq":
                eq[symbol.arguments[0].number].add(symbol.arguments[1].number)
            elif symbol.name == "goal_y":
                goal_yields.append(symbol.arguments[0].number)
            elif symbol.name in ["dont_care", "unknown"]:
                marks[symbol.name][symbol.arguments[0].number].add(symbol.arguments[1].number)
            elif symbol.name in ["non_sound", "non_closed", "non_safe", "non_covered"]:
                logging.warning(colored(f"Got {symbol.name}({symbol.arguments[0].number})", "magents", attrs=["bold"]))

        # Construct goal conditions
        conditions_for_goal: List[Dict[int, int]] = [dict(zip(self._solution, self._yields[i])) for i in goal_yields]

        # Construct decorations
        decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = {"dont_care": defaultdict(lambda: defaultdict(intbitset)), "unknown": defaultdict(lambda: defaultdict(intbitset))}
        for mark in marks.keys():
            for tr_idx, f_idxs in marks[mark].items():
                for tr_idx_2 in eq.get(tr_idx):
                    ext_edge: Tuple[int, Tuple[int, int]] = self._ext_pairs[tr_idx_2][1]
                    ext_state: Tuple[int, int] = (ext_edge[0], ext_edge[1][0])
                    decorations[mark][ext_state[0]][ext_state[1]] |= f_idxs

        return {
            "decorations": decorations,
            "conditions_for_goal": conditions_for_goal,
        }

    def recalculate(self, solution: Dict[str, Any], reasons: List[Dict[str, Any]]) -> Dict[str, Any]:
        for reason in reasons:
            logging.info(f"Recalculate: reason={reason}")
            if "non-sound" in reason:
                return {"decorations": None, "conditions_for_goal": None}
                for ext_state in reason.get("non-sound"):
                    self.add_ext_state_reachable(ext_state, force=True, soft=False)
            elif "non-goal" in reason:
                for ext_state in reason.get("non-goal"):
                    self.add_ext_state("non-goal", ext_state)
            else:
                raise RuntimeError(f"Unknown reason for recalculate(): reason={reason}")

        self._asp_solver: ASPSolver = self.get_new_solver()
        result: Dict[str, Any] = self.solve()
        logging.info(f"Recalculate: reasons={reasons}, result={result}")
        return result

# Adds features to satisfy pending requirements and simplify resulting policy
class PolicyFinalizer:
    def __init__(self,
                 benchmark: Any,
                 preprocessing_data: Dict[str, Any],
                 r_idx_to_ext_state: List[Tuple[int, int]],
                 ext_state_to_r_idx: Dict[Tuple[int, int], int],
                 annotated_requirements: List[Tuple[Dict[str, Any], intbitset]],
                 **kwargs):
        self._benchmark: Any = benchmark
        self._r_idx_to_ext_state: List[Tuple[int, int]] = r_idx_to_ext_state
        self._ext_state_to_r_idx: Dict[Tuple[int, int], int] = ext_state_to_r_idx
        self._annotated_requirements: List[Tuple[Dict[str, Any], intbitset]] = annotated_requirements
        self._soft_constraints: bool = kwargs.get("soft_constraints", False)

        self._width: int = preprocessing_data.get("width")
        self._feature_pool: List[Feature] = preprocessing_data.get("feature_pool")
        self._ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]] = preprocessing_data.get("ext_state_to_ext_edge")
        self._ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = preprocessing_data.get("ext_state_to_feature_valuations")
        self._bad_ext_edges: Set[Tuple[int, Tuple[int, int]]] = preprocessing_data.get("bad_ext_edges")
        self._ext_successors: Dict[Tuple[int, int], List[Tuple[str, int]]] = preprocessing_data.get("ext_successors")
        self._goal_ext_states: List[Tuple[int, int]] = preprocessing_data.get("goal_ext_states")
        self._non_goal_ext_states: List[Tuple[int, int]] = preprocessing_data.get("non_goal_ext_states")

        # Features
        self._f_idx_to_feature_index: Dict[int, int] = preprocessing_data.get("f_idx_to_feature_index")
        self._relevant_features: List[Tuple[int, Feature]] = preprocessing_data.get("relevant_features")
        self._numerical_features: List[Tuple[int, Feature]] = [(f_idx, feature) for f_idx, feature in self._relevant_features if isinstance(feature.dlplan_feature, dlplan_core.Numerical)]
        self._numerical_f_idxs: intbitset = intbitset([f_idx for f_idx, _ in self._numerical_features])

        # Requirements
        self._requirements: List[intbitset] = [requirement for _, requirement in self._annotated_requirements]
        self._requirements_for_goals: List[intbitset] = [requirement for annotation, requirement in self._annotated_requirements if annotation.get("key") == "Goal"]

        # Asp-based calculation
        self._solution: List[int] = None

    def _get_ext_state(self, r_idx: int) -> Tuple[int, int]:
        return self._r_idx_to_ext_state[r_idx]

    def _get_r_idx(self, ext_state: Tuple[int, int]) -> int:
        return self._ext_state_to_r_idx.get(ext_state)

    def _get_feature_valuations(self, ext_state: Tuple[int, int]) -> np.ndarray:
        feature_valuations: np.ndarray = self._ext_state_to_feature_valuations.get(ext_state)
        if feature_valuations is None:
            assert self._feature_pool is not None
            feature_valuations: np.ndarray = self._benchmark.get_feature_valuations(*ext_state, self._feature_pool)
        return feature_valuations

    def _score_fn(self, f_idx: int, pending_requirements: List[int]) -> Tuple[Union[int, float]]:
        feature_index: int = self._f_idx_to_feature_index[f_idx]
        feature_complexity: int = self._relevant_features[feature_index][1].complexity - 1
        solved_pending_requirements: List[int] = [i for i in pending_requirements if f_idx in self._requirements[i]]
        return (len(solved_pending_requirements) / feature_complexity,)

    """
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
                valuations: np.ndarray = self._get_feature_valuations(ext_state)[self._solution]
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

        self._conditions_for_goal: Set[Tuple[Tuple[int, int]]] = set()
        self._conditions_for_non_goal: Set[Tuple[Tuple[int, int]]] = set()
        for pair in [annotation.get("pair") for annotation, _ in self._annotated_requirements if annotation.get("key") == "Goal"]:
            goal_ext_state: Tuple[int, int] = (pair[0], pair[1][0])
            non_goal_ext_state: Tuple[int, int] = (pair[0], pair[1][1])
            goal_boolean_valuation: Tuple[int] = ext_state_to_valuations_boolean.get(goal_ext_state)
            non_goal_boolean_valuation: Tuple[int] = ext_state_to_valuations_boolean.get(non_goal_ext_state)
            if non_goal_boolean_valuation is None:
                # This can happen because set of transitions above doesn't contain all the relevant transitions
                valuations: np.ndarray = self._get_feature_valuations(non_goal_ext_state)[self._solution]
                boolean_valuations: List[int] = [1 if value > 0 else 0 for value in valuations]
                ext_state_to_valuations[non_goal_ext_state] = valuations
                ext_state_to_valuations_boolean[non_goal_ext_state] = boolean_valuations
                non_goal_boolean_valuation = boolean_valuations
            self._conditions_for_goal.add(tuple([(i, goal_boolean_valuation[i]) for i in goal_separating_features]))
            self._conditions_for_non_goal.add(tuple([(i, non_goal_boolean_valuation[i]) for i in goal_separating_features]))

        if len(self._conditions_for_goal & self._conditions_for_non_goal) > 0:
            logging.warning(f"Non-empty intersection of feature valuations for goal and non-goal states: {self._conditions_for_goal & self._conditions_for_non_goal}")
        logging.info(f"conditions_for_goal: features={self._solution}, valuations={sorted(self._conditions_for_goal)}")

        # Construct solver
        fact_signatures: List[Tuple[Any]] = [
            ("soft_constraints", (), "soft_constraints."),
            ("hard_constraints", (), "hard_constraints."),
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

        # Either soft or hard constraints
        facts["constraints"].append(asp_solver.make_fact("soft_constraints" if self._soft_constraints else "hard_constraints"))

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
            if any([all([_yield[j] == value for j, value in feature_valuation]) for feature_valuation in self._conditions_for_goal]):
                facts["goal/1"].append(asp_solver.make_fact("goal", i))
                logging.info(f"GOAL: yield={_yield}")

        return asp_solver, facts, transitions

    def _calculate_decorations_asp(self,
                                   solution: intbitset,
                                   r_idx_to_info: Dict[int, Tuple[int, intbitset]],
                                   **kwargs) -> Dict[str, Dict[int, Dict[int, intbitset]]]:
        local_timer: Timer = Timer()
        asp_solver, facts, transitions = self._construct_asp_solver_and_facts(solution, r_idx_to_info)
        asp_solver.ground(facts, dump_asp_program=False)
        symbols, cost, exit_code = asp_solver.optimize_model()
        local_timer.stop()
        logging.debug(f"{local_timer.get_elapsed_sec():.02f} second(s) for ASP solver")

        # If no solution, return None
        if symbols is None: return None

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
            elif symbol.name in ["non_sound", "non_closed", "non_safe"]:
                logging.warning(colored(f"Got {symbol.name}({symbol.arguments[0].number})", "magents", attrs=["bold"]))

        # Construct decorations
        decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = {"dont_care": defaultdict(lambda: defaultdict(intbitset)), "unknown": defaultdict(lambda: defaultdict(intbitset))}
        for mark in marks.keys():
            for tr_idx, f_idxs in marks[mark].items():
                for tr_idx_2 in eq.get(tr_idx):
                    ext_edge: Tuple[int, Tuple[int, int]] = transitions[tr_idx_2]
                    ext_state: Tuple[int, int] = (ext_edge[0], ext_edge[1][0])
                    decorations[mark][ext_state[0]][ext_state[1]] |= f_idxs

        return decorations
    """

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
    def __call__(self, f_idxs: intbitset, r_idx_to_info: Dict[int, Tuple[int, intbitset]], **kwargs) -> Tuple[List[int], Dict[str, Dict[int, Dict[int, intbitset]]]]:
        # Solve pending requirements
        if kwargs.get("solve_pending_requirements", False):
            solution, _ = self._solve_pending_requirements(f_idxs)
            if solution is None:
                return None, None
            elif solution != f_idxs:
                logging.info(f"{len(solution - f_idxs)} feature(s) added to solution: {sorted(solution - f_idxs)}")
        else:
            solution: intbitset = f_idxs

        cost: int = sum([self._relevant_features[self._f_idx_to_feature_index[f_idx]][1].complexity - 1 for f_idx in solution])
        logging.info(f"Solution {sorted(solution)} has cost {cost}")

        # Decorations
        decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = dict()
        if kwargs.get("simplify_policy", False):
            if len(solution) >= kwargs.get("threshold_for_asp_based_simplification", 12):
                assert False
                logging.info(f"Calculating decorations with NAIVE solver: {len(solution)} feature(s)")
                decorations = self._calculate_decorations_naive(solution, r_idx_to_info)
                result: Dict[str, Any] = {"decorations": decorations}
            else:
                result: Dict[str, Any] = self._calculate_decorations_asp(solution, r_idx_to_info)

        return {
            "cost": cost,
            "f_idxs": sorted(solution),
            "r_idxs": None,
            "decorations": result.get("decorations"),
            "conditions_for_goal": result.get("conditions_for_goal"),
        }

    def v2(self, f_idxs: intbitset, r_idx_to_info: Dict[int, Tuple[int, intbitset]], **kwargs) -> Tuple[List[int], Dict[str, Dict[int, Dict[int, intbitset]]]]:
        # Solve pending requirements
        if kwargs.get("solve_pending_requirements", False):
            solution, _ = self._solve_pending_requirements(f_idxs)
            if solution is None:
                return None, None
            elif solution != f_idxs:
                logging.info(f"{len(solution - f_idxs)} feature(s) added to solution: {sorted(solution - f_idxs)}")
        else:
            solution: intbitset = f_idxs

        cost: int = sum([self._relevant_features[self._f_idx_to_feature_index[f_idx]][1].complexity - 1 for f_idx in solution])
        logging.info(f"Solution {sorted(solution)} has cost {cost}")

        # Decorations
        decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = dict()
        if kwargs.get("simplify_policy", False):
            if len(solution) >= kwargs.get("threshold_for_asp_based_simplification", 12):
                logging.info(f"Calculating decorations with NAIVE solver: {len(solution)} feature(s)")
                decorations = self._calculate_decorations_naive(solution, r_idx_to_info)
                result: Dict[str, Any] = {"decorations": decorations}
            else:
                self._finalizer: ASPFinalizer = ASPFinalizer(self._benchmark, self._r_idx_to_ext_state, self._ext_state_to_ext_edge, self._ext_state_to_feature_valuations, self._goal_ext_states, self._non_goal_ext_states, self._requirements_for_goals, self._annotated_requirements, self._numerical_f_idxs, self._feature_pool, self._width)
                self._finalizer.construct(sorted(solution), r_idx_to_info, self._soft_constraints)
                result: Dict[str, Any] = self._finalizer.solve()

        return {
            "cost": cost,
            "f_idxs": sorted(solution),
            "r_idxs": None,
            "decorations": result.get("decorations"),
            "conditions_for_goal": result.get("conditions_for_goal"),
        }

    def recalculate(self, solution: Dict[str, Any], reason: Dict[str, Any]) -> Dict[str, Any]:
        return self._finalizer.recalculate(solution, reason)

