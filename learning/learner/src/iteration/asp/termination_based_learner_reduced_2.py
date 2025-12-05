import os
import logging
import tempfile
import numpy as np
from intbitset import intbitset

from termcolor import colored
from typing import Set, FrozenSet, Tuple, List, Union, Dict, Any, Optional, Union, Generator
from collections import OrderedDict, defaultdict, deque
from itertools import product
from pathlib import Path

import dlplan.core as dlplan_core
import dlplan.policy as dlplan_policy

from ..feature_pool import Feature
from ..feature_pool_utils import prune_features_with_same_feature_change_AND_boolean_valuation_reduced_v2
from ..state_pair_equivalence_utils import make_conditions, make_effects
from ..iteration_data import IterationData
from ..statistics import Statistics
from ...util import change_dir, Timer

from .asp_solver import ASPSolver
from .returncodes import ClingoExitCode

from ...state_space import PDDLInstance, StateFactory, get_plan, get_plan_v2

from .m_pairs import MPairs
from .rule_viewer import RuleViewer

from .greedy_solver import GreedySolver
from .rule_elimination import RuleElimination


class TerminationBasedLearnerReduced:
    def __init__(self, benchmark: Any, timers: Statistics, **kwargs):
        self._benchmark: Any = benchmark
        self._state_factory: StateFactory = self._benchmark._state_factory
        self._timers = timers if timers is not None else Statistics()
        self._options: Dict[str, Any] = kwargs
        self._width: int = kwargs.get("width", 0)

        # Stores edges for ext_states found by planner to avoid calling planning multiple times from same state
        self._ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]] = dict()

    def prepare_iteration(self, iteration_data: IterationData):
        logging.info(colored(f"Preparing iteration...", "blue"))

        # DLPlan state and feature evaluations for relevant states
        self._ext_state_to_dlplan_state_index: Dict[Tuple[int, int], int] = iteration_data.ext_state_to_dlplan_state_index
        self._ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = iteration_data.ext_state_to_feature_valuations
        self._instance_idx_to_denotations_caches: Dict[int, dlplan_core.DenotationsCaches] = iteration_data.instance_idx_to_denotations_caches
        self._iteration_ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]] = dict()

        # Feature pool
        self._feature_pool: List[Feature] = iteration_data.feature_pool
        self._numerical_features: List[Tuple[int, Feature]] = [(f_idx, feature) for f_idx, feature in enumerate(self._feature_pool) if isinstance(feature.dlplan_feature, dlplan_core.Numerical)]
        self._boolean_features: List[Tuple[int, Feature]] = [(f_idx, feature) for f_idx, feature in enumerate(self._feature_pool) if isinstance(feature.dlplan_feature, dlplan_core.Boolean)]
        logging.info(f"Features: #total={len(self._feature_pool)}, #numerical={len(self._numerical_features)}, #boolean={len(self._boolean_features)}")

        # Relevant features
        self._relevant_features: List[Tuple[int, Feature]] = None
        self._relevant_features_idxs: intbitset = None
        self._f_idx_to_feature_index: Dict[int, int] = None

        # Paths and vertices
        self._paths_with_idx: List[Tuple[int, Tuple[int]]] = iteration_data.paths_with_idx
        self._relevant_vertices: Dict[int, Set[int]] = iteration_data.relevant_vertices
        self._non_covered_vertices: Dict[int, Set[int]] = iteration_data.non_covered_vertices
        self._vertices_in_deadend_paths: Dict[int, Set[int]] = iteration_data.vertices_in_deadend_paths
        self._iteration_deadend_paths: List[Tuple[Tuple[int, int]]] = iteration_data.deadend_paths

    def solutions(self, non_covered_ext_states: List[Tuple[int, int]], **kwargs) -> Generator[Dict[str, Any], None, None]:
        # Options passed to ctor (possibly) overrided by kwargs
        options: Dict[str, Any] = dict(self._options)
        options.update(kwargs)

        # Print paths
        for instance_idx, path in self._paths_with_idx:
            logging.info(f"Path: instance_idx={instance_idx}, path={path}, size={len(path)}")
            instance_data: PDDLInstance = self._benchmark._instance_datas[instance_idx]
            assert instance_data.idx == instance_idx

            trajectory: List[Tuple[int, str]] = self._benchmark._preprocessed_datas[instance_idx]["trajectories"][0]
            for i, (state_idx, action) in enumerate(trajectory):
                logging.info(colored(f"  {state_idx}.{instance_data.get_dlplan_state(*instance_data.get_state(state_idx))}", "blue"))
                if action != "GOAL":
                    logging.info(colored(f"  {action}", "green"))

        # Preprocess termination
        preprocessing_data: Dict[str, Any] = self._preprocess_termination(**options)
        ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]] = preprocessing_data.get("ext_state_to_ext_edge")
        viewer: RuleViewer = preprocessing_data.get("rule_viewer")

        # Solver
        solver: RuleElimination = RuleElimination(self._benchmark, preprocessing_data, non_sound_ext_states=set(), **options)

        # Enumerate solutions
        for solution in solver.solutions(non_covered_ext_states, **options):
            f_idxs: List[int] = solution.get("f_idxs")
            r_idxs: List[int] = solution.get("r_idxs")
            decorations: Dict[str, Dict[int, intbitset]] = solution.get("decorations")
            ext_edges: List[Tuple[int, Tuple[int, int]]] = [ext_state_to_ext_edge.get(viewer.get_ext_state(r_idx)) for r_idx in r_idxs]
            rules_with_decorations: Set[Tuple[dlplan_policy.Rule, FrozenSet[int], FrozenSet[int]]] = self._get_rules_with_decorations(ext_edges, f_idxs, decorations)
            yield {
                "cost": solution.get("cost"),
                "f_idxs": f_idxs,
                "ext_edges": ext_edges,
                "rules_with_decorations": rules_with_decorations,
                "conditions_for_goal": solution.get("conditions_for_goal"),
                "verify_closure": solution.get("verify_closure"),
                "verify_bad_edges": solution.get("verify_bad_edges"),
                "solver": solver,
            }

    def solve(self, **kwargs) -> Dict[str, Any]:
        options: Dict[str, Any] = dict(self._options)
        options.update(kwargs)

        # Check that width is 0
        if not options.get("rule_elimination", False) and options.get("width", 0) > 0:
            logging.error(colored(f"Direct solver calculates policies not sketches. Illegal value {options.get('width')} for width", "red", attrs=["bold"]))
            raise RuntimeError(f"Illegal value of width > 0 (width={options.get('width')}) for non-enumerative solver")

        # Print paths
        for instance_idx, path in self._paths_with_idx:
            logging.info(f"Path: instance_idx={instance_idx}, path={path}, size={len(path)}")
            instance_data: PDDLInstance = self._benchmark._instance_datas[instance_idx]
            assert instance_data.idx == instance_idx

            trajectory: List[Tuple[int, str]] = self._benchmark._preprocessed_datas[instance_idx]["trajectories"][0]
            for i, (state_idx, action) in enumerate(trajectory):
                logging.info(colored(f"  {state_idx}.{instance_data.get_dlplan_state(*instance_data.get_state(state_idx))}", "blue"))
                if action != "GOAL":
                    logging.info(colored(f"  {action}", "green"))

        # Preprocess termination
        preprocessing_data: Dict[str, Any] = self._preprocess_termination(**options)
        ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]] = preprocessing_data.get("ext_state_to_ext_edge")
        viewer: RuleViewer = preprocessing_data.get("rule_viewer")

        # Greedy solver
        if not options.get("rule_elimination", False):
            logging.info(f"INITIALIZING SOLVER...")
            solver: GreedySolver = GreedySolver(preprocessing_data, self._state_factory, **options)
            logging.info(f"DONE")

            self._timers.resume("pricing/algorithm")
            solution: Dict[str, Any] = solver.solve(**options)
            self._timers.stop("pricing/algorithm")

        else:
            logging.info(f"INITIALIZING SOLVER...")
            solver: RuleElimination = RuleElimination(self._benchmark, preprocessing_data, non_sound_ext_states=set(), **options)
            logging.info(f"DONE")

            self._timers.resume("pricing/algorithm")
            solution: Dict[str, Any] = solver.one_solution(options.get("non_covered_ext_states"), **options)

        if solution is not None:
            # Construct solution as set of rules paired with unknowns and don't cares.
            # Each path edge generates one rule that is paired with unknowns/don't_cares for source states (if any)
            logging.info(f"Constructing policy...")
            f_idxs: List[int] = solution.get("f_idxs")
            r_idxs: List[int] = solution.get("r_idxs")
            decorations: Dict[str, Dict[int, intbitset]] = solution.get("decorations")
            ext_edges: List[Tuple[int, Tuple[int, int]]] = self._iteration_ext_edges if r_idxs is None else [ext_state_to_ext_edge.get(viewer.get_ext_state(r_idx)) for r_idx in r_idxs]
            rules_with_decorations: Set[Tuple[dlplan_policy.Rule, FrozenSet[int], FrozenSet[int]]] = self._get_rules_with_decorations(ext_edges, f_idxs, decorations)
            return {
                "cost": solution.get("cost"),
                "f_idxs": f_idxs,
                "rules_with_decorations": rules_with_decorations,
                "conditions_for_goal": solution.get("conditions_for_goal"),
            }
        else:
            raise RuntimeError("NO SOLUTION FOUND FOR SELECTED IDXS")

    def repair(self, solution: Dict[str, Any], reasons: List[Dict[str, Any]]) -> Dict[str, Any]:
        f_idxs: List[int] = solution.get("f_idxs")
        ext_edges: List[Tuple[int, Tuple[int, int]]] = solution.get("ext_edges")
        result: Dict[str, Any] = solution.get("solver").repair(solution, reasons)
        rules_with_decorations: Set[Tuple[dlplan_policy.Rule, FrozenSet[int], FrozenSet[int]]] = self._get_rules_with_decorations(ext_edges, f_idxs, result.get("decorations"))
        return {
            "cost": solution.get("cost"),
            "f_idxs": f_idxs,
            "ext_edges": ext_edges,
            "rules_with_decorations": rules_with_decorations,
            "conditions_for_goal": result.get("conditions_for_goal"),
            "verify_closure": solution.get("verify_closure"),
            "verify_bad_edges": solution.get("verify_bad_edges"),
            "solver": solution.get("solver"),
        }

    def get_rule_for_edge(self, instance_idx: int, edge: Tuple[int, int], feature_pool: List[Feature], f_idxs: Optional[List[int]] = None) -> Any:
        policy_builder: dlplan_core.PolicyBuilder = self._state_factory.domain_data().policy_builder
        src_state_idx, dst_state_idx = edge
        src_feature_values: np.ndarray = self._ext_state_to_feature_valuations.get((instance_idx, src_state_idx))
        dst_feature_values: np.ndarray = self._ext_state_to_feature_valuations.get((instance_idx, dst_state_idx))
        conditions = make_conditions(policy_builder, feature_pool, src_feature_values, f_idxs)
        effects = make_effects(policy_builder, feature_pool, src_feature_values, dst_feature_values, f_idxs)
        return policy_builder.make_rule(conditions, effects)

    ### PRIVATE METHODS

    def _get_rules_with_decorations(self, ext_edges: List[Tuple[int, Tuple[int, int]]], f_idxs: List[int], decorations: Dict[str, Dict[int, intbitset]]) -> Set[Tuple[dlplan_policy.Rule, FrozenSet[int], FrozenSet[int]]]:
        if decorations is None: return None
        rules_with_decorations: Set[Tuple[dlplan_policy.Rule, FrozenSet[int], FrozenSet[int]]] = set()
        for instance_idx, edge in ext_edges:
            unknowns: FrozenSet[int] = frozenset(decorations.get("unknown", dict()).get(instance_idx, dict()).get(edge[0], set()))
            dont_cares: FrozenSet[int] = frozenset(decorations.get("dont_care", dict()).get(instance_idx, dict()).get(edge[0], set()))
            rule: dlplan_policy.Rule = self.get_rule_for_edge(instance_idx, edge, self._feature_pool, f_idxs)
            rules_with_decorations.add((rule, unknowns, dont_cares))
        return rules_with_decorations
 
    def _get_effects_on_features(self, src_feature_values: np.ndarray, dst_feature_values: np.ndarray, relevant_features_idxs: Optional[intbitset] = None) -> Dict[str, intbitset]:
        effects: List[str] = list(map(lambda delta: "inc" if delta > 0 else "eqv" if delta == 0 else "dec", np.sign(dst_feature_values - src_feature_values)))
        unfiltered_effects: Dict[str, intbitset] = {key: intbitset([f_idx for f_idx, chg in enumerate(effects) if chg == key]) for key in ["inc", "eqv", "dec"]}
        if relevant_features_idxs is not None:
            return {key: features & relevant_features_idxs for key, features in unfiltered_effects.items()}
        else:
            return unfiltered_effects

    def _calculate_ext_states_and_edges(self):
        # Classify ext_states
        ext_states_in_relevant: Set[Tuple[int, int]] = frozenset([(instance_idx, state_idx) for instance_idx, state_idxs in self._relevant_vertices.items() for state_idx in state_idxs])
        non_goal_ext_states_in_paths: Set[Tuple[int, int]] = frozenset([(instance_idx, ex_state) for instance_idx, path in self._paths_with_idx for ex_state in path[:-1]])
        goal_ext_states_in_paths: Set[Tuple[int, int]] = frozenset([(instance_idx, path[-1]) for instance_idx, path in self._paths_with_idx])
        ext_states_in_deadend_paths: Set[Tuple[int, int]] = frozenset([ext_state for deadend_path in self._iteration_deadend_paths for ext_state in deadend_path])
        deadend_ext_states: Set[Tuple[int, int]] = frozenset([deadend_path[-1] for deadend_path in self._iteration_deadend_paths])
        non_covered_ext_states: Set[Tuple[int, int]] = frozenset([(instance_idx, state_idx) for instance_idx, state_idxs in self._non_covered_vertices.items() for state_idx in state_idxs])

        # Edges on example paths
        ex_paths: List[Tuple[Tuple[int, int]]] = [tuple((instance_idx, state) for state in path) for instance_idx, path in self._paths_with_idx]
        ext_edges: List[Tuple[int, Tuple[int, int]]] = [(instance_idx, ex_edge) for instance_idx, path in self._paths_with_idx for ex_edge in zip(path[:-1], path[1:])]

        # Make sure ext_state_to_ext_edge includes edges on example paths
        for ext_edge in ext_edges:
            src_ext_state: Tuple[int, int] = (ext_edge[0], ext_edge[1][0])
            if src_ext_state not in self._iteration_ext_state_to_ext_edge:
                self._iteration_ext_state_to_ext_edge[src_ext_state] = ext_edge
            if src_ext_state not in self._ext_state_to_ext_edge:
                self._ext_state_to_ext_edge[src_ext_state] = ext_edge

        # Obtain edges for example ext_states off paths
        for ext_state in non_covered_ext_states:
            instance_idx, state_idx = ext_state
            ext_edge: Tuple[int, Tuple[int, int]] = self._ext_state_to_ext_edge.get(ext_state)
            if ext_edge is None:
                logging.info(f"Get *example* edge for {state_idx}.{self._state_factory.get_dlplan_state(instance_idx, state_idx)}")
                status, plan = self._state_factory.get_plan_for_state_idx(instance_idx, state_idx)
                if len(plan) > 0:
                    edge: Tuple[Tuple[int, Any], Tuple[int, Any]] = self._state_factory.get_state_trajectory_from_state_idx_and_plan(instance_idx, state_idx, plan[:1])
                    ext_edge: Tuple[int, Tuple[int, int]] = (instance_idx, (state_idx, edge[1][0]))
                    logging.info(f"    Target {ext_edge[1][1]}.{self._state_factory.get_dlplan_state(instance_idx, ext_edge[1][1])}")
                else:
                    assert not self._state_factory.is_goal_state(instance_idx, state_idx)
                    ext_edge: Tuple[int, Tuple[int, int]] = (instance_idx, (state_idx, ))
                self._ext_state_to_ext_edge[ext_state] = ext_edge
            self._iteration_ext_state_to_ext_edge[ext_state] = ext_edge

            if len(ext_edge[1]) == 1:
                logging.error(f"ERROR: Unexpected deadend state .. it should have been detected earlier!")
                raise RuntimeError(f"Unexpected deadend state")
            else:
                ext_edges.append(ext_edge)
                logging.info(f"Adding ext edge: {ext_edge}")
                self._state_factory.register_deadend_value(instance_idx, state_idx, False)

        logging.info(f"        ext_states_in_relevant: {sorted(ext_states_in_relevant)}")
        logging.info(f"  non_goal_ext_states_in_paths: {sorted(non_goal_ext_states_in_paths)}")
        logging.info(f"      goal_ext_states_in_paths: {sorted(goal_ext_states_in_paths)}")
        logging.info(f"   ext_states_in_deadend_paths: {sorted(ext_states_in_deadend_paths)}")
        logging.info(f"            deadend_ext_states: {sorted(deadend_ext_states)}")
        logging.info(f"        non_covered_ext_states: {sorted(non_covered_ext_states)}")

        # All ext states classfied into deadends, goal, ex (i.e., example) and alive (i.e. non-example alive)
        self._iteration_ext_states_in_deadend_paths: Set[Tuple[int, int]] = ext_states_in_deadend_paths
        self._iteration_deadend_ext_states: Set[Tuple[int, int]] = deadend_ext_states
        self._iteration_non_covered_ext_states: Set[Tuple[int, int]] = non_covered_ext_states
        self._iteration_goal_ext_states: Set[Tuple[int, int]] = goal_ext_states_in_paths
        self._iteration_non_goal_ext_states: Set[Tuple[int, int]] = non_goal_ext_states_in_paths
        self._iteration_ex_ext_states: Set[Tuple[int, int]] = non_goal_ext_states_in_paths | non_covered_ext_states
        self._iteration_ext_edges: List[Tuple[int, Tuple[int, int]]] = ext_edges
        self._iteration_ex_paths: List[Tuple[Tuple[int, int]]] = ex_paths

        logging.info(f"The following are consolidated ext states:")
        logging.info(f"   ext_states_in_deadend_paths: {sorted(self._iteration_ext_states_in_deadend_paths)}")
        logging.info(f"        non_covered_ext_states: {sorted(self._iteration_non_covered_ext_states)}")
        logging.info(f"            deadend_ext_states: {sorted(self._iteration_deadend_ext_states)}")
        logging.info(f"               goal_ext_states: {sorted(self._iteration_goal_ext_states)}")
        logging.info(f"                 ex_ext_states: {sorted(self._iteration_ex_ext_states)}")
        logging.info(f"                     ext_edges: {sorted(self._iteration_ext_edges)}")
        logging.info(f"                      ex_paths: {sorted(self._iteration_ex_paths)}")

    def _calculate_rule_effects(self, ext_edges: List[Tuple[int, Tuple[int, int]]]) -> Dict[Tuple[int, Tuple[int, int]], Dict[str, intbitset]]:
        rule_effects: Dict[Tuple[int, Tuple[int, int]], Dict[str, intbitset]] = dict()
        for ext_edge in ext_edges:
            instance_idx, (src_state_idx, dst_state_idx) = ext_edge
            assert self._ext_state_to_feature_valuations.get((instance_idx, src_state_idx)) is not None, f"INEXISTENT ENTRY: {(instance_idx, src_state_idx)}; keys: {self._ext_state_to_feature_valuations.keys()}"
            assert self._ext_state_to_feature_valuations.get((instance_idx, dst_state_idx)) is not None, f"INEXISTENT ENTRY: {(instance_idx, dst_state_idx)}; keys: {self._ext_state_to_feature_valuations.keys()}"
            src_feature_values: np.ndarray = self._ext_state_to_feature_valuations.get((instance_idx, src_state_idx))
            dst_feature_values: np.ndarray = self._ext_state_to_feature_valuations.get((instance_idx, dst_state_idx))
            assert ext_edge not in rule_effects
            rule_effects[ext_edge] = self._get_effects_on_features(src_feature_values, dst_feature_values, self._relevant_features_idxs)
        return rule_effects

    def _calculate_indexing_data_structures(self, rule_effects: Dict[Tuple[int, Tuple[int, int]], Dict[str, intbitset]]) -> Dict[str, Any]:
        logging.info(f"Computing indexing data structures...")
        local_timer: Timer = Timer()
        self._timers.resume("indexing")

        numerical_features_idxs: intbitset = intbitset([f_idx for f_idx, feature in self._relevant_features if isinstance(feature.dlplan_feature, dlplan_core.Numerical)])
        features_decreased_by_some_rule: intbitset = intbitset().union(*[effects.get("dec", intbitset()) for effects in rule_effects.values()])
        features_increased_by_some_rule: intbitset = intbitset().union(*[effects.get("inc", intbitset()) for effects in rule_effects.values()])
        features_not_increased_by_any_rule: intbitset = self._relevant_features_idxs - features_increased_by_some_rule
        features_not_decreased_by_any_rule: intbitset = self._relevant_features_idxs - features_decreased_by_some_rule

        ext_states_by_bvalue_on_feature: Dict[Tuple[int, int], Set[Tuple[int, int]]] = defaultdict(set) # ext_states_by_bvalue_on_feature[(f_idx, bvalue)] -> {(instance_idx, state_idx)}
        ext_states_by_change_on_feature: Dict[Tuple[int, str], Set[Tuple[int, int]]] = defaultdict(set) # ext_states_by_change_on_feature[(f_idx, change)] -> {(instance_idx, state_idx)}
        features_by_bvalue_on_ext_state: Dict[Tuple[Tuple[int, int], int], intbitset] = defaultdict(intbitset) # features_by_bvalue_on_ext_state[((instance_idx, state_idx), bvalue)] -> {f_idx}
        features_by_change_on_ext_state: Dict[Tuple[Tuple[int, int], str], intbitset] = defaultdict(intbitset) # features_by_change_on_ext_state[((instance_idx, state_idx), change)] -> {f_idx}

        for ext_state in self._iteration_ex_ext_states | self._iteration_non_covered_ext_states | self._iteration_goal_ext_states | self._iteration_ext_states_in_deadend_paths:
            instance_idx, state_idx = ext_state
            feature_values: np.ndarray = self._ext_state_to_feature_valuations.get(ext_state)
            for f_idx, f_value in enumerate(feature_values):
                if f_idx in self._relevant_features_idxs:
                    f_bvalue = 0 if f_value == 0 else 1
                    ext_states_by_bvalue_on_feature[(f_idx, f_bvalue)].add(ext_state)
                    features_by_bvalue_on_ext_state[(ext_state, f_bvalue)].add(f_idx)

        for ext_edge in self._iteration_ext_edges:
            instance_idx, (src_state_idx, _) = ext_edge
            for key, f_idxs in rule_effects.get(ext_edge).items():
                assert key in ["eqv", "dec", "inc"]
                for f_idx in f_idxs:
                    ext_states_by_change_on_feature[(f_idx, key)].add((instance_idx, src_state_idx))
                    features_by_change_on_ext_state[((instance_idx, src_state_idx), key)].add(f_idx)

        ext_states_by_bvalue_on_feature: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {key: frozenset(ext_states) for key, ext_states in ext_states_by_bvalue_on_feature.items()}
        ext_states_by_change_on_feature: Dict[Tuple[int, str], Set[Tuple[int, int]]] = {key: frozenset(ext_states) for key, ext_states in ext_states_by_change_on_feature.items()}

        local_timer.stop()
        self._timers.stop("indexing")
        logging.info(f"{local_timer.get_elapsed_sec():0.2f} second(s) for indexing")

        return {
            "relevant_features": list(self._relevant_features),
            "relevant_features_idxs": self._relevant_features_idxs,
            "numerical_features_idxs": numerical_features_idxs,
            "features_decreased_by_some_rule": features_decreased_by_some_rule,
            "features_increased_by_some_rule": features_increased_by_some_rule,
            "features_not_increased_by_any_rule": features_not_increased_by_any_rule,
            "features_not_decreased_by_any_rule": features_not_decreased_by_any_rule,
            "ext_states_by_bvalue_on_feature": ext_states_by_bvalue_on_feature,
            "ext_states_by_change_on_feature": ext_states_by_change_on_feature,
            "features_by_bvalue_on_ext_state": features_by_bvalue_on_ext_state,
            "features_by_change_on_ext_state": features_by_change_on_ext_state,
            "ext_state_to_feature_valuations": self._ext_state_to_feature_valuations,
        }

    def _calculate_requirements_for_good_transitions(self, features_by_change_on_ext_state: Dict[Tuple[Tuple[int, int], str], intbitset]) -> Any:
        requirements_for_good_transitions: Dict[Tuple[int, int], intbitset] = dict()
        for ext_state in self._iteration_ex_ext_states:
            changing_features: intbitset = features_by_change_on_ext_state.get((ext_state, "dec"), intbitset()) | features_by_change_on_ext_state.get((ext_state, "inc"), intbitset())
            requirements_for_good_transitions[ext_state] = changing_features & self._relevant_features_idxs
        return requirements_for_good_transitions

    def _calculate_features_that_separate_ext_pairs(self,
                                                    instance_idx_to_ext_pairs: Dict[int, Tuple[int, int]],
                                                    features_by_bvalue_on_ext_state: Dict[Tuple[Tuple[int, int], int], intbitset]) -> Dict[Tuple[int, Tuple[int, int]], intbitset]:
        ext_pair_to_separating_features: Dict[Tuple[int, Tuple[int, int]], intbitset] = dict()
        for instance_idx, pairs in instance_idx_to_ext_pairs.items():
            for pair in pairs:
                ext_state_1: Tuple[int, int] = (instance_idx, pair[0])
                ext_state_2: Tuple[int, int] = (instance_idx, pair[1])
                features_by_bvalue_0_on_ext_state_1: intbitset = features_by_bvalue_on_ext_state.get((ext_state_1, 0), intbitset())
                features_by_bvalue_1_on_ext_state_1: intbitset = features_by_bvalue_on_ext_state.get((ext_state_1, 1), intbitset())
                features_by_bvalue_0_on_ext_state_2: intbitset = features_by_bvalue_on_ext_state.get((ext_state_2, 0), intbitset())
                features_by_bvalue_1_on_ext_state_2: intbitset = features_by_bvalue_on_ext_state.get((ext_state_2, 1), intbitset())
                separating_features: intbitset = (features_by_bvalue_0_on_ext_state_1 & features_by_bvalue_1_on_ext_state_2) | (features_by_bvalue_1_on_ext_state_1 & features_by_bvalue_0_on_ext_state_2)
                ext_pair_to_separating_features[(instance_idx, pair)] = separating_features & self._relevant_features_idxs
        return ext_pair_to_separating_features


    def _calculate_features_that_separate_ext_pairs_v2(self,
                                                       instance_idx_to_state_idx_to_state_idxs: Dict[int, Dict[int, List[int]]],
                                                       features_by_bvalue_on_ext_state: Dict[Tuple[Tuple[int, int], int], intbitset],
                                                       features_by_change_on_ext_state: Dict[Tuple[Tuple[int, int], str], intbitset]) -> Dict[int, Dict[int, List[Tuple[int, intbitset]]]]:
        instance_idx_to_state_idx_to_separating_features: Dict[int, Dict[int, Liat[Tuple[int, intbitset]]]] = defaultdict(lambda: defaultdict(intbitset))
        for instance_idx, state_idx_to_state_idxs in instance_idx_to_state_idx_to_state_idxs.items():
            for state_idx_1, state_idxs in state_idx_to_state_idxs.items():
                ext_state_1: Tuple[int, int] = (instance_idx, state_idx_1)
                for state_idx_2 in state_idxs:
                    ext_state_2: Tuple[int, int] = (instance_idx, state_idx_2)
                    features_by_bvalue_0_on_ext_state_1: intbitset = features_by_bvalue_on_ext_state.get((ext_state_1, 0), intbitset())
                    features_by_bvalue_1_on_ext_state_1: intbitset = features_by_bvalue_on_ext_state.get((ext_state_1, 1), intbitset())
                    features_by_bvalue_0_on_ext_state_2: intbitset = features_by_bvalue_on_ext_state.get((ext_state_2, 0), intbitset())
                    features_by_bvalue_1_on_ext_state_2: intbitset = features_by_bvalue_on_ext_state.get((ext_state_2, 1), intbitset())
                    separating_features: intbitset = (features_by_bvalue_0_on_ext_state_1 & features_by_bvalue_1_on_ext_state_2) | (features_by_bvalue_1_on_ext_state_1 & features_by_bvalue_0_on_ext_state_2)
                    instance_idx_to_state_idx_to_separating_features[instance_idx][state_idx_1].append((state_idx_2, separating_features & self._relevant_features_idxs))
        return instance_idx_to_state_idx_to_separating_features

    # DEPRECATED: It isn't used internally...
    def _calculate_features_that_change_differently_at_ext_pairs(self,
                                                                 ext_pairs: Set[Tuple[int, Tuple[int, int]]],
                                                                 features_by_bvalue_on_ext_state: Dict[Tuple[Tuple[int, int], int], intbitset],
                                                                 features_by_change_on_ext_state: Dict[Tuple[Tuple[int, int], str], intbitset]) -> Dict[Tuple[int, Tuple[int, int]], intbitset]:
        ext_pair_to_separating_features: Dict[Tuple[int, Tuple[int, int]], intbitset] = dict()
        for instance_idx, pair in ext_pairs:
            ext_states: List[Tuple[int, int]] = [(instance_idx, pair[i]) for i in [0, 1]]
            features_by_bvalue_on_ext_states: List[List[intbitset]] = [[features_by_bvalue_on_ext_state.get((ext_state[i], bvalue), intbitset()) for bvalue in [0, 1]] for i in [0, 1]]
            features_by_change_on_ext_states: List[List[intbitset]] = [[features_by_change_on_ext_state.get((ext_states[i], change), intbitset()) for change in ["dec", "inc", "eqv"]] for i in [0, 1]]

            features_with_different_bvalue: intbitset = features_by_bvalue_on_ext_states[0][0] & features_by_bvalue_on_ext_states[1][1]
            features_with_different_bvalue |= features_by_bvalue_on_ext_states[0][1] & features_by_bvalue_on_ext_states[1][0]

            features_that_change_differently: intbitset = features_by_change_on_ext_states[0][0] & features_by_change_on_ext_states[1][1]
            features_that_change_differently |= features_by_change_on_ext_states[0][0] & features_by_change_on_ext_states[1][2]
            features_that_change_differently |= features_by_change_on_ext_states[0][1] & features_by_change_on_ext_states[1][0]
            features_that_change_differently |= features_by_change_on_ext_states[0][1] & features_by_change_on_ext_states[1][2]
            features_that_change_differently |= features_by_change_on_ext_states[0][2] & features_by_change_on_ext_states[1][0]
            features_that_change_differently |= features_by_change_on_ext_states[0][2] & features_by_change_on_ext_states[1][1]
            separating_features: intbitset = features_by_change_on_ext_states | features_that_change_differently
            ext_pair_to_separating_features[(instance_idx, pair)] = separating_features
        return ext_pair_to_separating_features

    # DEPRECATED BY _calculate_separating_features_for_deadend_paths_v2()
    def _calculate_separating_features_for_deadend_paths(self,
                                                         features_by_change_on_ext_state: Dict[Tuple[Tuple[int, int], str], intbitset]) -> Dict[Tuple[Tuple[int, int]], List[intbitset]]:
        logging.info(f"Deadend paths: {self._iteration_deadend_paths}")
        deadend_path_to_separating_features: Dict[Tuple[Tuple[int, int]], List[intbitset]] = dict()
        for path in self._iteration_deadend_paths:
            # For each deadend path, look for *last* transition (S,S') in path such that S is an example state.
            # Create a requirement that "separates" (S,S') from # current example transitions.
            logging.debug(f"Deadend path: {path}")
            for transition in zip(path[:-1], path[1:]):
                marked_ext_edge: Tuple[int, Tuple[int, int]] = (transition[0][0], (transition[0][1], transition[1][1]))
                if marked_ext_edge in self._iteration_ext_edges:
                    logging.debug(f"  Ext-edge: {marked_ext_edge}")
                else:
                    logging.info(f"  Ext-edge: {marked_ext_edge}*")
                    break

            # Feature changes across marked ext-edge
            instance_idx, (src_state_idx, dst_state_idx) = marked_ext_edge
            src_feature_values: np.ndarray = self._ext_state_to_feature_valuations.get((instance_idx, src_state_idx))
            dst_feature_values: np.ndarray = self._ext_state_to_feature_valuations.get((instance_idx, dst_state_idx))
            feature_changes_on_marked_ext_edge: Dict[str, intbitset] = self._get_effects_on_features(src_feature_values, dst_feature_values, self._relevant_features_idxs)
            features_by_change_on_marked_ext_edge: List[intbitset] = [feature_changes_on_marked_ext_edge.get(change) for change in ["dec", "inc", "eqv"]]
            logging.info(f"feature_changes_on_marked_ext_edge: {feature_changes_on_marked_ext_edge}")

            # Requirement that separates marked ext-edges from ext-edges in policy
            requirements: List[intbitset] = []
            for ext_edge in self._iteration_ext_edges:
                ext_state: Tuple[int, int] = (ext_edge[0], ext_edge[1][0])
                features_by_change_on_ext_edge: List[List[intbitset]] = [features_by_change_on_ext_state.get((ext_state, change), intbitset()) for change in ["dec", "inc", "eqv"]]
                requirement: intbitset = intbitset().union(*[features_by_change_on_marked_ext_edge[i] - features_by_change_on_ext_edge[i] for i in [0, 1, 2]])
                requirements.append(requirement)
                logging.debug(f"Ext-state: {ext_state}, requirement={requirement}")
            deadend_path_to_separating_features[path] = requirements
            assert all([len(requirement) > 0 for requirement in requirements])
        return deadend_path_to_separating_features

    def _calculate_separating_features_for_deadend_paths_v2(self,
                                                            features_by_change_on_ext_state: Dict[Tuple[Tuple[int, int], str], intbitset]) -> Tuple[bool, Dict[Tuple[int, int], Dict[Tuple[int, Tuple[int, int]], intbitset]], Any]:
        logging.info(f"Calculating separating features for {len(self._iteration_deadend_paths)} deadend path(s) ...")
        ext_state_to_separating_features: Dict[Tuple[int, int], Dict[Tuple[int, Tuple[int, int]], intbitset]] = defaultdict(lambda: defaultdict(intbitset))
        marked_ext_edges: Set[Tuple[int, Tuple[int, int]]] = set()
        for path_id, path in enumerate(self._iteration_deadend_paths):
            logging.info(f"Deadend path: {path_id}.{path}")
            assert len(path) > 1

            # Marked edge to separate from example edges is last edge in path
            instance_idx: int = path[0][0]
            marked_ext_edge: Tuple[int, Tuple[int, int]] = (instance_idx, (path[-2][1], path[-1][1]))
            marked_ext_edges.add(marked_ext_edge)
            logging.info(f"  Mark ext-edge={marked_ext_edge}")

            # Feature changes across marked ext-edge
            instance_idx, (src_state_idx, dst_state_idx) = marked_ext_edge
            src_feature_values: np.ndarray = self._ext_state_to_feature_valuations.get((instance_idx, src_state_idx))
            dst_feature_values: np.ndarray = self._ext_state_to_feature_valuations.get((instance_idx, dst_state_idx))
            feature_changes_on_marked_ext_edge: Dict[str, intbitset] = self._get_effects_on_features(src_feature_values, dst_feature_values, self._relevant_features_idxs)
            features_by_change_on_marked_ext_edge: List[intbitset] = [feature_changes_on_marked_ext_edge.get(change) for change in ["dec", "inc", "eqv"]]
            logging.debug(f"    feature_changes_on_marked_ext_edge: {dict([(key, feature_changes_on_marked_ext_edge.get(key)) for key in ['dec', 'inc']])}")

            # Requirement that separates marked ext-edges from ext-edges in policy
            for ext_edge in self._iteration_ext_edges:
                ext_state: Tuple[int, int] = (ext_edge[0], ext_edge[1][0])
                features_by_change_on_ext_edge: List[List[intbitset]] = [features_by_change_on_ext_state.get((ext_state, change), intbitset()) for change in ["dec", "inc", "eqv"]]
                requirement: intbitset = intbitset().union(*[features_by_change_on_marked_ext_edge[i] - features_by_change_on_ext_edge[i] for i in [0, 1, 2]])
                assert marked_ext_edge not in ext_state_to_separating_features[ext_state] or ext_state_to_separating_features[ext_state][marked_ext_edge] == requirement
                ext_state_to_separating_features[ext_state][marked_ext_edge] = requirement
                logging.debug(f"    Ext-state: {ext_state}, requirement={requirement}")
                if len(requirement) == 0: return False, None, marked_ext_edges
        return True, ext_state_to_separating_features, marked_ext_edges

    def _preprocess_termination(self, monotone_only_by_dec: bool, rule_elimination: bool, **kwargs) -> Dict[str, Any]:
        logging.info(f"Preprocessing termination of features...")
        self._timers.resume("preprocessing")
        self._timers.resume("feature/termination")
        self._timers.resume("pricing/preprocessing")
        local_timer: Timer = Timer()

        # Define relevant edges: those in paths and those for off-path relevant vertices
        self._calculate_ext_states_and_edges()

        # Compute relevant features
        if self._relevant_features is None or kwargs.get("recompute_relevant_features", False):
            remove_features_with_constant_bvalue: bool = True
            self._relevant_features: List[Tuple[int, Feature]] = self._get_relevant_features(remove_features_with_constant_bvalue)
            self._relevant_features_idxs: intbitset = intbitset([f_idx for f_idx, _ in self._relevant_features])
            self._f_idx_to_feature_index: Dict[int, int] = {f_idx: index for index, (f_idx, feature) in enumerate(self._relevant_features)}
            logging.info(f"{len(self._relevant_features)} relevant feature(s)")

        # Extract change of relevant features along paths
        rule_effects: Dict[Tuple[int, Tuple[int, int]], Dict[str, intbitset]] = self._calculate_rule_effects(self._iteration_ext_edges)

        # Calculate indexing data structures
        ds: Dict[str, Any] = self._calculate_indexing_data_structures(rule_effects)

        # Calculate companions m_pairs
        lazy: bool = False
        if not rule_elimination:
            m_pairs: MPairs = MPairs(ds, monotone_only_by_dec=monotone_only_by_dec, lazy=lazy, timers=self._timers)
            viewer: RuleViewer = None
        else:
            # Calculate monotonicity counters
            m_pairs = None
            viewer: RuleViewer = RuleViewer(ds, monotone_only_by_dec=monotone_only_by_dec, timers=self._timers)

        if not rule_elimination:
            features_by_bvalue_on_ext_state: Dict[Tuple[Tuple[int, int], int], intbitset] = m_pairs._features_by_bvalue_on_ext_state
            features_by_change_on_ext_state: Dict[Tuple[Tuple[int, int], str], intbitset] = m_pairs._features_by_change_on_ext_state
            monotone_features: intbitset = m_pairs.monotone_features()

            if lazy:
                #usable_features: intbitset = m_pairs._relevant_features_idxs
                fg_companions: Dict[int, intbitset] = None
                gf_companions: Dict[int, intbitset] = None
            else:
                # Remove non-terminating features
                usable_features: intbitset = m_pairs.usable_features()
                pruned_relevant_features: List[Tuple[int, Feature]] = [(f_idx, feature) for f_idx, feature in self._relevant_features if f_idx in usable_features]
                logging.info(f"{len(pruned_relevant_features)} feature(s) after removing non-terminating features (num_pruned={len(self._relevant_features) - len(pruned_relevant_features)})")
                logging.info(f"{len(monotone_features)} monotone feature(s)")
                self._relevant_features: List[Tuple[int, Feature]] = pruned_relevant_features
                self._relevant_features_idxs: intbitset = intbitset([f_idx for f_idx, _ in self._relevant_features])
                self._f_idx_to_feature_index: Dict[int, int] = {f_idx: index for index, (f_idx, feature) in enumerate(self._relevant_features)}
            assert m_pairs._relevant_features_idxs == self._relevant_features_idxs
        else:
            features_by_bvalue_on_ext_state: Dict[Tuple[Tuple[int, int], int], intbitset] = viewer._features_by_bvalue_on_ext_state
            features_by_change_on_ext_state: Dict[Tuple[Tuple[int, int], str], intbitset] = viewer._features_by_change_on_ext_state
            #usable_features: intbitset = viewer._relevant_features_idxs
            assert viewer._relevant_features_idxs == self._relevant_features_idxs

        # Requirements for good transitions
        requirements_for_good_transitions = self._calculate_requirements_for_good_transitions(features_by_change_on_ext_state)
        logging.info(f"{len(requirements_for_good_transitions)} requirement(s) for good transitions")

        # Features that separate goal from non-goal states in example paths
        goal_vertices: Dict[int, intbitset] = defaultdict(intbitset)
        for instance_idx, state_idx in self._iteration_goal_ext_states:
            goal_vertices[instance_idx].add(state_idx)

        non_goal_vertices: Dict[int, intbitset] = defaultdict(intbitset)
        for instance_idx, state_idx in self._iteration_ex_ext_states | self._iteration_non_covered_ext_states | self._iteration_ext_states_in_deadend_paths:
            non_goal_vertices[instance_idx].add(state_idx)

        instance_idx_to_goal_state_idx_to_state_idxs: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        instance_idx_to_goal_and_non_goal_pairs: Dict[int, Set[Tuple[int, int]]] = dict()
        for instance_idx, goal_state_idxs in goal_vertices.items():
            instance_idx_to_goal_and_non_goal_pairs[instance_idx] = frozenset(product(goal_state_idxs, non_goal_vertices.get(instance_idx, [])))
        for instance_idx, goal_state_idxs in goal_vertices.items():
            for goal_state_idx in goal_state_idxs:
                instance_idx_to_goal_state_idx_to_state_idxs[instance_idx][goal_state_idx] = frozenset(non_goal_vertices.get(instance_idx, []))
        logging.info(f"{sum([len(pairs) for pairs in instance_idx_to_goal_and_non_goal_pairs.values()])} goal-and-non-goal state pairs")

        goal_ext_pair_to_separating_features: Dict[Tuple[int, Tuple[int, int]], intbitset] = self._calculate_features_that_separate_ext_pairs(instance_idx_to_goal_and_non_goal_pairs,
                                                                                                                                              features_by_bvalue_on_ext_state)
        #ext_state_to_separating_features_for_goals: Dict[Tuple[int, int], List[intbitset]] = self._calculate_features_that_separate_ext_pairs_v2(instance_idx_to_goal_state_idx_to_state_idxs,
        #                                                                                                                                         features_by_bvalue_on_ext_state)
        goal_separating_features: List[intbitset] = list(goal_ext_pair_to_separating_features.values())

        # Calculate separating features for discovered deadends
        ext_state_to_separating_features_for_deadend_paths: Dict[Tuple[int, int], Dict[Tuple[int, Tuple[int, int]], intbitset]] = None
        bad_ext_edges: Set[Tuple[int, Tuple[int, int]]] = set()
        status, ext_state_to_separating_features_for_deadend_paths, bad_ext_edges = self._calculate_separating_features_for_deadend_paths_v2(features_by_change_on_ext_state)
        if not status:
            logging.warning(f"No feature to separate transition to deadend state")
            raise RuntimeError(f"No feature to separate transition to deadend state")
        deadend_separating_features: List[intbitset] = [separating_features for separating_features_for_deadend_paths in ext_state_to_separating_features_for_deadend_paths.values() for separating_features in separating_features_for_deadend_paths.values()]

        # Calculate features that separate sibling edges (if applicable)
        ext_sibling_to_separating_features: Dict[Tuple[int, int, Tuple[int, int]], intbitset] = dict()
        ext_successors: Dict[Tuple[int, int], List[Tuple[str, int]]] = defaultdict(list)
        if rule_elimination:
            for ext_edge in self._iteration_ext_edges:
                instance_idx, (src_state_idx, dst_state_idx) = ext_edge
                ext_state: Tuple[int, int] = (instance_idx, src_state_idx)
                instance_data: PDDLInstance = self._benchmark._instance_datas[instance_idx]
                successors: List[Tuple[Tuple[int, Any], str]] = instance_data.get_successors(src_state_idx)

                # Store successor information
                for (succ_state_idx, succ_state), action in successors:
                    ext_successors[ext_state].append((action, succ_state_idx))

        local_timer.stop()
        self._timers.stop("pricing/preprocessing")
        self._timers.stop("feature/termination")
        self._timers.stop("preprocessing")
        logging.info(f"{local_timer.get_elapsed_sec():0.2f} second(s) for preprocessing termination")

        return {
            "width": self._width,
            "feature_pool": self._feature_pool,
            "relevant_features": self._relevant_features,
            "f_idx_to_feature_index": self._f_idx_to_feature_index,
            "lower_bound_on_rank": None,
            "requirements_for_good_transitions": requirements_for_good_transitions,
            "ext_state_to_ext_edge": self._iteration_ext_state_to_ext_edge,
            "ext_state_to_feature_valuations": self._ext_state_to_feature_valuations,
            "goal_ext_pair_to_separating_features": goal_ext_pair_to_separating_features,
            "ext_state_to_separating_features_for_deadend_paths": ext_state_to_separating_features_for_deadend_paths,
            "bad_ext_edges": bad_ext_edges,
            "ext_sibling_to_separating_features": ext_sibling_to_separating_features,
            "m_pairs": m_pairs,
            "rule_viewer": viewer,
            "ex_ext_states": self._iteration_ex_ext_states,
            "ext_successors": ext_successors,
            "goal_ext_states": self._iteration_goal_ext_states,
            "non_goal_ext_states": self._iteration_non_goal_ext_states,
            "ex_paths": self._iteration_ex_paths,
        }

    def _get_relevant_features(self, remove_features_with_constant_bvalue: bool = False) -> List[Tuple[int, Feature]]:
        logging.info(f"Pruning redundant fetures...")
        features: List[Feature] = self._feature_pool

        dlplan_state_pairs: List[Tuple[dlplan_core.State, dlplan_core.State]] = []
        for (instance_idx, (src_state_idx, dst_state_idx)) in self._iteration_ext_edges:
            src_dlplan_state: dlplan_core.State = self._benchmark.get_dlplan_state(instance_idx, src_state_idx)
            dst_dlplan_state: dlplan_core.State = self._benchmark.get_dlplan_state(instance_idx, dst_state_idx)
            dlplan_state_pairs.append((src_dlplan_state, dst_dlplan_state))
        logging.info(f"{len(dlplan_state_pairs)} pair(s) of dlplan states")

        selected_features_v2: List[Tuple[int, Feature]] = prune_features_with_same_feature_change_AND_boolean_valuation_reduced_v2(self._state_factory,
                                                                                                                                   dlplan_state_pairs,
                                                                                                                                   self._instance_idx_to_denotations_caches,
                                                                                                                                   features,
                                                                                                                                   remove_features_with_constant_bvalue)
        num_pruned_v2 = len(features) - len(selected_features_v2)
        logging.info(f"{len(selected_features_v2)} feature(s) after pruning features with same change across transitions AND same boolean valuation (num_pruned={num_pruned_v2})")

        return selected_features_v2

