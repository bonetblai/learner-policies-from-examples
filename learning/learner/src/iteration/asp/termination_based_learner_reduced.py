import os
import logging
import tempfile
import numpy as np

# Bitset-based unordered sets of unsigned integers
from intbitset import intbitset

from termcolor import colored
from typing import Set, Tuple, List, Union, Dict, Any, Optional, Union
from collections import OrderedDict, defaultdict, deque
from itertools import product
from pathlib import Path

import dlplan.core as dlplan_core
import dlplan.policy as dlplan_policy

from ..feature_pool import Feature
from ..feature_pool_utils import prune_features_with_same_feature_change_AND_boolean_valuation_reduced_v1, prune_features_with_same_feature_change_AND_boolean_valuation_reduced_v2
from ..state_pair_equivalence_utils import make_conditions, make_effects
from ..iteration_data import IterationData
from ..statistics import Statistics
from ...util import change_dir, Timer

from .asp_solver import ASPSolver
from .returncodes import ClingoExitCode

from ...state_space import PDDLInstance, StateFactory, get_plan, get_plan_v2

LIST_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

from .m_pairs import MPairs
from .m_pairs_contextual import MPairsContextual
from .greedy_solver import GreedySolver
from .greedy_solver_contextual import GreedySolverContextual
from .greedy_solver_contextual_alt import GreedySolverContextualAlt


class TerminationBasedLearnerReduced:
    def __init__(self,
                 state_factory: StateFactory,
                 disable_greedy_solver: bool = False,
                 disable_optimization_decorations: bool = False,
                 solver_prefix: Optional[str] = None,
                 timers: Optional[Statistics] = None,
                 **kwargs):
        self._state_factory: StateFactory = state_factory
        self._disable_greedy_solver = disable_greedy_solver
        self._disable_optimization_decorations = disable_optimization_decorations
        self._solver_prefix = solver_prefix
        self._using_tarski = state_factory.using_tarski()
        self._instance_datas: List[PDDLInstance] = state_factory._instances
        self._preprocessed_datas: List[Dict[str, Any]] = None
        self._timers = timers if timers is not None else Statistics()
        assert solver_prefix == "termination9"

        # Stores edges for ext_states found by planner to avoid calling planning multiple times from same state
        self._ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]] = dict()

    def preprocess_instances(self, planner: str) -> List[int]:
        self._timers.resume("preprocessing")
        self._timers.resume("planner")

        # Create folder for storing plans (preprocessing)
        folder_for_plans: Path = Path(f"plans.{planner}")
        folder_for_plans.mkdir(exist_ok=True)

        # Read plans from existing files (if possible)
        self._preprocessed_datas: List[Dict[str, Any]] = []
        for instance_data in self._instance_datas:
            instance_filepath: Path = Path(instance_data.instance_filepath())
            plan_filepath: Path = folder_for_plans / f"{instance_filepath.name}.plan.{planner}"
            preprocessed_data: Dict[str, Any] = {
                "instance_idx": instance_data.idx,
                "instance_filepath": instance_filepath,
                "plan_filepath": plan_filepath,
            }
            if plan_filepath.exists():
                with plan_filepath.open("r") as fd:
                    plan: List[str] = [line.strip() for line in fd.readlines()]
                is_unsolvable: bool = "NO-PLAN" in plan
                preprocessed_data["planner_output"] = (not is_unsolvable, plan)
                preprocessed_data["read_from_file"] = True
                preprocessed_data["complete"] = True
            else:
                preprocessed_data["read_from_file"] = False
                preprocessed_data["complete"] = False
            self._preprocessed_datas.append(preprocessed_data)

        # For remaining instances, preprocess them
        family_name: str = self._state_factory.family_name()
        temp_folder = tempfile.TemporaryDirectory(prefix=f"{family_name}.preprocess_instances.", dir=".")
        logging.info(f"Preprocessing instances in folder '{temp_folder.name}'...")
        with change_dir(temp_folder.name):
            for preprocessed_data in self._preprocessed_datas:
                instance_idx: int = preprocessed_data.get("instance_idx")
                if not preprocessed_data.get("complete", False):
                    data: Dict[str, Any] = self._preprocess_instance(temp_folder.name, instance_idx, planner, remove_files=False)
                    assert preprocessed_data.get("plan_filepath").name == data.get("plan_filepath").name, (preprocessed_data, data)
                    preprocessed_data["planner_output"] = data.get("planner_output")
                    preprocessed_data["read_from_file"] = False
                    preprocessed_data["complete"] = True

        # Create folder with plans for instances
        for preprocessed_data in self._preprocessed_datas:
            plan_filename: str = preprocessed_data.get("plan_filepath").name
            if (Path(temp_folder.name) / plan_filename).exists() and not (folder_for_plans / plan_filename).exists():
                (Path(temp_folder.name) / plan_filename).rename(folder_for_plans / plan_filename)

        # Cleanup temporaty folder
        temp_folder.cleanup()

        # Compute idxs to be removed
        idxs_to_be_removed: List[int] = []
        for instance_idx, preprocessed_data in enumerate(self._preprocessed_datas):
            assert instance_idx == preprocessed_data.get("instance_idx")
            planner_output: Optional[Tuple[bool, List[str]]] = preprocessed_data.get("planner_output")
            if planner_output is None:
                idxs_to_be_removed.append(instance_idx)
            else:
                status, plan = planner_output
                if not status or len(plan) == 0:
                    # If plan visit the same state twice, we should remove it.
                    # However, we don't have the trajectory here...
                    idxs_to_be_removed.append(instance_idx)

        # State maps for creation of facts
        self._ext_state_to_global_state_index: Dict[Tuple[int, int], int] = dict()
        self._global_state_index_to_ext_state: Dict[int, Tuple[int, int]] = dict()

        self._timers.stop("preprocessing")
        self._timers.stop("planner")
        return idxs_to_be_removed

    def remove_and_reorder_instances(self, sorted_instance_idxs: List[int]):
        self._timers.resume("preprocessing")
        assert len(self._instance_datas) == len(self._preprocessed_datas)
        for instance_data, preprocessed_data in zip(self._instance_datas, self._preprocessed_datas):
            assert instance_data.idx == preprocessed_data.get("instance_idx")
            assert instance_data.instance_filepath() == preprocessed_data.get("instance_filepath")

        preprocessed_datas: List[Dict[str, Any]] = [self._preprocessed_datas[instance_idx] for instance_idx in sorted_instance_idxs]
        for instance_idx, preprocessed_data in enumerate(preprocessed_datas):
            preprocessed_data["instance_idx"] = instance_idx
        self._preprocessed_datas: List[Dict[str, Any]] = preprocessed_datas

        self._state_factory.remove_and_reorder_instances(sorted_instance_idxs)
        self._instance_datas = self._state_factory._instances

        assert len(self._instance_datas) == len(self._preprocessed_datas)
        for instance_data, preprocessed_data in zip(self._instance_datas, self._preprocessed_datas):
            assert instance_data.idx == preprocessed_data.get("instance_idx")
            assert instance_data.instance_filepath() == preprocessed_data.get("instance_filepath")
        self._timers.stop("preprocessing")

    def non_loopy_state_trajectories(self, instance_idx: int) -> List[bool]:
        self._timers.resume("preprocessing")
        revisits_states: List[bool] = self._revisits_states_in_low_level_trajectory(instance_idx)
        self._timers.stop("preprocessing")
        return [not value for value in revisits_states]

    def preprocess_instances_final(self):
        self._timers.resume("preprocessing")
        for instance_idx, preprocessed_data in enumerate(self._preprocessed_datas):
            assert instance_idx == preprocessed_data.get("instance_idx")
            self._preprocess_instance_final(preprocessed_data)
        self._timers.stop("preprocessing")

    def prepare_iteration(self, iteration_data: IterationData):
        logging.info(colored(f"Preparing iteration...", "blue"))

        # DLPlan state and feature evaluations for relevant states
        self._iteration_ext_state_to_dlplan_state_index: Dict[Tuple[int, int], int] = iteration_data.ext_state_to_dlplan_state_index
        self._iteration_ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = iteration_data.ext_state_to_feature_valuations
        self._iteration_instance_idx_to_denotations_caches: Dict[int, dlplan_core.DenotationsCaches] = iteration_data.instance_idx_to_denotations_caches
        self._iteration_ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]] = dict()

        # Feature pool
        self._iteration_feature_pool: List[Feature] = iteration_data.feature_pool
        self._iteration_numerical_features: List[Tuple[int, Feature]] = [(f_idx, feature) for f_idx, feature in enumerate(self._iteration_feature_pool) if isinstance(feature.dlplan_feature, dlplan_core.Numerical)]
        self._iteration_boolean_features: List[Tuple[int, Feature]] = [(f_idx, feature) for f_idx, feature in enumerate(self._iteration_feature_pool) if isinstance(feature.dlplan_feature, dlplan_core.Boolean)]
        logging.info(f"Features: #total={len(self._iteration_feature_pool)}, #numerical={len(self._iteration_numerical_features)}, #boolean={len(self._iteration_boolean_features)}")

        # Paths and vertices
        self._iteration_paths_with_idxs: List[Tuple[int, Tuple[int]]] = iteration_data.paths_with_idxs
        self._iteration_relevant_vertices: Dict[int, Set[int]] = iteration_data.relevant_vertices
        self._iteration_non_covered_vertices: Dict[int, Set[int]] = iteration_data.non_covered_vertices
        self._iteration_vertices_in_deadend_paths: Dict[int, Set[int]] = iteration_data.vertices_in_deadend_paths
        self._iteration_deadend_paths: List[Tuple[Tuple[int, int]]] = iteration_data.deadend_paths

        # Data filled by preprocess termination
        self._relevant_features: List[Tuple[int, Feature]] = None
        self._f_idx_to_feature_index: Dict[int, int] = None
        self._preprocessing_data: Dict[str, Any] = None

    def solve(self, **kwargs) -> Tuple[bool, Set[int], Set[Tuple[Any, Set[int], Set[int]]], int]:
        # Other arguments
        timeout_in_seconds_per_step: Optional[float] = kwargs.get("timeout_in_seconds_per_step")
        timeout_in_seconds: Optional[float] = kwargs.get("timeout_in_seconds")
        if timeout_in_seconds_per_step is not None and timeout_in_seconds is None:
            timeout_in_seconds = 60 * timeout_in_seconds_per_step
        simplify_policy: bool = kwargs.get("simplify_policy", False)
        simplify_only_conditions: bool = kwargs.get("simplify_only_conditions", False)
        separate_siblings: bool = kwargs.get("separate_siblings", False)
        contextual: bool = kwargs.get("contextual", False)
        monotone_only_by_dec: bool = kwargs.get("monotone_only_by_dec", False)
        uniform_costs: bool = kwargs.get("uniform_costs", False)
        verbose: bool = kwargs.get("verbose", False)
        dump_asp_program: bool = kwargs.get("dump_asp_program", False)
        previous_feature_rank: int = kwargs.get("previous_feature_rank")

        # Print paths
        for instance_idx, path in self._iteration_paths_with_idxs:
            logging.info(f"Path: instance_idx={instance_idx}, path={path}, size={len(path)}")
            instance_data: PDDLInstance = self._instance_datas[instance_idx]
            assert instance_data.idx == instance_idx

            trajectory: List[Tuple[int, str]] = self._preprocessed_datas[instance_idx]["trajectories"][0]
            for i, (state_idx, action) in enumerate(trajectory):
                logging.info(colored(f"  {state_idx}.{instance_data.get_dlplan_state(*instance_data.get_state(state_idx))}", "blue"))
                if action != "GOAL":
                    logging.info(colored(f"  {action}", "green"))

        # Preprocess termination
        self._timers.resume("preprocessing")
        self._preprocessing_data: Dict[str, Any] = self._preprocess_termination(contextual, monotone_only_by_dec, separate_siblings)
        self._relevant_features: List[Tuple[int, Feature]] = self._preprocessing_data.get("relevant_features")
        self._f_idx_to_feature_index: Dict[int, int] = self._preprocessing_data.get("f_idx_to_feature_index")
        self._timers.stop("preprocessing")

        # Greedy solver
        logging.info(f"INITIALIZING SOLVER...")
        options_for_greedy_solver: Dict[str, Any] = {
            "simplify_policy": simplify_policy,
            "simplify_only_conditions": simplify_only_conditions,
            "uniform_costs": uniform_costs,
            "monotone_only_by_dec": monotone_only_by_dec,
            "dump_asp_program": dump_asp_program,
            "verify_termination": False, #True,
            "optimality": False,
        }
        if not contextual:
            greedy_solver: GreedySolver = GreedySolver(self._preprocessing_data, self._state_factory, **options_for_greedy_solver)
        else:
            #greedy_solver: GreedySolverContextual = GreedySolverContextual(self._preprocessing_data, self._state_factory, **options_for_greedy_solver)
            greedy_solver: GreedySolverContextualAlt = GreedySolverContextualAlt(self._preprocessing_data, self._state_factory, **options_for_greedy_solver)
        logging.info(f"DONE")

        if True:
            self._timers.resume("pricing/algorithm")
            status, chosen, cost, decorations, feature_ranks = greedy_solver.solve(**options_for_greedy_solver)
            self._timers.stop("pricing/algorithm")
        else:
            status, chosen, cost, decorations, feature_ranks = greedy_solver.dfs_solve(**options_for_greedy_solver)

        if options_for_greedy_solver.get("verify_termination", False):
            terminates: bool = self._verify_termination(self._relevant_features, feature_ranks, self._f_idx_to_feature_index, self._solver_prefix, dump_asp_program)
            if not terminates:
                logging.error(colored(f"Solution {sorted(chosen)} does not pass termination test", "red"))
                raise RuntimeError(f"Solution does not pass termination test")

        if status:
            # Construct solution as set of rules paired with unknowns and don't cares.
            # Each path edge generates one rule that is paired with unknowns/don't_cares for source states (if any)
            logging.info(f"Constructing policy...")
            rules_with_decorations: Set[Any, Set[int], Set[int]] = set()
            for instance_idx, edge in self._iteration_ext_edges:
                unknowns: Set[int] = frozenset(decorations.get("unknown", dict()).get(instance_idx, dict()).get(edge[0], set()))
                dont_cares: Set[int] = frozenset(decorations.get("dont_care", dict()).get(instance_idx, dict()).get(edge[0], set()))
                rule: Any = self.get_rule_for_edge(instance_idx, edge, self._iteration_feature_pool, f_idxs=sorted(chosen))
                rules_with_decorations.add((rule, unknowns, dont_cares))
            return True, chosen, rules_with_decorations, None
        else:
            return False, None, None, None

    def get_rule_for_edge(self, instance_idx: int, edge: Tuple[int, int], feature_pool: List[Feature], f_idxs: Optional[List[int]] = None) -> Any:
        policy_builder: dlplan_core.PolicyBuilder = self._state_factory.domain_data().policy_builder
        src_state_idx, dst_state_idx = edge
        src_feature_values: np.ndarray = self._iteration_ext_state_to_feature_valuations.get((instance_idx, src_state_idx))
        dst_feature_values: np.ndarray = self._iteration_ext_state_to_feature_valuations.get((instance_idx, dst_state_idx))
        conditions = make_conditions(policy_builder, feature_pool, src_feature_values, f_idxs)
        effects = make_effects(policy_builder, feature_pool, src_feature_values, dst_feature_values, f_idxs)
        return policy_builder.make_rule(conditions, effects)

    ### PRIVATE METHODS

    def _preprocess_instance(self, folder_name: str, instance_idx: int, planner: str, remove_files: bool = True) -> Dict[str, Any]:
        instance_data: PDDLInstance = self._instance_datas[instance_idx]
        assert instance_data.idx == instance_idx

        domain_filepath: Path = instance_data.domain_filepath()
        instance_filepath: Path = instance_data.instance_filepath()
        plan_filepath: Path = Path(f"{instance_filepath.name}.plan.{planner}")
        #status, plan = get_plan(domain_filepath, instance_filepath, plan_filepath, planner, remove_files=remove_files)
        status, plan = get_plan_v2(domain_filepath, instance_filepath, plan_filepath, planner, remove_files=remove_files)

        return {
            "instance_idx": instance_idx,
            "instance_filepath": Path(instance_filepath),
            "plan_filepath": plan_filepath,
            "planner_output": (status, plan),
        }

    def _revisits_states_in_low_level_trajectory(self, instance_idx: int) -> List[bool]:
        preprocessed_data: Dict[str, Any] = self._preprocessed_datas[instance_idx]
        assert instance_idx == preprocessed_data.get("instance_idx")
        assert self._instance_datas[instance_idx].instance_filepath() == preprocessed_data.get("instance_filepath")

        if preprocessed_data.get("revisits_states") is None:
            status, plan = preprocessed_data.get("planner_output")
            assert status == True
            low_level_state_trajectory: List[Any] = self._state_factory.get_low_level_state_trajectory_from_plan(instance_idx, plan)
            revisits_states: bool = len(low_level_state_trajectory) > len(set(low_level_state_trajectory))
            preprocessed_data.update({"revisits_states": [revisits_states]})
        return preprocessed_data.get("revisits_states")

    def _preprocess_instance_final(self, preprocessed_data: Dict[str, Any]):
        instance_idx: int = preprocessed_data.get("instance_idx")
        assert self._instance_datas[instance_idx].instance_filepath() == preprocessed_data.get("instance_filepath")

        if preprocessed_data.get("marked_state_trajectories") is None:
            status, plan = preprocessed_data.get("planner_output")
            assert status == True

            state_trajectory: List[int] = [state_idx for state_idx, _ in self._state_factory.get_state_trajectory_from_plan(instance_idx, plan)]
            trajectory: List[Tuple[int, str]] = list(zip(state_trajectory[:-1], plan)) + [(state_trajectory[-1], "GOAL")]
            logging.debug(f"State trajectory: {state_trajectory}")
            logging.debug(f"Trajectory: {trajectory}")
            preprocessed_data.update({"marked_state_trajectories": [state_trajectory], "trajectories": [trajectory]})

            # Register all states in marked state trajectory as non-deadend states
            for state_idx in state_trajectory:
                self._state_factory.register_deadend_value(instance_idx, state_idx, False)

    def _get_effects_on_features(self, src_feature_values: np.ndarray, dst_feature_values: np.ndarray, relevant_features: Optional[intbitset] = None) -> Dict[str, intbitset]:
        effects: List[str] = list(map(lambda delta: "inc" if delta > 0 else "eqv" if delta == 0 else "dec", np.sign(dst_feature_values - src_feature_values)))
        unfiltered_effects: Dict[str, intbitset] = {key: intbitset([f_idx for f_idx, chg in enumerate(effects) if chg == key]) for key in ["inc", "eqv", "dec"]}
        if relevant_features is not None:
            return {key: features & relevant_features for key, features in unfiltered_effects.items()}
        else:
            return unfiltered_effects

    def _calculate_ext_states_and_edges(self):
        # Classify ext_states
        ext_states_in_relevant: Set[Tuple[int, int]] = frozenset([(instance_idx, state_idx) for instance_idx, state_idxs in self._iteration_relevant_vertices.items() for state_idx in state_idxs])
        non_goal_ext_states_in_paths: Set[Tuple[int, int]] = frozenset([(instance_idx, ex_state) for instance_idx, path in self._iteration_paths_with_idxs for ex_state in path[:-1]])
        goal_ext_states_in_paths: Set[Tuple[int, int]] = frozenset([(instance_idx, path[-1]) for instance_idx, path in self._iteration_paths_with_idxs])
        ext_states_in_deadend_paths: Set[Tuple[int, int]] = frozenset([ext_state for deadend_path in self._iteration_deadend_paths for ext_state in deadend_path])
        deadend_ext_states: Set[Tuple[int, int]] = frozenset([deadend_path[-1] for deadend_path in self._iteration_deadend_paths])
        non_covered_ext_states: Set[Tuple[int, int]] = frozenset([(instance_idx, state_idx) for instance_idx, state_idxs in self._iteration_non_covered_vertices.items() for state_idx in state_idxs])

        # Edges on example paths
        ext_edges: List[Tuple[int, Tuple[int, int]]] = [(instance_idx, ex_edge) for instance_idx, path in self._iteration_paths_with_idxs for ex_edge in zip(path[:-1], path[1:])]

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
        self._iteration_ex_ext_states: Set[Tuple[int, int]] = non_goal_ext_states_in_paths | non_covered_ext_states
        self._iteration_ext_edges: List[Tuple[int, Tuple[int, int]]] = ext_edges

        logging.info(f"The following are consolidated ext states:")
        logging.info(f"   ext_states_in_deadend_paths: {sorted(self._iteration_ext_states_in_deadend_paths)}")
        logging.info(f"        non_covered_ext_states: {sorted(self._iteration_non_covered_ext_states)}")
        logging.info(f"            deadend_ext_states: {sorted(self._iteration_deadend_ext_states)}")
        logging.info(f"               goal_ext_states: {sorted(self._iteration_goal_ext_states)}")
        logging.info(f"                 ex_ext_states: {sorted(self._iteration_ex_ext_states)}")
        logging.info(f"                     ext_edges: {sorted(self._iteration_ext_edges)}")

    def _calculate_rule_effects(self, ext_edges: List[Tuple[int, Tuple[int, int]]], relevant_features_idxs: intbitset) -> Dict[Tuple[int, Tuple[int, int]], Dict[str, intbitset]]:
        rule_effects: Dict[Tuple[int, Tuple[int, int]], Dict[str, intbitset]] = dict()
        for ext_edge in ext_edges:
            instance_idx, (src_state_idx, dst_state_idx) = ext_edge
            assert self._iteration_ext_state_to_feature_valuations.get((instance_idx, src_state_idx)) is not None, f"INEXiSTENT ENTRY: {(instance_idx, src_state_idx)}; keys: {self._iteration_ext_state_to_feature_valuations.keys()}"
            assert self._iteration_ext_state_to_feature_valuations.get((instance_idx, dst_state_idx)) is not None, f"INEXiSTENT ENTRY: {(instance_idx, dst_state_idx)}; keys: {self._iteration_ext_state_to_feature_valuations.keys()}"
            src_feature_values: np.ndarray = self._iteration_ext_state_to_feature_valuations.get((instance_idx, src_state_idx))
            dst_feature_values: np.ndarray = self._iteration_ext_state_to_feature_valuations.get((instance_idx, dst_state_idx))
            assert ext_edge not in rule_effects
            rule_effects[ext_edge] = self._get_effects_on_features(src_feature_values, dst_feature_values, relevant_features_idxs)
        return rule_effects

    def _calculate_indexing_data_structures(self,
                                            rule_effects: Dict[Tuple[int, Tuple[int, int]], Dict[str, intbitset]],
                                            relevant_features: List[Tuple[int, Feature]]) -> Dict[str, Any]:
        logging.info(f"Computing indexing data structures...")
        local_timer: Timer = Timer()
        self._timers.resume("indexing")

        relevant_features_idxs: intbitset = intbitset([f_idx for f_idx, _ in relevant_features])
        numerical_features_idxs: intbitset = intbitset([f_idx for f_idx, feature in relevant_features if isinstance(feature.dlplan_feature, dlplan_core.Numerical)])
        features_decreased_by_some_rule: intbitset = intbitset().union(*[effects.get("dec", intbitset()) for effects in rule_effects.values()])
        features_increased_by_some_rule: intbitset = intbitset().union(*[effects.get("inc", intbitset()) for effects in rule_effects.values()])
        features_not_increased_by_any_rule: intbitset = relevant_features_idxs - features_increased_by_some_rule
        features_not_decreased_by_any_rule: intbitset = relevant_features_idxs - features_decreased_by_some_rule

        ext_states_by_bvalue_on_feature: Dict[Tuple[int, int], Set[Tuple[int, int]]] = defaultdict(set) # ext_states_by_bvalue_on_feature[(f_idx, bvalue)] -> {(instance_idx, state_idx)}
        ext_states_by_change_on_feature: Dict[Tuple[int, str], Set[Tuple[int, int]]] = defaultdict(set) # ext_states_by_change_on_feature[(f_idx, change)] -> {(instance_idx, state_idx)}
        features_by_bvalue_on_ext_state: Dict[Tuple[Tuple[int, int], int], intbitset] = defaultdict(intbitset) # features_by_bvalue_on_ext_state[((instance_idx, state_idx), bvalue)] -> {f_idx}
        features_by_change_on_ext_state: Dict[Tuple[Tuple[int, int], str], intbitset] = defaultdict(intbitset) # features_by_change_on_ext_state[((instance_idx, state_idx), change)] -> {f_idx}

        for ext_state in self._iteration_ex_ext_states | self._iteration_non_covered_ext_states | self._iteration_goal_ext_states | self._iteration_ext_states_in_deadend_paths:
            instance_idx, state_idx = ext_state
            feature_values: np.ndarray = self._iteration_ext_state_to_feature_valuations.get(ext_state)
            for f_idx, f_value in enumerate(feature_values):
                if f_idx in relevant_features_idxs:
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
            "relevant_features": list(relevant_features),
            "relevant_features_idxs": relevant_features_idxs,
            "numerical_features_idxs": numerical_features_idxs,
            "features_decreased_by_some_rule": features_decreased_by_some_rule,
            "features_increased_by_some_rule": features_increased_by_some_rule,
            "features_not_increased_by_any_rule": features_not_increased_by_any_rule,
            "features_not_decreased_by_any_rule": features_not_decreased_by_any_rule,
            "ext_states_by_bvalue_on_feature": ext_states_by_bvalue_on_feature,
            "ext_states_by_change_on_feature": ext_states_by_change_on_feature,
            "features_by_bvalue_on_ext_state": features_by_bvalue_on_ext_state,
            "features_by_change_on_ext_state": features_by_change_on_ext_state,
        }

    def _calculate_ext_states_to_contexts(self,
                                          usable_features: intbitset,
                                          features_by_change_on_ext_state: Dict[Tuple[Tuple[int, int], str], intbitset]) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
        ext_states_to_contexts: Dict[Tuple[int, int], Set[Tuple[int, int]]] = defaultdict(set)
        # Context for rule r (associated with ext-state) is singleton-valuation \nu = {g_idx <- bvalue} such that
        # r belongs to \varrho(R, \nu) = { rules whose condition is consistent with \nu, and do not change g_idx }
        for (ext_state, change), g_idxs in features_by_change_on_ext_state.items():
            if change == "eqv":
                feature_values: np.ndarray = self._iteration_ext_state_to_feature_valuations.get(ext_state)
                for g_idx in g_idxs & usable_features:
                    bvalue = 0 if feature_values[g_idx] == 0 else 1
                    ext_states_to_contexts[ext_state].add((g_idx, bvalue))
        for ext_state, contexts in ext_states_to_contexts.items():
            logging.debug(f"CONTEXT: {ext_state} -> {len(contexts)} context(s)")
        return ext_states_to_contexts

    def _calculate_requirements_for_good_transitions(self,
                                                     usable_features: intbitset,
                                                     features_by_change_on_ext_state: Dict[Tuple[Tuple[int, int], str], intbitset],
                                                     ext_states_to_contexts: Dict[Tuple[int, int], Set[Tuple[int, int]]] = None,
                                                     m_pairs: MPairsContextual = None) -> Any:
        if ext_states_to_contexts is None:
            assert m_pairs is None
            requirements_for_good_transitions: Dict[Tuple[int, int], intbitset] = dict()
            for ext_state in self._iteration_ex_ext_states:
                changing_features: intbitset = features_by_change_on_ext_state.get((ext_state, "dec"), intbitset()) | features_by_change_on_ext_state.get((ext_state, "inc"), intbitset())
                requirements_for_good_transitions[ext_state] = changing_features & usable_features
            return requirements_for_good_transitions
        else:
            # For each rule r, there is pair (f, \nu) such that
            # 1. f changes in r
            # 2. f is monotone in R given \nu
            # 3. f is monotone (i.e., \nu is empty) OR r belongs to \varrho(R, \nu)

            assert m_pairs is not None
            nu_context_to_index: OrderedDict[Tuple[int, int], int] = OrderedDict()
            fnu_pair_to_index: OrderedDict[Tuple[int, int], int] = OrderedDict()
            fnu_idx_to_direction: List[str] = []
            requirements_for_good_transitions: Dict[Tuple[int, int], intbitset] = defaultdict(intbitset)
            for (ext_state, change), f_idxs in features_by_change_on_ext_state.items():
                if change != "eqv":
                    monotone_f_idxs: intbitset = f_idxs & m_pairs.monotone_features()
                    assert monotone_f_idxs.issubset(usable_features)
                    non_monotone_f_idxs: intbitset = (f_idxs - monotone_f_idxs) & usable_features

                    # Empty contexts for monotone features that change in r
                    nu_idx: int = nu_context_to_index.setdefault((), len(nu_context_to_index))
                    for f_idx in monotone_f_idxs:
                        fnu_idx: int = fnu_pair_to_index.setdefault((f_idx, nu_idx), len(fnu_pair_to_index))
                        requirements_for_good_transitions[ext_state].add(fnu_idx)
                        if fnu_idx >= len(fnu_idx_to_direction):
                            fnu_idx_to_direction.append(change)

                    # Contexts (g_idx, bvalue) for conditional monotone features that change in r
                    contexts: Set[Tuple[int, int]] = ext_states_to_contexts.get(ext_state, [])
                    for g_idx, bvalue in contexts: # This is the context \nu = {g_idx <- bvalue} such that rule r belongs to \varrho(R, \nu)
                        nu_idx: int = nu_context_to_index.setdefault((g_idx, bvalue), len(nu_context_to_index))
                        monotone_given_nu_f_idxs: intbitset = non_monotone_f_idxs & m_pairs.f_idxs_for_g_idx(g_idx)[bvalue] # This is set of f_idx that change with r and are monotone given \nu
                        for f_idx in monotone_given_nu_f_idxs:
                            fnu_idx: int = fnu_pair_to_index.setdefault((f_idx, nu_idx), len(fnu_pair_to_index))
                            requirements_for_good_transitions[ext_state].add(fnu_idx)
                            if fnu_idx >= len(fnu_idx_to_direction):
                                fnu_idx_to_direction.append(change)

            # Complete requirements
            uncovered_ext_states: Set[Tuple[int, int]] = self._iteration_ex_ext_states - set(requirements_for_good_transitions.keys())
            for ext_state in uncovered_ext_states:
                requirements_for_good_transitions[ext_state] = intbitset()

            for ext_state, requirement in requirements_for_good_transitions.items():
                logging.info(f"REQUIREMENT: {ext_state} -> {len(requirement)} fnu-pair(s)")

            return requirements_for_good_transitions, nu_context_to_index, fnu_pair_to_index, fnu_idx_to_direction

    def _calculate_features_that_separate_ext_pairs(self,
                                                    instance_idx_to_ext_pairs: Dict[int, Tuple[int, int]],
                                                    features_by_bvalue_on_ext_state: Dict[Tuple[Tuple[int, int], int], intbitset],
                                                    usable_features: intbitset) -> Dict[Tuple[int, Tuple[int, int]], intbitset]:
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
                ext_pair_to_separating_features[(instance_idx, pair)] = separating_features & usable_features
        return ext_pair_to_separating_features


    def _calculate_features_that_separate_ext_pairs_v2(self,
                                                    instance_idx_to_state_idx_to_state_idxs: Dict[int, Dict[int, List[int]]],
                                                    features_by_bvalue_on_ext_state: Dict[Tuple[Tuple[int, int], int], intbitset],
                                                    features_by_change_on_ext_state: Dict[Tuple[Tuple[int, int], str], intbitset],
                                                    usable_features: intbitset) -> Dict[int, Dict[int, List[Tuple[int, intbitset]]]]:
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
                    instance_idx_to_state_idx_to_separating_features[instance_idx][state_idx_1].append((state_idx_2, separating_features & usable_features))
        return instance_idx_to_state_idx_to_separating_features

    def _calculate_features_that_change_differently_at_ext_pairs(self,
                                                                 ext_pairs: Set[Tuple[int, Tuple[int, int]]],
                                                                 features_by_bvalue_on_ext_state: Dict[Tuple[Tuple[int, int], int], intbitset],
                                                                 features_by_change_on_ext_state: Dict[Tuple[Tuple[int, int], str], intbitset],
                                                                 usable_features: intbitset) -> Dict[Tuple[int, Tuple[int, int]], intbitset]:
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
                                                         features_by_change_on_ext_state: Dict[Tuple[Tuple[int, int], str], intbitset],
                                                         relevant_features_idxs: intbitset) -> Dict[Tuple[Tuple[int, int]], List[intbitset]]:
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
            src_feature_values: np.ndarray = self._iteration_ext_state_to_feature_valuations.get((instance_idx, src_state_idx))
            dst_feature_values: np.ndarray = self._iteration_ext_state_to_feature_valuations.get((instance_idx, dst_state_idx))
            feature_changes_on_marked_ext_edge: Dict[str, intbitset] = self._get_effects_on_features(src_feature_values, dst_feature_values, relevant_features_idxs)
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
                                                            features_by_change_on_ext_state: Dict[Tuple[Tuple[int, int], str], intbitset],
                                                            relevant_features_idxs: intbitset) -> Tuple[bool, Dict[Tuple[int, int], Dict[Tuple[int, Tuple[int, int]], intbitset]]]:
        logging.info(f"Deadend paths: {self._iteration_deadend_paths}")
        ext_state_to_separating_features: Dict[Tuple[int, int], Dict[Tuple[int, Tuple[int, int]], intbitset]] = defaultdict(lambda: defaultdict(intbitset))
        marked_ext_edges: Set[Tuple[int, Tuple[int, int]]] = set()
        for path in self._iteration_deadend_paths:
            logging.info(f"Deadend path: {path}")
            assert len(path) > 1

            # Marked edge to separate from example edges is last edge in path
            instance_idx: int = path[0][0]
            marked_ext_edge: Tuple[int, Tuple[int, int]] = (instance_idx, (path[-2][1], path[-1][1]))
            marked_ext_edges.add(marked_ext_edge)
            logging.info(f"  Ext-edge: {marked_ext_edge}*")

            # Feature changes across marked ext-edge
            instance_idx, (src_state_idx, dst_state_idx) = marked_ext_edge
            src_feature_values: np.ndarray = self._iteration_ext_state_to_feature_valuations.get((instance_idx, src_state_idx))
            dst_feature_values: np.ndarray = self._iteration_ext_state_to_feature_valuations.get((instance_idx, dst_state_idx))
            feature_changes_on_marked_ext_edge: Dict[str, intbitset] = self._get_effects_on_features(src_feature_values, dst_feature_values, relevant_features_idxs)
            features_by_change_on_marked_ext_edge: List[intbitset] = [feature_changes_on_marked_ext_edge.get(change) for change in ["dec", "inc", "eqv"]]
            logging.debug(f"feature_changes_on_marked_ext_edge: {dict([(key, feature_changes_on_marked_ext_edge.get(key)) for key in ['dec', 'inc']])}")

            # Requirement that separates marked ext-edges from ext-edges in policy
            for ext_edge in self._iteration_ext_edges:
                ext_state: Tuple[int, int] = (ext_edge[0], ext_edge[1][0])
                features_by_change_on_ext_edge: List[List[intbitset]] = [features_by_change_on_ext_state.get((ext_state, change), intbitset()) for change in ["dec", "inc", "eqv"]]
                requirement: intbitset = intbitset().union(*[features_by_change_on_marked_ext_edge[i] - features_by_change_on_ext_edge[i] for i in [0, 1, 2]])
                assert marked_ext_edge not in ext_state_to_separating_features[ext_state] or ext_state_to_separating_features[ext_state][marked_ext_edge] == requirement
                ext_state_to_separating_features[ext_state][marked_ext_edge] = requirement
                logging.debug(f"Ext-state: {ext_state}, requirement={requirement}")
                if len(requirement) == 0: return False, None, marked_ext_edges
        return True, ext_state_to_separating_features, marked_ext_edges

    def _preprocess_termination(self, contextual: bool = False, monotone_only_by_dec: bool = False, separate_siblings: bool = False) -> Dict[str, Any]:
        logging.info(f"Preprocessing termination of features...")
        local_timer: Timer = Timer()
        self._timers.resume("feature/termination")
        self._timers.resume("pricing/preprocessing")

        # Define relevant edges: those in paths and those for off-path relevant vertices
        self._calculate_ext_states_and_edges()

        # Get relevant features
        remove_features_with_constant_bvalue: bool = True
        relevant_features: List[Tuple[int, Feature]] = self._get_relevant_features(remove_features_with_constant_bvalue)
        relevant_features_idxs: intbitset = intbitset([f_idx for f_idx, _ in relevant_features])
        f_idx_to_feature_index: Dict[int, int] = {f_idx: index for index, (f_idx, feature) in enumerate(relevant_features)}
        logging.info(f"{len(relevant_features)} relevant feature(s)")

        # Extract change of relevant features along paths
        rule_effects: Dict[Tuple[int, Tuple[int, int]], Dict[str, intbitset]] = self._calculate_rule_effects(self._iteration_ext_edges, relevant_features_idxs)

        # Calculate indexing data structures
        ds: Dict[str, Any] = self._calculate_indexing_data_structures(rule_effects, relevant_features)

        # Calculate companions m_pairs
        lazy: bool = False
        if not contextual:
            m_pairs: MPairs = MPairs(ds, monotone_only_by_dec=monotone_only_by_dec, lazy=lazy, timers=self._timers)
        else:
            m_pairs: MPairsContextual = MPairsContextual(ds, monotone_only_by_dec=monotone_only_by_dec, lazy=lazy, timers=self._timers)

        features_by_bvalue_on_ext_state: Dict[Tuple[Tuple[int, int], int], intbitset] = m_pairs._features_by_bvalue_on_ext_state
        features_by_change_on_ext_state: Dict[Tuple[Tuple[int, int], str], intbitset] = m_pairs._features_by_change_on_ext_state
        monotone_features: intbitset = m_pairs.monotone_features()

        if lazy:
            usable_features: intbitset = m_pairs._relevant_features_idxs
            pruned_relevant_features: List[Tuple[int, Feature]] = relevant_features
            fg_companions: Dict[int, intbitset] = None
            gf_companions: Dict[int, intbitset] = None
        else:
            # Remove non-terminating features
            usable_features: intbitset = m_pairs.usable_features()
            pruned_relevant_features: List[Tuple[int, Feature]] = [(f_idx, feature) for f_idx, feature in relevant_features if f_idx in usable_features]
            f_idx_to_feature_index: Dict[int, int] = {f_idx: index for index, (f_idx, feature) in enumerate(pruned_relevant_features)}
            logging.info(f"{len(pruned_relevant_features)} feature(s) after removing non-terminating features (num_pruned={len(relevant_features) - len(pruned_relevant_features)})")
            logging.info(f"{len(monotone_features)} monotone feature(s)")

        # Requirements for good transitions
        if not contextual:
            requirements_for_good_transitions = self._calculate_requirements_for_good_transitions(usable_features, features_by_change_on_ext_state)
            nu_context_to_index = None
            fnu_pair_to_index = None
            fnu_idx_to_direction = None
        else:
            ext_states_to_contexts: Dict[Tuple[int, int], Set[Tuple[int, int]]] = self._calculate_ext_states_to_contexts(usable_features, features_by_change_on_ext_state)
            requirements_for_good_transitions, nu_context_to_index, fnu_pair_to_index, fnu_idx_to_direction = self._calculate_requirements_for_good_transitions(usable_features,
                                                                                                                                                                features_by_change_on_ext_state,
                                                                                                                                                                ext_states_to_contexts,
                                                                                                                                                                m_pairs)
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
                                                                                                                                              features_by_bvalue_on_ext_state,
                                                                                                                                              usable_features)
        #ext_state_to_separating_features_for_goals: Dict[Tuple[int, int], List[intbitset]] = self._calculate_features_that_separate_ext_pairs_v2(instance_idx_to_goal_state_idx_to_state_idxs,
        #                                                                                                                                         features_by_bvalue_on_ext_state,
        #                                                                                                                                         usable_features)
        goal_separating_features: List[intbitset] = list(goal_ext_pair_to_separating_features.values())

        # Calculate separating features for discovered deadends
        ext_state_to_separating_features_for_deadend_paths: Dict[Tuple[int, int], Dict[Tuple[int, Tuple[int, int]], intbitset]] = None
        bad_ext_edges: Set[Tuple[int, Tuple[int, int]]] = set()
        status, ext_state_to_separating_features_for_deadend_paths, bad_ext_edges = self._calculate_separating_features_for_deadend_paths_v2(features_by_change_on_ext_state, relevant_features_idxs)
        if not status:
            logging.error(f"ERROR: No feature to separate transition to deadend state")
            raise RuntimeError(f"ERROR: No feature to separate transition to deadend state")
        deadend_separating_features: List[intbitset] = [separating_features for separating_features_for_deadend_paths in ext_state_to_separating_features_for_deadend_paths.values() for separating_features in separating_features_for_deadend_paths.values()]

        local_timer.stop()
        self._timers.stop("pricing/preprocessing")
        self._timers.stop("feature/termination")
        logging.info(f"{local_timer.get_elapsed_sec():0.2f} second(s) for preprocessing termination")

        # Calculate features that separate sibling edges (if applicable)
        ext_sibling_to_separating_features: Dict[Tuple[int, int, Tuple[int, int]], intbitset] = dict()
        if separate_siblings:
            assert False
            for ext_edge in self._iteration_ext_edges:
                instance_idx, (src_state_idx, dst_state_idx) = ext_edge
                ext_state: Tuple[int, int] = (instance_idx, src_state_idx)
                instance_data: PDDLInstance = self._instance_datas[instance_idx]
                src_feature_values: np.ndarray = self._iteration_ext_state_to_feature_valuations.get((instance_idx, src_state_idx))
                features_by_change_on_ext_edge: List[List[intbitset]] = [features_by_change_on_ext_state.get((ext_state, change), intbitset()) for change in ["dec", "inc", "eqv"]]

                successors: List[Tuple[Tuple[int, Any], str]] = instance_data.get_successors(src_state_idx)

                # Skip this transition if it is a mark/move-mark transition
                action_for_ext_edge = [action for (succ_state_idx, succ_state), action in successors if succ_state_idx == dst_state_idx][0]
                if action_for_ext_edge.startswith("mark-") or action_for_ext_edge.startswith("move-mark-"): continue

                for succ_state_idx in [succ_state_idx for (succ_state_idx, succ_state), action in successors if succ_state_idx != dst_state_idx]:
                    sibling_feature_values: np.ndarray = self._iteration_ext_state_to_feature_valuations.get((instance_idx, succ_state_idx))
                    feature_changes_on_sibling_ext_edge: Dict[str, intbitset] = self._get_effects_on_features(src_feature_values, sibling_feature_values, relevant_features_idxs)
                    features_by_change_on_sibling_ext_edge: List[intbitset] = [feature_changes_on_sibling_ext_edge.get(change) for change in ["dec", "inc", "eqv"]]
                    requirement: intbitset = intbitset().union(*[features_by_change_on_ext_edge[i] - features_by_change_on_sibling_ext_edge[i] for i in [0, 1, 2]])
                    ext_sibling: Tuple[int, int, Tuple[int, int]] = (instance_idx, src_state_idx, (dst_state_idx, succ_state_idx))
                    assert ext_sibling not in ext_sibling_to_separating_features
                    ext_sibling_to_separating_features[ext_sibling] = requirement
                    logging.debug(f"Sibling: ext_sibling={ext_sibling}, requirement={requirement}")

        return {
            "relevant_features": pruned_relevant_features,
            "f_idx_to_feature_index": f_idx_to_feature_index,
            "lower_bound_on_rank": None,
            "requirements_for_good_transitions": requirements_for_good_transitions,
            "nu_context_to_index": nu_context_to_index,
            "fnu_pair_to_index": fnu_pair_to_index,
            "fnu_idx_to_direction": fnu_idx_to_direction,
            "ext_state_to_ext_edge": self._iteration_ext_state_to_ext_edge,
            "ext_state_to_feature_valuations": self._iteration_ext_state_to_feature_valuations,
            "goal_ext_pair_to_separating_features": goal_ext_pair_to_separating_features,
            "ext_state_to_separating_features_for_deadend_paths": ext_state_to_separating_features_for_deadend_paths,
            "bad_ext_edges": bad_ext_edges,
            "ext_sibling_to_separating_features": ext_sibling_to_separating_features,
            "m_pairs": m_pairs,
            "ex_ext_states": self._iteration_ex_ext_states,
            #"facts": {
            #    "monotone": set([ASPSolver.make_fact("monotone", f_idx) for f_idx in monotone_features]),
            #    "changing_features": set([ASPSolver.make_fact("changed", self._get_global_state_index_from_ext_state(ext_state), f_idx) for ext_state, f_idxs in requirements_for_good_transitions.items() for f_idx in f_idxs]),
            #    "goal_separating_features_1": set([ASPSolver.make_fact("goal_separating", index) for index in range(len(goal_separating_features))]),
            #    "goal_separating_features_2": set([ASPSolver.make_fact("goal_separating", index, f_idx) for index, f_idxs in enumerate(goal_separating_features) for f_idx in f_idxs]),
            #    "deadend_separating_features_1": set([ASPSolver.make_fact("deadend_separating", index) for index in range(len(deadend_separating_features))]),
            #    "deadend_separating_features_2": set([ASPSolver.make_fact("deadend_separating", index, f_idx) for index, f_idxs in enumerate(deadend_separating_features) for f_idx in f_idxs]),
            #}
        }

    def _get_relevant_features(self, remove_features_with_constant_bvalue: bool = False) -> List[Tuple[int, Feature]]:
        logging.info(f"Pruning redundant fetures...")
        features: List[Feature] = self._iteration_feature_pool

        dlplan_state_pairs: List[Tuple[dlplan_core.State, dlplan_core.State]] = []
        for (instance_idx, (src_state_idx, dst_state_idx)) in self._iteration_ext_edges:
            src_dlplan_state: dlplan_core.State = self._state_factory.get_dlplan_state(instance_idx, src_state_idx)
            dst_dlplan_state: dlplan_core.State = self._state_factory.get_dlplan_state(instance_idx, dst_state_idx)
            dlplan_state_pairs.append((src_dlplan_state, dst_dlplan_state))
        logging.info(f"{len(dlplan_state_pairs)} pair(s) of dlplan states")

        selected_features_v2: List[Tuple[int, Feature]] = prune_features_with_same_feature_change_AND_boolean_valuation_reduced_v2(self._state_factory,
                                                                                                                                   dlplan_state_pairs,
                                                                                                                                   self._iteration_instance_idx_to_denotations_caches,
                                                                                                                                   features,
                                                                                                                                   remove_features_with_constant_bvalue)
        num_pruned_v2 = len(features) - len(selected_features_v2)
        logging.info(f"{len(selected_features_v2)} feature(s) after pruning features with same change across transitions AND same boolean valuation (num_pruned={num_pruned_v2})")

        return selected_features_v2

    def _get_global_state_index_from_ext_state(self, ext_state: Tuple[int, int], verbose: bool = True) -> int:
        global_state_index = self._ext_state_to_global_state_index.get(ext_state)
        if global_state_index is None:
            global_state_index = len(self._ext_state_to_global_state_index)
            self._ext_state_to_global_state_index[ext_state] = global_state_index
            self._global_state_index_to_ext_state[global_state_index] = ext_state
            if verbose: logging.info(f"STATE_MAP: {ext_state} -> {global_state_index}")
        return global_state_index

    def _get_ext_state_from_global_state_index(self, global_state_index: int, verbose: bool = True) -> Tuple[int, int]:
        ext_state: Tuple[int, int] = self._global_state_index_to_ext_state.get(global_state_index)
        assert ext_state is not None
        return ext_state

    def _verify_termination(self,
                            relevant_features: List[Tuple[int, Feature]],
                            feature_ranks: Dict[int, int],
                            f_idx_to_feature_index: Dict[int, int],
                            solver_prefix: str,
                            dump_asp_program: bool = False,
                            **kwargs) -> bool:
        # List of facts
        facts: List[Dict[str, Set[Any]]] = []

        # Facts for selected features
        selected_features: List[Tuple[int, feature]] = [relevant_features[f_idx_to_feature_index[f_idx]] for f_idx in feature_ranks.keys()]
        facts.append(self._create_facts_for_features(selected_features))

        # Facts for xedges and xstates
        facts.append(self._create_xedge_facts_for_ext_edges(self._iteration_ext_edges))
        facts.append(self._create_xstate_facts(self._iteration_ex_ext_states))
        dst_ext_states_in_edges: Set[Tuple[int, int]] = set([(instance_idx, edge[1]) for instance_idx, edge in self._iteration_ext_edges])

        # Facts for states
        state_space_facts: Dict[str, Set[Any]] = defaultdict(set)
        for ext_state in self._iteration_ex_ext_states | self._iteration_non_covered_ext_states | self._iteration_goal_ext_states | self._iteration_ext_states_in_deadend_paths | dst_ext_states_in_edges:
            global_state_index = self._get_global_state_index_from_ext_state(ext_state)

            if ext_state not in (dst_ext_states_in_edges | self._iteration_ext_states_in_deadend_paths) - self._iteration_non_covered_ext_states - self._iteration_ex_ext_states:
                # Avoid state(<global_state_index>) facts for "fringe" ext_states as this causes UNSAT verification because of covering clauses
                state_space_facts["state"].add(ASPSolver.make_fact("state", global_state_index))

            #feature_values: Tuple[int] = self._iteration_ext_state_to_feature_valuations.get(ext_state)
            feature_values: np.ndarray = self._iteration_ext_state_to_feature_valuations.get(ext_state)
            assert feature_values is not None, f"Non available feature valuations for ext_state={ext_state}"
            for f_idx, _ in selected_features:
                value = feature_values[f_idx]
                if type(value) == bool:
                    state_space_facts["value"].add(ASPSolver.make_fact("value", global_state_index, f_idx, 1 if value else 0))
                else:
                    state_space_facts["value"].add(ASPSolver.make_fact("value", global_state_index, f_idx, value))

        for ext_state in self._iteration_goal_ext_states:
            global_state_index = self._get_global_state_index_from_ext_state(ext_state)
            state_space_facts["goal"].add(ASPSolver.make_fact("goal", global_state_index))

        for ext_state in self._iteration_deadend_ext_states:
            global_state_index = self._get_global_state_index_from_ext_state(ext_state)
            state_space_facts["deadend"].add(ASPSolver.make_fact("deadend", global_state_index))

        facts.append(state_space_facts)

        # Facts for ranks
        sorted_features: List[int] = [f_idx for _, f_idx in sorted([(rank, f_idx) for f_idx, rank in feature_ranks.items()])]
        ranks: List[Tuple[int, int]] = list(zip(sorted_features, range(1, 2 + len(sorted_features))))
        facts.append(self._create_facts_for_ranks(ranks))

        # Create solver
        fact_signatures: List[Tuple[Any]] = [
            ("boolean", ("f",), "boolean(f)."),
            ("numerical", ("f",), "numerical(f)."),
            ("complexity", ("f", "c"), "complexity(f,c)."),
            ("xedge", ("s", "t"), "xedge(s,t)."),
            ("xstate", ("s",), "xstate(s)."),
            ("state", ("s",), "state(s)."),
            ("goal", ("s",), "goal(s)."),
            ("deadend", ("s",), "deadend(s)."),
            #("edge", ("s", "t"), "edge(s,t)."),
            ("value", ("s", "f", "v"), "value(s,f,v)."),
            #("unknown", ("s", "f"), "unknown(s,f)."),
            #("goal_separating", ("i",), "goal_separating(i)."),
            #("goal_separating", ("i", "f",), "goal_separating(i,f)."),
            #("changed", ("s", "f"), "changed(s,f)."),
            ("rank", ("f", "i"), "rank(f,i)."),
        ]

        arguments: List[str] = ["--parallel-mode=16", "-n", "0"]
        loads: List[str] = [str(LIST_DIR / f"{solver_prefix}_verify_termination.lp")]
        solver: ASPSolver = ASPSolver(arguments=arguments, fact_signatures=fact_signatures, loads=loads)
        logging.info(f"Arguments: {arguments}")

        # Ground solver with facts and check
        solver.ground(*facts, dump_asp_program=dump_asp_program)
        model, exit_code = solver.first_model() #, **options_for_solver)
        logging.info(f"Verification of termination: exit_code: {exit_code}")
        return exit_code == ClingoExitCode.SATISFIABLE

    def _create_facts_for_features(self, relevant_features: List[Tuple[int, Feature]], uniform_costs: bool = False) -> Dict[str, Set[Any]]:
        facts: Dict[str, Set[Any]] = defaultdict(set)
        for f_idx, feature in relevant_features:
            if isinstance(feature.dlplan_feature, dlplan_core.Numerical):
                facts["numerical"].add(ASPSolver.make_fact("numerical", f_idx))
            else:
                facts["boolean"].add(ASPSolver.make_fact("boolean", f_idx))
            facts["complexity"].add(ASPSolver.make_fact("complexity", f_idx, 1 if uniform_costs else feature.complexity))
        return facts

    def _create_facts_for_state_space(self,
                                      instance_idx: int,
                                      relevant_features: List[Tuple[int, Feature]],
                                      relevant_vertices: Dict[int, Set[int]],
                                      vertices_in_example_paths: Set[int]) -> Dict[str, Set[Any]]:
        assert relevant_vertices.get(instance_idx) is not None
        instance_data: PDDLInstance = self._instance_datas[instance_idx]
        assert instance_data.idx == instance_idx

        facts: Dict[str, Set[Any]] = defaultdict(set)

        # Vertices in edges whose source vertex is relevant, not a goal, and not in an example path
        sources: List[int] = [vertex for vertex in relevant_vertices.get(instance_idx) if vertex not in vertices_in_example_paths and not self._state_factory.is_goal_state(instance_idx, vertex)]
        edge_vertices: Set[int] = set(sources)
        for state_idx in sources:
            successors: List[Tuple[Tuple[int, Any], str]] = self._state_factory.get_successors(instance_idx, state_idx)
            edge_vertices |= set([succ_state_idx for (succ_state_idx, _), _ in successors])
            for (succ_state_idx, _), _ in successors:
                source = self._get_global_state_index_from_ext_state((instance_idx, state_idx))
                target = self._get_global_state_index_from_ext_state((instance_idx, succ_state_idx))
                facts["edge"].add(ASPSolver.make_fact("edge", source, target))

        # Values for relevant or edge vertices
        for vertex in relevant_vertices.get(instance_idx) | edge_vertices:
            ext_state: Tuple[int, int] = (instance_idx, vertex)
            global_state_index = self._get_global_state_index_from_ext_state(ext_state)
            assert global_state_index is not None, f"Undefined global_state_index for {ext_state}"

            if vertex in relevant_vertices.get(instance_idx):
                facts["state"].add(ASPSolver.make_fact("state", global_state_index))

            #feature_values: Tuple[int] = self._iteration_ext_state_to_feature_valuations.get(ext_state)
            feature_values: np.ndarray = self._iteration_ext_state_to_feature_valuations.get(ext_state)
            assert feature_values is not None, f"Non available feature valuations for ext_state={ext_state}"
            for f_idx, _ in relevant_features:
                value = feature_values[f_idx]
                if type(value) == bool:
                    facts["value"].add(ASPSolver.make_fact("value", global_state_index, f_idx, 1 if value else 0))
                else:
                    facts["value"].add(ASPSolver.make_fact("value", global_state_index, f_idx, value))

        # Goals for relevant or edge vertices
        for vertex in [vertex for vertex in (relevant_vertices.get(instance_idx) | edge_vertices) if self._state_factory.is_goal_state(instance_idx, vertex)]:
            global_state_index = self._get_global_state_index_from_ext_state((instance_idx, vertex))
            assert global_state_index is not None
            facts["goal"].add(ASPSolver.make_fact("goal", global_state_index))

        # Deadend for relevant or edge vertices
        for vertex in [vertex for vertex in (relevant_vertices.get(instance_idx) | edge_vertices) if self._state_factory.is_deadend_state(instance_idx, vertex)]:
            global_state_index = self._get_global_state_index_from_ext_state((instance_idx, vertex))
            assert global_state_index is not None
            facts["deadend"].add(ASPSolver.make_fact("deadend", global_state_index))

        return facts

    def _create_xedge_fact_for_ext_edge(self, ext_edge: Tuple[int, Tuple[int, int]]) -> Any:
        instance_idx, edge = ext_edge
        source = self._get_global_state_index_from_ext_state((instance_idx, edge[0]))
        target = self._get_global_state_index_from_ext_state((instance_idx, edge[1]))
        assert source is not None and target is not None
        return ASPSolver.make_fact("xedge", source, target)

    def _create_xedge_facts_for_ext_edges(self, ext_edges: List[Tuple[int, Tuple[int, int]]]) -> Dict[str, Set[Any]]:
        return {"xedge": set([self._create_xedge_fact_for_ext_edge(ext_edge) for ext_edge in ext_edges])}

    def _create_xstate_fact(self, instance_idx: int, state_idx: int) -> Any:
        global_state_index: int = self._get_global_state_index_from_ext_state((instance_idx, state_idx))
        return ASPSolver.make_fact("xstate", global_state_index)

    def _create_xstate_facts(self, ext_states: Set[Tuple[int, int]]) -> Dict[str, Set[Any]]:
        return {"xstate": set([self._create_xstate_fact(instance_idx, state_idx) for instance_idx, state_idx in ext_states])}

    # NOT USED
    def _create_facts_for_path(self, instance_idx: int, path: List[int]) -> Dict[str, Set[Any]]:
        ext_edges: List[Tuple[int, Tuple[int, int]]] = [(instance_idx, edge) for edge in zip(path[:-1], path[1:])]
        return self._create_xedge_facts_for_ext_edges(xedges)

    # NOT USED
    def _create_facts_for_unknowns(self, unknowns: Set[Tuple[int, int]]) -> Dict[str, Set[Any]]:
        return {"unknown": set([ASPSolver.make_fact("unknown", global_state_index, f_idx) for global_state_index, f_idx in unknowns])}

    # NOT USED
    def _create_facts_for_chosen_features(self, chosen: Set[int]) -> Dict[str, Set[Any]]:
        return {"chosen": set([ASPSolver.make_fact("chosen", f_idx) for f_idx in chosen])}

    def _create_facts_for_ranks(self, ranks: Set[Tuple[int, int]]) -> Dict[str, Set[Any]]:
        return {"rank": set([ASPSolver.make_fact("rank", f_idx, rank) for f_idx, rank in ranks])}
# END-OF class TerminationBasedLearnerReduced

