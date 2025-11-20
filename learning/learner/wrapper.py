import logging
from termcolor import colored
from typing import Set, List, Tuple, MutableSet, Dict, Optional, Any, Generator

import random
import numpy as np
from intbitset import intbitset
from collections import defaultdict, Counter

import dlplan.core as dlplan_core

from .benchmark import Benchmark
from .feature_pool import FeaturePool
from .src.util import Timer, write_file, write_file_lines, change_dir, memory_usage, print_separation_line
from .src.state_space import PDDLInstance
from .src.preprocessing import InstanceData
from .src.iteration import compute_feature_valuations_for_dlplan_state
from .src.iteration import IterationData, Statistics, SketchReduced, D2sepDlplanPolicyFactory
from .src.iteration import SolutionGenerator


class WrapperBase:
    def __init__(self, benchmark: Benchmark, feature_pool: FeaturePool, learner: Any, timers: Statistics, **kwargs):
        self._benchmark: Benchmark = benchmark
        self._feature_pool: FeaturePool = feature_pool
        self._learner: Any = learner
        self._timers: Statistics = timers
        self._options: Dict[str, Any] = kwargs
        self._width: int = learner._width

        self._policy_builder: PolicyFactory = self._benchmark._state_factory.domain_data().policy_builder

        self._folder_name_for_iterations: str = kwargs.get("folder_name_for_iterations")
        self._folder_name_for_output: str = kwargs.get("folder_name_for_output")
        self._instance_selection: str = kwargs.get("instance_selection")
        self._first_instance: str = kwargs.get("first_instance")
        self._enable_dump_files: bool = kwargs.get("enable_dump_files", False)
        self._dump_asp_program: bool = kwargs.get("dump_asp_program", False)

        self._instance_datas: List[PDDLInstance] = self._benchmark._instance_datas
        #self._dlplan_states: List[dlplan_core.State] = list(self._benchmark._dlplan_states)
        self._ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = {ext_state: np.array(feature_valuations) for ext_state, feature_valuations in self._benchmark._ext_state_to_feature_valuations.items()}
        #self._ext_state_to_dlplan_state_index: Dict[Tuple[int, int], int] = {ext_state: dlplan_state_index for ext_state, dlplan_state_index in self._benchmark._ext_state_to_dlplan_state_index.items()}
        self._instance_idx_to_denotations_caches: Dict[int, dlplan_core.DenotationsCaches] = self._benchmark._instance_idx_to_denotations_caches
        self._setup_registry: Dict[int, intbitset] = defaultdict(intbitset)

    def setup_ext_state(self, ext_state: Tuple[int, int], verbose: bool = False) -> bool:
        """
        instance_data: PDDLInstance = self._instance_datas[instance_idx]
        ext_state: Tuple[int, int] = (instance_idx, state_idx)
        if ext_state not in self._ext_state_to_dlplan_state_index:
            dlplan_state_index: int = len(self._dlplan_states)
            self._ext_state_to_dlplan_state_index[ext_state] = dlplan_state_index
            state: Any = self._benchmark.get_state(*ext_state)
            dlplan_state: dlplan_core.State = instance_data.get_dlplan_state(state_idx, state)
            self._dlplan_states.append(dlplan_state)
            logging.debug(f"EXT_STATE_TO_DLPLAN_INDEX: {ext_state} -> {dlplan_state_index} -> {dlplan_state}")
        if ext_state not in self._ext_state_to_feature_valuations:
            dlplan_state_index: int = self._ext_state_to_dlplan_state_index.get(ext_state)
            dlplan_state: dlplan_core.State = self._dlplan_states[dlplan_state_index]
            self._ext_state_to_feature_valuations[ext_state] = compute_feature_valuations_for_dlplan_state(dlplan_state, self._feature_pool._pool, self._instance_idx_to_denotations_caches)
        """
        if ext_state not in self._ext_state_to_feature_valuations:
            if verbose: logging.info(f"Setup_ext_state: ext_state={ext_state}")
            self._ext_state_to_feature_valuations[ext_state] = self._benchmark.get_feature_valuations(*ext_state, self._feature_pool._pool)
            return True
        else:
            return False

    def get_unsolved_instances(self,
                               iteration_data: IterationData,
                               instance_datas: List[PDDLInstance],
                               sketch: SketchReduced,
                               **kwargs) -> List[Tuple[PDDLInstance, Any]]:
        options: Dict[str, Any] = dict(self._options)
        options.update(kwargs)

        unsolved_instances: List[Tuple[PDDLInstance, Any]] = []
        for instance_data in instance_datas:
            solves, reason = sketch.solves(None, iteration_data, instance_data, **options)
            assert not solves or reason is None
            if not solves:
                unsolved_instances.append((instance_data, reason))
                if options.get("stop_at_first_unsolved_instance", True): break
        return unsolved_instances

    def get_initial_idxs(self) -> List[int]:
        available_idxs: List[int] = self._benchmark._instance_idxs
        if self._first_instance is not None and self._first_instance in available_idxs:
            return [self._first_instance]
        elif self._instance_selection is None or self._instance_selection in ["forward", "forward+"]:
            return [min(available_idxs)]
        elif self._instance_selection in ["backward", "backward+"]:
            return [max(available_idxs)]
        elif self._instance_selection in ["random", "random+"]:
            return [random.choice(available_idxs)]
        raise RuntimeError(f"ERROR: unexpected instance selection stragegy '{self._instance_selection}'")

    def advance_idxs(self, current_idxs: List[int], previous_idxs: List[int], unsolved_idxs: List[int]) -> List[int]:
        available_idxs: List[int] = self._benchmark._instance_idxs
        seen_idxs: List[int] = list(set(current_idxs + previous_idxs))
        max_seen_idx: int = max(seen_idxs)

        next_idx: int = None
        if self._instance_selection is None or self._instance_selection == "forward":
            choices_from_unsolved: List[int] = [instance_idx for instance_idx in unsolved_idxs if instance_idx > max_seen_idx]
            choices_from_available: List[int] = [instance_idx for instance_idx in available_idxs if instance_idx > max_seen_idx]
            if len(choices_from_unsolved) > 0:
                next_idx = min(choices_from_unsolved)
            elif len(choices_from_available) > 0:
                next_idx = min(choices_from_available)

        elif self._instance_selection == "forward+":
            choices_from_unsolved_0: List[int] = [instance_idx for instance_idx in unsolved_idxs if instance_idx < max_seen_idx]
            choices_from_unsolved_1: List[int] = [instance_idx for instance_idx in unsolved_idxs if instance_idx > max_seen_idx]
            choices_from_available: List[int] = [instance_idx for instance_idx in available_idxs if instance_idx > max_seen_idx]
            if len(choices_from_unsolved_0) > 0:
                next_idx = max(choices_from_unsolved_0)
            elif len(choices_from_unsolved_1) > 0:
                next_idx = min(choices_from_unsolved_1)
            elif len(choices_from_available) > 0:
                next_idx = min(choices_from_available)

        elif self._instance_selection in ["backward", "backward+"]:
            next_idx = max(unsolved_idxs) if len(unsolved_idxs) > 0 else None

        elif self._instance_selection in ["random", "random+"]:
            next_idx = random.choice(unsolved_idxs) if len(unsolved_idxs) > 0 else None

        else:
            raise RuntimeError(f"ERROR: unexpected instance selection stragegy '{self._instance_selection}'")

        if next_idx is None:
            logging.info(colored(f"** NO MORE TRAINING INSTANCES", "red"))
            raise RuntimeError("ERROR: ** NO MORE TRAINING INSTANCES")

        if self._instance_selection is None or not self._instance_selection.endswith("+"):
            next_idxs: List[int] = [next_idx]
        elif next_idx > max_seen_idx:
            next_idxs: List[int] = [next_idx]
        else:
            # TODO: CHECK: Following assertion may fail if --randomized_sketch_test
            #        1. Learned policy "passes" verification over learning instances.
            #        2. Verification over all instances goes again over learning instances.
            #        3. Second time, over learning instances, verification fails because it reaches a state not seen before.
            assert next_idx not in current_idxs, "See CHECK note in source file"
            next_idxs: List[int] = current_idxs + [next_idx]

        return next_idxs


class Wrapper(WrapperBase):
    def __init__(self, benchmark: Benchmark, feature_pool: FeaturePool, learner: Any, timers: Statistics, **kwargs):
        super().__init__(benchmark, feature_pool, learner, timers, **kwargs)

    def _inner_step(self, iteration_data: IterationData) -> Dict[str, Any]:
        self._timers.resume("preprocessing")

        # Relevant vertices must be those in example paths, deadend paths, and non-covered vertices
        for instance_idx, relevant_vertices in iteration_data.relevant_vertices.items():
            vertices_in_paths: Set[int] = set().union(*[path for idx, path in iteration_data.paths_with_idx if idx == instance_idx])
            assert vertices_in_paths.issubset(relevant_vertices)
            assert iteration_data.vertices_in_deadend_paths.get(instance_idx, set()).issubset(relevant_vertices)
            assert iteration_data.non_covered_vertices.get(instance_idx, set()).issubset(relevant_vertices)

        # Setup information for relevant vertices and off-path neighbors
        for instance_idx, relevant_vertices in iteration_data.relevant_vertices.items():
            logging.info(f"ITERATION: instance_idx={instance_idx}, #relevant={len(relevant_vertices)}")
            instance_data: PDDLInstance = self._instance_datas[instance_idx]

            # Setup information for relevant vertices and successors
            logging.info(f"Initializing {len(relevant_vertices)} relevant vertices...")
            for state_idx in relevant_vertices:
                if state_idx not in self._setup_registry.get(instance_idx, []):
                    self.setup_ext_state((instance_idx, state_idx))
                    if not instance_data.is_goal_state(state_idx):
                        successors: List[Tuple[Tuple[int, Any], str]] = instance_data.get_successors(state_idx)
                        logging.info(f"Setup ext_state={(instance_idx, state_idx)}, #successors={len(successors)}")
                        for (succ_state_idx, succ_state), operator in successors:
                            self.setup_ext_state((instance_idx, succ_state_idx))
                    else:
                        logging.info(f"Setup ext_state={(instance_idx, state_idx)} (goal state)")
                    self._setup_registry[instance_idx].add(state_idx)

        self._timers.stop("preprocessing")
        self._timers.resume("learner")
        # TODO: CHECK timers

        # Prepare learner for this iteration
        num_relevant_vertices = sum(map(lambda item: len(item), iteration_data.relevant_vertices.values()))
        logging.info(f"Bundle: bundle={iteration_data.paths_with_idx}, #relevant_vertices={num_relevant_vertices}, relevant_vertices={iteration_data.relevant_vertices}")

        # Solve
        try:
            solution: Dict[str, Any] = self._learner.solve(**self._options)
            solution.pop("conditions_for_goal") # Don't perform goal-condition test during sketch testing as "non-goal" failures are not repaired
        except RuntimeError as rt:
            logging.info(str(rt))
            self._timers.print(title="Finalizing due to RuntimeError exception...", logger=True)
            raise

        return solution

    def _inner_loop(self, iteration_data: IterationData) -> SketchReduced:
        # Inner loop, at each iteration one or more relevant vertices are added
        iteration_data.inner_iterations: int = 0
        while True:
            logging.info(f"New Inner ITERATION {iteration_data.inner_iterations} (total={self._total_inner_iterations}): Memory usage={memory_usage() / 1024:0.2f} GiB")
            iteration_data.inner_iterations += 1
            self._total_inner_iterations += 1

            # Compute solution
            solution: Dict[str, Any] = self._inner_step(iteration_data)
            f_idxs: List[int] = solution.get("f_idxs")
            rules_with_decorations: Set[Tuple[dlplan_policy.Rule, FrozenSet[int], FrozenSet[int]]] = solution.get("rules_with_decorations")
            conditions_for_goal: List[Dict[int, int]] = solution.get("conditions_for_goal")

            # Create sketch from solution
            dlplan_policy = D2sepDlplanPolicyFactory().make_dlplan_policy_from_rules_with_decorations(f_idxs, rules_with_decorations, self._policy_builder, iteration_data)
            sketch = SketchReduced(self._benchmark._state_factory, iteration_data.feature_pool, f_idxs, dlplan_policy, self._width)
            logging.info("Learned the following sketch:")
            sketch.print(logger=True)

            # Analyze learned sketch
            self._timers.resume("verification")
            logging.info(colored(f"Verifying learned sketch of width={self._width} on TRAINING instances {iteration_data.current_idxs}...", "blue"))
            unsolved_instances: List[Tuple[PDDLInstance, Any]] = self.get_unsolved_instances(iteration_data, iteration_data.instance_datas, sketch, **{"conditions_for_goal": conditions_for_goal})
            self._timers.stop("verification")

            if len(unsolved_instances) == 0:
                logging.info(f"Sketch SOLVES TRAINING instances {iteration_data.current_idxs}")
                return sketch
            else:
                # Sketch doesn't solve the instance used for learning
                old_len = sum(map(lambda states: len(states), iteration_data.relevant_vertices.values()))
                for instance_data, reason in unsolved_instances:
                    logging.info(f"  {instance_data.instance_filepath()} (idx={instance_data.idx})")
                    logging.info(f"  Reason: {reason}")

                    if "non-closed" in reason:
                        for instance_idx, state_idx in reason.get("non-closed", []):
                            if instance_idx in iteration_data.current_idxs:
                                iteration_data.non_covered_vertices[instance_idx].add(state_idx)
                                iteration_data.relevant_vertices[instance_idx].add(state_idx)
                                logging.info(f"Adding ext_state {(instance_idx, state_idx)} to non-covered vertices")
                            else:
                                unsolved_idxs.add(instance_idx)
                    elif "non-sound" in reason:
                        for instance_idx, state_idx in reason.get("non-sound", []):
                            if instance_idx in iteration_data.current_idxs:
                                iteration_data.non_covered_vertices[instance_idx].add(state_idx)
                                iteration_data.relevant_vertices[instance_idx].add(state_idx)
                                logging.info(f"Adding ext_state {(instance_idx, state_idx)} to non-covered vertices")
                            else:
                                unsolved_idxs.add(instance_idx)
                    elif "deadend" in reason:
                        if not self._options.get("deadends", False): logging.warning(f"DEADEND reached even though 'deadends=False'")
                        deadend_path: Tuple[Tuple[int, int]] = reason.get("deadend")
                        logging.info(f"{len(iteration_data.deadend_paths)} (current) deadend path(s); paths={iteration_data.deadend_paths}")
                        assert deadend_path is not None and deadend_path not in iteration_data.deadend_paths
                        iteration_data.deadend_paths.append(deadend_path)

                        instance_idx = deadend_path[0][0]
                        vertices_in_deadend_path: Set[int] = set([state_idx for _, state_idx in deadend_path])
                        iteration_data.vertices_in_deadend_paths[instance_idx] |= vertices_in_deadend_path
                        iteration_data.relevant_vertices[instance_idx] |= vertices_in_deadend_path
                        logging.info(f"Adding {vertices_in_deadend_path} to relevant_vertices")
                    else:
                        raise RuntimeError(f"Unknown reason: {reason}")

    def _outer_loop(self) -> Tuple[SketchReduced, List[IterationData]]:
        self._total_inner_iterations: int = 0
        trace: List[IterationData] = []
        previous_idxs: List[int] = []

        # Get initial set of instances
        current_idxs: List[int] = self.get_initial_idxs()
        assert len(current_idxs) > 0

        # Outer loop
        while True:
            logging.info(f"ADVANCING to TRAINING instances {current_idxs}")
            logging.info(colored(f"Iteration: {len(trace)}", "red"))
            with change_dir(f"outer-{len(trace)}", enable=self._enable_dump_files or self._dump_asp_program):
                for instance_idx in current_idxs:
                    logging.info(f"    id: {instance_idx}.[{self._instance_datas[instance_idx].instance_filepath()}]")

                # Consolidate data for iteration
                iteration_data = IterationData()
                iteration_data.current_idxs: List[int] = current_idxs
                iteration_data.preprocessed_datas: List[Dict[str, Any]] = [self._benchmark._preprocessed_datas[instance_idx] for instance_idx in current_idxs]
                iteration_data.instance_datas: List[InstanceData] = [self._instance_datas[instance_idx] for instance_idx in current_idxs]
                iteration_data.paths_with_idx: List[Tuple[int, Tuple[int]]] = list(zip(current_idxs, [datum.get("marked_state_trajectories")[0] for datum in iteration_data.preprocessed_datas]))
                iteration_data.relevant_vertices: Dict[int, Set[int]] = {instance_idx: set(self._benchmark._initial_relevant_vertices.get(instance_idx)) for instance_idx in current_idxs}
                iteration_data.feature_pool: List[Feature] = self._feature_pool._pool

                #iteration_data.ext_state_to_dlplan_state_index: Dict[Tuple[int, int], int] = self._ext_state_to_dlplan_state_index
                iteration_data.ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = self._ext_state_to_feature_valuations
                iteration_data.instance_idx_to_denotations_caches: Dict[int, dlplan_core.DenotationsCaches] = self._instance_idx_to_denotations_caches

                iteration_data.non_covered_vertices: Dict[int, Set[int]] = defaultdict(set)
                iteration_data.vertices_in_deadend_paths: Dict[int, Set[int]] = defaultdict(set)
                iteration_data.deadend_paths: List[Tuple[Tuple[int, int]]] = []

                # Update trace for outer loop
                trace.append(iteration_data)

                # Sanity check
                for instance_idx, path in iteration_data.paths_with_idx:
                    _relevant_vertices_for_instance_idx: Set[int] = set(iteration_data.relevant_vertices.get(instance_idx))
                    assert set(path).issubset(_relevant_vertices_for_instance_idx)

                # Flag learner for new inner iteration
                self._learner.prepare_iteration(iteration_data)

                # Solve current idxs
                sketch: SketchReduced = self._inner_loop(iteration_data)

                # Verify whether sketch solves all instances in benchmark
                self._timers.resume("verification")
                logging.info(colored("Verifying learned sketch of width={self._width} on ALL instances ({len(self._instance_datas_)} instance(s))...", "blue"))
                unsolved_instances: List[Tuple[PDDLInstance, Any]] = self.get_unsolved_instances(iteration_data, self._instance_datas, sketch)
                self._timers.stop("verification")

                if len(unsolved_instances) == 0:
                    logging.info(colored(f"Sketch SOLVES ALL {len(self._instance_datas)} instance(s)! [Bundle: {iteration_data.paths_with_idx}]", "blue"))
                    return sketch, trace

                # Sketch doesn't solve an instance different from the one used for learning
                logging.info(f"Sketch doesn't solve an instance outside learning set {current_idxs}")
                for instance_data, reason in unsolved_instances:
                    logging.info(f"  {instance_data.instance_filepath()} (idx={instance_data.idx})")
                    logging.info(f"  Reason: {reason}")

                previous_idxs += current_idxs
                current_idxs: List[int] = self.advance_idxs(current_idxs, previous_idxs, [instance_data.idx for instance_data, _ in unsolved_instances])

        return None, trace

    def learn(self):
        logging.info(f"Writing iteration information to '{self._folder_name_for_iterations}'")
        with change_dir(self._folder_name_for_iterations):
            sketch, trace = self._outer_loop()

        logging.info(f"Writing output to '{self._folder_name_for_output}'")
        with change_dir(self._folder_name_for_output):
            print_separation_line(logger=True)
            logging.info(colored("Summary:", "green"))

            learning_statistics = Statistics()
            learning_statistics.add("num_training_instances", len(self._instance_datas))
            learning_statistics.add("num_outer_iterations", len(trace))
            learning_statistics.add("num_total_inner_iterations", self._total_inner_iterations)
            learning_statistics.add("num_last_inner_iterations", trace[-1].inner_iterations)
            learning_statistics.add("num_selected_training_instances (|P|)", len(trace[-1].instance_datas))
            learning_statistics.add("num_states_in_selected_training_instances (|S|)", sum(len(vertices) for vertices in trace[-1].relevant_vertices.values()))
            learning_statistics.add("num_features_in_pool (|F|)", len(trace[-1].feature_pool))
            learning_statistics.print(title="Learning statistics:", logger=True)

            self._timers.add("memory/total", memory_usage() / 1024)
            self._timers.print(logger=True)

            print_separation_line(logger=True)

            logging.info("Resulting sketch:")
            sketch.print(logger=True)
            print_separation_line(logger=True)

            write_file(f"sketch_{self._width}.txt", str(sketch._dlplan_policy))
            write_file(f"sketch_minimized_{self._width}.txt", str(sketch._dlplan_policy))

            logging.info(f"SUCCESS {self._timers.get_elapsed_sec('feature/pool'):.02f} feature-pool {self._timers.get_elapsed_sec('preprocessing'):.02f} preprocessing {self._timers.get_elapsed_sec('asp'):.02f} ASP {self._timers.get_elapsed_sec('total'):.02f} total")
            print_separation_line(logger=True)


class WrapperEnumeration(WrapperBase):
    def __init__(self, benchmark: Benchmark, feature_pool: FeaturePool, learner: Any, timers: Statistics, **kwargs):
        super().__init__(benchmark, feature_pool, learner, timers, **kwargs)

    def _get_node_statistics(self) -> Dict[str, Any]:
        return SolutionGenerator.get_node_statistics()

    def _solutions(self, iteration_data: IterationData) -> Generator[Tuple[List[int], Set[Any], int], None, None]:
        # Relevant vertices must be those in example paths, deadend paths, and non-covered vertices
        for instance_idx, relevant_vertices in iteration_data.relevant_vertices.items():
            vertices_in_paths: Set[int] = set().union(*[path for idx, path in iteration_data.paths_with_idx if idx == instance_idx])
            assert vertices_in_paths.issubset(relevant_vertices)
            assert iteration_data.vertices_in_deadend_paths.get(instance_idx, set()).issubset(relevant_vertices)
            assert iteration_data.non_covered_vertices.get(instance_idx, set()).issubset(relevant_vertices)

        # Setup information for relevant vertices and off-path neighbors
        self._timers.resume("preprocessing")
        for instance_idx, relevant_vertices in iteration_data.relevant_vertices.items():
            logging.info(f"ITERATION: instance_idx={instance_idx}, #relevant={len(relevant_vertices)}")
            instance_data: PDDLInstance = self._instance_datas[instance_idx]

            # Setup information for relevant vertices and successors
            logging.info(f"Initializing {len(relevant_vertices)} relevant vertices...")
            for state_idx in relevant_vertices:
                if state_idx not in self._setup_registry.get(instance_idx, []):
                    self.setup_ext_state((instance_idx, state_idx))
                    if not instance_data.is_goal_state(state_idx):
                        successors: List[Tuple[Tuple[int, Any], str]] = instance_data.get_successors(state_idx)
                        logging.info(f"Setup ext_state={(instance_idx, state_idx)}, goal=False, #successors={len(successors)}")
                        for (succ_state_idx, succ_state), operator in successors:
                            self.setup_ext_state((instance_idx, succ_state_idx))
                    else:
                        logging.info(f"Setup ext_state={(instance_idx, state_idx)}, goal=True")
                    self._setup_registry[instance_idx].add(state_idx)

        self._timers.stop("preprocessing")

        yield from self._learner.solutions(**self._options)

    def _inner_loop(self, iteration_data: IterationData) -> Tuple[SketchReduced, Set[int]]:
        # Inner loop, at each iteration one or more relevant vertices are added
        iteration_data.inner_iterations: int = 0
        while True:
            logging.info(f"New Inner ITERATION {iteration_data.inner_iterations} (total={self._total_inner_iterations}): Memory usage={memory_usage() / 1024:0.2f} GiB")
            iteration_data.inner_iterations += 1
            self._total_inner_iterations += 1
            unsolved_idxs: Set[int] = set()

            failed_training_instances: List[int] = []
            solution_idx_to_unsolved_instances: List[List[Tuple[PDDLInstance, Any]]] = []
            for solution_idx, solution in enumerate(self._solutions(iteration_data)):
                f_idxs: List[int] = solution.get("f_idxs")
                rules_with_decorations: Set[Tuple[dlplan_policy.Rule, FrozenSet[int], FrozenSet[int]]] = solution.get("rules_with_decorations")
                conditions_for_goal: List[Dict[int, int]] = solution.get("conditions_for_goal")

                logging.info(colored(f"Got solution {1 + solution_idx}: cost={solution.get('cost')}, f_idxs={f_idxs}, conditions_for_goal={conditions_for_goal}", "green"))
                assert f_idxs is not None

                # Create sketch from solution
                dlplan_policy = D2sepDlplanPolicyFactory().make_dlplan_policy_from_rules_with_decorations(f_idxs, rules_with_decorations, self._policy_builder, iteration_data)
                sketch = SketchReduced(self._benchmark._state_factory, iteration_data.feature_pool, f_idxs, dlplan_policy, self._width)
                logging.info("Sketch:")
                sketch.print(logger=True)

                # Analyze learned sketch
                self._timers.resume("verification")
                logging.info(colored(f"Verifying learned sketch of width={self._width} on TRAINING instances {iteration_data.current_idxs}...", "blue"))
                unsolved_instances: List[Tuple[PDDLInstance, Any]] = self.get_unsolved_instances(iteration_data, iteration_data.instance_datas, sketch, **{"conditions_for_goal": conditions_for_goal})
                self._timers.stop("verification")

                if len(unsolved_instances) == 0:
                    logging.info(f"Sketch SOLVES TRAINING instances {iteration_data.current_idxs}")

                    # Verify whether sketch solves all instances in benchmark
                    self._timers.resume("verification")
                    logging.info(colored("Verifying learned sketch of width={self._width} on ALL instances...", "blue"))
                    unsolved_instances: List[Tuple[PDDLInstance, Any]] = self.get_unsolved_instances(iteration_data, self._instance_datas, sketch, **{"conditions_for_goal": conditions_for_goal})
                    self._timers.stop("verification")

                    if len(unsolved_instances) == 0:
                        logging.info(colored(f"Sketch SOLVES ALL {len(self._instance_datas)} instance(s)! [Bundle: {iteration_data.paths_with_idx}]", "blue"))
                        node_statistics: Dict[str, Any] = self._get_node_statistics()
                        logging.info(f"Statistics: {node_statistics}")
                        return sketch, None
                else:
                    failed_training_instances.append(solution_idx)

                # If control reaches here, there are unsolved instances for this solution
                new_non_covered_vertices: bool = False
                solution_idx_to_unsolved_instances.append(unsolved_instances)
                for instance_data, reason in unsolved_instances:
                    if "non-closed" in reason:
                        # TODO: If some non-closed state, try to resolve it via policy simplification before solving for termination
                        for instance_idx, state_idx in reason.get("non-closed", []):
                            if instance_idx in iteration_data.current_idxs:
                                iteration_data.non_covered_vertices[instance_idx].add(state_idx)
                                iteration_data.relevant_vertices[instance_idx].add(state_idx)
                                logging.info(f"Adding ext_state {(instance_idx, state_idx)} to non-covered vertices")
                            else:
                                unsolved_idxs.add(instance_idx)
                    elif "non-sound" in reason:
                        # TODO: If some non-sound state, try to resolve it via policy simplification before solving for termination
                        for instance_idx, state_idx in reason.get("non-sound", []):
                            if instance_idx in iteration_data.current_idxs:
                                iteration_data.non_covered_vertices[instance_idx].add(state_idx)
                                iteration_data.relevant_vertices[instance_idx].add(state_idx)
                                logging.info(f"Adding ext_state {(instance_idx, state_idx)} to non-covered vertices")
                            else:
                                unsolved_idxs.add(instance_idx)
                    elif "deadend" in reason:
                        # TODO
                        raise RuntimeError("Case of 'deadend' in reason not yet implemented")
                        """
                        if not self._options.get("deadends", False): logging.warning(f"WARNING: DEADEND reached even though 'deadends=False'")
                        deadend_path: Tuple[Tuple[int, int]] = reason.get("deadend")
                        logging.info(f"{len(iteration_data.deadend_paths)} deadend path(s); paths={iteration_data.deadend_paths}")
                        assert deadend_path is not None and deadend_path not in iteration_data.deadend_paths
                        iteration_data.deadend_paths.append(deadend_path)

                        instance_idx = deadend_path[0][0]
                        vertices_in_deadend_path: Set[int] = set([state_idx for _, state_idx in deadend_path])
                        iteration_data.vertices_in_deadend_paths[instance_idx] |= vertices_in_deadend_path
                        iteration_data.relevant_vertices[instance_idx] |= vertices_in_deadend_path
                        logging.info(f"Adding {vertices_in_deadend_path} to relevant_vertices")
                        """
                    else:
                        raise RuntimeError(f"Unknown reason: {reason}")

            # There are no more solutions, either because ran out of resources (i.e., #solutions or max cost bound),
            # or there is not sufficient expressivity in pool of features
            # TODO: CHECK CASE
            raise RuntimeError("Ran out of solutions")

            node_statistics: Dict[str, Any] = self._get_node_statistics()
            logging.info(f"Statistics: {node_statistics}")

            # If no solution, return to outer loop
            if len(solution_idx_to_unsolved_instances) == 0:
                return None, None

            # If sketch solves training instances but not benchmark, return to outer loop
            if len(unsolved_idxs) > 0:
                assert len(set(iteration_data.current_idxs) & unsolved_idxs) == 0
                return None, unsolved_idxs

        raise RuntimeError("Control shouldn't reach here")

    def _outer_loop(self) -> Tuple[SketchReduced, List[IterationData]]:
        self._total_inner_iterations: int = 0
        trace: List[IterationData] = []
        previous_idxs: List[int] = []

        # Get initial set of instances
        current_idxs: List[int] = self.get_initial_idxs()
        assert len(current_idxs) > 0

        # Outer loop
        while True:
            logging.info(f"ADVANCING to TRAINING instances {current_idxs}")
            logging.info(colored(f"Iteration: {len(trace)}", "red"))
            with change_dir(f"outer-{len(trace)}", enable=self._enable_dump_files or self._dump_asp_program):
                for instance_idx in current_idxs:
                    logging.info(f"    id: {instance_idx}.[{self._instance_datas[instance_idx].instance_filepath()}]")

                # Consolidate data for iteration
                iteration_data = IterationData()
                iteration_data.current_idxs: List[int] = current_idxs
                iteration_data.preprocessed_datas: List[Dict[str, Any]] = [self._benchmark._preprocessed_datas[instance_idx] for instance_idx in current_idxs]
                iteration_data.instance_datas: List[InstanceData] = [self._instance_datas[instance_idx] for instance_idx in current_idxs]
                iteration_data.paths_with_idx: List[Tuple[int, Tuple[int]]] = list(zip(current_idxs, [datum.get("marked_state_trajectories")[0] for datum in iteration_data.preprocessed_datas]))
                iteration_data.relevant_vertices: Dict[int, Set[int]] = {instance_idx: set(self._benchmark._initial_relevant_vertices.get(instance_idx)) for instance_idx in current_idxs}
                iteration_data.feature_pool: List[Feature] = self._feature_pool._pool

                #iteration_data.ext_state_to_dlplan_state_index: Dict[Tuple[int, int], int] = self._ext_state_to_dlplan_state_index
                iteration_data.ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = self._ext_state_to_feature_valuations
                iteration_data.instance_idx_to_denotations_caches: Dict[int, dlplan_core.DenotationsCaches] = self._instance_idx_to_denotations_caches

                iteration_data.non_covered_vertices: Dict[int, Set[int]] = defaultdict(set)
                iteration_data.vertices_in_deadend_paths: Dict[int, Set[int]] = defaultdict(set)
                iteration_data.deadend_paths: List[Tuple[Tuple[int, int]]] = []

                # Update trace for outer loop
                trace.append(iteration_data)

                # Sanity check
                for instance_idx, path in iteration_data.paths_with_idx:
                    _relevant_vertices_for_instance_idx: Set[int] = set(iteration_data.relevant_vertices.get(instance_idx))
                    assert set(path).issubset(_relevant_vertices_for_instance_idx)

                # Flag learner for new inner iteration
                self._learner.prepare_iteration(iteration_data)

                # Solve current idxs
                sketch, unsolved_idxs = self._inner_loop(iteration_data)
                assert sketch is None or unsolved_idxs is None

                if sketch is None and unsolved_idxs is None:
                    # No solution found, abort
                    break
                elif unsolved_idxs is None:
                    # Sketch solves the whole benchmark
                    return sketch, trace

                # Sketch doesn't solve an instance different from the one used for learning
                previous_idxs += current_idxs
                current_idxs: List[int] = self.advance_idxs(current_idxs, previous_idxs, unsolved_idxs)

        return None, trace

    def learn(self):
        logging.info(f"Writing iteration information to '{self._folder_name_for_iterations}'")
        with change_dir(self._folder_name_for_iterations):
            sketch, trace = self._outer_loop()

        # Report statistics
        print_separation_line(logger=True)
        logging.info(colored("Summary:", "green"))

        learning_statistics = Statistics()
        learning_statistics.add("num_training_instances", len(self._instance_datas))
        learning_statistics.add("num_outer_iterations", len(trace))
        learning_statistics.add("num_total_inner_iterations", self._total_inner_iterations)
        learning_statistics.add("num_last_inner_iterations", trace[-1].inner_iterations)
        learning_statistics.add("num_selected_training_instances (|P|)", len(trace[-1].instance_datas))
        learning_statistics.add("num_states_in_selected_training_instances (|S|)", sum(len(vertices) for vertices in trace[-1].relevant_vertices.values()))
        learning_statistics.add("num_features_in_pool (|F|)", len(trace[-1].feature_pool))
        learning_statistics.print(title="Learning statistics:", logger=True)

        self._timers.add("memory/total", memory_usage() / 1024)
        self._timers.print(logger=True)
        print_separation_line(logger=True)

        # If no sketch found, terminate; else, write sketch to output folder
        if sketch is None:
            logging.info(f"FAILURE (no sketch learned)")
            print_separation_line(logger=True)
        else:
            logging.info(f"Writing output to '{self._folder_name_for_output}'")
            with change_dir(self._folder_name_for_output):
                logging.info("Resulting sketch:")
                sketch.print(logger=True)
                print_separation_line(logger=True)

                write_file(f"sketch_{self._width}.txt", str(sketch._dlplan_policy))
                write_file(f"sketch_minimized_{self._width}.txt", str(sketch._dlplan_policy))

                logging.info(f"SUCCESS {self._timers.get_elapsed_sec('feature/pool'):.02f} feature-pool {self._timers.get_elapsed_sec('preprocessing'):.02f} preprocessing {self._timers.get_elapsed_sec('asp'):.02f} ASP {self._timers.get_elapsed_sec('total'):.02f} total")
                print_separation_line(logger=True)


class WrapperEnumerationV2(WrapperBase):
    def __init__(self, benchmark: Benchmark, feature_pool: FeaturePool, learner: Any, timers: Statistics, **kwargs):
        super().__init__(benchmark, feature_pool, learner, timers, **kwargs)

    def _get_node_statistics(self) -> Dict[str, Any]:
        return SolutionGenerator.get_node_statistics()

    def _solutions(self, iteration_data: IterationData) -> Generator[Tuple[List[int], Set[Any], int], None, None]:
        # Relevant vertices must be those in example paths, deadend paths, and non-covered vertices
        for instance_idx, relevant_vertices in iteration_data.relevant_vertices.items():
            vertices_in_paths: Set[int] = set().union(*[path for idx, path in iteration_data.paths_with_idx if idx == instance_idx])
            assert vertices_in_paths.issubset(relevant_vertices)
            assert iteration_data.vertices_in_deadend_paths.get(instance_idx, set()).issubset(relevant_vertices)
            assert iteration_data.non_covered_vertices.get(instance_idx, set()).issubset(relevant_vertices)

        # Setup information for relevant vertices and off-path neighbors
        self._timers.resume("preprocessing")
        for instance_idx, relevant_vertices in iteration_data.relevant_vertices.items():
            logging.info(f"ITERATION: instance_idx={instance_idx}, #relevant={len(relevant_vertices)}")
            instance_data: PDDLInstance = self._instance_datas[instance_idx]

            # Setup information for relevant vertices and successors
            logging.info(f"Initializing {len(relevant_vertices)} relevant vertices...")
            for state_idx in relevant_vertices:
                if state_idx not in self._setup_registry.get(instance_idx, []):
                    self.setup_ext_state((instance_idx, state_idx))
                    if not instance_data.is_goal_state(state_idx):
                        successors: List[Tuple[Tuple[int, Any], str]] = instance_data.get_successors(state_idx)
                        logging.info(f"Setup ext_state={(instance_idx, state_idx)}, goal=False, #successors={len(successors)}")
                        for (succ_state_idx, succ_state), operator in successors:
                            self.setup_ext_state((instance_idx, succ_state_idx))
                    else:
                        logging.info(f"Setup ext_state={(instance_idx, state_idx)}, goal=True")
                    self._setup_registry[instance_idx].add(state_idx)

        self._timers.stop("preprocessing")

        non_covered_ext_states: List[Tuple[int, int]] = [(instance_idx, state_idx) for instance_idx, state_idxs in iteration_data.non_covered_vertices.items() for state_idx in state_idxs]
        yield from self._learner.solutions(non_covered_ext_states, **self._options)

    def _repair(self, solution: Dict[str, Any], reasons: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._learner.repair(solution, reasons)

    def _inner_loop(self, iteration_data: IterationData) -> Tuple[SketchReduced, Set[int]]:
        # Inner loop, at each iteration one or more relevant vertices are added
        iteration_data.inner_iterations: int = 0
        max_num_restarts: int = 25
        max_num_failures: int = 1
        max_num_ext_states_added: int = 2

        num_ext_states_added: int = 0
        for enumeration_id in range(max_num_restarts):
            logging.info(f"New enumeration {iteration_data.inner_iterations} (total={self._total_inner_iterations}): Memory usage={memory_usage() / 1024:0.2f} GiB")
            with change_dir(f"enum-{enumeration_id}", enable=self._enable_dump_files or self._dump_asp_program):
                iteration_data.inner_iterations += 1
                self._total_inner_iterations += 1

                num_failures: int = 0
                non_sound_ext_states: List[Tuple[int, int]] = []
                for solution_idx, solution in enumerate(self._solutions(iteration_data)):
                    logging.info(colored(f"Got solution {1 + solution_idx}: cost={solution.get('cost')}, f_idxs={solution.get('f_idxs')}", "green"))
                    assert solution.get("f_idxs") is not None

                    while solution.get("rules_with_decorations") is not None:
                        f_idxs: List[int] = solution.get("f_idxs")
                        rules_with_decorations: Set[Tuple[dlplan_policy.Rule, FrozenSet[int], FrozenSet[int]]] = solution.get("rules_with_decorations")
                        conditions_for_goal: List[Dict[int, int]] = solution.get("conditions_for_goal")
                        logging.info(f"Conditions for goal: f_idxs={f_idxs}, conditions={conditions_for_goal}")

                        # Create sketch from solution
                        dlplan_policy = D2sepDlplanPolicyFactory().make_dlplan_policy_from_rules_with_decorations(f_idxs, rules_with_decorations, self._policy_builder, iteration_data)
                        sketch = SketchReduced(self._benchmark._state_factory, iteration_data.feature_pool, f_idxs, dlplan_policy, self._width)
                        logging.info("Sketch:")
                        sketch.print(logger=True)

                        # Check verification
                        assert "verify_closure" in solution and "verify_bad_edges" in solution, solution
                        verify_closure: List[Tuple[Tuple[int, int]]] = solution.get("verify_closure")
                        verify_bad_edges: List[Tuple[int, Tuple[int, int]]] = solution.get("verify_bad_edges")
                        if verify_closure is not None and len(verify_closure) > 0:
                            logging.warning(f"Verify closure: {verify_closure}")
                            assert False
                        if verify_bad_edges is not None and len(verify_bad_edges) > 0:
                            logging.warning(f"Verify bad edges: {verify_bad_edges}")
                            assert False

                        # Analyze learned sketch
                        self._timers.resume("verification")
                        logging.info(colored(f"Verifying learned sketch of width={self._width} on TRAINING instances {iteration_data.current_idxs}...", "blue"))
                        unsolved_instances: List[Tuple[PDDLInstance, Any]] = self.get_unsolved_instances(iteration_data, iteration_data.instance_datas, sketch, **{"conditions_for_goal": conditions_for_goal})
                        self._timers.stop("verification")

                        if len(unsolved_instances) == 0:
                            logging.info(f"Sketch SOLVES TRAINING instances {iteration_data.current_idxs}")

                            # Verify whether sketch solves all instances in benchmark
                            self._timers.resume("verification")
                            logging.info(colored("Verifying learned sketch of width={self._width} on ALL instances...", "blue"))
                            unsolved_instances: List[Tuple[PDDLInstance, Any]] = self.get_unsolved_instances(iteration_data, self._instance_datas, sketch, **{"conditions_for_goal": conditions_for_goal})
                            self._timers.stop("verification")

                            if len(unsolved_instances) == 0:
                                logging.info(colored(f"Sketch SOLVES ALL {len(self._instance_datas)} instance(s)! [Bundle: {iteration_data.paths_with_idx}]", "blue"))
                                node_statistics: Dict[str, Any] = self._get_node_statistics()
                                logging.info(f"Statistics: {node_statistics}")
                                return sketch, None

                        # If control reaches here, there are unsolved instances for this solution
                        reasons: List[Dict[str, Any]] = []
                        for instance_data, reason in unsolved_instances:
                            if "non-closed" in reason:
                                raise RuntimeError("Case of 'non-closed' in reason not yet implemented")
                            elif "non-sound" in reason:
                                for instance_idx, state_idx in reason.get("non-sound", []):
                                    if instance_idx in iteration_data.current_idxs:
                                        logging.warning(f"Adding ext_state {(instance_idx, state_idx)} to non-sound vertices")
                                        assert state_idx not in iteration_data.non_covered_vertices.get(instance_idx, set())
                                        non_sound_ext_states.append((instance_idx, state_idx))
                                        #logging.warning(f"Adding ext_state {(instance_idx, state_idx)} to non-covered vertices")
                                        #assert state_idx not in iteration_data.non_covered_vertices.get(instance_idx, set())
                                        #assert state_idx not in iteration_data.relevant_vertices.get(instance_idx, set())
                                        #iteration_data.non_covered_vertices[instance_idx].add(state_idx)
                                        #iteration_data.relevant_vertices[instance_idx].add(state_idx)
                                        #reason: Dict[str, Any] = {"non-sound": {(instance_idx, state_idx)}}
                                        #reasons.append(reason)
                                        num_failures += 1
                            elif "non-goal" in reason:
                                ext_state, conditions = reason.get("non-goal")
                                reason: Dict[str, Any] = {"non-goal": {ext_state}}
                                reasons.append(reason)
                                num_failures += 1
                            elif "deadend" in reason:
                                raise RuntimeError("Case of 'deadend' in reason not yet implemented")
                            else:
                                raise RuntimeError(f"Unknown reason: {reason}")

                        # Try to recover by recalculating decorations
                        # NOTES: Repair is not sufficient. Need to incorporate new state and action for it, and recompute.
                        # Maybe after failed repair, until no solution.
                        if len(reasons) > 0:
                            solution: Dict[str, Any] = self._repair(solution, reasons)
                        else:
                            break

                    # If number of discovered non-sound states is too high, break to
                    # add them to relevant vertices and start again enumeration of solutions
                    if num_failures >= max_num_failures:
                        num_ext_states_added_iteration: int = 0
                        counter_for_non_sound_ext_states: Counter = Counter(non_sound_ext_states)
                        print(f"counter_for_non_sound_ext_states: {counter_for_non_sound_ext_states}")
                        for (instance_idx, state_idx), count in counter_for_non_sound_ext_states.most_common(max_num_ext_states_added):
                            assert state_idx not in iteration_data.non_covered_vertices[instance_idx]
                            logging.warning(f"Adding ext_state {(instance_idx, state_idx)} to non-covered vertices (count={count})")
                            iteration_data.non_covered_vertices[instance_idx].add(state_idx)
                            iteration_data.relevant_vertices[instance_idx].add(state_idx)
                            num_ext_states_added_iteration += 1
                            num_ext_states_added += 1
                        logging.warning(f"RESTART: enumeration_id={enumeration_id + 1}, #failures={num_failures}, #non_sound_ext_states={len(frozenset(non_sound_ext_states))}, #added={(num_ext_states_added_iteration, num_ext_states_added)}")
                        break
                    else:
                        logging.warning(f"MOVE TO NEXT SOLUTION")

                # Setup information for next enumeration
                self._timers.resume("preprocessing")

                # Relevant vertices must be those in example paths, deadend paths, and non-covered vertices
                for instance_idx, relevant_vertices in iteration_data.relevant_vertices.items():
                    vertices_in_paths: Set[int] = set().union(*[path for idx, path in iteration_data.paths_with_idx if idx == instance_idx])
                    assert vertices_in_paths.issubset(relevant_vertices)
                    assert iteration_data.vertices_in_deadend_paths.get(instance_idx, set()).issubset(relevant_vertices)
                    assert iteration_data.non_covered_vertices.get(instance_idx, set()).issubset(relevant_vertices)

                self._timers.stop("preprocessing")
                continue
                raise RuntimeError("Ran out of solutions")



                # There are no more solutions, either because ran out of resources (i.e., #solutions or max cost bound),
                # or there is not sufficient expressivity in pool of features
                raise RuntimeError("Ran out of solutions")

                node_statistics: Dict[str, Any] = self._get_node_statistics()
                logging.info(f"Statistics: {node_statistics}")

                # If no solution, return to outer loop
                if len(solution_idx_to_unsolved_instances) == 0:
                    return None, None

                # If sketch solves training instances but not benchmark, return to outer loop
                if len(unsolved_idxs) > 0:
                    assert len(set(iteration_data.current_idxs) & unsolved_idxs) == 0
                    return None, unsolved_idxs

        raise RuntimeError("Control shouldn't reach here")

    def _outer_loop(self) -> Tuple[SketchReduced, List[IterationData]]:
        self._total_inner_iterations: int = 0
        trace: List[IterationData] = []
        previous_idxs: List[int] = []

        # Get initial set of instances
        current_idxs: List[int] = self.get_initial_idxs()
        #logging.warning("INITIAL IDXS SET TO [0, 1]")
        #current_idxs: List[int] = [0, 1]
        assert len(current_idxs) > 0

        # Outer loop
        while True:
            logging.info(f"ADVANCING to TRAINING instances {current_idxs}")
            logging.info(colored(f"Iteration: {len(trace)}", "red"))
            with change_dir(f"outer-{len(trace)}", enable=self._enable_dump_files or self._dump_asp_program):
                for instance_idx in current_idxs:
                    logging.info(f"    id: {instance_idx}.[{self._instance_datas[instance_idx].instance_filepath()}]")

                # Consolidate data for iteration
                iteration_data = IterationData()
                iteration_data.current_idxs: List[int] = current_idxs
                iteration_data.preprocessed_datas: List[Dict[str, Any]] = [self._benchmark._preprocessed_datas[instance_idx] for instance_idx in current_idxs]
                iteration_data.instance_datas: List[InstanceData] = [self._instance_datas[instance_idx] for instance_idx in current_idxs]
                iteration_data.paths_with_idx: List[Tuple[int, Tuple[int]]] = list(zip(current_idxs, [datum.get("marked_state_trajectories")[0] for datum in iteration_data.preprocessed_datas]))
                iteration_data.relevant_vertices: Dict[int, Set[int]] = {instance_idx: set(self._benchmark._initial_relevant_vertices.get(instance_idx)) for instance_idx in current_idxs}
                iteration_data.feature_pool: List[Feature] = self._feature_pool._pool

                #iteration_data.ext_state_to_dlplan_state_index: Dict[Tuple[int, int], int] = self._ext_state_to_dlplan_state_index
                iteration_data.ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = self._ext_state_to_feature_valuations
                iteration_data.instance_idx_to_denotations_caches: Dict[int, dlplan_core.DenotationsCaches] = self._instance_idx_to_denotations_caches

                iteration_data.non_covered_vertices: Dict[int, Set[int]] = defaultdict(set)
                iteration_data.vertices_in_deadend_paths: Dict[int, Set[int]] = defaultdict(set)
                iteration_data.deadend_paths: List[Tuple[Tuple[int, int]]] = []

                # Update trace for outer loop
                trace.append(iteration_data)

                # Sanity check
                for instance_idx, path in iteration_data.paths_with_idx:
                    _relevant_vertices_for_instance_idx: Set[int] = set(iteration_data.relevant_vertices.get(instance_idx))
                    assert set(path).issubset(_relevant_vertices_for_instance_idx)

                # Flag learner for new inner iteration
                self._learner.prepare_iteration(iteration_data)

                # Solve current idxs
                sketch, unsolved_idxs = self._inner_loop(iteration_data)
                assert sketch is None or unsolved_idxs is None

                if sketch is None and unsolved_idxs is None:
                    # No solution found, abort
                    logging.warning("CONTINUE")
                    continue
                    break
                elif unsolved_idxs is None:
                    # Sketch solves the whole benchmark
                    return sketch, trace

                # Sketch doesn't solve an instance different from the one used for learning
                previous_idxs += current_idxs
                current_idxs: List[int] = self.advance_idxs(current_idxs, previous_idxs, unsolved_idxs)

        return None, trace

    def learn(self):
        logging.info(f"Writing iteration information to '{self._folder_name_for_iterations}'")
        with change_dir(self._folder_name_for_iterations):
            sketch, trace = self._outer_loop()

        # Report statistics
        print_separation_line(logger=True)
        logging.info(colored("Summary:", "green"))

        learning_statistics = Statistics()
        learning_statistics.add("num_training_instances", len(self._instance_datas))
        learning_statistics.add("num_outer_iterations", len(trace))
        learning_statistics.add("num_total_inner_iterations", self._total_inner_iterations)
        learning_statistics.add("num_last_inner_iterations", trace[-1].inner_iterations)
        learning_statistics.add("num_selected_training_instances (|P|)", len(trace[-1].instance_datas))
        learning_statistics.add("num_states_in_selected_training_instances (|S|)", sum(len(vertices) for vertices in trace[-1].relevant_vertices.values()))
        learning_statistics.add("num_features_in_pool (|F|)", len(trace[-1].feature_pool))
        learning_statistics.print(title="Learning statistics:", logger=True)

        self._timers.add("memory/total", memory_usage() / 1024)
        self._timers.print(logger=True)
        print_separation_line(logger=True)

        # If no sketch found, terminate; else, write sketch to output folder
        if sketch is None:
            logging.info(f"FAILURE (no sketch learned)")
            print_separation_line(logger=True)
        else:
            logging.info(f"Writing output to '{self._folder_name_for_output}'")
            with change_dir(self._folder_name_for_output):
                logging.info("Resulting sketch:")
                sketch.print(logger=True)
                print_separation_line(logger=True)

                write_file(f"sketch_{self._width}.txt", str(sketch._dlplan_policy))
                write_file(f"sketch_minimized_{self._width}.txt", str(sketch._dlplan_policy))

                logging.info(f"SUCCESS {self._timers.get_elapsed_sec('feature/pool'):.02f} feature-pool {self._timers.get_elapsed_sec('preprocessing'):.02f} preprocessing {self._timers.get_elapsed_sec('asp'):.02f} ASP {self._timers.get_elapsed_sec('total'):.02f} total")
                print_separation_line(logger=True)


