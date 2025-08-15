import logging, sys
import uuid
from termcolor import colored
from typing import Set, List, Tuple, MutableSet, Dict, Optional, Any

import math, random
import numpy as np
from itertools import product
from collections import defaultdict
from pathlib import Path

from dlplan.policy import PolicyMinimizer
import dlplan.core as dlplan_core

#from .src.preprocessing import compute_domain_data

from .src.exit_codes import ExitCode
from .src.iteration import TerminationBasedLearnerReduced
from .src.iteration import generate_features, post_process_features, read_features_from_repositories, write_features_to_repository, find_feature_repositories, compute_feature_valuations_for_dlplan_state
from .src.iteration import EncodingType, IterationData, PlainStatistics, Statistics, SketchReduced, D2sepDlplanPolicyFactory
from .src.util import Timer, create_experiment_workspace, change_working_directory, write_file, write_file_lines, change_dir, memory_usage, add_console_handler, print_separation_line
from .src.preprocessing import InstanceData, PreprocessingData, StateFinder, compute_instance_datas, compute_tuple_graphs

from .src.state_space import StateFactory, PDDLInstance

def _get_mimir_state_repr(instance_data: InstanceData, state: Any, include_static_atoms: bool = False) -> str:
    assert False, f"{Path(__file__).name}"
    problem = instance_data.mimir_ss.get_problem()
    pddl_factories = instance_data.mimir_ss.get_pddl_factories()
    static_literals: List[str] = [str(literal) for literal in problem.get_static_initial_literals()]
    fluent_atoms: List[str] = [str(atom) for atom in pddl_factories.get_fluent_ground_atoms_from_indices(state.get_fluent_atoms())]
    derived_atoms: List[str] = [str(atom) for atom in pddl_factories.get_derived_ground_atoms_from_indices(state.get_derived_atoms())]
    atoms: List[str] = static_literals + fluent_atoms + derived_atoms if include_static_atoms else fluent_atoms + derived_atoms
    state_repr: str = f"[{', '.join(atoms)}]"
    return state_repr

def _compute_unsolved_instances(iteration_data: IterationData,
                                instance_datas: List[PDDLInstance],
                                sketch: SketchReduced,
                                stop_at_first_unsolved_instance: bool = False,
                                **kwargs) -> List[Tuple[PDDLInstance, Any]]:
    unsolved_instances: List[Tuple[PDDLInstance, Any]] = []
    for instance_data in instance_datas:
        solves, reason = sketch.solves(None, iteration_data, instance_data, **kwargs)
        assert not solves or reason is None
        if not solves:
            unsolved_instances.append((instance_data, reason))
            if stop_at_first_unsolved_instance: break
    return unsolved_instances

def _initial_selected_idxs(available_instance_idxs: List[int],
                           instance_selection: Optional[str],
                           first_instance: Optional[int]) -> List[int]:
    if first_instance is not None and first_instance in available_instance_idxs:
        return [first_instance]
    elif instance_selection is None or instance_selection in ["forward", "forward+"]:
        return [min(available_instance_idxs)]
    elif instance_selection in ["backward", "backward+"]:
        return [max(available_instance_idxs)]
    elif instance_selection in ["random", "random+"]:
        return [random.choice(available_instance_idxs)]
    elif instance_selection == "test":
        pass
    raise RuntimeError(f"ERROR: unexpected instance selection stragegy '{instance_selection}'")

def _advance_selected_idxs(selected_idxs: List[int],
                           previous_selected_idxs: List[int],
                           unsolved_instances: List[Tuple[PDDLInstance, Any]],
                           iteration_data: IterationData,
                           available_instance_idxs: List[int],
                           instance_selection: Optional[str]) -> List[int]:

    unsolved_instance_idxs: List[int] = [instance_data.idx for instance_data, _ in unsolved_instances]
    seen_instance_idxs: List[int] = list(set(selected_idxs + previous_selected_idxs))
    max_seen_instance_idx = max(seen_instance_idxs)

    next_instance_idx: int = None
    if instance_selection is None or instance_selection == "forward":
        choices_from_unsolved: List[int] = [instance_idx for instance_idx in unsolved_instance_idxs if instance_idx > max_seen_instance_idx]
        choices_from_available: List[int] = [instance_idx for instance_idx in available_instance_idxs if instance_idx > max_seen_instance_idx]
        if len(choices_from_unsolved) > 0:
            next_instance_idx = min(choices_from_unsolved)
        elif len(choices_from_available) > 0:
            next_instance_idx = min(choices_from_available)

    elif instance_selection == "forward+":
        choices_from_unsolved_0: List[int] = [instance_idx for instance_idx in unsolved_instance_idxs if instance_idx < max_seen_instance_idx]
        choices_from_unsolved_1: List[int] = [instance_idx for instance_idx in unsolved_instance_idxs if instance_idx > max_seen_instance_idx]
        choices_from_available: List[int] = [instance_idx for instance_idx in available_instance_idxs if instance_idx > max_seen_instance_idx]
        if len(choices_from_unsolved_0) > 0:
            next_instance_idx = max(choices_from_unsolved_0)
        elif len(choices_from_unsolved_1) > 0:
            next_instance_idx = min(choices_from_unsolved_1)
        elif len(choices_from_available) > 0:
            next_instance_idx = min(choices_from_available)

    elif instance_selection in ["backward", "backward+"]:
        choices: List[int] = unsolved_instance_idxs
        next_instance_idx = max(choices) if len(choices) > 0 else None

    elif instance_selection in ["random", "random+"]:
        choices: List[int] = [instance_data.idx for instance_data, _ in unsolved_instances]
        next_instance_idx = random.choice(choices) if len(choices) > 0 else None

    else:
        raise RuntimeError(f"ERROR: unexpected instance selection stragegy '{instance_selection}'")

    if next_instance_idx is None:
        logging.info(colored(f"** NO MORE TRAINING INSTANCES", "red"))
        raise RuntimeError("ERROR: ** NO MORE TRAINING INSTANCES")

    if instance_selection is None or not instance_selection.endswith("+"):
        next_instance_idxs = [next_instance_idx]
    elif next_instance_idx > max_seen_instance_idx:
        next_instance_idxs = [next_instance_idx]
    else:
        assert next_instance_idx not in selected_idxs
        next_instance_idxs = selected_idxs + [next_instance_idx]

    logging.info(f"ADVANCING to TRAINING instances: {next_instance_idxs}")
    return next_instance_idxs


def reduced_termination_based_learn_sketch_for_problem_class(
    domain_filepath: Path,
    problems_directory: Path,
    max_num_instances: int,
    workspace: Path,
    width: int,
    disable_closed_Q: bool = False,
    randomized_sketch_test: bool = False,
    disable_feature_generation: bool = False,
    generate_all_distance_features: bool = False,
    enable_incomplete_feature_pruning: bool = False,
    enable_pruning_features_always_positive: bool = False,
    enable_pruning_features_large_decrease: bool = False,
    concept_complexity_limit: int = 9,
    role_complexity_limit: int = 9,
    boolean_complexity_limit: int = 9,
    count_numerical_complexity_limit: int = 9,
    distance_numerical_complexity_limit: int = 9,
    feature_limit: int = 1000000,
    strict_gc2_features: bool = False,
    additional_booleans: List[str] = None,
    additional_numericals: List[str] = None,
    disable_feature_repositories: bool = False,
    store_features: bool = False,
    flexible_repositories: bool = False,
    all_repositories: bool = False,
    enable_dump_files: bool = False,
    instance_selection: Optional[str] = None,
    first_instance: Optional[int] = None,
    planner: Optional[str] = None,
    max_feature_depth: Optional[int] = None,
    analyze_features: Optional[str] = None,
    timeout_in_seconds_per_step: Optional[float] = None,
    timeout_in_seconds: Optional[float] = None,
    disable_greedy_solver: Optional[bool] = None,
    disable_optimization_decorations: Optional[bool] = None,
    deadends: Optional[bool] = None,
    solver_prefix: Optional[str] = None,
    simplify_policy: bool = False,
    simplify_only_conditions: bool = False,
    separate_siblings: bool = False,
    contextual: bool = False,
    monotone_only_by_dec: bool = False,
    uniform_costs: bool = False,
    verbose: bool = False,
    dump_asp_program: bool = False,
    preprocess_only: bool = False,
    features_only: bool = False,
    **kwargs):

    # Setup arguments and workspace
    if additional_booleans is None:
        additional_booleans = []
    if additional_numericals is None:
        additional_numericals = []
    instance_filepaths = list(problems_directory.iterdir())
    create_experiment_workspace(workspace)
    change_working_directory(workspace)

    # Create UUID for avoiding clashes with other processes
    uuid_str: str = uuid.uuid4().hex
    iterations_folder_name: str = f"iterations.{uuid_str}"
    output_folder_name: str = f"output.{uuid_str}"
    feature_repository_name: str = f"repo_{uuid_str}.frepo"
    logging_file_name: str = f"logging.txt"

    # Setup logger
    logger = logging.getLogger()
    logger_level = logging.INFO
    logger.setLevel(logger_level)
    logger_formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')

    logger_handler_stdout = logging.StreamHandler(sys.stdout)
    logger_handler_stdout.setFormatter(logger_formatter)
    #logger_handler.terminator = "" # Change end-of-output terminator from "\n" to ""
    logger.addHandler(logger_handler_stdout)

    try:
        from os import makedirs
        makedirs(iterations_folder_name)
    except FileExistsError:
        logging.error("Iterations folder {iterations_folder_name} exists! Cannot continue...")
        raise

    logger_handler_file = logging.FileHandler(Path(iterations_folder_name) / logging_file_name)
    logger_handler_file.setFormatter(logger_formatter)
    logger.addHandler(logger_handler_file)

    # First logging entries
    logging.info(f"Call: python {' '.join(sys.argv)}")
    logging.info(colored(f"UUID: {uuid_str}", "green"))

    # Keep track of time
    timers: Statistics = Statistics()
    timers.add_timers(["total", "preprocessing", "feature/pool", "feature/termination", "learner", "verification", "asp", "planner", "m_pairs", "indexing", "pricing/preprocessing", "pricing/algorithm"])
    timers_formatting: Dict[str, Any] = {
        #"title": "Title:",
        "subtitle1": "Memory statistics:",
        "subtitle2": "Time statistics:",
        "prefixes": {
            "memory/total": "Total memory",
            "timer/total": "Total time",
            "timer/preprocessing": "Preprocessing time",
            "timer/planner": "Planner time",
            "timer/asp": "ASP time",
            "timer/verification": "Verification time",
            "timer/feature/pool": "Feature pool generation time",
            "timer/feature/termination": "Preprocessing of feature termination time",
            "timer/learner": "Learner time",
            "timer/m_pairs": "MPairs time",
            "timer/indexing": "Indexing time",
            "timer/pricing/preprocessing": "Pricing/preprocessing time",
            "timer/pricing/algorithm": "Pricing/algorithm time",
        },
        "suffixes": {
            "memory/total": "GiB.",
            "timer/planner": "seconds (accounted for in preprocessing time).",
            "timer/feature/termination": "seconds (includes MPairs time).",
        },
        "formatters": {
            "memory/total": "0.2f",
        },
    }
    timers.register_formatting(**timers_formatting)
    timers.resume("total")

    # Obtain family name from domain path
    family_name: str = str(domain_filepath.parent.name)

    # Initialize state factory
    logging.info(f"Read and curate training instances...")
    state_factory: StateFactory = StateFactory(family_name, domain_filepath, instance_filepaths, planner, deadends)
    domain_data = state_factory.domain_data()

    # Create termination-based learner and do preprocessing
    learner_options = {
        "state_factory": state_factory,
        "disable_greedy_solver": disable_greedy_solver,
        "disable_optimization_decorations": disable_optimization_decorations,
        "solver_prefix": solver_prefix,
        "timers": timers,
        "preprocessing_timer": timers.get_timer("preprocessing"),
        "planner_timer": timers.get_timer("planner"),
        "asp_timer": timers.get_timer('asp'),
    }
    learner: TerminationBasedLearnerReduced = TerminationBasedLearnerReduced(**learner_options)
    idxs_to_be_removed: Set[int] = set(learner.preprocess_instances(planner))
    if preprocess_only:
        logging.info(f"{len(learner._instance_datas)} instance(s) preprocessed, {len(idxs_to_be_removed)} instance(s) to be removed. Done!")
        return

    if len(idxs_to_be_removed) > 0:
        logging.info(f"{len(idxs_to_be_removed)} instance(s) to be removed:")
        for instance_idx in idxs_to_be_removed:
            logging.info(f"  {instance_idx}.[{state_factory.get_instance(instance_idx).instance_filepath()}]")

    idxs_to_be_kept: List[int] = [instance_idx for instance_idx in range(len(state_factory._instances)) if instance_idx not in idxs_to_be_removed]
    sorted_idxs_to_be_kept: List[int] = sorted(idxs_to_be_kept, key=lambda idx: len(learner._preprocessed_datas[idx].get("planner_output")[1]), reverse=True)
    sorted_idxs_to_be_kept = sorted_idxs_to_be_kept if max_num_instances is None else sorted_idxs_to_be_kept[:max_num_instances]

    # Additionally, prune instances whose plan revisits a state. For this, plan's state trajectory must be generated
    non_loopy_idxs_to_be_kept: List[int] = [instance_idx for instance_idx in sorted_idxs_to_be_kept if all(learner.non_loopy_state_trajectories(instance_idx))]
    if len(non_loopy_idxs_to_be_kept) < len(sorted_idxs_to_be_kept):
        logging.info(f"Further removal of {len(sorted_idxs_to_be_kept) - len(non_loopy_idxs_to_be_kept)} loopy instance(s):")
        for instance_idx in sorted_idxs_to_be_kept:
            if instance_idx not in non_loopy_idxs_to_be_kept:
                logging.info(f"  {instance_idx}.[{state_factory.get_instance(instance_idx).instance_filepath()}]")

    # Remove and re-order instances
    previous_number_instances: int = len(learner._instance_datas)
    learner.remove_and_reorder_instances(non_loopy_idxs_to_be_kept)
    assert len(non_loopy_idxs_to_be_kept) == len(learner._instance_datas), (len(non_loopy_idxs_to_be_kept), len(learner._instance_datas))
    logging.info(f"{len(non_loopy_idxs_to_be_kept)} instance(s) after removal and re-ordering; {previous_number_instances - len(non_loopy_idxs_to_be_kept)} instance(s) removed")
    learner.preprocess_instances_final()

    if len(state_factory._instances) == 0:
        logging.info(f"Zero instances remain. Terminating...")
        timers.print(title="Finalizing...", logger=True)
        raise RuntimeError("Zero instances remain. Terminating...")

    # Define instance collections
    instance_datas: List[PDDLInstance] = list(state_factory._instances)
    instance_idxs: List[int] = list(range(len(instance_datas)))
    instance_idx_to_denotations_caches: Dict[int, dlplan_core.DenotationsCaches] = {instance_idx: dlplan_core.DenotationsCaches() for instance_idx in instance_idxs}

    # Initial set of relevant vertices
    initial_relevant_vertices: Dict[int, Set[int]] = dict()
    for preprocessed_data in learner._preprocessed_datas:
        assert not any(preprocessed_data.get("revisits_states"))
        instance_idx: int = preprocessed_data.get("instance_idx")
        marked_state_trajectories: List[Tuple[int]] = preprocessed_data.get("marked_state_trajectories")
        assert len(marked_state_trajectories) == 1, f"More than one marked path per instance isn't supported yet (instance_idx={instance_idx})"
        assert instance_idx not in initial_relevant_vertices
        initial_relevant_vertices[instance_idx] = set(marked_state_trajectories[0])
    logging.info(f"{sum([len(vertices) for vertices in initial_relevant_vertices.values()])} initial relevant vertice(s)")

    timers.resume("preprocessing")

    # Dlplan states
    dlplan_states: List[dlplan_core.State] = []
    ext_state_to_dlplan_state_index: Dict[Tuple[int, int], int] = dict()
    ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = dict()
    for instance_idx, vertices in initial_relevant_vertices.items():
        instance_data: PDDLInstace = state_factory.get_instance(instance_idx)
        for state_idx in vertices:
            ext_state: Tuple[int, int] = (instance_idx, state_idx)
            if ext_state not in ext_state_to_dlplan_state_index:
                dlplan_state_index: int = len(dlplan_states)
                ext_state_to_dlplan_state_index[ext_state] = dlplan_state_index
                state: Any = state_factory.get_state(*ext_state)
                dlplan_state: dlplan_core.State = instance_data.get_dlplan_state(state_idx, state)
                dlplan_states.append(dlplan_state)
                logging.debug(f"EXT_STATE_TO_DLPLAN_INDEX: {ext_state} -> {dlplan_state_index} -> {learner._ext_state_to_global_state_index.get(ext_state)} : {dlplan_state}")

    timers.stop("preprocessing")
    timers.resume("feature/pool")

    # Features
    feature_pool_options: Dict[str, Any] = {
        "disable_feature_generation": disable_feature_generation,
        "generate_all_distance_features": generate_all_distance_features,
        "concept_complexity_limit": concept_complexity_limit,
        "role_complexity_limit": role_complexity_limit,
        "boolean_complexity_limit": boolean_complexity_limit,
        "count_numerical_complexity_limit": count_numerical_complexity_limit,
        "distance_numerical_complexity_limit": distance_numerical_complexity_limit,
        "feature_limit": feature_limit,
        "strict_gc2_features": strict_gc2_features,
        "additional_booleans": additional_booleans,
        "additional_numericals": additional_numericals,
        "max_feature_depth": max_feature_depth,
        "analyze_features": analyze_features,
    }

    # Revise if there is a feature repository that match active instances and feature parameters.
    # If so, read the features in the repo instead of generating them. Else, generate the features
    # and store them.

    feature_parameters: Dict[str, Any] = dict(feature_pool_options)
    feature_parameters.update({"planner": planner})
    feature_parameters.pop("analyze_features")
    feature_parameters.pop("additional_booleans")
    feature_parameters.pop("additional_numericals")
    feature_parameters.pop("max_feature_depth")

    instance_names: List[str] = sorted([instance_data.instance_filepath().name for instance_data in learner._instance_datas])
    feature_repository_folder: Path = Path("feature_repositories")
    feature_repositories: List[Path] = find_feature_repositories(feature_repository_folder, feature_parameters, instance_names, all_repositories=all_repositories, flexible=flexible_repositories) if not disable_feature_repositories else None

    if feature_repositories is not None and len(feature_repositories) > 0:
        logging.info(colored(f"Found compatible feature repositories [{', '.join([feature_repository.name for feature_repository in feature_repositories])}]", "blue"))
        feature_pool, feature_statistics = read_features_from_repositories(feature_repositories,
                                                                           domain_data.syntactic_element_factory,
                                                                           **feature_pool_options)
    else:
        logging.info(colored("Generating features...", "blue"))
        feature_pool, _, feature_statistics = generate_features(domain_data.syntactic_element_factory,
                                                                dlplan_states,
                                                                instance_idx_to_denotations_caches,
                                                                **feature_pool_options)
    logging.info(f"Feature statistics: {feature_statistics}")

    if store_features and (feature_repositories is None or len(feature_repositories) == 0):
        feature_statistics.update({"uuid": uuid_str, "family": family_name})
        feature_repository: Path = feature_repository_folder / feature_repository_name
        write_features_to_repository(feature_pool, feature_parameters, feature_statistics, instance_names, feature_repository)
        logging.info(colored(f"{len(feature_pool)} feature(s) written to '{feature_repository}'", "blue"))

    # Post-processing of features (e.g., pruned by max depth)
    logging.info(colored("Post-processing features...", "blue"))
    feature_pool = post_process_features(feature_pool, **feature_pool_options)

    timers.stop("feature/pool")
    logging.info(colored(f"Got {len(feature_pool)} Feature(s) in {timers.get_elapsed_sec('feature/pool'):.02f} second(s)", "blue"))

    timers.stop("preprocessing")
    if features_only:
        timers.print(title="Finalizing because option '--features_only'", logger=True)
        return

    # Learn sketch
    with change_dir(iterations_folder_name):
        previous_selected_idxs: List[int] = []
        try:
            selected_idxs: List[int] = _initial_selected_idxs(instance_idxs, instance_selection, first_instance)
        except RuntimeError as rt:
            logging.info(str(rt))
            timers.print(title="Finalizing due to RuntimeError exception...", logger=True)
            raise
        logging.info(f"ADVANCING to TRAINING instances: {selected_idxs}")

        i = 0
        total_inner_iterations = 0
        while True:
            logging.info(colored(f"Iteration: {i}", "red"))
            with change_dir(str(i), enable=enable_dump_files or dump_asp_program):
                iteration_instance_datas: List[PDDLInstance] = [instance_datas[instance_idx] for instance_idx in selected_idxs]
                for instance_data in iteration_instance_datas:
                    logging.info(f"    id: {instance_data.idx}.[{instance_data.instance_filepath()}]")

                # Consider all bundles: product of paths in each instance in iteration
                good_sketch = False
                iteration_instance_idxs: List[int] = [instance_data.idx for instance_data in iteration_instance_datas]
                iteration_preprocessed_data: List[Dict[str, Any]] = [learner._preprocessed_datas[instance_idx] for instance_idx in iteration_instance_idxs]
                unsolved_instances_for_bundles: Dict[int, List[PDDLInstance, Any]] = dict()
                num_bundles = math.prod([len(preprocessed_data.get("marked_state_trajectories")) for preprocessed_data in iteration_preprocessed_data])
                assert num_bundles == 1
                for bundle_index, bundle in enumerate(product(*[preprocessed_data.get("marked_state_trajectories") for preprocessed_data in iteration_preprocessed_data])):
                    bundle_with_idxs: List[Tuple[int, Tuple[int]]] = list(zip(iteration_instance_idxs, bundle))

                    # Iteration data contains initial set of relevant vertices, and related info
                    iteration_data = IterationData()
                    iteration_data.feature_pool: List[Feature] = feature_pool
                    iteration_data.instance_datas = iteration_instance_datas
                    iteration_data.paths_with_idxs: List[Tuple[int, Tuple[int]]] = bundle_with_idxs
                    iteration_data.relevant_vertices: Dict[int, Set[int]] = {instance_idx: set(initial_relevant_vertices.get(instance_idx)) for instance_idx in iteration_instance_idxs}
                    iteration_data.vertices_in_bundle: Dict[int, Set[int]] = {instance_idx: set(vertices) for instance_idx, vertices in iteration_data.relevant_vertices.items()}

                    iteration_data.ext_state_to_dlplan_state_index: Dict[Tuple[int, int], int] = ext_state_to_dlplan_state_index
                    iteration_data.ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = ext_state_to_feature_valuations
                    iteration_data.instance_idx_to_denotations_caches: Dict[int, dlplan_core.DenotationsCaches] = instance_idx_to_denotations_caches

                    iteration_data.non_covered_vertices: Dict[int, Set[int]] = defaultdict(set)
                    iteration_data.vertices_in_deadend_paths: Dict[int, Set[int]] = defaultdict(set)
                    iteration_data.deadend_paths: List[Tuple[Tuple[int, int]]] = []

                    for instance_idx, path in bundle_with_idxs:
                        relevant_vertices_for_instance_idx: Set[int] = set(iteration_data.relevant_vertices.get(instance_idx))
                        assert set(path).issubset(relevant_vertices_for_instance_idx)

                    learner.prepare_iteration(iteration_data)

                    # Inner loop, at each iteration one or more relevant vertices are added
                    inner_iterations = 0
                    while True:
                        logging.info(f"New Inner ITERATION {inner_iterations} (total={total_inner_iterations}): Memory usage={memory_usage() / 1024:0.2f} GiB")
                        inner_iterations += 1
                        total_inner_iterations += 1
                        timers.resume("preprocessing")

                        # Relevant vertices must be those in example paths, deadend paths, and non-covered vertices
                        for instance_idx, relevant_vertices in iteration_data.relevant_vertices.items():
                            vertices_in_paths: Set[int] = set().union(*[path for idx, path in bundle_with_idxs if idx == instance_idx])
                            assert vertices_in_paths.issubset(relevant_vertices)
                            assert iteration_data.vertices_in_deadend_paths.get(instance_idx, set()).issubset(relevant_vertices)
                            assert iteration_data.non_covered_vertices.get(instance_idx, set()).issubset(relevant_vertices)

                        #logging.info(f"New Inner ITERATION 1b: Memory usage={memory_usage() / 1024:0.2f} GiB")

                        # Setup information for relevant vertices and off-path neighbors
                        for instance_idx, relevant_vertices in iteration_data.relevant_vertices.items():
                            logging.info(f"ITERATION: instance_idx={instance_idx}, #relevant={len(relevant_vertices)}")
                            instance_data: PDDLInstance = instance_datas[instance_idx]
                            assert instance_data.idx == instance_idx

                            # Setup information for relevant vertices
                            for state_idx in relevant_vertices:
                                ext_state: Tuple[int, int] = (instance_idx, state_idx)
                                if ext_state not in ext_state_to_dlplan_state_index:
                                    dlplan_state_index: int = len(dlplan_states)
                                    ext_state_to_dlplan_state_index[ext_state] = dlplan_state_index
                                    state: Any = state_factory.get_state(*ext_state)
                                    dlplan_state: dlplan_core.State = instance_data.get_dlplan_state(state_idx, state)
                                    dlplan_states.append(dlplan_state)
                                    logging.debug(f"EXT_STATE_TO_DLPLAN_INDEX: {ext_state} -> {dlplan_state_index} -> {learner._ext_state_to_global_state_index.get(ext_state)} : {dlplan_state}")
                                if ext_state not in ext_state_to_feature_valuations:
                                    dlplan_state_index: int = ext_state_to_dlplan_state_index.get(ext_state)
                                    dlplan_state: dlplan_core.State = dlplan_states[dlplan_state_index]
                                    ext_state_to_feature_valuations[ext_state] = compute_feature_valuations_for_dlplan_state(dlplan_state, feature_pool, instance_idx_to_denotations_caches)

                            # Setup information for successors of non-goal relevant vertices
                            for state_idx in [state_idx for state_idx in relevant_vertices if not instance_data.is_goal_state(state_idx)]:
                                ext_state: Tuple[int, int] = (instance_idx, state_idx)
                                successors: List[Tuple[Tuple[int, Any], str]] = instance_data.get_successors(state_idx)
                                #logging.info(f"ITERATION: ext_state={ext_state}, #SUCC={len(instance_data.get_successors(state_idx))}")
                                for (succ_state_idx, succ_state), operator in successors:
                                    succ_ext_state: Tuple[int, int] = (instance_idx, succ_state_idx)
                                    if succ_ext_state not in ext_state_to_dlplan_state_index:
                                        succ_dlplan_state_index: int = len(dlplan_states)
                                        ext_state_to_dlplan_state_index[succ_ext_state] = succ_dlplan_state_index
                                        #succ_state: Any = state_factory.get_state(*succ_ext_state)
                                        assert succ_state == state_factory.get_state(*succ_ext_state)
                                        succ_dlplan_state: dlplan_core.State = instance_data.get_dlplan_state(succ_state_idx, succ_state)
                                        dlplan_states.append(succ_dlplan_state)
                                        logging.debug(f"EXT_STATE_TO_DLPLAN_INDEX: {succ_ext_state} -> {succ_dlplan_state_index} -> {learner._ext_state_to_global_state_index.get(succ_ext_state)} : {succ_dlplan_state}")
                                    if succ_ext_state not in ext_state_to_feature_valuations:
                                        succ_dlplan_state_index: int = ext_state_to_dlplan_state_index.get(succ_ext_state)
                                        succ_dlplan_state: dlplan_core.State = dlplan_states[succ_dlplan_state_index]
                                        ext_state_to_feature_valuations[succ_ext_state] = compute_feature_valuations_for_dlplan_state(succ_dlplan_state, feature_pool, instance_idx_to_denotations_caches)

                            """
                            # Setup information for relevant vertices
                            for state_idx in relevant_vertices:
                                ext_state: Tuple[int, int] = (instance_idx, state_idx)
                                if ext_state not in ext_state_to_dlplan_state_index:
                                    dlplan_state_index: int = len(dlplan_states)
                                    ext_state_to_dlplan_state_index[ext_state] = dlplan_state_index
                                    state: Any = state_factory.get_state(*ext_state)
                                    dlplan_state: dlplan_core.State = instance_data.get_dlplan_state(state_idx, state)
                                    dlplan_states.append(dlplan_state)
                                    logging.debug(f"EXT_STATE_TO_DLPLAN_INDEX: {ext_state} -> {dlplan_state_index} -> {learner._ext_state_to_global_state_index.get(ext_state)} : {dlplan_state}")
                                if ext_state not in ext_state_to_feature_valuations:
                                    dlplan_state_index: int = ext_state_to_dlplan_state_index.get(ext_state)
                                    dlplan_state: dlplan_core.State = dlplan_states[dlplan_state_index]
                                    ext_state_to_feature_valuations[ext_state] = compute_feature_valuations_for_dlplan_state(dlplan_state, feature_pool, instance_idx_to_denotations_caches)

                            # Setup information for neighbors of off-path relevant vertices
                            for off_path_state_idx in [vertex for vertex in relevant_vertices - iteration_data.vertices_in_bundle.get(instance_idx) if not instance_data.is_goal_state(vertex)]:
                                logging.info(f"ITERATION: off_path_state_idx={off_path_state_idx}, #SUCC={len(instance_data.get_successors(off_path_state_idx))}")
                                for (succ_state_idx, succ_state), operator in instance_data.get_successors(off_path_state_idx):
                                    succ_ext_state: Tuple[int, int] = (instance_idx, succ_state_idx)
                                    if succ_ext_state not in ext_state_to_dlplan_state_index:
                                        succ_dlplan_state_index: int = len(dlplan_states)
                                        ext_state_to_dlplan_state_index[succ_ext_state] = succ_dlplan_state_index
                                        succ_state: Any = state_factory.get_state(*succ_ext_state)
                                        succ_dlplan_state: dlplan_core.State = instance_data.get_dlplan_state(succ_state_idx, succ_state)
                                        dlplan_states.append(succ_dlplan_state)
                                        logging.debug(f"EXT_STATE_TO_DLPLAN_INDEX: {succ_ext_state} -> {succ_dlplan_state_index} -> {learner._ext_state_to_global_state_index.get(succ_ext_state)} : {succ_dlplan_state}")
                                    if succ_ext_state not in ext_state_to_feature_valuations:
                                        succ_dlplan_state_index: int = ext_state_to_dlplan_state_index.get(succ_ext_state)
                                        succ_dlplan_state: dlplan_core.State = dlplan_states[succ_dlplan_state_index]
                                        ext_state_to_feature_valuations[succ_ext_state] = compute_feature_valuations_for_dlplan_state(succ_dlplan_state, feature_pool, instance_idx_to_denotations_caches)
                            """

                        #logging.info(f"New Inner ITERATION 2: Memory usage={memory_usage() / 1024:0.2f} GiB")
                        timers.stop("preprocessing")
                        timers.resume("learner")

                        # Prepare learner for this iteration
                        num_relevant_vertices = sum(map(lambda item: len(item), iteration_data.relevant_vertices.values()))
                        logging.info(f"Bundle: index={bundle_index}/{num_bundles-1}, bundle={bundle_index}.{bundle_with_idxs}, #relevant_vertices={num_relevant_vertices}, relevant_vertices={iteration_data.relevant_vertices}")

                        # Solve ASP programs
                        try:
                            solve_options: Dict[str, Any] = {
                                "timeout_in_seconds_per_step": timeout_in_seconds_per_step,
                                "timeout_in_seconds": timeout_in_seconds,
                                "simplify_policy": simplify_policy,
                                "simplify_only_conditions": simplify_only_conditions,
                                "separate_siblings": separate_siblings,
                                "contextual": contextual,
                                "monotone_only_by_dec": monotone_only_by_dec,
                                "uniform_costs": uniform_costs,
                                "verbose": verbose,
                                "dump_asp_program": dump_asp_program,
                                "novel": True,
                            }
                            status, feature_idxs, rules_with_decorations, _ = learner.solve(**solve_options)
                        except RuntimeError as rt:
                            logging.info(str(rt))
                            timers.print(title="Finalizing due to RuntimeError exception...", logger=True)
                            raise
                        #logging.info(f"New Inner ITERATION 3: Memory usage={memory_usage() / 1024:0.2f} GiB")

                        # If there is a solution, verify whether it works for the entire training set
                        if status:
                            # Create sketch from solution of LP
                            policy_builder: PolicyFactory = state_factory.domain_data().policy_builder
                            dlplan_policy = D2sepDlplanPolicyFactory().make_dlplan_policy_from_rules_with_decorations(feature_idxs, rules_with_decorations, policy_builder, iteration_data)
                            sketch = SketchReduced(dlplan_policy, width)
                            logging.info("Learned the following sketch:")
                            sketch.print(logger=True)

                            # Minimize sketch
                            if False:
                                sketch_minimized = sketch.minimize(domain_data.policy_builder)
                                logging.info("Minimized learned sketch:")
                                sketch_minimized.print(logger=True)
                            else:
                                sketch_minimized = sketch

                            # Analyze learned sketch
                            timers.resume("verification")
                            logging.info(colored(f"Verifying learned sketch on TRAINING instances {selected_idxs}...", "blue"))
                            solves_options: Dict[str, Any] = {
                                "test_goal_separating_features": False,
                                "randomized_sketch_test": randomized_sketch_test,
                                "max_non_covered_ext_states": 1,
                                "state_factory": state_factory,
                            }
                            unsolved_instances: List[Tuple[PDDLInstance, Any]] = _compute_unsolved_instances(iteration_data, iteration_data.instance_datas, sketch_minimized, debug=False, **solves_options)
                            timers.stop("verification")

                            if len(unsolved_instances) == 0:
                                logging.info(f"Sketch SOLVES TRAINING instances: {selected_idxs}")

                                timers.resume("verification")
                                logging.info(colored("Verifying learned sketch on ALL instances...", "blue"))
                                unsolved_instances: List[Tuple[PDDLInstance, Any]] = _compute_unsolved_instances(iteration_data, instance_datas, sketch_minimized, stop_at_first_unsolved_instance=True, **solves_options)
                                timers.stop("verification")

                                if len(unsolved_instances) == 0:
                                    logging.info(colored(f"Sketch SOLVES ALL {len(instance_datas)} instance(s)! [Bundle: {bundle_index}.{bundle_with_idxs}]", "blue"))
                                    good_sketch = True
                                else:
                                    # Sketch doesn't solve an instance different from the one used for learning
                                    logging.info(f"Sketch doesn't solve an instance different from the one used for learning (idxs={[instance_idx for instance_idx, _ in bundle_with_idxs]}):")
                                    for instance_data, reason in unsolved_instances:
                                        logging.info(f"  {instance_data.instance_filepath()} (idx={instance_data.idx})")
                                        logging.info(f"  Reason: {reason}")
                                    unsolved_instances_for_bundles[bundle_index] = unsolved_instances
                                #logging.info(f"New Inner ITERATION 4: Memory usage={memory_usage() / 1024:0.2f} GiB")
                                break
                            else:
                                # Sketch doesn't solve the instance used for learning
                                old_len = sum(map(lambda states: len(states), iteration_data.relevant_vertices.values()))
                                logging.info(f"Sketch doesn't solve an instance used for learning (idxs={[instance_idx for instance_idx, _ in bundle_with_idxs]}):")
                                for instance_data, reason in unsolved_instances:
                                    logging.info(f"  {instance_data.instance_filepath()} (idx={instance_data.idx})")
                                    logging.info(f"  Reason: {reason}")

                                    if "non-covered" in reason:
                                        for instance_idx, state_idx in [ext_state for ext_state in reason.get("non-covered", [])]:
                                            assert state_idx not in iteration_data.non_covered_vertices.get(instance_idx, [])
                                            iteration_data.non_covered_vertices[instance_idx].add(state_idx)
                                            iteration_data.relevant_vertices[instance_idx].add(state_idx)
                                            logging.info(f"Adding ext_state {(instance_idx, state_idx)} to non-covered vertices")

                                    elif "deadend" in reason:
                                        if not deadends: logging.warning(f"WARNING: DEADEND reached even though 'deadends=False'")
                                        deadend_path: Tuple[Tuple[int, int]] = reason.get("deadend")
                                        logging.info(f"{len(iteration_data.deadend_paths)} deadend path(s); paths={iteration_data.deadend_paths}")
                                        assert deadend_path is not None and deadend_path not in iteration_data.deadend_paths
                                        iteration_data.deadend_paths.append(deadend_path)

                                        instance_idx = deadend_path[0][0]
                                        vertices_in_deadend_path: Set[int] = set([state_idx for _, state_idx in deadend_path])
                                        iteration_data.vertices_in_deadend_paths[instance_idx] |= vertices_in_deadend_path
                                        iteration_data.relevant_vertices[instance_idx] |= vertices_in_deadend_path
                                        logging.info(f"Adding {vertices_in_deadend_path} to relevant_vertices")
                        else:
                            logging.info(f"UNSAT LP for bundle: {bundle_index}.{bundle_with_idxs}")
                            unsolved_instances: List[Tuple[PDDLInstance, Any]] = []
                            break

                    #logging.info(f"New Inner ITERATION 5: Memory usage={memory_usage() / 1024:0.2f} GiB")
                    logging.info(f"Inner iterations: {inner_iterations}")
                    logging.info(f"Total inner iterations: {total_inner_iterations}")

                    # If good sketch is found, stop iteration over bundles
                    if good_sketch: break

                timers.stop("learner")

                if good_sketch:
                    break
                else:
                    unsolved_instances = [pair for unsolved_instances_for_bundle in unsolved_instances_for_bundles.values() for pair in unsolved_instances_for_bundle]
                    new_previous_selected_idxs = previous_selected_idxs + selected_idxs
                    try:
                        new_selected_idxs = _advance_selected_idxs(selected_idxs, previous_selected_idxs, unsolved_instances, iteration_data, instance_idxs, instance_selection)
                    except RuntimeError as rt:
                        logging.info(str(rt))
                        timers.print(title="Finalizing due to RuntimeError exception...", logger=True)
                        raise
                    previous_selected_idxs = new_previous_selected_idxs
                    selected_idxs = new_selected_idxs
                    i += 1

    timers.stop("total")

    # Output the result
    with change_dir(output_folder_name):
        print_separation_line(logger=True)
        logging.info(colored("Summary:", "green"))

        learning_statistics = Statistics()
        learning_statistics.add("num_training_instances", len(instance_datas))
        learning_statistics.add("num_selected_training_instances (|P|)", len(iteration_data.instance_datas))
        learning_statistics.add("num_states_in_selected_training_instances (|S|)", sum(len(vertices) for vertices in iteration_data.relevant_vertices.values()))
        learning_statistics.add("num_features_in_pool (|F|)", len(iteration_data.feature_pool))
        learning_statistics.print(title="Learning statistics:", logger=True)

        timers.add("memory/total", memory_usage() / 1024)
        timers.print(logger=True)

        print_separation_line(logger=True)

        logging.info("Resulting sketch:")
        sketch.print(logger=True)
        print_separation_line(logger=True)

        # Minimize sketch
        if False:
            logging.info("Resulting minimized sketch:")
            sketch_minimized = sketch.minimize(domain_data.policy_builder)
            sketch_minimized.print(logger=True)
            print_separation_line(logger=True)
        else:
            sketch_minimized = sketch

        write_file(f"sketch_{width}.txt", str(sketch.dlplan_policy))
        write_file(f"sketch_minimized_{width}.txt", str(sketch_minimized.dlplan_policy))

        """
        # TO BE REINSTATED: Denotations of relevant vertices
        # Write denotations of selected features over states
        denotations = []
        sketch_features = []
        for feature in sketch_minimized.dlplan_policy.get_booleans():
            sketch_features.append(feature)
            denotations.append(f"Boolean {str(feature)}\n")

        for feature in sketch_minimized.dlplan_policy.get_numericals():
            sketch_features.append(feature)
            denotations.append(f"Numerical {str(feature)}\n")

        for instance_data in iteration_data.instance_datas:
            for gfa_state in instance_data.gfa.get_states():
                dlplan_state = preprocessing_data.state_finder.get_dlplan_ss_state(gfa_state)
                denotations.append(f"\n{str(dlplan_state)}\n")
                for feature in sketch_features:
                    key = feature.get_key()
                    index = int(key[1:])
                    f = iteration_data.feature_pool[index].dlplan_feature
                    assert str(feature.get_element()) == str(f), f"|{str(feature.get_element())}| != |{str(f)}|"
                    value = f.evaluate(dlplan_state)
                    denotations.append(f"  {key}={value}")
                denotations.append("\n")
                for rule in sketch_minimized.dlplan_policy.evaluate_conditions(dlplan_state):
                    denotations.append(f"  {rule}\n")

        write_file_lines(f"denotations.txt", denotations)
        """

        logging.info(f"SUCCESS {timers.get_elapsed_sec('feature/pool'):.02f} feature-pool {timers.get_elapsed_sec('preprocessing'):.02f} preprocessing {timers.get_elapsed_sec('asp'):.02f} ASP {timers.get_elapsed_sec('total'):.02f} total")
        print_separation_line(logger=True)
