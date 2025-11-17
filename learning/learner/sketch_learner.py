import logging, sys
import uuid
from termcolor import colored
from pathlib import Path
from typing import Set, List, Tuple, MutableSet, Dict, Optional, Any

from .benchmark import Benchmark
from .feature_pool import FeaturePool
from .wrapper import Wrapper, WrapperEnumeration, WrapperEnumerationV2

from .src.util import Timer
from .src.iteration import Statistics
from .src.iteration import TerminationBasedLearnerReduced
from .src.iteration.asp.termination_based_learner_reduced_2 import TerminationBasedLearnerReduced as TerminationBasedLearnerReduced2


def sketch_learner(
    # Benchmark required
    domain_filepath: Path,
    problems_directory: Path,
    workspace: Path,
    # Benchmark others
    preprocess_only: bool = False,
    planner: str = None,
    deadends: bool = None,
    max_num_instances: int = None,
    force_preprocessing: bool = False,
    benchmark_only: bool = False,
    # Feature generation
    disable_feature_generation: bool = False,
    force_feature_generation: bool = False,
    generate_all_distance_features: bool = False,
    concept_complexity_limit: int = 9,
    role_complexity_limit: int = 9,
    boolean_complexity_limit: int = 9,
    count_numerical_complexity_limit: int = 9,
    distance_numerical_complexity_limit: int = 9,
    feature_limit: int = 1000000,
    strict_gc2_features: bool = False,
    extended_features: bool = False,
    # Feature postprocessing
    max_feature_depth: int = None,
    analyze_features: bool = False,
    additional_booleans: List[str] = None,
    additional_numericals: List[str] = None,
    # Feature repositories
    all_repositories: bool = False,
    disable_feature_repositories: bool = False,
    repository: str = None,
    flexible_repositories: bool = False,
    store_features: bool = False,
    # Feature others
    features_only: bool = False,
    # Learner
    width: int = 0,
    monotone_only_by_dec: bool = False,
    rule_elimination: bool = False,
    simplify_policy: bool = False,
    simplify_only_conditions: bool = False,
    uniform_costs: bool = False,
    # Wrapper
    first_instance: int = None,
    instance_selection: str = None,
    randomized_sketch_test: bool = False,
    enumerate_solutions: bool = False,
    # REST
    #disable_closed_Q: bool = False,
    enable_dump_files: bool = False,
    timeout_in_seconds_per_step: Optional[float] = None,
    timeout_in_seconds: Optional[float] = None,
    disable_greedy_solver: Optional[bool] = None,
    verbose: bool = False,
    dump_asp_program: bool = False,
    **kwargs):

    # Create UUID for avoiding clashes with other processes
    uuid_str: str = uuid.uuid4().hex

    # Setup logger
    logger = logging.getLogger()
    logger_level = logging.INFO
    logger.setLevel(logger_level)
    logger_formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')

    logger_handler_stdout = logging.StreamHandler(sys.stdout)
    logger_handler_stdout.setFormatter(logger_formatter)
    #logger_handler.terminator = "" # Change end-of-output terminator from "\n" to ""
    logger.addHandler(logger_handler_stdout)

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

    ####### BENCHMARK
    benchmark_options = {
        "planner": planner,
        "deadends": deadends,
        "preprocess_only": preprocess_only,
        "max_num_instances": max_num_instances,
        "force_preprocessing": force_preprocessing,
    }
    benchmark: Benchmark = Benchmark(domain_filepath, problems_directory, workspace, timers, **benchmark_options)
    if benchmark_only:
        timers.print(title="Finalizing early because option '--benchmark_only'", logger=True)
        return

    ####### FEATURE POOL
    feature_pool_options = {
        # Generation
        "disable_feature_generation": disable_feature_generation,
        "force_feature_generation": force_feature_generation,
        "generate_all_distance_features": generate_all_distance_features,
        "concept_complexity_limit": concept_complexity_limit,
        "role_complexity_limit": role_complexity_limit,
        "boolean_complexity_limit": boolean_complexity_limit,
        "count_numerical_complexity_limit": count_numerical_complexity_limit,
        "distance_numerical_complexity_limit": distance_numerical_complexity_limit,
        "feature_limit": feature_limit,
        "strict_gc2_features": strict_gc2_features,
        "extended_features": extended_features,
        # Post processing
        "max_feature_depth": max_feature_depth,
        "analyze_features": analyze_features,
        "additional_booleans": additional_booleans or [],
        "additional_numericals": additional_numericals or [],
        # Repositories
        "all_repositories": all_repositories,
        "disable_feature_repositories": disable_feature_repositories,
        "flexible_repositories": flexible_repositories,
        "store_features": store_features,
        "repository": repository,
        "uuid_str": uuid_str,
    }
    feature_pool: FeaturePool = FeaturePool(benchmark, timers, **feature_pool_options)
    if features_only:
        timers.print(title="Finalizing early because option '--features_only'", logger=True)
        return

    ####### PREPARE FOR LEARNING
    folder_name_for_iterations: str = f"iterations.{uuid_str}"
    folder_name_for_output: str = f"output.{uuid_str}"
    try:
        from os import makedirs
        makedirs(folder_name_for_iterations)
    except FileExistsError:
        logging.error("Iterations folder {folder_name_for_iterations} exists! Cannot continue...")
        raise

    logging_file_name: str = f"logging.txt"
    logger_handler_file = logging.FileHandler(Path(folder_name_for_iterations) / logging_file_name)
    logger_handler_file.setFormatter(logger_formatter)
    logger.addHandler(logger_handler_file)

    ####### LEARNER
    learner_options = {
        "monotone_only_by_dec": monotone_only_by_dec,
        "rule_elimination": rule_elimination,
        "simplify_policy": simplify_policy,
        "simplify_only_conditions": simplify_only_conditions,
        "uniform_costs": uniform_costs,
        "width": width,
    }
    learner: TerminationBasedLearnerReduced2 = TerminationBasedLearnerReduced2(benchmark, timers, **learner_options)

    ####### WRAPPER
    wrapper_options = {
        "folder_name_for_iterations": folder_name_for_iterations,
        "folder_name_for_output": folder_name_for_output,
        "first_instance": first_instance,
        "instance_selection": instance_selection,
        "max_non_covered_ext_states": 1,
        "randomized_sketch_test": randomized_sketch_test,
        "test_goal_separating_features": False,
        #"solve_pending_requirements": True,
        "solve_pending_requirements": False,
        "hard_constraints": True,
        "max_num_solutions": 1000,
        "max_cost_bound": 100,
        "max_f_idxs": 5,
        "uniform_costs": uniform_costs,
        "dump_asp_program": dump_asp_program,
    }
    WrapperClass = WrapperEnumerationV2 if enumerate_solutions else Wrapper
    wrapper: WrapperBase = WrapperClass(benchmark, feature_pool, learner, timers, **wrapper_options)
    wrapper.learn()

