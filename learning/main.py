#!/usr/bin/env python3

import argparse, sys
from pathlib import Path
from typing import Dict, Optional, Any

from learner.termination_based_learner_reduced import reduced_termination_based_learn_sketch_for_problem_class
from learner.sketch_learner import sketch_learner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sketch learner.")

    # Arguments for benchmark files
    benchmark = parser.add_argument_group("Benchmark")
    benchmark.add_argument("--domain_filepath", type=Path, required=True, help="The path to the domain file")
    benchmark.add_argument("--problems_directory", type=Path, required=True, help="The directory containing the problem files")
    benchmark.add_argument("--workspace", type=Path, required=True, help="The directory containing intermediate files")
    benchmark.add_argument("--preprocess_only", action=argparse.BooleanOptionalAction, default=False, help="Only preprocess instances")
    benchmark.add_argument("--planner", type=str, default="bfws", choices=["bfws", "siw", "siw_plus", "siw+bfws"], help="Set planner (default: 'bfws')")
    benchmark.add_argument("--deadends", action=argparse.BooleanOptionalAction, default=False, help="Instruct the solver there are deadends in the domain")
    benchmark.add_argument("--max_num_instances", type=int, default=None, help="Maximum number of instances to process (default: None)")
    benchmark.add_argument("--force_preprocessing", action=argparse.BooleanOptionalAction, default=False, help="Force preprocessing of PDDL instances")
    benchmark.add_argument("--benchmark_only", action=argparse.BooleanOptionalAction, default=False, help="Only compute benchmark (default: False)")

    # Arguments for feature generation
    feature_generation = parser.add_argument_group("Feature generation")
    feature_generation.add_argument("--disable_feature_generation", action=argparse.BooleanOptionalAction, default=False, help="Whether to enable feature generation")
    feature_generation.add_argument("--force_feature_generation", action=argparse.BooleanOptionalAction, default=False, help="Force generation of features")
    feature_generation.add_argument("--generate_all_distance_features", action=argparse.BooleanOptionalAction, default=False, help="Whether to enable generation of all distance features")
    feature_generation.add_argument("--complexity_limit", type=int, default=None, help="Sets complexity limit for boolean, count, and distance features (default: None)")
    feature_generation.add_argument("--concept_complexity_limit", type=int, default=9, help="The complexity limit for concepts (default: 9)")
    feature_generation.add_argument("--role_complexity_limit", type=int, default=9, help="The complexity limit for roles (default: 9)")
    feature_generation.add_argument("--boolean_complexity_limit", type=int, default=9, help="The complexity limit for boolean features (default: 9)")
    feature_generation.add_argument("--count_numerical_complexity_limit", type=int, default=9, help="The complexity limit for count numerical features (default: 9)")
    feature_generation.add_argument("--distance_numerical_complexity_limit", type=int, default=9, help="The complexity limit for distance numerical features (default: 9)")
    feature_generation.add_argument("--feature_limit", type=int, default=1000000, help="The limit for the number of features (default: 1,000,000)")
    feature_generation.add_argument("--strict_gc2_features", action=argparse.BooleanOptionalAction, default=False, help="Only generate GC2 (Guarded C2) features")
    feature_generation.add_argument("--extended_features", action=argparse.BooleanOptionalAction, default=False, help="Generate extended features (i.e., non-C2 features)")

    # Arguments for feature post-processing
    feature_post_processing = parser.add_argument_group("Feature post-processing")
    feature_post_processing.add_argument("--max_feature_depth", type=int, default=None, help="Limit features by max depth (default: None)")
    feature_post_processing.add_argument("--analyze_features", type=str, default=None, help="Do analysis of features for specified domain (default: None)")
    feature_post_processing.add_argument("--additional_booleans", nargs='*', default=None, help="Additional boolean features to include (default: None)")
    feature_post_processing.add_argument("--additional_numericals", nargs='*', default=None, help="Additional numerical features to include (default: None)")
    #feature_post_processing.add_argument("--max_feature_rank", type=int, default=None, help="Maximum feature rank (default: None)")
    #feature_post_processing.add_argument("--enable_goal_separating_features", action=argparse.BooleanOptionalAction, default=False, help="Whether to enable goal separating features")

    # Arguments for pruning features
    #feature_pruning = parser.add_argument_group("Feature pruning")
    #feature_pruning.add_argument("--enable_incomplete_feature_pruning", action=argparse.BooleanOptionalAction, default=False, help="Whether to enable incomplete feature pruning")
    #feature_pruning.add_argument("--enable_pruning_features_always_positive", action=argparse.BooleanOptionalAction, default=False, help="Whether to enable pruning of features that never reach 0/False")
    #feature_pruning.add_argument("--enable_pruning_features_large_decrease", action=argparse.BooleanOptionalAction, default=False, help="Whether to enable pruning of features that decrease by more than 1")

    # Arguments for feature repositories
    feature_repositories = parser.add_argument_group("Feature repositories")
    feature_repositories.add_argument("--all_repositories", action=argparse.BooleanOptionalAction, default=False, help="Incorporate features from all repositories")
    feature_repositories.add_argument("--disable_feature_repository", action=argparse.BooleanOptionalAction, default=False, help="Don't look for compatible feature repositories")
    feature_repositories.add_argument("--flexible_repositories", action=argparse.BooleanOptionalAction, default=False, help="More flexible when looking for compatible feature repository")
    feature_repositories.add_argument("--store_features", action=argparse.BooleanOptionalAction, default=False, help="Whether features should be stored to a file")
    feature_repositories.add_argument("--repository", type=str, default=None, help="Force feature repository (default: None)")

    # Arguments for feature others
    feature_others = parser.add_argument_group("Feature others")
    feature_others.add_argument("--features_only", action=argparse.BooleanOptionalAction, default=False, help="Stop after computing features")

    # Arguments for learner
    learner = parser.add_argument_group("Learner")
    learner.add_argument("--width", type=int, default=0, help="The upper bound on the sketch width (default: 0)")
    learner.add_argument("--monotone_only_by_dec", action=argparse.BooleanOptionalAction, default=False, help="(Conditional) monotonicity only by decrements")
    learner.add_argument("--rule_elimination", action=argparse.BooleanOptionalAction, default=False, help="Learning by rule elimination")
    learner.add_argument("--simplify_policy", action=argparse.BooleanOptionalAction, default=False, help="Whether to add don't care conditions and unknown effects to projected rules")
    learner.add_argument("--simplify_only_conditions", action=argparse.BooleanOptionalAction, default=False, help="If simplify policy, simplify only conditions")
    learner.add_argument("--threshold_for_asp_based_simplification", type=int, default=8, help="Threshold limit for ASP-based simplification (default: 8)")
    learner.add_argument("--uniform_costs", action=argparse.BooleanOptionalAction, default=False, help="Optimize number of features rather than the sum of complexities")

    # Arguments for solutions
    solutions = parser.add_argument_group("Solutions")
    solutions.add_argument("--max_solutions", type=int, default=1000, help="Maximum number of solutions for enumerator (default: 1000)")
    solutions.add_argument("--max_features", type=int, default=15, help="Maximum number of features in a sketch (default: 15)")
    solutions.add_argument("--cost_bound", type=int, default=100, help="Maximum aggregated cost for sketch (default: 100)")

    # Arguments for backtracks
    backtracks = parser.add_argument_group("Backtracks")
    backtracks.add_argument("--max_backtracks", type=int, default=1000, help="Maximum number of backtracks (default: 1000)")
    backtracks.add_argument("--max_restarts", type=int, default=10, help="Maximum number of restarts (default: 10)")
    backtracks.add_argument("--backtrack_depth", type=int, default=0, help="Depth of backtracks (default: 0)")

    # Arguments for wrapper
    wrapper = parser.add_argument_group("Wrapper")
    wrapper.add_argument("--first_instance", type=int, default=None, help="First instance to solve (default: None)")
    wrapper.add_argument("--instance_selection", type=str, default="forward", choices=["forward", "forward+", "backward", "backward+", "random", "random+", "test"], help="Set strategy for selection of training instances (default: 'forward')")
    wrapper.add_argument("--recompute_relevant_features", action=argparse.BooleanOptionalAction, default=False, help="Wether to recompute relevant features for each added edge (expensive) (default: False)")
    wrapper.add_argument("--randomized_sketch_test", type=int, default=None, help="Whether sketch is randomized rather than fully tested (decreases verification time substantially) (default: None)")
    wrapper.add_argument("--enumerate_solutions", action=argparse.BooleanOptionalAction, default=False, help="Enumerate solutions (default: False)")

    # General options
    general_options = parser.add_argument_group("General options")
    general_options.add_argument("--disable_state_space_expansion", action=argparse.BooleanOptionalAction, default=False, help="Disable full expansion of state space")
    general_options.add_argument("--enable_dump_files", action=argparse.BooleanOptionalAction, default=False, help="Whether data should be written to files")
    general_options.add_argument("--dump_asp_program", action=argparse.BooleanOptionalAction, default=False, help="Dump ASP program")
    general_options.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Produce additional output")

    rest_options = parser.add_argument_group("REST options")
    rest_options.add_argument("--disable_closed_Q", action=argparse.BooleanOptionalAction, default=False, help="Whether the search space is closed")

    # THE FOLLOWING NEED TO BE CLASSIFIED INTO GROUPS
    other_options = parser.add_argument_group("Other options")
    other_options.add_argument("--max_num_states_per_instance", type=int, default=10000, help="The maximum number of states per instance (default: 10,000)")
    other_options.add_argument("--max_time_per_instance", type=int, default=10000, help="The maximum time (in seconds) per instance (default: 10,000)")
    other_options.add_argument("--max_num_rules", type=int, default=4, help="The maximum number of rules used in the explicit encoding (default: 4)")
    other_options.add_argument("--coalesce_instances", action=argparse.BooleanOptionalAction, default=False, help="Coalesce training instances")

    other_options.add_argument("--disable_select_all_features", action=argparse.BooleanOptionalAction, default=False, help="Disable select all features to split optimization problem")
    other_options.add_argument("--disable_greedy_solver", action=argparse.BooleanOptionalAction, default=False, help="Disable greedy solver")
    other_options.add_argument("--disable_greedy_solver_for_choosing_transitions", action=argparse.BooleanOptionalAction, default=False, help="Disable greedy solver for choosing transitions")
    other_options.add_argument("--disable_greedy_solver_for_min_cost_hitting_sets", action=argparse.BooleanOptionalAction, default=False, help="Disable greedy solver for min-cost hitting set problems")
    other_options.add_argument("--timeout_in_seconds_per_step", type=float, default=1200, help="Timeout in seconds for improvement step for the ASP solver (default: 60)")
    other_options.add_argument("--timeout_in_seconds", type=float, default=3600, help="Timeout in seconds for total time for the ASP solver (default: 3600)")

    # Parse arguments
    args = parser.parse_args()

    # Entailed options
    if args.complexity_limit != None:
        args.boolean_complexity_limit = args.complexity_limit
        args.count_numerical_complexity_limit = args.complexity_limit
        args.distance_numerical_complexity_limit = args.complexity_limit
        if args.complexity_limit > args.concept_complexity_limit:
            args.concept_complexity_limit = args.complexity_limit
        if args.complexity_limit > args.role_complexity_limit:
            args.role_complexity_limit = args.complexity_limit

    # Check for incompatible combination of options
    if not args.rule_elimination and args.width > 0:
        raise RuntimeError("Width must be 0 for feature-elimiation solver")

    print(f"Call: python {' '.join(sys.argv)}")

    learner_options: Dict[str, Any] = {
        # Benchmark
        "preprocess_only": args.preprocess_only,
        "planner": args.planner,
        "deadends": args.deadends,
        "max_num_instances": args.max_num_instances,
        "force_preprocessing": args.force_preprocessing,
        "benchmark_only": args.benchmark_only,
        # Feature generation
        "disable_feature_generation": args.disable_feature_generation,
        "force_feature_generation": args.force_feature_generation,
        "generate_all_distance_features": args.generate_all_distance_features,
        "concept_complexity_limit": args.concept_complexity_limit,
        "role_complexity_limit": args.role_complexity_limit,
        "boolean_complexity_limit": args.boolean_complexity_limit,
        "count_numerical_complexity_limit": args.count_numerical_complexity_limit,
        "distance_numerical_complexity_limit": args.distance_numerical_complexity_limit,
        "feature_limit": args.feature_limit,
        "strict_gc2_features": args.strict_gc2_features,
        "extended_features": args.extended_features,
        # Feature post processing
        "max_feature_depth": args.max_feature_depth,
        "analyze_features": args.analyze_features,
        "additional_booleans": args.additional_booleans,
        "additional_numericals": args.additional_numericals,
        # Feature repositories
        "all_repositories": args.all_repositories,
        "disable_feature_repositories": args.disable_feature_repository,
        "flexible_repositories": args.flexible_repositories,
        "store_features": args.store_features,
        "repository": args.repository,
        # Feature others
        "features_only": args.features_only,
        # Learner
        "width": args.width,
        "monotone_only_by_dec": args.monotone_only_by_dec,
        "rule_elimination": args.rule_elimination,
        "simplify_policy": args.simplify_policy or args.simplify_only_conditions,
        "simplify_only_conditions": args.simplify_only_conditions,
        "threshold_for_asp_based_simplification": args.threshold_for_asp_based_simplification,
        "uniform_costs": args.uniform_costs,
        # Solutions
        "max_solutions": args.max_solutions,
        "max_features": args.max_features,
        "cost_bound": args.cost_bound,
        # Backtracks
        "max_backtracks": args.max_backtracks,
        "max_restarts": args.max_restarts,
        "backtrack_depth": args.backtrack_depth,
        # Wrapper
        "first_instance": args.first_instance,
        "instance_selection": args.instance_selection,
        "recompute_relevant_features": args.recompute_relevant_features,
        "randomized_sketch_test": args.randomized_sketch_test,
        "enumerate_solutions": args.enumerate_solutions,
        # REST
        #"disable_closed_Q": args.disable_closed_Q,
        "enable_dump_files": args.enable_dump_files,
        "timeout_in_seconds_per_step": args.timeout_in_seconds_per_step,
        "timeout_in_seconds": args.timeout_in_seconds,
        "disable_greedy_solver": args.disable_greedy_solver,
        "verbose": args.verbose,
        "dump_asp_program": args.dump_asp_program,
    }
    print(learner_options["planner"])

    learner = sketch_learner
    learner(args.domain_filepath.resolve(), args.problems_directory.resolve(), args.workspace.resolve(), **learner_options)

