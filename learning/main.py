#!/usr/bin/env python3

import argparse, sys
from pathlib import Path
from typing import Dict, Optional, Any

from learner.termination_based_learner_reduced import reduced_termination_based_learn_sketch_for_problem_class
from learner.src.iteration import EncodingType


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sketch learner.")

    required = parser.add_argument_group("Required")
    required.add_argument("--domain_filepath", type=Path, required=True, help="The path to the domain file")
    required.add_argument("--problems_directory", type=Path, required=True, help="The directory containing the problem files")
    required.add_argument("--workspace", type=Path, required=True, help="The directory containing intermediate files")

    # Arguments for generation of features
    feature_generation = parser.add_argument_group("Feature generation")
    feature_generation.add_argument("--disable_feature_generation", action=argparse.BooleanOptionalAction, default=False, help="Whether to enable feature generation")
    feature_generation.add_argument("--max_feature_depth", type=int, default=None, help="Limit features by max depth (default: None)")
    feature_generation.add_argument("--max_feature_rank", type=int, default=None, help="Maximum feature rank (default: None)")
    feature_generation.add_argument("--strict-gc2-features", action=argparse.BooleanOptionalAction, default=False, help="Only generate GC2 (Guarded C2) features")
    feature_generation.add_argument("--enable_goal_separating_features", action=argparse.BooleanOptionalAction, default=False, help="Whether to enable goal separating features")
    feature_generation.add_argument("--generate_all_distance_features", action=argparse.BooleanOptionalAction, default=False, help="Whether to enable generation of all distance features")
    feature_generation.add_argument("--complexity_limit", type=int, default=None, help="Sets complexity limit for boolean, count, and distance features (default: None)")
    feature_generation.add_argument("--concept_complexity_limit", type=int, default=9, help="The complexity limit for concepts (default: 9)")
    feature_generation.add_argument("--role_complexity_limit", type=int, default=9, help="The complexity limit for roles (default: 9)")
    feature_generation.add_argument("--boolean_complexity_limit", type=int, default=9, help="The complexity limit for boolean features (default: 9)")
    feature_generation.add_argument("--count_numerical_complexity_limit", type=int, default=9, help="The complexity limit for count numerical features (default: 9)")
    feature_generation.add_argument("--distance_numerical_complexity_limit", type=int, default=9, help="The complexity limit for distance numerical features (default: 9)")
    feature_generation.add_argument("--feature_limit", type=int, default=1000000, help="The limit for the number of features (default: 1,000,000)")
    feature_generation.add_argument("--additional_booleans", nargs='*', default=None, help="Additional boolean features to include (default: None)")
    feature_generation.add_argument("--additional_numericals", nargs='*', default=None, help="Additional numerical features to include (default: None)")
    feature_generation.add_argument("--analyze_features", type=str, default=None, help="Do analysis of features for specified domain (default: None)")

    # Arguments for pruning features
    feature_pruning = parser.add_argument_group("Feature pruning")
    feature_pruning.add_argument("--enable_incomplete_feature_pruning", action=argparse.BooleanOptionalAction, default=False, help="Whether to enable incomplete feature pruning")
    feature_pruning.add_argument("--enable_pruning_features_always_positive", action=argparse.BooleanOptionalAction, default=False, help="Whether to enable pruning of features that never reach 0/False")
    feature_pruning.add_argument("--enable_pruning_features_large_decrease", action=argparse.BooleanOptionalAction, default=False, help="Whether to enable pruning of features that decrease by more than 1")

    # Arguments for repositories of features
    feature_repository = parser.add_argument_group("Feature repositories")
    feature_repository.add_argument("--disable_feature_repository", action=argparse.BooleanOptionalAction, default=False, help="Don't look for compatible feature repositories")
    feature_repository.add_argument("--flexible_repositories", action=argparse.BooleanOptionalAction, default=False, help="More flexible when looking for compatible feature repository")
    feature_repository.add_argument("--all_repositories", action=argparse.BooleanOptionalAction, default=False, help="Incorporate features from all repositories")
    feature_repository.add_argument("--store_features", action=argparse.BooleanOptionalAction, default=False, help="Whether features should be stored to a file")

    # Learner options
    learner_options = parser.add_argument_group("Learner options")
    learner_options.add_argument("--width", type=int, default=0, help="The upper bound on the sketch width (default: 0)")

    # Genex options
    genex_options = parser.add_argument_group("Genex options")
    genex_options.add_argument("--contextual", action=argparse.BooleanOptionalAction, default=False, help="(Conditional) monotonicity by contexts")
    genex_options.add_argument("--monotone_only_by_dec", action=argparse.BooleanOptionalAction, default=False, help="(Conditional) monotonicity only by decrements")
    genex_options.add_argument("--uniform_costs", action=argparse.BooleanOptionalAction, default=False, help="Optimize number of features rather than the sum of complexities")
    genex_options.add_argument("--simplify_policy", action=argparse.BooleanOptionalAction, default=False, help="Whether to add don't care conditions and unknown effects to projected rules")
    genex_options.add_argument("--simplify_only_conditions", action=argparse.BooleanOptionalAction, default=False, help="If simplify policy, simplify only conditions")
    genex_options.add_argument("--separate_siblings", action=argparse.BooleanOptionalAction, default=False, help="Whether to separate chosen transition from sibling transitions")

    # Wrapper options
    wrapper_options = parser.add_argument_group("Wrapper options")
    wrapper_options.add_argument("--planner", type=str, default="bfws", choices=["bfws", "siw", "siw_plus", "siw+bfws"], help="Set planner (default: 'bfws')")
    wrapper_options.add_argument("--deadends", action=argparse.BooleanOptionalAction, default=False, help="Instruct the solver there are deadends in the domain")
    wrapper_options.add_argument("--first_instance", type=int, default=None, help="First instance to solve (default: None)")
    wrapper_options.add_argument("--instance_selection", type=str, default="forward", choices=["forward", "forward+", "backward", "backward+", "random", "random+", "test"], help="Set strategy for selection of training instances (default: 'forward')")
    wrapper_options.add_argument("--disable_closed_Q", action=argparse.BooleanOptionalAction, default=False, help="Whether the search space is closed")
    wrapper_options.add_argument("--randomized_sketch_test", type=int, default=None, help="Whether sketch is randomized rather than fully tested (decreases verification time substantially) (default: None)")

    # General options
    general_options = parser.add_argument_group("General options")
    general_options.add_argument("--features_only", action=argparse.BooleanOptionalAction, default=False, help="Stop after computing features")
    general_options.add_argument("--preprocess_only", action=argparse.BooleanOptionalAction, default=False, help="Only preprocess instances")
    general_options.add_argument("--disable_state_space_expansion", action=argparse.BooleanOptionalAction, default=False, help="Disable full expansion of state space")
    general_options.add_argument("--max_num_instances", type=int, default=None, help="Maximum number of instances to process (default: None)")
    general_options.add_argument("--enable_dump_files", action=argparse.BooleanOptionalAction, default=False, help="Whether data should be written to files")
    general_options.add_argument("--dump_asp_program", action=argparse.BooleanOptionalAction, default=False, help="Dump ASP program")
    general_options.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Produce additional output")

    # THE FOLLOWING NEED TO BE CLASSIFIED INTO GROUPS
    other_options = parser.add_argument_group("Other options")
    other_options.add_argument("--max_num_states_per_instance", type=int, default=10000, help="The maximum number of states per instance (default: 10,000)")
    other_options.add_argument("--max_time_per_instance", type=int, default=10000, help="The maximum time (in seconds) per instance (default: 10,000)")
    other_options.add_argument("--encoding_type", type=EncodingType, default=EncodingType.D2, choices=[EncodingType.D2, EncodingType.EXPLICIT, EncodingType.D2_LTL, EncodingType.D2r, EncodingType.D2g, EncodingType.SOUNDNESS, EncodingType.TERMINATION], help="The encoding type for the sketch learner (default: 'd2'")
    other_options.add_argument("--max_num_rules", type=int, default=4, help="The maximum number of rules used in the explicit encoding (default: 4)")
    other_options.add_argument("--coalesce_instances", action=argparse.BooleanOptionalAction, default=False, help="Coalesce training instances")

    other_options.add_argument("--disable_select_all_features", action=argparse.BooleanOptionalAction, default=False, help="Disable select all features to split optimization problem")
    other_options.add_argument("--disable_greedy_solver", action=argparse.BooleanOptionalAction, default=False, help="Disable greedy solver")
    other_options.add_argument("--disable_greedy_solver_for_choosing_transitions", action=argparse.BooleanOptionalAction, default=False, help="Disable greedy solver for choosing transitions")
    other_options.add_argument("--disable_greedy_solver_for_min_cost_hitting_sets", action=argparse.BooleanOptionalAction, default=False, help="Disable greedy solver for min-cost hitting set problems")
    other_options.add_argument("--core_type", type=int, default=None, help="Core type: 0=naive, 1=number-goals, 2=optimal-plans, 3=goal-orderings (default: None)")
    other_options.add_argument("--slicing_method", type=str, default="number_subgoals", choices=["number_subgoals", "simple_paths", "subgoal_subsets", "planner"], help="Set method for slicing state space (default: 'number_subgoals')")
    other_options.add_argument("--timeout_in_seconds_per_step", type=float, default=1200, help="Timeout in seconds for improvement step for the ASP solver (default: 60)")
    other_options.add_argument("--timeout_in_seconds", type=float, default=3600, help="Timeout in seconds for total time for the ASP solver (default: 3600)")
    other_options.add_argument("--disable_optimization_decorations", action=argparse.BooleanOptionalAction, default=False, help="Disable optimization of decorations in sketch")
    other_options.add_argument("--solver_prefix", type=str, default=None, help="Prefix for solver's name (default: None)")

    args = parser.parse_args()
    if args.complexity_limit != None:
        args.boolean_complexity_limit = args.complexity_limit
        args.count_numerical_complexity_limit = args.complexity_limit
        args.distance_numerical_complexity_limit = args.complexity_limit
        if args.complexity_limit > args.concept_complexity_limit:
            args.concept_complexity_limit = args.complexity_limit
        if args.complexity_limit > args.role_complexity_limit:
            args.role_complexity_limit = args.complexity_limit

    print(f"Call: python {' '.join(sys.argv)}")

    learner_options: Dict[str, Any] = {
        "domain_filepath": args.domain_filepath.resolve(),
        "problems_directory": args.problems_directory.resolve(),
        "max_num_instances": args.max_num_instances,
        "workspace": args.workspace.resolve(),
        "width": args.width,
        "disable_closed_Q": args.disable_closed_Q,
        "randomized_sketch_test": args.randomized_sketch_test,
        "disable_feature_generation": args.disable_feature_generation,
        "generate_all_distance_features": args.generate_all_distance_features,
        "enable_incomplete_feature_pruning": args.enable_incomplete_feature_pruning,
        "enable_pruning_features_always_positive": args.enable_pruning_features_always_positive,
        "enable_pruning_features_large_decrease": args.enable_pruning_features_large_decrease,
        "concept_complexity_limit": args.concept_complexity_limit,
        "role_complexity_limit": args.role_complexity_limit,
        "boolean_complexity_limit": args.boolean_complexity_limit,
        "count_numerical_complexity_limit": args.count_numerical_complexity_limit,
        "distance_numerical_complexity_limit": args.distance_numerical_complexity_limit,
        "feature_limit": args.feature_limit,
        "strict_gc2_features": args.strict_gc2_features,
        "additional_booleans": args.additional_booleans,
        "additional_numericals": args.additional_numericals,
        "disable_feature_repositories": args.disable_feature_repository,
        "store_features": args.store_features,
        "flexible_repositories": args.flexible_repositories,
        "all_repositories": args.all_repositories,
        "enable_dump_files": args.enable_dump_files,
        "instance_selection": args.instance_selection,
        "first_instance": args.first_instance,
        "planner": args.planner,
        "max_feature_depth": args.max_feature_depth,
        "analyze_features": args.analyze_features,
        "timeout_in_seconds_per_step": args.timeout_in_seconds_per_step,
        "timeout_in_seconds": args.timeout_in_seconds,
        "disable_greedy_solver": args.disable_greedy_solver,
        "disable_optimization_decorations": args.disable_optimization_decorations,
        "deadends": args.deadends,
        "solver_prefix": args.solver_prefix,
        "simplify_policy": args.simplify_policy,
        "simplify_only_conditions": args.simplify_only_conditions,
        "separate_siblings": args.separate_siblings,
        "contextual": args.contextual,
        "monotone_only_by_dec": args.monotone_only_by_dec,
        "uniform_costs": args.uniform_costs,
        "verbose": args.verbose,
        "dump_asp_program": args.dump_asp_program,
        "preprocess_only": args.preprocess_only,
        "features_only": args.features_only,
    }

    learner = reduced_termination_based_learn_sketch_for_problem_class
    learner(**learner_options)

