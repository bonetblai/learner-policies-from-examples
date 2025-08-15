import logging
from termcolor import colored
from typing import Tuple, List, Dict, Set, Optional, Any
from pathlib import Path
from datetime import datetime

from bloom_filter2 import BloomFilter
from tqdm import tqdm

from enum import Enum
from collections import defaultdict, Counter
from itertools import product
import gzip

import pymimir as mm
import dlplan.generator as dlplan_generator
import dlplan.core as dlplan_core

from .feature_pool import Feature
from .iteration_data import IterationData

from ..preprocessing import PreprocessingData, StateFinder
from ..state_space import StateFactory
from ..util import Timer


class FeatureChange(Enum):
    UP = 0
    DOWN = 1
    BOT = 2

def _is_matching_repository(feature_repository: Path, feature_parameters: Dict[str, Any], instance_names: List[str], flexible: bool = False) -> Optional[Dict[str, Any]]:
    flexible_feature_parameters: Dict[str, Any] = dict(feature_parameters)
    if flexible: flexible_feature_parameters.pop('planner')
    compressed: bool = feature_repository.name.endswith("gz") or feature_repository.name.endswith("gzip")
    with (feature_repository.open("r") if not compressed else gzip.open(feature_repository, "rt")) as fd:
        header: List[str] = [fd.readline().strip("\n"), fd.readline().strip("\n"), fd.readline().strip("\n")]
        read_parameters: Dict[str, Any] = eval(header[0])
        if "max_feature_depth" in read_parameters:
            read_parameters.pop("max_feature_depth")
        if flexible: read_parameters.pop('planner')
        read_names: List[str] = eval(header[1])
        read_statistics: Dict[str, Any] = eval(header[2])
        if read_parameters == flexible_feature_parameters and (flexible or read_names == instance_names):
            return read_statistics
        else:
            return None

def find_feature_repositories(folder: Path, feature_parameters: Dict[str, Any], instance_names: List[str], all_repositories: bool = False, flexible: bool = False) -> List[Path]:
    # Collect compatible repositories
    repositories: List[Path] = list(folder.glob("**/*.frepo")) + list(folder.glob("**/*.frepo.gz"))
    compatible_repositories: List[Tuple[Path, Dict[str, Any]]] = []
    if all_repositories:
        return repositories
    else:
        for feature_repository in repositories:
            statistics: Dict[str, Any] = _is_matching_repository(feature_repository, feature_parameters, instance_names, flexible)
            if statistics is not None:
                compatible_repositories.append((feature_repository, statistics))

        # Find best compatible repository as the one with more features
        best_index = -1
        for index, (feature_repository, statistics) in enumerate(compatible_repositories):
            if best_index == -1 or (compatible_repositories[best_index][1].get("features", -1) < statistics.get("features", 0)):
                best_index = index
        return None if best_index == -1 else [compatible_repositories[best_index][0]]

def parse_feature(syntactic_element_factory: Any, feature_name: str) -> Feature:
    feature: Feature = None
    if feature_name.startswith("n_"):
        # This is a numerical feature
        try:
            numerical_parsed = syntactic_element_factory.parse_numerical(feature_name)
        except Exception as inst:
            logging.warning(colored(f"[{Path(__file__).name}] Failed parse of numerical feature '{feature_name}'; skipping...", "magenta"))
        else:
            max_depth_complexity = _max_depth(str(feature_name))
            feature: Feature = Feature(numerical_parsed, numerical_parsed.compute_complexity() + 1, max_depth_complexity + 1)
    elif feature_name.startswith("b_"):
        # This is a boolean feature
        try:
            boolean_parsed = syntactic_element_factory.parse_boolean(feature_name)
        except Exception as inst:
            logging.warning(colored(f"[{Path(__file__).name}] Failed parse of boolean feature '{feature_name}'; skipping...", "magenta"))
        else:
            max_depth_complexity = _max_depth(str(feature_name))
            feature: Feature = Feature(boolean_parsed, boolean_parsed.compute_complexity() + 1, max_depth_complexity + 1)
    else:
        raise RuntimeError("Unrecognized feature '{feature_name}'")
    return feature

def read_features_from_repositories(feature_repositories: List[Path],
                                    syntactic_element_factory: Any,
                                    **kwargs) -> Tuple[List[Feature], Dict[str, Any]]:
    feature_names: Set[str] = set()
    features: List[Feature] = []
    list_r_statistics: List[Dict[str, Any]] = []
    for feature_repository in feature_repositories:
        r_features, r_statistics = _read_features_from_repository(feature_repository, syntactic_element_factory, **kwargs)
        list_r_statistics.append(r_statistics)
        for feature in r_features:
            feature_name: str = str(feature.dlplan_feature)
            if feature_name not in feature_names:
                feature_names.add(feature_name)
                features.append(feature)

    if len(feature_repositories) == 1:
        return features, list_r_statistics[0]
    else:
        # Statistics about number of features
        statistics: Dict[str, Any] = {
            "features": len(features),
            "numerical_features": len([feature for feature in features if isinstance(feature.dlplan_feature, dlplan_core.Numerical)]),
            "boolean_features": len([feature for feature in features if isinstance(feature.dlplan_feature, dlplan_core.Boolean)]),
            "max_complexity": max([feature.complexity for feature in features]) - 1,
            "max_depth_complexity": max([feature.max_depth_complexity for feature in features]) - 1,
        }
        features: List[Feature] = sorted(features, key=lambda feature: (feature.complexity, feature.max_depth_complexity))
        return features, statistics

def _read_features_from_repository(feature_repository: Path,
                                   syntactic_element_factory: Any,
                                   **kwargs) -> Tuple[List[Feature], Dict[str, Any]]:
    if not feature_repository.exists():
        logging.error(f"Feature repository '{feature_repository}' does not exist")
        return None

    compressed: bool = feature_repository.name.endswith("gz") or feature_repository.name.endswith("gzip")

    features: List[Feature] = []
    with (feature_repository.open("r") if not compressed else gzip.open(feature_repository, "rt")) as fd:
        r_parameters: Dict[str, Any] = eval(fd.readline().strip("\n"))
        r_instance_names: List[str] = eval(fd.readline().strip("\n"))
        r_statistics: Dict[str, Any] = eval(fd.readline().strip("\n"))

        timer: Timer = Timer()

        num_features: int = 0
        while True:
            feature_name: str = fd.readline().strip("\n")
            if not feature_name: break
            feature: Feature = parse_feature(syntactic_element_factory, feature_name)
            if feature is not None:
                features.append(feature)
                num_features += 1

        timer.stop()
        logging.info(f"{num_features} feature(s) read from '{feature_repository}' in {timer.get_elapsed_sec():0.2f} second(s)")

    additional_feature_names: List[str] = kwargs.get("additional_numericals", []) + kwargs.get("additional_booleans", [])
    for feature_name in additional_feature_names:
        feature: Feature = parse_feature(syntactic_element_factory, feature_name)
        if feature is not None:
            features.append(feature)
    if len(additional_feature_names) > 0:
        logging.info(f"{len(additional_feature_names)} additional feature(s)")

    return features, r_statistics

def write_features_to_repository(features: List[Feature],
                                 feature_parameters: Dict[str, Any],
                                 feature_statistics: Dict[str, Any],
                                 instance_names: List[str],
                                 feature_repository: Path,
                                 compressed: bool = True):
    feature_repository.parent.mkdir(parents=True, exist_ok=True)
    with (feature_repository.open("w") if not compressed else gzip.open(feature_repository.with_suffix(feature_repository.suffix + ".gz"), "wt")) as fd:
        fd.write(f"{str(feature_parameters)}\n")
        fd.write(f"{instance_names}\n")
        fd.write(f"{str(feature_statistics)}\n")
        for feature in features:
            feature_name: str = str(feature._dlplan_feature)
            fd.write(f"{feature_name}\n")

def post_process_features(features: List[Feature], **kwargs) -> List[Feature]:
    max_feature_depth = kwargs.get("max_feature_depth")
    max_depth_pruning_timer: Timer = Timer(stopped=True)

    # PRUNE generated features by max-depth complexity
    max_depth_pruning_timer.resume()
    if max_feature_depth is not None:
        logging.info(f"Pruning features by max_depth={max_feature_depth}:")

        numerical_features: List[Feature] = [feature for feature in features if isinstance(feature.dlplan_feature, dlplan_core.Numerical)]
        boolean_features: List[Feature] = [feature for feature in features if isinstance(feature.dlplan_feature, dlplan_core.Boolean)]
        numerical_features_new: List[Feature] = [feature for feature in numerical_features if _max_depth(str(feature.dlplan_feature)) <= max_feature_depth]
        boolean_features_new: List[Feature] = [feature for feature in boolean_features if _max_depth(str(feature.dlplan_feature)) <= max_feature_depth]

        pruning_stats = {
            "numerical": {
                "pruned": len(numerical_features) - len(numerical_features_new),
                "remain": len(numerical_features_new),
                "reduction": 100 * (1.0 - float(len(numerical_features_new)) / float(len(numerical_features)))
            },
            "boolean": {
                "pruned": len(boolean_features) - len(boolean_features_new),
                "remain": len(boolean_features_new),
                "reduction": 100 * (1.0 - float(len(boolean_features_new)) / float(len(boolean_features)))
            }
        }
        for key, stats in pruning_stats.items():
            logging.info(f"    {key.title()}s: {stats['pruned']} feature(s) pruned; {stats['remain']} feature(s) remain [{stats['reduction']:.1f}% reduction]")

        features: List[Feature] = numerical_features_new + boolean_features_new
    max_depth_pruning_timer.stop()

    return features

def get_statistics(features: List[Feature]) -> Dict[str, Any]:
    num_features = len(features)
    num_numerical_features = len([feature for feature in features if isinstance(feature.dlplan_feature, dlplan_core.Numerical)])
    max_complexity = -1
    max_depth = max([_max_depth(str(feature.dlplan_feature)) for feature in features])
    return {
        "features": num_features,
        "numerical": numerical,
        "boolean": num_features - num_numerical_features,
        "max_complexity": max_complexity,
        "max_depth": max_depth,
    }

def generate_features(syntactic_element_factory: Any,
                      dlplan_states: List[dlplan_core.State],
                      instance_idx_to_denotations_caches: Dict[int, dlplan_core.DenotationsCaches],
                      disable_feature_generation: bool = False,
                      generate_all_distance_features: bool = False,
                      concept_complexity_limit: int = 9,
                      role_complexity_limit: int = 9,
                      boolean_complexity_limit: int = 9,
                      count_numerical_complexity_limit: int = 9,
                      distance_numerical_complexity_limit: int = 9,
                      feature_limit: int = 1000,
                      strict_gc2_features: bool = False,
                      additional_booleans: List[str] = [],
                      additional_numericals: List[str] = [],
                      max_feature_depth: Optional[int] = None,
                      analyze_features: Optional[str] = None) -> Tuple[List[Feature], Dict[str, int]]:
    # Statistics
    generation_timer = Timer(stopped=True)
    distance_features_timer = Timer(stopped=True)
    additional_features_timer = Timer(stopped=True)
    statistics: Dict[str, Any] = {"dlplan_states": len(dlplan_states), "instances": len(instance_idx_to_denotations_caches)}

    # List of features
    features: List[Feature] = []

    # Generate features
    if not disable_feature_generation:
        feature_generator = dlplan_generator.FeatureGenerator()
        feature_generator.set_generate_inclusion_boolean(False)
        feature_generator.set_generate_diff_concept(False)
        feature_generator.set_generate_or_concept(False)
        feature_generator.set_generate_projection_concept(False)
        feature_generator.set_generate_subset_concept(False)
        feature_generator.set_generate_compose_role(False)
        feature_generator.set_generate_diff_role(False)
        feature_generator.set_generate_identity_role(False)
        feature_generator.set_generate_not_role(False)
        feature_generator.set_generate_or_role(False)
        feature_generator.set_generate_top_role(False)
        feature_generator.set_generate_transitive_reflexive_closure_role(False)

        logging.warning(colored(f"[{Path(__file__).name}] Disabling til_c roles because testing script doesn't support them!", "red"))
        feature_generator.set_generate_til_c_role(False)

        if strict_gc2_features:
            # Disable all non-GC2 concepts and roles
            feature_generator.set_generate_equal_concept(False)  # CHECK
            feature_generator.set_generate_and_role(False)       # CHECK

        # Generate features
        generation_timer.resume()
        [generated_booleans, generated_numericals, generated_concepts, generated_roles] = feature_generator.generate(
            syntactic_element_factory,
            dlplan_states,
            concept_complexity_limit,
            role_complexity_limit,
            boolean_complexity_limit,
            count_numerical_complexity_limit,
            distance_numerical_complexity_limit,
            2147483647,  # max time limit,
            feature_limit)
        generation_timer.stop()

        # Populated list of features
        for numerical in generated_numericals:
            max_depth_complexity = _max_depth(str(numerical))
            features.append(Feature(numerical, numerical.compute_complexity() + 1, max_depth_complexity + 1))
        for boolean in generated_booleans:
            max_depth_complexity = _max_depth(str(boolean))
            features.append(Feature(boolean, boolean.compute_complexity() + 1, max_depth_complexity + 1))

        statistics.update({"features": len(features), "numerical_features": len(generated_numericals), "boolean_features": len(generated_booleans), "concepts": len(generated_concepts), "roles": len(generated_roles)})
        for key, value in statistics.items():
            logging.info(f"{key}: {value}")

        # Extend features with distance features for each pair of concepts and each role
        distance_features_timer.resume()
        if False and generate_all_distance_features:
            logging.info(f"Generating all distance features with {len(generated_concepts)} concept(s) and {len(generated_roles)} role(s)...")
            num_distance_features = 0
            existing_distance_features: Set[str] = set([str(feature._dlplan_feature) for feature in features if str(feature._dlplan_feature).startswith("n_concept_distance")])
            for concept1, concept2 in product(generated_concepts, generated_concepts):
                oncept1_max_depth_complexity = _max_depth(str(concept1))
                oncept2_max_depth_complexity = _max_depth(str(concept2))
                if concept1.compute_complexity() + concept2.compute_complexity() + 1 < distance_numerical_complexity_limit:
                    complexity_budget = distance_numerical_complexity_limit - concept1.compute_complexity() - concept2.compute_complexity()
                    for role in generated_roles:
                        if role.compute_complexity() + 1 < complexity_budget:
                            feature_str = f"n_concept_distance({concept1},{role},{concept2})"
                            if feature_str not in existing_distance_features:
                                parsed = syntactic_element_factory.parse_numerical(feature_str)
                                assert parsed.compute_complexity() <= distance_numerical_complexity_limit
                                feature: Feature = Feature(parsed, parsed.compute_complexity() + 1, max(concept1_max_depth_complexity, concept2_max_depth_complexity) + 2)
                                features.append(feature)
                                num_distance_features += 1
            logging.info(f"{num_distance_features} distance features(s) added")
        distance_features_timer.stop()

    # Generate additional numerical and boolean features
    additional_features_timer.resume()
    additional_features: Dict[str, int] = dict()
    for numerical in additional_numericals:
        try:
            numerical_parsed = syntactic_element_factory.parse_numerical(numerical)
        except Exception as inst:
            logging.warning(colored(f"[{Path(__file__).name}] Failed parse of numerical feature '{numerical}'; skipping...", "magenta"))
        else:
            max_depth_complexity = _max_depth(str(numerical))
            feature: Feature = Feature(numerical_parsed, numerical_parsed.compute_complexity() + 1, max_depth_complexity + 1)
            features.append(feature)
            logging.info(f"Additional numerical feature: {len(features) - 1}.{features[-1]._dlplan_feature} / {features[-1]._complexity}")
            additional_features[str(feature._dlplan_feature)] = -1
    for boolean in additional_booleans:
        try:
            boolean_parsed = syntactic_element_factory.parse_boolean(boolean)
        except Exception as inst:
            logging.warning(colored(f"[{Path(__file__).name}] Failed parse of boolean feature '{boolean}'; skipping...", "magenta"))
        else:
            max_depth_complexity = _max_depth(str(boolean))
            feature: Feature = Feature(boolean_parsed, boolean_parsed.compute_complexity() + 1, max_depth_complexity + 1)
            features.append(feature)
            logging.info(f"Additional boolean feature: {len(features) - 1}.{features[-1]._dlplan_feature} / {features[-1]._complexity}")
            additional_features[str(feature._dlplan_feature)] = -1
    additional_features_timer.stop()

    if analyze_features is not None:
        logging.info(f"Analysis: Concepts and roles")
        #_analyze_concepts_and_roles(analyze_features, generated_concepts, generated_roles)
        _analyze_features(analyze_features, features)

    # Fill in remaining statistics
    time_statistics: Dict[str, Any] = {
        "generation_time": generation_timer.get_elapsed_sec(),
        "distance_features_time": distance_features_timer.get_elapsed_sec(),
        "additional_features_time": additional_features_timer.get_elapsed_sec(),
        "timestamp": str(datetime.now()),
    }
    statistics.update(time_statistics)

    return features, additional_features, statistics


def compute_feature_pool(preprocessing_data: PreprocessingData,
                         iteration_data: IterationData,
                         gfa_state_id_to_tuple_graph: Dict[int, mm.TupleGraph],
                         state_finder: StateFinder,
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
                         feature_limit: int = 1000,
                         strict_gc2_features: bool = False,
                         additional_booleans: List[str] = [],
                         additional_numericals: List[str] = [],
                         max_feature_depth: Optional[int] = None,
                         analyze_features: Optional[str] = None) -> List[Feature]:
    assert False

    # Get concrete dlplan states of global states
    dlplan_ss_states: Set[dlplan_core.State] = set()
    for gfa_state in iteration_data.gfa_states:
        dlplan_ss_states.add(state_finder.get_dlplan_ss_state(gfa_state))
    dlplan_ss_states: List[dlplan_core.State] = list(dlplan_ss_states)

    # Generate features for dlplan_ss_states
    features, additional_features, statistics = generate_features(
        preprocessing_data.syntactic_element_factory,
        dlplan_ss_states,
        None,
        disable_feature_generation,
        generate_all_distance_features,
        concept_complexity_limit,
        role_complexity_limit,
        boolean_complexity_limit,
        count_numerical_complexity_limit,
        distance_numerical_complexity_limit,
        feature_limit,
        strict_gc2_features,
        additional_booleans,
        additional_numericals,
        max_feature_depth,
        analyze_features)

    # Post-processing of features (e.g., pruned by max depth)
    logging.info(colored("Post-processing features...", "blue"))
    features_options: Dict[str, Any] = {
        "max_feature_depth": max_feature_depth,
        "analyze_features": analyze_features,
    }
    features = post_process_features(features, **features_options)

    # Construct equivalence classes for features modulo (non-boolean) valuation across gfa states
    logging.info(f"Constructing equivalence classes of features modulo (non-boolean) valuation across gfa states...")
    valuation_idx_to_valuation: List[Tuple[int]] = []
    valuation_to_valuation_idx: Dict[Tuple[int], int] = dict()
    valuation_idx_to_equivalence_class: Dict[int, List[int]] = defaultdict(list)

    for f_idx, feature in enumerate(features):
        valuation: List[int] = []
        for gfa_state in iteration_data.gfa_states:
            instance_idx = gfa_state.get_faithful_abstraction_index()
            instance_data = preprocessing_data.instance_datas[instance_idx]
            dlplan_source_ss_state = state_finder.get_dlplan_ss_state(gfa_state)
            valuation.append(int(feature.dlplan_feature.evaluate(dlplan_source_ss_state, instance_data.denotations_caches)))

        valuation: Tuple[int] = tuple(valuation)
        valuation_idx = valuation_to_valuation_idx.get(valuation)
        if valuation_idx is None:
            valuation_idx = len(valuation_idx_to_valuation)
            valuation_idx_to_valuation.append(valuation)
            valuation_to_valuation_idx[valuation] = valuation_idx
        valuation_idx_to_equivalence_class[valuation_idx].append(f_idx)

    logging.info(f"{len(features)} feature(s), {len(valuation_idx_to_valuation)} equivalence class(es)")

    if analyze_features is not None:
        logging.info(f"Analysis 3: Features after constructing equivalence classes")
        _analyze_features(analyze_features, features)

    # PRUNE features with identical valuation; keep one representative from each class
    logging.info(f"Pruning features with identical valuations...")
    num_features_before_pruning = len(features)
    features_to_keep: List[int] = [sorted(eqclass, key=lambda item: features[f_idx].complexity)[0] for eqclass in valuation_idx_to_equivalence_class.values()]
    features = [features[f_idx] for f_idx in features_to_keep]
    logging.info(f"{len(features)} feature(s) after pruning those with identical valuation (num_pruned={num_features_before_pruning - len(features)})")

    if analyze_features is not None:
        logging.info(f"Analysis 4: Features after reducing equivalence classes")
        _analyze_features(analyze_features, features)

    # PRUNE features that never reach 0/False
    if enable_pruning_features_always_positive or enable_incomplete_feature_pruning:
        logging.info(f"Pruning features that never reach 0/False...")
        num_pruned, selected_features = 0, []
        for feature in features:
            always_non_zero = True
            for instance_data in iteration_data.instance_datas:
                for dlplan_state in instance_data.dlplan_ss.get_states().values():
                    val = int(feature.dlplan_feature.evaluate(dlplan_state, instance_data.denotations_caches))
                    if val == 0:
                        always_non_zero = False
                        break
            num_pruned += 1 if always_non_zero else 0
            if not always_non_zero:
                selected_features.append(feature)
        features = selected_features
        logging.info(f"{len(features)} feature(s) after pruning always positive features (num_pruned={num_pruned})")

    # PRUNE features that decrease by more than 1 on a state transition
    if enable_pruning_features_large_decrease or enable_incomplete_feature_pruning:
        logging.info(f"Pruning features that decrease more than 1 on a state transition...")
        num_pruned, soft_changing_features = 0, set()
        for feature in features:
            is_soft_changing = True
            for gfa_state in iteration_data.gfa_states:
                dlplan_source_ss_state = state_finder.get_dlplan_ss_state(gfa_state)
                instance_idx = gfa_state.get_faithful_abstraction_index()
                instance_data = preprocessing_data.instance_datas[instance_idx]
                source_val = int(feature.dlplan_feature.evaluate(dlplan_source_ss_state, instance_data.denotations_caches))

                gfa = instance_data.gfa
                gfa_state_global_idx = gfa_state.get_global_index()
                gfa_state_idx = gfa.get_abstract_state_index(gfa_state_global_idx)
                gfa_states = gfa.get_states()
                for gfa_state_prime_idx in gfa.get_forward_adjacent_state_indices(gfa_state_idx):
                    gfa_state_prime = gfa_states[gfa_state_prime_idx]
                    dlplan_target_ss_state = state_finder.get_dlplan_ss_state(gfa_state_prime)
                    target_val = int(feature.dlplan_feature.evaluate(dlplan_target_ss_state, instance_data.denotations_caches))
                    if source_val in {0, 2147483647} or target_val in {0, 2147483647}:
                        # Allow arbitrary changes on border values
                        continue
                    if source_val > target_val and (source_val > target_val + 1):
                        is_soft_changing = False
                        break
                    if target_val > source_val and (target_val > source_val + 1):
                        is_soft_changing = False
                        break
                if not is_soft_changing:
                    break
            num_pruned += 1 if not is_soft_changing else 0
            if is_soft_changing:
                soft_changing_features.add(feature)
        features = list(soft_changing_features)
        logging.info(f"{len(features)} feature(s) after pruning features that decrease by more than 1 (num_pruned={num_pruned})")

    # Calculate indices of additional features
    for f_name in additional_features.keys():
        f_idxs: List[int] = [f_idx for f_idx, feature in enumerate(features) if f_name == str(feature._dlplan_feature)]
        assert len(f_idxs) <= 1
        assert additional_features.get(f_name) == -1
        additional_features[f_name] = f_idxs[0] if len(f_idxs) > 0 else -1

    # Construct equivalence classes for features modulo boolean valuations across gfa states
    logging.info(f"Construct equivalece classes for features modulo boolean valuations across gfa states...")
    boolean_valuation_idx_to_boolean_valuation: List[Tuple[bool]] = []
    boolean_valuation_to_boolean_valuation_idx: Dict[Tuple[bool], int] = dict()
    boolean_valuation_idx_to_boolean_equivalence_class: Dict[int, List[int]] = defaultdict(list)

    for f_idx, feature in enumerate(features):
        valuation: List[int] = []
        for gfa_state in iteration_data.gfa_states:
            instance_idx = gfa_state.get_faithful_abstraction_index()
            instance_data = preprocessing_data.instance_datas[instance_idx]
            dlplan_source_ss_state = state_finder.get_dlplan_ss_state(gfa_state)
            valuation.append(int(feature.dlplan_feature.evaluate(dlplan_source_ss_state, instance_data.denotations_caches)))

        boolean_valuation: Tuple[bool] = tuple([value > 0 for value in valuation])
        boolean_valuation_idx = boolean_valuation_to_boolean_valuation_idx.get(boolean_valuation)
        if boolean_valuation_idx is None:
            boolean_valuation_idx = len(boolean_valuation_idx_to_boolean_valuation)
            boolean_valuation_idx_to_boolean_valuation.append(boolean_valuation)
            boolean_valuation_to_boolean_valuation_idx[boolean_valuation] = boolean_valuation_idx
        boolean_valuation_idx_to_boolean_equivalence_class[boolean_valuation_idx].append(f_idx)

    logging.info(f"{len(features)} feature(s), {len(boolean_valuation_idx_to_boolean_valuation)} Boolean equivalence class(es)")

    # Construct maps from f_idx to boolean valuation indices
    f_idx_to_boolean_valuation_idx: Dict[int, int] = dict()
    for boolean_valuation_idx, equivalence_class in boolean_valuation_idx_to_boolean_equivalence_class.items():
        for f_idx in equivalence_class:
            f_idx_to_boolean_valuation_idx[f_idx] = boolean_valuation_idx

    logging.info(f"Boolean equivalence classes for additional features:")
    for f_name, f_idx in additional_features.items():
        if f_idx != -1:
            equivalence_class = boolean_valuation_idx_to_boolean_equivalence_class.get(f_idx_to_boolean_valuation_idx.get(f_idx))
            logging.info(f"  f{f_idx} -> [{', '.join(['f' + str(i) for i in equivalence_class])}]")
            if len(equivalence_class) > 1:
                for f_idx2 in equivalence_class:
                    feature = features[f_idx2]
                    logging.info(f"  - f{f_idx2}.{str(feature._dlplan_feature)} / {feature.complexity}")

    # PRUNE features that do have same feature change along all state pairs AND the same boolean valuation
    # Last condition added by Blai
    logging.info(f"Pruning features that do have same feature change along all state pairs AND the same boolean valuation...")
    selected_features: List[Tuple[int, Feature]] = prune_features_with_same_feature_change_AND_boolean_valuation(preprocessing_data,
                                                                                                                 iteration_data.gfa_states,
                                                                                                                 gfa_state_id_to_tuple_graph,
                                                                                                                 state_finder,
                                                                                                                 features,
                                                                                                                 f_idx_to_boolean_valuation_idx,
                                                                                                                 additional_features)

    num_pruned = len(features) - len(selected_features)
    logging.info(f"{len(selected_features)} feature(s) after pruning features with same change across transitions AND same boolean valuation (num_pruned={num_pruned})")
    features = [feature for _, feature in selected_features]

    if analyze_features is not None:
        logging.info(f"Analysis 5: Final features")
        _analyze_features(analyze_features, features)

    return features


def prune_features_with_same_feature_change_AND_boolean_valuation(preprocessing_data: PreprocessingData,
                                                                  gfa_states: List[mm.GlobalFaithfulAbstractState],
                                                                  gfa_state_id_to_tuple_graph: Dict[int, mm.TupleGraph],
                                                                  state_finder: StateFinder,
                                                                  features: List[Feature],
                                                                  f_idx_to_boolean_valuation_idx: Optional[Dict[int, int]] = None,
                                                                  additional_features: Optional[Dict[str, int]] = None) -> List[Tuple[int, Feature]]:
    # Local denotations caches (saves a lot of memory)
    denotations_caches: dlplan_core.DenotationsCaches = dlplan_core.DenotationsCaches()

    if f_idx_to_boolean_valuation_idx is None:
        # Construct equivalence classes for features modulo boolean valuations across gfa states
        logging.info(f"Construct equivalece classes for features modulo boolean valuations across gfa states...")
        boolean_valuation_idx_to_boolean_valuation: List[Tuple[bool]] = []
        boolean_valuation_to_boolean_valuation_idx: Dict[Tuple[bool], int] = dict()
        boolean_valuation_idx_to_boolean_equivalence_class: Dict[int, List[int]] = defaultdict(list)

        for f_idx, feature in enumerate(features):
            valuation: List[int] = []
            for gfa_state in gfa_states:
                instance_idx = gfa_state.get_faithful_abstraction_index()
                instance_data = preprocessing_data.instance_datas[instance_idx]
                dlplan_source_ss_state = state_finder.get_dlplan_ss_state(gfa_state)
                #valuation.append(int(feature.dlplan_feature.evaluate(dlplan_source_ss_state, instance_data.denotations_caches)))
                valuation.append(int(feature.dlplan_feature.evaluate(dlplan_source_ss_state, denotations_caches)))

            boolean_valuation: Tuple[bool] = tuple([value > 0 for value in valuation])
            boolean_valuation_idx = boolean_valuation_to_boolean_valuation_idx.get(boolean_valuation)
            if boolean_valuation_idx is None:
                boolean_valuation_idx = len(boolean_valuation_idx_to_boolean_valuation)
                boolean_valuation_idx_to_boolean_valuation.append(boolean_valuation)
                boolean_valuation_to_boolean_valuation_idx[boolean_valuation] = boolean_valuation_idx
            boolean_valuation_idx_to_boolean_equivalence_class[boolean_valuation_idx].append(f_idx)

        logging.info(f"{len(features)} feature(s), {len(boolean_valuation_idx_to_boolean_valuation)} Boolean equivalence class(es)")

        # Construct maps from f_idx to boolean valuation indices
        f_idx_to_boolean_valuation_idx: Dict[int, int] = dict()
        for boolean_valuation_idx, equivalence_class in boolean_valuation_idx_to_boolean_equivalence_class.items():
            for f_idx in equivalence_class:
                f_idx_to_boolean_valuation_idx[f_idx] = boolean_valuation_idx


    changes_to_f_idxs: Dict[Tuple[Any], List[int]] = defaultdict(list)
    for f_idx, feature in enumerate(features):
        changes: List[Any] = []
        for gfa_state in gfa_states:
            instance_idx = gfa_state.get_faithful_abstraction_index()
            instance_data = preprocessing_data.instance_datas[instance_idx]

            if instance_data.gfa.is_deadend_state(gfa_state.get_faithful_abstract_state_index()):
                continue

            tuple_graph = gfa_state_id_to_tuple_graph[gfa_state.get_global_index()]
            tuple_graph_vertices_by_distance = tuple_graph.get_vertices_grouped_by_distance()

            dlplan_source_ss_state = state_finder.get_dlplan_ss_state(gfa_state)
            #source_val = int(feature.dlplan_feature.evaluate(dlplan_source_ss_state, instance_data.denotations_caches))
            source_val = int(feature.dlplan_feature.evaluate(dlplan_source_ss_state, denotations_caches))

            for tuple_vertex_group in tuple_graph_vertices_by_distance:
                for tuple_vertex in tuple_vertex_group:
                    for mimir_ss_state_prime in tuple_vertex.get_states():
                        gfa_state_prime = state_finder.get_gfa_state_from_ss_state_idx(instance_idx, instance_data.mimir_ss.get_state_index(mimir_ss_state_prime))
                        dlplan_target_ss_state = state_finder.get_dlplan_ss_state(gfa_state_prime)
                        instance_prime_idx = gfa_state_prime.get_faithful_abstraction_index()
                        instance_data_prime = preprocessing_data.instance_datas[instance_prime_idx]
                        #target_val = int(feature.dlplan_feature.evaluate(dlplan_target_ss_state, instance_data_prime.denotations_caches))
                        target_val = int(feature.dlplan_feature.evaluate(dlplan_target_ss_state, denotations_caches))
                        if source_val < target_val:
                            changes.append(FeatureChange.UP)
                        elif source_val > target_val:
                            changes.append(FeatureChange.DOWN)
                        else:
                            changes.append(FeatureChange.BOT)
        changes: Tuple[Any] = tuple(changes)

        existing_f_idxs: List[int] = changes_to_f_idxs.get(changes, [])
        boolean_equivalent: List[int] = [f_idx2 for f_idx2 in existing_f_idxs if f_idx_to_boolean_valuation_idx[f_idx] == f_idx_to_boolean_valuation_idx[f_idx2]]
        if len(boolean_equivalent) > 0:
            assert len(boolean_equivalent) == 1
            existing_f_idx = boolean_equivalent[0]
            existing_feature = features[existing_f_idx]
            # Either prune f_idx or replace the equivalent one with f_idx
            if feature.complexity < existing_feature.complexity:
                if additional_features is not None and str(existing_feature._dlplan_feature) in additional_features:
                    logging.info(f"FEATURE {existing_f_idx}.{existing_feature._dlplan_feature}/{existing_feature.complexity} PRUNED-BY {f_idx}.{feature._dlplan_feature}/{feature.complexity}")
                changes_to_f_idxs[changes] = [f_idx] + [f_idx2 for f_idx2 in existing_f_idxs if f_idx2 != existing_f_idx]
            elif additional_features is not None and str(feature._dlplan_feature) in additional_features:
                logging.info(f"FEATURE {f_idx}.{feature._dlplan_feature}/{feature.complexity} PRUNED-BY {existing_f_idx}.{existing_feature._dlplan_feature}/{existing_feature.complexity}")
        else:
            # This is a new change, add it
            changes_to_f_idxs[changes].append(f_idx)

    selected_features: List[Tuple[int, Feature]] = [(f_idx, features[f_idx]) for f_idxs in changes_to_f_idxs.values() for f_idx in f_idxs]
    return selected_features


def prune_features_with_same_feature_change_AND_boolean_valuation_reduced_v1(state_factory: StateFactory,
                                                                             dlplan_states: List[dlplan_core.State],
                                                                             instance_idx_to_denotations_caches: Dict[int, dlplan_core.DenotationsCaches],
                                                                             features: List[Feature]) -> List[Tuple[int, Feature]]:
    # Construct equivalence classes for features modulo boolean valuations across states
    logging.info(f"Construct equivalece classes for features modulo boolean valuations across states...")
    boolean_valuation_idx_to_boolean_valuation: List[Tuple[bool]] = []
    boolean_valuation_to_boolean_valuation_idx: Dict[Tuple[bool], int] = dict()
    boolean_valuation_idx_to_boolean_equivalence_class: Dict[int, List[int]] = defaultdict(list)

    bloom: BloomFilter = BloomFilter(max_elements=len(features), error_rate=0.1)
    logging.info(f"Bloom filter for constructing f_idx_to_boolean_valuation_idx: {bloom}")

    for f_idx, feature in enumerate(features):
        valuation: List[int] = [int(feature.dlplan_feature.evaluate(dlplan_state, instance_idx_to_denotations_caches.get(dlplan_state.get_instance_info().get_index()))) for dlplan_state in dlplan_states]
        boolean_valuation: Tuple[bool] = tuple([value > 0 for value in valuation])

        if boolean_valuation in bloom:
            boolean_valuation_idx = boolean_valuation_to_boolean_valuation_idx.get(boolean_valuation)
            if boolean_valuation_idx is None:
                boolean_valuation_idx = len(boolean_valuation_idx_to_boolean_valuation)
                boolean_valuation_idx_to_boolean_valuation.append(boolean_valuation)
                boolean_valuation_to_boolean_valuation_idx[boolean_valuation] = boolean_valuation_idx
            boolean_valuation_idx_to_boolean_equivalence_class[boolean_valuation_idx].append(f_idx)
        else:
            boolean_valuation_idx = len(boolean_valuation_idx_to_boolean_valuation)
            boolean_valuation_idx_to_boolean_valuation.append(boolean_valuation)
            boolean_valuation_to_boolean_valuation_idx[boolean_valuation] = boolean_valuation_idx
            boolean_valuation_idx_to_boolean_equivalence_class[boolean_valuation_idx].append(f_idx)
            bloom.add(boolean_valuation)
    logging.info(f"{len(features)} feature(s), {len(boolean_valuation_idx_to_boolean_valuation)} Boolean equivalence class(es)")

    # Construct maps from f_idx to boolean valuation indices
    f_idx_to_boolean_valuation_idx: Dict[int, int] = dict()
    for boolean_valuation_idx, equivalence_class in boolean_valuation_idx_to_boolean_equivalence_class.items():
        for f_idx in equivalence_class:
            f_idx_to_boolean_valuation_idx[f_idx] = boolean_valuation_idx

    # Extract non-redundant features
    bloom: BloomFilter = BloomFilter(max_elements=len(features), error_rate=0.1)
    logging.info(f"Bloom filter for pruning features: {bloom}")

    logging.warning(f"[{Path(__file__).name}] *** Order of iteration should be swapped, first do the expansion, then iterate over f_idxs")
    changes_to_f_idxs: Dict[Tuple[Any], List[int]] = defaultdict(list)
    for f_idx, feature in enumerate(features):
        changes: List[int] = []
        for dlplan_state in dlplan_states:
            instance_idx: int = dlplan_state.get_instance_info().get_index()
            denotations_caches: dlplan_core.DenotationsCaches = instance_idx_to_denotations_caches.get(instance_idx)
            state_idx: int = dlplan_state.get_index()

            instance_data: PDDLInstance = state_factory.get_instance(instance_idx)
            if instance_data.is_deadend_state(state_idx):
                continue

            # ASSUMPTION: Successors are always ordered in the same way
            source_val: int = int(feature.dlplan_feature.evaluate(dlplan_state, denotations_caches))
            for (succ_state_idx, succ_state), action in instance_data.get_successors(state_idx):
                succ_dlplan_state: dlplan_core.State = instance_data.get_dlplan_state(succ_state_idx, succ_state)
                target_val: int = int(feature.dlplan_feature.evaluate(succ_dlplan_state, denotations_caches))
                if source_val < target_val:
                    changes.append(FeatureChange.UP.value)
                elif source_val > target_val:
                    changes.append(FeatureChange.DOWN.value)
                else:
                    changes.append(FeatureChange.BOT.value)
        changes: Tuple[int] = tuple(changes)

        if changes in bloom:
            existing_f_idxs: List[int] = changes_to_f_idxs.get(changes, [])
            boolean_equivalent: List[int] = [f_idx2 for f_idx2 in existing_f_idxs if f_idx_to_boolean_valuation_idx[f_idx] == f_idx_to_boolean_valuation_idx[f_idx2]]
            if len(boolean_equivalent) > 0:
                assert len(boolean_equivalent) == 1
                existing_f_idx = boolean_equivalent[0]
                existing_feature = features[existing_f_idx]
                # Either prune f_idx or replace the equivalent one with f_idx
                if feature.complexity < existing_feature.complexity:
                    changes_to_f_idxs[changes] = [f_idx] + [f_idx2 for f_idx2 in existing_f_idxs if f_idx2 != existing_f_idx]
            else:
                # This is a new change, add it
                changes_to_f_idxs[changes].append(f_idx)
        else:
            # This is a new change, add it
            changes_to_f_idxs[changes].append(f_idx)
            bloom.add(changes)

    selected_features: List[Tuple[int, Feature]] = [(f_idx, features[f_idx]) for f_idxs in changes_to_f_idxs.values() for f_idx in f_idxs]
    return selected_features


def prune_features_with_same_feature_change_AND_boolean_valuation_reduced_v2(state_factory: StateFactory,
                                                                             dlplan_state_pairs: List[Tuple[dlplan_core.State, dlplan_core.State]],
                                                                             instance_idx_to_denotations_caches: Dict[int, dlplan_core.DenotationsCaches],
                                                                             features: List[Feature],
                                                                             remove_features_with_constant_bvalue: bool = False) -> List[Tuple[int, Feature]]:
    # Local denotations caches (saves a lot of memory)
    denotations_caches: dlplan_core.DenotationsCaches = dlplan_core.DenotationsCaches()

    # List of unique dlplan states
    unique_dlplan_states: List[dlplan_core.State] = []
    set_unique_dlplan_states: Set[dlplan_core.State] = set()
    for dlplan_state_pair in dlplan_state_pairs:
        for dlplan_state in dlplan_state_pair:
            if dlplan_state not in set_unique_dlplan_states:
                unique_dlplan_states.append(dlplan_state)
                set_unique_dlplan_states.add(dlplan_state)

    # Construct equivalence classes for features modulo boolean valuations across states
    logging.info(f"Construct equivalece classes for features modulo boolean valuations across states...")
    boolean_valuation_idx_to_boolean_valuation: List[Tuple[bool]] = []
    boolean_valuation_to_boolean_valuation_idx: Dict[Tuple[bool], int] = dict()
    boolean_valuation_idx_to_boolean_equivalence_class: Dict[int, List[int]] = defaultdict(list)

    bloom: BloomFilter = BloomFilter(max_elements=len(features), error_rate=0.1)
    logging.info(f"Bloom filter for constructing f_idx_to_boolean_valuation_idx: {bloom}")

    for f_idx, feature in enumerate(features):
        #valuation: List[int] = [int(feature.dlplan_feature.evaluate(dlplan_state, instance_idx_to_denotations_caches.get(dlplan_state.get_instance_info().get_index()))) for dlplan_state in unique_dlplan_states]
        valuation: List[int] = [int(feature.dlplan_feature.evaluate(dlplan_state, denotations_caches)) for dlplan_state in unique_dlplan_states]
        boolean_valuation: Tuple[bool] = tuple([value > 0 for value in valuation])

        if boolean_valuation in bloom:
            boolean_valuation_idx = boolean_valuation_to_boolean_valuation_idx.get(boolean_valuation)
            if boolean_valuation_idx is None:
                boolean_valuation_idx = len(boolean_valuation_idx_to_boolean_valuation)
                boolean_valuation_idx_to_boolean_valuation.append(boolean_valuation)
                boolean_valuation_to_boolean_valuation_idx[boolean_valuation] = boolean_valuation_idx
            boolean_valuation_idx_to_boolean_equivalence_class[boolean_valuation_idx].append(f_idx)
        else:
            boolean_valuation_idx = len(boolean_valuation_idx_to_boolean_valuation)
            boolean_valuation_idx_to_boolean_valuation.append(boolean_valuation)
            boolean_valuation_to_boolean_valuation_idx[boolean_valuation] = boolean_valuation_idx
            boolean_valuation_idx_to_boolean_equivalence_class[boolean_valuation_idx].append(f_idx)
            bloom.add(boolean_valuation)
    logging.info(f"{len(features)} feature(s), {len(boolean_valuation_idx_to_boolean_valuation)} Boolean equivalence class(es)")

    # Construct maps from f_idx to boolean valuation indices
    f_idx_to_boolean_valuation_idx: Dict[int, int] = dict()
    for boolean_valuation_idx, equivalence_class in boolean_valuation_idx_to_boolean_equivalence_class.items():
        for f_idx in equivalence_class:
            f_idx_to_boolean_valuation_idx[f_idx] = boolean_valuation_idx

    # Remove features with constant boolean value (if applicable)
    simplified_features: List[Tuple[int, Feature]] = None
    if remove_features_with_constant_bvalue:
        simplified_features: List[Tuple[int, Feature]] = []
        for f_idx, feature in enumerate(features):
            boolean_valuation: Tuple[bool] = boolean_valuation_idx_to_boolean_valuation[f_idx_to_boolean_valuation_idx[f_idx]]
            if len(set(boolean_valuation)) == 2:
                simplified_features.append((f_idx, feature))
        logging.info(f"{len(simplified_features)} feature(s) after removal of features with constant boolean value (num_pruned={len(features) - len(simplified_features)})")
    else:
        simplified_features: List[Tuple[int, Feature]] = list(enumerate(features))

    # Extract non-redundant features
    bloom: BloomFilter = BloomFilter(max_elements=len(simplified_features), error_rate=0.1)
    logging.info(f"Bloom filter for pruning features: {bloom}")

    """
    # Compute valuations
    dlplan_state_to_valuation: Dict[dlplan_state.State, Tuple[int]] = dict()
    for dlplan_state in unique_dlplan_states:
        instance_idx: int = dlplan_state.get_instance_info().get_index()
        denotations_caches: dlplan_core.DenotationsCaches = instance_idx_to_denotations_caches.get(instance_idx)
        valuation: List[int] = [int(feature.dlplan_feature.evaluate(dlplan_state, denotations_caches)) for _, feature in simplified_features]
        dlplan_state_to_valuation[dlplan_state] = tuple(valuation)
    """

    # Calcualte changes for features over pairs
    #changes_to_f_idxs: Dict[Tuple[Any], List[int]] = defaultdict(list)
    changes_to_indices: Dict[Tuple[Any], List[int]] = defaultdict(list)
    for idx, (f_idx, feature) in enumerate(simplified_features):
        changes: List[int] = []
        for dlplan_state_pair in dlplan_state_pairs:
            instance_idx: int = dlplan_state_pair[0].get_instance_info().get_index()
            assert instance_idx == dlplan_state_pair[1].get_instance_info().get_index()
            #denotations_caches: dlplan_core.DenotationsCaches = instance_idx_to_denotations_caches.get(instance_idx)

            #values: List[int, int] = [dlplan_state_to_valuation[dlplan_state][f_idx] for dlplan_state in dlplan_state_pair]
            values: List[int, int] = [int(feature.dlplan_feature.evaluate(dlplan_state, denotations_caches)) for dlplan_state in dlplan_state_pair]
            if values[0] < values[1]:
                changes.append(FeatureChange.UP.value)
            elif values[0] > values[1]:
                changes.append(FeatureChange.DOWN.value)
            else:
                changes.append(FeatureChange.BOT.value)
        changes: Tuple[int] = tuple(changes)

        if changes in bloom:
            #existing_f_idxs: List[int] = changes_to_f_idxs.get(changes, [])
            #boolean_equivalent: List[int] = [f_idx2 for f_idx2 in existing_f_idxs if f_idx_to_boolean_valuation_idx[f_idx] == f_idx_to_boolean_valuation_idx[f_idx2]]
            existing_indices: List[int] = changes_to_indices.get(changes, [])
            boolean_equivalent: List[int] = [idx2 for idx2 in existing_indices if f_idx_to_boolean_valuation_idx[f_idx] == f_idx_to_boolean_valuation_idx[simplified_features[idx2][0]]]
            if len(boolean_equivalent) > 0:
                assert len(boolean_equivalent) == 1
                #existing_f_idx: int = boolean_equivalent[0]
                #existing_feature: Feature = features[existing_f_idx]
                existing_idx: int = boolean_equivalent[0]
                existing_feature: Feature = simplified_features[existing_idx][1]
                # Either prune idx or replace the equivalent one with idx
                if feature.complexity < existing_feature.complexity:
                    #changes_to_f_idxs[changes] = [f_idx] + [f_idx2 for f_idx2 in existing_f_idxs if f_idx2 != existing_f_idx]
                    changes_to_indices[changes] = [idx] + [idx2 for idx2 in existing_indices if idx2 != existing_idx]
            else:
                # This is a new change, add it
                #changes_to_f_idxs[changes].append(f_idx)
                changes_to_indices[changes].append(idx)
        else:
            # This is a new change, add it
            #changes_to_f_idxs[changes].append(f_idx)
            changes_to_indices[changes].append(idx)
            bloom.add(changes)

    #surviving_features: List[Tuple[int, Feature]] = [(f_idx, features[f_idx]) for f_idxs in changes_to_f_idxs.values() for f_idx in f_idxs]
    surviving_features: List[Tuple[int, Feature]] = [simplified_features[idx] for indices in changes_to_indices.values() for idx in indices]
    return surviving_features


def _max_depth(item: str, recursion_depth: int = 0) -> int:
    logging.debug(f"max_depth: {recursion_depth * ' '}item={item}")
    if item[0] == "(":
        logging.debug(f"max_depth: {recursion_depth * ' '}item={item}")
        # Item is a list of items
        assert item[-1] == ")", item
        elements = []
        index, nesting = 0, 0
        for i, c in enumerate(item[1:-1]):
            if c == "(":
                nesting += 1
            elif c == ")":
                nesting -= 1
                assert nesting >= 0, item
            elif c == "," and nesting == 0:
                elements.append(item[1+index:1+i])
                index = 1+i
        assert item[index] in ["(", ","], (item, index)
        elements.append(item[1+index:-1])
        max_depth = max([_max_depth(element) for element in elements])
    elif item in ["c_top", "c_bot"]:
        # Basic concepts
        logging.debug(f"max_depth: {recursion_depth * ' '}item={item}, max_depth=0")
        max_depth = 0
    elif item.startswith("c_primitive") or item.startswith("r_primitive") or item.startswith("c_one_of"):
        # Item is a primitive concept or role
        logging.debug(f"max_depth: {recursion_depth * ' '}item={item}, max_depth=1")
        max_depth = 0
    elif item.startswith("b_nullary"):
        return 1
    elif item[:2] in ["n_", "b_"]:
        # Item is a feature
        assert item[-1] == ")", item
        leftpar_idx = item.index("(")
        max_depth = _max_depth(item[leftpar_idx:])
        logging.debug(f"max_depth: {recursion_depth * ' '}item={item}, max_depth={max_depth}")
    else:
        # Item is a non-primitive concept or role
        assert item[:2] in ["n_", "b_", "c_", "r_"], item
        assert item[-1] == ")", item
        leftpar_idx = item.index("(")
        max_depth = 1 + _max_depth(item[leftpar_idx:])
        logging.debug(f"max_depth: {recursion_depth * ' '}item={item}, max_depth={max_depth}")
    return max_depth


def _analyze_concepts_and_roles(domain: str, concepts, roles):
    concept_counter = Counter([str(concept) for concept in concepts])
    role_counter = Counter([str(role) for role in roles])
    assert concept_counter.most_common()[0][1] == 1 and role_counter.most_common()[0][1] == 1

    relevant_concepts: Dict[str, List[Tuple[str,str]]] = {
        "delivery": [
            ("c_some(r_inverse(r_primitive(at,0,1)),c_primitive(truck,0))", "concept-1"),
            ("c_some(r_inverse(r_primitive(at,0,1)),c_primitive(package,0))", "concept-1"),
            ("c_some(r_inverse(r_primitive(at_g,0,1)),c_top))", "concept-1"),
            ("c_and(c_some(r_inverse(r_primitive(at,0,1)),c_primitive(package,0)),c_not(c_some(r_inverse(r_primitive(at_g,0,1)),c_top)))", "concept-1"),
            ("c_and(c_not(c_some(r_inverse(r_primitive(at_g,0,1)),c_top)),c_some(r_inverse(r_primitive(at,0,1)),c_primitive(package,0)))", "concept-1"),
            ("c_not(c_some(r_inverse(r_primitive(at_g,0,1)),c_top))", "concept-1"),
            ("c_and(c_primitive(package,0),c_not(c_equal(r_primitive(at,0,1),r_primitive(at_g,0,1))))", "concept-1"),
            ("c_and(c_not(c_equal(r_primitive(at,0,1),r_primitive(at_g,0,1))),c_primitive(package,0))", "concept-1"),
            ("c_not(c_equal(r_primitive(at,0,1),r_primitive(at_g,0,1)))", "concept-1"),
            ("c_equal(r_primitive(at,0,1),r_primitive(at_g,0,1))", "concept-1"),
            ("c_and(c_all(r_inverse(r_primitive(at_g,0,1)),c_bot),c_some(r_inverse(r_primitive(at,0,1)),c_primitive(package,0)))", "AAAI-19: Cup"),
            ("c_and(c_some(r_inverse(r_primitive(at,0,1)),c_primitive(package,0)),c_all(r_inverse(r_primitive(at_g,0,1)),c_bot))", "AAAI-19: Cup"),
            ("c_some(r_inverse(r_primitive(at,0,1)),c_primitive(truck,0))", "AAAI-19: Ct"),
        ],
        "delivery-AAAI19": [
            ("c_all(r_inverse(r_primitive(at_g,0,1)),c_bot)", "AAAI-19: Cup: component"),
            ("c_some(r_inverse(r_primitive(at,0,1)),c_primitive(package,0))", "AAAI-19: Cup: component"),
            ("c_and(c_all(r_inverse(r_primitive(at_g,0,1)),c_bot),c_some(r_inverse(r_primitive(at,0,1)),c_primitive(package,0)))", "AAAI-19: Cup"),
            ("c_and(c_some(r_inverse(r_primitive(at,0,1)),c_primitive(package,0)),c_all(r_inverse(r_primitive(at_g,0,1)),c_bot))", "AAAI-19: Cup"),
            ("c_some(r_inverse(r_primitive(at,0,1)),c_primitive(truck,0))", "AAAI-19: Ct"),
        ],
        "blocks_4_clear": [
            ("c_one_of(b1)", "concept-1"),
            ("c_primitive(on-table,0)", "concept-1"),
            ("c_some(r_transitive_closure(r_primitive(on,0,1)),c_one_of(b1)))", "concept-1"),
        ],
        "logistics-1truck": [
            #("c_primitive(obj,0)", "objects"),
            #("c_primitive(truck,0)", "trucks"),
            ("c_equal(r_primitive(at,0,1),r_primitive(at_g,0,1))", "Goal counter"),
            ("c_some(r_primitive(in,0,1),c_primitive(truck,0))", "Objects in trucks"),
            ("c_some(r_inverse(r_primitive(in,0,1)),c_primitive(obj,0))", "Vehicles that contain objects"),
            #("c_some(r_inverse(r_primitive(at_g,0,1)),c_some(r_primitive(in,0,1),c_primitive(truck,0)))", "Target location for objects in trucks"),
            ("c_some(r_inverse(r_primitive(at,0,1)),c_some(r_inverse(r_primitive(in,0,1)),c_primitive(obj,0)))", "Location of vehicles that contain objects"),
            ("c_and(c_primitive(truck,0),c_some(r_primitive(at,0,1),c_some(r_inverse(r_primitive(at_g,0,1)),c_some(r_primitive(in,0,1),c_primitive(truck,0)))))", "trucks at target locations of objects in trucks"),

            #("c_some(r_inverse(r_primitive(at,0,1)),c_some(r_inverse(r_primitive(in,0,1)),c_primitive(package,0)))", "target locations that contain a LOADED truck"),
            #("c_some(r_inverse(r_primitive(at,0,1)),c_some(r_inverse(r_primitive(in,0,1)),c_primitive(package,0)))", "target locations that contain a LOADED truck"),
            #("c_and(c_some(r_inverse(r_primitive(at,0,1)),c_some(r_inverse(r_primitive(in,0,1)),c_primitive(package,0))),c_some(r_inverse(r_primitive(at,0,1),c_primitive(package,0))))", "target locations that contain a LOADED truck"),
        ],
    }

    for name, description in relevant_concepts.get(domain, []):
        max_depth = _max_depth(name)
        if concept_counter[name] > 0:
            logging.info(colored(f"    FOUND: {description}: {name} / (?,{max_depth})", "blue"))
        else:
            logging.info(f"  MISSING: {description}: {name} / (?,{max_depth})")

    relevant_roles: Dict[str, List[Tuple[str,str]]] = {
        "delivery": [
            ("r_primitive(at,0,1)", "role-1"),
            ("r_primitive(at_g,0,1)", "role-1"),
            ("r_primitive(adjacent,0,1)", "role-1"),
            ("r_inverse(r_primitive(at,0,1))", "role-1"),
            ("r_inverse(r_primitive(at_g,0,1))", "role-1"),
        ],
        "delivery-AAAI19": [
            ("r_inverse(r_primitive(at,0,1))", "AAAI-19: Inverse AT"),
            ("r_inverse(r_primitive(at_g,0,1))", "AAAI-19: Inverse ATg"),
        ],
        "blocks_4_clear": [
            ("r_inverse(r_primitive(on,0,1))", "role-1"),
            ("r_transitive_closure(r_primitive(on,0,1))", "role-1"),
        ],
        "logistics-1truck": [
            #("r_primitive(in,0,1)", "primitive in(?obj1, ?obj2)"),
        ],
    }

    for name, description in relevant_roles.get(domain, []):
        max_depth = _max_depth(name)
        if role_counter[name] > 0:
            logging.info(colored(f"    FOUND: {description}: {name} / (?,{max_depth})", "blue"))
        else:
            logging.info(f"  MISSING: {description}: {name} / (?,{max_depth})")


def _analyze_features(domain: str, features: List[Feature]):
    feature_counter = Counter([str(feature._dlplan_feature) for feature in features])
    #assert feature_counter.most_common()[0][1] == 1, feature_counter.most_common()[0]

    relevant_features: Dict[str, List[Tuple[str, str]]] = {
        "delivery": [
            ("n_concept_distance(c_some(r_inverse(r_primitive(at,0,1)),c_primitive(truck,0)),r_primitive(adjacent,0,1),c_some(r_inverse(r_primitive(at_g,0,1)),c_top))", "Distance to target cell"),
            ("n_concept_distance(c_some(r_inverse(r_primitive(at,0,1)),c_primitive(truck,0)),r_primitive(adjacent,0,1),c_and(c_some(r_inverse(r_primitive(at,0,1)),c_primitive(package,0)),c_not(c_some(r_inverse(r_primitive(at_g,0,1)),c_top))))", "Distance to closest undelivered paackage"),
            ("n_concept_distance(c_some(r_inverse(r_primitive(at,0,1)),c_primitive(truck,0)),r_primitive(adjacent,0,1),c_and(c_not(c_some(r_inverse(r_primitive(at_g,0,1)),c_top)),c_some(r_inverse(r_primitive(at,0,1)),c_primitive(package,0))))", "Distance to closest undelivered paackage"),
            ("n_count(c_and(c_primitive(package,0),c_not(c_equal(r_primitive(at,0,1),r_primitive(at_g,0,1)))))", "Goal counter"),
            ("b_empty(c_primitive(empty,0))", "Empty truck"),
            ("n_count(c_primitive(empty,0))", "AAAI-19: Empty truck"),
            ("b_empty(c_primitive(empty,0))", "AAAI-19: Empty truck"),
            ("n_count(c_not(c_equal(r_primitive(at,0,1),r_primitive(at_g,0,1))))", "AAAI-19: Goal counter"),
            ("n_concept_distance(c_some(r_inverse(r_primitive(at,0,1)),c_primitive(truck,0)),r_primitive(adjacent,0,1),c_and(c_some(r_inverse(r_primitive(at,0,1)),c_primitive(package,0)),c_all(r_inverse(r_primitive(at_g,0,1)),c_bot)))", "AAAI-19: Du"),
            ("n_concept_distance(c_some(r_inverse(r_primitive(at,0,1)),c_primitive(truck,0)),r_primitive(adjacent,0,1),c_some(r_inverse(r_primitive(at_g,0,1)),c_top))", "AAAI-19: Dt"),
        ],
        "delivery-AAAI19": [
            ("n_count(c_primitive(empty,0))", "AAAI-19: Empty truck"),
            ("b_empty(c_primitive(empty,0))", "AAAI-19: Empty truck"),
            ("n_count(c_not(c_equal(r_primitive(at,0,1),r_primitive(at_g,0,1))))", "AAAI-19: Goal counter"),
            #("n_concept_distance(c_some(r_inverse(r_primitive(at,0,1)),c_primitive(truck,0)),r_primitive(adjacent,0,1),c_and(c_some(r_inverse(r_primitive(at,0,1)),c_primitive(package,0)),c_all(r_inverse(r_primitive(at_g,0,1)),c_bot)))", "AAAI-19: Du"),
            #("n_concept_distance(c_some(r_inverse(r_primitive(at,0,1)),c_primitive(truck,0)),r_primitive(adjacent,0,1),c_some(r_inverse(r_primitive(at_g,0,1)),c_top))", "AAAI-19: Dt"),
            ("n_concept_distance(c_some(r_inverse(r_primitive(at,0,1)),c_primitive(truck_u,0)),r_primitive(adjacent,0,1),c_and(c_some(r_inverse(r_primitive(at,0,1)),c_primitive(package_u,0)),c_all(r_inverse(r_primitive(at_g,0,1)),c_bot)))", "AAAI-19: Du"),
            ("n_concept_distance(c_some(r_inverse(r_primitive(at,0,1)),c_primitive(truck_u,0)),r_primitive(adjacent,0,1),c_some(r_inverse(r_primitive(at_g,0,1)),c_top))", "AAAI-19: Dt"),
        ],
        "blocks_4_clear": [
            ("n_count(c_some(r_transitive_closure(r_primitive(on,0,1)),c_one_of(b1)))", "Number of blocks above b1"),
            ("n_concept_distance(c_one_of(b1),r_inverse(r_primitive(on,0,1)),c_primitive(clear,0))", "Number of blocks above b1"),
            ("n_count(c_primitive(on-table,0))", "Number of blocks on the table"),
        ],
        "logistics-1truck": [
            ("n_count(c_and(c_primitive(obj,0),c_not(c_equal(r_primitive(at,0,1),r_primitive(at_g,0,1)))))", "Goal counter (?,?)"),
            ("n_count(c_and(c_primitive(truck,0),c_some(r_primitive(at,0,1),c_some(r_inverse(r_primitive(at_g,0,1)),c_some(r_primitive(in,0,1),c_primitive(truck,0))))))", "Trucks at target locations of objects in trucks (?,?)"),
            ("n_count(c_and(c_primitive(truck,0),c_some(r_primitive(at,0,1),c_some(r_inverse(r_primitive(at,0,1)),c_and(c_primitive(obj,0),c_not(c_equal(r_primitive(at,0,1),r_primitive(at_g,0,1))))))))", "Trucks at locations of objects not in target locations (?,?)"),
            ("n_count(c_and(c_some(r_primitive(in,0,1),c_primitive(truck,0)),c_some(r_primitive(at_g,0,1),c_not(c_some(r_inverse(r_primitive(at,0,1)),c_primitive(truck,0))))))", "Packages in trucks whose target location is not location of any truck (?,?)"),
            ("n_count(c_and(c_some(r_primitive(at_g,0,1),c_not(c_some(r_inverse(r_primitive(at,0,1)),c_primitive(truck,0)))),c_some(r_primitive(in,0,1),c_primitive(truck,0))))", "Packages in trucks whose target location is not location of any truck (?,?)"),
        ],
        "logistics-1truck-indexicals": [
            ("n_count(c_and(c_primitive(obj,0),c_not(c_equal(r_primitive(at,0,1),r_primitive(at_g,0,1)))))", "Goal counter (8,3)"),
            ("n_count(c_and(c_not(c_equal(r_primitive(at,0,1),r_primitive(at_g,0,1))),c_primitive(obj,0)))", "Goal counter (8,3)"),
            ("n_count(c_primitive(marked,0))", "Marked packages (3,0)"),
            ("n_count(c_primitive(markable,0))", "Markable packages (3,0)"),
            ("n_count(c_and(c_primitive(truck,0),c_some(r_inverse(r_primitive(in,0,1)),c_primitive(marked,0))))", "Trucks that contain marked objects (6,2)"),
            ("n_count(c_and(c_some(r_inverse(r_primitive(in,0,1)),c_primitive(marked,0)),c_primitive(truck,0)))", "Trucks that contain marked objects (6,2)"),
            ("n_count(c_and(c_primitive(truck,0),c_some(r_primitive(at,0,1),c_some(r_inverse(r_primitive(at,0,1)),c_primitive(marked,0)))))", "Trucks at CURRENT location of marked objects (10,4)"),
            ("n_count(c_and(c_some(r_primitive(at,0,1),c_some(r_inverse(r_primitive(at,0,1)),c_primitive(marked,0))),c_primitive(truck,0)))", "Trucks at CURRENT location of marked objects (10,4)"),
            ("n_count(c_and(c_primitive(truck,0),c_some(r_primitive(at,0,1),c_some(r_inverse(r_primitive(at_g,0,1)),c_primitive(marked,0)))))", "Trucks at TARGET location of marked objects (10,4)"),
            ("n_count(c_and(c_some(r_primitive(at,0,1),c_some(r_inverse(r_primitive(at_g,0,1)),c_primitive(marked,0))),c_primitive(truck,0)))", "Trucks at TARGET location of marked objects (10,4)"),
        ],
    }

    for name, description in relevant_features.get(domain, []):
        max_depth = _max_depth(name)
        if feature_counter[name] > 0:
            f_idxs = [f_idx for f_idx, feature in enumerate(features) if str(feature._dlplan_feature) == name]
            assert len(f_idxs) > 0
            f_idx = f_idxs[0]
            complexity = features[f_idx].complexity
            logging.info(colored(f"    FOUND: {description}: {f_idx}.{name} / ({complexity},{max_depth})", "blue"))
        else:
            logging.info(f"  MISSING: {description}: {name} / (?,{max_depth})")
