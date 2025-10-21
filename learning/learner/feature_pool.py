import logging
from termcolor import colored
from pathlib import Path
from typing import Set, List, Tuple, MutableSet, Dict, Optional, Any, Generator

import dlplan.core as dlplan_core

from .benchmark import Benchmark
from .src.util import Timer
from .src.iteration import generate_features, post_process_features, read_features_from_repositories, write_features_to_repository, find_feature_repositories
from .src.iteration import Statistics


class FeaturePool:
    def __init__(self, benchmark: Benchmark, timers: Statistics, **kwargs):
        self._benchmark: Benchmark = benchmark
        self._timers: Statistics = timers

        self._timers.resume("feature/pool")

        # Revise if there is a feature repository that match active instances and feature parameters.
        # If so, read the features in the repo instead of generating them. Else, generate the features
        # and store them.
        parameters: Dict[str, Any] = {
            "disable_feature_generation": kwargs.get("disable_feature_generation"),
            "generate_all_distance_features": kwargs.get("generate_all_distance_features"),
            "concept_complexity_limit": kwargs.get("concept_complexity_limit"),
            "role_complexity_limit": kwargs.get("role_complexity_limit"),
            "boolean_complexity_limit": kwargs.get("boolean_complexity_limit"),
            "count_numerical_complexity_limit": kwargs.get("count_numerical_complexity_limit"),
            "distance_numerical_complexity_limit": kwargs.get("distance_numerical_complexity_limit"),
            "feature_limit": kwargs.get("feature_limit"),
            "strict_gc2_features": kwargs.get("strict_gc2_features"),
            "planner": benchmark._planner,
        }

        disable_feature_repositories: bool = kwargs.get("disable_feature_repositories", False)
        all_repositories: bool = kwargs.get("all_repositories", False)
        flexible_repositories: bool = kwargs.get("flexible_repositories", False)
        store_features: bool = kwargs.get("store_features", False)
        repository_folder: Path = Path("feature_repositories")
        uuid_str: str = kwargs.get("uuid_str")

        instance_names: List[str] = sorted([instance_data.instance_filepath().name for instance_data in benchmark._instance_datas])

        if not disable_feature_repositories:
            repositories: List[Path] = find_feature_repositories(repository_folder, parameters, instance_names, all_repositories=all_repositories, flexible=flexible_repositories)
        else:
            repositories: List[Path] = None

        if repositories is not None and len(repositories) > 0:
            logging.info(colored(f"Found compatible feature repositories [{', '.join([repository.name for repository in repositories])}]", "blue"))
            pool, statistics = read_features_from_repositories(repositories,
                                                               self._benchmark._domain_data.syntactic_element_factory,
                                                               **kwargs)
        else:
            logging.info(colored("Generating features...", "blue"))
            pool, _, statistics = generate_features(self._benchmark._domain_data.syntactic_element_factory,
                                                    self._benchmark._dlplan_states,
                                                    self._benchmark._instance_idx_to_denotations_caches,
                                                    **kwargs)
        logging.info(f"Feature statistics: {statistics}")

        if store_features and (repositories is None or len(repositories) == 0):
            repository_name: str = kwargs.get("feature_repository_name") or f"repo_{uuid_str}.frepo"
            statistics.update({"uuid": uuid_str, "family": self._benchmark._family_name})
            repository: Path = repository_folder / repository_name
            write_features_to_repository(pool, parameters, statistics, instance_names, repository)
            logging.info(colored(f"{len(pool)} feature(s) written to '{repository}'", "blue"))

        # Post-processing of features (e.g., pruned by max depth)
        logging.info(colored("Post-processing features...", "blue"))
        self._pool = post_process_features(pool, **kwargs)
        self._statistics = statistics

        self._timers.stop("feature/pool")
        logging.info(colored(f"Got {len(pool)} Feature(s) in {self._timers.get_elapsed_sec('feature/pool'):.02f} second(s)", "blue"))

