import logging, sys
import tempfile
import uuid
from termcolor import colored
from typing import Set, List, Tuple, MutableSet, Dict, Optional, Any, Generator

import math, random
import numpy as np

from intbitset import intbitset
from collections import defaultdict
from pathlib import Path

import dlplan.core as dlplan_core

from .src.iteration import generate_features, post_process_features, read_features_from_repositories, write_features_to_repository, find_feature_repositories, compute_feature_valuations_for_dlplan_state
from .src.iteration import IterationData, Statistics, SketchReduced, D2sepDlplanPolicyFactory
from .src.util import Timer, create_experiment_workspace, change_working_directory, write_file, write_file_lines, change_dir, memory_usage, print_separation_line
from .src.preprocessing import InstanceData
from .src.state_space import StateFactory, PDDLInstance, get_plan_v2


class Benchmark:
    def __init__(self, domain_filepath: Path, problems_directory: Path, workspace: Path, timers: Statistics, **kwargs):
        self._domain_filepath: Path = domain_filepath
        self._problems_directory: Path = problems_directory
        self._workspace: Path = workspace
        self._timers: Statistics = timers
        self._planner: str = kwargs.get("planner")
        self._deadends: bool = kwargs.get("deadends")
        self._force_preprocessing: bool = kwargs.get("force_preprocessing", False)

        self._instance_filepaths = list(self._problems_directory.iterdir())
        create_experiment_workspace(self._workspace)
        change_working_directory(self._workspace)

        # Obtain family name from domain path
        self._family_name: str = str(self._domain_filepath.parent.name)

        # Initialize state factory
        logging.info(f"Read and curate training instances...")
        self._state_factory: StateFactory = StateFactory(self._family_name,
                                                         self._domain_filepath,
                                                         self._instance_filepaths,
                                                         self._planner,
                                                         self._deadends)
        self._domain_data: DomainData = self._state_factory.domain_data()

        # Preprocess instances
        idxs_to_be_removed: Set[int] = set(self.preprocess_instances())
        if kwargs.get("preprocess_only", False):
            logging.info(f"{len(self._state_factory._instances)} instance(s) preprocessed, {len(idxs_to_be_removed)} instance(s) to be removed. Done!")
            return

        if len(idxs_to_be_removed) > 0:
            logging.info(f"{len(idxs_to_be_removed)} instance(s) to be removed:")
            for instance_idx in idxs_to_be_removed:
                logging.info(f"  {instance_idx}.[{self.get_instance(instance_idx).instance_filepath()}]")

        max_num_instances: int = kwargs.get("max_num_instances", int(1e6))
        idxs_to_be_kept: List[int] = [instance_idx for instance_idx in range(len(self._state_factory._instances)) if instance_idx not in idxs_to_be_removed]
        sorted_idxs_to_be_kept: List[int] = sorted(idxs_to_be_kept, key=lambda idx: len(self._preprocessed_datas[idx].get("planner_output")[1]), reverse=True)
        sorted_idxs_to_be_kept = sorted_idxs_to_be_kept if max_num_instances is None else sorted_idxs_to_be_kept[:max_num_instances]

        # Additionally, prune instances whose plan revisits a state. For this, plan's state trajectory must be generated
        non_loopy_idxs_to_be_kept: List[int] = [instance_idx for instance_idx in sorted_idxs_to_be_kept if all(self._non_loopy_state_trajectories(instance_idx))]
        if len(non_loopy_idxs_to_be_kept) < len(sorted_idxs_to_be_kept):
            logging.info(f"Further removal of {len(sorted_idxs_to_be_kept) - len(non_loopy_idxs_to_be_kept)} loopy instance(s):")
            for instance_idx in sorted_idxs_to_be_kept:
                if instance_idx not in non_loopy_idxs_to_be_kept:
                    logging.info(f"  {instance_idx}.[{self.get_instance(instance_idx).instance_filepath()}]")

        # Remove and re-order instances
        previous_number_instances: int = len(self._state_factory._instances)
        self.remove_and_reorder_instances(non_loopy_idxs_to_be_kept)
        assert len(non_loopy_idxs_to_be_kept) == len(self._state_factory._instances), (len(non_loopy_idxs_to_be_kept), len(self._state_factory._instances))
        logging.info(f"{len(non_loopy_idxs_to_be_kept)} instance(s) after removal and re-ordering; {previous_number_instances - len(non_loopy_idxs_to_be_kept)} instance(s) removed")
        self.preprocess_instances_final()

        if len(self._state_factory._instances) == 0:
            logging.info(f"Zero instances remain. Terminating...")
            self._timers.print(title="Finalizing...", logger=True)
            raise RuntimeError("Zero instances remain. Terminating...")

        # Define instance collections
        self._instance_datas: List[PDDLInstance] = list(self._state_factory._instances)
        self._instance_idxs: List[int] = list(range(len(self._instance_datas)))
        self._instance_idx_to_denotations_caches: Dict[int, dlplan_core.DenotationsCaches] = {instance_idx: dlplan_core.DenotationsCaches() for instance_idx in self._instance_idxs}

        # Initial set of relevant vertices
        self._initial_relevant_vertices: Dict[int, Set[int]] = dict()
        for preprocessed_data in self._preprocessed_datas:
            assert not any(preprocessed_data.get("revisits_states"))
            instance_idx: int = preprocessed_data.get("instance_idx")
            marked_state_trajectories: List[Tuple[int]] = preprocessed_data.get("marked_state_trajectories")
            assert len(marked_state_trajectories) == 1, f"More than one marked path per instance isn't supported yet (instance_idx={instance_idx})"
            assert instance_idx not in self._initial_relevant_vertices
            self._initial_relevant_vertices[instance_idx] = set(marked_state_trajectories[0])
        logging.info(f"{sum([len(vertices) for vertices in self._initial_relevant_vertices.values()])} initial relevant vertice(s)")

        self._timers.resume("preprocessing")

        # Dlplan states
        self._dlplan_states: List[dlplan_core.State] = []
        self._ext_state_to_dlplan_state_index: Dict[Tuple[int, int], int] = dict()
        self._ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = dict()
        for instance_idx, vertices in self._initial_relevant_vertices.items():
            instance_data: PDDLInstace = self.get_instance(instance_idx)
            for state_idx in vertices:
                ext_state: Tuple[int, int] = (instance_idx, state_idx)
                if ext_state not in self._ext_state_to_dlplan_state_index:
                    dlplan_state_index: int = len(self._dlplan_states)
                    self._ext_state_to_dlplan_state_index[ext_state] = dlplan_state_index
                    state: Any = self.get_state(*ext_state)
                    dlplan_state: dlplan_core.State = instance_data.get_dlplan_state(state_idx, state)
                    self._dlplan_states.append(dlplan_state)
                    logging.debug(f"EXT_STATE_TO_DLPLAN_INDEX: {ext_state} -> {dlplan_state_index} -> {dlplan_state}")

        self._timers.stop("preprocessing")

    def preprocess_instances(self) -> List[int]:
        self._timers.resume("preprocessing")
        self._timers.resume("planner")

        # Create folder for storing plans (preprocessing)
        folder_for_plans: Path = Path(f"plans.{self._planner}")
        folder_for_plans.mkdir(exist_ok=True)

        # Read plans from existing files (if possible)
        self._preprocessed_datas: List[Dict[str, Any]] = []
        for instance_data in self._state_factory._instances:
            instance_filepath: Path = Path(instance_data.instance_filepath())
            plan_filepath: Path = folder_for_plans / f"{instance_filepath.name}.plan.{self._planner}"
            preprocessed_data: Dict[str, Any] = {
                "instance_idx": instance_data.idx,
                "instance_filepath": instance_filepath,
                "plan_filepath": plan_filepath,
            }
            if not self._force_preprocessing and plan_filepath.exists():
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
        if any([not preprocessed_data.get("complete", False) for preprocessed_dat in self._preprocessed_datas]):
            temp_folder = tempfile.TemporaryDirectory(prefix=f"{self._family_name}.preprocess_instances.", dir=".")
            logging.info(f"Preprocessing instances in folder '{temp_folder.name}'...")
            with change_dir(temp_folder.name):
                for preprocessed_data in self._preprocessed_datas:
                    instance_idx: int = preprocessed_data.get("instance_idx")
                    if not preprocessed_data.get("complete", False):
                        data: Dict[str, Any] = self._preprocess_instance(temp_folder.name, instance_idx, remove_files=False)
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

        self._timers.stop("preprocessing")
        self._timers.stop("planner")
        return idxs_to_be_removed

    def preprocess_instances_final(self):
        self._timers.resume("preprocessing")
        for instance_idx, preprocessed_data in enumerate(self._preprocessed_datas):
            assert instance_idx == preprocessed_data.get("instance_idx")
            self._preprocess_instance_final(preprocessed_data)
        self._timers.stop("preprocessing")

    def remove_and_reorder_instances(self, sorted_instance_idxs: List[int]):
        self._timers.resume("preprocessing")
        assert len(self._state_factory._instances) == len(self._preprocessed_datas)
        for instance_data, preprocessed_data in zip(self._state_factory._instances, self._preprocessed_datas):
            assert instance_data.idx == preprocessed_data.get("instance_idx")
            assert instance_data.instance_filepath() == preprocessed_data.get("instance_filepath")

        preprocessed_datas: List[Dict[str, Any]] = [self._preprocessed_datas[instance_idx] for instance_idx in sorted_instance_idxs]
        for instance_idx, preprocessed_data in enumerate(preprocessed_datas):
            preprocessed_data["instance_idx"] = instance_idx
        self._preprocessed_datas: List[Dict[str, Any]] = preprocessed_datas

        self._state_factory.remove_and_reorder_instances(sorted_instance_idxs)

        assert len(self._state_factory._instances) == len(self._preprocessed_datas)
        for instance_data, preprocessed_data in zip(self._state_factory._instances, self._preprocessed_datas):
            assert instance_data.idx == preprocessed_data.get("instance_idx")
            assert instance_data.instance_filepath() == preprocessed_data.get("instance_filepath")
        self._timers.stop("preprocessing")

    def get_instance(self, instance_idx: int) -> PDDLInstance:
        return self._state_factory.get_instance(instance_idx)

    def get_state(self, instance_idx: int, state_idx: int) -> Any:
        return self._state_factory.get_state(instance_idx, state_idx)

    def get_dlplan_state(self, instance_idx: int, state_idx: int) -> dlplan_core.State:
        return self._state_factory.get_dlplan_state(instance_idx, state_idx)

    def get_feature_valuations(self, instance_idx: int, state_idx: int, feature_pool: List[Any]) -> np.ndarray:
        dlplan_state: dlplan_core.State = self.get_dlplan_state(instance_idx, state_idx)
        feature_valuations: np.ndarray = compute_feature_valuations_for_dlplan_state(dlplan_state, feature_pool, self._instance_idx_to_denotations_caches)
        return feature_valuations

    def exploration(self, instance_idx: int, state_idx: int, width: int, caching: bool) -> intbitset:
        return self._state_factory.exploration(instance_idx, state_idx, width, caching)

    def _preprocess_instance(self, folder_name: str, instance_idx: int, remove_files: bool = True) -> Dict[str, Any]:
        instance_data: PDDLInstance = self._state_factory._instances[instance_idx]
        assert instance_data.idx == instance_idx

        domain_filepath: Path = instance_data.domain_filepath()
        instance_filepath: Path = instance_data.instance_filepath()
        plan_filepath: Path = Path(f"{instance_filepath.name}.plan.{self._planner}")
        status, plan = get_plan_v2(domain_filepath, instance_filepath, plan_filepath, self._planner, remove_files=remove_files)

        return {
            "instance_idx": instance_idx,
            "instance_filepath": Path(instance_filepath),
            "plan_filepath": plan_filepath,
            "planner_output": (status, plan),
        }

    def _preprocess_instance_final(self, preprocessed_data: Dict[str, Any]):
        instance_idx: int = preprocessed_data.get("instance_idx")
        assert self._state_factory._instances[instance_idx].instance_filepath() == preprocessed_data.get("instance_filepath")

        if preprocessed_data.get("marked_state_trajectories") is None:
            status, plan = preprocessed_data.get("planner_output")
            assert status == True

            state_trajectory: Tuple[int] = tuple([state_idx for state_idx, _ in self._state_factory.get_state_trajectory_from_plan(instance_idx, plan)])
            trajectory: List[Tuple[int, str]] = list(zip(state_trajectory[:-1], plan)) + [(state_trajectory[-1], "GOAL")]
            logging.debug(f"State trajectory: {state_trajectory}")
            logging.debug(f"Trajectory: {trajectory}")
            preprocessed_data.update({"marked_state_trajectories": [state_trajectory], "trajectories": [trajectory]})

            # Register all states in marked state trajectory as non-deadend states
            for state_idx in state_trajectory:
                self._state_factory.register_deadend_value(instance_idx, state_idx, False)

    def _non_loopy_state_trajectories(self, instance_idx: int) -> List[bool]:
        self._timers.resume("preprocessing")
        revisits_states: List[bool] = self._revisits_states_in_low_level_trajectory(instance_idx)
        self._timers.stop("preprocessing")
        return [not value for value in revisits_states]

    def _revisits_states_in_low_level_trajectory(self, instance_idx: int) -> List[bool]:
        preprocessed_data: Dict[str, Any] = self._preprocessed_datas[instance_idx]
        assert instance_idx == preprocessed_data.get("instance_idx")
        assert self._state_factory._instances[instance_idx].instance_filepath() == preprocessed_data.get("instance_filepath")

        if preprocessed_data.get("revisits_states") is None:
            status, plan = preprocessed_data.get("planner_output")
            assert status == True
            low_level_state_trajectory: List[Any] = self._state_factory.get_low_level_state_trajectory_from_plan(instance_idx, plan)
            revisits_states: bool = len(low_level_state_trajectory) > len(set(low_level_state_trajectory))
            preprocessed_data.update({"revisits_states": [revisits_states]})
        return preprocessed_data.get("revisits_states")

