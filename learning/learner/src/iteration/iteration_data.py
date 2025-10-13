from dataclasses import dataclass
from typing import Set, List, Dict, Union, Tuple, Any

import numpy as np

from .feature_pool import Feature
import dlplan.core as dlplan_core
from ..preprocessing import InstanceData


@dataclass
class IterationData:
    """ Store data that is being computed in each iteration of learning sketches. """
    current_idxs: List[int] = None
    preprocessed_datas: List[Dict[str, Any]] = None
    instance_datas: List[InstanceData] = None
    paths_with_idx: List[Tuple[int, Tuple[int]]] = None
    relevant_vertices: Dict[int, Set[int]] = None
    feature_pool: List[Feature] = None

    ext_state_to_dlplan_state_index: Dict[Tuple[int, int], int] = None
    ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = None
    instance_idx_to_denotations_caches: Dict[int, dlplan_core.DenotationsCaches] = None

    non_covered_vertices: Dict[int, Set[int]] = None
    vertices_in_deadend_paths: Dict[int, Set[int]] = None
    deadend_paths: List[Tuple[Tuple[int, int]]] = None

    inner_iterations: int = None

    def __repr__(self) -> str:
        return f"IterationData[current_idxs={self.current_idxs}, paths_with_idx={self.paths_with_idx}, inner_iterations={self.inner_iterations}, relevant_vertices={self.relevant_vertices}, non_covered_vertices={self.non_covered_vertices}, vertices_in_deadend_paths={self.vertices_in_deadend_paths}, deadend_paths={self.deadend_paths}]"

