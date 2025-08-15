import logging
from typing import Tuple, List, Dict, Union, Any
from pathlib import Path

import numpy as np
import dlplan.core as dlplan_core

from .feature_pool import Feature
from .iteration_data import IterationData
from ..preprocessing import PreprocessingData


def compute_per_state_feature_valuations(
        preprocessing_data: PreprocessingData,
        iteration_data: IterationData) -> None:
    """ Evaluate features on representative concrete state of all global faithful abstract states.
    """
    assert False
    gfa_state_global_idx_to_feature_evaluations: Dict[int, List[Union[bool, int]]] = dict()
    for gfa_state in iteration_data.gfa_states:
        instance_idx = gfa_state.get_faithful_abstraction_index()
        instance_data = preprocessing_data.instance_datas[instance_idx]
        dlplan_ss_state = preprocessing_data.state_finder.get_dlplan_ss_state(gfa_state)
        global_state_global_idx = gfa_state.get_global_index()
        state_feature_valuations: List[Union[bool, int]] = []
        for feature in iteration_data.feature_pool:
            state_feature_valuations.append(feature.dlplan_feature.evaluate(dlplan_ss_state, instance_data.denotations_caches))
        gfa_state_global_idx_to_feature_evaluations[global_state_global_idx] = state_feature_valuations

    return gfa_state_global_idx_to_feature_evaluations

def compute_feature_valuations_for_dlplan_state(dlplan_state: dlplan_core.State,
                                                features: List[Feature],
                                                instance_idx_to_denotations_caches: Dict[int, dlplan_core.DenotationsCaches]) -> np.ndarray: #Tuple[int]:
    instance_idx: int = dlplan_state.get_instance_info().get_index()
    state_idx: int = dlplan_state.get_index()
    ext_state: Tuple[int, int] = (instance_idx, state_idx)
    #denotations_caches: dlplan_core.DenotationsCaches = instance_idx_to_denotations_caches.get(instance_idx)
    denotations_caches: dlplan_core.DenotationsCaches = dlplan_core.DenotationsCaches()
    assert denotations_caches is not None, f"NON-EXISTENT DENOTATIONS CACHE FOR INSTANCE {instance_idx}"

    lower_bound, upper_bound = np.iinfo(np.int16).min // 2, np.iinfo(np.uint16).max // 2
    feature_valuations: List[int] = []
    for feature in features:
        #value = feature.dlplan_feature.evaluate(dlplan_state)
        value = feature.dlplan_feature.evaluate(dlplan_state, denotations_caches)
        feature_valuations.append(min(max(value, lower_bound), upper_bound))
        assert feature_valuations[-1] is not None, f"INVALID FEATURE VALUATION: {dlplan_state}, {feature}"
    feature_valuations: np.ndarray = np.array(feature_valuations, dtype=np.int16)
    feature_valuations.setflags(write=False)
    return feature_valuations

def compute_feature_valuations_for_dlplan_states(dlplan_states: List[dlplan_core.State],
                                                 features: List[Feature],
                                                 instance_idx_to_denotations_caches: Dict[int, dlplan_core.DenotationsCaches]) -> List[np.ndarray]: #List[Tuple[int]]:
    assert False
    return [compute_feature_valuations_for_dlplan_state(dlplan_state, features, instance_idx_to_denotations_caches) for dlplan_state in dlplan_states]

def compute_feature_valuations_dict_for_dlplan_states(dlplan_states: List[dlplan_core.State],
                                                      features: List[Feature],
                                                      instance_idx_to_denotations_caches: Dict[int, dlplan_core.DenotationsCaches]) -> Dict[Tuple[int, int], np.ndarray]: #Dict[Tuple[int, int], Tuple[int]]:
    assert False
    return {(dlplan_state.get_instance_info().get_index(), dlplan_state.get_index()): compute_feature_valuations_for_dlplan_state(dlplan_state, features, instance_idx_to_denotations_caches) for dlplan_state in dlplan_states}

