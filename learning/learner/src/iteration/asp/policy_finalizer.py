import os
import logging
import random
import numpy as np

# Bitset-based unordered sets of unsigned integers
from intbitset import intbitset

from pathlib import Path
from termcolor import colored
from typing import Set, Tuple, List, Union, Dict, Any, Optional, Union, Callable, Generator
from collections import defaultdict
from itertools import product

import dlplan.core as dlplan_core

from ..feature_pool import Feature
from ...util import Timer

from .asp_solver import ASPSolver

LIST_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


class Verifier:
    def __init__(self,
                 solution: List[int],
                 r_idx_to_info: Dict[int, Tuple[int, intbitset]],
                 decorations: Dict[str, Dict[int, Dict[int, intbitset]]],
                 conditions_for_goal: List[Dict[int, int]],
                 r_idx_to_ext_state: List[Tuple[int, int]],
                 ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]],
                 ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray],
                 numerical_f_idxs: intbitset,
                 disable_yields: bool = False):
        self._solution: List[int] = solution
        self._r_idx_to_info: Dict[int, Tuple[int, intbitset]] = r_idx_to_info
        self._decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = decorations
        self._conditions_for_goal: List[Dict[int, int]] = conditions_for_goal

        self._r_idx_to_ext_state: List[Tuple[int, int]] = r_idx_to_ext_state
        self._ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]] = ext_state_to_ext_edge
        self._ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = ext_state_to_feature_valuations
        self._numerical_f_idxs: intbitset = numerical_f_idxs

        self._r_idxs: intbitset = intbitset(self._r_idx_to_info.keys())
        self._dont_care: Dict[int, Dict[int, intbitset]] = self._decorations.get("dont_care", dict())
        self._unknown: Dict[int, Dict[int, intbitset]] = self._decorations.get("unknown", dict())

        if disable_yields:
            self._yields: List[Tuple[int]] = None
            self._yield_to_yield_id: Dict[Tuple[int], int] = None
            self._yields_goal: intbitset = None
            self._yields_alive: intbitset = None
        else:
            self._yields: List[Tuple[int]] = list(product(*[(0, 1) for _ in self._solution]))
            self._yield_to_yield_id: Dict[Tuple[int], int] = {yield_: yield_id for yield_id, yield_ in enumerate(self._yields)}

            # Goal yields
            self._yields_goal: intbitset = None
            if self._conditions_for_goal is not None:
                self._yields_goal: intbitset = intbitset()
                for yield_id, yield_ in enumerate(self._yields):
                    for condition in self._conditions_for_goal:
                        compatible: bool = True
                        for i, f_idx in enumerate(self._solution):
                            if (yield_[i] > 0 and condition[f_idx] == 0) or (yield_[i] == 0 and condition[f_idx] > 0):
                                compatible = False
                                break
                        if compatible: self._yields_goal.add(yield_id)

            # Compute possible outputs for each rule
            self._compute_possible_outputs_for_each_rule()

            # Alive yields
            self._yields_alive: intbitset = intbitset(range(len(self._yields)))
            if self._yields_goal is not None: self._yields_alive -= self._yields_goal
            #print(f"Before FP: yields_for_r_idxs={sorted(self._yields_for_r_idxs)}, issubset={self._yields_for_r_idxs.issubset(self._yields_alive)}")
            #print(f"Before FP:      yields_alive={sorted(self._yields_alive)}")
            change_alive: bool = True
            while change_alive:
                change_alive: bool = False
                yields_to_remove: intbitset = intbitset()
                for yield_id in self._yields_alive - self._yields_for_r_idxs:
                    if len(self._output_yield_to_r_idx_to_input_yields[yield_id]) == 0:
                        yields_to_remove.add(yield_id)
                    else:
                        support: intbitset = intbitset.union(*[r_idx_support for r_idx, r_idx_support in self._output_yield_to_r_idx_to_input_yields[yield_id].items()])
                        if len(support & self._yields_alive) == 0:
                            yields_to_remove.add(yield_id)
                change_alive: bool = len(yields_to_remove) > 0
                self._yields_alive -= yields_to_remove
            #print(f" After FP: yields_for_r_idxs={sorted(self._yields_for_r_idxs)}, issubset={self._yields_for_r_idxs.issubset(self._yields_alive)}")
            #print(f" After FP:      yields_alive={sorted(self._yields_alive)}")

    def _is_boolean(self, f_idx: int) -> bool:
        return f_idx not in self._numerical_f_idxs

    def _is_numerical(self, f_idx: int) -> bool:
        return f_idx in self._numerical_f_idxs

    def _get_ext_state(self, r_idx: int) -> Tuple[int, int]:
        return self._r_idx_to_ext_state[r_idx]

    def _get_ext_edge(self, r_idx: int) -> Tuple[int, Tuple[int, int]]:
        return self._ext_state_to_ext_edge.get(self._get_ext_state(r_idx))

    def _get_feature_valuations(self, ext_state: Tuple[int, int]) -> np.ndarray:
        feature_valuations: np.ndarray = self._ext_state_to_feature_valuations.get(ext_state)
        assert feature_valuations is not None
        return feature_valuations

    def _is_applicable(self, input_yield: List[int], src_features: np.ndarray, dont_care: intbitset, verbose: bool = False) -> bool:
        for i, f_idx in enumerate(self._solution):
            if f_idx not in dont_care and ((input_yield[i] > 0 and src_features[i] == 0) or (input_yield[i] == 0 and src_features[i] > 0)):
                return False
        if verbose: logging.info(f"_is_applicable: input={input_yield}, src={src_features}, dont_care={sorted(dont_care)}, value=True")
        return True

    def _compatible(self, src_features: np.ndarray, dst_features: np.ndarray, effects: List[str]) -> bool:
        for i, (f_idx, effect) in enumerate(zip(self._solution, effects)):
            if effect == "false" and dst_features[i] > 0:
                return False
            elif effect == "true" and dst_features[i] == 0:
                return False
            elif effect == "inc" and src_features[i] >= dst_features[i]:
                return False
            elif effect == "dec" and src_features[i] <= dst_features[i]:
                return False
            elif effect == "eqv" and src_features[i] != dst_features[i]:
                return False
            else:
                # Effect is either 'unk', or some of the above but compatible. Do nothing!
                pass
        return True

    def _get_effects(self, src_features: np.ndarray, dst_features: np.ndarray, dont_care: intbitset, unknown: intbitset, verbose: bool = False) -> List[str]:
        effects: List[str] = []
        for i, f_idx in enumerate(self._solution):
            if f_idx in unknown:
                effects.append("unk")
            elif f_idx in dont_care:
                if self._is_boolean(f_idx):
                    assert dst_features[i] in [0, 1]
                    effects.append("true" if dst_features[i] == 1 else "false")
                else:
                    effects.append("eqv" if src_features[i] == dst_features[i] else ("dec" if src_features[i] > dst_features[i] else "inc"))
            elif src_features[i] == dst_features[i]:
                effects.append("eqv")
            elif self._is_numerical(f_idx):
                effects.append("dec" if src_features[i] > dst_features[i] else "inc")
            else:
                assert dst_features[i] in [0, 1]
                effects.append("true" if dst_features[i] == 1 else "false")
        assert all([effect in ["false", "true", "inc", "dec", "eqv", "unk"] for effect in effects])
        if verbose: logging.info(f"_get_effects: src={src_features}, dst={dst_features}, dont_care={sorted(dont_care)}, unknown={sorted(unknown)}, effects={effects}")
        return effects

    def _get_effects_for_yield(self, input_yield: List[int], effects: List[str], verbose: bool = False) -> List[str]:
        # Each item in effects is "false", "true", "inc", "dec", "eqv", or "unk"
        effects_dict: Dict[str, str] = {"false": "0", "true": "1", "inc": "1", "dec": "01", "unk": "01"}
        effects_for_yield: List[str] = [f"{input_yield[i]}" if effect == "eqv" else effects_dict.get(effect) for i, effect in enumerate(effects)]
        assert all([effect in ["0", "1", "01"] for effect in effects_for_yield])
        if verbose: logging.info(f"_get_effects_for_yield: input={input_yield}, effects={effects}, effects_for_yield={effects_for_yield}")
        return effects_for_yield

    def _get_possible_outputs(self, yield_to_yield_id: Dict[Tuple[int], int], effects_for_yield: List[str], verbose: bool = False) -> intbitset:
        # Each item in effects_for_yield is "0", "1", or "01"
        generator: List[Tuple[int]] = [(0, 1) if effect == "01" else (int(effect),) for effect in effects_for_yield]
        outputs: intbitset = intbitset([yield_to_yield_id.get(output_yield) for output_yield in product(*generator)])
        if verbose: logging.info(f"_get_possible_outputs: effects_for_yield={effects_for_yield}, outputs={sorted(outputs)}")
        return outputs

    def _compute_possible_outputs_for_each_rule(self):
        self._yields_for_r_idxs: intbitset = intbitset()
        self._input_yields_to_r_idxs: List[intbitset] = [intbitset() for _ in self._yields]
        self._output_yield_to_r_idx_to_input_yields: List[Dict[int, intbitset]] = [defaultdict(intbitset) for _ in self._yields]
        for r_idx in self._r_idxs:
            r_idx_ext_edge: Tuple[int, Tuple[int, int]] = self._get_ext_edge(r_idx)
            r_idx_dont_care: intbitset = self._dont_care.get(r_idx_ext_edge[0], dict()).get(r_idx_ext_edge[1][0], intbitset())
            r_idx_unknown: intbitset = self._unknown.get(r_idx_ext_edge[0], dict()).get(r_idx_ext_edge[1][0], intbitset())
            r_idx_src_features: np.ndarray = self._get_feature_valuations((r_idx_ext_edge[0], r_idx_ext_edge[1][0]))[self._solution]
            r_idx_dst_features: np.ndarray = self._get_feature_valuations((r_idx_ext_edge[0], r_idx_ext_edge[1][1]))[self._solution]
            r_idx_effects: List[str] = self._get_effects(r_idx_src_features, r_idx_dst_features, r_idx_dont_care, r_idx_unknown)
            for input_yield_id, input_yield in enumerate(self._yields):
                if self._is_applicable(input_yield, r_idx_src_features, r_idx_dont_care):
                    self._input_yields_to_r_idxs[input_yield_id].add(r_idx)
                    effects_for_yield: List[str] = self._get_effects_for_yield(input_yield, r_idx_effects)
                    output_yield_ids: intbitset = self._get_possible_outputs(self._yield_to_yield_id, effects_for_yield)
                    for output_yield_id in output_yield_ids:
                        self._output_yield_to_r_idx_to_input_yields[output_yield_id][r_idx].add(input_yield_id)
            yield_for_r_idx: Tuple[int] = tuple(1 if value > 0 else 0 for value in r_idx_src_features)
            self._yields_for_r_idxs.add(self._yield_to_yield_id.get(yield_for_r_idx))

    def verify_closure(self, verbose: bool = False) -> List[Tuple[Tuple[int, int]]]:
        # Sketch is closed if for each alive yield, there is a rule that is applicable
        yields_with_no_rule: List[int] = []
        for yield_id in self._yields_alive:
            if len(self._input_yields_to_r_idxs[yield_id]) == 0:
                logging.warning(f"[Verifier] No rule for yield {yield_id}.{self._yields[yield_id]}")
                yields_with_no_rule.append(yield_id)
        return [tuple(zip(self._solution, self._yields[yield_id])) for yield_id in yields_with_no_rule]

    def verify_bad_edges(self, bad_ext_edges: Set[Tuple[int, Tuple[int, int]]], verbose: bool = False) -> List[Tuple[int, Tuple[int, int]]]:
        accepted_bad_ext_edges: List[Tuple[int, Tuple[int, int]]] = []
        for bad_ext_edge in bad_ext_edges:
            src_features: np.ndarray = self._get_feature_valuations((bad_ext_edge[0], bad_ext_edge[1][0]))[self._solution]
            dst_features: np.ndarray = self._get_feature_valuations((bad_ext_edge[0], bad_ext_edge[1][1]))[self._solution]
            for r_idx in self._r_idxs:
                r_idx_ext_edge: Tuple[int, Tuple[int, int]] = self._get_ext_edge(r_idx)
                r_idx_dont_care: intbitset = self._dont_care.get(r_idx_ext_edge[0], dict()).get(r_idx_ext_edge[1][0], intbitset())
                r_idx_unknown: intbitset = self._unknown.get(r_idx_ext_edge[0], dict()).get(r_idx_ext_edge[1][0], intbitset())
                r_idx_src_features: np.ndarray = self._get_feature_valuations((r_idx_ext_edge[0], r_idx_ext_edge[1][0]))[self._solution]
                r_idx_dst_features: np.ndarray = self._get_feature_valuations((r_idx_ext_edge[0], r_idx_ext_edge[1][1]))[self._solution]
                r_idx_effects: List[str] = self._get_effects(r_idx_src_features, r_idx_dst_features, r_idx_dont_care, r_idx_unknown)
                if self._is_applicable(src_features, r_idx_src_features, r_idx_dont_care) and self._compatible(src_features, dst_features, r_idx_effects):
                    accepted_bad_ext_edges.append(bad_ext_edge)
                    logging.warning(f"[Verifier] Accepted bad ext-edge={bad_ext_edge}")
                    if verbose:
                        logging.info(f"    R_idx {r_idx} is compatible with bad ext-edge {bad_ext_edge}:")
                        logging.info(f"          bad_src_features: {src_features}")
                        logging.info(f"          bad_dst_features: {dst_features}")
                        logging.info(f"        r_idx_src_features: {r_idx_src_features}")
                        logging.info(f"        r_idx_dst_features: {r_idx_dst_features}")
                        logging.info(f"           r_idx_dont_care: {r_idx_dont_care}")
                        logging.info(f"             r_idx_unknown: {r_idx_unknown}")
                        logging.info(f"             r_idx_effects: {r_idx_effects}")
                        logging.info(f"                  solution: {self._solution}")
                        logging.info(f"                compatible: {self._compatible(src_features, dst_features, r_idx_effects)}")
        assert len(accepted_bad_ext_edges) == 0
        return accepted_bad_ext_edges


class _PolicyFinalizer:
    def __init__(self,
                 benchmark: Any,
                 preprocessing_data: Dict[str, Any],
                 r_idx_to_ext_state: List[Tuple[int, int]],
                 ext_state_to_r_idx: Dict[Tuple[int, int], int],
                 annotated_requirements: List[Tuple[Dict[str, Any], intbitset]],
                 non_sound_ext_states: Set[Tuple[int, int]],
                 **kwargs):
        self._benchmark: Any = benchmark
        self._r_idx_to_ext_state: List[Tuple[int, int]] = r_idx_to_ext_state
        self._ext_state_to_r_idx: Dict[Tuple[int, int], int] = ext_state_to_r_idx
        self._annotated_requirements = annotated_requirements
        self._non_sound_ext_states: Set[Tuple[int, int]] = non_sound_ext_states
        self._options: Dict[str, Any] = kwargs

        self._ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]] = preprocessing_data.get("ext_state_to_ext_edge")
        self._ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = preprocessing_data.get("ext_state_to_feature_valuations")
        self._goal_ext_states: List[Tuple[int, int]] = preprocessing_data.get("goal_ext_states")
        self._non_goal_ext_states: List[Tuple[int, int]] = preprocessing_data.get("non_goal_ext_states")
        self._bad_ext_edges: Set[Tuple[int, Tuple[int, int]]] = preprocessing_data.get("bad_ext_edges")
        self._width: int = preprocessing_data.get("width")
        assert self._bad_ext_edges is not None

        self._feature_pool: List[Feature] = preprocessing_data.get("feature_pool")
        self._f_idx_to_feature_index: Dict[int, int] = preprocessing_data.get("f_idx_to_feature_index")
        self._relevant_features: List[Tuple[int, Feature]] = preprocessing_data.get("relevant_features")
        self._numerical_features: List[Tuple[int, Feature]] = [(f_idx, feature) for f_idx, feature in self._relevant_features if isinstance(feature.dlplan_feature, dlplan_core.Numerical)]
        self._numerical_f_idxs: intbitset = intbitset([f_idx for f_idx, _ in self._numerical_features])

    def is_numerical(self, f_idx: int) -> bool:
        return f_idx in self._numerical_f_idxs

    def is_boolean(self, f_idx: int) -> bool:
        return f_idx not in self._numerical_f_idxs

    def get_ext_state(self, r_idx: int) -> Tuple[int, int]:
        return self._r_idx_to_ext_state[r_idx]

    def get_ext_edge(self, r_idx: int) -> Tuple[int, Tuple[int, int]]:
        return self._ext_state_to_ext_edge.get(self.get_ext_state(r_idx))

    def get_r_idx(self, ext_state: Tuple[int, int]) -> int:
        return self._ext_state_to_r_idx.get(ext_state)

    def get_feature_valuations(self, ext_state: Tuple[int, int]) -> np.ndarray:
        feature_valuations: np.ndarray = self._ext_state_to_feature_valuations.get(ext_state)
        if feature_valuations is None:
            assert self._feature_pool is not None
            feature_valuations: np.ndarray = self._benchmark.get_feature_valuations(*ext_state, self._feature_pool)
        return feature_valuations


class _NaiveFinalizer(_PolicyFinalizer):
    def __init__(self,
                 benchmark: Any,
                 preprocessing_data: Dict[str, Any],
                 r_idx_to_ext_state: List[Tuple[int, int]],
                 ext_state_to_r_idx: Dict[Tuple[int, int], int],
                 annotated_requirements: List[Tuple[Dict[str, Any], intbitset]],
                 non_sound_ext_states: Set[Tuple[int, int]],
                 **kwargs):
        super().__init__(benchmark,
                         preprocessing_data,
                         r_idx_to_ext_state,
                         ext_state_to_r_idx,
                         annotated_requirements,
                         non_sound_ext_states,
                         **kwargs)

    def _calculate_decorations(self,
                               solution: intbitset,
                               r_idx_to_info: Dict[int, Tuple[int, intbitset]],
                               **kwargs) -> Dict[str, Dict[int, Dict[int, intbitset]]]:
        # Necessary f_idxs: these are needed to guarantee good transitions are separated from bad transitions
        r_idx_to_necessary_f_idxs: Dict[int, List[intbitset]] = defaultdict(list)
        for annotation, requirement in self._annotated_requirements:
            if annotation["key"] == "Deadend":
                ext_state: Tuple[int, int] = annotation["ext_state"]
                r_idx: int = self.get_r_idx(ext_state)
                assert r_idx is not None
                necessary_f_idxs: intbitset = requirement & solution
                assert len(necessary_f_idxs) > 0
                r_idx_to_necessary_f_idxs[r_idx].append(necessary_f_idxs)

        if len(r_idx_to_necessary_f_idxs) > 0:
            logging.info(f"r_idx_to_necessary_f_idxs: {r_idx_to_necessary_f_idxs}")

        # Calculate decorations: "remove" conditions/effects of features that come "after" the "chosen" feature for each rule
        decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = {"dont_care": defaultdict(lambda: defaultdict(intbitset)), "unknown": defaultdict(lambda: defaultdict(intbitset))}
        for r_idx, (f_idx, above) in r_idx_to_info.items():
            singleton: intbitset = intbitset([f_idx])
            instance_idx, state_idx = self.get_ext_state(r_idx)
            assert instance_idx not in decorations["unknown"] or state_idx not in decorations["unknown"][instance_idx]
            assert instance_idx not in decorations["dont_care"] or state_idx not in decorations["dont_care"][instance_idx]
            f_idxs_to_remove: intbitset = solution - above - singleton
            f_idxs_to_preserve: intbitset = intbitset()
            list_necessary_f_idxs: List[intbitset] = r_idx_to_necessary_f_idxs.get(r_idx, [])
            for necessary_f_idxs in list_necessary_f_idxs:
                if necessary_f_idxs.issubset(f_idxs_to_remove) and len(necessary_f_idxs & f_idxs_to_preserve) == 0:
                    # Add one f_idx to preserve list
                    f_idxs_to_preserve.add(list(necessary_f_idxs)[0])
            f_idxs_to_remove -= f_idxs_to_preserve
            decorations["dont_care"][instance_idx][state_idx] = f_idxs_to_remove
            if not kwargs.get("simplify_only_conditions", False):
                decorations["unknown"][instance_idx][state_idx] = f_idxs_to_remove

        return decorations

    def solve(self, solution: List[int], r_idx_to_info: Dict[int, Tuple[int, intbitset]], **kwargs) -> Dict[str, Any]:
        # If no policy simplification, return solution
        if not kwargs.get("simplify_policy", False):
            decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = None
        else:
            decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = self._calculate_decorations(intbitset(solution), r_idx_to_info, **kwargs)

        # Verify closure and bad edges
        conditions_for_goal: List[Dict[int, int]] = None
        verifier: Verifier = Verifier(solution,
                                      r_idx_to_info,
                                      decorations,
                                      conditions_for_goal,
                                      self._r_idx_to_ext_state,
                                      self._ext_state_to_ext_edge,
                                      self._ext_state_to_feature_valuations,
                                      self._numerical_f_idxs,
                                      disable_yields=True)

        # Return result
        result: Dict[str, Any] = {
            "cost": sum([self._relevant_features[self._f_idx_to_feature_index[f_idx]][1].complexity - 1 for f_idx in solution]),
            "f_idxs": sorted(solution),
            "decorations": decorations,
            "conditions_for_goal": None,
            "verify_closure": None,
            "verify_bad_edges": verifier.verify_bad_edges(self._bad_ext_edges, verbose=True),
        }
        return result

    def repair(self, solution: Dict[str, Any], reasons: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise RuntimeError("_NaiveFinalizer: repair() not yet implemented")


class _ASPFinalizer(_PolicyFinalizer):
    def __init__(self,
                 benchmark: Any,
                 preprocessing_data: Dict[str, Any],
                 r_idx_to_ext_state: List[Tuple[int, int]],
                 ext_state_to_r_idx: Dict[Tuple[int, int], int],
                 annotated_requirements: List[Tuple[Dict[str, Any], intbitset]],
                 non_sound_ext_states: Set[Tuple[int, int]],
                 **kwargs):
        super().__init__(benchmark,
                         preprocessing_data,
                         r_idx_to_ext_state,
                         ext_state_to_r_idx,
                         annotated_requirements,
                         non_sound_ext_states,
                         **kwargs)

        self._asp_solver: ASPSolver = None
        self._solution: List[int] = None
        self._r_idx_to_info:  Dict[int, Tuple[int, intbitset]] = None
        self._ext_pairs: List[Tuple[str, Tuple[int, Tuple[int, int]]]] = None
        self._ext_pairs_set: Set[Any] = None
        self._pair_idx_to_r_idx: List[int] = None
        self._pair_idx_to_changes: List[Tuple[str]] = None
        self._reachable: List[Tuple[Tuple[int, int], bool, bool]] = None
        self._reachable_set: Set[Tuple[int, int]] = None
        self._ext_states: List[Tuple[str, Tuple[int, int]]] = None
        self._ext_states_set: Set[Tuple[int, int]] = None
        self._ext_state_to_boolean_valuation: Dict[Tuple[int, int], List[int]] = None
        self._goal_separating_features: List[Any] = None
        self._facts: Dict[str, List[Any]] = None

        self._hard_constraints: bool = kwargs.get("hard_constraints", False)
        self._dump_asp_program: bool = kwargs.get("dump_asp_program", False)

    def _add_ext_pair(self, pair_type: str, ext_pair: Tuple[int, Tuple[int, int]], r_idx: int):
        assert pair_type in ["good", "bad"]
        assert ext_pair not in self._ext_pairs_set

        # Add ext_pair
        pair_idx: int = len(self._ext_pairs)
        self._ext_pairs.append((pair_type, ext_pair))
        self._ext_pairs_set.add(ext_pair)
        self._pair_idx_to_r_idx.append(r_idx)

        # Calculate boolean valuations and changes for ext_pair
        ext_state_src: Tuple[int, int] = (ext_pair[0], ext_pair[1][0])
        ext_state_dst: Tuple[int, int] = (ext_pair[0], ext_pair[1][1])
        ext_states: List[Tuple[int, int]] = [ext_state_src, ext_state_dst]
        valuations: List[np.ndarray] = [self.get_feature_valuations(ext_state)[self._solution] for ext_state in ext_states]
        boolean_valuations: List[List[int]] = [[1 if value > 0 else 0 for value in valuation] for valuation in valuations]
        for i in [0, 1]:
            if ext_states[i] not in self._ext_state_to_boolean_valuation:
                self._ext_state_to_boolean_valuation[ext_states[i]] = boolean_valuations[i]
        changes: np.ndarray = valuations[1] - valuations[0]
        changes: Tuple[str] = tuple("inc" if d > 0 else ("dec" if d < 0 else "eqv") for d in changes)
        self._pair_idx_to_changes.append(changes)

        # Create facts for source/3, change/3, and fixed/2
        assert pair_type in ["good", "bad"]
        self._facts[f"{pair_type}/1"].append(self._asp_solver.make_fact(pair_type, pair_idx))
        for i, f_idx in enumerate(self._solution):
            self._facts["source/3"].append(self._asp_solver.make_fact("source", pair_idx, f_idx, boolean_valuations[0][i]))
        for i, change in enumerate(changes):
            self._facts["change/3"].append(self._asp_solver.make_fact("change", pair_idx, self._solution[i], change))

        # Create facts for fixed/2
        if pair_type == "good":
            f_idx, above = self._r_idx_to_info.get(r_idx)
            self._facts["fixed/2"].append(self._asp_solver.make_fact("fixed", pair_idx, f_idx))
            for g_idx in above:
                self._facts["fixed/2"].append(self._asp_solver.make_fact("fixed", pair_idx, g_idx))

    def _add_ext_state_reachable(self, ext_state: Tuple[int, int], force: bool, soft: bool):
        assert ext_state not in self._reachable_set

        reachable_idx: int = len(self._reachable)
        self._reachable.append((ext_state, force, soft))
        self._reachable_set.add(ext_state)

        # Calculate boolean valuation for ext_state
        valuation: np.ndarray = self.get_feature_valuations(ext_state)[self._solution]
        boolean_valuation: List[int] = [1 if value > 0 else 0 for value in valuation]
        if ext_state not in self._ext_state_to_boolean_valuation:
            self._ext_state_to_boolean_valuation[ext_state] = boolean_valuation
        logging.info(f"A.0: ext_state={ext_state}, reachable_idx={reachable_idx}, valuation={valuation}, boolean={boolean_valuation}")

        # Create facts for reachable/1, force/1, and soft/1
        self._facts["reachable/1"].append(self._asp_solver.make_fact("reachable", reachable_idx))
        if force: self._facts["force/1"].append(self._asp_solver.make_fact("force", reachable_idx))
        if soft: self._facts["soft/1"].append(self._asp_solver.make_fact("soft", reachable_idx))

        new_facts = [self._facts["reachable/1"][-1]]
        if force: new_facts.append(self._facts["force/1"][-1])
        if soft: new_facts.append(self._facts["soft/1"][-1])
        logging.info(f"A.2: new_facts={new_facts}")

        # Create facts for value_r/3
        for i, f_idx in enumerate(self._solution):
            self._facts["value_r/3"].append(self._asp_solver.make_fact("value_r", reachable_idx, f_idx, boolean_valuation[i]))

        # Create facts for transition/2, and change/3
        # Make width exploration from ext-state S to obtain all reachable states S'.
        # Add pairs (S,S') as transition/2 atoms, and add change/3 atoms
        seen_changes: Set[Tuple[str]] = set()
        explored: FrozenSet[Tuple[int, Tuple[int, int]]] = self._benchmark.exploration(*ext_state, self._width, caching=True)
        reachable: intbitset = intbitset([ext_state[1]] + [state_idx for _, (_, state_idx) in explored])
        logging.info(f"A.4: #explored={len(explored)}, #reachable={len(reachable)}")
        for state_idx in reachable:
            ext_state_dst: Tuple[int, int] = (ext_state[0], state_idx)
            ext_pair: Tuple[int, Tuple[int, int]] = (ext_state[0], (ext_state[1], state_idx))
            assert ext_pair not in self._ext_pairs_set

            # Calculate boolean valuation for (next) state_idx, and changes for ext_pair
            valuation_dst: np.ndarray = self.get_feature_valuations(ext_state_dst)[self._solution]
            boolean_valuation_dst: List[int] = [1 if value > 0 else 0 for value in valuation_dst]
            if ext_state_dst not in self._ext_state_to_boolean_valuation:
                self._ext_state_to_boolean_valuation[ext_state_dst] = boolean_valuation_dst
            changes: np.ndarray = valuation_dst - valuation
            changes: Tuple[str] = tuple("inc" if d > 0 else ("dec" if d < 0 else "eqv") for d in changes)

            # Create facts if this is a "new pair"
            if changes not in seen_changes:
                seen_changes.add(changes)

                # Add pair
                pair_idx: int = len(self._ext_pairs)
                self._ext_pairs.append(("reachable", ext_pair))
                self._pair_idx_to_r_idx.append(None)
                self._pair_idx_to_changes.append(changes)

                logging.info(f"A.5: New change: ext_pair={ext_pair}, pair_idx={pair_idx}, valuation={valuation_dst}, change={changes}")

                # Create facts for transition/2 and change/3
                self._facts["transition/2"].append(self._asp_solver.make_fact("transition", reachable_idx, pair_idx))
                for i, f_idx in enumerate(self._solution):
                    self._facts["source/3"].append(self._asp_solver.make_fact("source", pair_idx, f_idx, boolean_valuation[i]))
                    #logging.info(f"A.5: - Source: pair_idx={pair_idx}, f_idx={f_idx}, bvalue={boolean_valuation[i]}")
                for i, change in enumerate(changes):
                    self._facts["change/3"].append(self._asp_solver.make_fact("change", pair_idx, self._solution[i], change))
                    #logging.info(f"A.5: - Change: pair_idx={pair_idx}, f_idx={self._solution[i]}, change={change}")
                logging.debug(f"Pair {ext_pair} added as atom 'transition({reachable_idx},{pair_idx})': changes={changes}")

    def _add_bad_ext_edges(self, bad_ext_edges: List[Tuple[int, Tuple[int, int]]]):
        for ext_edge in bad_ext_edges:
            logging.info(f"Add bad ext-edge={ext_edge}")
            self._add_ext_pair("bad", ext_edge, None)

    def _add_ext_state(self, state_type: str, ext_state: Tuple[int, int]):
        assert state_type in ["goal", "non-goal"]
        assert ext_state not in self._ext_states_set
        ext_state_idx: int = len(self._ext_states)
        self._ext_states.append((state_type, ext_state))

        valuation: np.ndarray = self.get_feature_valuations(ext_state)[self._solution]
        boolean_valuation: List[int] = [1 if value > 0 else 0 for value in valuation]
        if ext_state not in self._ext_state_to_boolean_valuation:
            self._ext_state_to_boolean_valuation[ext_state] = boolean_valuation

        self._facts["state/2"].append(self._asp_solver.make_fact("state", state_type, ext_state_idx))
        for i, f_idx in enumerate(self._solution):
            self._facts["value_s/3"].append(self._asp_solver.make_fact("value_s", ext_state_idx, f_idx, boolean_valuation[i]))

    def _get_new_solver(self) -> ASPSolver:
        fact_signatures: List[Tuple[Any]] = [
            ("feature", ("f",), "feature(f)."),
            ("boolean", ("f",), "boolean(f)."),
            ("good", ("g",), "good(g)."),
            ("bad", ("b",), "bad(b)."),
            ("fixed", ("g", "f"), "fixed(g,f)."),
            ("source", ("t", "f", "v"), "source(t,f,v)."),
            ("change", ("t", "f", "c"), "change(t,f,c)."),
            ("yield", ("y",), "yield(y)."),
            #("yield", ("g", "y",), "yield(g,y)."),
            ("value", ("y", "f", "v",), "value(y,f,v)."),
            ("goal", ("y",), "goal(y)."),
            ("reachable", ("r",), "reachable(r)."),
            ("value_r", ("r", "f", "v"), "value_r(r,f,v)."),
            ("transition", ("r", "t"), "transition(r,t)."),
            ("force", ("r",), "force(r)."),
            ("soft", ("r",), "soft(r)."),
            ("state", ("t", "s",), "state(t,s)."),
            ("value_s", ("s", "f", "v"), "value_s(s,f,v)."),
            ("hard_constraints", (), "hard_constraints."),
        ]
        arguments: List[str] = ["--parallel-mode=16", "-n", "0"]
        #loads: List[str] = [str(LIST_DIR / f"relax_transitions.lp")]
        loads: List[str] = [str(LIST_DIR / f"relax_transitions_v2.lp")]
        asp_solver: ASPSolver = ASPSolver(arguments=arguments, fact_signatures=fact_signatures, loads=loads)
        return asp_solver

    def _construct(self, solution: List[int], r_idx_to_info: Dict[int, Tuple[int, intbitset]]):
        # Base elements
        self._asp_solver: ASPSolver = self._get_new_solver()
        self._solution: List[int] = solution
        self._r_idx_to_info: Dict[int, Tuple[int, intbitset]] = r_idx_to_info
        self._ext_pairs: List[Tuple[str, Tuple[int, Tuple[int, int]]]] = []
        self._ext_pairs_set: Set[Any] = set()
        self._pair_idx_to_r_idx: List[int] = []
        self._pair_idx_to_changes: List[Tuple[str]] = []
        self._reachable: List[Tuple[Tuple[int, int], bool, bool]] = []
        self._reachable_set: Set[Tuple[int, int]] = set()
        self._ext_states: List[Tuple[str, Tuple[int, int]]] = []
        self._ext_states_set: Set[Tuple[int, int]] = set()
        self._ext_state_to_boolean_valuation: Dict[Tuple[int, int], List[int]] = dict()
        self._goal_separating_features: List[Any] = []
        self._facts: Dict[str, List[Any]] = defaultdict(list)

        # Good transitions
        for r_idx in r_idx_to_info.keys():
            ext_state: Tuple[int, int] = self.get_ext_state(r_idx)
            ext_edge: Tuple[int, Tuple[int, int]] = self._ext_state_to_ext_edge.get(ext_state)
            assert ext_state is not None and ext_edge is not None
            self._add_ext_pair("good", ext_edge, r_idx)

        # Goal and non-goal states
        for ext_state in  self._goal_ext_states:
            self._add_ext_state("goal", ext_state)
        for ext_state in  self._non_goal_ext_states:
            self._add_ext_state("non-goal", ext_state)

        """
        # Valuations that separate goal from non-goal states
        pending_goal_requirements: List[int] = list(range(len(self._requirements_for_goals)))
        while len(pending_goal_requirements) > 0:
            candidates: List[Tuple[int, int]] = [(i, sum([1 if f_idx in self._requirements_for_goals[j] else 0 for j in pending_goal_requirements])) for i, f_idx in enumerate(solution)]
            candidates_sorted: List[Tuple[int, int]] = sorted(candidates, key=lambda p: p[1], reverse=True)
            assert candidates_sorted[0][1] > 0
            self._goal_separating_features.append(candidates_sorted[0][0])
            pending_goal_requirements: List[int] = [i for i in pending_goal_requirements if solution[candidates_sorted[0][0]] not in self._requirements_for_goals[i]]

        for pair in [annotation.get("pair") for annotation, _ in self._annotated_requirements if annotation.get("key") == "Goal"]:
            goal_ext_state: Tuple[int, int] = (pair[0], pair[1][0])
            non_goal_ext_state: Tuple[int, int] = (pair[0], pair[1][1])
            goal_boolean_valuation: Tuple[int] = self._ext_state_to_boolean_valuation.get(goal_ext_state)
            non_goal_boolean_valuation: Tuple[int] = self._ext_state_to_boolean_valuation.get(non_goal_ext_state)
            if non_goal_boolean_valuation is None:
                # This can happen because set of transitions above doesn't contain all the relevant transitions
                valuation: np.ndarray = self.get_feature_valuations(non_goal_ext_state)[solution]
                non_goal_boolean_valuation: List[int] = [1 if value > 0 else 0 for value in valuation]
                self._ext_state_to_boolean_valuation[non_goal_ext_state] = non_goal_boolean_valuation
            self._conditions_for_goal.add(tuple([(i, goal_boolean_valuation[i]) for i in self._goal_separating_features]))
            self._conditions_for_non_goal.add(tuple([(i, non_goal_boolean_valuation[i]) for i in self._goal_separating_features]))

        if len(self._conditions_for_goal & self._conditions_for_non_goal) > 0:
            logging.warning(f"Non-empty intersection of feature valuations for goal and non-goal states: {self._conditions_for_goal & self._conditions_for_non_goal}")
        logging.info(f"conditions_for_goal: features={self._solution}, valuations={sorted(self._conditions_for_goal)}")
        """

        # Deadend paths
        self._add_bad_ext_edges(self._bad_ext_edges)

        # Base facts
        if self._hard_constraints:
            self._facts["constraints"].append(self._asp_solver.make_fact("hard_constraints"))

        # feature/1 and boolean/1
        for f_idx in self._solution:
            self._facts["feature/1"].append(self._asp_solver.make_fact("feature", f_idx))
            if f_idx not in self._numerical_f_idxs:
                self._facts["feature/1"].append(self._asp_solver.make_fact("boolean", f_idx))

        # yield/1, value/3, and goal/1
        self._yields: List[Tuple[int]] = list(product(*[(0, 1) for _ in self._solution]))
        for i, _yield in enumerate(self._yields):
            self._facts["yield/1"].append(self._asp_solver.make_fact("yield", i))
            for j, f_idx in enumerate(self._solution):
                self._facts["value/3"].append(self._asp_solver.make_fact("value", i, f_idx, _yield[j]))
            """
            if any([all([_yield[j] == value for j, value in condition]) for condition in self._conditions_for_goal]):
                self._facts["goal/1"].append(self._asp_solver.make_fact("goal", i))
                logging.info(f"GOAL: yield={_yield}")
            """

        # Non-sound ext states
        for ext_state in self._non_sound_ext_states:
            logging.warning(f"Declare ext_state={ext_state} as non-sound")
            self._add_ext_state_reachable(ext_state, force=True, soft=False)

    def solve(self, solution: List[int], r_idx_to_info: Dict[int, Tuple[int, intbitset]], **kwargs) -> Dict[str, Any]:
        # If no policy simplification, return solution
        if not kwargs.get("simplify_policy", False):
            assert solution is not None, "CHECK CASE: Unexpected solution=None"
            result: Dict[str, Any] = {
                "cost": sum([self._relevant_features[self._f_idx_to_feature_index[f_idx]][1].complexity - 1 for f_idx in solution]),
                "f_idxs": sorted(solution),
                "decorations": None,
                "conditions_for_goal": None,
                "verify_closure": None,
                "verify_bad_edges": None,
            }
            return result

        # If no ASP solver, construct one and set initial theory
        if self._asp_solver is None:
            self._construct(solution, r_idx_to_info)

        # Ground facts and call solver
        local_timer: Timer = Timer()
        self._asp_solver.ground(self._facts, dump_asp_program=self._dump_asp_program)
        symbols, cost, exit_code = self._asp_solver.optimize_model()
        local_timer.stop()
        logging.info(f"{local_timer.get_elapsed_sec():.02f} second(s) for ASP solver; exit_code={exit_code}")

        # If no solution, return None
        if symbols is None:
            return {
                "cost": None,
                "f_idxs": None,
                "decorations": None,
                "conditions_for_goal": None,
            }

        # Read symbols
        eqclass: intbitset = intbitset()
        eq: Dict[int, intbitset] = defaultdict(intbitset)
        goal_yields: List[int] = []
        marks: Dict[str, Dict[int, intbitset]] = {"dont_care": defaultdict(intbitset), "unknown": defaultdict(intbitset)}
        for symbol in symbols:
            if symbol.name == "class":
                eqclass.add(symbol.arguments[0].number)
            elif symbol.name == "eq":
                eq[symbol.arguments[0].number].add(symbol.arguments[1].number)
            elif symbol.name == "goal_y":
                goal_yields.append(symbol.arguments[0].number)
            elif symbol.name in ["dont_care", "unknown"]:
                marks[symbol.name][symbol.arguments[0].number].add(symbol.arguments[1].number)
            elif symbol.name in ["non_sound", "non_closed", "non_safe", "non_covered"]:
                logging.warning(colored(f"Got {symbol.name}({symbol.arguments[0].number})", "magents", attrs=["bold"]))

        # Construct goal conditions
        conditions_for_goal: List[Dict[int, int]] = [dict(zip(self._solution, self._yields[i])) for i in goal_yields]

        # Construct decorations
        decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = {"dont_care": defaultdict(lambda: defaultdict(intbitset)), "unknown": defaultdict(lambda: defaultdict(intbitset))}
        for mark in marks.keys():
            for tr_idx, f_idxs in marks[mark].items():
                for tr_idx_2 in eq.get(tr_idx):
                    ext_edge: Tuple[int, Tuple[int, int]] = self._ext_pairs[tr_idx_2][1]
                    ext_state: Tuple[int, int] = (ext_edge[0], ext_edge[1][0])
                    decorations[mark][ext_state[0]][ext_state[1]] |= f_idxs

        # Verify closure and bad edges
        verifier: Verifier = Verifier(solution,
                                      r_idx_to_info,
                                      decorations,
                                      conditions_for_goal,
                                      self._r_idx_to_ext_state,
                                      self._ext_state_to_ext_edge,
                                      self._ext_state_to_feature_valuations,
                                      self._numerical_f_idxs)

        # Return result
        result: Dict[str, Any] = {
            "cost": None if solution is None else sum([self._relevant_features[self._f_idx_to_feature_index[f_idx]][1].complexity - 1 for f_idx in solution]),
            "f_idxs": solution,
            "decorations": decorations,
            "conditions_for_goal": conditions_for_goal,
            "verify_closure": verifier.verify_closure(verbose=True),
            "verify_bad_edges": verifier.verify_bad_edges(self._bad_ext_edges, verbose=True),
        }
        return result

    def repair(self, solution: Dict[str, Any], reasons: List[Dict[str, Any]]) -> Dict[str, Any]:
        logging.info(f"Repair: Reasons={reasons}")
        for reason in reasons:
            if "non-sound" in reason:
                ###return {"decorations": None, "conditions_for_goal": None}
                for ext_state in reason.get("non-sound"):
                    self._add_ext_state_reachable(ext_state, force=True, soft=False)
            elif "non-goal" in reason:
                for ext_state in reason.get("non-goal"):
                    self._add_ext_state("non-goal", ext_state)
            else:
                raise RuntimeError(f"Unknown reason to repair: reason={reason}")
        logging.info(f"Repair: Call solver...")
        self._asp_solver: ASPSolver = self._get_new_solver()
        result: Dict[str, Any] = self.solve(None, None)
        logging.info(f"Repair: Result={result}")
        if "non-sound" in reasons[0] and result.get("decorations") is not None: logging.info("SUCCESSFUL REPAIR")
        return result


# Adds features to satisfy pending requirements and simplify resulting policy
class PolicyFinalizer:
    def __init__(self,
                 benchmark: Any,
                 preprocessing_data: Dict[str, Any],
                 r_idx_to_ext_state: List[Tuple[int, int]],
                 ext_state_to_r_idx: Dict[Tuple[int, int], int],
                 annotated_requirements: List[Tuple[Dict[str, Any], intbitset]],
                 non_sound_ext_states: Set[Tuple[int, int]],
                 **kwargs):
        self._benchmark: Any = benchmark
        self._preprocessing_data: Dict[str, Any] = preprocessing_data
        self._r_idx_to_ext_state: List[Tuple[int, int]] = r_idx_to_ext_state
        self._ext_state_to_r_idx: Dict[Tuple[int, int], int] = ext_state_to_r_idx
        self._annotated_requirements: List[Tuple[Dict[str, Any], intbitset]] = annotated_requirements
        self._non_sound_ext_states: Set[Tuple[int, int]] = non_sound_ext_states

        self._options: Dict[str, Any] = {
            "benchmark": self._benchmark,
            "preprocessing_data": self._preprocessing_data,
            "r_idx_to_ext_state": self._r_idx_to_ext_state,
            "ext_state_to_r_idx": self._ext_state_to_r_idx,
            "annotated_requirements": self._annotated_requirements,
            "non_sound_ext_states": self._non_sound_ext_states,
        }
        self._options.update(kwargs)

        # Features
        self._f_idx_to_feature_index: Dict[int, int] = preprocessing_data.get("f_idx_to_feature_index")
        self._relevant_features: List[Tuple[int, Feature]] = preprocessing_data.get("relevant_features")

        # Finalizer
        self._finalizer: _PolicyFinalizer = None
        self._width: int = preprocessing_data.get("width")

    def solve_pending_requirements(self, f_idxs: intbitset) -> Tuple[intbitset, intbitset]:
        def get_pending_requirements(solution: intbitset) -> List[int]:
            assert self._width > 0 or len([i for i, (annotation, requirement) in enumerate(self._annotated_requirements) if annotation["key"] == "Edge" and len(requirement & solution) == 0]) == 0
            return [i for i, (annotation, requirement) in enumerate(self._annotated_requirements) if annotation["key"] != "Edge" and len(requirement & solution) == 0]

        # Score function for choosing features
        def score_fn(f_idx: int, pending_requirements: List[int]) -> Tuple[Union[int, float]]:
            feature_index: int = self._f_idx_to_feature_index[f_idx]
            feature_complexity: int = self._relevant_features[feature_index][1].complexity - 1
            solved_pending_requirements: List[int] = [i for i in pending_requirements if f_idx in self._annotated_requirements[i][1]]
            return (len(solved_pending_requirements) / feature_complexity,)

        solution: intbitset = intbitset(f_idxs)
        pending_requirements: List[int] = get_pending_requirements(solution)
        #pending_requirements: List[int] = [i for i, (annotation, requirement) in enumerate(self._annotated_requirements) if annotation["key"] != "Edge" and len(requirement & solution) == 0]
        while len(pending_requirements) > 0:
            logging.info(f"[solve_pending_requirements] pending_requirements: {pending_requirements}")
            eligible_features_with_score: List[Tuple[int, Tuple[float]]] = [(f_idx, score_fn(f_idx, pending_requirements)) for f_idx, _ in self._relevant_features if f_idx not in solution]
            eligible_features_with_non_zero_score: List[Tuple[int, Tuple[float]]] = [(f_idx, score) for f_idx, score in eligible_features_with_score if score[0] > 0]
            sorted_eligible_features: List[Tuple[int, Tuple[float]]] = sorted(eligible_features_with_non_zero_score, key=lambda item: item[1], reverse=True)

            if len(sorted_eligible_features) == 0:
                logging.info("No feature for solving pending requirements:")
                for annotation, requirement in [self._annotated_requirements[i] for i in pending_requirements]:
                    if annotation["key"] == "Edge":
                        ext_state: Tuple[int, int] = annotation["ext_state"]
                        r_idx: int = self._get_r_idx(ext_state)
                        logging.info(f"  Unexpected pending requirement: key='Edge', ext_state={ext_state}, r_idx={r_idx}")
                        logging.info(f"    This should have detected during rule elimination by _rec_solve")
                    elif annotation["key"] == "Goal":
                        pair: Any = annotation["pair"]
                        logging.info(f"  Goal requirement: pair={pair}, requirement={sorted(requirement)}")
                    elif annotation["key"] == "Deadend":
                        ext_state: Tuple[int, int] = annotation["ext_state"]
                        path: path = annotation["path"]
                        logging.info(f"  Deadend requirement: ext_state={ext_state}, path={path}, requirement={sorted(requirement)}")
                    elif annotation["key"] == "Sibling":
                        ext_sibling: Tuple[int, int] = annotation["ext_sibling"]
                        logging.info(f"  Sibling requirement: ext_sibling={ext_sibling}, requirement={sorted(requirement)}")
                    else:
                        raise RuntimeError(f"Unknown requirement key '{annotation['key']}'")
                return None, None

            # Choose a best eligible feature
            best_f_idxs: List[Tuple[int]] = [f_idx for f_idx, score in sorted_eligible_features if score == sorted_eligible_features[0][1]]
            best_f_idx: int = random.choice(best_f_idxs)
            best_score: Tuple[float] = score_fn(best_f_idx, pending_requirements)
            complexity: int = self._relevant_features[self._f_idx_to_feature_index[best_f_idx]][1].complexity - 1
            logging.info(f"[solve_pending_requirements] f{best_f_idx}.{self._relevant_features[self._f_idx_to_feature_index[best_f_idx]][1]._dlplan_feature}/{complexity}, score={best_score}")

            solution.add(best_f_idx)
            #pending_requirements_reduced: List[int] = [i for i in pending_requirements if len(self._annotated_requirements[i][1] & solution) == 0]
            pending_requirements_reduced: List[int] = get_pending_requirements(solution)
            assert intbitset(pending_requirements_reduced).issubset(intbitset(pending_requirements))
            assert len(pending_requirements_reduced) < len(pending_requirements)
            pending_requirements = pending_requirements_reduced

        if len(solution - f_idxs) > 0:
            logging.info(f"[solve_pending_requirements] added={sorted(solution - f_idxs)}")

        return solution, solution - f_idxs

    # This is the intended "external access point" (API)
    def __call__(self, f_idxs: intbitset, r_idx_to_info: Dict[int, Tuple[int, intbitset]], **kwargs) -> Dict[str, Any]:
        if self._options.get("solve_pending_requirements", False):
            solution, difference = self.solve_pending_requirements(f_idxs)
            if solution != f_idxs:
                logging.info(f"Patched solution={sorted(solution)}, difference={sorted(difference)}")
        else:
            solution: intbitset = f_idxs

        threshold_for_asp_based_simplification: int = kwargs.get("threshold_for_asp_based_simplification") or 10
        if len(solution) > threshold_for_asp_based_simplification:
            logging.info(f"Calculate decorations with NAIVE solver for {len(solution)} feature(s)")
            self._finalizer: _PolicyFinalizer = _NaiveFinalizer(**self._options)
        else:
            logging.info(f"Calculate decorations with ASP solver for {len(solution)} feature(s)")
            self._finalizer: _PolicyFinalizer = _ASPFinalizer(**self._options)

        result: Dict[str, Any] = self._finalizer.solve(sorted(solution), r_idx_to_info, **kwargs)
        return result

    def repair(self, solution: Dict[str, Any], reason: Dict[str, Any]) -> Dict[str, Any]:
        return self._finalizer.repair(solution, reason)

