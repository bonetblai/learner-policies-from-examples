import logging
import math
import random

from collections import defaultdict, deque
from termcolor import colored
from typing import Dict, Set, FrozenSet, List, Deque, MutableSet, Tuple, Any, Optional
from pathlib import Path
import numpy as np
from intbitset import intbitset

import dlplan.core as dlplan_core
import dlplan.policy as dlplan_policy
from dlplan.policy import PolicyMinimizer

from .feature_pool import Feature
from .iteration_data import IterationData
from .feature_decoder import FeatureDecoder
from ..preprocessing import PreprocessingData, InstanceData
from ..state_space import StateFactory
from ..util import Timer


class SketchReduced:
    def __init__(self, state_factory: StateFactory, feature_pool: List[Feature], f_idxs: List[int], dlplan_policy: dlplan_policy.Policy, width: int):
        self._state_factory: StateFactory = state_factory
        self._feature_pool: List[Feature] = feature_pool
        self._f_idxs: List[int] = f_idxs
        self._dlplan_policy: dlplan_policy.Policy = dlplan_policy
        self._width: int = width

    def _get_state_values(self, instance_data: InstanceData, state_idx: int) -> Dict[int, int]:
        state: Any = instance_data.get_state(state_idx)[1]
        dlplan_state: dlplan_core.State = instance_data.get_dlplan_state(state_idx, state)
        state_values: Dict[int, int] = {f_idx: self._feature_pool[f_idx]._dlplan_feature.evaluate(dlplan_state) for f_idx in self._f_idxs}
        return state_values

    def _values_satisfy_condition(self, values: Dict[int, int], condition: Dict[int, int]) -> bool:
        for f_idx, bvalue in condition.items():
            value: int = values[f_idx]
            if (bvalue == 0 and value > 0) or (bvalue == 1 and value == 0):
                return False
        return True

    def _values_satisfy_conditions(self, values: Dict[int, int], conditions: List[Dict[int, int]]) -> bool:
        for condition in conditions:
            if self._values_satisfy_condition(values, condition):
                return True
        return False

    def _state_satisfies_conditions(self, instance_data: InstanceData, state_idx: int, conditions: List[Dict[int, int]]) -> bool:
        return self._values_satisfy_conditions(self._get_state_values(instance_data, state_idx), conditions)

    def _values_get_satisfied_conditions(self, values: Dict[int, int], conditions: List[Dict[int, int]]) -> List[Dict[int, int]]:
        satisfied_conditions: List[Dict[int, int]] = []
        for condition in conditions:
            if self._values_satisfy_condition(values, condition):
                satisfied_conditions.append(condition)
        return satisfied_conditions

    def _state_get_satisfied_conditions(self, instance_data: InstanceData, state_idx: int, conditions: List[Dict[int, int]]) -> List[Dict[int, int]]:
        return self._values_get_satisfied_conditions(self._get_state_values(instance_data, state_idx), conditions)

    def _get_width_successors(self, instance_data: InstanceData, state_idx: int) -> List[Tuple[Tuple[int, Any], str]]:
        if self._width == 0:
            successors: List[Tuple[Tuple[int, Any], str]] = [((succ_state_idx, succ_state), action) for (succ_state_idx, succ_state), action in instance_data.get_successors(state_idx) if succ_state_idx != state_idx]
        else:
            successors: List[int] = list(instance_data.exploration(state_idx, self._width, caching=True) - intbitset([state_idx]))
            successors: List[Tuple[Tuple[int, Any], str]] = [(instance_data.get_state(succ_state_idx), "<not-avail>") for succ_state_idx in successors]
        logging.debug(f"[_get_width_successors] ext_state={(instance_data.idx, state_idx)}, width={self._width}, #successors={len(successors)}")
        return successors

    def _verify_policy_for_bounded_width(self,
                                         preprocessing_data: PreprocessingData,
                                         iteration_data: IterationData,
                                         instance_data: InstanceData,
                                         randomized_sketch_test: int,
                                         max_non_covered_ext_states: int,
                                         **kwargs) -> Tuple[bool, Dict[int, Set[int]], Dict[str, Any]]:
        logging.debug(f"[VERIFY] INSTANCE_INDEX: {instance_data.idx}")

        # Get valuations for goals (if any)
        conditions_for_goal: List[Dict[int, int]] = kwargs.get("conditions_for_goal")

        queue: Deque[int] = deque()
        visited: MutableSet[int] = set()
        parent: Dict[int, int] = dict()
        for initial_state_idx, _ in [instance_data.get_initial_state()]:
            logging.debug(f"[VERIFY] initial_state_idx={initial_state_idx}")
            queue.append(initial_state_idx)
            visited.add(initial_state_idx)
            parent[initial_state_idx] = -1

        # byproduct for acyclicity check
        subgoal_states_per_r_reachable_state: Dict[int, Set[int]] = defaultdict(set)

        # Keep track of time and report progress every 60 seconds
        timers: Dict[str, Timer] = {
            "search": Timer(),
            "expansion": Timer(stopped=True),
            "successors": Timer(stopped=True),
        }
        last_record: float = 0
        reporting_period: float = 60

        # Non-covered states are divided into "non-closed" (those for which no rule condition is consistent with them), and
        # "non-sound" (those that are not closed but no rule is compatible with a transition for them)
        non_closed_ext_states: Set[Tuple[int, int]] = set()
        non_sound_ext_states: Set[Tuple[int, int]] = set()

        # Perform width breadth-first exploration of reachable states
        expanded: int = 0
        generated: int = 0
        while queue:
            timers["search"].stop()
            if timers["search"].get_elapsed_sec() - last_record > reporting_period:
                last_record = timers["search"].get_elapsed_sec()
                elapsed_times: str = "(" + ", ".join([f"{timers[key].get_elapsed_sec():0.2f}" for key in ["search", "expansion", "successors"]]) + ")"
                logging.info(f"[VERIFY] elapsed_times={elapsed_times}, expanded={expanded}, generated={generated}, #queue={len(queue)}")
            timers["search"].resume()

            root_state_idx = queue.pop()
            _, root_state = instance_data.get_state(root_state_idx)
            root_dlplan_state = instance_data.get_dlplan_state(root_state_idx, root_state)

            logging.debug(f"[VERIFY] DEQUED: {root_state_idx}.{root_dlplan_state}")

            satisfies_conditions_for_goal: bool = None if conditions_for_goal is None else self._state_satisfies_conditions(instance_data, root_state_idx, conditions_for_goal)
            if instance_data.is_goal_state(root_state_idx):
                logging.debug(colored(f"[VERIFY] GOAL state: instance_idx={instance_data.idx}, {root_state_idx}.{root_dlplan_state}", "red"))
                if satisfies_conditions_for_goal is not None and not satisfies_conditions_for_goal:
                    logging.warning(colored(f"[VERIFY] GOAL ext_state={(instance_data.idx, root_state_idx)} doesn't satisfy goal conditions {conditions_for_goal}", "magenta"))
                continue
            elif satisfies_conditions_for_goal is not None and satisfies_conditions_for_goal:
                logging.info(colored(f"NON-GOAL ext_state={(instance_data.idx, root_state_idx)} satisfies goal conditions {conditions_for_goal}", "magenta"))
                reason: Dict[str, Any] = {"non-goal": ((instance_data.idx, root_state_idx), self._state_get_satisfied_conditions(instance_data, root_state_idx, conditions_for_goal))}
                return False, None, reason

            is_deadend: bool = instance_data.is_deadend_state(root_state_idx)

            if is_deadend:
                logging.info(colored(f"Sketch reaches DEADEND for {instance_data.instance_filepath()}/{instance_data.idx}, {root_state_idx}.{root_dlplan_state}", "red"))
                state_idx_path: List[int] = []
                state_idx = root_state_idx
                while state_idx != -1:
                    state_idx_path.append(state_idx)
                    state_idx = parent.get(state_idx)
                state_idx_path = list(reversed(state_idx_path))
                for state_idx in state_idx_path:
                    logging.debug(colored(f"    {state_idx}.{instance_data.get_dlplan_state(*instance_data.get_state(state_idx))}", "red"))
                reason: Dict[str, Any] = {"deadend": tuple([(instance_data.idx, state_idx) for state_idx in state_idx_path])}
                return False, None, reason

            # Number rules whose conditions are satisfied at current state
            num_consistent_conditions = len(self._dlplan_policy.evaluate_conditions(root_dlplan_state))

            # Expand state (iterate over width-successors)
            timers["expansion"].resume()
            expanded += 1

            timers["successors"].resume()
            width_successors: List[Tuple[Tuple[int, Any], str]] = self._get_width_successors(instance_data, root_state_idx)
            generated += len(width_successors)
            random.shuffle(width_successors)
            timers["successors"].stop()

            num_alive_width_successors = 0
            for i, ((succ_state_idx, succ_state), action) in enumerate(width_successors):
                succ_dlplan_state = instance_data.get_dlplan_state(succ_state_idx, succ_state)
                logging.debug(f"[VERIFY]   SUCCESSOR: {succ_state_idx}.{succ_dlplan_state} [action={action}]")

                # Is transition compatible with policy?
                rule = self._dlplan_policy.evaluate(root_dlplan_state, succ_dlplan_state)
                if rule is not None:
                    subgoal_states_per_r_reachable_state[root_state_idx].add(succ_state_idx)
                    logging.debug(f"[VERIFY]               ADD: {root_state_idx} -> {succ_state_idx}")
                    logging.debug(f"                           RULE: {rule}")
                    if succ_state_idx not in visited:
                        visited.add(succ_state_idx)
                        queue.append(succ_state_idx)
                        parent[succ_state_idx] = root_state_idx
                    num_alive_width_successors += 1

                    # If randomized test, do limited exploration
                    if randomized_sketch_test is not None and num_alive_width_successors >= randomized_sketch_test:
                        break
            timers["expansion"].stop()

            # If no successors, this is a deadend state. It may fail above check as deadend check may be incomplete
            if len(width_successors) == 0:
                logging.debug(colored(f"[VERIFY] DEADEND state is r_reachable (no successors): instance_idx={instance_data.idx}, {root_state_idx}.{root_dlplan_state}", "red"))
                state_idx_path: List[int] = []
                state_idx = root_state_idx
                while state_idx != -1:
                    state_idx_path.append(state_idx)
                    state_idx = parent.get(state_idx)
                state_idx_path = list(reversed(state_idx_path))
                for state_idx in state_idx_path:
                    logging.debug(colored(f"    {state_idx}.{instance_data.get_dlplan_state(*instance_data.get_state(state_idx))}", "red"))
                reason: Dict[str, Any] = {"deadend": tuple([(instance_data.idx, state_idx) for state_idx in state_idx_path])}
                return False, None, reason

            if num_alive_width_successors == 0:
                if num_consistent_conditions == 0:
                    non_closed_ext_states.add((instance_data.idx, root_state_idx))
                    logging.info(colored(f"Sketch isn't CLOSED for {instance_data.instance_filepath()}/{instance_data.idx}, {root_state_idx}.{root_dlplan_state} [values={self._get_state_values(instance_data, root_state_idx)}]", "red"))
                else:
                    non_sound_ext_states.add((instance_data.idx, root_state_idx))
                    logging.info(colored(f"Sketch isn't SOUND for {instance_data.instance_filepath()}/{instance_data.idx}, {root_state_idx}.{root_dlplan_state} [values={self._get_state_values(instance_data, root_state_idx)}]", "red"))
                if len(non_closed_ext_states) + len(non_sound_ext_states) >= max_non_covered_ext_states:
                    break

        if len(non_closed_ext_states) > 0:
            reason: Dict[str, Any] = {"non-closed": non_closed_ext_states}
            return False, None, reason
        elif len(non_sound_ext_states) > 0:
            reason: Dict[str, Any] = {"non-sound": non_sound_ext_states}
            return False, None, reason
        else:
            logging.debug(colored(f"Sketch has BOUNDED WIDTH on {instance_data.instance_filepath()}/{instance_data.idx}", "blue"))
            return True, subgoal_states_per_r_reachable_state, None

    def _verify_acyclicity(self,
                           instance_data: InstanceData,
                           r_compatible_successors: Dict[int, int]) -> Tuple[bool, Dict[str, Set[Tuple[int, int]]]]:
        """
        Returns True iff sketch is acyclic, i.e., no infinite trajectories s1,s2,... are possible.
        """
        logging.debug(f"[VERIFY ACYCLICITY] INSTANCE_INDEX: {instance_data.idx}")

        state_idxs_generated: Set[int] = set()
        for root_state_idx, successors in r_compatible_successors.items():
            # The depth-first search is the iterative version where the current path is explicit in the stack.
            # https://en.wikipedia.org/wiki/Depth-first_search
            if root_state_idx not in state_idxs_generated:
                stack: List[Tuple[int, Any]] = [(root_state_idx, iter(successors))]
                #state_idx_path: List[int] = []
                #state_idxs_in_path: Set[int] = set()
                #frontier: Set[int] = set([root_state_idx])  # the generated states, to ensure that they are only added once to the stack
                state_idxs_in_path: Set[int] = set([root_state_idx])
                state_idxs_generated.add(root_state_idx)
                while stack:
                    state_idx, iterator = stack[-1]
                    assert state_idx in state_idxs_in_path
                    assert not instance_data.is_deadend_state(state_idx)
                    #state_idxs_in_path.add(state_idx)
                    #state_idx_path.append(state_idx)

                    if instance_data.is_goal_state(state_idx):
                        # Pop stack (backtrack)
                        stack.pop(-1)
                        state_idxs_in_path.discard(state_idx)
                    else:
                        try:
                            state_prime_idx = next(iterator)
                            if state_prime_idx in state_idxs_in_path:
                                # Cycle detected
                                state_idx_path: List[int] = [state_idx for state_idx, _ in stack] + [state_prime_idx]
                                logging.info(colored(f"Sketch CYCLES on {instance_data.instance_filepath()}/{instance_data.idx}", "red"))
                                logging.info(f"state_idxs in cycle: {list(state_idxs_in_path)}")
                                logging.info(f"Path (cyclic): {state_idx_path}")

                                logging.info(f"Transitions:")
                                for src_state_idx, dst_state_idx in zip(state_idx_path[:-1], state_idx_path[1:]):
                                    src_dlplan_state = self._state_factory.get_dlplan_state(instance_data.idx, src_state_idx)
                                    dst_dlplan_state = self._state_factory.get_dlplan_state(instance_data.idx, dst_state_idx)
                                    rule = self._dlplan_policy.evaluate(src_dlplan_state, dst_dlplan_state)
                                    logging.info(f"    {(src_state_idx, dst_state_idx)}: {rule}")
                                    assert rule is not None

                                assert False, "CYCLE"
                                return False, {"cycle": {(instance_data.idx, state_idx) for state_idx in state_idxs_in_path}}
                            elif state_prime_idx not in state_idxs_generated:
                                # Push state_idx into stack
                                stack.append((state_prime_idx, iter(r_compatible_successors.get(state_prime_idx, []))))
                                state_idxs_in_path.add(state_prime_idx)
                                state_idxs_generated.add(state_prime_idx)
                        except StopIteration:
                            # Pop stack (backtrack)
                            stack.pop(-1)
                            state_idxs_in_path.discard(state_idx)
                    """
                    try:
                        state_prime_idx = next(iterator)
                        if instance_data.is_goal_state(state_prime_idx):
                            continue
                        elif state_prime_idx in state_idxs_in_path:
                            logging.info(colored(f"Sketch CYCLES on {instance_data.instance_filepath()}/{instance_data.idx}", "red"))
                            logging.info(f"state_idxs in cycle: {state_idxs_in_path.union([state_prime_idx])}")
                            logging.info(f"Path (cyclic): {state_idx_path + [state_prime_idx]}")
                            if state_factory is not None:
                                logging.info(f"Transitions:")
                                for src_state_idx, dst_state_idx in zip(state_idx_path, state_idx_path[1:] + [state_prime_idx]):
                                    src_dlplan_state = state_factory.get_dlplan_state(instance_data.idx, src_state_idx)
                                    dst_dlplan_state = state_factory.get_dlplan_state(instance_data.idx, dst_state_idx)
                                    rule = self._dlplan_policy.evaluate(src_dlplan_state, dst_dlplan_state)
                                    logging.info(f"    {(src_state_idx, dst_state_idx)}: {rule}")
                            assert False, "CYCLE"
                            return False, {"cycle": {(instance_data.idx, state_idx) for state_idx in state_idxs_in_path.union([state_prime_idx])}}
                        elif state_prime_idx not in frontier:
                            frontier.add(state_prime_idx)
                            stack.append((state_prime_idx, iter(r_compatible_successors.get(state_prime_idx, []))))
                    except StopIteration:
                        state_idxs_in_path.discard(state_idx)
                        state_idx_path.pop()
                        stack.pop(-1)
                    """

        logging.debug(colored(f"Sketch is ACYCLIC on {instance_data.instance_filepath()}", "blue"))
        return True, None

    def _compute_state_b_values(self,
                                booleans: List[dlplan_policy.NamedBoolean],
                                numericals: List[dlplan_policy.NamedNumerical],
                                state: dlplan_core.State,
                                denotations_caches: dlplan_core.DenotationsCaches) -> Tuple[bool]:
        return tuple([boolean.get_element().evaluate(state, denotations_caches) for boolean in booleans] + [numerical.get_element().evaluate(state, denotations_caches) > 0 for numerical in numericals])

    def _verify_goal_separating_features(self,
                                         preprocessing_data: PreprocessingData,
                                         iteration_data: IterationData,
                                         instance_data: InstanceData) -> bool:
        goal_b_values = set()
        nongoal_b_values = set()
        booleans = self._dlplan_policy.get_booleans()
        numericals = self._dlplan_policy.get_numericals()
        for gfa_state_idx, gfa_state in enumerate(instance_data.gfa.get_states()):
            new_instance_idx = gfa_state.get_abstraction_index()
            new_instance_data = preprocessing_data.instance_datas[new_instance_idx]
            dlplan_state = preprocessing_data.state_finder.get_dlplan_ss_state(gfa_state)
            b_values = self._compute_state_b_values(booleans, numericals, dlplan_state, new_instance_data.denotations_caches)
            separating = True
            if instance_data.gfa.is_goal_state(gfa_state_idx):
                goal_b_values.add(b_values)
                if b_values in nongoal_b_values:
                    separating = False
            else:
                nongoal_b_values.add(b_values)
                if b_values in goal_b_values:
                    separating = False
            if not separating:
                print("Features do not separate goals from non goals")
                print("Booleans:")
                print("State:", str(dlplan_state))
                print("b_values:", b_values)
                return False
        return True

    def solves(self,
               preprocessing_data: PreprocessingData,
               iteration_data: IterationData,
               instance_data: InstanceData,
               **kwargs)  -> Tuple[bool, Dict[str, Any]]:
        """
        Returns True iff the sketch solves the instance, i.e.,
            (1) subproblems are safe,
            (2) sketch only classifies delta optimal state pairs as good,
            (3) sketch is acyclic, and
            (4) sketch features separate goals from nongoal states.
        """
        #logging.info(f"[solves] instance_data={instance_data}")

        test_goal_separating_features: bool = kwargs.get("test_goal_separating_features", False)
        randomized_sketch_test: int = kwargs.get("randomized_sketch_test", None)

        num_tests: int = 1 if randomized_sketch_test is None else 10
        subgoal_states_per_r_reachable_state: Dict[int, Set[int]] = defaultdict(set)
        logging.debug(colored(f"Verifying sketch solvability on {instance_data.instance_filepath()}/{instance_data.idx}{'' if num_tests == 1 else ' (' + str(num_tests) + ' times)'}", "blue"))
        for _ in range(num_tests):
            status, sample_subgoal_states_per_r_reachable_state, reason = self._verify_policy_for_bounded_width(preprocessing_data, iteration_data, instance_data, **kwargs)
            assert not status or reason is None
            if not status:
                return False, reason
            for key, value in sample_subgoal_states_per_r_reachable_state.items():
                subgoal_states_per_r_reachable_state[key] |= value

        logging.debug(f"*** WARNING ({Path(__file__)}: SKIPPING _verify_goal_separating_features")
        if test_goal_separating_features:
            goal_separation, failure_states = self._verify_goal_separating_features(preprocessing_data, iteration_data, instance_data)
            assert not goal_separation or len(failure_states) == 0
            if not goal_separation:
                return False, None

        status, reason = self._verify_acyclicity(instance_data, subgoal_states_per_r_reachable_state)
        assert not status or reason is None
        if not status:
            return False, reason

        logging.info(colored(f"Sketch SOLVES {instance_data.instance_filepath()}/{instance_data.idx}", "blue"))
        return True, None

    def minimize(self, policy_builder: Any):
        #return SketchReduced(PolicyMinimizer().minimize(self._dlplan_policy, policy_builder), self._width)
        raise RuntimeError("Policy minimization must be fixed; not yet done!")

    def print(self, logger: bool = False):
        sketch_lines: List[str] = str(self._dlplan_policy).splitlines()

        """
        features: Dict[str, List] = dict()
        for feature_type in ["(:booleans", "(:numericals"]:
            features_str = [line for line in sketch_lines if line.startswith(feature_type)][0]
            features_list = features_str[1+len(feature_type):-1].split(" ")
            if len(features_list) == 1:
                features[feature_type[2:]] = []
            else:
                fids = [int(item[2:]) for i, item in enumerate(features_list) if i % 2 == 0]
                names = [item[:-1].strip('"') for i, item in enumerate(features_list) if i % 2 == 1]
                features[feature_type[2:]] = list(zip(fids, names))
        """

        decoder: FeatureDecoder = FeatureDecoder()
        features: List[Tuple[int, int, str]] = []
        max_complexity: int = 0
        sum_complexity: int = 0
        for feature in list(self._dlplan_policy.get_booleans()) + list(self._dlplan_policy.get_numericals()):
            fid: int = int(feature.get_key()[1:])
            complexity: int = feature.get_element().compute_complexity()
            decoded: str = decoder(str(feature.get_element()))
            features.append((fid, complexity, decoded))
            max_complexity = max(max_complexity, complexity)
            sum_complexity += complexity
        features: List[Tuple[int, int, str]] = sorted(features, key=lambda t: t[1])

        if logger:
            for line in sketch_lines:
                logging.info(line)
            #logging.info(str(self._dlplan_policy))
            logging.info(f"Numer of sketch rules: {len(self._dlplan_policy.get_rules())}")
            logging.info(f"Number of selected features: {len(self._dlplan_policy.get_booleans()) + len(self._dlplan_policy.get_numericals())}")
            logging.info(f"Complexity of selected features: max={max_complexity}, sum={sum_complexity}")

            for fid, complexity, decoded in features:
                logging.info(f"Feature f{fid}/{complexity}: {decoded}")

        else:
            for line in sketch_lines:
                print(line)
            #print(str(self._dlplan_policy))
            print(f"Numer of sketch rules: {len(self._dlplan_policy.get_rules())}")
            print(f"Number of selected features: {len(self._dlplan_policy.get_booleans()) + len(self._dlplan_policy.get_numericals())}")
            print(f"Complexity of selected features: max={max_complexity}, sum={sum_complexity}")

            for fid, complexity, decoded in features:
                print(f"Feature f{fid}/{complexity}: {decoded}")

