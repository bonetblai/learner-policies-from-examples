import logging
import math
import random

from collections import defaultdict, deque
from termcolor import colored
from typing import Dict, Set, List, Deque, MutableSet, Tuple, Any, Optional
from pathlib import Path

import dlplan.core as dlplan_core
import dlplan.policy as dlplan_policy
from dlplan.policy import PolicyMinimizer

from .iteration_data import IterationData
from ..preprocessing import PreprocessingData, InstanceData
from ..util import Timer


class SketchReduced:
    def __init__(self, dlplan_policy: dlplan_policy.Policy, width: int):
        self.dlplan_policy = dlplan_policy
        self.width = width

    def _verify_policy_for_bounded_width(self,
                                         preprocessing_data: PreprocessingData,
                                         iteration_data: IterationData,
                                         instance_data: InstanceData,
                                         randomized_sketch_test: int,
                                         max_non_covered_ext_states: int,
                                         **kwargs) -> Tuple[bool, Dict[int, Set[int]], Dict[str, Any]]:
        """
        Performs forward search over R-reachable states.
        Initially, the R-reachable states are all initial states.
        For each R-reachable state there must be a satisfied subgoal tuple.
        If optimal width is required, we do not allow R-compatible states
        that are closer than the closest satisfied subgoal tuple.
        """
        debug = kwargs.get("debug", False)
        logging_level = logging.root.level
        new_logging_level = logging.DEBUG if debug else logging_level
        logging.root.setLevel(new_logging_level)

        logging.debug(f"VERIFY WIDTH: INSTANCE_INDEX: {instance_data.idx}")

        queue: Deque[int] = deque()
        visited: MutableSet[int] = set()
        parent: Dict[int, int] = dict()
        for initial_state_idx, _ in [instance_data.get_initial_state()]:
            logging.debug(f"VERIFY WIDTH: initial_state_idx={initial_state_idx}")
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

        expanded: int = 0
        generated: int = 0
        non_covered_ext_states: Set[Tuple[int, int]] = set()
        while queue:
            timers["search"].stop()
            if timers["search"].get_elapsed_sec() - last_record > reporting_period:
                last_record = timers["search"].get_elapsed_sec()
                elapsed_times: str = "(" + ", ".join([f"{timers[key].get_elapsed_sec():0.2f}" for key in ["search", "expansion", "successors"]]) + ")"
                logging.info(f"VERIFY WIDTH: elapsed_times={elapsed_times}, expanded={expanded}, generated={generated}, #queue={len(queue)}")
            timers["search"].resume()

            root_state_idx = queue.pop()
            _, root_state = instance_data.get_state(root_state_idx)
            root_dlplan_state = instance_data.get_dlplan_state(root_state_idx, root_state)

            logging.debug(f"VERIFY WIDTH: DEQUED: {root_state_idx}.{root_dlplan_state}")

            if instance_data.is_goal_state(root_state_idx):
                logging.debug(colored(f"VERIFY WIDTH: GOAL state: instance_idx={instance_data.idx}, {root_state_idx}.{root_dlplan_state}", "red"))
                continue

            logging.root.setLevel(logging_level)
            is_deadend: bool = instance_data.is_deadend_state(root_state_idx)
            logging.root.setLevel(new_logging_level)

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
                logging.root.setLevel(logging_level)
                return False, None, {"deadend": tuple([(instance_data.idx, state_idx) for state_idx in state_idx_path])}

            # Expand state
            timers["expansion"].resume()
            expanded += 1

            timers["successors"].resume()
            successors: List[Tuple[Tuple[int, Any], str]] = instance_data.get_successors(root_state_idx)
            generated += len(successors)
            random.shuffle(successors)
            timers["successors"].stop()

            num_alive_successors = 0
            for i, ((succ_state_idx, succ_state), action) in enumerate(successors):
                succ_dlplan_state = instance_data.get_dlplan_state(succ_state_idx, succ_state)
                logging.debug(f"VERIFY WIDTH:   SUCCESSOR: {succ_state_idx}.{succ_dlplan_state} [action={action}]")

                # Is transition compatible with policy?
                rule = self.dlplan_policy.evaluate(root_dlplan_state, succ_dlplan_state)
                if rule is not None:
                    subgoal_states_per_r_reachable_state[root_state_idx].add(succ_state_idx)
                    logging.debug(f"VERIFY WIDTH:               ADD: {root_state_idx} -> {succ_state_idx}")
                    logging.debug(f"                           RULE: {rule}")
                    if succ_state_idx not in visited:
                        visited.add(succ_state_idx)
                        queue.append(succ_state_idx)
                        parent[succ_state_idx] = root_state_idx
                    num_alive_successors += 1

                    # If randomized test, do limited exploration
                    if randomized_sketch_test is not None and num_alive_successors >= randomized_sketch_test:
                        break
            timers["expansion"].stop()

            # If no successors, this is a deadend state. It may fail above check as deadend check may be incomplete
            if len(successors) == 0:
                logging.debug(colored(f"VERIFY WIDTH: DEADEND state is r_reachable (no successors): instance_idx={instance_data.idx}, {root_state_idx}.{root_dlplan_state}", "red"))
                state_idx_path: List[int] = []
                state_idx = root_state_idx
                while state_idx != -1:
                    state_idx_path.append(state_idx)
                    state_idx = parent.get(state_idx)
                state_idx_path = list(reversed(state_idx_path))
                for state_idx in state_idx_path:
                    logging.debug(colored(f"    {state_idx}.{instance_data.get_dlplan_state(*instance_data.get_state(state_idx))}", "red"))
                logging.root.setLevel(logging_level)
                return False, None, {"deadend": tuple([(instance_data.idx, state_idx) for state_idx in state_idx_path])}

            if num_alive_successors == 0:
                non_covered_ext_states.add((instance_data.idx, root_state_idx))
                logging.info(colored(f"Sketch isn't CLOSED for {instance_data.instance_filepath()}/{instance_data.idx}, {root_state_idx}.{root_dlplan_state}", "red"))
                if len(non_covered_ext_states) >= max_non_covered_ext_states:
                    break

        if len(non_covered_ext_states) > 0:
            logging.root.setLevel(logging_level)
            return False, None, {"non-covered": non_covered_ext_states}
        else:
            logging.debug(colored(f"Sketch has BOUNDED WIDTH on {instance_data.instance_filepath()}/{instance_data.idx}", "blue"))
            logging.root.setLevel(logging_level)
            return True, subgoal_states_per_r_reachable_state, None

    def _verify_acyclicity(self,
                           instance_data: InstanceData,
                           r_compatible_successors: Dict[int, int],
                           state_factory: Optional[Any] = None) -> Tuple[bool, Dict[str, Set[Tuple[int, int]]]]:
        """
        Returns True iff sketch is acyclic, i.e., no infinite trajectories s1,s2,... are possible.
        """
        logging.debug(f"VERIFY ACYCLICITY: INSTANCE_INDEX: {instance_data.idx}")

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
                                if state_factory is not None:
                                    logging.info(f"Transitions:")
                                    for src_state_idx, dst_state_idx in zip(state_idx_path[:-1], state_idx_path[1:]):
                                        src_dlplan_state = state_factory.get_dlplan_state(instance_data.idx, src_state_idx)
                                        dst_dlplan_state = state_factory.get_dlplan_state(instance_data.idx, dst_state_idx)
                                        rule = self.dlplan_policy.evaluate(src_dlplan_state, dst_dlplan_state)
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
                                    rule = self.dlplan_policy.evaluate(src_dlplan_state, dst_dlplan_state)
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
        """
        Returns True iff sketch features separate goal from nongoal states.
        """
        goal_b_values = set()
        nongoal_b_values = set()
        booleans = self.dlplan_policy.get_booleans()
        numericals = self.dlplan_policy.get_numericals()
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
            (4) sketch features separate goals from nongoal states. """

        test_goal_separating_features: bool = kwargs.get("test_goal_separating_features", False)
        randomized_sketch_test: int = kwargs.get("randomized_sketch_test", None)

        num_tests: int = 1 if randomized_sketch_test is None else 10
        subgoal_states_per_r_reachable_state: Dict[int, Set[int]] = defaultdict(set)
        logging.info(colored(f"Verifying sketch solvability on {instance_data.instance_filepath()}/{instance_data.idx}{'' if num_tests == 1 else ' (' + str(num_tests) + ' times)'}", "blue"))
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

        status, reason = self._verify_acyclicity(instance_data, subgoal_states_per_r_reachable_state, kwargs.get("state_factory"))
        assert not status or reason is None
        if not status:
            return False, reason

        logging.info(colored(f"Sketch SOLVES {instance_data.instance_filepath()}/{instance_data.idx}", "blue"))
        return True, None

    def minimize(self, policy_builder: Any):
        #return SketchReduced(PolicyMinimizer().minimize(self.dlplan_policy, policy_builder), self.width)
        raise RuntimeError("Policy minimization must be fixed; not yet done!")

    def parse_feature(self, feature: str) -> List:
        # Split feature by top-level commas
        def split(f: str) -> List:
            i, level = 0, 0
            token, tokens = "", []
            while i < len(f):
                #print(f"DEBUG: f=|{f}|, i={i}, level={level}, char=|{f[i]}|, token=|{token}|, tokens={tokens}")
                char = f[i]
                if char == "(":
                    token += char
                    level += 1
                elif char == ")":
                    token += char
                    level -= 1
                elif char == "," and level == 0:
                    tokens.append(token)
                    token = ""
                else:
                    token += char
                i += 1
            if token != "":
                tokens.append(token)
                token = ""
            #print(f"DEBUG: f=|{f}|, i={i}, level={level}, char=END, token=|{token}|, tokens={tokens}")
            return tokens

        # Parse
        def parse(f: str, vars: List[int] = None, indent: int = 0) -> List[Any]:
            #print(f"parse:{'  ' * indent}f=|{f}|, vars={vars}")
            # Parse Boolean features
            if f.startswith("b_empty"):
                assert f.endswith(")")
                tokens = split(f[8:-1])
                assert len(tokens) == 1
                vars = [0, 1] if tokens[0].startswith("r_") else [0]
                return ["b_empty", vars, parse(tokens[0], vars, 1 + indent)]
            elif f.startswith("b_nullary"):
                assert f.endswith(")")
                tokens = split(f[10:-1])
                assert len(tokens) == 1
                return ["b_nullary", [], tokens[0]]
            elif f.startswith("b_"):
                raise RuntimeError(f"Unexpected Boolean |{f}|")

            # Parse numerical features
            elif f.startswith("n_count"):
                assert f.endswith(")")
                tokens = split(f[8:-1])
                assert len(tokens) == 1
                vars = [0, 1] if tokens[0].startswith("r_") else [0]
                return ["n_count", vars, parse(tokens[0], vars, 1 + indent)]
            elif f.startswith("n_concept_distance"):
                assert f.endswith(")")
                tokens = split(f[19:-1])
                assert len(tokens) == 3
                vars = [0, 1]
                return ["n_concept_distance", vars] + [parse(tokens[0], vars[:1], 1 + indent), parse(tokens[1], vars, 1 + indent), parse(tokens[2], vars[1:], 1 + indent)]
            elif f.startswith("n_"):
                raise RuntimeError(f"Unexpected numerical |{f}|")

            # Parse roles
            elif f.startswith("r_primitive"):
                assert f.endswith(")")
                tokens = split(f[12:-1])
                assert len(tokens) == 3
                return ["r_primitive", vars[-2:], tokens[0]]
            elif f.startswith("r_inverse"):
                assert f.endswith(")")
                tokens = split(f[10:-1])
                assert len(tokens) == 1
                return parse(tokens[0], [vars[-1], vars[-2]], 1 + indent)
            elif f.startswith("r_and"):
                assert f.endswith(")")
                tokens = split(f[6:-1])
                assert len(tokens) == 2
                return ["r_and", vars] + [parse(token, vars, 1 + indent) for token in tokens]
            elif f.startswith("r_transitive_closure"):
                assert f.endswith(")")
                tokens = split(f[21:-1])
                assert len(tokens) == 1
                return ["r_transitive_closure", vars, parse(tokens[0], vars, 1 + indent)]
            elif f.startswith("r_restrict"):
                assert f.endswith(")")
                tokens = split(f[11:-1])
                assert len(tokens) == 2
                return ["r_restrict", vars, parse(tokens[0], vars[-2:], 1 + indent), parse(tokens[1], vars[-1:], 1 + indent)]
            elif f.startswith("r_"):
                raise RuntimeError(f"Unexpected role |{f}|")

            # Parse concepts
            elif f.startswith("c_bot"):
                return ["c_bot", []]
            elif f.startswith("c_top"):
                return ["c_top", []]
            elif f.startswith("c_one_of"):
                assert f.endswith(")")
                tokens = split(f[9:-1])
                assert len(tokens) == 1
                return ["c_one_of", vars[-1:], tokens[0]]
            elif f.startswith("c_primitive"):
                assert f.endswith(")")
                tokens = split(f[12:-1])
                assert len(tokens) == 2
                return ["c_primitive", vars[-1:], tokens[0]]
            elif f.startswith("c_not"):
                assert f.endswith(")")
                tokens = split(f[6:-1])
                assert len(tokens) == 1
                return ["c_not", vars, parse(tokens[0], vars, 1 + indent)]
            elif f.startswith("c_and"):
                assert f.endswith(")")
                tokens = split(f[6:-1])
                assert len(tokens) == 2, tokens
                return ["c_and", vars] + [parse(token, vars, 1 + indent) for token in tokens]
            elif f.startswith("c_equal"):
                assert f.endswith(")")
                tokens = split(f[8:-1])
                assert len(tokens) == 2
                vars = vars + [1 + max(vars)]
                return ["c_equal", vars] + [parse(token, vars, 1 + indent) for token in tokens]
            elif f.startswith("c_some"):
                assert f.endswith(")")
                tokens = split(f[7:-1])
                assert len(tokens) == 2
                var = 1 + max(vars)
                return ["c_some", [vars[-1], var]] + [parse(token, vars + [var], 1 + indent) for token in tokens]
            elif f.startswith("c_all"):
                assert f.endswith(")")
                tokens = split(f[6:-1])
                assert len(tokens) == 2
                var = 1 + max(vars)
                return ["c_all", [vars[-1], var]] + [parse(token, vars + [var], 1 + indent) for token in tokens]
            elif f.startswith("c_"):
                raise RuntimeError(f"Unexpected concept |{f}|")

            # Unexpected
            else:
                raise RuntimeError(f"Unexpected |{f}|")

        return parse(feature)

    def decode_feature(self, feature: str) -> str:
        parsed: List = self.parse_feature(feature)
        #print(f"feature: {feature}")
        #print(f" parsed: {parsed}")
        def decode(f: List, formula: bool, indent: int = 0) -> str:
            #print(f"decode: {'  ' * indent}{f}")
            assert type(f) == list and len(f) > 0

            # Boolean features
            if f[0] == "b_empty":
                role_or_concept: str = decode(f[2], formula=False, indent=1 + indent)
                return f"Empty({role_or_concept})"
            elif f[0] == "b_nullary":
                return f"{f[2]()}" if formula else f"<nullary({f[2]})>"
            elif f[0].startswith("b_"):
                logging.warning(f"Unexpected Boolean {f}") 
                return f"<unexpected-boolean({f})>"

            # Numerical features
            elif f[0] == "n_count":
                role_or_concept: str = decode(f[2], formula=False, indent=1 + indent)
                return f"Cardinality({role_or_concept})"
            elif f[0] == "n_concept_distance":
                concept1: str = decode(f[2], formula=False, indent=1 + indent)
                role: str = decode(f[3], formula=True, indent=1 + indent)
                concept2: str = decode(f[4], formula=False, indent=1 + indent)
                return f"Distance({concept1}, {role}, {concept2}))"
            elif f[0].startswith("n_"):
                logging.warning(f"Unexpected numerical {f}") 
                return f"<unexpected-numerical({f})>"

            # Roles
            elif f[0] == "r_primitive":
                vars: List[str] = [f"x{i}" for i in f[1]]
                role: str = f[2]
                vpair: str = "(" + ",".join(vars[-2:]) + ")"
                return f"{role}{vpair}" if formula else f"{{{vpair} : {role}{vpair}}}"
            elif f[0] == "r_and":
                vars: List[str] = [f"x{i}" for i in f[1]]
                role1: str = decode(f[2], formula=True, indent=1 + indent)
                role2: str = decode(f[3], formula=True, indent=1 + indent)
                vpair: str = "(" + ",".join(vars[-2:]) + ")"
                return f"[{role1} & {role2}]" if formula else f"{{{vpair} : {role1} & {role2}}}"
            elif f[0].startswith("r_transitive_closure"):
                vars: List[str] = [f"x{i}" for i in f[1]]
                role: str = decode(f[2], formula=True, indent=1 + indent)
                vpair: str = "(" + ",".join(vars[-2:]) + ")"
                i = role.rfind(vpair)
                assert i != -1, f"role=|{role}|, vpair=|{vpair}|"
                naked_role: str = role[:i]
                tc_formula: str = f"TC[{naked_role}]{vpair}"
                return tc_formula if formula else f"{{{vpair} : {tc_formula}}}"
            elif f[0].startswith("r_restrict"):
                vars: List[str] = [f"x{i}" for i in f[1]]
                role: str = decode(f[2], formula=True, indent=1 + indent)
                concept: str = decode(f[3], formula=True, indent=1 + indent)
                vpair: str = "(" + ",".join(vars[-2:]) + ")"
                return f"[{role} & {concept}]" if formula else f"{{{vpair} : {role} & {concept}}}"
            elif f[0].startswith("r_"):
                logging.warning(f"Unexpected role {f}") 
                return f"<unexpected-role({f})>"

            # Concepts
            elif f[0] == "c_bot":
                return "False" if formula else "None"
            elif f[0] == "c_top":
                return "True" if formula else "All"
            elif f[0] == "c_one_of":
                var : str = f"x{f[1][-1]}"
                return f"[{var} = {f[2]}]" if formula else f"{{{f[2]}}}"
            elif f[0] == "c_primitive":
                vars: List[str] = [f"x{i}" for i in f[1]]
                concept: str = f[2]
                vpair: str = "(" + ",".join(vars) + ")"
                return f"{concept}{vpair}" if formula else f"{{{vars[0]} : {concept}{vpair}}}"
            elif f[0] == "c_not":
                vars: List[str] = [f"x{i}" for i in f[1]]
                concept: str = decode(f[2], formula, indent=1 + indent)
                vpair: str = "(" + ",".join(vars) + ")"
                return f"-{concept}" if formula else f"\compl({concept})"
            elif f[0] == "c_and":
                concept1: str = decode(f[2], formula, 1 + indent)
                concept2: str = decode(f[3], formula, 1 + indent)
                return f"[{concept1} & {concept2}]" if formula else f"[{concept1} \cap {concept2}]"
            elif f[0] == "c_equal":
                vars: List[str] = [f"x{i}" for i in f[1]]
                var, qvar = vars[0], vars[1]
                role1: str = decode(f[2], formula=True, indent=1 + indent)
                role2: str = decode(f[3], formula=True, indent=1 + indent)
                if formula:
                    return f"[Forall {qvar}.[{role1} <=> {role1}]]({var})"
                else:
                    return f"{{{var} : {{{qvar} : {role1}}} = {{{qvar} : {role2}}}}}"
            elif f[0] == "c_some":
                vars: List[str] = [f"x{i}" for i in f[1]]
                var, qvar = vars[0], vars[1]
                role: str = decode(f[2], formula=True, indent=1 + indent)
                concept: str = decode(f[3], formula=True, indent=1 + indent)
                if formula:
                    return f"[Exist {qvar}.[{role} & {concept}]]({var})"
                else:
                    return f"{{{var} : Exists {qvar}.[{role} & {concept}]}}"
            elif f[0] == "c_all":
                vars: List[str] = [f"x{i}" for i in f[1]]
                var, qvar = vars[0], vars[1]
                role: str = decode(f[2], formula=True, indent=1 + indent)
                concept: str = decode(f[3], formula=True, indent=1 + indent)
                if formula:
                    return f"[Forall {qvar}.[{role} => {concept}]]({var})"
                else:
                    return f"{{{var} : Forall {qvar}.[{role} => {concept}]}}"
            elif f[0].startswith("c_"):
                logging.warning(f"Unexpected concept {f}") 
                return f"<unexpected-concept({f})>"

            # Unexpected
            else:
                raise RuntimeError(f"Unexpected '{f}'")

        return decode(parsed, formula=False)

    def print(self, logger: bool = False):
        sketch_lines: List[str] = str(self.dlplan_policy).splitlines()

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

        if logger:
            for line in sketch_lines:
                logging.info(line)
            #logging.info(str(self.dlplan_policy))
            logging.info(f"Numer of sketch rules: {len(self.dlplan_policy.get_rules())}")
            logging.info(f"Number of selected features: {len(self.dlplan_policy.get_booleans()) + len(self.dlplan_policy.get_numericals())}")
            logging.info(f"Maximum complexity of selected feature: {max([0] + [boolean.get_element().compute_complexity() for boolean in self.dlplan_policy.get_booleans()] + [numerical.get_element().compute_complexity() for numerical in self.dlplan_policy.get_numericals()])}")

            for feature_type, features in features.items():
                for fid, feature in features:
                    logging.info(f"Feature f{fid}: {self.decode_feature(feature)}")

        else:
            for line in sketch_lines:
                print(line)
            #print(str(self.dlplan_policy))
            print(f"Numer of sketch rules: {len(self.dlplan_policy.get_rules())}")
            print(f"Number of selected features: {len(self.dlplan_policy.get_booleans()) + len(self.dlplan_policy.get_numericals())}")
            print(f"Maximum complexity of selected feature: {max([0] + [boolean.get_element().compute_complexity() for boolean in self.dlplan_policy.get_booleans()] + [numerical.get_element().compute_complexity() for numerical in self.dlplan_policy.get_numericals()])}")

