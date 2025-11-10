import os
import logging
import random
import heapq
import numpy as np
from intbitset import intbitset

from pathlib import Path
from termcolor import colored
from typing import Set, FrozenSet, Tuple, List, Union, Dict, Any, Optional, Union, Callable, Generator
from collections import defaultdict

import dlplan.core as dlplan_core

from ...state_space import StateFactory
from ..feature_pool import Feature
from ...util import Timer

from .rule_viewer import RuleViewer
from .transitive_closure import TransitiveClosure


def _partition_r_idxs_with_f_idx(r_idxs: intbitset, f_idx: int, viewer: RuleViewer, boolean_projection: bool = True) -> Dict[Tuple[Tuple[int, int]], intbitset]:
    partition: Dict[Tuple[Tuple[int, int]], intbitset] = defaultdict(intbitset)
    for r_idx in r_idxs:
        projection: Tuple[Tuple[int, int]] = viewer.project_condition(r_idx, f_idx, boolean_projection=boolean_projection)
        partition[projection].add(r_idx)
    return partition


# A node in the search tree represents an elimination tree for rules.
class Node:
    # Class members that store initial and goals for search (must be updated before search with class method initialize())
    _width: int = None
    _max_f_idxs: int = None
    _uniform_costs: bool = None
    _k_reachable: Dict[Tuple[int, int], intbitset] = None
    _ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]] = None
    _ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = None

    def __init__(self):
        # Branches:
        # 1) A branch is a {0,1}-string
        # 2) Branches are mapped to f_idxs, r_idxs at branch, and active segments
        # 3) Branches can be classified as OPEN or TERMINAL:
        #    - an OPEN branch needs to be assigned to f_idx or moved to TERMINAL
        #    - a branch is TERMINAL if all its segments can be solved with IW-search (includes case of no segments)
        # 4) A node is TERMINAL if it doesn't have OPEN branches
        self._branch_to_branch_idx: Dict[str, int] = None
        self._branch_idx_to_branch: List[str] = None
        self._branch_idx_to_f_idx: List[int] = None
        self._branch_idx_to_r_idxs: List[intbitset] = None
        self._branch_idx_to_segments: List[List[Tuple[Tuple[int, int]]]] = None
        self._OPEN: intbitset = None
        self._TERMINAL: intbitset = None

        # Other information:
        # - f_idxs is set of f_idxs assigned to branches (two branches can be assigned to same f_idx)
        # - cost is sum of the cost of f_idxs assigned to branches, repeated occurrences of f_idx are accounted
        self._f_idxs: intbitset = None
        self._cost: int = 0

    def __repr__(self):
        branches: str = ", ".join([str(p) for p in zip(self._branch_idx_to_branch, self._branch_idx_to_f_idx)])
        open_branches: str = ", ".join(sorted([f"'{self._branch_idx_to_branch[i]}'" for i in self._OPEN], key=lambda item: (len(item), item)))
        terminal_branches: str = ", ".join(sorted([f"'{self._branch_idx_to_branch[i]}'" for i in self._TERMINAL], key=lambda item: (len(item), item)))
        f_idxs: str = ", ".join([str(f_idx) for f_idx in sorted(self._f_idxs)])
        return f"Node[{{{branches}}}, OPEN={{{open_branches}}}, TERMINAL={{{terminal_branches}}}, cost={self._cost}, f_idxs={{{f_idxs}}}]"

    def __lt__(self, n: "Node"):
        return self._cost < n._cost

    def _clone(self) -> "Node":
        clone: "Node" = Node()
        clone._branch_to_branch_idx: Dict[str, int] = dict(self._branch_to_branch_idx)
        clone._branch_idx_to_branch: List[str] = list(self._branch_idx_to_branch)
        clone._branch_idx_to_f_idx: List[int] = list(self._branch_idx_to_f_idx)
        clone._branch_idx_to_r_idxs: List[intbitset] = list(self._branch_idx_to_r_idxs)                        # bitsets are shared between node and clone
        clone._branch_idx_to_segments: List[List[Tuple[Tuple[int, int]]]] = list(self._branch_idx_to_segments) # segments are shared between node and clone
        clone._OPEN: intbitset = intbitset(self._OPEN)
        clone._TERMINAL: intbitset = intbitset(self._TERMINAL)
        clone._f_idxs: intbitset = intbitset(self._f_idxs)
        clone._cost: int = self._cost
        return clone

    def _create_successor(self, branch_idx: int, g_idx: int, viewer: RuleViewer) -> "Node":
        assert self._branch_idx_to_f_idx[branch_idx] == -1
        assert branch_idx in self._OPEN
        assert len(viewer.r_idxs()) > 0
        # Assumption: rules changed by g_idxs associated with parent are "removed"

        # Create successor node that assigns g_idx to branch
        successor: "Node" = self._clone()
        successor._OPEN.remove(branch_idx)

        # If g_idx is None, the branch is solvable by IW and this successor just removes the branch from OPEN
        if g_idx is None:
            assert branch_idx in self._TERMINAL
            return successor

        # Update set of chosen f_idxs and cost
        successor._branch_idx_to_f_idx[branch_idx] = g_idx
        if g_idx not in successor._f_idxs:
            successor._f_idxs.add(g_idx)
            successor._cost += viewer.f_idx_to_feature(g_idx).complexity - 1 if not self._uniform_costs else 1

        # Remove rules that change g_idx
        r_idxs_for_node: intbitset = intbitset(viewer.r_idxs())
        rules_that_change_g_idx: intbitset = viewer.r_idxs_that_change(g_idx)
        for r_idx in rules_that_change_g_idx: viewer.remove_rule(r_idx)

        # If there are remaining rules, create children
        if len(viewer.r_idxs()) > 0:
            segments: List[Tuple[Tuple[int, int]]] = self._branch_idx_to_segments[branch_idx]
            partition_rules_by_bvalue_on_g_idx: Dict[Tuple[Tuple[int, int]], intbitset] = _partition_r_idxs_with_f_idx(viewer.r_idxs(), g_idx, viewer)
            for key, r_idxs in partition_rules_by_bvalue_on_g_idx.items():
                assert len(r_idxs) > 0 and len(key) == 1 and key[0][0] == g_idx and key[0][1] in [0, 1]
                new_branch: str = successor._branch_idx_to_branch[branch_idx] + str(key[0][1])
                new_branch_idx: int = len(successor._branch_idx_to_branch)
                successor._branch_to_branch_idx[new_branch] = new_branch_idx
                successor._branch_idx_to_branch.append(new_branch)
                successor._branch_idx_to_f_idx.append(-1)
                successor._branch_idx_to_r_idxs.append(r_idxs)

                # Calculate segments for new branch
                removed_r_idxs: intbitset = r_idxs_for_node - r_idxs
                removed_ext_states: FrozenSet[Tuple[int, int]] = frozenset([viewer.get_ext_state(r_idx) for r_idx in removed_r_idxs])
                segments_for_new_branch: List[Tuple[Tuple[int, int]]] = []
                for segment in segments:
                    ext_states_in_segment: FrozenSet[Tuple[int, int]] = frozenset(segment)
                    indices: List[int] = [-1] + sorted([segment.index(ext_state) for ext_state in removed_ext_states if ext_state in ext_states_in_segment]) + [len(segment) - 1]
                    non_trivial_sub_segments: List[Tuple[Tuple[int, int]]] = [segment[1 + start:1 + end] for start, end in zip(indices[:-1], indices[1:]) if end - start > 2]
                    segments_for_new_branch.extend(non_trivial_sub_segments)
                successor._branch_idx_to_segments.append(segments_for_new_branch)

                # Insert new branch into OPEN and TERMINAL sets
                successor._OPEN.add(new_branch_idx)
                if successor.is_terminal_branch(new_branch_idx):
                    successor._TERMINAL.add(new_branch_idx)

        # Restore rules that change g_idx
        for r_idx in rules_that_change_g_idx: viewer.restore_rule(r_idx)

        return successor

    def _remove_rules_for_branch(self, branch_idx: int, viewer: RuleViewer, restore_rules: bool) -> List[intbitset]:
        branch: str = self._branch_idx_to_branch[branch_idx]
        list_removed_rules: List[intbitset] = []
        for i, vertex in enumerate(branch):
            prefix_idx: int = self._branch_to_branch_idx.get(branch[:i])
            f_idx: int = self._branch_idx_to_f_idx[prefix_idx]
            bvalue: int = int(vertex)
            assert f_idx is not None and f_idx >= 0 and bvalue in [0, 1]

            # Remove rules that change f_idx
            list_removed_rules.append(viewer.r_idxs_that_change(f_idx))
            assert len(list_removed_rules[-1]) > 0
            for r_idx in list_removed_rules[-1]: viewer.remove_rule(r_idx)

            # Remove rules whose condition is inconsistent with boolean valuation for f_idx
            partition_by_bvalue_on_f_idx: Dict[Tuple[Tuple[int, int]], intbitset] = _partition_r_idxs_with_f_idx(viewer.r_idxs(), f_idx, viewer)
            list_removed_rules.append(partition_by_bvalue_on_f_idx.get(((f_idx, 1 - bvalue),), intbitset()))
            for r_idx in list_removed_rules[-1]: viewer.remove_rule(r_idx)

        # This is a proper terminal branch if there are remaining rules
        assert len(viewer.r_idxs()) > 0, f'Improper tip branch "{branch}" for {self}'

        # Restore viewer if requested
        if restore_rules:
            for removed_rules in reversed(list_removed_rules):
                for r_idx in removed_rules: viewer.restore_rule(r_idx)

        return list_removed_rules

    @classmethod
    def initialize(cls,
                   width: int,
                   max_f_idxs: int,
                   uniform_costs: bool,
                   k_reachable: Dict[Tuple[int, int], intbitset],
                   ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]],
                   ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray]):
        cls._width: int = width
        cls._max_f_idxs: int = max_f_idxs
        cls._uniform_costs: bool = uniform_costs
        cls._k_reachable: Dict[Tuple[int, int], intbitset] = k_reachable
        cls._ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]] = ext_state_to_ext_edge
        cls._ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = ext_state_to_feature_valuations

    @classmethod
    def get_root_node(cls, r_idxs_for_root_node: intbitset, ex_paths: List[Tuple[Tuple[int, int]]]) -> "Node":
        root: "Node" = Node()
        root._branch_to_branch_idx: Dict[str, int] = {"": 0}
        root._branch_idx_to_branch: List[str] = [""]
        root._branch_idx_to_f_idx: List[int] = [-1]
        root._branch_idx_to_r_idxs: Lit[intbitset] = [r_idxs_for_root_node]
        segments_for_root_node: List[Tuple[Tuple[int, int]]] = [path for path in ex_paths if len(path) > 2]
        root._branch_idx_to_segments: List[List[Tuple[Tuple[int, int]]]] = [segments_for_root_node]
        root._OPEN: intbitset = intbitset([0])
        root._TERMINAL: intbitset = intbitset([0]) if root.is_terminal_branch(0) else intbitset()
        root._f_idxs: intbitset = intbitset()
        root._cost: int = 0
        return root

    def is_terminal(self, verbose: bool = False) -> bool:
        terminal: bool = len(self._OPEN) == 0
        if verbose and terminal: print(f"{self} is TERMINAL")
        return terminal

    def is_terminal_branch(self, branch_idx: int, verbose: bool = False) -> bool:
        segments: List[Tuple[Tuple[int, int]]] = self._branch_idx_to_segments[branch_idx]
        if verbose: print(f"Branch={branch_idx}.'{self._branch_idx_to_branch[branch_idx]}', r_idxs={sorted(self._branch_idx_to_r_idxs[branch_idx])}, segments={segments}")
        for segment in segments:
            assert len(segment) > 2
            reachable: Set[Tuple[int, int]] = self._exploration([segment[0]])
            if verbose: print(f"Exploration: segment={segment}, reachable={sorted(reachable)}, terminal={segment[-1] in reachable}")
            if segment[-1] not in reachable: return False
        if verbose: print(f"Branch is TERMINAL")
        return True

    def _exploration(self, initial_ext_states: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        reachable: Set[Tuple[int, int]] = set()
        for instance_idx, src_idx in initial_ext_states:
            for state_idx in self._k_reachable.get((instance_idx, src_idx)):
                reachable.add((instance_idx, state_idx))
        return reachable

    def get_successors(self, branch_selection_heuristic: Callable["Node", str], viewer: RuleViewer, cost_bound: int, **kwargs) -> Tuple[List["Node"], int]:
        # Select branch to expand using provided heuristic
        branch_idx: int = branch_selection_heuristic(self)
        assert branch_idx in self._OPEN and self._branch_idx_to_f_idx[branch_idx] == -1
        terminal_branch: bool = self.is_terminal_branch(branch_idx)

        # List of sets of removed rules
        list_removed_rules: List[intbitset] = self._remove_rules_for_branch(branch_idx, viewer, restore_rules=False)

        # Calculate monotone features for branch
        monotone_g_idxs: intbitset = viewer.monotone_features(kwargs.get("monotone_only_by_dec", False))
        splits: Set[FrozenSet] = set()
        successors: List["Node"] = []

        # At the end, next_cost_bound must lower bound the cost of successors "pruned by cost bound"
        next_cost_bound: int = int(1e6)
        lower_bound: int = self._cost
        #lower_bound: int = self._cost + (len(self._OPEN - self._TERMINAL) - 1) * (1 if self._uniform_costs else viewer._min_complexity)
        #print(f"LowerBound: branch_idx={branch_idx}")

        # Generate successors for branch:
        # 1) one successor for each g_idx that is monotone for the set of non-removed rules at branch and that is changed by at least one such rule
        # 2) prune g_idxs that result in the same "split" (which is calculated with plain valuations and not boolean valuations)
        # 3) Additionally, if branch is terminal, the same node appears as successor but with the branch moved to non_terminals

        for cost, g_idx in sorted([(viewer.f_idx_to_feature(g_idx).complexity - 1, g_idx) for g_idx in monotone_g_idxs]):
            revised_cost: int = 1 if self._uniform_costs else cost
            if g_idx in self._f_idxs or ((cost_bound is None or lower_bound + revised_cost <= cost_bound) and (self._max_f_idxs is None or len(self._f_idxs) < self._max_f_idxs)):
                r_idxs_that_change_g_idx: intbitset = viewer.r_idxs_that_change(g_idx)
                if len(r_idxs_that_change_g_idx) > 0:
                    partition: Dict[Tuple[Tuple[int, int]], intbitset] = _partition_r_idxs_with_f_idx(viewer.r_idxs() - r_idxs_that_change_g_idx, g_idx, viewer, boolean_projection=False)
                    split: FrozenSet = frozenset(partition.values())
                    if split not in splits:
                        heapq.heappush(successors, self._create_successor(branch_idx, g_idx, viewer))
                        splits.add(split)
            elif g_idx not in self._f_idxs and cost_bound is not None and (self._max_f_idxs is None or len(self._f_idxs) < self._max_f_idxs):
                # Branch isn't assigned to g_idx because it violated cost bound
                next_cost_bound = min(next_cost_bound, lower_bound + revised_cost)

        # If branch is terminal, create successor where branch is just removed from OPEN
        if branch_idx in self._TERMINAL:
            heapq.heappush(successors, self._create_successor(branch_idx, None, viewer))

        # Restore rules following bottom-up branch traversal
        for removed_rules in reversed(list_removed_rules):
            for r_idx in removed_rules: viewer.restore_rule(r_idx)

        assert next_cost_bound > cost_bound
        return [heapq.heappop(successors) for _ in range(len(successors))], next_cost_bound

    def get_r_idx_to_info(self, viewer: RuleViewer) -> Dict[int, Tuple[int, intbitset]]:
        r_idxs: intbitset = viewer.r_idxs()
        r_idx_to_info: Dict[int, Tuple[int, intbitset]] = dict()
        for branch_idx, f_idx in enumerate(self._branch_idx_to_f_idx):
            if f_idx != -1:
                branch: str = self._branch_idx_to_branch[branch_idx]
                above: intbitset = intbitset([self._branch_idx_to_f_idx[self._branch_to_branch_idx.get(branch[:i])] for i in range(len(branch))])
                list_removed_rules: List[intbitset] = self._remove_rules_for_branch(branch_idx, viewer, restore_rules=False)

                # Fill r_idx_to_info data structure
                for r_idx in viewer.r_idxs_that_change(f_idx):
                    assert r_idx not in r_idx_to_info
                    r_idx_to_info[r_idx] = (f_idx, intbitset(above))

                # Restore rules following bottom-up branch traversal
                for removed_rules in reversed(list_removed_rules):
                    for r_idx in removed_rules: viewer.restore_rule(r_idx)

        return r_idx_to_info


class SolutionGenerator:
    # Statistics
    _layers: List[List[int]] = []
    _num_generated: List[int] = []
    _num_expanded: List[int] = []
    _num_terminals: List[int] = []
    _num_solutions: List[int] = []

    def __init__(self, preprocessing_data: Dict[str, Any], state_factory: StateFactory, **kwargs):
        self._preprocessing_data: Dict[str, Any] = preprocessing_data
        self._state_factory: StateFactory = state_factory
        assert self._preprocessing_data is not None

        self._requirements_for_good_transitions: Dict[Tuple[int, int], intbitset] = self._preprocessing_data.get("requirements_for_good_transitions")
        self._goal_ext_pair_to_separating_features: Dict[Tuple[int, Tuple[int, int]], intbitset] = self._preprocessing_data.get("goal_ext_pair_to_separating_features")
        #self._deadend_path_to_separating_features: Dict[Tuple[Tuple[int, int]], List[intbitset]] = self._preprocessing_data.get("deadend_path_to_separating_features")
        self._ext_state_to_separating_features_for_deadend_paths: Dict[Tuple[int, int], Dict[Tuple[int, Tuple[int, int]], intbitset]] = self._preprocessing_data.get("ext_state_to_separating_features_for_deadend_paths")
        self._ext_sibling_to_separating_features: Dict[Tuple[int, int, Tuple[int, int]], intbitset] = self._preprocessing_data.get("ext_sibling_to_separating_features")
        self._ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]] = self._preprocessing_data.get("ext_state_to_ext_edge")
        self._ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = self._preprocessing_data.get("ext_state_to_feature_valuations")
        self._ex_paths: List[Tuple[Tuple[int, int]]] = self._preprocessing_data.get("ex_paths")

        # Features
        self._relevant_features: List[Tuple[int, Feature]] = self._preprocessing_data.get("relevant_features")
        self._numerical_features: List[Tuple[int, Feature]] = [(f_idx, feature) for f_idx, feature in self._relevant_features if isinstance(feature.dlplan_feature, dlplan_core.Numerical)]
        self._numerical_f_idxs: intbitset = intbitset([f_idx for f_idx, _ in self._numerical_features])

        # Rule viewer
        self._viewer: RuleViewer = self._preprocessing_data.get("rule_viewer")

        # Construct requirements, one per ex-edge and one per pair of goal and non-goal xstates
        self._annotated_requirements: List[Tuple[Dict[str, Any], intbitset]] = []
        self._annotated_requirements.extend([({"key": "Edge", "ext_state": ext_state}, requirement) for ext_state, requirement in self._requirements_for_good_transitions.items()])
        self._annotated_requirements.extend([({"key": "Goal", "pair": pair}, separating_features) for pair, separating_features in self._goal_ext_pair_to_separating_features.items()])
        self._annotated_requirements.extend([({"key": "Deadend", "ext_state": ext_state, "path": path}, separating_features) for ext_state, separating_features_for_deadend_paths in self._ext_state_to_separating_features_for_deadend_paths.items() for path, separating_features in separating_features_for_deadend_paths.items()])
        self._annotated_requirements.extend([({"key": "Sibling", "ext_sibling": ext_sibling}, separating_features) for ext_sibling, separating_features in self._ext_sibling_to_separating_features.items()])
        self._requirements: List[intbitset] = [requirement for _, requirement in self._annotated_requirements]
        self._requirements_for_goals: List[intbitset] = [requirement for annotation, requirement in self._annotated_requirements if annotation.get("key") == "Goal"]
        self._requirements_for_deadends: List[intbitset] = [requirement for annotation, requirement in self._annotated_requirements if annotation.get("key") == "Deadend"]
        self._num_requirements: Dict[str, int] = {key: sum([1 if annotation.get("key") == key else 0 for annotation, _ in self._annotated_requirements]) for key in ["Edge", "Goal", "Deadend", "Sibling"]}
        logging.info(f"{len(self._requirements)} requirement(s) split as {self._num_requirements}")

        # Calculate k-reachable pairs
        self._width: int = kwargs.get("width", 0)
        local_timer: Timer = Timer()
        self._k_reachable: Dict[Tuple[int, int], intbitset] = self._calculate_reachable_pairs()
        local_timer.stop()
        logging.info(f"{sum([len(reachable) for reachable in self._k_reachable.values()]) if self._width > 0 else 0} reachability pair(s) calculated in {local_timer.get_elapsed_sec():0.2f} second(s)")

        # Other options
        self._max_f_idxs: int = kwargs.get("max_f_idxs", int(1e6))
        self._uniform_costs: bool = kwargs.get("uniform_costs", False)

    def _calculate_reachable_pairs(self) -> Dict[Tuple[int, int], intbitset]:
        # Get ext-edges in example paths
        ext_edges: List[Tuple[int, Tuple[int, int]]] = list(self._ext_state_to_ext_edge.values())
        instance_idx_to_src_idxs: Dict[int, intbitset] = defaultdict(intbitset)
        instance_idx_to_dst_idxs: Dict[int, intbitset] = defaultdict(intbitset)
        for instance_idx, (src_idx, dst_idx) in ext_edges:
            instance_idx_to_src_idxs[instance_idx].add(src_idx)
            instance_idx_to_dst_idxs[instance_idx].add(dst_idx)
        instance_idx_to_state_idxs: Dict[int, intbitset] = {instance_idx: src_idxs | instance_idx_to_dst_idxs[instance_idx] for instance_idx, src_idxs in instance_idx_to_src_idxs.items()}

        # For each ext-edge, do a k-width exploration to calculate k-reachable states
        k_reachable: Dict[Tuple[int, int], intbitset] = dict()
        for instance_idx, (src_idx, dst_idx) in ext_edges:
            state_idxs: intbitset = instance_idx_to_state_idxs[instance_idx]
            explored: FrozenSet[Tuple[int, Tuple[int, int]]] = self._state_factory.exploration(instance_idx, src_idx, self._width, caching=True, verbose=False)
            reachable: intbitset = intbitset([src_idx] + [dst_idx for _, (_, dst_idx) in explored])
            k_reachable[(instance_idx, src_idx)] = reachable & state_idxs
        return k_reachable

    def _node_generator(self, branch_selection_heuristic: Callable[Node, str], **kwargs) -> Generator[Node, None, None]:
        def _rec_generator(depth: int, node: Node, cost_bound: int, prune_solutions_with_cost_bound: bool = True) -> Generator[Node, None, int]:
            self._num_generated[-1] += 1
            next_cost_bound: int = int(1e6)
            assert node._cost <= cost_bound
            if node.is_terminal(verbose=not prune_solutions_with_cost_bound or node._cost == cost_bound):
                self._num_terminals[-1] += 1
                if not prune_solutions_with_cost_bound or node._cost == cost_bound:
                    self._num_solutions[-1] += 1
                    yield node
            else:
                self._num_expanded[-1] += 1
                successors, next_cost_bound = node.get_successors(branch_selection_heuristic, self._viewer, cost_bound, **kwargs)
                # CHECK: WHAT IS NEXT COST BOUND?
                for successor in successors:
                    ncb: int = yield from _rec_generator(1 + depth, successor, cost_bound, prune_solutions_with_cost_bound)
                    next_cost_bound = min(next_cost_bound, ncb)
            return next_cost_bound

        max_cost_bound: int = kwargs.get("max_cost_bound", 100)
        Node.initialize(self._width, self._max_f_idxs, self._uniform_costs, self._k_reachable, self._ext_state_to_ext_edge, self._ext_state_to_feature_valuations)

        self._layers.append([])
        self._num_generated.append(0)
        self._num_expanded.append(0)
        self._num_terminals.append(0)
        self._num_solutions.append(0)
        last_num_generated: int = 0
        last_num_expanded: int = 0
        last_num_terminals: int = 0
        last_num_solutions: int = 0

        root_node: Node = Node.get_root_node(self._viewer.all_rules(), self._ex_paths)

        if True:
            cost_bound: int = 0
            while cost_bound <= max_cost_bound:
                self._layers[-1].append(cost_bound)
                next_cost_bound: int = yield from _rec_generator(0, root_node, cost_bound)
                logging.info(f"Cost-layer: cost={cost_bound}, #generated={self._num_generated[-1] - last_num_generated}, #expanded={self._num_expanded[-1] - last_num_expanded}, #terminals={self._num_terminals[-1] - last_num_terminals}, #solutions={self._num_solutions[-1] - last_num_solutions}")
                last_num_generated: int = self._num_generated[-1]
                last_num_expanded: int = self._num_expanded[-1]
                last_num_terminals: int = self._num_terminals[-1]
                last_num_solutions: int = self._num_solutions[-1]
                cost_bound: int = next_cost_bound
        else:
            yield from _rec_generator(0, root_node, max_cost_bound, prune_solutions_with_cost_bound=False)

    def _node_generator_v2(self, branch_selection_heuristic: Callable[Node, str], **kwargs) -> Generator[Node, None, None]:
        def _one_solution(depth: int, node: Node, cost_bound: int) -> Generator[Node, None, int]:
            if node.is_terminal():
                if node._cost <= cost_bound:
                    self._solutions[-1] += 1
                    yield node
                    return node._cost
            else:
                self._num_expanded[-1] += 1
                successors, next_cost_bound = node.get_successors(branch_selection_heuristic, self._viewer, cost_bound, **kwargs)
                # TODO: make get_successors() to be generator as there could be many successors
                for successor in successors:
                    self._num_generated[-1] += 1
                    cost: int = yield from _one_solution(1 + depth, successor, cost_bound)
                    if cost > 0: return cost
            return -1

        Node.initialize(self._width, self._ext_state_to_ext_edge, self._ext_state_to_feature_valuations, self._k_reachable, self._max_f_idxs, self._uniform_costs)
        max_cost_bound: int = kwargs.get("max_cost_bound", 100)

        self._num_generated.append(0)
        self._num_expanded.append(0)
        self._num_terminals.append(0)
        self._num_solutions.append(0)

        root_node: Node = Node.get_root_node()
        cost: int = yield from _one_solution(0, root_node, max_cost_bound)
        while cost > 0:
            cost: int = yield from _one_solution(0, root_node, cost - 1)

    def _solutions(self, **kwargs) -> Generator[Any, None, None]:
        # Branch selection heuristic: select a tip with greatest number of r_idxs
        def _branch_selection_heuristic_1(node: Node) -> str:
            assert len(node._OPEN) > 0
            branches_with_score: List[Tuple[int, int]] = [(branch_idx, len(node._branch_idx_to_r_idxs[branch_idx])) for branch_idx in node._OPEN]
            sorted_branches_with_score: List[Tuple[int, int]] = sorted(branches_with_score, key=lambda p: p[1], reverse=True)
            return sorted_branches_with_score[0][0]

        # Branch selection heuristic: select a shallowest tip
        def _branch_selection_heuristic_2(node: Node) -> str:
            assert len(node._OPEN) > 0
            branches_with_score: List[Tuple[int, int]] = [(branch_idx, len(node._branch_idx_to_branch[branch_idx])) for branch_idx in node._OPEN]
            sorted_branches_with_score: List[Tuple[int, int]] = sorted(branches_with_score, key=lambda p: p[1], reverse=False)
            return sorted_branches_with_score[0][0]

        # Reset rule viewer
        self._viewer.reset()

        # Generate solutions in increasing order of cost
        max_num_solutions: int = kwargs.get("max_num_solutions")
        for i, node in enumerate(self._node_generator(_branch_selection_heuristic_1, **kwargs)):
            logging.info(f"Yield {1 + i} {node}")
            yield node._f_idxs, node.get_r_idx_to_info(self._viewer)
            if max_num_solutions is not None and max_num_solutions <= 1 + i: break
        logging.info(f"Nodes: #expanded={self._num_expanded}, #generated={self._num_generated}")

    def __call__(self, **kwargs) -> Generator[Any, None, None]:
        yield from self._solutions(**kwargs)

    @classmethod
    def get_node_statistics(cls) -> Dict[str, Any]:
        return {
            "searches": len(cls._layers),
            "num_layers": [len(layer) for layer in cls._layers],
            "layers": list(cls._layers),
            "generated": list(cls._num_generated),
            "expanded": list(cls._num_expanded),
            "terminals": list(cls._num_terminals),
            "solutions": list(cls._num_solutions),
        }

