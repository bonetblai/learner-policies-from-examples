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


class Node:
    # Class members that store initial and goals for search (must be updated before search with class method initialize())
    _width: int = None
    _ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]] = None
    _inits: FrozenSet[Tuple[int, int]] = None
    _goals: FrozenSet[Tuple[int, int]] = None
    _k_reachable: Dict[Tuple[int, int], intbitset] = None

    def __init__(self):
        self._branch_to_f_idx: Dict[str, int] = {"": -1}
        self._branch_to_num_r_idxs: Dict[str, int] = {"": -1}
        self._f_idxs: intbitset = intbitset()
        self._num_tips: int = 1
        self._cost: int = 0

        # Support for early termination (i.e., computation of sketches)
        self._preset: Set[Tuple[int, int]] = None
        self._TC: Dict[int, TransitiveClosure] = None

        # Invariant: state s is in termini iff (s0,s) in T* and s not in preset
        self._termini: Set[Tuple[int, int]] = None

    def __repr__(self):
        return f"Node[{[(key, self._branch_to_f_idx[key]) for key in sorted(self._branch_to_f_idx.keys())]}, cost={self._cost}, f_idxs={sorted(self._f_idxs)}]"

    def __lt__(self, other: "Node"):
        return self._cost < other._cost

    def _clone(self) -> "Node":
        clone: "Node" = Node()
        clone._branch_to_f_idx: Dict[str, int] = dict(self._branch_to_f_idx)
        clone._branch_to_num_r_idxs: Dict[str, int] = dict(self._branch_to_num_r_idxs)
        clone._f_idxs: intbitset = intbitset(self._f_idxs)
        clone._num_tips: int = self._num_tips
        clone._cost: int = self._cost
        clone._preset: Set[Tuple[int, int]] = set(self._preset)
        clone._TC: Dict[int, TransitiveClosure] = {instance_idx: TransitiveClosure(tc) for instance_idx, tc in self._TC.items()}
        return clone

    def _get_f_idxs_along_branch(self, branch: str) -> intbitset:
        f_idxs: List[int] = [self._branch_to_f_idx.get(branch[:i]) for i in range(1 + len(branch))]
        assert all([f_idx is not None and f_idx >= 0 for f_idx in f_idxs])
        return f_idxs

    def _create_successor(self, branch: str, g_idx: int, viewer: RuleViewer) -> "Node":
        #logging.info(f"[_create_successor] branch='{branch}', g_idx={g_idx}")
        assert self._branch_to_f_idx.get(branch) == -1
        assert len(viewer.r_idxs()) > 0
        # Assumption: rules changed by g_idxs associated with parent are "removed"

        # Create successor node that assigns g_idx to branch
        successor: "Node" = self._clone()
        successor._branch_to_f_idx[branch] = g_idx
        successor._num_tips -= 1
        if g_idx not in successor._f_idxs:
            successor._f_idxs.add(g_idx)
            successor._cost += viewer.f_idx_to_feature(g_idx).complexity - 1

        # Remove rules that change g_idx
        rules_that_change_g_idx: intbitset = viewer.r_idxs_that_change(g_idx)
        for r_idx in rules_that_change_g_idx: viewer.remove_rule(r_idx)

        # If there are remaining rules, create children
        if len(viewer.r_idxs()) > 0:
            partition_rules_by_bvalue_on_g_idx: Dict[Tuple[Tuple[int, int]], intbitset] = _partition_r_idxs_with_f_idx(viewer.r_idxs(), g_idx, viewer)
            for key, r_idxs in partition_rules_by_bvalue_on_g_idx.items():
                assert len(r_idxs) > 0 and len(key) == 1 and key[0][0] == g_idx and key[0][1] in [0, 1]
                child_branch: str = branch + str(key[0][1])
                assert successor._branch_to_f_idx.get(child_branch) is None
                successor._branch_to_f_idx[child_branch] = -1
                successor._branch_to_num_r_idxs[child_branch] = len(r_idxs)
                successor._num_tips += 1

        # Restore rules that change g_idx
        for r_idx in rules_that_change_g_idx: viewer.restore_rule(r_idx)

        # Calculate preset and termini
        if Node._width > 0:
            # Augment preset and transitive closure with k-reachable pairs that are compatible
            # with (the standard simplification of) a rule that changes g_idx
            f_idxs: List[int] = successor._get_f_idxs_along_branch(branch)
            for r_idx in rules_that_change_g_idx:
                #logging.info(f"Rule {r_idx} changes {g_idx}")
                ext_state: Tuple[int, int] = viewer.get_ext_state(r_idx)
                ext_edge: Tuple[int, Tuple[int, int]] = Node._ext_state_to_ext_edge[ext_state]
                ext_edge_src_values: np.ndarray = Node._ext_state_to_feature_valuations[(ext_edge[0], ext_edge[1][0])].take(f_idxs)
                ext_edge_dst_values: np.ndarray = Node._ext_state_to_feature_valuations[(ext_edge[0], ext_edge[1][1])].take(f_idxs)
                ext_edge_condition: np.ndarray = np.sign(ext_edge_src_values)
                ext_edge_change: np.ndarray = np.sign(ext_edge_dst_values - ext_edge_src_values)
                assert f_idxs[-1] == g_idx and ext_edge_change[len(f_idxs) - 1] != 0
                #logging.info(f"f_idxs={f_idxs}, r_idx={r_idx}, ext_edge={ext_edge}, ext_edge_condition={ext_edge_condition}, ext_edge_change={ext_edge_change}")
                for (instance_idx, src_state_idx), k_reachable in self._k_reachable.items():
                    for dst_state_idx in k_reachable:
                        ext_pair_src_values: np.ndarray = Node._ext_state_to_feature_valuations[(instance_idx, src_state_idx)].take(f_idxs)
                        ext_pair_dst_values: np.ndarray = Node._ext_state_to_feature_valuations[(instance_idx, dst_state_idx)].take(f_idxs)
                        ext_pair_condition: np.ndarray = np.sign(ext_pair_src_values)
                        ext_pair_change: np.ndarray = np.sign(ext_pair_dst_values - ext_pair_src_values)
                        if np.all(ext_edge_condition == ext_pair_condition) and np.all(ext_edge_change == ext_pair_change):
                            if instance_idx not in successor._TC: successor._TC[instance_idx] = TransitiveClosure()
                            if not successor._TC[instance_idx].edge(src_state_idx, dst_state_idx):
                                successor._TC[instance_idx].update(src_state_idx, dst_state_idx)
                                successor._preset.add((instance_idx, src_state_idx))
                                #logging.info(f"ADD TC: r_idx={r_idx}, edge={(src_state_idx, dst_state_idx)}")
            #logging.info(f"Preset: {sorted([state_idx for _, state_idx in successor._preset])}")

            # Calculate termini = { s : (s0,s) \in T* such that s \notin \pre(T) }
            successor._termini: Set[Tuple[int, int]] = set()
            for instance_idx, s0_idx in Node._inits:
                assert instance_idx in successor._TC
                successor._termini.update([(instance_idx, state_idx) for state_idx in successor._TC.get(instance_idx).start_at(s0_idx) if (instance_idx, state_idx) not in successor._preset])
            #logging.info(f"Termini: {successor._termini}")

        logging.debug(f"Successor: {successor}, termini={successor._termini}, terminal={successor.is_terminal()}")
        return successor

    # This only need to be called when nodes are manually created
    def _remove_solved_tips(self, viewer: RuleViewer):
        def _dfs_traversal(branch: str):
            f_idx: int = self._branch_to_f_idx.get(branch)
            if f_idx is None: return

            # Get active rules at this node (i.e., active given viewer filtered by boolean value of parent f_idx)
            if branch != "":
                parent_f_idx: int = self._branch_to_f_idx.get(branch[:-1])
                bvalue: int = int(branch[-1])
                partition_by_parent_f_idx: Dict[Tuple[Tuple[int, int]], intbitset] = _partition_r_idxs_with_f_idx(viewer.r_idxs(), parent_f_idx, viewer)
                rules_to_remove_due_to_parent_f_idx: intbitset = partition_by_parent_f_idx.get(((parent_f_idx, 1 - bvalue),), intbitset())
            else:
                rules_to_remove_due_to_parent_f_idx: intbitset = intbitset()

            for r_idx in rules_to_remove_due_to_parent_f_idx:
                viewer.remove_rule(r_idx)

            # A branch mapped to a valid f_idx represents an internal node; recurse on both children to collect successors
            if f_idx >= 0:
                assert len(viewer.r_idxs()) > 0

                # Remove rules that change f_idx
                rules_that_change_f_idx: intbitset = viewer.r_idxs_that_change(f_idx)
                for r_idx in rules_that_change_f_idx:
                    viewer.remove_rule(r_idx)

                # Recurse on children
                _dfs_traversal(branch + "0")
                _dfs_traversal(branch + "1")

                # Restore removed rules due to f_idx and parent_f_idx
                for r_idx in rules_that_change_f_idx:
                    viewer.restore_rule(r_idx)

            # else If no alive rules, this is new a TERMINAL node; mark as such and backtrack
            elif len(viewer.r_idxs()) == 0:
                self._branch_to_f_idx.pop(branch)
                self._num_tips -= 1

            # Restore rules removed to parent f_idx
            for r_idx in rules_to_remove_due_to_parent_f_idx:
                viewer.restore_rule(r_idx)

        # Perform a DFS traversal from root
        _dfs_traversal("")

    def _remove_rules_for_branch(self, branch: str, viewer: RuleViewer, restore_rules: bool) -> List[intbitset]:
        list_removed_rules: List[intbitset] = []
        for i, vertex in enumerate(branch):
            f_idx: int = self._branch_to_f_idx.get(branch[:i])
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

        # This is a proper tip branch is there are remaining rules
        assert len(viewer.r_idxs()) > 0, f'Improper tip branch "{branch}" for {self}'

        # Restore viewer if requested
        if restore_rules:
            for removed_rules in reversed(list_removed_rules):
                for r_idx in removed_rules: viewer.restore_rule(r_idx)

        return list_removed_rules

    @classmethod
    def initialize(cls,
                   width: int,
                   ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]],
                   ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray],
                   k_reachable: Dict[Tuple[int, int], intbitset]):
        cls._width: int = width
        if cls._width > 0:
            ext_edges_preset: FrozenSet[Tuple[int, int]] = frozenset([(instance_idx, src_state_idx) for instance_idx, (src_state_idx, _) in ext_state_to_ext_edge.values()])
            ext_edges_poset: FrozenSet[Tuple[int, int]] = frozenset([(instance_idx, dst_state_idx) for instance_idx, (_, dst_state_idx) in ext_state_to_ext_edge.values()])
            cls._ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]] = ext_state_to_ext_edge
            cls._ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = ext_state_to_feature_valuations
            cls._inits: FrozenSet[Tuple[int, int]] = ext_edges_preset - ext_edges_poset
            cls._goals: FrozenSet[Tuple[int, int]] = ext_edges_poset - ext_edges_preset
            cls._k_reachable: Dict[Tuple[int, int], intbitset] = k_reachable

    @classmethod
    def get_root_node(cls) -> "Node":
        root: "Node" = Node()
        root._TC: Dict[int, TransitiveClosure] = dict()
        root._preset: FrozenSet[Tuple[int, int]] = frozenset()
        root._termini: FrozenSet[Tuple[int, int]] = frozenset()
        return root

    def is_full_tree(self) -> bool:
        return self._num_tips == 0

    def is_terminal(self) -> bool:
        if self._width == 0:
            return self.is_full_tree()
        elif self.is_full_tree():
            return True
        else:
            # A node is terminal iff GOALS == TERMINI where GOALS determined by example paths,
            # and TERMINI = { s : (s0,s) \in Kleene such that s \notin \pre(T) }
            assert self._goals is not None and self._termini is not None, (self._goals, self._termini)
            return self._goals == self._termini

    def get_successors(self, branch_selection_heuristic: Callable["Node", str], viewer: RuleViewer, cost_bound: int, **kwargs) -> Tuple[List["Node"], int]:
        # Select branch to expand using provided heuristic
        branch: str = branch_selection_heuristic(self)
        assert branch is None or self._branch_to_f_idx.get(branch) == -1
        #print(f"Successors: {self}, branch={branch}")

        # If no branch available, there are no successors
        if branch is None: return [], int(1e6)

        # List of sets of removed rules
        list_removed_rules: List[intbitset] = self._remove_rules_for_branch(branch, viewer, restore_rules=False)

        # Calculate monotone features for branch
        splits: Set[FrozenSet] = set()
        successors: List["Node"] = []
        monotone_g_idxs: intbitset = viewer.monotone_features(kwargs.get("monotone_only_by_dec", False))

        # Generate successors for branch:
        # 1) one successor for each g_idx that is monotone for the set of non-removed rules at branch and that is changed by at least one such rule
        # 2) prune g_idxs that result in the same "split" (which is calculated with plain valuations and not boolean valuations)
        next_cost_bound: int = int(1e6)
        lower_bound = self._cost #+ (self._num_tips - 1) * viewer._min_complexity # Cannot add this term as tips can be resolved with features already included in node at no further cost
        for cost, g_idx in sorted([(viewer.f_idx_to_feature(g_idx).complexity - 1, g_idx) for g_idx in monotone_g_idxs]):
            if g_idx in self._f_idxs or cost_bound is None or lower_bound + cost <= cost_bound:
                r_idxs_that_change_g_idx: intbitset = viewer.r_idxs_that_change(g_idx)
                if len(r_idxs_that_change_g_idx) > 0:
                    partition: Dict[Tuple[Tuple[int, int]], intbitset] = _partition_r_idxs_with_f_idx(viewer.r_idxs() - r_idxs_that_change_g_idx, g_idx, viewer, boolean_projection=False)
                    split: FrozenSet = frozenset(partition.values())
                    if split not in splits:
                        heapq.heappush(successors, self._create_successor(branch, g_idx, viewer))
                        splits.add(split)
            else:
                next_cost_bound = min(next_cost_bound, lower_bound + cost)

        # Restore rules following bottom-up branch traversal
        for removed_rules in reversed(list_removed_rules):
            for r_idx in removed_rules: viewer.restore_rule(r_idx)

        #print(f"succ={successors}")
        return [heapq.heappop(successors) for _ in range(len(successors))], next_cost_bound

    def get_r_idx_to_info(self, viewer: RuleViewer) -> Dict[int, Tuple[int, intbitset]]:
        r_idxs: intbitset = viewer.r_idxs()
        r_idx_to_info: Dict[int, Tuple[int, intbitset]] = dict()
        for branch, f_idx in self._branch_to_f_idx.items():
            if f_idx != -1:
                above: intbitset = intbitset([self._branch_to_f_idx.get(branch[:i]) for i in range(len(branch))])
                list_removed_rules: List[intbitset] = self._remove_rules_for_branch(branch, viewer, restore_rules=False)

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

    def _calculate_reachable_pairs(self) -> Dict[Tuple[int, int], intbitset]:
        if self._width == 0:
            return None
        else:
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
                k_reachable[(instance_idx, src_idx)] = self._state_factory.explore(instance_idx, src_idx, self._width, caching=True) & state_idxs
            return k_reachable

    def _node_generator(self, branch_selection_heuristic: Callable[Node, str], **kwargs) -> Generator[Node, None, None]:
        def _rec_generator(depth: int, node: Node, cost_bound: int, prune_solutions_with_cost_bound: bool = True) -> Generator[Node, None, int]:
            self._num_generated[-1] += 1
            next_cost_bound: int = int(1e6)
            if node.is_terminal():
                self._num_terminals[-1] += 1
                if not prune_solutions_with_cost_bound or node._cost == cost_bound:
                    self._num_solutions[-1] += 1
                    yield node
            else:
                self._num_expanded[-1] += 1
                successors, next_cost_bound = node.get_successors(branch_selection_heuristic, self._viewer, cost_bound, **kwargs)
                # TODO: make get_successors() to be generator as there could be many successors
                for successor in successors:
                    ncb: int = yield from _rec_generator(1 + depth, successor, cost_bound, prune_solutions_with_cost_bound)
                    next_cost_bound = min(next_cost_bound, ncb)
            return next_cost_bound

        Node.initialize(self._width, self._ext_state_to_ext_edge, self._ext_state_to_feature_valuations, self._k_reachable)
        max_cost_bound: int = kwargs.get("max_cost_bound", 100)

        self._layers.append([])
        self._num_generated.append(0)
        self._num_expanded.append(0)
        self._num_terminals.append(0)
        self._num_solutions.append(0)
        last_num_generated: int = 0
        last_num_expanded: int = 0
        last_num_terminals: int = 0
        last_num_solutions: int = 0

        if True:
            cost_bound: int = 0
            while cost_bound <= max_cost_bound:
                self._layers[-1].append(cost_bound)
                next_cost_bound: int = yield from _rec_generator(0, Node.get_root_node(), cost_bound)
                logging.info(f"Cost-layer: cost={cost_bound}, #generated={self._num_generated[-1] - last_num_generated}, #expanded={self._num_expanded[-1] - last_num_expanded}, #terminals={self._num_terminals[-1] - last_num_terminals}, #solutions={self._num_solutions[-1] - last_num_solutions}")
                last_num_generated: int = self._num_generated[-1]
                last_num_expanded: int = self._num_expanded[-1]
                last_num_terminals: int = self._num_terminals[-1]
                last_num_solutions: int = self._num_solutions[-1]
                cost_bound: int = next_cost_bound
        else:
            yield from _rec_generator(0, Node.get_root_node(), max_cost_bound, prune_solutions_with_cost_bound=False)


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

        Node.initialize(self._width, self._ext_state_to_ext_edge, self._ext_state_to_feature_valuations, self._k_reachable)
        max_cost_bound: int = kwargs.get("max_cost_bound", 100)

        self._num_generated.append(0)
        self._num_expanded.append(0)
        self._num_terminals.append(0)
        self._num_solutions.append(0)

        cost: int = yield from _one_solution(0, Node.get_root_node(), max_cost_bound)
        while cost > 0:
            cost: int = yield from _one_solution(0, Node.get_root_node(), cost - 1)

    def _solutions(self, **kwargs) -> Generator[Any, None, None]:
        # Branch selection heuristic: select a tip with greatest number of r_idxs
        def _branch_selection_heuristic_1(node: Node) -> str:
            branches_with_score: List[Tuple[str, int]] = [(branch, node._branch_to_num_r_idxs.get(branch)) for branch, f_idx in node._branch_to_f_idx.items() if f_idx == -1]
            sorted_branches_with_score: List[Tuple[str, int]] = sorted(branches_with_score, key=lambda p: p[1], reverse=True)
            return None if len(sorted_branches_with_score) == 0 else sorted_branches_with_score[0][0]

        # Branch selection heuristic: select a shallowest tip
        def _branch_selection_heuristic_2(node: Node) -> str:
            branches_with_score: List[Tuple[str, int]] = [(branch, len(branch)) for branch, f_idx in node._branch_to_f_idx.items() if f_idx == -1]
            sorted_branches_with_score: List[Tuple[str, int]] = sorted(branches_with_score, key=lambda p: p[1], reverse=False)
            return None if len(sorted_branches_with_score) == 0 else sorted_branches_with_score[0][0]

        # Reset rule viewer
        self._viewer.reset()

        # Generate solutions in increasing order of cost
        max_num_solutions: int = kwargs.get("max_num_solutions", 100)
        for i, node in enumerate(self._node_generator(_branch_selection_heuristic_1, **kwargs)):
            logging.info(f"Got node {node}")
            yield node._f_idxs, node.get_r_idx_to_info(self._viewer)
            if i + 1 == max_num_solutions: break
        logging.debug(f"Nodes: #expanded={self._num_expanded}, #generated={self._num_generated}")

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

