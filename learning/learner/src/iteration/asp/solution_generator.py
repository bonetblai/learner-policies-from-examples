import os
import logging
import random
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
from .policy_finalizer import PolicyFinalizer
from .exceptions import NoFeature, MaxRestarts


def _partition_r_idxs_with_f_idx(r_idxs: intbitset, f_idx: int, viewer: RuleViewer, boolean_projection: bool = True) -> Dict[Tuple[Tuple[int, int]], intbitset]:
    partition: Dict[Tuple[Tuple[int, int]], intbitset] = defaultdict(intbitset)
    for r_idx in r_idxs:
        projection: Tuple[Tuple[int, int]] = viewer.project_condition(r_idx, f_idx, boolean_projection=boolean_projection)
        partition[projection].add(r_idx)
    return partition


class Backtrack(RuntimeError):
    # Parameters set by initialize()
    _max_restarts: int = None
    _max_backtracks: int = None
    _backtrack_depth: int = None

    # Class variables
    _num_restarts: int = None
    _num_backtracks: int = None
    _seq: int = 0

    @classmethod
    def initialize(cls, max_restarts: int, max_backtracks: int, backtrack_depth: int):
        cls._max_restarts: int = max_restarts
        cls._max_backtracks: int = max_backtracks
        cls._backtrack_depth: int = backtrack_depth
        cls._num_backtracks: int = None
        cls._num_restarts: int = 0

    @classmethod
    def reset(cls):
        cls._num_backtracks: int = 0

    def __init__(self, depth):
        self._id: int = Backtrack._seq
        self._depth: int = depth
        Backtrack._seq += 1

    def __str__(self):
        return f"BT[id={self._id}, depth={self._depth}]"


# A branch in an elimination tree. A node consists of several branches.
class Branch:
    @classmethod
    def make_root_branch(cls, r_idxs: intbitset, ex_paths: List[Tuple[Tuple[int, int]]], r_idxs_other: intbitset = None) -> "Branch":
        branch: "Branch" = Branch(0, "")
        branch._r_idxs: intbitset = r_idxs
        branch._r_idxs_other: intbitset = r_idxs_other or intbitset()
        branch._segments: List[Tuple[Tuple[int, int]]] = [path for path in ex_paths if len(path) > 2]
        return branch

    def __init__(self, branch_idx: int, branch_id: str):
        self._idx: int = branch_idx
        self._id: str = branch_id
        self._f_idx: int = -1
        self._r_idxs: intbitset = None
        self._r_idxs_change: intbitset = None
        self._r_idxs_other: intbitset = None
        self._r_idxs_other_change: intbitset = None
        self._segments: List[Tuple[Tuple[int, int]]] = None
        self._f_idx_str: str = None

    def __repr__(self):
        f_idx: str = self._f_idx_str or str(self._f_idx)
        r_idxs: List[int] = sorted(self._r_idxs)
        r_idxs_change: List[int] = sorted(self._r_idxs_change or [])
        r_idxs_other: List[int] = sorted(self._r_idxs_other)
        r_idxs_other_change: List[int] = sorted(self._r_idxs_other_change or [])
        return f"Branch['{self._id}', f_idx={f_idx}, r_idxs={r_idxs}, change={r_idxs_change}, other={r_idxs_other}, other_change={r_idxs_other_change}, segments={self._segments}]"

    def _clone(self) -> "Branch":
        clone: "Branch" = Branch(self._idx, self._id)
        clone._f_idx: int = self._f_idx
        clone._r_idxs: intbitset = self._r_idxs
        clone._r_idxs_change: intbitset = self._r_idxs_change
        clone._r_idxs_other: intbitset = self._r_idxs_other
        clone._r_idxs_other_change: intbitset = self._r_idxs_other_change
        clone._segments: List[Tuple[Tuple[int, int]]] = self._segments
        return clone


# A node in the search tree represets an elimination tree that is made of several branches.
class Node:
    # Class members that must be initialized with class method before search starts
    _width: int = None
    _max_features: int = None
    _uniform_costs: bool = None
    _k_reachable: Dict[Tuple[int, int], intbitset] = None
    _ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]] = None
    _ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = None

    @classmethod
    def initialize(cls,
                   width: int,
                   max_features: int,
                   uniform_costs: bool,
                   k_reachable: Dict[Tuple[int, int], intbitset],
                   ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]],
                   ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray]):
        cls._width: int = width
        cls._max_features: int = max_features
        cls._uniform_costs: bool = uniform_costs
        cls._k_reachable: Dict[Tuple[int, int], intbitset] = k_reachable
        cls._ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]] = ext_state_to_ext_edge
        cls._ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = ext_state_to_feature_valuations

    @classmethod
    def make_root_node(cls, r_idxs: intbitset, ex_paths: List[Tuple[Tuple[int, int]]], r_idxs_other: intbitset = None) -> "Node":
        root: "Node" = Node()
        root._branch_id_to_branch_idx: Dict[str, int] = {"": 0}
        root._branch_idx_to_branch: List[Branch] = [Branch.make_root_branch(r_idxs, ex_paths, r_idxs_other)]
        root._OPEN: intbitset = intbitset([0])
        root._TERMINAL: intbitset = intbitset([0]) if root.is_terminal_branch(0) else intbitset()
        root._f_idxs: intbitset = intbitset()
        root._cost: int = 0
        root._score: Tuple[Union[int, float]] = None
        return root

    @classmethod
    def _exploration(cls, initial_ext_states: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        reachable: Set[Tuple[int, int]] = set()
        for instance_idx, src_idx in initial_ext_states:
            for state_idx in cls._k_reachable.get((instance_idx, src_idx)):
                reachable.add((instance_idx, state_idx))
        return reachable

    def __init__(self):
        self._branch_id_to_branch_idx: Dict[str, int] = None
        self._branch_idx_to_branch: List[Branch] = None
        self._OPEN: intbitset = None
        self._TERMINAL: intbitset = None
        self._f_idxs: intbitset = None
        self._cost: int = 0
        self._score: Tuple[Union[int, float]] = None

    def __repr__(self):
        branches: str = ", ".join([str((branch._id, branch._f_idx)) for branch in self._branch_idx_to_branch])
        open_branches: str = ", ".join(sorted([f"'{self.get_branch(i)._id}'" for i in self._OPEN], key=lambda item: (len(item), item)))
        terminal_branches: str = ", ".join(sorted([f"'{self.get_branch(i)._id}'" for i in self._TERMINAL], key=lambda item: (len(item), item)))
        f_idxs: str = ", ".join([str(f_idx) for f_idx in sorted(self._f_idxs)])
        return f"Node[{{{branches}}}, OPEN={{{open_branches}}}, TERMINAL={{{terminal_branches}}}, cost={self._cost}, f_idxs={{{f_idxs}}}]"

    def __lt__(self, n: "Node"):
        return self._score < n._score

    def _clone(self) -> "Node":
        clone: "Node" = Node()
        clone._branch_id_to_branch_idx: Dict[str, int] = dict(self._branch_id_to_branch_idx)
        clone._branch_idx_to_branch: List[Branch] = list(self._branch_idx_to_branch)    # Branches shared between node and clone
        clone._OPEN: intbitset = intbitset(self._OPEN)
        clone._TERMINAL: intbitset = intbitset(self._TERMINAL)
        clone._f_idxs: intbitset = intbitset(self._f_idxs)
        clone._cost: int = self._cost
        clone._score: Tuple[Union[int, float]] = self._score
        return clone

    def _make_successor(self, branch_idx: int, g_idx: int, viewer: RuleViewer, partition_g_idx: Dict[Tuple[Tuple[int, int]], intbitset] = None) -> "Node":
        # Assumption: rules changed by g_idxs associated with parent are "removed"
        assert len(viewer.r_idxs()) > 0
        assert branch_idx in self._OPEN
        assert self.get_branch(branch_idx)._f_idx == -1, str(self)

        # Create successor node that assigns g_idx to branch
        successor: "Node" = self._clone()
        successor._OPEN.remove(branch_idx)

        # If g_idx is None, the branch is solvable by IW and this successor just removes the branch from OPEN
        if g_idx is None:
            assert branch_idx in self._TERMINAL
            assert len(self.get_branch(branch_idx)._r_idxs_other) == 0
            return successor

        # Clone branch
        branch: Branch = self.get_branch(branch_idx)._clone()
        successor._branch_idx_to_branch[branch_idx] = branch
        assert branch._r_idxs_change is None

        # Update set of chosen f_idxs and cost
        branch._f_idx: int = g_idx
        g_idx_complexity: int = viewer.f_idx_to_feature(g_idx).complexity - 1 if not self._uniform_costs else 1
        branch._f_idx_str: str = f"{g_idx}.{viewer.f_idx_to_feature(g_idx)._dlplan_feature}"
        g_idx_in_f_idxs: bool = g_idx in successor._f_idxs
        if g_idx not in successor._f_idxs:
            successor._f_idxs.add(g_idx)
            successor._cost += g_idx_complexity

        # Calculate alive r_idxs
        r_idxs_alive: intbitset = viewer.r_idxs()
        assert (branch._r_idxs | branch._r_idxs_other) == r_idxs_alive, f"r_idx={sorted(branch._r_idxs)}, other={sorted(branch._r_idxs_other)}, alive={sorted(r_idxs_alive)}"

        # Rules/ext_states that change g_idx
        r_idxs_that_change_g_idx: intbitset = viewer.r_idxs_that_change(g_idx)
        branch._r_idxs_change: intbitset = branch._r_idxs & r_idxs_that_change_g_idx
        branch._r_idxs_other_change: intbitset = branch._r_idxs_other & r_idxs_that_change_g_idx
        ext_states_that_change_g_idx: FrozenSet[Tuple[int, int]] = frozenset([viewer.get_ext_state(r_idx) for r_idx in branch._r_idxs_change])

        # Split other r_idxs by value of g_idx
        r_idxs_other_by_value: Dict[int, intbitset] = {key[0][1]: block for key, block in _partition_r_idxs_with_f_idx(branch._r_idxs_other, g_idx, viewer).items()}

        # Split segments by change of g_idx
        decode = lambda ext_edge: ((ext_edge[0], ext_edge[1][0]), (ext_edge[0], ext_edge[1][1]))
        coalesced_r_idx_segments_by_value: List[List[Tuple[Tuple[int, int]]]] = [[], []]
        for segment in branch._segments:
            assert len(segment) > 0
            indices: List[int] = [0] + [1 + i for i, ext_state in enumerate(segment) if ext_state in ext_states_that_change_g_idx] + [len(segment)]
            sub_segments: List[Tuple[Tuple[int, int]]] = [segment[start : 1 + end] for start, end in zip(indices[:-1], indices[1:])]
            assert len(sub_segments) > 0
            r_idxs_in_sub_segments: List[intbitset] = [intbitset([viewer.get_r_idx(ext_state) for ext_state in sub_segment[:-1]]) for sub_segment in sub_segments]
            r_idx_blocks: List[Tuple[int, intbitset]] = [(key[0][1], block) for r_idxs in r_idxs_in_sub_segments for key, block in _partition_r_idxs_with_f_idx(r_idxs, g_idx, viewer).items()]
            r_idx_segments_by_value: List[List[List[int]]] = [[], []]
            for sub_segment in sub_segments:
                sub: List[int] = []
                sub_type: Tuple[int, int] = None
                for i, ext_state in enumerate(sub_segment[:-1]):
                    r_idx: int = viewer.get_r_idx(ext_state)
                    if r_idx is None:
                        # ext_state is the last (goal) state in the segment, skip it
                        continue
                    else:
                        r_idx_type: List[Tuple[int, int]] = [(key, j) for j, (key, block) in enumerate(r_idx_blocks) if r_idx in block]
                        # Check that r_idx belongs to a unique block; i.e., its type contains exactly 1 match
                        assert len(r_idx_type) == 1, f"sub_segment={sub_segment}, r_idx_blocks={r_idx_blocks}, r_idx={r_idx}, r_idx_type={r_idx_type}"
                        if sub_type is None or sub_type == r_idx_type[0]:
                            sub.append(r_idx)
                        else:
                            r_idx_segments_by_value[sub_type[0]].append(sub)
                            sub: List[int] = []
                        sub_type: Tuple[int, int] = r_idx_type[0]
                if len(sub) > 0: r_idx_segments_by_value[sub_type[0]].append(sub)
            coalesced_r_idx_segments_by_value[0].extend(r_idx_segments_by_value[0])
            coalesced_r_idx_segments_by_value[1].extend(r_idx_segments_by_value[1])

        # Construct child branches
        for key, r_idx_segments in enumerate(coalesced_r_idx_segments_by_value):
            r_idxs_other: intbitset = r_idxs_other_by_value.get(key, intbitset()) - r_idxs_that_change_g_idx
            if len(r_idx_segments) > 0 or len(r_idxs_other) > 0:
                new_branch_id: str = branch._id + str(key)
                new_branch_idx: int = len(successor._branch_idx_to_branch)
                new_branch: Branch = Branch(new_branch_idx, new_branch_id)
                successor._branch_id_to_branch_idx[new_branch_id] = new_branch_idx
                successor._branch_idx_to_branch.append(new_branch)

                # Segments
                segments: List[Tuple[Tuple[int, int]]] = [tuple(viewer.get_ext_state(r_idx) for r_idx in segment[:-1]) + decode(Node._ext_state_to_ext_edge.get(viewer.get_ext_state(segment[-1]))) for segment in r_idx_segments]
                new_branch._segments: List[Tuple[Tuple[int, int]]] = segments

                # Rules for new branch
                new_branch._r_idxs: intbitset = (branch._r_idxs & intbitset([r_idx for r_idx_segment in r_idx_segments for r_idx in r_idx_segment])) - branch._r_idxs_change
                new_branch._r_idxs_other: intbitset = r_idxs_other

                # Classify new branch as OPEN, TERMINAL, or both
                if successor.is_terminal_branch(new_branch_idx):
                    successor._TERMINAL.add(new_branch_idx)
                    if len(new_branch._r_idxs) > 0:
                        successor._OPEN.add(new_branch_idx)
                else:
                    successor._OPEN.add(new_branch_idx)

        # Check that the new successor node is sound
        if not successor.verify():
            print("Successor:")
            print(f"  Branch='{branch._id}', g_idx={g_idx}")
            successor.dump("  ")
            assert successor.verify(2, True)

        # Setup score that is used when ordering successors by max score
        r_idxs_to_remove: intbitset = branch._r_idxs_change | branch._r_idxs_other_change
        if partition_g_idx is None:
            partition_g_idx: Dict[Tuple[Tuple[int, int]], intbitset] = _partition_r_idxs_with_f_idx(viewer.r_idxs(), g_idx, viewer)
        block_sizes: List[int] = [len(block - r_idxs_to_remove) for _, block in partition_g_idx.items() if len(block - r_idxs_to_remove) > 0]
        num_blocks: int = len(block_sizes)
        std_block_sizes: float = -np.std(block_sizes) if num_blocks > 0 else 0
        monotone_by_dec: int = 1 if viewer.is_monotone(g_idx, monotone_only_by_dec=True) else 0
        successor._score: Tuple[Union[int, float]] = (monotone_by_dec, len(r_idxs_to_remove) / g_idx_complexity, 1 if g_idx_in_f_idxs else 0, num_blocks, std_block_sizes)

        return successor

    def _remove_rules_for_branch(self, branch_idx: int, viewer: RuleViewer, restore_rules: bool, verbose: bool = False) -> List[intbitset]:
        branch: Branch = self.get_branch(branch_idx)
        list_removed_rules: List[intbitset] = []
        for i, vertex in enumerate(branch._id):
            prefix_idx: int = self.get_branch_idx(branch._id[:i])
            f_idx: int = self.get_branch(prefix_idx)._f_idx
            bvalue: int = int(vertex)
            assert f_idx >= 0 and bvalue in [0, 1]
            if verbose: print(f"_remove_rules_for_branch: branch={branch}, i={i}, vertex={vertex}, prefix_id={branch._id[:i]}, prefix_idx={prefix_idx}, f_idx={f_idx}, bvalue={bvalue}")

            # Remove rules that change f_idx
            list_removed_rules.append(viewer.r_idxs_that_change(f_idx))
            if len(list_removed_rules[-1]) == 0: self.dump()
            assert len(list_removed_rules[-1]) > 0, str(list_removed_rules)
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

    def verify(self, indent: int = 0, verbose: bool = False) -> bool:
        if not self.verify_r_idxs(indent, verbose): return False
        if not self.verify_segmentation(indent, verbose): return False
        return True

    def verify_r_idxs(self, indent: int = 0, verbose: bool = False) -> bool:
        for branch in self._branch_idx_to_branch:
            if branch._r_idxs_change is not None and not branch._r_idxs_change.issubset(branch._r_idxs):
                if verbose: print(f"{' ' * indent}r_idxs={sorted(branch._r_idxs_change)} that change for '{branch._id}'/f_idx={branch._f_idx} isn't subset of r_idxs={sorted(branch._r_idxs)}")
                return False

            if len(branch._r_idxs & branch._r_idxs_other) > 0:
                if verbose: print(f"{' ' * indent}r_idxs={sorted(branch._r_idxs)} isn't disjoint of other r_idxs={sorted(branch._r_idxs_other)} for '{branch._id}'/f_idx={branch._f_idx}")
                return False

            if branch._id + "0" in self._branch_id_to_branch_idx or branch._id + "1" in self._branch_id_to_branch_idx:
                # This is a parent branch
                children: List[Branch] = [self.get_branch(self.get_branch_idx(branch._id + x)) for x in ["0", "1"] if branch._id + x in self._branch_id_to_branch_idx]
                r_idxs: initbitset = intbitset.union(*[child_branch._r_idxs for child_branch in children])
                if r_idxs != (branch._r_idxs - branch._r_idxs_change):
                    if verbose: print(f"{' ' * indent}Union of child r_idxs={sorted(r_idxs)} isn't equal to non-change r_idxs={sorted(branch._r_idxs - branch._r_idxs_change)} for branch '{branch._id}'")
                    return False
        return True

    def verify_segmentation(self, indent: int = 0, verbose: bool = False) -> bool:
        entire_segments: List[Tuple[Tuple[int, int]]] = self.get_branch(0)._segments
        branch_ids: List[str] = [branch._id for branch in self._branch_idx_to_branch]
        branch_id_prefixes: FrozenSet[str] = frozenset([branch_id[:i] for branch_id in branch_ids for i in range(0, len(branch_id))])
        tip_branch_ids: List[str] = [branch_id for branch_id in branch_ids if branch_id not in branch_id_prefixes]

        segments: List[Tuple[Tuple[int, int]]] = [segment for branch_id in tip_branch_ids for segment in self.get_branch(self.get_branch_idx(branch_id))._segments]
        segment_dict: Dict[Tuple[int, int], Tuple[Tuple[int, int]]] = {segment[0]: segment for segment in segments}
        for entire_segment in entire_segments:
            tip_ext_state: Tuple[int, int] = entire_segment[0]
            reconstruction: List[Tuple[int, int]] = [tip_ext_state]
            if verbose: print(f"{' ' * indent}Reconstruction {reconstruction}")
            while segment_dict.get(tip_ext_state) is not None:
                reconstruction.extend(segment_dict.get(tip_ext_state)[1:])
                tip_ext_state: Tuple[int, int] = reconstruction[-1]
                assert tip_ext_state not in reconstruction[:-1], f"entire_segment={entire_segment}, reconstruction={reconstruction}"
                if verbose: print(f"{' ' * indent}Reconstruction {reconstruction}")
            if tuple(reconstruction) != entire_segment:
                if verbose:
                    print(f"{' ' * indent}Entire segment {entire_segment} cannot be reconstructed for length={branch_length}:")
                    print(f"{' ' * indent}Reconstruction {reconstruction}")
                    print(f"{' ' * indent}Tip_branch_ids {tip_branch_ids}")
                    print(f"{' ' * indent}      Segments {segments}")
                    print(f"{' ' * indent}          Dict {segment_dict}")
                    print(f"{' ' * indent} Tip_ext_state {tip_ext_state}")
                return False
        return True

    def dump(self, prefix: str = "", indent: int = 0):
        print(f"{' ' * indent}{prefix}{self}")
        for branch_idx, branch in enumerate(self._branch_idx_to_branch):
            print(f"{' ' * (indent + len(prefix))}{branch_idx}.{branch}")

    def get_branch(self, branch_idx: int) -> Branch:
        return self._branch_idx_to_branch[branch_idx]

    def get_branch_idx(self, branch_id: str) -> int:
        return self._branch_id_to_branch_idx.get(branch_id)

    def is_terminal(self, verbose: bool = False) -> bool:
        terminal: bool = len(self._OPEN) == 0
        if verbose and terminal: print(f"{self} is TERMINAL")
        return terminal

    def is_terminal_branch(self, branch_idx: int, verbose: bool = False) -> bool:
        # Branch is terminal iff there are no other r_idxs AND every segment is solved by IW
        branch: Branch = self.get_branch(branch_idx)
        if verbose: print(str(branch))
        if len(branch._r_idxs_other) > 0:
            return False
        elif len(branch._r_idxs) == 0:
            return True
        elif self._width == 0:
            return False
        else:
            if verbose: print(f"Segments: {branch._segments}")
            for segment in branch._segments:
                assert len(segment) > 1
                if len(segment) == 2:
                    solved: bool = True
                    if verbose: print(f"Exploration: segment={segment}, solved={solved}")
                else:
                    reachable: Set[Tuple[int, int]] = Node._exploration([segment[0]])
                    solved: bool = segment[-1] in reachable
                    if verbose: print(f"Exploration: segment={segment}, reachable={sorted(reachable)}, solved={solved}")
                if not solved: return False
            if verbose: print(f"Branch is TERMINAL")
            return True

    def get_successors(self, branch_selection_heuristic: Callable["Node", Tuple[int, bool]], viewer: RuleViewer, cost_bound: int, **kwargs) -> Tuple[List["Node"], int, bool]:
        # Select branch to expand using provided heuristic
        branch_idx, random_tie_breaking_for_branch = branch_selection_heuristic(self)
        if kwargs.get("verbose", False): print(f"branch_idx={branch_idx}, random_tie_breaking={random_tie_breaking_for_branch}")
        if branch_idx is None: return [], int(1e6), random_tie_breaking_for_branch
        branch: Branch = self.get_branch(branch_idx)
        assert branch_idx in self._OPEN and branch._f_idx == -1

        # List of sets of removed rules
        list_removed_rules: List[intbitset] = self._remove_rules_for_branch(branch_idx, viewer, restore_rules=False, verbose=kwargs.get("verbose", False))
        if kwargs.get("verbose", False): print(f"list_removed_rules={list_removed_rules}")

        # Calculate monotone features for branch
        monotone_g_idxs: intbitset = viewer.monotone_features(kwargs.get("monotone_only_by_dec", False))
        if kwargs.get("verbose", False): print(f"monotone_g_idxs={monotone_g_idxs}")

        # At the end, next_cost_bound must lower bound the cost of successors "pruned by cost bound"
        next_cost_bound: int = int(1e6)
        lower_bound: int = self._cost
        #lower_bound: int = self._cost + (len(self._OPEN - self._TERMINAL) - 1) * (1 if self._uniform_costs else viewer._min_complexity)
        if kwargs.get("verbose", False):
            print(f"next_cost_bound={next_cost_bound}, lower_bound={lower_bound}")
            print(f"cost_bound={cost_bound}, max_features={self._max_features}, #f_idxs={len(self._f_idxs)}")

        # Generate successors for branch:
        # 1) one successor for each g_idx that is monotone for the set of non-removed rules at branch and that is changed by at least one such rule
        # 2) prune g_idxs that result in the same "split" (which is calculated with plain valuations and not boolean valuations)
        # 3) Additionally, if branch is terminal, the same node appears as successor but with the branch moved to non_terminals

        successors: List["Node"] = []
        for cost, g_idx in sorted([(viewer.f_idx_to_feature(g_idx).complexity - 1, g_idx) for g_idx in monotone_g_idxs]):
            revised_cost: int = 1 if self._uniform_costs else cost
            if g_idx in self._f_idxs or ((cost_bound is None or lower_bound + revised_cost <= cost_bound) and (self._max_features is None or len(self._f_idxs) < self._max_features)):
                r_idxs_that_change_g_idx: intbitset = viewer.r_idxs_that_change(g_idx)
                if kwargs.get("verbose", False):
                    print(f"  Case 1: cost={cost}, g_idx={g_idx}, r_idxs={sorted(viewer.r_idxs())}, r_idxs_that_change_g_idx={sorted(r_idxs_that_change_g_idx)}")
                if len((branch._r_idxs | branch._r_idxs_other) & r_idxs_that_change_g_idx) > 0:
                    successors.append(self._make_successor(branch_idx, g_idx, viewer))
            elif g_idx not in self._f_idxs and cost_bound is not None and (self._max_features is None or len(self._f_idxs) < self._max_features):
                if kwargs.get("verbose", False): print(f"  Case 2: cost={cost}, g_idx={g_idx}")
                # Branch isn't assigned to g_idx because it violates cost bound
                next_cost_bound = min(next_cost_bound, lower_bound + revised_cost)
            else:
                if kwargs.get("verbose", False): print(f"  Case 3: cost={cost}, g_idx={g_idx}")
        if kwargs.get("verbose", False): print(f"Successors: {successors}")

        # If branch is terminal, create successor where branch is just removed from OPEN
        if kwargs.get("verbose", False): print(f"Branch {branch_idx} is TERMINAL")
        if branch_idx in self._TERMINAL:
            if kwargs.get("verbose", False): print(f"Branch {branch_idx} is TERMINAL")
            successors.append(self._make_successor(branch_idx, None, viewer))

        # Restore rules following bottom-up branch traversal
        for removed_rules in reversed(list_removed_rules):
            for r_idx in removed_rules: viewer.restore_rule(r_idx)

        assert cost_bound is None or next_cost_bound > cost_bound, (next_cost_bound, cost_bound)
        random.shuffle(successors)
        successors: List["Node"] = sorted(successors, reverse=True)
        random_tie_breaking_for_successors: bool = len([successor for successor in successors if successor._score == successors[0]._score]) > 1
        return successors, next_cost_bound, random_tie_breaking_for_branch | random_tie_breaking_for_successors

    def get_r_idx_to_info(self, viewer: RuleViewer) -> Dict[int, Tuple[int, intbitset]]:
        r_idxs: intbitset = viewer.r_idxs()
        r_idx_to_info: Dict[int, Tuple[int, intbitset]] = dict()

        num_standard_rules: int = 0
        num_other_rules: int = 0
        for branch_idx, branch in enumerate(self._branch_idx_to_branch):
            if branch._f_idx != -1:
                above: intbitset = intbitset([self.get_branch(self.get_branch_idx(branch._id[:i]))._f_idx for i in range(len(branch._id))])
                assert -1 not in above
                assert branch._r_idxs_change is not None and branch._r_idxs_other_change is not None
                for r_idx in branch._r_idxs_change | branch._r_idxs_other_change:
                    assert r_idx not in r_idx_to_info
                    r_idx_to_info[r_idx] = (branch._f_idx, intbitset(above))
                num_standard_rules += len(branch._r_idxs_change)
                num_other_rules += len(branch._r_idxs_other_change)
        assert len(self.get_branch(0)._r_idxs_other) == num_other_rules, (len(self.get_branch(0)._r_idxs_other), num_other_rules)

        return r_idx_to_info

    # Branch selection heuristic: select a tip with greatest number of r_idxs
    def bsh_tip_with_most_r_idxs(self) -> Tuple[int, bool]:
        assert len(self._OPEN) > 0
        branches: List[Branch] = [self.get_branch(branch_idx) for branch_idx in self._OPEN]
        branches_with_score: List[Tuple[int, Tuple[int]]] = [(branch._idx, (len(branch._r_idxs), len(branch._r_idxs_other))) for branch in branches]
        if len(branches_with_score) > 0:
            random.shuffle(branches_with_score)
            sorted_branches_with_score: List[Tuple[int, Tuple[int]]] = sorted(branches_with_score, key=lambda p: p[1], reverse=True)
            random_tie_breaking_for_branch: bool = len([branch for branch, score in sorted_branches_with_score if score == sorted_branches_with_score[0][1]]) > 1
            return sorted_branches_with_score[0][0], random_tie_breaking_for_branch
        else:
            return None, False

    # Branch selection heuristic: select a shallowest tip
    def bsh_shallowest_tip(self) -> Tuple[int, bool]:
        assert len(self._OPEN) > 0
        branches: List[Branch] = [self.get_branch(branch_idx) for branch_idx in self._OPEN]
        branches_with_score: List[Tuple[int, int]] = [(branch._idx, len(branch._id)) for branch in branches]
        if len(branches_with_score) > 0:
            random.shuffle(branches_with_score)
            sorted_branches_with_score: List[Tuple[int, int]] = sorted(branches_with_score, key=lambda p: p[1], reverse=False)
            random_tie_breaking_for_branch: bool = len([branch for branch, score in sorted_branches_with_score if score == sorted_branches_with_score[0][1]]) > 1
            return sorted_branches_with_score[0][0], random_tie_breaking_for_branch
        else:
            return None, False

    # Branch selection heuristic: select a leftmost tip
    def bsh_leftmost_tip(self) -> Tuple[int, bool]:
        assert len(self._OPEN) > 0
        branches: List[Branch] = [self.get_branch(branch_idx) for branch_idx in self._OPEN]
        branches_with_score: List[Tuple[int, str]] = [(branch._idx, branch._id) for branch in branches]
        if len(branches_with_score) > 0:
            random.shuffle(branches_with_score)
            sorted_branches_with_score: List[Tuple[int, int]] = sorted(branches_with_score, key=lambda p: p[1])
            random_tie_breaking_for_branch: bool = len([branch for branch, score in sorted_branches_with_score if score == sorted_branches_with_score[0][1]]) > 1
            return sorted_branches_with_score[0][0], random_tie_breaking_for_branch
        else:
            return None, False


class SolutionGenerator:
    # Class variables for statistics
    _layers: List[List[int]] = []
    _num_generated: List[int] = []
    _num_expanded: List[int] = []
    _num_terminals: List[int] = []
    _num_solutions: List[int] = []

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

    def __init__(self, preprocessing_data: Dict[str, Any], state_factory: StateFactory, finalizer: PolicyFinalizer, **kwargs):
        self._preprocessing_data: Dict[str, Any] = preprocessing_data
        self._state_factory: StateFactory = state_factory
        self._finalizer: PolicyFinalizer = finalizer
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
        self._max_features: int = kwargs.get("max_features", int(1e6))
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

    def _node_generator(self, branch_selection_heuristic: Callable[Node, int], r_idxs_other: intbitset, **kwargs) -> Generator[Node, None, None]:
        def rec_generator(depth: int, node: Node, cost_bound: int, prune_solutions_with_cost_bound: bool = True) -> Generator[Node, None, int]:
            assert node._cost <= cost_bound
            SolutionGenerator._num_generated[-1] += 1
            next_cost_bound: int = int(1e6)
            if node.is_terminal():
                SolutionGenerator._num_terminals[-1] += 1
                if not prune_solutions_with_cost_bound or node._cost == cost_bound:
                    SolutionGenerator._num_solutions[-1] += 1
                    node.dump("YIELD ")
                    yield node
            else:
                SolutionGenerator._num_expanded[-1] += 1
                successors, next_cost_bound, _ = node.get_successors(branch_selection_heuristic, self._viewer, cost_bound, **kwargs)
                for succ in successors:
                    ncb: int = yield from rec_generator(1 + depth, succ, cost_bound, prune_solutions_with_cost_bound)
                    next_cost_bound = min(next_cost_bound, ncb)
            return next_cost_bound

        max_cost_bound: int = kwargs.get("cost_bound", 100)
        Node.initialize(self._width, self._max_features, self._uniform_costs, self._k_reachable, self._ext_state_to_ext_edge, self._ext_state_to_feature_valuations)

        SolutionGenerator._layers.append([])
        SolutionGenerator._num_generated.append(0)
        SolutionGenerator._num_expanded.append(0)
        SolutionGenerator._num_terminals.append(0)
        SolutionGenerator._num_solutions.append(0)

        last_num_generated: int = 0
        last_num_expanded: int = 0
        last_num_terminals: int = 0
        last_num_solutions: int = 0

        root_node: Node = Node.make_root_node(self._viewer.all_rules() - r_idxs_other, self._ex_paths, r_idxs_other)
        assert root_node.verify()

        cost_bound: int = 0
        while cost_bound <= max_cost_bound:
            self._layers[-1].append(cost_bound)
            next_cost_bound: int = yield from rec_generator(0, root_node, cost_bound)
            logging.info(f"Cost-layer: cost={cost_bound}, #generated={SolutionGenerator._num_generated[-1] - last_num_generated}, #expanded={SolutionGenerator._num_expanded[-1] - last_num_expanded}, #terminals={SolutionGenerator._num_terminals[-1] - last_num_terminals}, #solutions={SolutionGenerator._num_solutions[-1] - last_num_solutions}")
            last_num_generated: int = SolutionGenerator._num_generated[-1]
            last_num_expanded: int = SolutionGenerator._num_expanded[-1]
            last_num_terminals: int = SolutionGenerator._num_terminals[-1]
            last_num_solutions: int = SolutionGenerator._num_solutions[-1]
            cost_bound: int = next_cost_bound

    def _find_solution(self, branch_selection_heuristic: Callable[Node, int], r_idxs_other: intbitset, cost_bound: int, **kwargs) -> Dict[str, Any]:
        def search(depth: int, node: Node, cost_bound: int) -> Dict[str, Any]:
            nonlocal meaningful_restart
            #node.dump(depth * '  ' + '==> ')
            assert cost_bound is None or node._cost <= cost_bound
            SolutionGenerator._num_generated[-1] += 1
            if node.is_terminal():
                SolutionGenerator._num_terminals[-1] += 1
                r_idx_to_info: Dict[int, Tuple[int, intbitset]] = node.get_r_idx_to_info(self._viewer)
                result: Dict[str, Any] = self._finalizer(node._f_idxs, r_idx_to_info)
                if result is not None:
                    result.update({"node": node, "r_idx_to_info": r_idx_to_info, "r_idxs": list(r_idx_to_info.keys())})
                    SolutionGenerator._num_solutions[-1] += 1
                    return result
                else:
                    bt: Backtrack = Backtrack(Backtrack._backtrack_depth)
                    print(f"{depth * '  '}<== BACKTRACK (TERMINAL) @ {depth} ({bt})")
                    raise bt
            else:
                SolutionGenerator._num_expanded[-1] += 1
                successors, _, random_tie_breaking = node.get_successors(branch_selection_heuristic, self._viewer, cost_bound, **kwargs)
                meaningful_restart |= random_tie_breaking
                for succ in successors:
                    try:
                        solution: Dict[str, Any] = search(1 + depth, succ, cost_bound)
                    except Backtrack as bt:
                        if bt._depth < 0:
                            # This is a restart request, float it
                            #print(f"{depth * '  '}<== RESTART @ {depth} ({bt}, #f_idxs={len(node._f_idxs)}, max_features={Node._max_features})")
                            raise bt
                        elif bt._depth == 0:
                            # Backtrack level reached, if still within quota of bracktracks, continue, else request restart
                            #print(f"{depth * '  '}HANDLING {bt} @ {depth}")
                            Backtrack._num_backtracks += 1
                            if Backtrack._max_backtracks and Backtrack._max_backtracks <= Backtrack._num_backtracks:
                                #logging.warning(f"{depth * '  '}Max backtracks reached: #backtracks={Backtrack._num_backtracks}, #f_idxs={len(node._f_idxs)},  max_features={Node._max_features})")
                                #print(f"{depth * '  '}Max backtracks reached @ {depth} (#backtracks={Backtrack._num_backtracks})")
                                restart: Backtrack = Backtrack(-1)
                                #print(f"{depth * '  '}REQUEST restart {restart}")
                                raise restart
                            continue
                        else:
                            # Regular backtrack hasn't reached handling level, float it
                            bt._depth -= 1
                            #print(f"{depth * '  '}<== BACKTRACK (id={bt._id}, depth={bt._depth}, #f_idxs={len(node._f_idxs)}, max_features={Node._max_features})")
                            raise bt
                    assert solution is not None
                    return solution

                # No solution found below this node, backtrack
                bt: Backtrack = Backtrack(Backtrack._backtrack_depth)
                #print(f"{depth * '  '}<== BACKTRACK @ {depth} ({bt}, #succ={len(successors)}, #f_idxs={len(node._f_idxs)}, max_features={Node._max_features})")
                raise bt

        Backtrack.initialize(kwargs.get("max_restarts", 1), kwargs.get("max_backtracks"), kwargs.get("backtrack_depth", 0))
        Node.initialize(self._width, self._max_features, self._uniform_costs, self._k_reachable, self._ext_state_to_ext_edge, self._ext_state_to_feature_valuations)
        meaningful_restart: bool = True

        SolutionGenerator._layers.append([])
        SolutionGenerator._num_generated.append(0)
        SolutionGenerator._num_expanded.append(0)
        SolutionGenerator._num_terminals.append(0)
        SolutionGenerator._num_solutions.append(0)

        root_node: Node = Node.make_root_node(self._viewer.all_rules() - r_idxs_other, self._ex_paths, r_idxs_other)
        assert root_node.verify()

        # Perform up to max_restarts searches
        assert Backtrack._max_restarts > 0
        while meaningful_restart and Backtrack._num_restarts <= Backtrack._max_restarts:
            meaningful_restart: bool = False
            Backtrack.reset()
            logging.info(f"Searching for solution with cost_bound={cost_bound} [restart_idx={Backtrack._num_restarts}] ...")
            try:
                solution: Dict[str, Any] = search(0, root_node, cost_bound)
            except Backtrack as bt:
                Backtrack._num_restarts += 1
                print(f"TOP-LEVEL: bt={bt}, meaningful_restart={meaningful_restart}")
                continue
            assert solution is not None
            solution.get("node").dump("FOUND ")
            return solution
        logging.warning(f"NO SOLUTION FOUND AFTER {Backtrack._max_restarts} RESTART(s)")
        raise MaxRestarts(Backtrack._max_restarts, cost_bound)

    def solutions(self, other_ext_states: List[Tuple[int, int]], **kwargs) -> Generator[Dict[str, Any], None, None]:
        # Reset rule viewer
        self._viewer.reset()
        assert False, "ADAPT TO FINALIZER MOVED INTO GENERATOR"

        # Other r_idxs
        for ext_state in other_ext_states:
            if self._viewer.get_r_idx(ext_state) is None:
                print(f"NON_EXISTENT: {ext_state}")
        assert all([self._viewer.get_r_idx(ext_state) is not None for ext_state in other_ext_states])
        r_idxs_other: intbitset = intbitset([self._viewer.get_r_idx(ext_state) for ext_state in other_ext_states])

        # Generate solutions in increasing order of cost
        max_solutions: int = kwargs.get("max_solutions")
        branch_selection_heuristic: Callable[Node, Tuple[int, bool]] = lambda node: node.bsh_tip_with_most_r_idxs()
        for i, node in enumerate(self._node_generator(branch_selection_heuristic, r_idxs_other, **kwargs)):
            logging.info(f"Yield {1 + i} {node}")
            yield node._f_idxs, node.get_r_idx_to_info(self._viewer)
            if max_solutions and max_solutions <= 1 + i: break
        logging.info(f"No more solutions: {i} solution(s) using max_solutions={max_solutions}; #expanded={self._num_expanded}, #generated={self._num_generated}")

    def one_solution(self, other_ext_states: List[Tuple[int, int]], cost_bound: int = None, **kwargs) -> Dict[str, Any]:
        # Reset rule viewer
        self._viewer.reset()

        # Check there is solution
        ext_states_that_change_no_feature: List[Tuple[int, int]] = self._viewer.ext_states_that_change_no_feature()
        if len(ext_states_that_change_no_feature) > 0:
            logging.warning(f"The following ext_states (transitions) change no feature: {sorted(ext_states_that_change_no_feature)}")
            raise NoFeature([self._ext_state_to_ext_edge.get(ext_state) for ext_state in ext_edges_that_change_no_feature])

        # Other r_idxs
        for ext_state in other_ext_states:
            if self._viewer.get_r_idx(ext_state) is None:
                print(f"NON_EXISTENT: {ext_state}")
        assert all([self._viewer.get_r_idx(ext_state) is not None for ext_state in other_ext_states])
        r_idxs_other: intbitset = intbitset([self._viewer.get_r_idx(ext_state) for ext_state in other_ext_states])

        # Search for one solution
        if kwargs.get("branch_selection_heuristic", "leftmost") == "leftmost":
            branch_selection_heuristic: Callable[Node, Tuple[int, bool]] = lambda node: node.bsh_leftmost_tip()
        else:
            branch_selection_heuristic: Callable[Node, Tuple[int, bool]] = lambda node: node.bsh_shallowest_tip()
        solution: Dict[str, Any] = self._find_solution(branch_selection_heuristic, r_idxs_other, cost_bound, **kwargs)
        return solution

