import logging
import random
import heapq
import numpy as np

# Bitset-based unordered sets of unsigned integers
from intbitset import intbitset

from termcolor import colored
from typing import Set, Tuple, List, Union, Dict, Any, Optional, Union
from collections import OrderedDict, defaultdict, deque
from itertools import product

import dlplan.core as dlplan_core

from ..feature_pool import Feature
from ...util import Timer
from ...state_space import StateFactory

from .m_pairs import MPairs
from .stratified_policy import StratifiedPolicy


class GreedySolver:
    def __init__(self, preprocessing_data: Dict[str, Any], state_factory: StateFactory, **kwargs):
        self._preprocessing_data: Dict[str, Any] = preprocessing_data
        self._state_factory: StateFactory = state_factory
        self._simplify_policy: bool = kwargs.get("simplify_policy", False)
        self._simplify_only_conditions: bool = kwargs.get("simplify_only_conditions", False)
        self._uniform_costs: bool = kwargs.get("uniform_costs", False)
        self._monotone_only_by_dec: bool = kwargs.get("monotone_only_by_dec", False)
        assert self._preprocessing_data is not None

        # Extract relevant data from pre-processing
        self._relevant_features: List[Tuple[int, Feature]] = self._preprocessing_data.get("relevant_features")
        self._f_idx_to_feature_index: Dict[int, int] = self._preprocessing_data.get("f_idx_to_feature_index")
        assert self._relevant_features is not None

        self._requirements_for_good_transitions: Dict[Tuple[int, int], intbitset] = self._preprocessing_data.get("requirements_for_good_transitions")
        self._goal_ext_pair_to_separating_features: Dict[Tuple[int, Tuple[int, int]], intbitset] = self._preprocessing_data.get("goal_ext_pair_to_separating_features")
        #self._deadend_path_to_separating_features: Dict[Tuple[Tuple[int, int]], List[intbitset]] = self._preprocessing_data.get("deadend_path_to_separating_features")
        self._ext_state_to_separating_features_for_deadend_paths: Dict[Tuple[int, int], Dict[Tuple[int, Tuple[int, int]], intbitset]] = self._preprocessing_data.get("ext_state_to_separating_features_for_deadend_paths")
        self._ext_sibling_to_separating_features: Dict[Tuple[int, int, Tuple[int, int]], intbitset] = self._preprocessing_data.get("ext_sibling_to_separating_features")
        self._ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]] = self._preprocessing_data.get("ext_state_to_ext_edge")
        self._bad_ext_edges: Set[Tuple[int, Tuple[int, int]]] = self._preprocessing_data.get("bad_ext_edges")

        self._m_pairs: MPairs = self._preprocessing_data.get("m_pairs")
        self._monotone_features: intbitset = self._m_pairs.monotone_features()

        # Calculate numerical features
        self._numerical_features: List[Tuple[int, Feature]] = [(f_idx, feature) for f_idx, feature in self._relevant_features if isinstance(feature.dlplan_feature, dlplan_core.Numerical)]

        # Construct requirements, one per ex-edge and one per pair of goal and non-goal xstates
        self._annotated_requirements: List[Tuple[Dict[str, Any], intbitset]] = []
        self._annotated_requirements.extend([({"key": "Edge", "ext_state": ext_state}, requirement) for ext_state, requirement in self._requirements_for_good_transitions.items()])
        self._annotated_requirements.extend([({"key": "Goal", "pair": pair}, separating_features) for pair, separating_features in self._goal_ext_pair_to_separating_features.items()])
        self._annotated_requirements.extend([({"key": "Deadend", "ext_state": ext_state, "path": path}, separating_features) for ext_state, separating_features_for_deadend_paths in self._ext_state_to_separating_features_for_deadend_paths.items() for path, separating_features in separating_features_for_deadend_paths.items()])
        self._annotated_requirements.extend([({"key": "Sibling", "ext_sibling": ext_sibling}, separating_features) for ext_sibling, separating_features in self._ext_sibling_to_separating_features.items()])
        self._requirements: List[intbitset] = [requirement for _, requirement in self._annotated_requirements]
        self._requirements_for_deadends: List[intbitset] = [requirement for annotation, requirement in self._annotated_requirements if annotation.get("key") == "Deadend"]
        self._num_requirements: Dict[str, int] = {key: sum([1 if annotation.get("key") == key else 0 for annotation, _ in self._annotated_requirements]) for key in ["Edge", "Goal", "Deadend", "Sibling"]}
        logging.info(f"{len(self._requirements)} requirement(s) split as {self._num_requirements}")

        # Support for simplification of policies
        self._ex_ext_states: Set[Tuple[int, int]] = self._preprocessing_data.get("ex_ext_states")
        self._ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = self._preprocessing_data.get("ext_state_to_feature_valuations")

    def _calculate_ranks_from_chains(self, features: intbitset, feature_chains: List[Tuple[int]]) -> Dict[int, int]:
        # Calculate rank of selected features by solving a simple system of inequalities. For each feature f in solution,
        # create vertex -f. For each link g < f in a chain, create an edge from -g to -f with cost -1. Finally, create an
        # additional vertex S, with id=1,  that is connected to each vertex -f with an arc of zero cost. The system has
        # solution iff the graph has no cycle of negative cost, which is guaranteed by construction of the terminating set.
        # The rank of feature f is the cost of a min-cost path from vertex S to vertex -f.

        successors: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
        successors[1] = {(-f_idx, 0) for f_idx in features}
        for f_idx in features:
            f_idx_index = self._f_idx_to_feature_index[f_idx]
            f_idx_chain = feature_chains[f_idx_index]
            for (u, v) in zip(f_idx_chain[:-1], f_idx_chain[1:]):
                successors[-u].add((-v, -1))

        # Heap elements are pairs (dist, u) where dist is distance from S to node u.
        # (This relies on the fact that items are sorted with < so first element must be priority.)
        heap: List[Tuple[int, int]] = []
        heapq.heappush(heap, (0, 1))
        distances: Dict[int, int] = dict()

        while len(heap) > 0:
            dist, u = heapq.heappop(heap)
            if dist < distances.get(u, int(1e6)):
                distances[u] = dist
                for v, cost in successors.get(u, []):
                    heapq.heappush(heap, (dist + cost, v))

        feature_ranks: Dist[int, int] = {-u: -d for u, d in distances.items() if u < 0}
        return feature_ranks

    def _calculate_ranks(self, features: intbitset) -> Dict[int, int]:
        feature_ranks: Dict[int, int] = {f_idx: 0 for f_idx in features if f_idx in self._monotone_features}
        pending_features: intbitset = features - intbitset(feature_ranks.keys())
        change: bool = True

        while change and len(pending_features) > 0:
            g_idxs_with_ranks: List[Tuple[int, int]] = list(feature_ranks.items())
            for g_idx, rank in g_idxs_with_ranks:
                f_idxs: intbitset = (features & self._m_pairs.f_idxs_for_g_idx(g_idx)) - intbitset(feature_ranks.keys())
                feature_ranks.update({f_idx: 1 + rank for f_idx in f_idxs})
                pending_features -= f_idxs
                change |= len(f_idxs) > 0

        assert len(pending_features) == 0
        return feature_ranks

    def _does_it_solve_requirements(self, features: intbitset, ranks: Dict[int, int], check_support: bool = True, verbose: bool = False) -> bool:
        assert features is not None
        pending_requirements: intbitset = intbitset(range(len(self._requirements)))
        for f_idx in features:
            if check_support:
                # Check that feature is enabled: an enabler g_idx must be in features and have rank lesser than rank of r_idx
                f_idx_rank = ranks.get(f_idx)
                if f_idx_rank > 0 and len(intbitset([g_idx for g_idx in self._m_pairs.g_idxs_for_f_idx(f_idx) if ranks.get(g_idx, int(1e6)) < f_idx_rank]) & features) == 0:
                    if verbose: logging.info(f"Feature f_idx={f_idx} is not enabled; enablers={sorted(self._fg_companions.get(f_idx, []))}, features={features}")
                    return False
            pending_requirements -= intbitset([i for i in pending_requirements if f_idx in self._requirements[i]])

        if len(pending_requirements) > 0:
            if verbose: logging.info(f"{len(pending_requirements)} unsatisfied requirement(s)")
            return False
        else:
            return True

    def _revise_costs_and_chains(self, f_idxs: intbitset, feature_costs: List[int], feature_chains: List[Tuple[int]]):
        q: deque = deque(f_idxs)
        while len(q) > 0:
            g_idx: int = q.popleft()
            g_idx_index = self._f_idx_to_feature_index[g_idx]
            g_idx_complexity = self._relevant_features[g_idx_index][1].complexity
            g_idx_cost = feature_costs[g_idx_index]
            g_idx_chain = feature_chains[g_idx_index]
            assert g_idx_cost < int(1e6) and g_idx_chain is not None
            logging.debug(f"Dequeued: g_idx={g_idx}{'*' if g_idx in self._monotone_features else ''}, complexity={g_idx_complexity}, cost={g_idx_cost}")

            for f_idx in self._m_pairs.f_idxs_for_g_idx(g_idx):
                f_idx_index = self._f_idx_to_feature_index[f_idx]
                f_idx_complexity = self._relevant_features[f_idx_index][1].complexity
                f_idx_cost = feature_costs[f_idx_index]
                if g_idx_cost + f_idx_complexity < f_idx_cost:
                    new_f_idx_cost = g_idx_cost + f_idx_complexity
                    new_f_idx_chain = g_idx_chain + (f_idx,)
                    logging.debug(f"  Revise cost of f_idx={f_idx}{'*' if f_idx in self._monotone_features else ''} from {f_idx_cost} to {new_f_idx_cost} ; chain={new_f_idx_chain}")
                    feature_costs[f_idx_index] = new_f_idx_cost
                    feature_chains[f_idx_index] = new_f_idx_chain
                    q.append(f_idx)

    def _score_fn(self, f_idx: int, pending_requirements: intbitset, feature_costs: List[int], feature_chains: List[Tuple[int]]) -> int:
        feature_index: int = self._f_idx_to_feature_index[f_idx]
        feature_complexity: int = self._relevant_features[feature_index][1].complexity
        feature_cost = feature_costs[feature_index]
        feature_chain: intbitset = intbitset(feature_chains[feature_index])
        return sum([1 if len(feature_chain & self._requirements[i]) > 0 else 0 for i in pending_requirements]) / feature_cost

    def _simple_score_fn(self, f_idx: int, pending_requirements: intbitset) -> float:
        feature_index: int = self._f_idx_to_feature_index[f_idx]
        feature_complexity: int = self._relevant_features[feature_index][1].complexity
        return sum([1 for i in pending_requirements if f_idx in self._requirements[i]]) / feature_complexity

    def _simplify_terminating_set(self, terminating_set: intbitset, feature_ranks: Dict[int, int]) -> intbitset:
        removed: List[int] = []
        new_terminating_set: intbitset = intbitset(terminating_set)
        expendables: List[int] = [f_idx for f_idx in terminating_set if self._does_it_solve_requirements(terminating_set - intbitset([f_idx]), feature_ranks, verbose=False)]

        while len(expendables) > 0:
            expendables_with_scores: List[Tuple[int, int]] = [(f_idx, self._relevant_features[self._f_idx_to_feature_index[f_idx]][1].complexity) for f_idx in expendables]
            sorted_expendables_with_scores: List[Tuple[int, int]] = sorted(expendables_with_scores, key=lambda item: item[1], reverse=True)
            chosen: int = sorted_expendables_with_scores[0][0]
            logging.info(f"  Postprocessing: expendables={expendables}, chosen={chosen}")
            removed.append(chosen)
            new_terminating_set -= intbitset([chosen])
            expendables: List[int] = [f_idx for f_idx in new_terminating_set if self._does_it_solve_requirements(new_terminating_set - intbitset([f_idx]), feature_ranks, verbose=False)]

        if len(removed) > 0:
            cost = sum([self._relevant_features[self._f_idx_to_feature_index[f_idx]][1].complexity for f_idx in new_terminating_set])
            for f_idx in removed: feature_ranks.pop(f_idx)
            logging.info(f"  Postprocessing: {len(removed)} feature(s) removed: {removed}")
            logging.info(f"  Postprocessing: terminating_set: {sorted(new_terminating_set)}")
            logging.info(f"  Postprocessing: cost: {cost}")
            logging.info(f"  Postprocessing: ranks: {sorted([(f_idx, rank) for f_idx, rank in feature_ranks.items()], key=lambda item: item[1])}")

        return new_terminating_set

    def _calculate_decorations(self, terminating_feature_set: intbitset, feature_ranks: Dict[int, int]) -> Dict[str, Dict[int, Dict[int, intbitset]]]:
        # Calculate decorations as follows:
        #
        # 1. Each rule r is associated with a unique ext_state, denoted by ext_state[r]
        # 2. For each rule r, let f_idx[r] be a f_idx of *minimum* rank that changes at ext_state[r]
        # 3. For each rule r, let g_idx[r] be a g_idx that enables f_idx[r] of *minimum* rank. Clearly, rank[g_idx[r]] < rank[f_idx[r]]
        #
        # Unknowns:
        # 4. If rank[f_idx] > rank[f_idx[r]], set f_idx as unknown at rule r *if* f_idx doesn't satisfy a DEADEND requirement
        #
        # Don't cares:
        # 5. If f_idx[r] is   boolean, condition[r] = { bvalue[g_idx[r], ext_state[r]], bvalue[f_idx[r], ext_state[r]] }
        # 6. If f_idx[r] is numerical, condition[r] = { bvalue[g_idx[r], ext_state[r]] }
        # Additionally, in both cases, add bvalue[f_idx] to condition if f_idx is boolean that separates r from a deadend transition

        logging.info(f"Calculate decorations: features={list(terminating_feature_set)}, ranks={feature_ranks}")
        unknowns: Dict[int, Dict[int, intbitset]] = defaultdict(lambda: defaultdict(intbitset))
        dont_cares: Dict[int, Dict[int, intbitset]] = defaultdict(lambda: defaultdict(intbitset))
        numerical_f_idxs: intbitset = intbitset([f_idx for f_idx, _ in self._numerical_features])

        # Calculate f_idxs that separate ext_states from deadend paths as follows:
        # 1. Solve min-cost hitting set problem to choose f_idxs that solve all requirements for daadend-paths (called f_idxs_for_deadend_requirements)
        # 2. Associate each ext_state (i.e. rule r) with chosen features that hit requirements for ext_state
        # 3. If f_idx is associated with rule r and f_idx is boolean, then f_idx cannot be marked as don't care for rule r

        # Step 1: Solve min-cost hitting set problem
        if True:
            f_idxs_for_deadend_requirements: intbitset = intbitset()
            pending_deadend_requirements: intbitset = intbitset(range(len(self._requirements_for_deadends)))
            while len(pending_deadend_requirements) > 0:
                f_idxs_with_scores: List[Tuple[int, int]] = [(f_idx, sum([1 if f_idx in self._requirements_for_deadends[i] else 0 for i in pending_deadend_requirements])) for f_idx in terminating_feature_set - f_idxs_for_deadend_requirements]
                sorted_f_idxs_with_scores: List[Tuple[int, int]] = sorted(f_idxs_with_scores, key=lambda pair: pair[1], reverse=True)
                assert len(sorted_f_idxs_with_scores) > 0
                best_f_idx: int = sorted_f_idxs_with_scores[0][0]
                f_idxs_for_deadend_requirements.add(best_f_idx)
                pending_deadend_requirements: intbitset = pending_deadend_requirements - intbitset([i for i in pending_deadend_requirements if best_f_idx in self._requirements_for_deadends[i]])
        else:
            f_idxs_for_deadend_requirements: intbitset = intbitset(terminating_feature_set)
        assert all([len(requirement & f_idxs_for_deadend_requirements) > 0 for requirement in self._requirements_for_deadends])

        # Step 2: Associate each ext_state (i.e. rule r) with chosen features that hit requirements for ext_state
        marked_f_idxs: Dict[Tuple[int, int], intbitset] = dict()
        for ext_state, dict_for_separating_features  in self._ext_state_to_separating_features_for_deadend_paths.items():
            f_idxs: intbitset = intbitset().union(*dict_for_separating_features.values()) & f_idxs_for_deadend_requirements
            marked_f_idxs[ext_state] = f_idxs
            # CHECK: Can we reduced f_idx even further?

        # Iterate over example ext states
        for ex_ext_state in self._ex_ext_states:
            instance_idx, state_idx = ex_ext_state
            requirement: intbitset = self._requirements_for_good_transitions.get(ex_ext_state)
            assert requirement is not None, ex_ext_state
            assert len(requirement) > 0

            # Define f_idx for ex_ext_state[r]
            hitting_set: intbitset = requirement & terminating_feature_set
            assert len(hitting_set) > 0
            sorted_hitting_set: List[Tuple[int, int]] = sorted([(f_idx, feature_ranks.get(f_idx)) for f_idx in hitting_set], key=lambda item: item[1])
            f_idx_for_ext_state: int = sorted_hitting_set[0][0]

            # Define g_idx for ex_ext_state[r]
            non_dont_care_conditions: intbitset = intbitset()
            if f_idx_for_ext_state not in self._monotone_features:
                g_idxs: intbitset = self._m_pairs.g_idxs_for_f_idx(f_idx_for_ext_state) & terminating_feature_set
                assert len(g_idxs) > 0
                sorted_g_idxs: List[Tuple[int, int]] = sorted([(g_idx, feature_ranks.get(g_idx)) for g_idx in g_idxs], key=lambda item: item[1])
                g_idx_for_ext_state: int = sorted_g_idxs[0][0]
                non_dont_care_conditions.add(g_idx_for_ext_state)
            if f_idx_for_ext_state not in numerical_f_idxs:
                non_dont_care_conditions.add(f_idx_for_ext_state)

            # Step 3: Add conditions for deadend requirements; i.e., add boolean marked f_idxs for ext_state
            marked_f_idxs_for_ext_state = marked_f_idxs.get(ex_ext_state, intbitset())
            non_dont_care_conditions |= marked_f_idxs_for_ext_state - numerical_f_idxs

            unknown_effects: List[int] = [f_idx for f_idx in terminating_feature_set if feature_ranks.get(f_idx) > feature_ranks.get(f_idx_for_ext_state) and f_idx not in marked_f_idxs_for_ext_state]
            unknowns[instance_idx][state_idx] = intbitset(unknown_effects)
            dont_cares[instance_idx][state_idx] = terminating_feature_set - non_dont_care_conditions

        unknowns: Dict[int, Dict[int, intbitset]] = {instance_idx: dict(dict_for_unknowns) for instance_idx, dict_for_unknowns in unknowns.items()}
        dont_cares: Dict[int, Dict[int, intbitset]] = {instance_idx: dict(dict_for_dont_cares) for instance_idx, dict_for_dont_cares in dont_cares.items()}
        decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = {"unknown": unknowns, "dont_care": dont_cares}
        return decorations

    def _calculate_decorations2(self, features: intbitset, ranks: Dict[int, int], sigma: Set[Tuple[int, int]]) -> Dict[str, Dict[int, Dict[int, intbitset]]]:
        numerical_features: intbitset = intbitset([f_idx for f_idx, _ in self._numerical_features])
        policy: StratitiedPolicy = StratifiedPolicy(features, numerical_features, sigma, self._ext_state_to_ext_edge, self._ext_state_to_feature_valuations, self._bad_ext_edges)
        decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = policy.calculate_decorations(self._simplify_only_conditions)
        return decorations

    def _rec_descend(self, depth: int, depth_bound: int, incumbent_terminating_set: intbitset, incumbent_cost: int, pending_requirements: intbitset) -> Tuple[intbitset, int]:
        logging.info(f"dfs: {'  ' * depth}incumbent_terminating_set={sorted(incumbent_terminating_set)}, cost={incumbent_cost}, {len(pending_requirements)} pending requirement(s)")

        if len(pending_requirements) == 0:
            return incumbent_terminating_set, incumbent_cost
        elif depth_bound is not None and depth >= depth_bound:
            return None, incumbent_cost

        eligible_features: List[int] = list(self._m_pairs.f_idxs_for_g_idxs(incumbent_terminating_set))
        eligible_features_with_scores: List[Tuple[int, int]] = [(f_idx, self._simple_score_fn(f_idx, pending_requirements)) for f_idx in eligible_features]
        eligible_features_with_non_zero_scores: List[Tuple[int, int]] = [(f_idx, score) for f_idx, score in eligible_features_with_scores if score > 0]
        sorted_eligible_features_with_scores: List[Tuple[int, int]] = sorted(eligible_features_with_non_zero_scores, key=lambda item: item[1], reverse=True)

        for f_idx, score in sorted_eligible_features_with_scores:
            feature_index: int = self._f_idx_to_feature_index[f_idx]
            feature_cost: int = self._relevant_features[feature_index][1].complexity
            incumbent_terminating_set.add(f_idx)
            f_idx_pending_requirements: intbitset = pending_requirements - intbitset([i for i in pending_requirements if f_idx in self._requirements[i]])
            terminating_set, cost = self._rec_descend(1 + depth, depth_bound, incumbent_terminating_set, incumbent_cost + feature_cost, f_idx_pending_requirements)
            if terminating_set is not None: return terminating_set, cost
            incumbent_terminating_set.remove(f_idx)

        # No solution below this branch
        return None, incumbent_cost

    # Solver that recursively descend search tree looking for a terminating set that solves all requirements
    # It can be used with "lazy" m_pairs
    def dfs_solve(self, **kwargs) -> Tuple[bool, intbitset, int, Dict[str, Dict[int, Dict[int, intbitset]]], int]:
        local_timer: Timer = Timer()

        # Other arguments
        optimality: bool = kwargs.get("optimality", False)
        depth_bound: int = kwargs.get("depth_bound")

        # Obtain solution by exploring search tree with recursive descend
        terminating_set, cost = self.rec_descend(0, depth_bound, intbitset(), 0, intbitset(range(len(self._requirements))))
        assert terminating_set is not None

        # Calculate rank of selected features
        feature_ranks: Dict[int, int] = self._calculate_ranks(terminating_set)
        logging.info(f"Ranks: {sorted([(f_idx, rank) for f_idx, rank in feature_ranks.items()], key=lambda item: item[1])}")
        assert self._does_it_solve_requirements(terminating_set, feature_ranks, verbose=True)

        # Simplify set of selected features
        terminating_set: intbitset = self._simplify_terminating_set(terminating_set, feature_ranks)

        # Calculate decorations
        if self._simplify_policy:
            assert self._num_requirements["Deadend"] == 0, "RULE SIMPLIFICATION NOT YET READY FOR DEADEND"
            decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = self._calculate_decorations(terminating_set, feature_ranks)
            logging.info(f"Decorations: {decorations}")
        else:
            decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = dict()

        local_timer.stop()
        logging.info(f"Greedy solver finished in {local_timer.get_elapsed_sec():0.2f} second(s)")
        return True, terminating_set, [cost], decorations, feature_ranks

    # Solver that at each iteration solves a requiements. Number of iteration is thus bounded by number of requirements.
    def solve(self, **kwargs) -> Tuple[bool, intbitset, int, Dict[str, Dict[int, Dict[int, intbitset]]], int]:
        logging.info(f"Starting greedy solver...")
        local_timer: Timer = Timer()

        # Define costs and chains for features, and revision function
        feature_costs: List[int] = [int(1e6) for _ in self._relevant_features]
        feature_chains: List[Tuple[int]] = [None for _ in self._relevant_features]

        # Initialize costs and chains for monotone features, and propagate
        logging.info(f"Constructing costs and chains for {len(self._relevant_features)} feature(s) using dynamic programming...")
        for f_idx in self._monotone_features:
            f_idx_index = self._f_idx_to_feature_index[f_idx]
            f_idx_complexity = self._relevant_features[f_idx_index][1].complexity
            feature_costs[f_idx_index] = f_idx_complexity
            feature_chains[f_idx_index] = (f_idx,)
        self._revise_costs_and_chains(self._monotone_features, feature_costs, feature_chains)

        # Score function
        # Grow terminating set until all requirements are satisfied
        sigma: Set[Tuple[int, int]] = set()
        terminating_set: intbitset = intbitset()
        pending_requirements: List[int] = list(range(len(self._requirements)))
        while len(pending_requirements) > 0:
            # Features with non-zero cost are eligible to enter the solution
            eligible_features: List[int] = [f_idx for f_idx, _ in self._relevant_features if feature_costs[self._f_idx_to_feature_index[f_idx]] > 0]

            # Sort eligible features by score
            eligible_features_with_scores: List[Tuple[int, int]] = [(f_idx, self._score_fn(f_idx, pending_requirements, feature_costs, feature_chains)) for f_idx in eligible_features]
            eligible_features_with_non_zero_score: List[Tuple[int, int]] = [(f_idx, score) for f_idx, score in eligible_features_with_scores if score > 0]
            sorted_eligible_features_with_non_zero_score: List[Tuple[int, int]] = sorted(eligible_features_with_non_zero_score, key=lambda item: item[1], reverse=True)

            # Check for early termination due to non-existence of solution
            if len(sorted_eligible_features_with_non_zero_score) == 0:
                logging.warning(f"No eligible features: terminating_set={sorted(terminating_set)}")
                logging.warning(f"Pending requirements:")
                for r_idx in pending_requirements:
                    annotation, requirement = self._annotated_requirements[r_idx]
                    key: str = annotation.get("key")
                    if key == "Edge":
                        ext_state: Tuple[int, int] = annotation.get("ext_state")
                        ext_edge: Tuple[int, Tuple[int, int]] = self._ext_state_to_ext_edge.get(ext_state)
                        src_dlplan_state: dlplan_core.State = self._state_factory.get_dlplan_state(ext_edge[0], ext_edge[1][0])
                        dst_dlplan_state: dlplan_core.State = self._state_factory.get_dlplan_state(ext_edge[0], ext_edge[1][1])
                        logging.warning(f"  r_idx={r_idx}, key={key}, ext_edge={ext_edge}, requirement={requirement}")
                        logging.warning(f"      src_state: {ext_edge[1][0]}.{src_dlplan_state}")
                        logging.warning(f"      dst_state: {ext_edge[1][1]}.{dst_dlplan_state}")
                    elif key == "Goal":
                        path: Tuple[Tuple[int, int]] = annotation.get("path")
                        pair: Tuple[int, Tuple[int, int]] = pair
                        dlplan_state_0: dlplan_core.State = self._state_factory.get_dlplan_state(pair[0], pair[1][0])
                        dlplan_state_1: dlplan_core.State = self._state_factory.get_dlplan_state(pair[0], pair[1][1])
                        logging.warning(f"  r_idx={r_idx}, key={key}, pair={pair}, requirement={requirement}")
                        logging.warning(f"      state_0: {pair[1][0]}.{dlplan_state_0}")
                        logging.warning(f"      state_1: {pair[1][1]}.{dlplan_state_1}")
                    elif key == "Deadend":
                        ext_edge: Tuple[int, Tuple[int, int]] = annotation.get("path")
                        src_dlplan_state: dlplan_core.State = self._state_factory.get_dlplan_state(ext_edge[0], ext_edge[1][0])
                        dst_dlplan_state: dlplan_core.State = self._state_factory.get_dlplan_state(ext_edge[0], ext_edge[1][1])
                        logging.warning(f"  r_idx={r_idx}, key={key}, ext_edge={ext_edge}, requirement={requirement}")
                        logging.warning(f"      src_state: {ext_edge[1][0]}.{src_dlplan_state}")
                        logging.warning(f"      dst_state: {ext_edge[1][1]}.{dst_dlplan_state}")
                    elif key == "Sibling":
                        ext_sibling: Tuple[int, int, Tuple[int, int]] = annotation.get("ext_sibling")
                        src_dlplan_state: dlplan_core.State = self._state_factory.get_dlplan_state(ext_sibling[0], ext_sibling[1])
                        dst1_dlplan_state: dlplan_core.State = self._state_factory.get_dlplan_state(ext_sibling[0], ext_sibling[2][0])
                        dst2_dlplan_state: dlplan_core.State = self._state_factory.get_dlplan_state(ext_sibling[0], ext_sibling[2][1])
                        logging.warning(f"  r_idx={r_idx}, key={key}, ext_sibling={ext_sibling}, requirement={requirement}")
                        logging.warning(f"       src_state: {ext_sibling[1]}.{src_dlplan_state}")
                        logging.warning(f"      dst1_state: {ext_sibling[2][0]}.{dst1_dlplan_state}")
                        logging.warning(f"      dst2_state: {ext_sibling[2][1]}.{dst2_dlplan_state}")
                        assert False
                raise RuntimeError(f"No eligible features")

            # Choose a best eligible feature
            best_eligible_features: List[Tuple[int]] = [f_idx for f_idx, score in sorted_eligible_features_with_non_zero_score if score == sorted_eligible_features_with_non_zero_score[0][1]]
            best_eligible_feature: int = random.choice(best_eligible_features)
            best_cost: int = feature_costs[self._f_idx_to_feature_index[best_eligible_feature]]
            best_chain: Tuple[int] = feature_chains[self._f_idx_to_feature_index[best_eligible_feature]]
            logging.info(f"#eligible={len(eligible_features)}, #best={len(best_eligible_features)}, score={self._score_fn(best_eligible_feature, pending_requirements, feature_costs, feature_chains):0.2f}, f_idx={best_eligible_feature}.{self._relevant_features[self._f_idx_to_feature_index[best_eligible_feature]][1]._dlplan_feature}, chain={best_chain}, cost={best_cost}")

            # Extend sigma
            sigma: Set[Tuple[int, int]] = sigma | set(zip(best_chain, (None,) + best_chain[:-1]))

            # Extend terminating set and revise feature costs and chains
            terminating_set |= intbitset(best_chain)
            revised_f_idxs: List[int] = []
            for f_idx in best_chain:
                f_idx_index = self._f_idx_to_feature_index[f_idx]
                f_idx_cost = feature_costs[f_idx_index]
                assert f_idx_cost < int(1e6)
                if f_idx_cost > 0:
                    feature_costs[f_idx_index] = 0
                    revised_f_idxs.append(f_idx)
            self._revise_costs_and_chains(revised_f_idxs, feature_costs, feature_chains)

            # Recompute pending requirements
            pending_requirements: List[int] = [i for i in pending_requirements if len(terminating_set & self._requirements[i]) == 0]
            logging.info(f"{len(pending_requirements)} pending requirement(s)")

        # (Don't) Simplify set of selected features
        terminating_set: intbitset = intbitset(terminating_set)
        feature_ranks: Dict[int, int] = self._calculate_ranks_from_chains(terminating_set, feature_chains)
        assert self._does_it_solve_requirements(terminating_set, feature_ranks, verbose=True)

        """
        # THIS IS BUGGY....
        terminating_set: intbitset = self._simplify_terminating_set(terminating_set, feature_ranks)
        feature_ranks: Dict[int, int] = self._calculate_ranks_from_chains(terminating_set, feature_chains)
        assert self._does_it_solve_requirements(terminating_set, feature_ranks, verbose=True)
        """

        cost = sum([self._relevant_features[self._f_idx_to_feature_index[f_idx]][1].complexity for f_idx in terminating_set])
        sigma: Set[Tuple[int, int]] = set([(f_idx, g_idx) for f_idx, g_idx in sigma if f_idx in terminating_set and (g_idx is None or g_idx in terminating_set)])

        logging.info(f"Terminating_set: {sorted(terminating_set)}")
        logging.info(f"Cost: {cost}")
        logging.info(f"Ranks: {sorted([(f_idx, rank) for f_idx, rank in feature_ranks.items()], key=lambda item: item[1])}")

        """
        # THIS IS BUGGY
        terminating_set: intbitset = intbitset(terminating_set)
        cost = sum([self._relevant_features[self._f_idx_to_feature_index[f_idx]][1].complexity for f_idx in terminating_set])
        logging.info(f"terminating_set: {sorted(terminating_set)}")
        logging.info(f"cost: {cost}")

        # Calculate rank of selected features
        feature_ranks: Dict[int, int] = self._calculate_ranks_from_chains(terminating_set, feature_chains)
        logging.info(f"Ranks: {sorted([(f_idx, rank) for f_idx, rank in feature_ranks.items()], key=lambda item: item[1])}")
        assert self._does_it_solve_requirements(terminating_set, feature_ranks, verbose=True)

        # Simplify set of selected features
        terminating_set: intbitset = self._simplify_terminating_set(terminating_set, feature_ranks)
        """

        # Calculate decorations
        if self._simplify_policy:
            #decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = self._calculate_decorations(terminating_set, feature_ranks)
            decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = self._calculate_decorations2(terminating_set, feature_ranks, sigma)
            logging.info(f"Decorations: {decorations}")
        else:
            decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = dict()

        local_timer.stop()
        logging.info(f"Greedy solver finished in {local_timer.get_elapsed_sec():0.2f} second(s)")
        return True, terminating_set, [cost], decorations, feature_ranks

