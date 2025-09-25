import logging
import numpy as np

# Bitset-based unordered sets of unsigned integers
from intbitset import intbitset

from termcolor import colored
from typing import Set, Tuple, List, Union, Dict, Any, Optional, Union
from collections import OrderedDict, defaultdict, deque

import dlplan.core as dlplan_core

from ..feature_pool import Feature
from ...util import Timer
from ...state_space import StateFactory


class Rule:
    def __init__(self, conditions: Dict[int, str], effects: Dict[int, str]):
        self._conditions: Dict[int, str] = conditions
        self._effects: Dict[int, str] = effects

    def clone(self) -> Any:
        return Rule(dict(self._conditions), dict(self._effects))

    # Conditions: condition on f_idx is either "eq" or "gt"; if f_idx not mapped, condition is "don't care"
    def conditions(self) -> Dict[int, str]:
        return self._conditions

    def condition(self, f_idx) -> str:
        return self._conditions.get(f_idx, "dc")

    def remove_condition(self, f_idx: int) -> str:
        return self._conditions.pop(f_idx, None)

    # Effects: effect on f_idx is either "dec", "inc", or "bot"; if f_idx not mapped, effect is "unk"
    def effects(self) -> Dict[int, str]:
        return self._effects

    def effect(self, f_idx) -> str:
        return self._effects.get(f_idx, "unk")

    def remove_effect(self, f_idx: int) -> str:
        return self._effects.pop(f_idx, None)

    # Rule is compatible with transition (s,t) if conditions are satisfied by s, and changes across (s,t) are comatible with effects
    def compatible(self, conditions_at_src: Dict[int, str], changes_across_transition: Dict[int, str]) -> bool:
        # Verify conditions
        for f_idx, condition in self._conditions.items():
            assert condition in ["eq", "gt"], f"Unexpected rule conditions: {self._conditions}"
            condition_at_src: str = conditions_at_src.get(f_idx)
            assert condition_at_src in ["eq", "gt"], f"Unexpected conditions_at_src: {conditions_at_src}"
            if condition != condition_at_src:
                return False

        # Verify effects
        for f_idx, effect in self._effects.items():
            assert effect in ["inc", "dec", "bot"], f"Unexpected rule effects: {self._effects}"
            change_across_transition: str = changes_across_transition.get(f_idx)
            assert change_across_transition in ["inc", "dec", "bot"], f"Unexpected changes_across_transition: {changes_across_transition}"
            if effect != change_across_transition:
                return False

        return True

    def __str__(self):
        conditions: List[str] = [f"{condition.upper()}(f{f_idx})" for f_idx, condition in sorted(self._conditions.items(), key=lambda item: item[0])]
        effects: List[str] = [f"{effect.upper()}(f{f_idx})" for f_idx, effect in sorted(self._effects.items(), key=lambda item: item[0])]
        return f"{{ {', '.join(conditions)} }} -> {{ {', '.join(effects)} }}"


class StratifiedPolicy:
    def __init__(self,
                 features: intbitset,
                 numerical_features: intbitset,
                 ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]],
                 ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray],
                 bad_ext_edges: Set[Tuple[int, Tuple[int, int]]]):
        self._features: intbitset = features
        self._numerical_features: intbitset = numerical_features
        self._ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = ext_state_to_feature_valuations

        # Calculate feature-based rules
        self._rules: List[Rule] = []
        self._rule_idx_to_ext_edge: List[Tuple[int, Tuple[int, int]]] = []
        self._calculate_rules(ext_state_to_ext_edge)

        # Bad edges
        self._bad_ext_edges: List[Tuple[int, Tuple[int, int]]] = list(bad_ext_edges)
        logging.info(f"StratifiedPolicy: bad_ext_edges={self._bad_ext_edges}")

    def _calculate_rules(self, ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]]):
        self._rules: List[Rule] = []
        self._rule_idx_to_ext_edge: List[Tuple[int, Tuple[int, int]]] = []
        for ext_state, ext_edge in ext_state_to_ext_edge.items():
            assert ext_state[0] == ext_edge[0] and ext_state[1] == ext_edge[1][0]
            instance_idx, (src_state_idx, dst_state_idx) = ext_edge
            src_feature_values: np.ndarray = self._ext_state_to_feature_valuations.get(ext_state)
            dst_feature_values: np.ndarray = self._ext_state_to_feature_valuations.get((instance_idx, dst_state_idx))
            conditions: Dict[int, str] = {h_idx: "eq" if src_feature_values[h_idx] == 0 else "gt" for h_idx in self._features}
            effects: List[Tuple[int, int]] = [(h_idx, dst_feature_values[h_idx] - src_feature_values[h_idx]) for h_idx in self._features]
            effects: Dict[int, str] = {h_idx: "bot" if d == 0 else "inc" if d > 0 else "dec" for h_idx, d in effects}
            logging.debug(f"ext_edge={ext_edge}: conditions={conditions}, effects={effects}")
            self._rules.append(Rule(conditions, effects))
            self._rule_idx_to_ext_edge.append(ext_edge)

    def is_monotone(self, f_idx: int, rules: List[Rule]) -> bool:
        f_idx_direction: str = None
        for r_idx, rule in enumerate(rules):
            rule_direction: str = rule.effect(f_idx)
            if rule_direction == "unk":
                return False
            elif rule_direction != "bot":
                if f_idx_direction is None:
                    f_idx_direction = rule_direction
                elif f_idx_direction != rule_direction:
                    return False
        return True

    def compatible(self, rule: Rule, ext_edge: Tuple[int, Tuple[int, int]]) -> bool:
        src_ext_state: Tuple[int, int] = (ext_edge[0], ext_edge[1][0])
        dst_ext_state: Tuple[int, int] = (ext_edge[0], ext_edge[1][1])
        src_feature_values: np.ndarray = self._ext_state_to_feature_valuations.get(src_ext_state)
        dst_feature_values: np.ndarray = self._ext_state_to_feature_valuations.get(dst_ext_state)
        conditions: Dict[int, str] = {h_idx: "eq" if src_feature_values[h_idx] == 0 else "gt" for h_idx in self._features}
        effects: List[Tuple[int, int]] = [(h_idx, dst_feature_values[h_idx] - src_feature_values[h_idx]) for h_idx in self._features]
        effects: Dict[int, str] = {h_idx: "bot" if d == 0 else "inc" if d > 0 else "dec" for h_idx, d in effects}
        return rule.compatible(conditions, effects)

    def compatible_edges(self, rules: List[Rule], ext_edges: List[Tuple[int, Tuple[int, int]]], single_edge: bool = True) -> bool:
        for ext_edge in ext_edges:
            for rule in rules:
                compatible: bool = self.compatible(rule, ext_edge)
                if single_edge and compatible:
                    return True
                elif not single_edge and not compatible:
                    return False
        return False if single_edge else True

    def accept_some_bad_edge(self, rules: List[Rule] = None) -> bool:
        _rules: List[Rule] = rules or self._rules
        return self.compatible_edges(_rules, self._bad_ext_edges)

    def print_rules(self, rules: List[Rule] = None, ext_edges: bool = True, indent: int = 0):
        _rules: List[Rule] = rules or self._rules
        for r_idx, rule in enumerate(_rules):
            logging.info(f"{' ' * indent}r{r_idx}.{rule}" + ("" if not ext_edges else f" [ext_edge={self._rule_idx_to_ext_edge[r_idx]}]"))


class StratifiedPolicyByFeatures(StratifiedPolicy):
    def __init__(self,
                 features: intbitset,
                 numerical_features: intbitset,
                 sigma: Set[Tuple[int, int]],
                 ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]],
                 ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray],
                 bad_ext_edges: Set[Tuple[int, Tuple[int, int]]]):
        super().__init__(features, numerical_features, ext_state_to_ext_edge, ext_state_to_feature_valuations, bad_ext_edges)
        self._sigma: Set[Tuple[int, int]] = sigma
        self._varrho: Dict[Tuple[int, str], Set[int]] = self._calculate_varrho(self._features, self._rules)
        logging.info(f"StratifiedPolicyByFeatures: sigma={self._sigma}")

    def _calculate_varrho(self, features: intbitset, rules: List[Rule]) -> Dict[Tuple[int, str], Set[int]]:
        opposite: Dict[str, str] = {"eq": "gt", "gt": "eq"}
        varrho: Dict[Tuple[int, str], Set[int]] = defaultdict(set)
        for g_idx in features:
            for r_idx, rule in enumerate(rules):
                if rule.effect(g_idx) not in ["dec", "inc"]:
                    for cond in ["eq", "gt"]:
                        if rule.condition(g_idx) != opposite[cond]:
                            varrho[(g_idx, cond)].add(r_idx)
        return varrho

    # Recompute varrho given condition to delete in rule
    def _revise_varrho_with_condition(self,
                                      r_idx: int,
                                      h_idx: int,
                                      rules: List[Rule],
                                      varrho: Dict[Tuple[int, str], Set[int]]) -> Tuple[Dict[Tuple[int, str], Set[int]], bool]:
        varrho_prime: Dict[Tuple[int, str], Set[int]] = defaultdict(set)
        varrho_prime.update({key: set(value) for key, value in varrho.items()})
        rule: Rule = rules[r_idx]
        condition: str = rule.condition(h_idx)
        effect: str = rule.effect(h_idx)
        change: bool = False

        assert condition in ["eq", "gt", "dc"]
        assert effect in ["inc", "dec", "bot", "unk"]

        # if EQ(h_idx) is removed from r_idx and r_idx may not change h_idx, r_idx must be added to \varrho[h_idx][GT]
        if condition == "eq" and effect in ["bot", "unk"]:
            assert r_idx not in varrho_prime.get((h_idx, "gt"), [])
            varrho_prime[(h_idx, "gt")].add(r_idx)
            change = True

        # if GT(h_idx) is removed from r_idx and r_idx may not change h_idx, r_idx must be added to \varrho[h_idx][EQ]
        if condition == "gt" and effect in ["bot", "unk"]:
            assert r_idx not in varrho_prime.get((h_idx, "eq"), [])
            varrho_prime[(h_idx, "eq")].add(r_idx)
            change = True

        return varrho_prime, change

    # Recompute varrho given effect to delete in rule
    def _revise_varrho_with_effect(self,
                                   r_idx: int,
                                   h_idx: int,
                                   rules: List[Rule],
                                   varrho: Dict[Tuple[int, str], Set[int]]) -> Tuple[Dict[Tuple[int, str], Set[int]], bool]:
        varrho_prime: Dict[Tuple[int, str], Set[int]] = defaultdict(set)
        varrho_prime.update({key: set(value) for key, value in varrho.items()})
        rule: Rule = rules[r_idx]
        condition: str = rule.condition(h_idx)
        effect: str = rule.effect(h_idx)
        change: bool = False

        assert condition in ["eq", "gt", "dc"]
        assert effect in ["inc", "dec", "bot", "unk"]

        # if INC(h_idx) or DEC(h_idx) is replaced by UNK(h_idx) in r_idx, r_idx must be added to \varrho[h_idx][condition]
        if effect in ["inc", "dec"]:
            assert r_idx not in varrho_prime.get((h_idx, "eq"), [])
            assert r_idx not in varrho_prime.get((h_idx, "gt"), [])
            if condition != "dc":
                varrho_prime[(h_idx, condition)].add(r_idx)
            else:
                varrho_prime[(h_idx, "eq")].add(r_idx)
                varrho_prime[(h_idx, "gt")].add(r_idx)
            change = True

        # if BOT(h_idx) is replaced by UNK(h_idx) in r_idx, nothing must be done as r_idx should already belong to correct places
        if effect == "bot" and condition != "dc":
            assert r_idx in varrho_prime.get((h_idx, condition))
        elif effect == "bot":
            assert r_idx in varrho_prime.get((h_idx, "eq"))
            assert r_idx in varrho_prime.get((h_idx, "gt"))

        return varrho_prime, change

    def is_stratified(self, rules: List[Rule] = None, varrho: Dict[Tuple[int, str], Set[int]] = None, debug: bool = False) -> bool:
        _rules: List[Rule] = rules or self._rules
        _varrho: Dict[Tuple[int, str], Set[int]] = varrho or self._varrho
        for f_idx, g_idx in self._sigma:
            if g_idx is None:
                if not self.is_monotone(f_idx, _rules):
                    return False
            else:
                if not self.is_monotone(f_idx, [_rules[r_idx] for r_idx in _varrho.get((g_idx, "eq"), [])]) or \
                   not self.is_monotone(f_idx, [_rules[r_idx] for r_idx in _varrho.get((g_idx, "gt"), [])]):
                    return False
        return True

    def calculate_decorations(self, simplify_only_conditions: bool = False) -> Dict[str, Dict[int, Dict[int, intbitset]]]:
        unknowns: Dict[int, Dict[int, intbitset]] = defaultdict(lambda: defaultdict(intbitset))
        dont_cares: Dict[int, Dict[int, intbitset]] = defaultdict(lambda: defaultdict(intbitset))

        # Input rules and varrho
        logging.info(f"calculate_decorations: Input rules:")
        rules: List[Rule] = [rule.clone() for rule in self._rules]
        varrho: Dict[Tuple[int, str], Set[int]] = self._varrho

        self.print_rules(rules=rules)
        self.print_varrho(varrho=varrho)
        assert self.is_stratified(rules=rules, varrho=varrho), "Input ruleset isn't stratified"
        assert not self.accept_some_bad_edge(), "Input ruleset accepts bad edges"
        logging.info(f"calculate_decorations: Input rules are stratified")

        # Add decorations until fixpoint
        change: bool = True
        while change:
            change = False

            # Simplify conditions (increase coverage)
            logging.info(f"calculate_decorations:   Simplify conditions...")
            for r_idx in range(len(rules)):
                conditions: List[Tuple[int, str]] = list(rules[r_idx].conditions().items())
                for h_idx, condition in conditions:
                    # Avoid adding don't care conditions for Boolean features as rule effect would then be compatible with BOT
                    if h_idx in self._numerical_features:
                        rule_prime: Rule = rules[r_idx].clone()
                        rule_prime.remove_condition(h_idx)
                        rules_prime: List[Rule] = rules[:r_idx] + [rule_prime] + rules[r_idx+1:]
                        if self.accept_some_bad_edge(rules_prime): continue
                        varrho_prime, varrho_change = self._revise_varrho_with_condition(r_idx, h_idx, rules, varrho)
                        if self.is_stratified(rules=rules_prime, varrho=varrho_prime):
                            ext_edge: Tuple[int, Tuple[int, int]] = self._rule_idx_to_ext_edge[r_idx]
                            instance_idx, state_idx = ext_edge[0], ext_edge[1][0]
                            logging.debug(f"calculate_decorations:     MARK.0: Mark f{h_idx} as DC in rule r{r_idx}.{rules[r_idx]} associated with ext_edge {ext_edge}")
                            dont_cares[instance_idx][state_idx].add(h_idx)
                            rules = rules_prime
                            varrho = varrho_prime
                            change = True
                            assert self.compatible(rule_prime, ext_edge)
            assert not self.accept_some_bad_edge(rules), f"Ruleset accepts bad edges"
            logging.info(f"calculate_decorations:   Ruleset remains stratified")

            # Simplify effects (increase trajectories)
            if not simplify_only_conditions:
                logging.info(f"calculate_decorations:   Simplify effects...")
                for r_idx in range(len(rules)):
                    effects: List[Tuple[int, str]] = list(rules[r_idx].effects().items())
                    assert len([1 for h_idx, effect in effects if effect != "bot"]) > 0
                    added_unk_effects: List[int] = []
                    for h_idx, effect in effects:
                        if len([1 for h_idx, effect in effects if effect != "bot"]) - len(added_unk_effects) == 1: break
                        rule_prime: Rule = rules[r_idx].clone()
                        rule_prime.remove_effect(h_idx)
                        rules_prime: List[Rule] = rules[:r_idx] + [rule_prime] + rules[r_idx+1:]
                        if self.accept_some_bad_edge(rules_prime): continue
                        varrho_prime, varrho_change = self._revise_varrho_with_effect(r_idx, h_idx, rules, varrho)
                        if self.is_stratified(rules=rules_prime, varrho=varrho_prime):
                            ext_edge: Tuple[int, Tuple[int, int]] = self._rule_idx_to_ext_edge[r_idx]
                            instance_idx, state_idx = ext_edge[0], ext_edge[1][0]
                            logging.debug(f"calculate_decorations:     MARK.1: Mark f{h_idx} as UNK in rule r{r_idx}.{rules[r_idx]} associated with ext_edge {ext_edge}")
                            unknowns[instance_idx][state_idx].add(h_idx)
                            rules = rules_prime
                            varrho = varrho_prime
                            change = True
                            added_unk_effects.append(h_idx)
                            assert self.compatible(rule_prime, ext_edge)
                    assert len([1 for h_idx, effect in effects if effect != "bot"]) - len(added_unk_effects) >= 1
                assert not self.accept_some_bad_edge(rules), f"Ruleset accepts bad edges"
                logging.info(f"calculate_decorations:   Ruleset remains stratified")

            logging.debug(f"calculate_decorations:   change={change}")

        # Simplified rules
        logging.info(f"calculate_decorations: Simplified rules:")
        self.print_rules(rules=rules, ext_edges=False)
        assert self.is_stratified(rules=rules, varrho=varrho, debug=True)

        # Return
        unknowns: Dict[int, Dict[int, intbitset]] = {instance_idx: dict(dict_for_unknowns) for instance_idx, dict_for_unknowns in unknowns.items()}
        dont_cares: Dict[int, Dict[int, intbitset]] = {instance_idx: dict(dict_for_dont_cares) for instance_idx, dict_for_dont_cares in dont_cares.items()}
        decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = {"unknown": unknowns, "dont_care": dont_cares}
        return decorations

    def print_varrho(self, varrho: Dict[Tuple[int, str], Set[int]] = None, indent: int = 0):
        _varrho: Dict[Tuple[int, str], Set[int]] = varrho or self._varrho
        for (g_idx, cond), r_idxs in _varrho.items():
            logging.info(f"{' ' * indent}Varrho[{g_idx}][{cond}] = {sorted(r_idxs)}")


class StratifiedPolicyByFeaturesContextual(StratifiedPolicyByFeatures):
    def __init__(self,
                 features: intbitset,
                 numerical_features: intbitset,
                 sigma: Dict[Tuple[int, Tuple[int, int]], str],
                 ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]],
                 ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray],
                 bad_ext_edges: Set[Tuple[int, Tuple[int, int]]]):
        super().__init__(features, numerical_features, sigma, ext_state_to_ext_edge, ext_state_to_feature_valuations, bad_ext_edges)
        self._monotone: intbitset = intbitset([f_idx for f_idx, nu_context in sigma if len(nu_context) == 0])
        logging.info(f"StratifiedPolicyByFeaturesContextual: mononote={self._monotone}")

        # Calculate list of contexts for sigma
        self._sigma_contexts: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        for f_idx, nu_context in self._sigma.keys():
            if len(nu_context) == 0:
                self._sigma_contexts[f_idx].append(nu_context)
            else:
                g_idx: int = nu_context[0]
                condition: str = "eq" if nu_context[1] == 0 else "gt"
                self._sigma_contexts[f_idx].append((g_idx, condition))
        logging.info(f"StratifiedPolicyByFeaturesContextual: sigma_contexts={dict(self._sigma_contexts)}")

    def is_stratified(self, rules: List[Rule] = None, varrho: Dict[Tuple[int, str], Set[int]] = None, debug: bool = False) -> bool:
        _rules: List[Rule] = rules or self._rules
        _varrho: Dict[Tuple[int, str], Set[int]] = varrho or self._varrho
        _prec: Dict[int, intbitset] = defaultdict(intbitset)

        # Check that all pairs (f,\nu) in sigma correspond to valid conditional monotonicities
        for f_idx, nu_context in self._sigma:
            if len(nu_context) == 0:
                if not self.is_monotone(f_idx, _rules):
                    return False
            else:
                g_idx, b_value = nu_context
                condition: str = "eq" if b_value == 0 else "gt"
                logging.debug(f"is_stratified: g_idx={g_idx}, condition={condition}, rules={_varrho.get((g_idx, condition), [])}")
                if not self.is_monotone(f_idx, [_rules[r_idx] for r_idx in _varrho.get((g_idx, condition), [])]):
                    return False
                _prec[f_idx].add(g_idx)

        # Check validity of termination theorem under assumption sigma is valid
        logging.info(f"is_stratified: {list(range(len(_rules)))}")
        for r_idx, rule in enumerate(_rules):
            f_idxs_change: intbitset = intbitset([f_idx for f_idx in self._features if rule.effect(f_idx) in ["dec", "inc"]])

            # If rule changes a monotone feature, condition is satisfied
            if len(f_idxs_change & self._monotone) > 0:
                logging.debug(f"  r{r_idx:02d}={rule}")
                logging.debug(f"  Cleared: changes monotone features {list(f_idxs_change & self._monotone)}")
                continue

            # There must be a pair (f, \nu) in \sigma such that f_idx is changed, f_idx is monotone given \nu, AND r_idx belongs to \varrho(R, \nu)
            found_pair: bool = False
            for f_idx in f_idxs_change:
                for nu_context in self._sigma_contexts.get(f_idx, []):
                    if r_idx in _varrho.get(nu_context, []):
                        found_pair = True
                        if debug:
                            logging.info(f"  r{r_idx:02d}={rule}")
                            logging.info(f"  Cleared: pair={(f_idx, nu_context)}, varrho={_varrho.get(nu_context)}")
                        break
                if found_pair: break

            if not found_pair:
                logging.info(f"is_stratified: Termination Theorem not satisfied: r{r_idx}.{rule}, f_idx={f_idx}")
                logging.info(f"is_stratified: sigma={self._sigma}")
                return False

        return True

