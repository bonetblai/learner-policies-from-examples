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
                 sigma: Set[Tuple[int, int]],
                 ext_state_to_ext_edge: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]],
                 ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray],
                 bad_ext_edges: Set[Tuple[int, Tuple[int, int]]]):
        self._features: intbitset = features
        self._numerical_features: intbitset = numerical_features
        self._sigma: Set[Tuple[int, int]] = sigma
        self._ext_state_to_feature_valuations: Dict[Tuple[int, int], np.ndarray] = ext_state_to_feature_valuations
        logging.info(f"StratifiedPolicy: Sigma={sigma}")

        self._rules: List[Rule] = []
        self._rule_idx_to_ext_edge: List[Tuple[int, Tuple[int, int]]] = []
        for ext_state, ext_edge in ext_state_to_ext_edge.items():
            assert ext_state[0] == ext_edge[0] and ext_state[1] == ext_edge[1][0]
            instance_idx, (src_state_idx, dst_state_idx) = ext_edge
            src_feature_values: np.ndarray = ext_state_to_feature_valuations.get(ext_state)
            dst_feature_values: np.ndarray = ext_state_to_feature_valuations.get((instance_idx, dst_state_idx))
            conditions: Dict[int, str] = {h_idx: "eq" if src_feature_values[h_idx] == 0 else "gt" for h_idx in features}
            effects: List[Tuple[int, int]] = [(h_idx, dst_feature_values[h_idx] - src_feature_values[h_idx]) for h_idx in features]
            effects: Dict[int, str] = {h_idx: "bot" if d == 0 else "inc" if d > 0 else "dec" for h_idx, d in effects}
            logging.debug(f"ext_edge={ext_edge}: conditions={conditions}, effects={effects}")
            self._rules.append(Rule(conditions, effects))
            self._rule_idx_to_ext_edge.append(ext_edge)

        opposite: Dict[str, str] = {"eq": "gt", "gt": "eq"}
        self._varrho: Dict[Tuple[int, str], Set[int]] = defaultdict(set)
        for g_idx in self._features:
            for r_idx, rule in enumerate(self._rules):
                if rule.effect(g_idx) not in ["dec", "inc"]:
                    for cond in ["eq", "gt"]:
                        if rule.condition(g_idx) != opposite[cond]:
                            self._varrho[(g_idx, cond)].add(r_idx)

        self._bad_ext_edges: List[Tuple[int, Tuple[int, int]]] = list(bad_ext_edges)
        logging.info(f"StratifiedPolicy: bad_ext_edges={self._bad_ext_edges}")

    # Recompute varrho given condition to delete in rule
    def _revise_varrho_condition(self,
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

        # if h_idx = 0 is removed from r_idx and r_idx may not change h_idx, r_idx must be added to \varrho[h_idx][gt]
        if condition == "eq" and effect in ["bot", "unk"]:
            assert r_idx not in varrho_prime.get((h_idx, "gt"), [])
            varrho_prime[(h_idx, "gt")].add(r_idx)
            change = True

        # if h_idx > 0 is removed from r_idx and r_idx may not change h_idx, r_idx must be added to \varrho[h_idx][eq]
        if condition == "gt" and effect in ["bot", "unk"]:
            assert r_idx not in varrho_prime.get((h_idx, "eq"), [])
            varrho_prime[(h_idx, "eq")].add(r_idx)
            change = True

        return varrho_prime, change

    # Recompute varrho given effect to delete in rule
    def _revise_varrho_effect(self,
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
            assert condition == "dc" or r_idx not in varrho_prime.get((h_idx, condition), [])
            if condition != "dc":
                varrho_prime[(h_idx, condition)].add(r_idx)
                change = True
            else:
                change = r_idx not in varrho_prime.get((h_idx, "eq"), []) or r_idx not in varrho_prime.get((h_idx, "gt"), [])
                varrho_prime[(h_idx, "eq")].add(r_idx)
                varrho_prime[(h_idx, "gt")].add(r_idx)

        # if BOT(h_idx) is replaced by UNK(h_idx) in r_idx, nothing must be done as r_idx should already belong to correct places
        if effect == "bot" and condition != "dc":
            assert r_idx in varrho_prime.get((h_idx, condition))
        elif effect == "bot":
            assert r_idx in varrho_prime.get((h_idx, "eq")) and r_idx in varrho_prime.get((h_idx, "gt"))

        return varrho_prime, change

    def monotone(self, f_idx: int, rules: List[Rule]) -> bool:
        f_idx_direction: str = None # "inc"=increase, "dec"=decrease, "eq"=equal, "unk"=unknown
        for r_idx, rule in enumerate(rules):
            rule_direction: str = rule.effect(f_idx)
            if rule_direction == "unk":
                #print(f"HOLA.0")
                return False
            elif rule_direction != "bot":
                if f_idx_direction is None:
                    f_idx_direction = rule_direction
                elif f_idx_direction != rule_direction:
                    #print(f"HOLA.1: r_idx={r_idx}, f_idx={f_idx}, f_idx_direction={f_idx_direction}, rule_direction={rule_direction}")
                    return False
        return True

    def stratified(self, rules: List[Rule] = None, varrho: Dict[Tuple[int, str], Set[int]] = None) -> bool:
        _rules: List[Rule] = rules if rules is not None else self._rules
        _varrho: Dict[Tuple[int, str], Set[int]] = varrho if varrho is not None else self._varrho
        for f_idx, g_idx in self._sigma:
            #print((f_idx, g_idx))
            if g_idx is None:
                if not self.monotone(f_idx, _rules):
                    #print(f"HOLA.2")
                    return False
            elif not self.monotone(f_idx, [_rules[r_idx] for r_idx in _varrho.get((g_idx, "eq"), [])]) or not self.monotone(f_idx, [_rules[r_idx] for r_idx in _varrho.get((g_idx, "gt"), [])]):
                #print(f"HOLA.3")
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
        _rules: List[Rule] = rules if rules is not None else self._rules
        return self.compatible_edges(_rules, self._bad_ext_edges)

    def calculate_decorations(self, simplify_only_conditions: bool = False) -> Dict[str, Dict[int, Dict[int, intbitset]]]:
        logging.info(f"StratifiedPolicy: Calculate decorations...")
        unknowns: Dict[int, Dict[int, intbitset]] = defaultdict(lambda: defaultdict(intbitset))
        dont_cares: Dict[int, Dict[int, intbitset]] = defaultdict(lambda: defaultdict(intbitset))

        # Input rules
        logging.info(f"StratifiedPolicy: Input rules:")
        self.print_rules()
        self.print_varrho()
        assert self.stratified(), "Input ruleset isn't stratified"
        assert not self.accept_some_bad_edge(), "Input ruleset accepts bad edges"

        # Add decorations until fixpoint
        rules: List[Rule] = [rule.clone() for rule in self._rules]
        varrho: Dict[Tuple[int, str], Set[int]] = self._varrho
        change: bool = True
        while change:
            change = False

            # Simplify conditions (increase coverage)
            logging.info(f"StratifiedPolicy:   Simplify conditions...")
            for r_idx in range(len(rules)):
                conditions: List[Tuple[int, str]] = list(rules[r_idx].conditions().items())
                for h_idx, condition in conditions:
                    # Avoid don't care conditions for Boolean features as effect is compatible with BOT
                    if h_idx in self._numerical_features and condition != "dc":
                        rule_prime: Rule = rules[r_idx].clone()
                        rule_prime.remove_condition(h_idx)
                        rules_prime: List[Rule] = rules[:r_idx] + [rule_prime] + rules[r_idx+1:]
                        if self.accept_some_bad_edge(rules_prime): continue
                        varrho_prime, varrho_change = self._revise_varrho_condition(r_idx, h_idx, rules, varrho)
                        if self.stratified(rules_prime, varrho_prime):
                            ext_edge: Tuple[int, Tuple[int, int]] = self._rule_idx_to_ext_edge[r_idx]
                            instance_idx, state_idx = ext_edge[0], ext_edge[1][0]
                            logging.debug(f"StratifiedPolicy:     MARK.0: Mark f{h_idx} as DC in rule r{r_idx}.{rules[r_idx]} associated with ext_edge {ext_edge}")
                            dont_cares[instance_idx][state_idx].add(h_idx)
                            rules = rules_prime
                            varrho = varrho_prime
                            change = True
                            assert self.stratified(rules, varrho), f"Ruleset isn't stratified after inserting DC(f{h_idx}) into rule r{r_idx}.{rules[r_idx]}"
            assert not self.accept_some_bad_edge(rules), f"Ruleset accepts bad edges"

            # Simplify effects (increase trajectories)
            if not simplify_only_conditions:
                logging.info(f"StratifiedPolicy:   Simplify effects...")
                for r_idx in range(len(rules)):
                    effects: List[Tuple[int, str]] = [(h_idx, effect) for h_idx, effect in rules[r_idx].effects().items() if effect != "unk"]
                    assert len([1 for h_idx, effect in effects if effect != "bot"]) > 0
                    unk_effects: List[int] = []
                    for h_idx, effect in effects:
                        if len([1 for h_idx, effect in effects if effect != "bot"]) - len(unk_effects) == 1: break
                        rule_prime: Rule = rules[r_idx].clone()
                        rule_prime.remove_effect(h_idx)
                        rules_prime: List[Rule] = rules[:r_idx] + [rule_prime] + rules[r_idx+1:]
                        if self.accept_some_bad_edge(rules_prime): continue
                        varrho_prime, varrho_change = self._revise_varrho_effect(r_idx, h_idx, rules, varrho)
                        if self.stratified(rules_prime, varrho_prime):
                            ext_edge: Tuple[int, Tuple[int, int]] = self._rule_idx_to_ext_edge[r_idx]
                            instance_idx, state_idx = ext_edge[0], ext_edge[1][0]
                            logging.debug(f"StratifiedPolicy:     MARK.1: Mark f{h_idx} as UNK in rule r{r_idx}.{rules[r_idx]} associated with ext_edge {ext_edge}")
                            unknowns[instance_idx][state_idx].add(h_idx)
                            rules = rules_prime
                            varrho = varrho_prime
                            change = True
                            unk_effects.append(h_idx)
                            assert self.stratified(rules, varrho), f"Ruleset isn't stratified after inserting UNK(f{h_idx}) into rule r{r_idx}.{rules[r_idx]}"
                    assert len([1 for h_idx, effect in effects if effect != "bot"]) - len(unk_effects) >= 1
                assert not self.accept_some_bad_edge(rules), f"Ruleset accepts bad edges"

            logging.info(f"StratifiedPolicy:   change={change}")

        # Simplified rules
        logging.info(f"StratifiedPolicy: Simplified rules:")
        self.print_rules(rules=rules, ext_edges=False)

        # Return
        unknowns: Dict[int, Dict[int, intbitset]] = {instance_idx: dict(dict_for_unknowns) for instance_idx, dict_for_unknowns in unknowns.items()}
        dont_cares: Dict[int, Dict[int, intbitset]] = {instance_idx: dict(dict_for_dont_cares) for instance_idx, dict_for_dont_cares in dont_cares.items()}
        decorations: Dict[str, Dict[int, Dict[int, intbitset]]] = {"unknown": unknowns, "dont_care": dont_cares}
        return decorations

    def print_rules(self, rules: List[Rule] = None, ext_edges: bool = True, indent: int = 0):
        _rules: List[Rule] = rules if rules is not None else self._rules
        for r_idx, rule in enumerate(_rules):
            logging.info(f"{' ' * indent}r{r_idx}.{rule}" + ("" if not ext_edges else f" [ext_edge={self._rule_idx_to_ext_edge[r_idx]}]"))

    def print_varrho(self, varrho: Dict[Tuple[int, str], Set[int]] = None, indent: int = 0):
        _varrho: Dict[Tuple[int, str], Set[int]] = varrho if varrho is not None else self._varrho
        for (g_idx, cond), r_idxs in self._varrho.items():
            logging.info(f"{' ' * indent}Varrho[{g_idx}][{cond}] = {sorted(r_idxs)}")

