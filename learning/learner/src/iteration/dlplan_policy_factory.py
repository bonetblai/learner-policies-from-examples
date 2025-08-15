import logging
import re

from abc import ABC, abstractmethod

import dlplan.core as dlplan_core
import dlplan.policy as dlplan_policy
from dlplan.policy import PolicyFactory

from clingo import Symbol
from typing import Dict, Set, List, Tuple, Union, MutableSet, Any
from termcolor import colored

from .iteration_data import IterationData
from ..preprocessing import PreprocessingData


class DlplanPolicyFactory(ABC):
    """ """
    @abstractmethod
    def make_dlplan_policy_from_answer_set(self, symbols: List[Symbol], policy_builder: PolicyFactory, iteration_data: IterationData):
        """
        Parses the facts of an answer set into a dlplan policy.
        """

    @abstractmethod
    def make_dlplan_policy_from_r_idxs_with_decorations(self, r_idxs_with_decorations: Set[Tuple[int, Set[int], Set[int]]], r_idx_to_rule: Dict[int, Any], policy_builder: PolicyFactory, iteration_data: IterationData):
        pass

    @abstractmethod
    def make_dlplan_policy_from_rules_with_decorations(self, rules_with_decorations: Set[Tuple[Any, Set[int], Set[int]]], policy_builder: PolicyFactory, iteration_data: IterationData):
        pass


def extract_f_idx_from_argument(string: str):
    """
    Input examples: n1231 for numerical or b123 for boolean
    with respective outputs 1231 and 123.
    """
    return int(re.findall(r"[bn](.+)", string)[0])


class ExplicitDlplanPolicyFactory(DlplanPolicyFactory):
    """
    Encoding where rules are explicit in the ASP encoding (ICAPS2022)
    """
    def make_dlplan_policy_from_answer_set(self, symbols: List[Symbol], policy_builder: PolicyFactory, iteration_data: IterationData):
        """ """
        selected_features = self._add_features(symbols, policy_builder, iteration_data)
        rules = self._add_rules(symbols, policy_builder, iteration_data, selected_features)
        return policy_builder.make_policy(rules)

    def _add_features(self, symbols: List[Symbol], policy_builder: PolicyFactory, iteration_data: IterationData):
        """ """
        selected_features = set()
        for symbol in symbols:
            if symbol.name == "select":
                f_idx = symbol.arguments[0].number
                selected_features.add(iteration_data.feature_pool[f_idx].dlplan_feature)
        return selected_features

    def _add_rules(self, symbols: List[Symbol], policy_builder: PolicyFactory, iteration_data: IterationData, selected_features: MutableSet[Union[dlplan_core.Boolean, dlplan_core.Numerical]]):
        """ """
        rules_dict = dict()
        for symbol in symbols:
            if symbol.name == "rule":
                r_idx = symbol.arguments[0].number
                rules_dict[r_idx] = [set(), set()]  # conditions and effects
        for symbol in symbols:
            if symbol.name in {"c_b_pos", "c_b_neg", "c_n_gt", "c_n_eq", "e_b_pos", "e_b_neg", "e_b_bot", "e_n_dec", "e_n_inc", "e_n_bot"}:
                r_idx = symbol.arguments[0].number
                f_idx = symbol.arguments[1].number
                feature = iteration_data.feature_pool[f_idx].dlplan_feature
                if feature not in selected_features:
                    continue
                if symbol.name == "c_b_pos":
                    rules_dict[r_idx][0].add(policy_builder.make_pos_condition(policy_builder.make_boolean(f"f{f_idx}", feature)))
                elif symbol.name == "c_b_neg":
                    rules_dict[r_idx][0].add(policy_builder.make_neg_condition(policy_builder.make_boolean(f"f{f_idx}", feature)))
                elif symbol.name == "c_n_gt":
                    rules_dict[r_idx][0].add(policy_builder.make_gt_condition(policy_builder.make_numerical(f"f{f_idx}", feature)))
                elif symbol.name == "c_n_eq":
                    rules_dict[r_idx][0].add(policy_builder.make_eq_condition(policy_builder.make_numerical(f"f{f_idx}", feature)))
                elif symbol.name == "e_b_pos":
                    rules_dict[r_idx][1].add(policy_builder.make_pos_effect(policy_builder.make_boolean(f"f{f_idx}", feature)))
                elif symbol.name == "e_b_neg":
                    rules_dict[r_idx][1].add(policy_builder.make_neg_effect(policy_builder.make_boolean(f"f{f_idx}", feature)))
                elif symbol.name == "e_b_bot":
                    rules_dict[r_idx][1].add(policy_builder.make_bot_effect(policy_builder.make_boolean(f"f{f_idx}", feature)))
                elif symbol.name == "e_n_dec":
                    rules_dict[r_idx][1].add(policy_builder.make_dec_effect(policy_builder.make_numerical(f"f{f_idx}", feature)))
                elif symbol.name == "e_n_inc":
                    rules_dict[r_idx][1].add(policy_builder.make_inc_effect(policy_builder.make_numerical(f"f{f_idx}", feature)))
                elif symbol.name == "e_n_bot":
                    rules_dict[r_idx][1].add(policy_builder.make_bot_effect(policy_builder.make_numerical(f"f{f_idx}", feature)))
        rules = set()
        for _, (conditions, effects) in rules_dict.items():
            rules.add(policy_builder.make_rule(conditions, effects))
        return rules

    def make_dlplan_policy_from_r_idxs_with_decorations(self, r_idxs_with_decorations: Set[Tuple[int, Set[int], Set[int]]], r_idx_to_rule: Dict[int, Any], policy_builder: PolicyFactory, iteration_data: IterationData):
        assert False, "make_dlplan_policy_from_r_idxs_with_decorations"

    def make_dlplan_policy_from_rules_with_decorations(self, rules_with_decorations: Set[Tuple[Any, Set[int], Set[int]]], policy_builder: PolicyFactory, iteration_data: IterationData):
        assert False, "make_dlplan_policy_from_rules_with_decorations"

class D2sepDlplanPolicyFactory(DlplanPolicyFactory):
    """
    Encoding where rules are implicit in the D2-separation.
    """
    def make_dlplan_policy_from_answer_set(self, symbols: List[Symbol], policy_builder: PolicyFactory, iteration_data: IterationData):
        dlplan_features = set()
        for symbol in symbols:
            #_print_symbol(symbol)
            if symbol.name == "select":
                f_idx = symbol.arguments[0].number
                dlplan_features.add(iteration_data.feature_pool[f_idx].dlplan_feature)
        rules = set()
        for symbol in symbols:
            #_print_symbol(symbol)
            if symbol.name == "good":
                r_idx = symbol.arguments[0].number if len(symbol.arguments) == 1 else symbol.arguments[1].number
                rule = iteration_data.state_pair_equivalences[r_idx]
                conditions = set()
                for condition in rule.get_conditions():
                    f_idx = int(condition.get_named_element().get_key()[1:])
                    dlplan_feature = iteration_data.feature_pool[f_idx].dlplan_feature
                    if dlplan_feature in dlplan_features:
                        conditions.add(condition)
                effects = set()
                for effect in rule.get_effects():
                    f_idx = int(effect.get_named_element().get_key()[1:])
                    dlplan_feature = iteration_data.feature_pool[f_idx].dlplan_feature
                    if dlplan_feature in dlplan_features:
                        effects.add(effect)
                rules.add(policy_builder.make_rule(conditions, effects))
        return policy_builder.make_policy(rules)

    def make_dlplan_policy_from_r_idxs_with_decorations(self,
                                                        feature_idxs: Set[int],
                                                        r_idxs_with_decorations: Set[Tuple[int, Set[int], Set[int]]],
                                                        r_idx_to_rule: Dict[int, Any],
                                                        policy_builder: PolicyFactory,
                                                        iteration_data: IterationData):
        rules: Set[Any] = set()
        feature_pool: List[Any] = iteration_data.feature_pool
        for r_idx, unknowns, dont_cares in r_idxs_with_decorations:
            rule = r_idx_to_rule.get(r_idx)
            assert rule is not None
            rules.add(self._get_policy_rule(feature_idxs, rule, unknowns, dont_cares, policy_builder, feature_pool))
        return policy_builder.make_policy(rules)

    def make_dlplan_policy_from_rules_with_decorations(self,
                                                       feature_idxs: Set[int],
                                                       rules_with_decorations: Set[Tuple[Any, Set[int], Set[int]]],
                                                       policy_builder: PolicyFactory,
                                                       iteration_data: IterationData):
        feature_pool: List[Any] = iteration_data.feature_pool
        rules: Set[Any] = set([self._get_policy_rule(feature_idxs, rule, unknowns, dont_cares, policy_builder, feature_pool) for rule, unknowns, dont_cares in rules_with_decorations])
        return policy_builder.make_policy(rules)

    def _get_policy_rule(self,
                         feature_idxs: Set[int],
                         rule: Any,
                         unknowns: Set[int],
                         dont_cares: Set[int],
                         policy_builder: Any,
                         feature_pool: List[Any]) -> Any:
        conditions: Set[Any] = set()
        for condition in rule.get_conditions():
            f_idx = int(condition.get_named_element().get_key()[1:])
            if (f_idx not in feature_idxs) or (f_idx in dont_cares): continue
            feature = feature_pool[f_idx].dlplan_feature
            if isinstance(condition, dlplan_policy.PositiveBooleanCondition):
                conditions.add(policy_builder.make_pos_condition(policy_builder.make_boolean(f"f{f_idx}", feature)))
            elif isinstance(condition, dlplan_policy.NegativeBooleanCondition):
                conditions.add(policy_builder.make_neg_condition(policy_builder.make_boolean(f"f{f_idx}", feature)))
            elif isinstance(condition, dlplan_policy.GreaterNumericalCondition):
                conditions.add(policy_builder.make_gt_condition(policy_builder.make_numerical(f"f{f_idx}", feature)))
            elif isinstance(condition, dlplan_policy.EqualNumericalCondition):
                conditions.add(policy_builder.make_eq_condition(policy_builder.make_numerical(f"f{f_idx}", feature)))
            else:
                raise RuntimeError(f"Cannot parse condition {str(condition)}")

        effects: Set[Any] = set()
        for effect in rule.get_effects():
            f_idx = int(effect.get_named_element().get_key()[1:])
            if (f_idx not in feature_idxs) or (f_idx in unknowns): continue
            feature = feature_pool[f_idx].dlplan_feature
            if isinstance(effect, dlplan_policy.PositiveBooleanEffect):
                effects.add(policy_builder.make_pos_effect(policy_builder.make_boolean(f"f{f_idx}", feature)))
            elif isinstance(effect, dlplan_policy.NegativeBooleanEffect):
                effects.add(policy_builder.make_neg_effect(policy_builder.make_boolean(f"f{f_idx}", feature)))
            elif isinstance(effect, dlplan_policy.UnchangedBooleanEffect):
                effects.add(policy_builder.make_bot_effect(policy_builder.make_boolean(f"f{f_idx}", feature)))
            elif isinstance(effect, dlplan_policy.IncrementNumericalEffect):
                effects.add(policy_builder.make_inc_effect(policy_builder.make_numerical(f"f{f_idx}", feature)))
            elif isinstance(effect, dlplan_policy.DecrementNumericalEffect):
                effects.add(policy_builder.make_dec_effect(policy_builder.make_numerical(f"f{f_idx}", feature)))
            elif isinstance(effect, dlplan_policy.UnchangedNumericalEffect):
                effects.add(policy_builder.make_bot_effect(policy_builder.make_numerical(f"f{f_idx}", feature)))
            else:
                raise RuntimeError(f"Cannot parse effect {str(effect)}")

        return policy_builder.make_rule(conditions, effects)


def _print_symbol(symbol):
    #symbols = ["select", "good", "bad", "d2_separate"]
    symbols = ["bad"]
    if symbol.name in symbols:
        logging.info(colored(f"ASP: Symbol: {symbol}", "red"))

