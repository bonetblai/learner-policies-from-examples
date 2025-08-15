import logging
import tempfile
from typing import List, Tuple, Dict, Union, Optional, Any
from pathlib import Path

from tarski.syntax.formulas import Atom as _TarskiAtom
from tarski.fstrips.problem import Problem as _TarskiInstance
from tarski.fstrips.action import Action as _TarskiAction
from tarski.fstrips.action import PlainOperator as _TarskiOperator
from tarski.model import Model as _TarskiState
from tarski.io import PDDLReader as _TarskiPDDLReader

from tarski.syntax.predicate import Predicate
from tarski.fstrips import fstrips

from tarski.grounding import LPGroundingStrategy
from tarski.grounding.lp_grounding import ground_problem_schemas_into_plain_operators
from tarski.syntax.transform.action_grounding import ground_schema_into_plain_operator_from_grounding
from tarski.search.model import progress
from tarski.search import GroundForwardSearchModel, BreadthFirstSearch

import dlplan.core as dlplan_core
from dlplan.core import VocabularyInfo, SyntacticElementFactory
from dlplan.policy import PolicyFactory

from ..util import change_dir, Timer
from ..preprocessing import DomainData
from .planner import get_plan, get_plan_v2


def parse_pddl_file(pddl_filepath: Path) -> Tuple[Any]:
    # Get a single string from file discarding PDDL comments
    parsed_lines: List[str] = []
    with pddl_filepath.open("r") as fd:
        for line in [line.strip(" ").strip(" \n").strip(" ") for line in fd.readlines()]:
            index = line.find(";")
            parsed_line = line if index == -1 else line[:index]
            if len(parsed_line) > 0:
                parsed_lines.append(parsed_line)
    collated: str = " ".join(parsed_lines)

    # Split single string into PDDL items, following parentization
    item_stack: List[Any] = []
    result: List[Any] = None
    token = None
    for i, c in enumerate(collated):
        logging.debug(f"i={i}, c={c}, token=|{token}|, stack={item_stack}, rest=|{collated[i:]}|")
        if c == "(":
            item_stack.append([])
            token = ""
        elif c == ")":
            assert len(item_stack) > 0
            if len(token) > 0: item_stack[-1].append(token)
            item_closed = tuple(item_stack.pop())
            if len(item_stack) > 0:
                item_stack[-1].append(item_closed)
            else:
                result = item_closed
            token = ""
        elif c == " ":
            if len(token) > 0: item_stack[-1].append(token)
            token = ""
        else:
            token += c
    assert len(item_stack) == 0
    return result

def str_from_parsed_pddl(parsed_pddl: Any) -> str:
    if type(parsed_pddl) == str:
        return parsed_pddl
    else:
        assert type(parsed_pddl) == tuple
        return "(" + " ".join([str_from_parsed_pddl(item) for item in parsed_pddl]) + ")"

class PDDLVocabulary:
    def __init__(self, domain_filepath: Path, instance_filepath: Path, using_tarski: bool = True):
        self._vocabulary_info: dlplan_core.VocabularyInfo = dlplan_core.VocabularyInfo()
        if using_tarski:
            reader: _TarskiPDDLReader = _TarskiPDDLReader(raise_on_error=True, case_insensitive=True)
            reader.parse_domain(str(domain_filepath))
            instance: _TarskiInstance = reader.parse_instance(str(instance_filepath))

            # Analyze schemas to calculate static predicates
            self._static_symbols: Set[Any] = set(instance.language.predicates)
            for schema_name in instance.actions:
                schema = instance.get_action(schema_name)
                for effect in schema.effects:
                    for predicate in self._get_affected_predicates(effect):
                        if predicate in self._static_symbols:
                            self._static_symbols.remove(predicate)
            self._fluent_symbols: Set[Any] = set(instance.language.predicates) - self._static_symbols

            # Analyze domain to calculate constants
            self._constants: List[Any] = [instance.language.get_constant(obj) for obj in  self._get_constants_from_pddl_domain(domain_filepath)]

            # Create vocabulary_info
            for predicate in instance.language.predicates:
                predicate_name: str = str(predicate.name)
                if predicate_name.endswith("_u") or predicate_name.endswith("_g"):
                    logging.error(f"ERROR: Predicate names ending with '_u' or '_g' are not supported because these suffixes are internally used; predicate={predicate_name}")
                    raise RuntimeError(f"Predicate names ending with '_u' or '_g' are not supported because these suffixes are internally used; predicate={predicate_name}")
                elif predicate_name not in ["=", "!=", "<", "<=", ">", ">="]:
                    is_static: bool = predicate in self._static_symbols
                    self._vocabulary_info.add_predicate(predicate_name, predicate.arity, is_static)
                    self._vocabulary_info.add_predicate(predicate_name + "_g", predicate.arity, is_static)

            # Add unary predicates for sorts to vocabulary
            for sort in instance.language.sorts:
                if str(sort.name) != "object":
                    self._vocabulary_info.add_predicate(str(sort.name) + "_u", 1, True)

            # Add constants to vocabulary
            for constant in self._constants:
                self._vocabulary_info.add_constant(str(constant.name))

        else:
            assert False, f"Non-tarski PDDLVocabulary for not yet implemented!"

    def _get_affected_predicates(self, effect: Any) -> List[Predicate]:
        if isinstance(effect, fstrips.AddEffect):
            return [effect.atom.predicate]
        elif isinstance(effect, fstrips.DelEffect):
            return [effect.atom.predicate]
        elif isinstance(effect, fstrips.UniversalEffect):
            return [predicate for subeffect in effect.effects for predicate in self._get_affected_predicates(subeffect)]
        else:
            logging.error(f"ERROR: Unexpected effect '{effect}'")
            raise RuntimeError(f"Unexpected effect '{effect}'")

    def _get_constants_from_pddl_domain(self, domain_filepath: Path) -> List[str]:
        # Get constants record in PDDL domain (if any)
        parsed_domain: Tuple[Any] = parse_pddl_file(domain_filepath)
        constants_record: List[str] = []
        for item in parsed_domain:
            if type(item) == tuple and item[0] == ":constants":
                constants_record = item[1:]

        # Remove type information (if any)
        remove_next_token = False
        constants: List[str] = []
        for token in constants_record:
            if token == "-":
                remove_next_token = True
            elif not remove_next_token:
                constants.append(token)
            else:
                remove_next_token = False
        return constants

    def static_symbols(self) -> List[Any]:
        return self._static_symbols

    def fluent_symbols(self) -> List[Any]:
        return self._fluent_symbols

    def constants(self) -> List[Any]:
        return self._constants

    def vocabulary_info(self) -> dlplan_core.VocabularyInfo:
        return self._vocabulary_info


class PDDLDomain:
    def __init__(self, domain_filepath: Path, vocabulary: PDDLVocabulary, using_tarski: bool = True):
        self._domain_filepath = domain_filepath
        self._vocabulary: PDDLVocabulary = vocabulary
        self._domain_data: DomainData = PDDLDomain.create_domain_data(self._domain_filepath, self._vocabulary.vocabulary_info())

    @classmethod
    def create_domain_data(cls, domain_filepath: Path, vocabulary_info: dlplan_core.VocabularyInfo) -> DomainData:
        syntactic_element_factory: SyntacticElementFactory = SyntacticElementFactory(vocabulary_info)
        policy_builder: PolicyFactory = PolicyFactory(syntactic_element_factory)
        return DomainData(str(domain_filepath), vocabulary_info, policy_builder, syntactic_element_factory)

    def domain_filepath(self) -> Path:
        return self._domain_filepath

    def vocabulary(self) -> PDDLVocabulary:
        return self._vocabulary

    def domain_data(self) -> DomainData:
        return self._domain_data


class MimirInstance:
    def __init__(self, domain: PDDLDomain, instance_filepath: Path):
        # Must do something with domain and instance file
        #assert False, "FIX: CALL Mimir API"
        pass

    def create_operator_from_action_str(self, action: str) -> Any:
        assert False, "FIX: CALL Mimir API: create_operator_from_action_str"

    def get_dlplan_state(self, state_idx: int, state: Any) -> dlplan_core.State:
        assert False, "FIX: CALL Mimir API: get_dlplan_state"

    def is_goal_state(self, state: Any) -> bool:
        assert False, "FIX: CALL Mimir API: is_goal_state"

    def is_deadend_state(self, state: Any) -> bool:
        assert False, "FIX: CALL Mimir API: is_deadend_state"

    def get_initial_state(self) -> Any:
        assert False, "FIX: CALL Mimir API: get_initial_state"

    def get_applicable_actions(self, state: Any) -> List[Any]:
        assert False, "FIX: CALL Mimir API: get_applicable_actions"

    def get_next_state(self, state: Any, action: str) -> Any:
        assert False, "FIX: CALL Mimir API: get_next_state"


class TarskiInstance:
    def __init__(self, domain: PDDLDomain, instance_idx: int, instance_filepath: Path, deadends: bool, planner: str):
        logging.info(instance_filepath)
        self._domain: PDDLDomain = domain
        self._instance_idx: int = instance_idx
        self._instance_filepath: Path = instance_filepath
        self._deadends: bool = deadends
        self._planner: str = planner

        self._initialized: bool = False
        self._initialized_final: bool = False

        self._grounded_model: Any = None
        self._grounding: LPGroundingStrategy = None

        self._reader = None
        self._instance = None
        self._static_atoms = None
        self._fluent_atoms = None

        self._instance_info: dlplan_core.InstanceInfo = None
        self._fluent_str_to_dlplan_atom: Dict[str, dlplan_core.Atom] = None

    def _initialize(self):
        if not self._initialized:
            self._initialized = True

            logging.info(f"Partial initialization of instance {self._instance_idx}.[{self._instance_filepath}]")
            self._reader: _TarskiPDDLReader = _TarskiPDDLReader(raise_on_error=True, case_insensitive=True)
            self._reader.parse_domain(str(self._domain.domain_filepath()))
            self._instance: _TarskiInstance = self._reader.parse_instance(str(self._instance_filepath))

            # Add lowercase keys to action dict
            self._actions: Dict[str, Any] = dict(self._instance.actions)
            self._actions.update({key.lower(): value for key, value in self._actions.items()})

            # Get different items from instance
            language = self._instance.language
            sorts = language.sorts
            objects = language.constants()
            constants = self._domain.vocabulary().constants()
            predicates = language.predicates
            goals = [self._instance.goal] if type(self._instance.goal) == _TarskiAtom else self._instance.goal.subformulas
            logging.debug(f"         SORTs: {sorts}")
            logging.debug(f"       OBJECTs: {objects}")
            logging.debug(f"     CONSTANTs: {constants}")
            logging.debug(f"    PREDICATEs: {predicates}")
            logging.debug(f"         GOALs: {goals}")

            # Extend language with unary predicates for sorts, and goal predicates (must be done before grounding)
            object_sort = language.get_sort("object")
            for sort in sorts:
                if str(sort.name) != "object":
                    signature_for_unary_predicate_for_sort = [str(sort.name) + "_u", object_sort]
                    language.predicate(*signature_for_unary_predicate_for_sort)

            for predicate in predicates:
                if str(predicate.name) not in ["=", "!="]:
                    signature_for_goal_predicate = [str(predicate.name) + "_g"] + [language.get_sort(arg) for arg in predicate.signature[1:]]
                    language.predicate(*signature_for_goal_predicate)

            # Add unary atoms for sorts and goal atoms
            for obj in objects:
                if str(obj.sort.name) != "object":
                    unary_predicate = language.get_predicate(str(obj.sort.name) + "_u")
                    self._instance.init.add(unary_predicate(obj))
            for goal in goals:
                predicate = goal.predicate
                goal_predicate = language.get_predicate(str(predicate.name) + "_g")
                self._instance.init.add(goal_predicate(*goal.subterms))
            logging.debug(f"    (NEW) INIT: {self._instance.init}")

            # Compute LP grounding
            self._grounding: LPGroundingStrategy = LPGroundingStrategy(self._instance, ground_actions=True)
            self._schema_bindings: Dict[str, List[Any]] = self._grounding.ground_actions()
            self._operators: List[Any] = [ground_schema_into_plain_operator_from_grounding(self._instance.actions.get(schema), binding) for schema, bindings in self._schema_bindings.items() for binding in bindings]
            logging.debug(f"STATIC SYMBOLS: {self._grounding.static_symbols}")
            logging.debug(f"FLUENT SYMBOLS: {self._grounding.fluent_symbols}")
            logging.debug(f"OPERATORS: {len(self._operators)}")

            # Ground fluents
            self._fluent_atoms = self._grounding.ground_state_variables()
            logging.debug(f"       FLUENTs: {self._fluent_atoms}")

            # Get static atoms from initial state
            #print(self._instance.init.as_atoms())
            self._static_atoms = [atom for atom in self._instance.init.as_atoms() if atom.predicate in self._grounding.static_symbols]
            logging.debug(f"  STATIC ATOMs: {self._static_atoms}")

    def _initialize_final(self):
        if not self._initialized:
            raise RuntimeError("_initialize_final() must be called after _initialize()")

        if not self._initialized_final:
            self._initialized_final = True

            logging.info(f"Final initialization of instance {self._instance_idx}.[{self._instance_filepath}]")

            # Create instance info and dlplan_atom map
            self._instance_info: dlplan_core.InstanceInfo = dlplan_core.InstanceInfo(self._instance_idx, self._domain.domain_data()._vocabulary_info)
            self._fluent_str_to_dlplan_atom: Dict[str, dlplan_core.Atom] = dict()
            for atom in self._static_atoms:
                assert str(atom.predicate.name) not in ["=", "!="]
                static_atom = self._instance_info.add_static_atom(str(atom.predicate.name), [str(obj) for obj in atom.subterms])
            for state_variable in self._fluent_atoms:
                atom = state_variable.to_atom()
                assert str(atom.predicate.name) not in ["=", "!="]
                fluent_atom = self._instance_info.add_atom(str(atom.predicate.name), [str(obj) for obj in atom.subterms])
                self._fluent_str_to_dlplan_atom[str(state_variable.to_atom())] = fluent_atom
            logging.debug(f"Instance info: {self._instance_info}")
            logging.debug(f"    FLUENT MAP: {self._fluent_str_to_dlplan_atom}")

            # Report some stats

    def _is_static(self, atom: Any) -> bool:
        return atom.predicate in self._grounding.static_symbols

    def _create_new_parsed_instance_from_state(self, state: _TarskiState) -> Tuple[Any]:
        # Create new initi spec by skipping _g and _u predicates
        new_init_spec: List[Any] = [":init"]
        for atom in state.as_atoms():
            if not str(atom.predicate.name).endswith("_g") and not str(atom.predicate.name).endswith("_u"):
                new_init_spec.append(tuple([str(atom.predicate.name)] + [str(term) for term in atom.subterms]))
        new_init_spec: Tuple[Any] = tuple(new_init_spec)
        #print(f"new_init_state: {new_init_spec}")

        # Parse instance file and create new parsed instance
        parsed_instance: Tuple[Any] = parse_pddl_file(self._instance_filepath)
        #print(f"parsed_instance: {parsed_instance}")
        new_parsed_instance: Tuple[Any] = tuple([item if type(item) != tuple or item[0] != ":init" else new_init_spec for item in parsed_instance])
        #print(f"new_parsed_instance: {new_parsed_instance}")
        return new_parsed_instance

    def re_index(self, new_instance_idx: int) -> bool:
        if self._initialized_final:
            return False
        else:
            assert self._instance_info is None and self._fluent_str_to_dlplan_atom is None
            self._instance_idx = new_instance_idx
            return True

    def initialized(self) -> bool:
        return self._initialized

    def domain_filepath(self) -> Path:
        return self._domain.domain_filepath()

    def instance_filepath(self) -> Path:
        return self._instance_filepath

    def instance_info(self) -> dlplan_core.InstanceInfo:
        return self._instance_info

    def create_operator_from_action_str(self, action: str) -> _TarskiOperator:
        items: List[str] = action.strip("()").split(" ")
        schema: _TarskiAction = self._actions.get(items[0])
        assert schema is not None, f"action={action}, items={items}"
        arguments: Tuple[str] = tuple(items[1:])
        operator: _TarskiOperator = ground_schema_into_plain_operator_from_grounding(schema, arguments)
        return operator

    def get_dlplan_state(self, state_idx: int, state: Any) -> dlplan_core.State:
        self._initialize_final()
        dlplan_state_atoms: List[dlplan_core.Atom] = [self._fluent_str_to_dlplan_atom.get(str(atom)) for atom in state.as_atoms() if not self._is_static(atom)]
        dlplan_state: dlplan_core.State = dlplan_core.State(state_idx, self._instance_info, dlplan_state_atoms)
        return dlplan_state

    def get_plan_for_state(self, state: _TarskiState, instance_filename: Optional[str] = None) -> Tuple[bool, List[str]]:
        # Need to create PDDL instance with initial state given by state and then call planner.
        # State is declared as deadend iff planner fails.
        new_parsed_instance: Tuple[Any] = self._create_new_parsed_instance_from_state(state)
        with change_dir("instances_for_get_plan", enable=True):
            assert instance_filename is not None
            instance_filepath: Path = Path(instance_filename)
            plan_filepath: Path = Path(f"{instance_filename}.plan.{self._planner}")

            with instance_filepath.open("w") as fp:
                fp.write(str_from_parsed_pddl(new_parsed_instance))

            status, plan = get_plan_v2(self.domain_filepath(), instance_filepath, plan_filepath, self._planner)
            if len(plan) == 0 and not self.is_goal_state(state):
                status = False

            """
            status, plan = get_plan(self.domain_filepath(), instance_filepath, plan_filepath, self._planner)
            # UNCHECKED ASSUMPTION: Plan indeed leads to a goal state
            if len(plan) == 0 and not self.is_goal_state(state):
                status = False
            """

        return status, plan

    def is_goal_state(self, state: _TarskiState) -> bool:
        return state[self._instance.goal]

    def is_deadend_state(self, state: _TarskiState, instance_filename: Optional[str] = None) -> bool:
        if self._deadends:
            status, plan = self.get_plan_for_state(state, instance_filename)
            return not status
        else:
            return False

    def get_initial_state(self) -> _TarskiState:
        self._initialize()
        return self._instance.init

    def get_applicable_operators(self, state: _TarskiState) -> List[Any]:
        #print(f"OPERATORS:  {len(self._operators)}")
        #print(f"APPLICABLE: {len([operator for operator in self._operators if state[operator.precondition]])}")
        return [operator for operator in self._operators if state[operator.precondition]]

    def get_next_state_with_operator(self, state: _TarskiState, operator: Any) -> _TarskiState:
        assert state[operator.precondition], f"Error: operator '{operator}' isn't applicable at {state.as_atoms()}"
        next_state: _TarskiState = progress(state, operator)
        return next_state


class PDDLInstance:
    def __init__(self, domain: PDDLDomain, instance_idx: int, instance_filepath: Path, planner: str, deadends: bool, using_tarski: bool = True):
        self.idx: int = instance_idx
        self._domain: PDDLDdomain = domain
        self._instance_filepath: Path = instance_filepath
        self._planner: str = planner
        self._deadends: bool = deadends
        self._using_tarski: bool = using_tarski

        if self._using_tarski:
            self._instance: TarskiInstance = TarskiInstance(self._domain, instance_idx, instance_filepath, self._deadends, self._planner)
        else:
            self._instance: MimirInstance = MimirInstance(self._domain, instance_idx, instance_filepath)

        # State repositories
        self._state_to_state_idx: Dict[Any, int] = dict()
        self._state_idx_to_state: List[Any] = []
        self._state_idx_to_dlplan_state: List[dlplan_core.State] = []

        # Successor and deadend caching
        self._state_idx_to_successors: List[Tuple[int, str]] = []
        self._state_idx_to_deadend_value: List[bool] = []

    def re_index(self, new_instance_idx: int) -> bool:
        if self._instance.re_index(new_instance_idx):
            self.idx = new_instance_idx
            return True
        else:
            return False

    def domain_filepath(self) -> Path:
        return self._domain.domain_filepath()

    def instance_filepath(self) -> Path:
        return self._instance.instance_filepath()

    def vocabulary_info(self) -> dlplan_core.VocabularyInfo:
        return self._domain.vocabulary_info()

    def instance_info(self) -> dlplan_core.InstanceInfo:
        return self._instance._instance_info

    def create_operator_from_action_str(self, action: str) -> Any:
        return self._instance.create_operator_from_action_str(action)

    def get_state_index(self, state: Any) -> Tuple[int, Any]:
        state_idx: int = self._state_to_state_idx.get(state)
        if state_idx is None:
            assert len(self._state_idx_to_state) == len(self._state_idx_to_dlplan_state) == len(self._state_idx_to_deadend_value) == len(self._state_idx_to_successors)
            state_idx: int = len(self._state_to_state_idx)
            self._state_to_state_idx[state] = state_idx
            self._state_idx_to_state.append(state)
            dlplan_state: dlplan_core.State = self._instance.get_dlplan_state(state_idx, state)
            self._state_idx_to_dlplan_state.append(dlplan_state)
            self._state_idx_to_successors.append(None)
            self._state_idx_to_deadend_value.append(None)
            logging.debug(f"STATE-CREATION (instance_idx may change if this was before reordering of instances): instance_idx={self.idx} -> {state_idx}.{dlplan_state}")
        return state_idx, state

    def get_state(self, state_idx: int) -> Tuple[int, Any]:
        return state_idx, self._state_idx_to_state[state_idx]

    def get_dlplan_state(self, state_idx: int, state: Any) -> dlplan_core.State:
        #return self._instance.get_dlplan_state(state_idx, state)
        return self._state_idx_to_dlplan_state[state_idx]

    def get_plan_for_state(self, state: Any, instance_filename: Optional[str] = None) -> Tuple[bool, List[str]]:
        return self._instance.get_plan_for_state(state, instance_filename)

    def is_goal_state(self, state_idx: int) -> bool:
        _, state = self.get_state(state_idx)
        return self._instance.is_goal_state(state)

    def is_deadend_state(self, state_idx: int) -> bool:
        is_deadend: bool = self._state_idx_to_deadend_value[state_idx]
        if is_deadend is None:
            instance_filename: str = f"problem_for_state_{state_idx}_in_instance_{self.idx}.pddl"
            _, state = self.get_state(state_idx)
            is_deadend: bool = self._instance.is_deadend_state(state, instance_filename)
            self.register_deadend_value(state_idx, is_deadend)
        return is_deadend

    def register_deadend_value(self, state_idx: int, value: bool):
        is_deadend: bool = self._state_idx_to_deadend_value[state_idx]
        assert is_deadend is None or (not is_deadend or value) or (is_deadend or not value)
        if is_deadend is None:
            self._state_idx_to_deadend_value[state_idx] = value

    def get_initial_state(self) -> Tuple[int, Any]:
        state: Any = self._instance.get_initial_state()
        return self.get_state_index(state)

    def get_applicable_actions(self, state_idx: int) -> List[str]:
        _, state = self.get_state(state_idx)
        return self._instance.get_applicable_actions(state)

    def get_next_state(self, state_idx: int, action: str) -> Tuple[int, Any]:
        _, state = self.get_state(state_idx)
        next_state: Any = self._instance.get_next_state_with_operator(state, self.create_operator_from_action_str(action))
        return self.get_state_index(next_state)

    def get_successors(self, state_idx: int) -> List[Tuple[Tuple[int, Any], str]]:
        successors: List[Tuple[int, str]] = self._state_idx_to_successors[state_idx]
        if successors is None:
            _, state = self.get_state(state_idx)
            applicable_operators: List[Any] = self._instance.get_applicable_operators(state)
            successors: List[Tuple[int, str]] = [(self.get_state_index(self._instance.get_next_state_with_operator(state, operator))[0], str(operator)) for operator in applicable_operators]
            self._state_idx_to_successors[state_idx] = successors
        return [(self.get_state(succ_state_idx), operator) for succ_state_idx, operator in successors]

    def get_low_level_initial_state(self) -> Any:
        state: Any = self._instance.get_initial_state()
        return state

    def get_next_low_level_state(self, state: Any, action: str) -> Any:
        next_state: Any = self._instance.get_next_state_with_operator(state, self.create_operator_from_action_str(action))
        return next_state

    def get_low_level_state_trajectory_from_plan(self, plan: List[str]) -> List[Any]:
        state_trajectory: List[Any] = [self.get_low_level_initial_state()]
        for action in plan:
            current_state: Any = state_trajectory[-1]
            next_state: Any = self.get_next_low_level_state(current_state, action)
            state_trajectory.append(next_state)
        return state_trajectory

class StateFactory:
    def __init__(self, family_name: str, domain_filepath: Path, instance_filepaths: List[Path], planner: str, deadends: Optional[bool] = None, using_tarski: bool = True):
        self._family_name: str = family_name
        self._planner: str = planner
        self._deadends: bool = True if deadends is None else deadends
        self._using_tarski: bool = using_tarski

        # Create vocabulary with domain and one instance
        self._vocabulary: PDDLVocabulary = PDDLVocabulary(domain_filepath, instance_filepaths[0], using_tarski)

        # Create domain
        self._domain: PDDLDomain = PDDLDomain(domain_filepath, self._vocabulary, using_tarski)

        # Read PDDL instances
        self._instances: List[PDDLInstance] = []
        for instance_idx, instance_filepath in enumerate(instance_filepaths):
            self._instances.append(PDDLInstance(self._domain, instance_idx, instance_filepath, planner, self._deadends, using_tarski))

    def remove_and_reorder_instances(self, sorted_instance_idxs: List[int]):
        new_instances: List[PDDLInstance] = []
        for instance_idx in sorted_instance_idxs:
            new_instance_idx: int = len(new_instances)
            instance: PDDLInstance = self._instances[instance_idx]
            if instance.re_index(new_instance_idx):
                new_instances.append(instance)
                assert new_instances[new_instance_idx].idx == new_instance_idx
                assert new_instances[new_instance_idx]._instance._instance_idx == new_instance_idx
            else:
                raise RuntimeError("Cannot re-index after *finally* initialized PDDLInstance(s)")
        self._instances = new_instances

    def family_name(self) -> str:
        return self._family_name

    def using_tarski(self) -> bool:
        return self._using_tarski

    def domain_data(self) -> DomainData:
        return self._domain.domain_data()

    def get_instance(self, instance_idx: int) -> PDDLInstance:
        return self._instances[instance_idx]

    def get_state_index(self, instance_idx: int, state: Any) -> int:
        return self._instances[instance_idx].get_state_index(state)[0]

    def get_state(self, instance_idx: int, state_idx: int) -> Any:
        return self._instances[instance_idx].get_state(state_idx)[1]

    def get_dlplan_state(self, instance_idx: int, state_idx: int) -> dlplan_core.State:
        state: Any = self.get_state(instance_idx, state_idx)
        return self._instances[instance_idx].get_dlplan_state(state_idx, state)

    def get_plan_for_state_idx(self, instance_idx: int, state_idx: int) -> Tuple[bool, List[str]]:
        state: Any = self.get_state(instance_idx, state_idx)
        instance_filename: str = f"problem_for_state_{state_idx}_in_instance_{instance_idx}.pddl"
        return self._instances[instance_idx].get_plan_for_state(state, instance_filename)

    def get_plan_for_state(self, instance_idx: int, state: Any) -> Tuple[bool, List[str]]:
        assert False, "Use get_plan_for_state_idx()"

    def is_goal_state(self, instance_idx: int, state_idx: int) -> bool:
        return self._instances[instance_idx].is_goal_state(state_idx)

    def is_deadend_state(self, instance_idx: int, state_idx: int) -> bool:
        return self._instances[instance_idx].is_deadend_state(state_idx)

    def register_deadend_value(self, instance_idx: int, state_idx: int, value: bool):
        self._instances[instance_idx].register_deadend_value(state_idx, value)

    def get_initial_state(self, instance_idx: int) -> Tuple[int, Any]:
        return self._instances[instance_idx].get_initial_state()

    def get_applicable_actions(self, instance_idx: int, state_idx: int) -> List[str]:
        return self._instances[instance_idx].get_applicable_actions(state_idx)

    def get_next_state(self, instance_idx: int, state_idx: int, action: str) -> Tuple[int, Any]:
        return self._instances[instance_idx].get_next_state(state_idx, action)

    def get_state_trajectory_from_state_idx_and_plan(self, instance_idx: int, state_idx: int, plan: List[str]) -> List[Tuple[int, Any]]:
        state_trajectory: List[Tuple[int, Any]] = [self._instances[instance_idx].get_state(state_idx)]
        for action in plan:
            current_state: Tuple[int, Any] = state_trajectory[-1]
            next_state: Tuple[int, Any] = self.get_next_state(instance_idx, current_state[0], action)
            state_trajectory.append(next_state)
        return state_trajectory

    def get_state_trajectory_from_plan(self, instance_idx: int, plan: List[str]) -> List[Tuple[int, Any]]:
        return self.get_state_trajectory_from_state_idx_and_plan(instance_idx, self.get_initial_state(instance_idx)[0], plan)

    def get_successors(self, instance_idx: int, state_idx: int) -> List[Tuple[Tuple[int, Any], str]]:
        return self._instances[instance_idx].get_successors(state_idx)

    def get_low_level_state_trajectory_from_plan(self, instance_idx: int, plan: List[str]) -> List[Any]:
        return self._instances[instance_idx].get_low_level_state_trajectory_from_plan(plan)

