import logging
from typing import Dict, Set, FrozenSet, List, Deque, MutableSet, Tuple, Any, Optional


class FeatureDecoder:
    def __init__(self):
        pass

    def _parse(self, feature: str) -> List[Any]:
        # Split feature by top-level "commas"
        def split(f: str) -> List[Any]:
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
            elif f.startswith("r_compose"):
                assert f.endswith(")")
                tokens = split(f[10:-1])
                assert len(tokens) == 2
                assert len(vars) >= 2
                qvar = 1 + max(vars)
                jvars = [[vars[-2], qvar], [qvar, vars[-1]]]
                return ["r_compose", vars + [qvar]] + [parse(token, _vars, 1 + indent) for token, _vars in zip(tokens, jvars)]
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
            elif f.startswith("c_or"):
                assert f.endswith(")")
                tokens = split(f[5:-1])
                assert len(tokens) == 2, tokens
                return ["c_or", vars] + [parse(token, vars, 1 + indent) for token in tokens]
            elif f.startswith("c_equal"):
                assert f.endswith(")")
                tokens = split(f[8:-1])
                assert len(tokens) == 2
                qvar = 1 + max(vars)
                vars = vars + [qvar]
                return ["c_equal", vars] + [parse(token, vars, 1 + indent) for token in tokens]
            elif f.startswith("c_subset"):
                assert f.endswith(")")
                tokens = split(f[9:-1])
                assert len(tokens) == 2
                qvar = 1 + max(vars)
                vars = vars + [qvar]
                return ["c_subset", vars] + [parse(token, vars, 1 + indent) for token in tokens]
            elif f.startswith("c_some"):
                assert f.endswith(")")
                tokens = split(f[7:-1])
                assert len(tokens) == 2
                qvar = 1 + max(vars)
                return ["c_some", [vars[-1], qvar]] + [parse(token, vars + [qvar], 1 + indent) for token in tokens]
            elif f.startswith("c_all"):
                assert f.endswith(")")
                tokens = split(f[6:-1])
                assert len(tokens) == 2
                qvar = 1 + max(vars)
                return ["c_all", [vars[-1], qvar]] + [parse(token, vars + [qvar], 1 + indent) for token in tokens]
            elif f.startswith("c_"):
                raise RuntimeError(f"Unexpected concept |{f}|")

            # Unexpected
            else:
                raise RuntimeError(f"Unexpected |{f}|")

        return parse(feature)

    def decode(self, feature: str) -> str:
        parsed: List[Any] = self._parse(feature)
        #print(f"feature: {feature}")
        #print(f" parsed: {parsed}")
        def _decode(f: List[Any], formula: bool, indent: int = 0) -> str:
            #print(f"decode: {'  ' * indent}{f}")
            assert type(f) == list and len(f) > 0

            # Boolean features
            if f[0] == "b_empty":
                role_or_concept: str = _decode(f[2], formula=False, indent=1 + indent)
                return f"Empty({role_or_concept})"
            elif f[0] == "b_nullary":
                return f"{f[2]()}" if formula else f"<nullary({f[2]})>"
            elif f[0].startswith("b_"):
                logging.warning(f"Unexpected Boolean {f}")
                return f"<unexpected-boolean({f})>"

            # Numerical features
            elif f[0] == "n_count":
                role_or_concept: str = _decode(f[2], formula=False, indent=1 + indent)
                return f"Cardinality({role_or_concept})"
            elif f[0] == "n_concept_distance":
                concept1: str = _decode(f[2], formula=False, indent=1 + indent)
                role: str = _decode(f[3], formula=True, indent=1 + indent)
                concept2: str = _decode(f[4], formula=False, indent=1 + indent)
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
                role1: str = _decode(f[2], formula=True, indent=1 + indent)
                role2: str = _decode(f[3], formula=True, indent=1 + indent)
                vpair: str = "(" + ",".join(vars[-2:]) + ")"
                return f"[{role1} & {role2}]" if formula else f"{{{vpair} : {role1} & {role2}}}"
            elif f[0].startswith("r_transitive_closure"):
                vars: List[str] = [f"x{i}" for i in f[1]]
                role: str = _decode(f[2], formula=True, indent=1 + indent)
                vpair: str = "(" + ",".join(vars[-2:]) + ")"
                i = role.rfind(vpair)
                assert i != -1, f"role=|{role}|, vpair=|{vpair}|"
                naked_role: str = role[:i]
                tc_formula: str = f"TC[{naked_role}]{vpair}"
                return tc_formula if formula else f"{{{vpair} : {tc_formula}}}"
            elif f[0].startswith("r_restrict"):
                vars: List[str] = [f"x{i}" for i in f[1]]
                role: str = _decode(f[2], formula=True, indent=1 + indent)
                concept: str = _decode(f[3], formula=True, indent=1 + indent)
                vpair: str = "(" + ",".join(vars[-2:]) + ")"
                return f"[{role} & {concept}]" if formula else f"{{{vpair} : {role} & {concept}}}"
            elif f[0] == "r_compose":
                vars: List[str] = [f"x{i}" for i in f[1]]
                role1: str = _decode(f[2], formula=True, indent=1 + indent)
                role2: str = _decode(f[3], formula=True, indent=1 + indent)
                assert len(vars) >= 3
                qvar = vars[-1]
                vpair: str = "(" + ",".join(vars[-3:-1]) + ")"
                return f"Join({role1}, {role2}]" if formula else f"{{{vpair} : Exists {qvar}.[{role1} & {role2}]}}"
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
                concept: str = _decode(f[2], formula, indent=1 + indent)
                vpair: str = "(" + ",".join(vars) + ")"
                return f"-{concept}" if formula else f"\compl({concept})"
            elif f[0] == "c_and":
                concept1: str = _decode(f[2], formula, 1 + indent)
                concept2: str = _decode(f[3], formula, 1 + indent)
                return f"[{concept1} & {concept2}]" if formula else f"[{concept1} \cap {concept2}]"
            elif f[0] == "c_or":
                concept1: str = _decode(f[2], formula, 1 + indent)
                concept2: str = _decode(f[3], formula, 1 + indent)
                return f"[{concept1} v {concept2}]" if formula else f"[{concept1} \cup {concept2}]"
            elif f[0] == "c_equal":
                vars: List[str] = [f"x{i}" for i in f[1]]
                var, qvar = vars[0], vars[1]
                role1: str = _decode(f[2], formula=True, indent=1 + indent)
                role2: str = _decode(f[3], formula=True, indent=1 + indent)
                if formula:
                    return f"[Forall {qvar}.[{role1} <=> {role2}]]({var})"
                else:
                    return f"{{{var} : {{{qvar} : {role1}}} = {{{qvar} : {role2}}}}}"
            elif f[0] == "c_subset":
                vars: List[str] = [f"x{i}" for i in f[1]]
                var, qvar = vars[0], vars[1]
                role1: str = _decode(f[2], formula=True, indent=1 + indent)
                role2: str = _decode(f[3], formula=True, indent=1 + indent)
                if formula:
                    return f"Subset({role1}, {role2})"
                else:
                    return f"{{{var} : Exists {qvar}.[{role1} & NOT {role2}]}}"
            elif f[0] == "c_some":
                vars: List[str] = [f"x{i}" for i in f[1]]
                var, qvar = vars[0], vars[1]
                role: str = _decode(f[2], formula=True, indent=1 + indent)
                concept: str = _decode(f[3], formula=True, indent=1 + indent)
                if formula:
                    return f"[Exist {qvar}.[{role} & {concept}]]({var})"
                else:
                    return f"{{{var} : Exists {qvar}.[{role} & {concept}]}}"
            elif f[0] == "c_all":
                vars: List[str] = [f"x{i}" for i in f[1]]
                var, qvar = vars[0], vars[1]
                role: str = _decode(f[2], formula=True, indent=1 + indent)
                concept: str = _decode(f[3], formula=True, indent=1 + indent)
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

        return _decode(parsed, formula=False)

    def __call__(self, feature: str) -> str:
        return self.decode(feature)

