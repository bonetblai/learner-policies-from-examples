import argparse
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

NOTYPE = "<notype>"

def strip_comment(line):
    post = ""
    for char in line:
        if char == ";":
            return post.strip(" ")
        else:
            post += char
    return post.strip(" ")

def strip_extra_blanks(line):
    post = ""
    tab = False
    for char in line:
        if char == "\t":
            post += " "
        elif char != " " or len(post) == 0 or post[-1] != " ":
            post += char
        tab = tab or char == "\t"
    return post if not tab else strip_extra_blanks(post)

def decompose(line):
    if line.find("(")  == -1:
        return line
    else:
        tokens = []
        token = ""
        level = -1
        for char in line:
            if char == "(":
                if level > -1:
                    token += char
                level += 1
            elif char == ")":
                if level > 0:
                    token += char
                level -= 1
            elif char == " " and level == 0:
                if token != "":
                    tokens.append(token)
                    token = ""
            else:
                token += char
        if token != "": tokens.append(token)
        return [decompose(token) for token in tokens]

def parse_objects(objs, debug=False):
    all_objects = []
    objects = []
    obj2type = dict()
    type2objs = defaultdict(list)
    read_obj_type = False

    for obj in objs:
        if obj == "-":
            read_obj_type = True
        elif not read_obj_type:
            objects.append(obj)
        else:
            obj_type = obj
            read_obj_type = False
            obj2type.update({obj: obj_type for obj in objects})
            type2objs[obj_type].extend(objects)
            all_objects.extend(objects)
            objects = []

    if len(objects) > 0:
        all_objects.extend(objects)
        if len(obj2type) > 0:
            obj2type.update({obj: "object" for obj in objects})
            type2objs["objects"].extend(objects)

    for obj in all_objects:
        if obj not in obj2type:
            obj2type[obj] = NOTYPE
            type2objs[NOTYPE].append(obj)

    if debug:
        print(f"all_objects: {all_objects}")
        print(f"   obj2type: {obj2type}")
        print(f"  type2objs: {dict(type2objs)}")

    return all_objects, obj2type, type2objs

def support_for_marks(objects):
    support = [["markable-0", obj] for obj in objects]
    return support

def add_support_for_marks(parsed):
    support = support_for_marks(parsed["objects"])
    patched = deepcopy(parsed)
    patched.update({"init": support + patched["init"]})
    return patched

def add_marking_actions(parsed, max_rank=0):
    predicates = deepcopy(parsed["predicates"])
    predicates += [[f"markable-{k}", "?x"] for k in range(1 + max_rank)]
    predicates += [[f"mark-{k}", "?x"] for k in range(1 + max_rank)] + [["mark", "?x"]]
    predicates += [[f"some-mark-{k}"] for k in range(1 + max_rank)]

    actions = deepcopy(parsed["actions"])

    # Add first-mark actions
    for k in range(1 + max_rank):
        name = f"mark-{k}"
        parameters = ["?x"]
        precondition = [[f"markable-{k}", "?x"], ["not", ["mark", "?x"]], ["not", [f"some-mark-{k}"]]]
        precondition += [[f"some-mark-{k-1}"]] if k > 0 else []
        effect = [[f"mark-{k}", "?x"], ["not", [f"markable-{k}", "?x"]], ["mark", "?x"], [f"some-mark-{k}"]]
        for j in range(1 + k, 1 + max_rank):
            effect += [["forall", ["?z"], ["and", [f"markable-{j}", "?z"], ["when", [f"mark-{j}", "?z"], ["and", ["not", [f"mark-{j}", "?z"]], ["not", ["mark", "?z"]]]]]]]
            effect += [["not", [f"some-mark-{j}"]]]
        action = [name, ":parameters", parameters, ":precondition", ["and"] + precondition, ":effect", ["and"] + effect]
        actions.append(action)

    # Add move-mark actions
    for k in range(1 + max_rank):
        name = f"move-mark-{k}"
        parameters = ["?x", "?y"]
        precondition = [[f"mark-{k}", "?x"], [f"markable-{k}", "?y"], ["not", ["mark", "?y"]]]
        #print(precondition)
        effect = [["not", [f"mark-{k}", "?x"]], ["not", ["mark", "?x"]], [f"mark-{k}", "?y"], ["not", [f"markable-{k}", "?y"]], ["mark", "?y"]]
        for j in range(1 + k, 1 + max_rank):
            effect += [["forall", ["?z"], ["and", [f"markable-{j}", "?z"], ["when", [f"mark-{j}", "?z"], ["and", ["not", [f"mark-{j}", "?z"]], ["not", ["mark", "?z"]]]]]]]
            effect += [["not", [f"some-mark-{j}"]]]
        #print(effect)
        action = [name, ":parameters", parameters, ":precondition", ["and"] + precondition, ":effect", ["and"] + effect]
        actions.append(action)

    patched = deepcopy(parsed)
    patched.update({"predicates": predicates, "actions": actions})
    return patched

def item_to_str(item):
    if item[0] == "and":
        return "(and " + " ".join(items_to_list_str(item[1:])) + ")"
    elif item[0] == "not":
        return "(not " + item_to_str(item[1]) + ")"
    elif item[0] == "forall":
        return "(forall (" + " ".join(items_to_list_str(item[1])) + ") " + item_to_str(item[2]) + ")"
    elif item[0] == "when":
        return "(when " + item_to_str(item[1]) + " " + item_to_str(item[2]) + ")"
    elif item[0][0] == "?":
        return item
    else:
        return "(" + " ".join(item) + ")"

def items_to_list_str(items):
    return [item_to_str(item) for item in items]

def objects_with_types(type2objs):
    objs_with_type = [" ".join(objects) + f" - {obj_type}" for obj_type, objects in type2objs.items() if obj_type != NOTYPE]
    objs_with_type.extend(type2objs.get(NOTYPE, []))
    return " ".join(objs_with_type)

def write_domain(parsed):
    domain = parsed["domain"]
    requirements = parsed["requirements"]
    types = parsed["types"]
    objects = objects_with_types(parsed["type2objs"])
    predicates = parsed["predicates"]
    actions = parsed["actions"]

    print(f"(define (domain {domain})")
    if len(requirements) > 0:
        print(f"    (:requirements {' '.join(requirements)})")
    if len(types) > 0:
        assert False
    if len(objects) > 0:
        print(f"    (:constants {objects})")
    print(f"    (:predicates")
    for atom in items_to_list_str(predicates):
        print(f"        {atom}")
    print(f"    )")
    for action in actions:
        name = action[0]
        parameters = action[2]
        precondition = action[4]
        if len(precondition) == 0:
            precondition = []
        elif precondition[0] != "and":
            precondition = [item_to_str(precondition)]
        else:
            precondition = items_to_list_str(precondition[1:])
        effect = action[6]
        if len(effect) == 0:
            effect = []
        elif effect[0] != "and":
            effect = [item_to_str(effect)]
        else:
            effect = items_to_list_str(effect[1:])

        print(f"    (:action {name}")
        print(f"        :parameters ({' '.join(parameters)})")
        print(f"        :precondition (and {' '.join(precondition)})")
        print(f"        :effect (and {' '.join(effect)})")
        print(f"    )")
    print(")")

def write_problem(parsed):
    problem = parsed["problem"]
    domain = parsed["domain"]
    objects = objects_with_types(parsed["type2objs"])
    init_atoms = items_to_list_str(parsed["init"])
    goal = parsed["goal"]
    goal_items = goal if goal[0] != "and" else goal[1:]
    goal_atoms = items_to_list_str(goal_items)

    print(f"(define (problem {problem})")
    print(f"    (:domain {domain})")
    print(f"    (:objects {objects})")
    print(f"    (:init")
    for atom in init_atoms:
        print(f"        {atom}")
    print(f"    )")
    print(f"    (:goal (and {' '.join(goal_atoms)}))")
    print(")")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse (and possibly patch) PDDL domain.")
    parser.add_argument("--add_marking_actions", type=int, default=None, help="Add support for marks for given max_rank. Default is None.")
    parser.add_argument("--dump_pddl", action="store_true", default=False, help="Dump (possibly patched) PDDL file. Default is False.")
    parser.add_argument("--debug", action="store_true", default=False, help="Turn on debugging. Default is False.")
    parser.add_argument("pddl_file", nargs=1, type=Path, help="Path to PDDL file.")
    args = parser.parse_args()

    # Read PDDL file
    with args.pddl_file[0].open("r") as fd:
        lines = [strip_comment(line.strip("\n").strip(" ")) for line in fd.readlines() if line.lstrip(" ")[0] != ";"]
        lines = [strip_extra_blanks(line) for line in lines if line != ""]
        lines = [line for line in lines if line != ""]
        if args.debug: print(f"lines: {lines}")
        single = " ".join(lines)
    if args.debug: print(f"single: |{single}|")

    # Decompose file
    decomposed_file = decompose(single)
    if args.debug: print(f"decomposed: {decomposed_file}")
    assert decomposed_file[0] == "define" and decomposed_file[1][0] == "domain"

    domain = decomposed_file[1][1]
    if args.debug: print(f"domain: |{domain}|")

    requirements = [struct[1:] for struct in decomposed_file if struct[0] == ":requirements"]
    requirements = requirements[0] if len(requirements) > 0 else []
    if args.debug: print(f"requirements: {requirements}")

    types = [struct[1:] for struct in decomposed_file if struct[0] == ":types"]
    types = types[0] if len(types) > 0 else []
    if args.debug: print(f"types: {types}")

    constants = [struct[1:] for struct in decomposed_file if struct[0] == ":constants"]
    constants = constants[0] if len(constants) > 0 else []
    objects, obj2type, type2objs = parse_objects(constants, args.debug)
    if args.debug: print(f"constants: {constants}")

    predicates = [struct[1:] for struct in decomposed_file if struct[0] == ":predicates"]
    predicates = predicates[0] if len(predicates) > 0 else []
    if args.debug: print(f"predicates: {predicates}")

    actions = [struct[1:] for struct in decomposed_file if struct[0] == ":action"]
    if args.debug: print(f"actions: {actions}")

    parsed = {
        "domain": domain,
        "requirements": requirements,
        "types": types,
        "constants": constants,
        "obj2type": obj2type,
        "type2objs": type2objs,
        "predicates": predicates,
        "actions": actions,
    }
    if args.debug: print(parsed)

    if args.add_marking_actions is not None:
        patched = add_marking_actions(parsed, args.add_marking_actions)
        if args.dump_pddl:
            write_domain(patched)
    elif args.dump_pddl:
        write_domain(parsed)

