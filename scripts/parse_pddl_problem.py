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
        if char != " " or len(post) == 0 or post[-1] != " ":
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
    parser = argparse.ArgumentParser(description="Parse (and possibly patch) PDDL problem.")
    parser.add_argument("--add_support_for_marks", action="store_true", default=False, help="Add support for marks. Default is False.")
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
    assert decomposed_file[0] == "define" and decomposed_file[1][0] == "problem" and decomposed_file[2][0] == ":domain"

    problem = decomposed_file[1][1]
    domain = decomposed_file[2][1]
    if args.debug:
        print(f"problem: |{problem}|")
        print(f"domain: |{domain}|")

    objects = [struct[1:] for struct in decomposed_file if struct[0] == ":objects"]
    objects = objects[0] if len(objects) > 0 else []
    objects, obj2type, type2objs = parse_objects(objects, args.debug)
    if args.debug: print(f"objects: {objects}")

    init = [struct[1:] for struct in decomposed_file if struct[0] == ":init"]
    init = init[0] if len(init) > 0 else []
    if args.debug: print(f"init: {init}")

    goal = [struct[1:] for struct in decomposed_file if struct[0] == ":goal"]
    goal = goal[0][0] if len(goal) > 0 else []
    if args.debug: print(f"goal: {goal}")

    parsed = {
        "problem": problem,
        "domain": domain,
        "objects": objects,
        "obj2type": obj2type,
        "type2objs": type2objs,
        "init": init,
        "goal": goal,
    }

    if args.add_support_for_marks:
        patched = add_support_for_marks(parsed)
        if args.dump_pddl:
            write_problem(patched)
    elif args.dump_pddl:
        write_problem(parsed)

