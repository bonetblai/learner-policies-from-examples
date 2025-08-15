#; child-snack task with 1 children and 0.0 gluten factor 
#; constant factor of 1.0
#; random seed: 1
#
#(define (problem prob-snack)
#  (:domain child-snack)
#  (:objects
#    child1 - child
#    bread1 - bread-portion
#    content1 - content-portion
#    tray1 - tray
#    table1 table2 table3 - place
#    sandw1 - sandwich
#  )
#  (:init
#     (at tray1 kitchen)
#     (at_kitchen_bread bread1)
#     (at_kitchen_content content1)
#     (not_allergic_gluten child1)
#     (waiting child1 table1)
#     (notexist sandw1)
#  )
#  (:goal
#    (and
#     (served child1)
#    )
#  )
#)

import subprocess

from sys import argv
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set

if __name__ == "__main__":
    if len(argv) < 2:
        usage(argv[0])
        exit(-1)

    problem_filepath: Path = Path(argv[1])

    # Parse file into single line without comments
    with problem_filepath.open("r") as fd:
        lines: List[str] = []
        for line in fd.readlines():
            line = line.strip("\n").strip(" ")
            if line.startswith(";") or line == "":
                # This is a comment, pass it through
                #print(line)
                pass
            else:
                # Remove comment
                pos = line.find(";")
                line = line if pos == -1 else line[:pos]

                # Remove extra whitespace

                # Append line
                lines.append(line)


    parsed: str = " ".join(lines)
    assert parsed.find("  ") == -1

    # Get objects
    typed_objects: Dict[str, Set[str]] = defaultdict(set)
    objects_start = parsed.find(":objects")
    objects_end = parsed.find(")", objects_start)

    token: str = ""
    objs: Set[str] = set()
    parsing_type: bool = False

    i = objects_start + 9
    assert parsed[i-2] == "s" and parsed[i-1] == " " and parsed[i] != " "

    while i < objects_end:
        c = parsed[i]
        if c == " ":
            #print(f"i={i}, start={objects_start}, end={objects_end}, token=|{token}|, parsed=|{parsed[:i+1]}|")
            objs.add(token)
            token = ""
            i += 1
            assert parsed[i] != " "
        elif c == "-":
            assert parsed[i+1] == " " and parsed[i+2] != " "
            # Read type and insert objects into dict
            token_end = parsed.find(" ", i+2)
            type_name = parsed[i+2: token_end]
            typed_objects[type_name] |= objs
            objs = set()
            token = ""
            i = token_end + 1
            #print(f"i={i}, start={objects_start}, end={objects_end}, parsed=|{parsed[:i+1]}|")
        else:
            token += c
            i += 1

    new_unary_atoms: List[str] = [f"({type_name}_t {obj})" for type_name, objects in typed_objects.items() for obj in objects]

    init_start = parsed.find(":init")
    new_parsed: str = parsed[:objects_start + 9] + " ".join([obj for objects in typed_objects.values() for obj in objects]) + parsed[objects_end: init_start + 6]
    new_parsed += " ".join(new_unary_atoms) + " "
    new_parsed += parsed[init_start + 6:]

    # Pretty printing
    pos = 0
    while pos != -1:
        segment_start = new_parsed.find("(:", pos + 1)
        segment = new_parsed[pos: segment_start]
        #print(f"pos={pos}, segment_start={segment_start}, segment=|{segment}|")
        if segment.startswith("(:"):
            print(f"    {segment}")
        else:
            print(segment)
        pos = segment_start
    print(")")

