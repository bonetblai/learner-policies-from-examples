import sys
from typing import List, Dict, Set
from collections import defaultdict

if __name__ == "__main__":
    option: str = sys.argv[1]
    final_args: str = sys.argv[2]
    splitted: List[str] = [arg.strip(" ") for arg in final_args.split(" ") if arg != ""]
    values: List[str] = []
    for i, arg in enumerate(splitted):
        if arg == option:
            values.append(splitted[i+1])
    if len(values) > 0: print(values[-1])

