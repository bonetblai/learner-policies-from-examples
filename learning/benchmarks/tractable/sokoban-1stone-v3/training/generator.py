import logging, sys
import random
from typing import List, Tuple, Dict, Set, Any, Optional
from itertools import product


def encode(col: int, row: int, nrows: int) -> int:
    return (col - 1) * nrows + (row - 1)

def decode(pos: int, nrows: int) -> Tuple[int, int]:
    col = pos // nrows
    row = pos % nrows
    return 1 + col, 1 + row

def generate(ncols: int, nrows: int):
    npos = ncols * nrows
    cols: List[int] = list(range(1, 1 + ncols))
    rows: List[int] = list(range(1, 1 + nrows))
    positions: List[Tuple[int, int]] = sorted(product(cols, rows))

    wall_positions: List[Tuple[int, int]] = []
    wall_positions: List[Tuple[int, int]] = wall_positions + [(col, 1) for col in range(1, 1 + ncols)]
    wall_positions: List[Tuple[int, int]] = wall_positions + [(1, row) for row in range(1, 1 + nrows)]
    wall_positions: List[Tuple[int, int]] = wall_positions + [(col, nrows) for col in range(1, 1 + ncols)]
    wall_positions: List[Tuple[int, int]] = wall_positions + [(ncols, row) for row in range(1, 1 + nrows)]

    inner_positions: List[Tuple[int, int]] = sorted(set(positions) - set(wall_positions))
    stone, player, goal = random.sample(inner_positions, 3)
    while stone == player or stone == goal or player == goal:
        stone, player, goal = random.sample(inner_positions, 3)

    clear_positions: Set[Tuple[int, int]] = set(positions) - set(wall_positions) - set([stone]) - set([player])

    logging.info(f"Generate Sokoban problem with 1 stone: ncols={ncols}, nrows={nrows}, stone={stone}, player={player}, goal={goal}")

    print(f";;  ", end="")
    for col in range(ncols):
        x_col = (1 + (col % 10)) % 10
        print(x_col, end="")
    print(f"")

    for row in range(nrows):
        x_row = (1 + (row % 10)) % 10
        print(f";; {x_row}", end="")
        for col in range(ncols):
            pos = (1 + col, 1 + row)
            if pos in wall_positions:
                print("#", end="")
            elif pos == stone:
                print("$", end="")
            elif pos == player:
                print("@", end="")
            elif pos == goal:
                print(".", end="")
            else:
                print(" ", end="")
        print("")

    print(f"")
    print(f"(define (problem x-microban-sequential-1stone-{ncols}-{nrows}-{encode(*stone, nrows)}-{encode(*player, nrows)}-{encode(*goal, nrows)})")
    print(f"  (:domain sokoban-sequential)")
    print(f"  (:objects")

    for col, row in positions:
        print(f"      pos-{col}-{row} - location")

    print(f"      stone-01 - stone")
    print(f"  )")
    print(f"  (:init")

    # Goal and non-goal positions
    for pos in positions:
        col, row = pos
        if pos == goal:
            print(f"      (IS-GOAL pos-{col}-{row})")
        else:
            print(f"      (IS-NONGOAL pos-{col}-{row})")

    # Right and down adjecency 
    print(f"")
    for pos in positions:
        col, row = pos
        if col < ncols:
            print(f"      (ADJ-RIGHT pos-{col}-{row} pos-{1+col}-{row})")
        if row < nrows:
            print(f"      (ADJ-RIGHT pos-{col}-{row} pos-{col}-{1+row})")

    # Locate player and stone
    print(f"")
    print(f"      (player pos-{player[0]}-{player[1]})")
    print(f"      (at stone-01 pos-{stone[0]}-{stone[1]})")

    # Clear
    for pos in positions:
        col, row = pos
        if pos in clear_positions:
            print(f"      (clear pos-{col}-{row})")

    print(f"  )")
    print(f"  (:goal (at-goal stone-01))")

    print(f")")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <rows> <cols>")
        exit(-1)

    nrows = int(sys.argv[1])
    ncols = int(sys.argv[2])

    generate(nrows, ncols)
