import logging

# Bitset-based unordered sets of unsigned integers
from intbitset import intbitset

from termcolor import colored
from typing import Set, Tuple, List, Union, Dict, Any, Optional, Union
from collections import defaultdict


class TransitiveClosure:
    def __init__(self, TC: Any = None, debug: bool = False):
        start_at: Dict[int, intbitset] = defaultdict(intbitset)
        end_at: Dict[int, intbitset] = defaultdict(intbitset)
        self._TC: Dict[str, Dict[int, intbitset]] = {"start-at": start_at, "end-at": end_at}
        self._acyclic: bool = True
        self._debug: bool = debug

        if TC is not None:
            self._acyclic = TC._acyclic
            for src, dst in TC.edges():
                self._add(src, dst)

    # Direct edge addition (no propagation)
    def _add(self, src: int, dst: int):
        self._TC["start-at"][src].add(dst)
        self._TC["end-at"][dst].add(src)

   # Compute edges that would result of adding some edge
    def _new_edges(self, src: int, dst: int) -> Set[Tuple[int, int]]:
        new_edges: Set[Tuple[int, int]] = set([(src, dst)])
        # Add (x,dst) if (x,src)
        new_edges |= set([(x, dst) for x in self.end_at(src)])
        # Add (src,y) if (dst,y)
        new_edges |= set([(src, y) for y in self.start_at(dst)])
        # Add (x,y) if (x,src) & (dst,y)
        new_edges |= set([(x, y) for x in self.end_at(src) for y in self.start_at(dst)])
        return new_edges - self.edges()

    def acyclic(self) -> bool:
        return self._acyclic

    def edge(self, src: int, dst: int) -> bool:
        return src in self.end_at(dst)

    def edges(self) -> Set[Tuple[int, int]]:
        return set([(src, dst) for src, list_dst in self._TC["start-at"].items() for dst in list_dst])

    def start_at(self, src: int) -> intbitset:
        return self._TC["start-at"].get(src, intbitset())

    def end_at(self, dst: int) -> intbitset:
        return self._TC["end-at"].get(dst, intbitset())

    def update(self, src: int, dst: int):
        new_edges: List[Tuple[int, int]] = self._new_edges(src, dst)
        for (src2, dst2) in new_edges:
            self._add(src2, dst2)
            self._acyclic = self._acyclic or src2 == dst2

    def update_with_path(self, path: Tuple[int]):
        for (src, dst) in zip(path[:-1], path[1:]):
            self.update(src, dst)

    def acyclic_if_path_added(self, path: Tuple[int]) -> bool:
        clone: TransitiveClosure = TransitiveClosure(self)
        clone.update_with_path(path)
        return clone.acyclic()

    def calculate_ranks(self, vertices: intbitset, unique: bool = False) -> Dict[int, int]:
        # If not acyclic, return None as there is no proper ranking
        if not self.acyclic_: return None

        # Vertices of rank = 0
        ranks: Dict[int, int] = {i: 0 for i in vertices if len(self.end_at(i)) == 0}
        assigned: intbitset = intbitset(list(ranks.keys()))

        # Vertices of rank > 0
        if not unique:
            rank = 1
            while len(assigned) < len(vertices):
                new_assigned: intbitset = intbitset()
                for j in vertices - assigned:
                    # Vertex j has rank R iff there is i with rank R-1 such that (i, j) in TC AND there is no k with (i, k) AND (k, j) in TC
                    below: List[int] = [i for i in self.end_at(j) if ranks.get(i) == rank - 1 and len(self.start_at(i) & self.end_at(j)) == 0]
                    if len(below) > 0:
                        ranks[j] = rank
                        new_assigned.add(j)
                        assigned.add(j)
                assert len(new_assigned) > 0
                assigned |= new_assigned
                rank += 1
        else:
            rank = 1
            while len(assigned) < len(vertices):
                for j in vertices - assigned:
                    if len(self.end_at(j) - assigned) == 0:
                        ranks[j] = rank
                        assigned.add(j)
                        break
                rank += 1

        return ranks

