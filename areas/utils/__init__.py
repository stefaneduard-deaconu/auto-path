# TODO move the neighbor generators in here
from typing import Callable


def grid_neighbors_gen(i: int, j: int, maxi: int, maxj: int,
                       include_corner_neighbors: bool = False) -> list[tuple[int, int]]:
    coords = [(i - 1, j) if i > 0 else None,
              (i, j - 1) if j > 0 else None,
              (i, j + 1) if j < maxj - 1 else None,
              (i + 1, j) if i < maxi - 1 else None]
    if include_corner_neighbors:
        coords.extend(
            [(i - 1, j - 1) if i > 0 and j > 0 else None,
             (i - 1, j + 1) if i > 0 and j < maxj - 1 else None,
             (i + 1, j + 1) if i < maxi - 1 and j < maxj - 1 else None,
             (i + 1, j - 1) if i < maxi - 1 and j > 0 else None])
    return list(filter(lambda x: x is not None,
                       coords))

def minmax(values: list, key: Callable = None):
    if not key:
        key = lambda x: x
    minimum = min(values, key=key)
    maximum = max(values, key=key)
    return minimum, maximum