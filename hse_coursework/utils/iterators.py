from itertools import islice
from typing import Iterable, Tuple, TypeVar

T = TypeVar("T")


def window(seq: Iterable[T], width: int = 2) -> Iterable[Tuple[T, ...]]:
    """
    Returns a sliding window over data from the iterable
    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    """
    it = iter(seq)
    result = tuple(islice(it, width))
    if len(result) == width:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
