# -*- coding: utf-8 -*-
def sets_intersect(sets):
    """
    the ``sets`` iterable must contain at least one set, because an empty
    iterable would result in a set containing everything
    """
    from functools import reduce
    sets = tuple(sets)
    assert len(sets) != 0
    return reduce(lambda A, B: set(A) & set(B), sets[1:], set(sets[0]))
