# -*- coding: utf-8 -*-
from functools import total_ordering


def object_order_key(v):
    # return id(type(v)), v
    return v.__class__.__name__, v


@total_ordering
class IdAndValue:

    def __init__(self, i, v):
        self.i = i
        self.v = v

    def __lt__(self, other):
        return self.v < other.v

    def __eq__(self, other):
        return self.v == other.v

    def __hash__(self):
        return hash(self.v)

    def __repr__(self):
        return 'IdAndValue(%r,%r)' % (self.i, self.v)
