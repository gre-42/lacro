# -*- coding: utf-8 -*-
import sys
from functools import partial, total_ordering

import numpy as np

from lacro.collections import ThreadGlobal
from lacro.string.misc import escape_latex, to_strp, to_xhtml


class ObjectNull(object):
    # def __eq__(self, other):
                # return type(other)==type(self)

    def __repr__(self):
        return 'ObjectNull()'

    def __str__(self):
        return '<onull>'


object_null = ObjectNull()
# object_null = '<onull>'
# object_null = None

int_null = sys.maxsize - 42
string_null = '<null>'
# string_undefined = '<undefined>'
float_null = np.float32(np.nan)

assert_deep_consistent = ThreadGlobal(True)

# --------
# - null -
# --------


def assert_valuetype_valid(typ, key='dummy'):
    # dict: train_test.factory_dictlist
    # tuple: classify/summary/app_results.py
    # np.bool_ would not support "null" values
    # if elemtype not in [str, np.int32, np.int64, np.float64, np.float32,
    #                    dict, tuple, np.ndarray, ListDict, UnknownType]:
    #    raise ValueError(F('Unknown elemtype for key "{k}": {elemtype}'))
    if typ == float:
        raise ValueError(f'Key {key!r}: float type not allowed, use '
                         f'np.float32 or np.float64 instead.')


def get_null(type, unknown_is_objnull=False, key='dummy'):
    assert_valuetype_valid(type)
    if type in [np.int32, np.int64]:
        return type(int_null)
    if type in [np.float32, np.float64]:
        return np.nan
    if type == str:
        return string_null
    if unknown_is_objnull:
        return object_null
    raise ValueError(f'Can not generate NaN for type {type.__name__!r} '
                     f'at key {key!r}')


def is_null(v, key='dummy'):
    assert_valuetype_valid(type(v))
    if type(v) in [np.float32, np.float64]:
        return np.isnan(v)
    elif type(v) in [np.int32, np.int64]:
        return v == int_null
    elif type(v) == str:
        return v == string_null
    elif type(v) == ObjectNull:
        return True
    else:
        null = get_null(type(v), unknown_is_objnull=True, key=key)
        # Without the type check,
        # (np.array([1,2,3]) == null) == np.array([False,False,False]).
        # We want a scalar result.
        return type(v) == type(null) and v == null


def null2str(v, k='dummy', null_val='null', float_fmt='{:3e}', **kwargs):
    return null_val if is_null(v, k) else to_strp(v, float_fmt=float_fmt,
                                                  **kwargs)


def null2htmlv(v, key='dummy', null_val='?', null_class=None, escape=False,
               float_fmt='{:3e}', **kwargs):
    if is_null(v, key):
        return (null_val if null_class is None else
                '<span class="{null_class}">{null_val}</span>'.format(
                    null_class=null_class,
                    null_val=null_val))
    else:
        my_to_str = partial(to_strp, float_fmt=float_fmt, **kwargs)
        return to_xhtml(v, to_str=my_to_str) if escape else my_to_str(v)


# noinspection PyUnusedLocal
def null2latexv(v, k='dummy', null_val='null', escape=True, **kwargs):
    if is_null(v, k):
        return null_val
    else:
        return escape_latex(str(v)) if escape else str(v)


@total_ordering
class NullComparable:
    """Comparable object from nullables

    null_small: null < nonnull
    not null_small: nonnull < null

    """

    def __init__(self, v, null_small=True):
        self.v = v
        self._null_small = null_small
        self._is_null = is_null(self.v)

    def __lt__(self, other: 'NullComparable'):
        if other._is_null:
            return (not self._null_small) and (not self._is_null)
        elif self._is_null:
            return self._null_small
        else:
            return self.v < other.v

    def __eq__(self, other: 'NullComparable'):
        return (self._is_null and other._is_null) or (self.v == other.v)

    def __hash__(self):
        return hash(self.v)

    def __repr__(self):
        return ('NullComparable(%r, null_small=%s)' %
                (self.v, self._null_small))


def date_distance(format):
    def dist(a, b):
        from datetime import datetime
        return abs(datetime.strptime(a, format) -
                   datetime.strptime(b, format))
    return dist


def closest_key(key, distance=lambda a, b: abs(a - b)):
    def operations(L):
        closest = np.argmin([distance(L[key], v) for v in L['leaves'][key]])
        return L['leaves'].row_tuple(closest)
    return operations
