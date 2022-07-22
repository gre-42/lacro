# -*- coding: utf-8 -*-
from collections import Counter
from functools import reduce

from lacro.assertions import (list_assert_no_duplicates,
                               lists_assert_disjoint, set_assert_contains,
                               set_assert_subset)


def list_sorted(keys, head_keys, tail_keys, order='increasing'):
    set_assert_contains(['keep', 'increasing', 'decreasing'], order)
    set_assert_subset(keys, head_keys + tail_keys)
    lists_assert_disjoint([head_keys, tail_keys])
    if order == 'keep':
        return (head_keys
                + list_minus(keys, list_union(head_keys, tail_keys))
                + tail_keys)
    else:
        return (head_keys
                + sorted(set(keys) - set(head_keys) - set(tail_keys),
                       reverse={'increasing': False,
                                'decreasing': True}[order]) +
                tail_keys)

# alternative: used OrderedSet, but this does not yet support indexing


def list_minus(L, R, must_exist=True, msg=''):
    if must_exist:
        set_assert_subset(L, R, msg=msg, unique=False)
    return [l for l in L if l not in R]


def list_union(*args, allow_duplicates=False, order='dont_care', msg=''):
    return lists_union(args, allow_duplicates=allow_duplicates, order=order,
                       msg=msg)


def lists_union(lsts, allow_duplicates=False, order='dont_care', msg=''):
    from lacro.iterators import iterables_concat
    set_assert_contains(['dont_care', 'keep', 're-sort'], order)
    if not allow_duplicates:
        lists_assert_disjoint(lsts, msg=msg)
        if order in ['dont_care', 'keep']:
            return list(iterables_concat(lsts))
        else:
            assert order == 're-sort'
            return sorted(iterables_concat(lsts))
    else:
        if order == 'dont_care':
            return set(iterables_concat(lsts))
        elif order == 're-sort':
            return sorted(set(iterables_concat(lsts)))
        else:
            assert order == 'keep'
            return list(list_removed_duplicates(tuple(iterables_concat(lsts))))

# def list_filter_by_value_slice(lst, slice):
    #import builtins
    # if type(slice) == builtins.slice:
        # return [v for i,v in enumerate(lst) if (
            # slice.start is None or ((slice.start>=0 and v>=slice.start) or (slice.start<0 and len(lst)+slice.start<=i)))
            # and (
            # slice.stop is None or ((slice.stop>=0 and v<=slice.stop) or (slice.stop<0 and len(lst)+slice.stop>=i)))]
    # else:
        # return slice


def slice_from_string(run_id):
    import re
    g_range = re.match('^ *([^:]+)?:([^:]+)? *$', run_id)

    if g_range is not None:
        return slice(*g_range.groups(None))
    else:
        return run_id


def list_filter_by_value_slices(lst, slices, key=lambda x: x, sort=False):
    if type(slices) == slice:
        slices = [slices]
    slices = [s if type(s) == slice else slice(s, s) for s in slices]
    if sort:
        lst = sorted(lst, key=key)
    return [v for i, v in enumerate(lst) if any(
            (s.start is None or (key(v) >= key(s.start)))

            and (s.stop is None or (key(v) <= key(s.stop)))
            for s in slices)]


def list_intersect(*args):
    args = tuple(args)
    if len(args) == 0:
        return []
    else:
        for a in args:
            list_assert_no_duplicates(a)
        return reduce(lambda A, B: [a for a in A if a in B], args[1:], args[0])


def lists_intersect(lsts):
    return list_intersect(*lsts)


def list_duplicates_ids(lst):
    from lacro.iterators import iterable_ids_of_unique
    lst = list(lst)
    ct = Counter(lst)
    return (i for i in iterable_ids_of_unique(lst) if ct[lst[i]] > 1)


def list_duplicates(lst, keep_order=True):
    if keep_order:
        lst = list(lst)
        return [lst[i] for i in sorted(list_duplicates_ids(lst))]
    else:
        return [k for k, c in Counter(lst).items() if c > 1]


def list_concat(*args):
    return lists_concat(args)


def lists_concat(lsts):
    return reduce(lambda a, b: a + b, lsts, [])
    # return list(iterables_concat(lsts)


def list_removed_duplicates(lst):
    from lacro.iterators import iterable_ids_of_unique
    ids = sorted(iterable_ids_of_unique(lst))
    return (lst[i] for i in ids)


def lists_concat2(lsts2):
    return [lists_concat(a) for a in zip(*lsts2)]
