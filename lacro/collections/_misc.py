# -*- coding: utf-8 -*-
import copy
import itertools
import sys
import threading
from collections import OrderedDict
from contextlib import contextmanager

from lacro.assertions import (list_assert_disjoint, set_assert_contains,
                              set_assert_subset)


class Func2List:
    """
    see also ``indexable_function``
    """

    def __init__(self, func, len_list):
        self.func = func
        self.len_list = len_list

    def __getitem__(self, id):
        if id >= self.len_list:
            raise IndexError('list index out of range')
        return self.func(id)

    def __len__(self):
        return self.len_list

# numpy.fromfunction does not evaluate on-demand


class IterWithLen:

    def __init__(self, iiter, len_list):
        self.iiter = iiter
        self.len_list = len_list

    def __iter__(self):
        return self.iiter

    def __len__(self):
        return self.len_list


class ListGets(list):

    def __getitem__(self, ids):
        from lacro.iterators import is_iterable
        if is_iterable(ids):
            return [list.__getitem__(self, i) for i in ids]
        else:
            return list.__getitem__(self, ids)


class DictGets(dict):

    def __getitem__(self, ids):
        from lacro.iterators import is_iterable
        if is_iterable(ids):
            return [dict.__getitem__(self, i) for i in ids]
        else:
            return dict.__getitem__(self, ids)


def _get_dct(o):
    return object.__getattribute__(o, '_dct')


def _get_forbidden(o):
    return object.__getattribute__(o, '_forbidden')


class ListKeysView:

    def __init__(self, keys):
        self._keys = keys

    def __eq__(self, other):
        return (type(other) == type(self) and
                set(self._keys) == set(other._keys))

    def __iter__(self):
        return iter(self._keys)


class Dict2Object:

    def __init__(self, _dct={}, _forbidden={}, **kwargs):
        _dct = dict(_dct)  # iterator of items
        list_assert_disjoint(_dct.keys(), kwargs.keys())
        _dct.update(kwargs)
        object.__setattr__(self, '_dct', _dct)
        # create copy to avoid updating the default value
        object.__setattr__(self, '_forbidden', dict(_forbidden))

    def keys(self):
        return ListKeysView(list(_get_dct(self).keys()))

    def values(self):
        return list(_get_dct(self).values())

    def items(self):
        return ((k, v) for k, v in _get_dct(self).items())

    def update(self, other):
        _get_dct(self).update(other)

    def updated(self, other):
        res = copy.copy(self)
        res.update(other)
        return res

    def union(self, dct):
        from lacro.collections import dict_union
        return Dict2Object(dict_union(_get_dct(self), dct))

    def __len__(self):
        return len(_get_dct(self))

    def __copy__(self):
        return Dict2Object(_get_dct(self), _get_forbidden(self))

    def __repr__(self):
        from lacro.string.misc import to_repr
        return to_repr(self)

    def __to_repr__(self, **kwargs):
        from lacro.string.misc import to_repr
        return 'Dict2Object(%s%s)' % (
            to_repr(dict(self.items()), **kwargs),
            (', _forbidden=%s' %
             to_repr(_get_forbidden(self), **kwargs))
            if _get_forbidden(self) != {} else '')

    def __str__(self):
        return self.__tostr__()

    def __tostr__(self, **kwargs):
        from lacro.collections import items_2_str
        return items_2_str(sorted(self.items()), **kwargs)

    def __getitem__(self, k):
        if k in _get_forbidden(self).keys():
            raise ValueError('Forbidden key %r was accessed.%s' %
                             (k, _get_forbidden(self)[k]))
        return _get_dct(self)[k]
        # return getattr(self, k) # only accepts strings, no int or other

    def __setitem__(self, k, val):
        _get_dct(self)[k] = val

    def __getattribute__(self, k):
        if k.startswith('__'):
            return object.__getattribute__(self, k)
        try:
            return self[k]
        except KeyError:
            return object.__getattribute__(self, k)

    def __setattr__(self, k, v):
        if k.startswith('__'):
            object.__setattr__(self, k, v)
        else:
            _get_dct(self)[k] = v

    def __delitem__(self, k):
        del _get_dct(self)[k]

    def __delattr__(self, k):
        del _get_dct(self)[k]

    def __contains__(self, k):
        raise ValueError('"in" operator not implemented because I do not like '
                         'it ("a in self" could mean '
                         '"a in self.keys()", "a in self.values()" or "a in '
                         'self.items()"). Use "a in self.keys()" instead')

    def __eq__(self, other):
        return (isinstance(other, Dict2Object) and
                _get_dct(self) == _get_dct(other) and
                _get_forbidden(self) == _get_forbidden(other))


class Incrementer:

    def __init__(self, initial_value=0):
        self.value = initial_value

    def __call__(self, inc=1):
        class DoInc:

            def __enter__(self1):
                self.value += inc

            def __exit__(self1, type, value, traceback):
                self.value -= inc
        return DoInc()

    def is_zero(self):
        return self.value == 0


class TeeIterator:

    def __init__(self, iterable, nthreads):
        self.n_collected = 0
        self.val_lock = threading.Lock()
        self.has_val = False
        self.iterator = iter(iterable)
        self.nthreads = nthreads
        self.barrier = threading.Barrier(nthreads)
        self.exception = None

    def subiterator(self):
        class SubIter:

            def __iter__(self1):
                return self1

            def __next__(self1):
                self.barrier.wait()
                with self.val_lock:
                    if self.exception is not None:
                        a, b, c = self.exception
                        raise a.with_traceback(b, c)
                    # first to enter writes value
                    if not self.has_val:
                        try:
                            self.val = next(self.iterator)
                        except StopIteration:
                            raise
                        except BaseException:
                            self.exception = sys.exc_info()
                            raise
                        self.has_val = True
                    self.n_collected += 1
                    # last to enter deletes value
                    if self.n_collected == self.nthreads:
                        self.has_val = False
                        self.n_collected = 0
                        return self.val
                return self.val
        return SubIter()


def nested_teeiterator(iterable, nesting_level, nthreads):
    if nesting_level == 0:
        return iterable
    else:
        return TeeIterator((nested_teeiterator(it, nesting_level - 1, nthreads)
                            for it in iterable), nthreads)


def nested_subiterator(iterable, nesting_level):
    if nesting_level == 0:
        return iterable
    else:
        return (nested_subiterator(it, nesting_level - 1)
                for it in iterable.subiterator())


def sync_tee(funcs, iterable, nesting_level=1):
    import lacro.parallel.threadmap as ht
    titer = nested_teeiterator(iterable, nesting_level, len(funcs))

    def op(i):
        return funcs[i](nested_subiterator(titer, nesting_level))
    return ht.map(op, range(len(funcs)),
                  nthreads='ONE_PER_ITEM')


class AppendAndIterate:

    def __iter__(self):
        while hasattr(self, 'v'):
            v = self.v
            delattr(self, 'v')
            yield v

    def append(self, v):
        assert not hasattr(self, 'v')
        self.v = v


class IterateGet:

    def __init__(self, func, *args, **kwargs):
        self.ai = AppendAndIterate()
        self.it = iter(func(self.ai, *args, **kwargs))

    def get(self, v):
        self.ai.append(v)
        return next(self.it)


class FunctionChain:
    """
    input: [f1, f2, f3], arg
    output: f3(f2(f1(arg)))
    """

    def __init__(self, funcs):
        self.funcs = funcs

    def __call__(self, arg):
        return (arg if len(self.funcs) == 0 else
                self.funcs[-1](FunctionChain(self.funcs[:-1])(arg)))


class SetWithKey:

    def __init__(self, *args, key=lambda v: v):
        self._set = set(*args)
        self._key = key

    def __contains__(self, v):
        return self._key(v) in self._set

    def add(self, v):
        self._set.add(self._key(v))


def re_inc_exc(include, exclude):
    for v in include + exclude:
        from lacro.string.misc import assert_valid_regex
        assert_valid_regex(v)
    # (?!...) : negative look-ahead
    # (s in include) or (s not in exclude)
    return '^(?:(?:%s)$|(?:(?!(?:%s)$).*))' % ('|'.join(include),
                                               '|'.join(exclude))


def filter_include(data, lst, include, must_exist=True):
    data = list(data)
    lst = list(lst)
    if len(data) != len(lst):
        raise ValueError('len(data) != len(lst)')

    if must_exist:
        set_assert_subset(lst, include)

    from lacro.iterators import eqzip
    return [d for d, s in eqzip(data, lst) if s in include]


def filter_exclude(data, lst, exclude, must_exist=True):
    data = list(data)
    lst = list(lst)
    if len(data) != len(lst):
        raise ValueError('len(data) != len(lst)')

    if must_exist:
        set_assert_subset(lst, exclude)

    from lacro.iterators import eqzip
    return [d for d, s in eqzip(data, lst) if s not in exclude]


class IntCounter:
    """
    see also lacro.dsync.int_counter
    """

    def __init__(self, smallest_number=0, largest_number=None, name='number'):
        self._used_numbers = []
        self._smallest_number = smallest_number
        self._largest_number = largest_number
        self._name = name

    def _get_next_number(self) -> int:
        for number in itertools.count(self._smallest_number):
            if number not in self._used_numbers:
                return number

    def generate(self):
        number = self._get_next_number()
        if ((self._largest_number is not None) and
                (number > self._largest_number)):
            raise ValueError('Exceeding largest %s %d' %
                             (self._name, self._largest_number))
        self._used_numbers.append(number)
        return number

    def delete(self, number):
        del self._used_numbers[self._used_numbers.index(number)]


class ArrayEqual:

    def __init__(self, v, silent=False):
        self.v = v
        self.silent = silent

    def __eq__(self, b):
        import numpy as np
        return np.array_equal(self.v, b)

    def __str__(self):
        return str(self.v) if self.silent else 'EQ(%s)' % self.v

    def __repr__(self):
        return 'ArrayEqual(%r)' % self.v

    def __hash__(self):
        return hash(self.v)


def resolved_tree_path(tree, path, msg=''):
    from lacro.collections import dict_get
    return (tree if len(path) == 0 else
            resolved_tree_path(dict_get(tree, path[0], msg=msg),
                               path[1:],
                               msg))


def conditional_product(args, vals1_args, update_vals1_args):
    if len(args) == 0:
        return [()]
    else:
        # print('--')
        # print(vals1(args[0], vals1_args))
        # print(list(conditional_product(args[1:], vals1,
        #  update_vals1_args(args[0], vals1_args), update_vals1_args)))
        return ((r0,) + r
                for r0 in args[0](vals1_args)
                for r in conditional_product(args[1:],
                                             update_vals1_args(r0, vals1_args),
                                             update_vals1_args))


def map_recursive(f, x):
    if type(x) in (list, tuple):
        return type(x)(map_recursive(f, v) for v in x)
    elif type(x) in (dict, OrderedDict):
        return type(x)(map_recursive(f, v) for v in x.items())
    else:
        return f(x)


class ThreadGlobal:

    def __init__(self, value):
        self._local = threading.local()
        self._global_value = value

    def set(self, value):
        assert ((threading.current_thread().name == 'MainThread') ==
                (not threading.current_thread().daemon))
        if threading.current_thread().name == 'MainThread':
            self._global_value = value
        self._local.value = value

    def get(self):
        if not hasattr(self._local, 'value'):
            self._local.value = self._global_value
        return self._local.value

    @contextmanager
    def with_value(self, value):
        old_value = self.get()
        self.set(value)
        try:
            yield
        finally:
            self.set(old_value)


def mindef(it, default):
    it = list(it)
    return default if len(it) == 0 else min(it)


def maxdef(it, default):
    it = list(it)
    return default if len(it) == 0 else max(it)


def array_ids_join(lst0, lst1, join_type, left_num_copies, right_num_copies,
                   keys='dummy', duplicates_error_string=None, order='left'):
    """
    allows duplicates in at most one of the two lists.

    Example:
        bad:

        ==== ====
        lst0 lst1
        ==== ====
        1    1
        1    1
        ==== ====

        good:

        ==== ====
        lst0 lst1
        ==== ====
        1    1
        1    2
        2    2
        ==== ====

        left/right_num_copies in [1, '*']

    """
    import numpy as np

    from lacro.io.string import print_err
    from lacro.iterators import eqzip
    from lacro.string.misc import checked_match
    left_num_copies, left_nocomment, left_comment = checked_match(
        r'(.*?)(?:{(//)?(.*)\})?$', left_num_copies).groups()
    right_num_copies, right_nocomment, right_comment = checked_match(
        r'(.*?)(?:{(//)?(.*)\})?$', right_num_copies).groups()

    for c in [left_num_copies, right_num_copies]:
        if c not in ['0-1', '1', '*', '+']:
            raise ValueError('num_copies (%s) not in {%s}' %
                             (c, ', '.join(['0-1', '1', '*', '+'])))

    if join_type not in ['inner', 'left', 'right', 'outer']:
        raise ValueError('Unknown join_type: "' + join_type + '"')

    if duplicates_error_string is None:
        def duplicates_error_string(i0, i1): return '\n'.join([
            'lst 0 (id: value)\n%s' % '\n'.join('%4d: %s' % (i, v)
                                                for i, v in enumerate(lst0)),
            'lst 1 (id: value)\n%s' % '\n'.join('%4d: %s' % (i, v)
                                                for i, v in enumerate(lst1)),
            'i0: %d lst0.shape[0]: %d' % (i0, lst0.shape[0]),
            'i1: %d lst1.shape[0]: %d' % (i1, lst1.shape[0]),
            'v0: %s' % (v0,),
            'v1: %s' % (v1,),
            'joining by keys %s' % keys])
    assert lst0.ndim == 1
    assert lst1.ndim == 1

    did0 = np.empty((max(lst0.shape[0], lst1.shape[0]),), dtype=int)
    did1 = np.empty((max(lst0.shape[0], lst1.shape[0]),), dtype=int)
    sid0 = np.empty((max(lst0.shape[0], lst1.shape[0]),), dtype=int)
    sid1 = np.empty((max(lst0.shape[0], lst1.shape[0]),), dtype=int)

    oid0 = np.argsort(lst0)
    oid1 = np.argsort(lst1)

    lst0 = lst0[oid0]
    lst1 = lst1[oid1]

    i0 = 0
    i1 = 0
    l0 = 0
    l1 = 0
    di = 0
    while i0 < lst0.shape[0] or i1 < lst1.shape[0]:
        if i0 == lst0.shape[0]:
            # simulate infinitely large last value after lst0
            v0 = None
            v1 = lst1[i1]
            v0_lt_v1 = False
            v1_lt_v0 = True
        if i1 == lst1.shape[0]:
            # simulate infinitely large last value after lst1
            v0 = lst0[i0]
            v1 = None
            v0_lt_v1 = True
            v1_lt_v0 = False
        if i0 < lst0.shape[0] and i1 < lst1.shape[0]:
            v0 = lst0[i0]
            v1 = lst1[i1]
            v0_lt_v1 = v0 < v1
            v1_lt_v0 = v1 < v0

        if v0_lt_v1:
            if left_num_copies in ['1', '+']:
                raise ValueError('Could not join by keys "' +
                                 repr(list(keys)) +
                                 '" with left_num_copies="' + left_num_copies +
                                 '". The right table is missing some values '
                                 'of the left table.\n' +
                                 duplicates_error_string(i0, i1))

            if (left_nocomment is None and
                    left_comment is not None):
                print_err(left_comment + '   ' +
                          '  '.join('%s: %s' % e for e in eqzip(keys, v0)))

            if join_type in ['left', 'outer']:
                sid0[l0] = i0
                did0[l0] = di
                l0 += 1
                di += 1
            i0 += 1
        elif v1_lt_v0:
            if right_num_copies in ['1', '+']:
                raise ValueError('Could not join by keys "' +
                                 repr(list(keys)) +
                                 '" with right_num_copies="' +
                                 right_num_copies +
                                 '". The left table is missing some values '
                                 'of the right table.\n' +
                                 duplicates_error_string(i0, i1))

            if (right_nocomment is None and
                    right_comment is not None):
                print_err(right_comment + '   ' +
                          '  '.join('%s: %s' % e for e in eqzip(keys, v1)))

            if join_type in ['right', 'outer']:
                sid1[l1] = i1
                did1[l1] = di
                l1 += 1
                di += 1
            i1 += 1
        else:
            sid0[l0] = i0
            sid1[l1] = i1
            did0[l0] = di
            did1[l1] = di
            l0 += 1
            l1 += 1
            di += 1

            oi0 = i0
            oi1 = i1

            if oi0 == lst0.shape[0] - 1 or lst0[oi0 + 1] != v0:
                i1 += 1
            if oi1 == lst1.shape[0] - 1 or lst1[oi1 + 1] != v1:
                i0 += 1

            if i0 == oi0 and i1 != oi1 and left_num_copies in ['0-1', '1']:
                raise ValueError('Could not join by keys "' +
                                 repr(list(keys)) +
                                 '" with left_num_copies="' + left_num_copies +
                                 '". The rows of the left table would be '
                                 'duplicated.\n' +
                                 duplicates_error_string(i0, i1))

            if i1 == oi1 and i0 != oi0 and right_num_copies in ['0-1', '1']:
                raise ValueError('Could not join by keys "' +
                                 repr(list(keys)) +
                                 '" with right_num_copies="' +
                                 right_num_copies +
                                 '". The rows of the right table would be '
                                 'duplicated.\n' +
                                 duplicates_error_string(i0, i1))

            if oi0 == i0 and oi1 == i1:
                raise ValueError('Both lists contain duplicates in their '
                                 'keys. At most one list is allowed to '
                                 'contain duplicates.\n' +
                                 duplicates_error_string(i0, i1))

    set_assert_contains(['left', 'right', 'sort'], order)
    if order == 'left':
        dranks = np.empty((di,), dtype=int)
        dranks[did1[:l1]] = oid1[sid1[:l1]] + l0
        dranks[did0[:l0]] = oid0[sid0[:l0]]
        dranks = np.argsort(np.argsort(dranks))
        # print(did0[:l0])
    elif order == 'right':
        dranks = np.empty((di,), dtype=int)
        dranks[did0[:l0]] = oid0[sid0[:l0]] + l1
        dranks[did1[:l1]] = oid1[sid1[:l1]]
        dranks = np.argsort(np.argsort(dranks))
    elif order == 'sort':
        dranks = np.arange(di, dtype=int)
    else:
        assert False
    # print(dranks)
    # rper0 = dranks[did0[:l0]]
    # rper1 = dranks[did1[:l1]]
    # print(rper0)
    # print(did1[:l1])
    # print(rper1)
    # source list: l
    # dest list: d
    # sorting permutation matrix: P
    # join source permutation matrix: J
    # join dest permutation matrix: D
    # l' = P l
    # D d = J l = J P l
    # inv(J P) D d = inv(J P) J P l
    # iJP0 = np.argsort(oid0[sid0[:l0]])
    # iJP1 = np.argsort(oid1[sid1[:l1]])
    return [oid0[sid0[:l0]],
            oid1[sid1[:l1]]], [dranks[did0[:l0]],
                               dranks[did1[:l1]]], di
