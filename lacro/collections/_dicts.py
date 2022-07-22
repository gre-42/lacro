# -*- coding: utf-8 -*-
import copy
import itertools
from collections import OrderedDict
from typing import Iterable, Union

from lacro.assertions import (asserted_of_type, dict_assert_function,
                               list_assert_no_duplicates,
                               lists_assert_disjoint, set_assert_contains,
                               set_assert_subset)

from ._items import items_2_str
from ._lists import list_intersect
from ._misc import Dict2Object
from ._value import object_order_key

# --------------
# - Dictionary -
# --------------


class HashableDict(dict):

    def __hash__(self):
        return hash(tuple(sorted(self.items())))


class HashableOrderedDict(OrderedDict):

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, repr(list(self.items())))

    def __hash__(self):
        return hash(tuple(self.items()))


class GetattrHashableOrderedDict(HashableOrderedDict):

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def dict_2_str(d, **kwargs):
    return items_2_str(sorted(d.items(),
                              key=lambda kv: object_order_key(kv[0]))
                       if type(d) == dict else d.items(), **kwargs)


def dict_get(d, k, msg='', sep='\n', order=True):
    from lacro.string.misc import iterable_2_repr
    # if "d" is a string and "k" is a substring inside that string,
    # then "k not in d == False", but d[k] returns a TypeError
    if type(d) == str:
        raise ValueError('dict_get called with key %r on string %r' % (k, d))
    if k not in d.keys():
        raise ValueError('%sCould not find key %r in dictionary with keys\n%s'
                         % (msg, k, iterable_2_repr(sorted(d.keys())
                                                    if order else
                                                    d.keys(), sep)))
    # Things to know about super
    # www.phyast.pitt.edu/~micheles/python/super.pdf
    return d.__getitem__(k)  # allows use of "super", while d[k] does not


def dict_gets(d, k):
    from lacro.collections import DictGets
    return DictGets(d)[k]


def dict_minus_val(l, r=(None,)):
    return type(l)((k, v) for k, v in l.items() if v not in r)


def dict_minus(l, r, must_exist=True):
    if must_exist:
        set_assert_subset(l.keys(), r)
    return type(l)((k, v) for k, v in l.items() if k not in r)


def dicts_union(dicts: Iterable[dict],
                allow_duplicates: Union[bool, str] = False,
                assert_type=True,
                msg='',
                dict_type=None):
    from lacro.collections import Dict2Object
    from lacro.iterators import iterables_concat
    dicts = list(dicts)  # rewinding iterator
    if dict_type is not None and len(dicts) == 0:
        return dict_type()
    assert len(dicts) > 0  # can not derive result type otherwise
    if dict_type is not None:
        asserted_of_type(dicts[0], dict_type)
    else:
        dict_type = type(dicts[0])
    if type(allow_duplicates) == str:
        assert allow_duplicates == 'nearly'
        dicts_assert_nearly_disjoint(dicts)
    else:
        assert type(allow_duplicates) == bool
        if not allow_duplicates:
            dicts_assert_disjoint(dicts, msg=msg)
    if assert_type:
        for d in dicts:
            asserted_of_type(d, [dict, OrderedDict, HashableOrderedDict,
                                 GetattrHashableOrderedDict,
                                 Dict2Object], varname='Dictionary')
    return dict_type(iterables_concat(a.items() for a in dicts))


def dict_union(*args: Union[dict, Dict2Object],
               allow_duplicates: Union[bool, str] = False,
               msg=''):
    """Returns the union of dictionaries and, if ``allow_duplicates=False``,
    raises an error if a key was contained in several of the dictionaries.

    Parameters
    ----------
    args : iterable
        Dictionaries to compute union of.
    allow_duplicates : boolean, optional
        If ``True``, raise an error if a key was contained in several of the
        dictionaries.
    msg : str, optional
        Message text appended to raised exception.

    Notes
    -----
    ``dict_union(..., allow_duplicates=True)`` is identical to
    ``dict_updated(..., must_exist=False)``

    """
    return dicts_union(args, allow_duplicates=allow_duplicates, msg=msg)


def dict_unite(*args):
    assert len(args) > 0
    dicts_assert_disjoint(args)
    for a in args[1:]:
        args[0].update(a)


def dict_updated(a, b, must_exist=True):
    """Updates the dictionary ``a`` using the values in ``b`` and, if
    ``must_exist=True``, raises an error if on of the values did not yet exist.

    Parameters
    ----------
    a : dict
        Dictionary that shall be updated
    b : dict
        Dictionary that is merged into a
    must_exist : bool, optional
        If ``True``, raises an error if on of the values did not yet exist.

    Notes
    -----
    ``dict_union(..., allow_duplicates=True)`` is identical to
    ``dict_updated(..., must_exist=False)``

    """
    if must_exist:
        set_assert_subset(a.keys(), b.keys())
    res = copy.copy(a)
    res.update(b)
    return res


def dict_update(a, b, must_exist=True):
    if must_exist:
        set_assert_subset(a.keys(), b.keys())
    a.update(b)


def dict_intersect(l, r, must_exist=True):
    list_assert_no_duplicates(r)
    if must_exist:
        set_assert_subset(l.keys(), r)
    return type(l)((k, l[k]) for k in list_intersect(r, list(l.keys())))


def dicts_assert_disjoint(dicts, msg=''):
    lists_assert_disjoint((list(d.keys()) for d in dicts), msg=msg)


def dicts_assert_nearly_disjoint(dicts, operator='equality'):
    from lacro.iterators import iterables_concat
    set_assert_contains(['identity', 'equality'], operator)
    for g, V in itertools.groupby(
            sorted(iterables_concat(d.items() for d in dicts),
                   key=lambda kv: object_order_key(kv[0])),
            key=lambda kv: kv[0]):
        V = list(V)
        # print(g, V)
        for _, vv in V:
            if ((vv is not V[0][1]) if operator == 'identity' else
                    (vv != V[0][1])):
                raise ValueError(f'Detected duplicates in key {g!r} that are '
                                 f'not identical\n{vv!r}\n{V[0][1]!r}')


def inverted_noninjective_dict(dct):
    return {v: [kv[0] for kv in KV]
            for v, KV in itertools.groupby(
                sorted(dct.items(), key=lambda kv: object_order_key(kv[1])),
                key=lambda kv: kv[1])}


def inverted_injective_dict(dct):
    idct = inverted_noninjective_dict(dct)
    dict_assert_function(idct, msg=('Inverted dictionary is not a function. '
                                    'What follows are keys and their target '
                                    'values in the inverted dictionary.'))
    return {k: v[0] for k, v in idct.items()}
    # return {v:k for k,v in dct.items()}


def dict_items(dct, keys):
    return ((k, dct[k]) for k in keys)


def dict_unique_value(dct):
    from lacro.iterators import single_element
    vals = inverted_noninjective_dict(dct)
    if len(vals) != 1:
        raise ValueError('Could not find a unique value in dict "%s". '
                         'Found "%s"' % (dct, vals))
    return single_element(vals.keys())
