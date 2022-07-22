#!/usr/bin/env pytest
# -*- coding: utf-8 -*-


import os.path
import re
from functools import partial
from typing import Callable, cast

import pytest

from lacro.collections import (AppendAndIterate, Cache, CompressedCache,
                               CompressedDict, Dict2Object, FunctionChain,
                               GetattrHashableOrderedDict, Incrementer,
                               IterateGet, ListKeysView, array_ids_join,
                               filter_exclude, filter_include, list_sorted,
                               re_inc_exc, sync_tee)
from lacro.collections._cache import tuple_allclose, tuple_types_torepr
from lacro.inspext.app import init_pytest_suite
from lacro.io.string import load_from_pickle, save_to_pickle
from lacro.path.pathmod import remove_files
from lacro.string.misc import to_repr

init_pytest_suite()


def srepr(s):
    return to_repr(s, sort_dict=True)


def test_dict2object_str():
    assert str(Dict2Object({'s': 5, 't': 8})) == 's:  5\nt:  8'
    assert srepr(Dict2Object({'s': 5, 't': 8})) == \
        "Dict2Object({'s': 5, 't': 8})"


def test_dict2object_func():
    a = Dict2Object(
        x=3,
        f=lambda: 5)
    assert a.x == 3
    assert cast(Callable, a.f)() == 5
    assert a['x'] == 3
    assert a['f']() == 5
    a.x = 4
    a.f = 2
    assert a.x == 4
    assert a.f == 2


def test_dict2object_1(tmpdir):
    j = partial(os.path.join, str(tmpdir))
    a = Dict2Object({'s': 5, 't': 8})
    assert a['s'] == 5
    a['s'] = -555
    assert srepr(a) == "Dict2Object({'s': -555, 't': 8})"
    a.s = -444
    assert srepr(a) == "Dict2Object({'s': -444, 't': 8})"
    # print('s' in a)
    f = Dict2Object({'s': 5, 't': 8}, _forbidden={'g': ' asd'})
    assert srepr(f) == \
        "Dict2Object({'s': 5, 't': 8}, " \
        "_forbidden={'g': ' asd'})"
    with pytest.raises(ValueError,
                       match="Forbidden key 'g' was accessed. asd"):
        # noinspection PyStatementEffect
        f.g
    with pytest.raises(ValueError,
                       match="Forbidden key 'g' was accessed. asd"):
        # noinspection PyStatementEffect
        f['g']
    # print(Dict2Object({'s':5,'t':8},_forbidden={'g':'asd'}).s)
    # print(Dict2Object({'s':5,'t':8},_forbidden={'s':'asd'}).s)
    # print(Dict2Object({'s':5,'t':8},_forbidden={'s':'asd'})['s'])
    assert Dict2Object(a=5).keys() == ListKeysView(['a'])
    assert Dict2Object(keys=5).keys == 5
    del a['s']
    assert srepr(a) == "Dict2Object({'t': 8})"
    save_to_pickle(j('a.blob'), a)
    assert load_from_pickle(j('a.blob')) == a
    remove_files([j('a.blob')])
    del a.t
    assert srepr(a) == "Dict2Object({})"


def test_getattr_hashable():
    d = GetattrHashableOrderedDict()
    d['a'] = 5
    assert d.a == 5
    with pytest.raises(AttributeError, match='^b$'):
        # noinspection PyStatementEffect
        d.b


def test_1():
    assert sync_tee(
        [lambda X:[x for x in X],
         lambda X: [2 * x for x in X],
         lambda X: [3 * x for x in X]],
        range(4)) == [[0, 1, 2, 3], [0, 2, 4, 6], [0, 3, 6, 9]]


def test_append_and_iterate():
    cont = AppendAndIterate()
    k = (i * 2 for i in cont)
    cont.append(3)
    assert next(k) == 6
    cont.append(4)
    assert next(k) == 8
    with pytest.raises(StopIteration):
        next(k)
    cont.append(4)
    with pytest.raises(AssertionError):
        cont.append(4)


def test_iterage_get():
    ig = IterateGet(lambda it: (2 * i for i in it))
    assert ig.get(4) == 8
    assert ig.get(3) == 6


def test_function_chain():
    assert FunctionChain(
        [lambda s: s + '1 ',
         lambda s: s + '2 ',
         lambda s: s + '3 '])('begin ') == 'begin 1 2 3 '


def test_compressed_dict():
    assert repr(CompressedDict.from_decompressed_items([('a', 'g')])) == \
        "CompressedDict.from_decompressed_items([('a', 'g')])"
    assert list(CompressedDict.from_decompressed_items([('a', 'g')])
                .decompressed_items()) == [('a', 'g')]
    assert re.search(r"^\[\('a', b'.*'\)\]$",
                     repr(list(CompressedDict
                               .from_decompressed_items([('a', 'g')])
                               .compressed_items())))
    assert re.search(r"^\[\('a', b'.*'\)\]$",
                     repr(list(CompressedDict
                               .from_decompressed_items([('a', [0] * 1000)])
                               .compressed_items())))


def test_cache(tmpdir):
    j = partial(os.path.join, str(tmpdir))
    cache = Cache(filename=j('asd.blob'))
    with pytest.raises(KeyError):
        cache.__getitem__('5')
    assert cache.get('5', lambda: 54321) == 54321
    assert cache['5'] == 54321
    assert cache.get(6 + 1e-12, lambda: 612) == 612
    assert list(cache.keys()) == ['5', 6.000000000001]
    cache._close_duplicates = 'raise'
    with pytest.raises(ValueError):
        cache.get(6, lambda: 612)
    cache._close_duplicates = 'rewrite'
    assert cache.get(6, lambda: 600) == 612
    assert list(cache.keys()) == ['5', 6]
    assert cache.get((1, 2, 3), lambda: 123) == 123
    assert cache.keys_tostr() == '''\
0: "'5'"
   str
1: '6'
   int
2: '(1, 2, 3)'
   (int, int, int)'''

    gcache = CompressedCache(j('asd.blob.lgz'))

    assert gcache.get('5', lambda: 7654) == 54321
    assert gcache.get('7', lambda: 7654) == 7654
    assert gcache.get(6, lambda: 7654) == 612
    gcache._close_duplicates = 'rewrite'
    assert gcache.get(6 + 1e-15, lambda: 615) == 612
    assert list(gcache.keys()) == ['5', (1, 2, 3), '7',
                                   6.000000000000001]

    gcache1 = CompressedCache(j('asd.blob.lgz'))
    assert gcache1.get('5', lambda: 55) == 54321
    assert gcache1.get('7', lambda: 55) == 7654

    gcache2 = CompressedCache(j('asd2.blob.lgz'))
    assert gcache2.get('5', lambda: 55) == 55
    assert gcache2.get('5', lambda: 88) == 55

    gcache3 = CompressedCache()
    assert gcache3.get('5', lambda: 99) == 99
    assert gcache3.get('5', lambda: 88) == 99

    remove_files([j('asd.blob.lgz'), j('asd2.blob.lgz')])


def test_tuple():
    assert tuple_types_torepr((1, 2, 1.3)) == '(int, int, float)'
    assert tuple_types_torepr(((1, 2, 1.3),)) == \
        '((int, int, float))'
    assert tuple_types_torepr((1, 2, [])) == '(int, int, list)'
    assert tuple_allclose((1, 2, 1.3), (1, 2, 1.3)) is True
    assert tuple_allclose((1, 2, 1.3), (1, 2, 1.4)) is False
    assert tuple_allclose((1, 2, 1.3), (1, 2,)) is False
    assert tuple_allclose((1, 2, 1.3), (1, 2.0,)) is False
    assert tuple_allclose((1, '2'), (1, '2')) is True
    assert tuple_allclose({1: 2, 3: 4}, {1: 2, 3: 4}) is True
    assert tuple_allclose({1: 2, 3: 4}, {1: 2, 3: 5}) is False
    assert tuple_allclose({1: 2, 3: 4}, {1: 2, 4: 4}) is False
    with pytest.raises(ValueError):
        tuple_allclose((1, '2'), (1, []))


def test_filter_include_exclude():
    include = ['ab', 'c']
    exclude = ['de', 'f']
    lst = include + exclude + ['def']
    data = [i + 1 for i in range(len(lst))]
    assert re_inc_exc(include, exclude) == \
        '^(?:(?:ab|c)$|(?:(?!(?:de|f)$).*))'
    assert filter_include(data, lst, include) == [1, 2]
    assert filter_exclude(data, lst, exclude) == [1, 2, 5]
    assert [s for s in lst
            if re.match(re_inc_exc(['ab'], ['.*']), s)] == ['ab']
    assert [s for s in lst
            if re.match(re_inc_exc([], ['de']), s)] == \
        ['ab', 'c', 'f', 'def']
    assert [s for s in lst
            if re.match(re_inc_exc([], ['d']), s)] == \
        ['ab', 'c', 'de', 'f', 'def']


def test_list_sorted():
    assert list_sorted(list('cab'), ['b'], [],
                       order='increasing') == \
                     list('bac')
    assert list_sorted(list('cab'), ['b'], [],
                       order='decreasing') == \
        list('bca')
    assert list_sorted(list('cab'), ['b'], [],
                       order='keep') == \
        list('bca')


@pytest.mark.parametrize(
    'lst0, lst1, join_type, left_num_copies, right_num_copies, '
    'oid0, oid1, did0, did1, di',
    [([0, 1, 2], [0, 2, 1], 'inner', '1', '1',
      [0, 1, 2], [0, 2, 1], [0, 1, 2], [0, 1, 2], 3),
     ([0, 1, 2], [2, 0], 'inner', '0-1', '1',
      [0, 2], [1, 0], [0, 1], [0, 1], 2),
     ([0, 1, 2], [2, 0], 'left', '0-1', '1',
      [0, 1, 2], [1, 0], [0, 1, 2], [0, 2], 3),
     ([2, 0], [0, 1, 2], 'right', '1', '0-1',
      [1, 0], [0, 1, 2], [1, 0], [1, 2, 0], 3)])
def test_array_ids_join(lst0, lst1, join_type, left_num_copies,
                        right_num_copies, oid0, oid1, did0, did1, di):
    import numpy as np
    from numpy.testing import assert_array_equal
    [oid0_, oid1_], [did0_, did1_], di_ = array_ids_join(
        np.array(lst0), np.array(lst1), join_type,
        left_num_copies, right_num_copies)
    assert_array_equal(oid0_, oid0)
    assert_array_equal(oid1_, oid1)
    assert_array_equal(did0_, did0)
    assert_array_equal(did1_, did1)
    assert di_ == di


def test_incrementer():
    inc = Incrementer()
    assert inc.value == 0
    with inc(3):
        assert inc.value == 3
        with inc(2):
            assert inc.value == 5
        assert inc.value == 3
    assert inc.value == 0
