#!/usr/bin/env pytest
# -*- coding: utf-8 -*-
import os.path
from functools import partial
from os import unlink

import pytest

from lacro.collections.dictlike_tuples import (dictlike_tuple,
                                               named_dictlike_tuple)
from lacro.inspext.app import init_pytest_suite
from lacro.io.string import load_from_pickle, save_to_pickle
from lacro.string.misc import to_str

init_pytest_suite()


GT = dictlike_tuple
GNT = named_dictlike_tuple

# global for "load_from_pickle" (__reduce__ is called only if pytest is used)
tuple_pname = GT('tuple_pname', parent_module_name=__name__)
tuple_pauto = GT('tuple_pauto', parent_module_name='<auto>')

ntuple_pname = GNT('ntuple_pname', ['x', 'y'], parent_module_name=__name__)
ntuple_pauto = GNT('ntuple_pauto', ['x', 'y'], parent_module_name=__name__)


@pytest.mark.parametrize('readonly', [False, True])
def test_a(readonly):
    A = GNT('A', ('x', 'y'), readonly=readonly, parent_module_name=__name__)
    a = A.C(1, 7)

    assert repr(a) == 'A.C(1, 7)'
    assert str(a) == 'A.C(1, 7)'
    assert to_str(a) == 'A.C(1, 7)'

    if readonly:
        with pytest.raises(ValueError, match='^A created as readonly$'):
            a.x = 30
    else:
        a.x = 30
        assert a == A.C(30, 7)


def test_0():
    TT = GT('MyType', parent_module_name=__name__)
    TT2 = GT('MyType2', parent_module_name=__name__)

    b = TT([1, 2, '3'])
    X = b + TT([1])
    #b + [1]
    assert b == TT.C(1, 2, '3')
    assert b != TT2.C(1, 2, '3')
    assert X == TT.C(1, 2, '3', 1)
    assert b[0] == 1
    assert b[1:] == TT.C(2, '3')


def test_1():
    TT = GNT('MyType', ['u', 'v', 'w'], parent_module_name=__name__)
    TT2 = GNT('MyType2', ['u', 'v', 'w'], parent_module_name=__name__)

    b = TT([1, 2, '3'])
    assert b == TT.C(1, 2, '3')
    assert b != TT2.C(1, 2, '3')
    assert b == TT.C(1, 2, w='3')
    assert [b.u, b.v, b.w] == [1, 2, '3']


@pytest.mark.parametrize('ttuple', [tuple_pname, tuple_pauto,
                                    ntuple_pname, ntuple_pauto])
def test_2(tmpdir, ttuple):
    j = partial(os.path.join, str(tmpdir))
    save_to_pickle(j('a.blob'), ttuple([(1, 2), (3, 4)]))
    assert load_from_pickle(j('a.blob')) == ttuple([(1, 2), (3, 4)])
    unlink(j('a.blob'))

    save_to_pickle(j('a.blob'), ttuple)
    assert load_from_pickle(j('a.blob')) == ttuple
    unlink(j('a.blob'))


@pytest.mark.parametrize('ttuple', [
    GT('MyType', parent_module_name=__name__),
    GNT('MyType', ['x', 'y'], parent_module_name=__name__)])
def test_3(ttuple):
    a = ttuple.C(1, 2)
    with pytest.raises(TypeError, match="^unhashable type: 'MyType'$"):
        hash(a)


@pytest.mark.parametrize('ttuple', [
    GT('MyType', hashable=True, parent_module_name=__name__),
    GNT('MyType', ['x', 'y'], hashable=True, parent_module_name=__name__)])
def test_4(ttuple):
    a = ttuple.C(1, 2)
    hash(a)
