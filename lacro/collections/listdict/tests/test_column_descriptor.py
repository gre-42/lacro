#!/usr/bin/env pytest
# -*- coding: utf-8 -*-

import pytest

from lacro.collections.listdict import SubsetOfType, t2c, unify_dspecs, v2c
from lacro.inspext.app import init_pytest_suite
from lacro.string.misc import to_str

init_pytest_suite()


def test_subset_of_type():
    assert str(SubsetOfType([1, 2, 3])) == 'S(1, 2, 3)'
    assert repr(SubsetOfType([1, 2, 3])) == \
        'SubsetOfType.C(1, 2, 3)'


def test_unify_dspecs():
    a = v2c([1, 2])
    b = v2c([1, 2, 3])
    u = unify_dspecs(dict(x=a), dict(x=b), unite_subset_of_type=True)
    assert str(u) == \
        "{'x': ColumnDescriptor(SubsetOfType.C(1, 2, 3), " \
        "int, int, None, ())}"
    assert to_str(unify_dspecs(dict(x=a), dict(x=b),
                               unite_subset_of_type=True)) == \
        "{'x': S(1, 2, 3)}"
    assert str(a) == 'S(1, 2)'


def test_unify_metas():
    a = t2c(int, meta={'k': 'v'})
    b = t2c(int, meta=None)
    with pytest.raises(ValueError,
                       match='^metas of joined keys do not match.*'):
        unify_dspecs(dict(x=a), dict(x=b), unite_subset_of_type=True)
    u1 = unify_dspecs(dict(x=a), dict(x=b), unite_subset_of_type=True,
                      unite_metas=True)
    assert str(u1) == "{'x': ColumnDescriptor(int, int, int, {'k': 'v'}, ())}"


def test_unify_metas_merge_none():
    a = t2c(int, meta={'k': 'v'})
    b = t2c(int, meta={'r': 'm'})
    with pytest.raises(ValueError,
                       match='^metas of joined keys do not match.*'):
        unify_dspecs(dict(x=a), dict(x=b), unite_subset_of_type=True)
    u1 = unify_dspecs(dict(x=a), dict(x=b), unite_subset_of_type=True,
                      unite_metas=True)
    assert str(u1) == \
        "{'x': ColumnDescriptor(int, int, int, {'k': 'v', 'r': 'm'}, ())}"


def test_unify_metas_merge_conflict():
    a = t2c(int, meta={'k': 'v'})
    b = t2c(int, meta={'k': 'x',
                       'r': 'm'})
    with pytest.raises(ValueError,
                       match="^Detected duplicates in key 'k' that are .*"):
        unify_dspecs(dict(x=a), dict(x=b), unite_subset_of_type=True,
                     unite_metas=True)


def test_unify_metas_merge_success():
    a = t2c(int, meta={'k': 'v',
                       's': 't'})
    b = t2c(int, meta={'k': 'v',
                       'r': 'm'})
    u = unify_dspecs(dict(x=a), dict(x=b), unite_subset_of_type=True,
                     unite_metas=True)
    assert str(u) == ("{'x': ColumnDescriptor(int, int, int, "
                      "{'k': 'v', 's': 't', 'r': 'm'}, ())}")
