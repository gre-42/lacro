#!/usr/bin/env pytest
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from lacro.collections.listdict._elements.values import (NullComparable,
                                                          float_null,
                                                          string_null)
from lacro.inspext.app import init_pytest_suite

init_pytest_suite()


def test_float():
    with pytest.raises(ValueError):
        NullComparable(1.0)


@pytest.mark.parametrize('a, b', [(np.int32(2), np.int32(3)),
                                  ('2', '3')])
def test_eq_2_3(a, b):
    assert NullComparable(a) == NullComparable(a)
    assert not (NullComparable(a) != NullComparable(a))
    assert NullComparable(b) != NullComparable(a)


@pytest.mark.parametrize('x1, x2', [(np.int32(1), np.int32(2)),
                                    ('1', '2')])
def test_lgt_1_2(x1, x2):
    assert not (NullComparable(x2) < NullComparable(x2))
    assert not (NullComparable(x2) > NullComparable(x2))
    assert NullComparable(x1) < NullComparable(x2)
    assert NullComparable(x2) > NullComparable(x1)


@pytest.mark.parametrize('x, null', [('1', string_null), (1, float_null)])
def test_lt_n_1(x, null):
    assert (NullComparable(null, null_small=True) <
            NullComparable(x, null_small=True))
    assert not (NullComparable(null, null_small=False) <
                NullComparable(x, null_small=False))


@pytest.mark.parametrize('x, null', [('1', string_null), (1, float_null)])
def test_lt_1_n(x, null):
    assert not (NullComparable(x, null_small=True) <
                NullComparable(null, null_small=True))
    assert (NullComparable(x, null_small=False) <
            NullComparable(null, null_small=False))


@pytest.mark.parametrize('null', [string_null, float_null])
def test_lt_n_n(null):
    assert not (NullComparable(null, null_small=True) <
                NullComparable(null, null_small=True))
    assert not (NullComparable(null, null_small=False) <
                NullComparable(null, null_small=False))


@pytest.mark.parametrize('null', [string_null, float_null])
@pytest.mark.parametrize('null_small', [True, False])
def test_eq_n_n(null, null_small):
    assert (NullComparable(null, null_small=null_small) ==
            NullComparable(null, null_small=null_small))


@pytest.mark.parametrize('null', [string_null, float_null])
@pytest.mark.parametrize('null_small', [True, False])
def test_ne_n_n(null, null_small):
    assert not (NullComparable(null, null_small=null_small) !=
                NullComparable(null, null_small=null_small))


@pytest.mark.parametrize('null', [string_null, float_null])
@pytest.mark.parametrize('null_small', [True, False])
def test_eq_n_1(null, null_small):
    assert not (NullComparable(null, null_small=null_small) ==
                NullComparable('1', null_small=null_small))


@pytest.mark.parametrize('null', [string_null, float_null])
@pytest.mark.parametrize('null_small', [True, False])
def test_ne_n_1(null, null_small):
    assert (NullComparable(null, null_small=null_small) !=
            NullComparable('1', null_small=null_small))


@pytest.mark.parametrize('null', [string_null, float_null])
@pytest.mark.parametrize('null_small', [True, False])
def test_eq_1_n(null, null_small):
    assert not (NullComparable('1', null_small=null_small) ==
                NullComparable(null, null_small=null_small))


@pytest.mark.parametrize('null', [string_null, float_null])
@pytest.mark.parametrize('null_small', [True, False])
def test_ne_1_n(null, null_small):
    assert (NullComparable('1', null_small=null_small) !=
            NullComparable(null, null_small=null_small))
