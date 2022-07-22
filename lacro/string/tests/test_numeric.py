#!/usr/bin/env pytest
# -*- coding: utf-8 -*-

import pytest
from simpleeval import FeatureNotAvailable

from lacro.inspext.app import init_pytest_suite
from lacro.string.numeric import (align_decimal, fcalculate,
                                  float_string_order_key, g_maxnum_2_str,
                                  i_maxnum_2_str, icalculate,
                                  int_string_order_key, shorten_float_str)

init_pytest_suite()


def test_shorten_float_str():
    assert shorten_float_str('{:3e}'.format(123)) == '1.230000e2'
    assert shorten_float_str('{:3e}'.format(0.123)) == '1.230000e-1'


def test_int_string_order_key():
    lst = [f'a{v}b' for v in range(9, 12)]
    assert sorted(reversed(lst), key=int_string_order_key) == lst
    assert sorted('ab012') == sorted('ab012', key=int_string_order_key)


def test_float_string_order_key():
    lst = [f'a{v / 10:g}b' for v in range(9, 12)]
    assert sorted(reversed(lst), key=float_string_order_key) == lst


def test_13():
    with pytest.raises(FeatureNotAvailable):
        icalculate('()')
    assert fcalculate('1+2.5') == 3.5
    with pytest.raises(ValueError):
        icalculate('1+2.5')
    assert fcalculate('1+2') == 3


def test_15():
    assert align_decimal(5, 3, 2) == '  5   '
    assert align_decimal(5.0, 3, 2) == '  5   '
    assert align_decimal(5.1, 3, 2) == '  5.1 '
    assert align_decimal(5.1, 3, 3) == '  5.1  '
    assert align_decimal(5.1, 2, 3) == ' 5.1  '
    assert align_decimal(5.1, 2, 0) == ' 5.1'
    assert align_decimal(5.11, 2, 0) == ' 5.11'
    assert align_decimal(5.11, 2, 0, fillchar='0') == '05.11'
    assert align_decimal(5, 2, 0) == ' 5'

    assert i_maxnum_2_str(5, maxnum=9) == '5'
    assert g_maxnum_2_str(5, maxnum=9) == '5'
    assert i_maxnum_2_str(5, maxnum=10) == ' 5'
    assert g_maxnum_2_str(5, maxnum=10) == ' 5'
    with pytest.raises(ValueError):
        i_maxnum_2_str(5.1, maxnum=10)
    # self.assertEqual(g_maxnum_2_str(5.1, maxnum=10000, fillchar='0'), ' 5.1')
