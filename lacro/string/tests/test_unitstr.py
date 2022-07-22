#!/usr/bin/env pytest
# -*- coding: utf-8 -*-

import os.path

import numpy as np

from lacro.inspext.app import init_pytest_suite
from lacro.path.pathabs import abspath_sl
from lacro.string.unitstr import (UnitFormatted, float_2_str_signif,
                                  precision_for_length, unit_formatter)

init_pytest_suite()


def test_01():
    assert float_2_str_signif(10000.55, 4) == '10000'
    assert precision_for_length(1.2, 4) == '1.20'


def test_00():
    assert unit_formatter.format(
        '{0} {0} {2:H} {2:Ht} {2:10H}', 1, 2, 0.0001) == \
        r'1 1 100μ 100{\boldmath$\mu$}       100μ'
    assert unit_formatter.format('{:5H}', 10) == '   10'
    assert unit_formatter.format('{:5Hc}', 10) == '  10 '
    assert unit_formatter.format('{:5H}', 10000) == '  10k'
    assert unit_formatter.format('{:5Hc}', 10000) == '  10k'
    # print('{} {} {}'.format(1, 2, 3))

    assert unit_formatter.format('--{:5HcnN}--', None) == '----'
    assert unit_formatter.format('--{:nN}--{:nN}', 3, None) == '--3--'
    assert unit_formatter.format('{:H3s} - (nan)', np.nan) == 'nan - (nan)'

    assert unit_formatter.format(
        '{:7%Hc} - {:7%Hc}', 1, 0.01) == '   100% -      1%'
    assert unit_formatter.format('{:7%Hc} -', 1) == '   100% -'
    assert unit_formatter.format('{:7%Hc} -', 0.0015) == '   1.5‰ -'
    assert unit_formatter.format('{:7%Hc} -', 0.001) == '     1‰ -'
    assert unit_formatter.format('{:7%Hfc} -', 0.001) == '   1.0‰ -'
    assert unit_formatter.format('{:7%Hc} - (nan)', np.nan) == \
        '   nan  - (nan)'
    assert unit_formatter.format('{:7%H3sc} -', 0.15) == '  15.0% -'
    assert unit_formatter.format('{:7%H3oc} -', 0.15) == '    15% -'
    assert unit_formatter.format('{0:{1}%H3oc} -', 0.15, 7) == '    15% -'
    assert unit_formatter.format('{:?val} -', False) == ' -'
    assert unit_formatter.format('True: {0:?val|{0}} -', True) ==\
        'True: val -'
    assert unit_formatter.format(
        'False: {0:?val|{0}} -', False) == 'False: False -'
    assert unit_formatter.format(
        '!True: {0:|val?{0}} -', True) == '!True: True -'
    assert unit_formatter.format(
        '!False: {0:|val?{0}} -', False) == '!False: val -'
    assert unit_formatter.format(
        '!!True: {0:|?val ?| {0}} -', True) == '!!True: val ?| True -'
    assert unit_formatter.format('{:?|} -', True) == ' -'
    assert unit_formatter.format('{0:j } -', [1, 2, 3, 4]) == '1 2 3 4 -'
    assert unit_formatter.format('{sFWHM:?{sFWHM:g}|}', sFWHM=3.1) == '3.1'
    assert unit_formatter.format('{v:?{v[0]:e} {v[1]:e}|}',
                                 v=[3.1, 6.4]) == '3.100000e+00 6.400000e+00'

    assert unit_formatter.format('{:H3s}', np.inf) == 'inf'
    assert unit_formatter.format('{:H3s}', -np.inf) == '-inf'

    assert unit_formatter.format('{://x}', 'a/b') == abspath_sl('a/b')
    assert unit_formatter.format('{:/x}', '/a/b') == '/a/b'
    assert unit_formatter.format('{:/x}', 'a/b') == '../a/b'
    assert unit_formatter.format('{:/x/y}', 'a/b') == '../../a/b'
    assert unit_formatter.format('{:/.}', 'a/b') == 'a/b'
    assert unit_formatter.format(
        '{:/..}', 'a/b') == os.path.relpath('a/b', '..')


def test_1():
    assert unit_formatter.format('{:7%eHc} -', 0.15) == '    15% -'
    assert unit_formatter.format('{:7%eHc} -', 0.015) == '   1.5% -'
    assert unit_formatter.format('{:7%eHc} -', 0.0015) == '   1.5‰ -'
    assert unit_formatter.format('{:7%eHc} -', 0.00015) == '1.5e-04 -'
    assert unit_formatter.format('{:7%eHc} -', 0.000015) == '1.5e-05 -'
    assert unit_formatter.format('{:6%eHc} -', 0.000015) == ' 2e-05 -'
    assert unit_formatter.format('{:5%eHc} -', 0.000015) == '2e-05 -'
    assert unit_formatter.format('{:6%eH4oc} -', 1.0983e-04) == ' 1e-04 -'


def test_o():
    v = ['a', 'b', 'cd']
    assert f'{UnitFormatted(v):j }' == 'a b cd'
