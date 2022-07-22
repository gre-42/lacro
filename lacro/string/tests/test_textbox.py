#!/usr/bin/env pytest
# -*- coding: utf-8 -*-

from lacro.string.textbox import *

se.init_pytest_suite()


def test_01():
    contents = Contents('')
    contents.insert(0, 0, 'a')
    assert contents.text == 'a'
    contents.insert(0, 1, '\n')
    assert contents.text == 'a\n'
    assert contents.yx_2_i(1, 0) == 2
    contents.insert(1, 0, 'b')
    assert contents.text == 'a\nb'

    assert contents.i_2_yx(0) == (0, 0)
    assert contents.i_2_yx(1) == (0, 1)
    assert contents.i_2_yx(2) == (1, 0)

    assert Contents('1234').clip(2, 3).text == '123'
