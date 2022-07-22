#!/usr/bin/env pytest
# -*- coding: utf-8 -*-

from collections import OrderedDict

import pytest

from lacro.inspext.app import init_pytest_suite
from lacro.string.misc import (F0, FF, _newline_join, box_chars_1,
                               checked_getenv, clean_using_dictionary,
                               colored_words, exp_latex, hyphenation,
                               hyphenation_cat, mywrap, opt_if_nonnull,
                               replace_all, text_sliding_window, textbox,
                               textline, to_repr, unique_substrings)

init_pytest_suite()


class A:

    def __repr__(self):
        return '%s()' % self.__class__.__name__


class B:

    # noinspection PyUnusedLocal
    def __to_repr__(self, **kwargs):
        return '%s<x\ny>' % self.__class__.__name__

    def __repr__(self):
        return '%s(x\ny)' % self.__class__.__name__


def srepr(s):
    return to_repr(s, sort_dict=True)


def test_0():
    assert to_repr(A) == f"<class '{__name__}.A'>"
    assert to_repr(A()) == "A()"

    assert to_repr(A, add_module=True) == f"<class '{__name__}.A'>"
    assert to_repr(A, type_repr=True,
                   add_module=True) == f"{__name__}.A"
    assert to_repr(A(), add_module=True) == f"{__name__}.A()"

    assert to_repr({1: 1, '2': '2'}, sort_dict=True) == \
        "{1: 1, '2': '2'}"
    assert to_repr({1: 1, '2': '2\n3'}, sort_dict=True) == \
        r"{1: 1, '2': '2\n3'}"
    assert to_repr({1: 1, '2': B()}, sort_dict=True) == \
        "{1: 1, '2': B<x\ny>}"

    assert to_repr(1) == "1"
    assert to_repr(1, add_module=True) == "1"

    assert to_repr(OrderedDict([(1, 1), ('2', B()), (3, 3)])) == \
        "OrderedDict([(1, 1), ('2', B<x\ny>),\n\n(3, 3)])"

    assert _newline_join(['1', '2\nN', '3'], ', ', ',\n') == \
        '1, 2\nN,\n3'


def test_1():
    assert text_sliding_window('a\n bb c dd', 'bb'.split(),
                               1, preserve='space+newline') == \
        '\n bb'
    assert text_sliding_window('a\n bb c dd', 'bb'.split(),
                               1, preserve='space') == \
        '  bb'
    assert text_sliding_window('a\n bb c dd', 'bb'.split(),
                               1, preserve='none') == \
        'bb'
    assert text_sliding_window('a bb c dd', 'bb c'.split(),
                               1, preserve='none') == \
        'bb c'
    assert text_sliding_window('a bb c dd', 'bb c'.split(),
                               3, preserve='none') == \
        'a bb c dd'
    assert text_sliding_window('a bb c dd', 'bb c'.split(),
                               2, preserve='none') == \
        'a bb c'
    assert text_sliding_window('a BB c dd', 'bb c'.split(),
                               2, preserve='none') == \
        'BB c'
    assert text_sliding_window('a BB c dd', 'bb c'.split(),
                               2, preserve='none',
                               ignore_case=True) == \
        'a BB c'

    assert text_sliding_window('a bb c dd', ''.split(),
                               1, preserve='none') == ''
    assert text_sliding_window('a bb c dd', 'a bb c dd'.split(),
                               1, preserve='none') == \
        'a bb c dd'

    assert text_sliding_window('a bb c dd', 'bb dd'.split(),
                               1, preserve='none') == \
        'bb\ndd'
    assert text_sliding_window('a bb c dd', 'BB dd'.split(),
                               1, preserve='none') == \
        'dd'
    assert text_sliding_window('a bb c dd', 'BB dd'.split(),
                               1, preserve='none',
                               ignore_case=True) == \
        'bb\ndd'
    assert text_sliding_window('a bb c dd', '.*'.split(),
                               1, preserve='none',
                               ignore_case=True) == \
        'a bb c dd'


def test_2():
    assert repr(colored_words('a bb cc dd ee',
                              {'dd': 'red',
                               'bb': ['bold',
                                      'black',
                                      'bg_blue']})) == \
        r"'a \x1b[1m\x1b[30m\x1b[44mbb\x1b[0m cc \x1b[31mdd\x1b[0m ee'"


def test_3():
    assert replace_all('1 2', {'1': 'f'}, whole_word=True,
                       must_exist=True) == 'f 2'
    assert replace_all(' 1 2', {'1': 'f'}, whole_word=True,
                       must_exist=True) == ' f 2'
    with pytest.raises(ValueError):
        replace_all('1 2', {'3': 'f'},
                    whole_word=True, must_exist=True)


def test_4():
    assert hyphenation('asdf', ['a-s']).replace('\u00AD', '|') == \
        'a|sdf'
    assert hyphenation_cat('asdf', ['a-s', 'f']).replace('\u00AD', '|') == \
        'a|sdf'
    assert hyphenation_cat('asdf', ['a-sd', 'f']).replace('\u00AD', '|') == \
        'a|sd|f'
    assert hyphenation_cat('ab', ['a', 'b']).replace('\u00AD', '|') == \
        'a|b'


def test_5():
    assert mywrap('asd\u00ADbcde', 5) == 'asd\nbcde'
    assert mywrap('a\u00ADbc', 5) == 'abc'


def test_6():
    assert unique_substrings(['123d6', '123ab6']) == ['d', 'ab']
    assert unique_substrings(['123d67', '123ab67']) == ['d', 'ab']
    assert unique_substrings(['123d67', '123ab']) == ['d67', 'ab']
    assert unique_substrings(['123d67', 'ab']) == ['123d67', 'ab']
    assert unique_substrings(['10', '1a_10']) == ['', 'a_']
    assert unique_substrings(['_10', '_1a_10']) == ['', 'a']


def test_7():
    assert exp_latex('%e' % 4.123) == '4.123000'
    assert exp_latex('%+e' % 4.123) == '+4.123000'
    assert exp_latex('%e' % -4.123) == '-4.123000'
    assert exp_latex('%e' % 40.123) == \
        r'4.012300$\cdot$10$^{\text{1}}$'
    assert exp_latex('%e' % 400.123) == \
        r'4.001230$\cdot$10$^{\text{2}}$'
    assert exp_latex('%e' % 0.123) == \
        r'1.230000$\cdot$10$^{\text{-1}}$'
    assert exp_latex('%e' % 0.0123) == \
        r'1.230000$\cdot$10$^{\text{-2}}$'
    assert exp_latex('%+e' % 0.0123) == \
        r'+1.230000$\cdot$10$^{\text{-2}}$'
    assert exp_latex('%+e' % -0.0123) == \
        r'-1.230000$\cdot$10$^{\text{-2}}$'


def test_9():
    import os
    assert 'asdf' not in os.environ
    with pytest.raises(
            ValueError,
            match='^Could not find environment variable "asdf"$'):
        checked_getenv('asdf')
    assert checked_getenv('asdf', default=41, type=int) == 41
    assert checked_getenv('asdf', 41, int) == 41
    os.environ['asdf'] = 'fj'
    assert checked_getenv('asdf') == 'fj'
    with pytest.raises(ValueError,
                       match='^Could not convert environment variable '
                             '"asdf" to type "int"$'):
        checked_getenv('asdf',
                       type=int)
    os.environ['asdf'] = '42'
    assert checked_getenv('asdf', type=int) == 42


def test_11():
    from lacro.run_in import system
    assert system.noninteractive(['pysed', '--debug', 's#a#b#g', '-r'],
                                 return_stdout='utf-8',
                                 stdin='aXX'.encode('utf-8')) == \
        'bXX'
    assert system.noninteractive(['pysed', '--debug', r's#a#b\#c\#d#g', '-r'],
                                 return_stdout='utf-8',
                                 stdin='aXX'.encode('utf-8')) == \
        'b#c#dXX'
    assert system.noninteractive(['pysed', '--debug', r's#a\#b#c\#d#g',
                                  '-r'], return_stdout='utf-8',
                                 stdin='a#bXX'.encode('utf-8')) == \
        'c#dXX'
    assert system.noninteractive(['pysed', '--debug', r's#a\t\#b#c\#d#g',
                                  '-r'], return_stdout='utf-8',
                                 stdin='a\t#bXX'.encode('utf-8')) == \
        'c#dXX'


def test_12():
    assert opt_if_nonnull(
        [(1, None), 2, (3, 'd')]) == ['2', '3 d']


def test_14():
    a = 4
    b = 7
    assert FF('- {a} {b} -') == '- 4 7 -'
    assert FF('- {a} {b} -', dct=dict(a=5, b=9)) == '- 5 9 -'
    assert FF('- {a} {b} -', dct=dict(a=5), udct=dict(b=9)) == '- 5 9 -'
    with pytest.raises(ValueError):
        FF('- {a} {x} -')
    assert F0('- {a} {b} -', a=a, b=b) == '- 4 7 -'
    assert F0('- {a} {b} -', udct=dict(a=a, b=b)) == '- 4 7 -'


def test_16():
    assert textline('test', length=10) == '== test =='
    assert textline('test', length=11) == '== test ==='
    assert textbox('a b c') == '''\
#########
# a b c #
#########'''
    assert textbox('a b\nc') == '''\
#######
# a b #
# c   #
#######'''
    assert textbox('a b\nc', box_chars_1) == '''\
┌─────┐
│ a b │
│ c   │
└─────┘'''


def test_18():
    assert clean_using_dictionary(['a', 'b'], ['s', 'b']) == 'b'
    assert clean_using_dictionary(['s', 't'], ['s', 'b']) == 's'
    assert clean_using_dictionary(['a', 'b'], ['s0', 'b0']) == \
        'b0'
    assert clean_using_dictionary(['s', 't'], ['s0', 'b0']) == \
        's0'
