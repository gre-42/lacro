#!/usr/bin/env pytest
# -*- coding: utf-8 -*-
from lacro.inspext.app import init_pytest_suite
from lacro.path.shlext import djoin, dquote, qjoin, qquote

init_pytest_suite()


def test_quote():
    assert qquote("'asd'") == '\'\'"\'"\'asd\'"\'"\'\''
    assert dquote("'asd'") == '"\'asd\'"'


def test_join():
    assert qjoin(["'asd'", "'bsd'"]) == (
        '\'\'"\'"\'asd\'"\'"\'\' '
        '\'\'"\'"\'bsd\'"\'"\'\'')
    assert djoin(["'asd'", "'bsd'"]) == (
        '"\'asd\'" '
        '"\'bsd\'"')


def test_join_sep():
    sep = '---'
    assert qjoin(["'asd'", "'bsd'"], sep) == (
        '\'\'"\'"\'asd\'"\'"\'\'' + sep
        + '\'\'"\'"\'bsd\'"\'"\'\'')
    assert djoin(["'asd'", "'bsd'"], sep) == (
        '"\'asd\'"' + sep
        + '"\'bsd\'"')
