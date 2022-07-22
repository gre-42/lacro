#!/usr/bin/env pytest
# -*- coding: utf-8 -*-
from lacro.inspext.app import init_pytest_suite
from lacro.path.pathmod import included_parent_dirs_sorted_relative

init_pytest_suite()


def test_0():
    assert included_parent_dirs_sorted_relative(['a/b/c', '11/22/33']) == \
        ['11', '11/22', '11/22/33', 'a', 'a/b', 'a/b/c']
