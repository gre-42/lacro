#!/usr/bin/env pytest
# -*- coding: utf-8 -*-
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from lacro.inspext.app import init_pytest_suite
from lacro.iterators import eqzip
from lacro.sort.array import (iter_argmin, map_maxsim_rowsX_to_rowsY,
                               searchunsorted, sort_rowsY_like_rowsX,
                               unique_2d, unique_diff)

init_pytest_suite()


a = np.array([
    [ 0, 0,  10],  # ?
    [ 0, 2,  -2],  # b
    [-2, 2.1, 0]])  # c

b = np.array([
    [ 1, -1,   0],  # c
    [ 0, 1,  -1]])  # b

c = np.array([
    [ 1, -1,   0],  # c
    [ 1, -1,   0],  # c
    [ 0, 1,  -1]])  # b


def test_sort_rows_0():
    M1_s, M1_r = sort_rowsY_like_rowsX(np.array([[ 1, -1,   0],
                                                 [ 0, 1,  -1]]),
                                       np.array([[ 0, 0,  10],
                                                 [ 0, 2,  -2],
                                                 [-2, 2.1, 0]]))
    assert_allclose(M1_s, [[2, -2.1, 0], [0,  2,  -2]])
    assert_allclose(M1_r, [[0,  0,  10]])


def test_sort_rows_1():
    for aa, bb in eqzip(map_maxsim_rowsX_to_rowsY(b, a),
                        (np.array([2, 1]),
                            [0],
                            np.array([-0.99990087,  1]))):
        assert_allclose(aa, bb)
    for aa, bb in eqzip(sort_rowsY_like_rowsX(b, a),
                        (np.array([[2, -2.1, -0],
                                   [0,  2  , -2]]),
                            np.array([[0,  0  , 10]]))):
        assert_array_equal(aa, bb)
    with pytest.raises(ValueError):
        sort_rowsY_like_rowsX(c, a)


def test_searchunsorted():
    def cmp(a, b, c, side='left'):
        c = np.array(c)
        assert_allclose(searchunsorted(np.array(a),
                                       np.array(b),
                                       side=side), c)
        assert_allclose(searchunsorted(np.array(a)[::-1],
                                       np.array(b), side=side),
                        ((len(a) - 1 - c) * (c != len(a)) +
                         c * (c == len(a))))
    cmp([1, 2, 3, 4], [1, 2, 3, 4], [0, 1, 2, 3])
    cmp([1, 2, 3, 4], [4, 3, 2, 1], [3, 2, 1, 0])
    cmp([1, 2, 3, 4], [4, 3, 2, 1, 1], [3, 2, 1, 0, 0])
    cmp([1, 2, 3, 4], [4, 3, 2, 1, 2], [3, 2, 1, 0, 1])
    cmp([1, 2, 3, 4], [3, 1.5, 3], [2, 1, 2])
    cmp([1, 2, 3, 4], [3, 3, 1.5], [2, 2, 1])
    cmp([1, 2, 3, 4], [4.1], [4])

    cmp([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], side='right')
    cmp([1, 2, 3, 4], [4, 3, 2, 1], [4, 3, 2, 1], side='right')
    assert_allclose(searchunsorted(np.array([4, 3, 2, 1]),
                                   np.array([3, 1.5, 3])), [1, 2, 1])


def test_unique_2d_0():
    M = np.array([[1, 2, 1], [0, 4, 1], [1, 2, 1], [1, 2, 1]]).astype(str)
    un, ui, inv, eqvs = unique_2d(M, return_index=True,
                                  return_inverse=True, return_equals=True)
    assert_array_equal(un, [list('041'), list('121')])
    assert_array_equal(ui, [[0, 1, 0], [1, 0, 0]])
    assert_array_equal(
        inv, [[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]])
    assert_array_equal(eqvs, [[False, True, False, False],
                              [True, False, True, True]])


def test_unique_2d_1():
    M = np.array(['a', 'z', 'c', 'c', 'd'])[:, None]
    un, inv = unique_2d(M, return_inverse=True)
    assert_array_equal(un, [['a'], ['c'], ['d'], ['z']])
    assert_array_equal(inv, [[0], [3], [1], [1], [2]])


def test_unique_2d_2():
    M = np.empty((4, 0))
    un, ui, inv, eqvs = unique_2d(M, return_index=True,
                                  return_inverse=True, return_equals=True)
    assert_array_equal(un, np.empty((1, 0)))
    assert_array_equal(ui, np.empty((1, 0)))
    assert_array_equal(inv, np.empty((4, 0)))
    assert_array_equal(eqvs, [[True, True, True, True]])


def test_argmin_iter():
    A = np.random.random((3, 4, 5))
    assert_array_equal(np.argmin(A, axis=0), iter_argmin(A))
    assert_array_equal(
        iter_argmin((-i * np.ones((2, 3)) * np.array([[1, 1, 1], [1, -1, 1]])
                 for i in range(6))), [[5, 5, 5], [5, 0, 5]])


def test_unique_diff():
    with pytest.raises(AssertionError):
        unique_diff(np.array([-1, 0, 2]))
    assert_array_equal(unique_diff(np.array([-2, 0, 2])), 2)
    assert_array_equal(unique_diff(np.array([-2, 0, 2]) + 1.5), 2)
