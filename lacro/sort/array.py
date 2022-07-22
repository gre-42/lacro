# -*- coding: utf-8 -*-
import numpy as np

import lacro.stdext as se
from lacro.array.assert_shape import asserted_shape
from lacro.array.modify_shape import modified_shape


def ranking(a, axis=-1):
    # http://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy
    return np.argsort(np.argsort(a, axis=axis), axis=axis)


def searchunsorted(ids1, ids2, side='left'):
    """
    side = 'right': return the first index of "ids1" where "ids2" is larger than "ids1"
    this means that searchunsorted([1,2,3,4], [1,2,3,4]) = [1,2,3,-1]
    side = 'left': result[with ids2 > max(ids1)] = -1
    """
    asserted_shape(ids1, (None,))
    asserted_shape(ids2, (None,))
    a1 = np.argsort(ids1)
    r1 = np.argsort(a1)
    r1 = np.concatenate([r1, [len(r1)]])
    res = r1[np.searchsorted(ids1[a1], ids2, side=side)]
    return res


def map_maxsim_rowsX_to_rowsY(X, Y, threshold=0, metric='correlation',
                              get_warnings=False, errors=[]):
    """
    returns two arrays of length X.size[0]
    """

    from lacro.statistics.similarity import cdist2
    C = 1 - cdist2(X, Y, metric=metric)  # cdist computes 1-correlation

    warnings = {}

    # check relation properties:
    axis1 = np.sum(np.abs(C) >= threshold, axis=1)
    # X -> Y is fully defined:
    if not np.all(axis1 >= 1):
        warnings['not fully defined'] = (
            'Relation X->Y is not fully defined: The following element(s) in '
            'X are not related to any element of Y.', np.where(axis1 < 1)[0])
    # X -> Y is uniquely defined
    if not np.all(axis1 <= 1):
        warnings['not uniquely defined'] = (
            'Relation X->Y is not uniquely defined: The following element(s) '
            'in X are related to more than one element of Y.',
            np.where(axis1 > 1)[0])

    axis0 = np.sum(np.abs(C) >= threshold, axis=0)
    # X -> M3 is injective
    if not np.all(axis0 <= 1):
        warnings['not injective'] = (
            'Relation X->Y is not injective: The following element(s) in Y '
            'are related to more than one element of X.',
            np.where(axis0 > 1)[0])
    # X -> Y is surjective
    if not np.all(axis0 >= 1):
        warnings['not surjective'] = (
            'Relation X->Y is not surjective: The following element(s) in Y '
            'are related not to any element of X', np.where(axis0 < 1)[0])

    # index matrix
    x2y = np.argmax(np.abs(C), axis=1)
    x2y[axis1 == 0] = -1

    x2y_complem = sorted(list(set(range(0, Y.shape[0])) -
                              set(x2y[axis1 != 0])))

    if len(set(x2y)) != len(x2y):
        warnings['not injective weak'] = (
            'Could not sort columns uniquely, the following rows were found '
            'in the reference more than once: %s' % se.list_duplicates(x2y))

    # similarity matrix
    x2sim = np.array([C[c, x2y[c]] if x2y[c] >= 0 else 0
                      for c in range(0, C.shape[0])])

    for err in errors:
        if err in warnings.keys():
            raise ValueError(warnings[err])

    return ([x2y, x2y_complem, x2sim, warnings] if get_warnings else
            [x2y, x2y_complem, x2sim])


def sort_rowsY_like_rowsX(X, Y, debug=False):
    """Sorts a data array according to a reference array.

    Sorts data (e.g. resting state ICA components) according to a
    reference (e.g. RSN atlas). Uses maximum absolute correlation to determine
    associated components.

    Parameters
    ----------
    Y : array_like
        Data to be sorted according to `X`, e.g. components from an ICA
        decomposition.
    X : array_like
        References, e.g. atlas.
    debug : bool, optional
        If True, the function returns an array with permutation and
        max. correlation. Defaults to False.

    Returns
    -------
    M1_s : numpy.ndarray
        Sorted components in `M1_s` in the right order (multiplied by -1 if
        correlation was negative)
    M1_r : numpy.ndarray
        Components from `Y` that weren't associated with any of the reference
        components will be in `M1_r`

    Raises
    ------
    ValueError:
        If # of reference components > # of data components
        OR if sorting cannot be done uniquely, i.e. same data
        component is associated with more than one reference
        component.

    Examples
    --------
    >>> sort_rowsY_like_rowsX(
            np.array([[ 1,-1,   0],
                      [ 0, 1,  -1]]),
            np.array([[ 0, 0,  10],
                      [ 0, 2,  -2],
                      [-2, 2.1, 0]]))
    [np.array([[2, -2.1, 0],
               [0,  2,  -2]]),
     np.array([[0,  0,  10]])]

    """

    [x2y, x2y_complem, x2sim] = map_maxsim_rowsX_to_rowsY(
        X, Y, errors=['not injective weak'])

    [M1_s, M1_r] = [np.sign(x2sim)[:, None] * Y[x2y, :], Y[x2y_complem, :]]

    if debug:
        return M1_s, M1_r, x2y, x2sim
    else:
        return M1_s, M1_r


def unique_ordered_1D(a, return_index=False, return_inverse=False,
                      ordered=True):
    assert return_index == False and return_inverse == True
    asserted_shape(a, (None,))
    if not ordered:
        return np.unique(a, return_inverse=True)
    else:
        # http://stackoverflow.com/questions/15637336/numpy-unique-with-order-preserved
        u, ind, inv = np.unique(a, return_index=True, return_inverse=True)
        sortid = np.argsort(ind)
        #u[inv] = a
        # u[s][?(inv)] = a

        # print(a)
        # print(u[inv])

        # inv(u) = a
        # F(s(u)) = a
        # inv(s^-1(s(u))) = a
        # => F(x) = inv(s^-1(x)) = inv*(s^-1)*x = x[(s^-1)[inv]]

        #iso = se.inverted_injective_dict(dict(enumerate(sortid)))
        #iso = np.array([iso[i] for i in range(len(iso))])
        iso = np.argsort(sortid)

        # print(iso[sortid])
        # print(sortid[iso])
        return u[sortid], iso[inv]


def unique_2d(a, return_index=False, return_inverse=False, return_equals=False,
              ordered=False):
    asserted_shape(a, (None, None))
    if a.shape[1] == 0:
        un = np.empty((min(1, a.shape[0]), 0), dtype=a.dtype)
        ui = np.empty((un.shape[0], 0), dtype=np.int64)
        inv = np.empty(a.shape, dtype=np.int64)
        eq = np.full((un.shape[0], a.shape[0]), True, dtype=bool)
    else:
        us, inv = zip(*[unique_ordered_1D(c, return_index=False,
                                        return_inverse=True, ordered=ordered)
                        for c in a.T])
        us = np.array(us).T
        inv = np.array(inv).T
        # print(inv)
        ui = np.array(sorted(set(tuple(r) for r in inv)))
        un = np.array([s[i] for s, i in se.eqzip(us.T, ui.T)]).T
        eq = np.array([np.all(inv == i, axis=1) for i in ui])
    res = (un,)
    if return_index:
        res = res + (ui,)
    if return_inverse:
        res = res + (inv,)
    if return_equals:
        res = res + (eq,)
    return res if len(res) > 1 else res[0]


def unique_Nd(a):
    u, inv = np.unique(a, return_inverse=True)
    res = np.empty(shape=a.shape, dtype=inv.dtype)
    return u, modified_shape(inv, a.shape)


def unique_diff(v):
    asserted_shape(v, (-2,))
    d = np.diff(v)
    m = (v[-1] - v[0]) / (len(v) - 1)
    assert np.max(np.abs(d - m)) < 3 * np.spacing(np.max(v))
    return m


def iter_argmin(iterable):
    """Equivalent to ``np.argmin(np.array(iterable), axis=0)``.

    Args:
        iterable: The iterable to compute the argmin of.

    Example:
        >>> A = np.random.random((3, 4, 5))
        >>> np.testing.assert_array_equal(np.argmin(A, axis=0), iter_argmin(A))

    """
    i_res = None
    for i, v in enumerate(iterable):
        if i_res is None:
            i_res = np.zeros(v.shape, dtype=np.int64)
            v_res = v
        else:
            ids = (v < v_res)
            i_res[ids] = i
            v_res[ids] = v[ids]
    return i_res
