# -*- coding: utf-8 -*-
from collections import OrderedDict
from typing import KeysView, List, Type, Union


def asserted_of_type(v: object,
                     type: Union[Type, List[Type], KeysView[Type]],
                     msg='',
                     varname=None):
    import builtins
    if builtins.type(type) not in [list, builtins.type({}.keys())]:
        types = [type]
    else:
        types = type
    type_v = builtins.type(v)
    if type_v not in types:
        if callable(msg):
            raise ValueError(msg(v, type))
        if type_v.__name__ not in [t.__name__ for t in types]:
            raise ValueError(
                '%s%s is of type "%s", but it should be one of {%s}' %
                (msg, repr(v) if varname is None else varname,
                 type_v.__name__, ', '.join(t.__name__ for t in types)))
        else:
            raise ValueError(
                '%s%s is of type "%s", but it should be one of {%s}' %
                (msg, repr(v) if varname is None else varname,
                 type_v, ', '.join(map(str, types))))
    return v


def asserted_is_type(type1: Type, type2: Union[Type, List[Type]], msg=''):
    if type(type2) != list:
        if type1 != type2:
            raise ValueError('%sType "%s" differst from "%s"' %
                             (msg, type1.__name__, type2.__name__))
    else:
        if type1 not in type2:
            raise ValueError('%sType "%s" is not one of {%s}' %
                             (msg, type1.__name__,
                              ', '.join(t.__name__ for t in type2)))
    return type1


def asserted_equal(a, b, msg=None, warn=False):
    if a != b:
        if msg is None:
            msg = ('%r != %r' % (a, b))
        if warn:
            from lacro.io.string import print_err
            print_err('WARNING: %s' % msg)
        else:
            raise ValueError(msg)
    return a


# -------------
# - Container -
# -------------


def _assert_array_container_comp(a, b, comp, rtol, atol, convert):
    def _comp2(x, y):
        return _assert_array_container_comp(x, y, comp, rtol, atol, convert)
    import numpy as np

    import lacro.math.npext as ne
    from lacro.array.assert_shape import asserted_shape
    from lacro.collections import Dict2Object
    from lacro.stdext import asserted_of_type
    if convert is not None:
        a = convert(a)
        b = convert(b)
    asserted_of_type(a, type(b))
    tpe = type(a)
    if tpe == Dict2Object:
        set_assert_equal(a.keys(), b.keys())
        assert a.keys() == b.keys()
        for k in a.keys():
            _comp2(a[k], b[k])
    elif tpe in [np.ndarray] + ne.int_types + ne.float_types:
        np.testing.assert_equal(np.asarray(a).dtype.type,
                                np.asarray(b).dtype.type)
        if tpe == np.ndarray:
            asserted_shape(a, b.shape)
        if comp == 'close':
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
        elif comp == 'equal':
            assert rtol is None
            assert atol is None
            np.testing.assert_array_equal(a, b)
        else:
            assert False
    elif tpe in (list, tuple):
        np.testing.assert_equal(len(a), len(b))
        for i in range(len(a)):
            _comp2(a[i], b[i])
    else:
        raise ValueError('Unknown type: %s' % tpe)


def assert_container_allclose(a, b, rtol=1e-07, atol=0,
                              convert=None):
    _assert_array_container_comp(a, b, comp='close', rtol=rtol, atol=atol,
                                 convert=convert)


def assert_container_equal(a, b, convert=None):
    _assert_array_container_comp(a, b, comp='equal', rtol=None, atol=None,
                                 convert=convert)


def assert_count_close(a, b, rcount=0, acount=0, rtol=1e-07, atol=0):
    import numpy as np
    np.testing.assert_equal(np.asarray(a).dtype.type,
                            np.asarray(b).dtype.type)
    diff = np.sum(~np.isclose(a, b, rtol=rtol, atol=atol))
    size = np.broadcast(a, b).size
    assert diff <= max(rcount * size, acount), \
        ('rdiff: %g, adiff: %d' % (diff / size, diff))

# --------
# - List -
# --------


def list_assert_no_duplicates(lst, msg=''):
    from lacro.collections import list_duplicates
    err = list_duplicates(lst)
    if len(err) > 0:
        raise ValueError('%sThe following items occured multiple times: %s' %
                         (msg, err))
    # for k,g in itertools.groupby(keys):
        # if len(list(g)) > 1:
        # raise ValueError('Multiple occurences of key "%s"' % k)
    return lst


def list_assert_empty(lst):
    if len(lst) != 0:
        raise ValueError('List %s should be empty' % lst)


def lists_assert_disjoint(lsts, msg=''):
    from lacro.collections import list_duplicates
    from lacro.iterators import iterables_concat
    from lacro.string.misc import iterable_2_str, to_str

    # rewinding iterator
    lsts = list(lsts)
    err = list_duplicates(iterables_concat(lsts))
    if len(err) > 0:
        msgs = ['%sDuplicates in lists\n%s' %
                (msg,
                 '\n\n\n'.join('Duplicates: %s \nList %5d: %s' %
                               (', '.join(to_str(e) for e in err if e in lst),
                                i, iterable_2_str(lst, to_str=to_str))
                               for i, lst in enumerate(lsts)
                               if len(set(err) & set(lst)) > 0))
                for to_str in [repr, to_str]]
        if msgs[0] == msgs[1]:
            del msgs[1]
        raise ValueError('\n---------\n'.join(msgs))


def list_assert_disjoint(*lsts):
    return lists_assert_disjoint(lsts)


def lists_assert_equal(lsts, error='', check_order=True):
    from lacro.collections import list_intersect, list_minus, lists_union
    from lacro.string.misc import iterable_2_repr
    lsts = list(lsts)
    diff = list_minus(lists_union(lsts, allow_duplicates=True),
                      list_intersect(*lsts))
    if len(diff) != 0:
        raise ValueError('%sLists not equal. Lists: %s\nDifference: %s' %
                         (error,
                          '\n\n'.join(map(iterable_2_repr, lsts)),
                          ', '.join(map(repr, diff))))
    if check_order:
        if len(lsts) == 0:
            return True
        else:
            err = [l for l in lsts[1:] if l != lsts[0]]
            if len(err) > 0:
                raise ValueError('%sOrdering of elements not identical in '
                                 'all lists. Differing lists\n%s' %
                                 (error,
                                  '\n\n'.join(repr(l)
                                              for l in [lsts[0]] + err)))


# -------
# - Set -
# -------


def set_assert_subset(sett, subset, delimiter=', ', max_len=200, msg='',
                      unique=True):
    from lacro.iterators import iterable_ids_of_unique
    from lacro.string.misc import iterable_2_repr, trunc

    # nex = set(subset) - set(sett)
    sett = list(sett)
    if unique:
        list_assert_no_duplicates(sett)
    sett1 = set(sett)
    nex = [l for l in subset if l not in sett1]
    nex = [nex[i] for i in sorted(iterable_ids_of_unique(nex))]
    if len(nex) > 0:
        raise ValueError(('%sElements \n{%s}\n do not exist in \n\n{%s}'
                          if len(nex) > 1 else
                          '%sElement \n%s\n does not exist in \n\n{%s}') %
                         (msg, iterable_2_repr(nex),
                          trunc(iterable_2_repr(sett, delimiter), max_len)))


def set_assert_contains(sett, element, delimiter=', ', max_len=200, msg=''):
    return set_assert_subset(sett, {element},
                             delimiter=delimiter, max_len=max_len, msg=msg)


def set_assert_equal(lst0, lst1):
    set_assert_subset(lst0, lst1)
    set_assert_subset(lst1, lst0)


def unittest_raw_class(klass, attrname=None):
    from contextlib import contextmanager
    from unittest import SkipTest, TestCase

    from lacro.io.string import print_err
    from lacro.string.numeric import int_string_order_key

    if not isinstance(klass, type):
        raise ValueError('Object is not a valid type')

    if klass.__base__ != TestCase:
        raise ValueError(f'Class "{klass.__name__}" does not inherit from '
                         f'unittest.TestCase')

    class SubKlass(klass):

        @contextmanager
        def subTest(self, **kwargs):
            def print_err_for_subtest(msg):
                print_err('%s%s' %
                          ('\n'.join('%s: %s' % (k, v)
                                    for k, v in sorted(kwargs.items())),
                          msg))
            try:
                yield
            except SkipTest as e:
                print_err_for_subtest(f' skipped ... {e}')
            except:
                print_err_for_subtest(' failed ...')
                raise

    a = SubKlass().setUp()
    if attrname is not None:
        if not hasattr(a, attrname):
            raise ValueError(f'Object of type "{klass.__name__}" has no '
                             f'attribute "{attrname}"')
        getattr(a, attrname)()
    else:
        for k in sorted(dir(a), key=int_string_order_key):
            if k.startswith('test'):
                try:
                    getattr(a, k)()
                except SkipTest as e:
                    print_err(f'{k} skipped ... {e}')
    if hasattr(a, 'tearDown'):
        a.tearDown()


def unittest_raw_module(module):
    from unittest import TestCase
    for _, klass in sorted(module.__dict__.items()):
        if isinstance(klass, type) and klass.__base__ == TestCase:
            unittest_raw_class(klass)


def unittest_raw():
    import sys
    from importlib import import_module

    from lacro.inspext.relative_importer import RelativeImporter

    ri = RelativeImporter()

    module = '__main__'
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            def find():
                for prefix in [module + '.', '']:
                    try:
                        return (ri.import_module(prefix + arg),
                                None, None)
                    except ImportError:
                        try:
                            return (None,
                                    ri.import_attribute(prefix + arg),
                                    None)
                        except ImportError:
                            if '.' in arg:
                                attrname = arg.split('.')[-1]
                                klassname = '.'.join(arg.split('.')[:-1])
                                try:
                                    return (None,
                                            ri.import_attribute(prefix +
                                                                klassname),
                                            attrname)
                                except ImportError:
                                    pass
            submodule, klass, attrname = find()
            if submodule is None and klass is None:
                raise ValueError(f'Could not import "{arg}"')
            if submodule is not None:
                unittest_raw_module(submodule)
            if klass is not None:
                unittest_raw_class(klass, attrname)
    else:
        module = import_module(module)
        unittest_raw_module(module)


def pytest_subprocess_env():
    import os
    import shlex
    import sys

    from lacro.path.shlext import qjoin
    if 'PYTEST_ADDOPTS' in os.environ.keys():
        env = dict(os.environ)
        l = shlex.split(env['PYTEST_ADDOPTS'])
        for flag in ['-k', '-m']:
            while flag in l:
                print(f'Removing {flag} flag from ${{PYTEST_ADDOPTS}} in '
                      'subtest', file=sys.stderr)
                i = l.index(flag)
                del l[i]
                if i == len(l):
                    raise ValueError(f'Detected {flag} flag at end of '
                                     '${PYTEST_ADDOPTS}')
                del l[i]
        env['PYTEST_ADDOPTS'] = qjoin(l)
    else:
        env = None
    return env


def verbose_allclose(a, b, **kwargs):
    import numpy as np
    if not np.allclose(a, b, **kwargs):
        raise AssertionError('a: %r\nb: %r' % (a, b))


# ........
# - Dict -
# --------


def dict_assert_function(dct, msg=('What follows are keys and their target '
                                   'values.')):
    """
    surjective and unique
    """
    too_many = OrderedDict((k, v) for k, v in dct.items() if len(v) > 1)
    too_few = OrderedDict((k, v) for k, v in dct.items() if len(v) == 0)
    if len(too_many) > 0 or len(too_few) > 0:
        raise ValueError('%s\n\nThe following keys have multiple target '
                         'values.\n%s\nThe following keys have no target '
                         'values\n%s' %
                         (msg, '\n'.join('%r: %r' % (k, OK)
                                         for k, OK in too_many.items()),
                          '\n'.join('%r: %r' % (k, OK)
                                    for k, OK in too_few.items())))


def dict_assert_injective(dct):
    from lacro.collections import inverted_noninjective_dict
    idct = inverted_noninjective_dict(dct)
    dict_assert_function(idct, msg=('Inverted dictionary is not a function. '
                                    'What follows are keys and their target '
                                    'values in the inverted dictionary.'))
    return dct


def dict_assert_inverse_surjective(dct, msg=''):
    from lacro.string.misc import iterable_2_repr
    too_few = [k for k, v in dct.items() if len(v) == 0]
    if len(too_few) > 0 or len(too_few) > 0:
        raise ValueError('%sThe following keys have no target values\n%s' %
                         (msg, iterable_2_repr(too_few)))
