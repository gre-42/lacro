# -*- coding: utf-8 -*-
import inspect
import itertools
from typing import Callable, Optional

from lacro.assertions import asserted_of_type
from lacro.iterators import eqzip
from lacro.string.misc import to_str as default_to_str


def _construct_f(module, func, args):
    from importlib import import_module
    return getattr(import_module(module), func)(*args)


def _gen_tuple_base(BaseTuple,
                    class_name: str,
                    parent_module_name: str,
                    to_str: Optional[Callable],
                    screpr: Optional[Callable],
                    hashable: bool,
                    is_iterable: bool):
    if parent_module_name is None:
        raise ValueError('Please pass '
                         '"parent_module_name={__name__, "<auto>"}", '
                         'inspect.stack(0) is slow')
    elif parent_module_name == '<auto>':
        frame = inspect.stack(0)[2]
        parent_module_name = inspect.getmodule(frame[0]).__name__
        # alternative implementation used in collections.namedtuple:
        # module = _sys._getframe(1).f_globals.get('__name__', '__main__')

    my_tostr = to_str
    del to_str

    class Tuple(BaseTuple):

        _data: tuple

        def __repr__(self):
            return '%s.C(%s)' % (class_name,
                                 ', '.join(repr(v)
                                           if screpr is None else
                                           screpr(v)
                                           for v in self._data))

        def __str__(self):
            return self.__tostr__()

        def __tostr__(self, **kwargs):
            # convert to tuple because _data is a list if it is not readonly
            if my_tostr is None:
                return '%s.C%s' % (class_name,
                                   default_to_str(tuple(self._data), **kwargs))
            else:
                return my_tostr(self, **kwargs)

        def __reduce__(self):
            return _construct_f, (parent_module_name,
                                  class_name,
                                  (self._data,))

        def __lt__(self, b):
            return self._data.__lt__(self.asserted_same_type(b))

        def __le__(self, b):
            return self._data.__le__(self.asserted_same_type(b))

        def __eq__(self, b):
            return type(b) == Tuple and self._data.__eq__(b._data)

        def __ne__(self, b):
            return not (self == b)

        def __ge__(self, b):
            return self._data.__ge__(self.asserted_same_type(b))

        def __gt__(self, b):
            return self._data.__gt__(self.asserted_same_type(b))

        @staticmethod
        def asserted_same_type(b):
            asserted_of_type(b, Tuple)
            return b

        if is_iterable:
            def __iter__(self):
                return iter(self._data)

        if hashable:
            def __hash__(self):
                return self._data.__hash__()

    Tuple.__name__ = class_name
    Tuple.__module__ = parent_module_name
    return Tuple


def named_dictlike_tuple(class_name, names, types=None, to_str=None,
                         screpr=None, hashable=False, readonly=True,
                         parent_module_name=None, is_iterable=True,
                         has_items=False):
    assert '_data' not in names
    if has_items:
        assert 'keys' not in names
        assert 'values' not in names
        assert 'items' not in names

    if types is not None:
        assert len(types) == len(names)

    class BaseTuple:

        def __init__(self, data):
            data = (tuple if readonly else list)(data)
            if len(names) != len(data):
                raise ValueError('len(names) != len(data), %d != %d\n\n'
                                 '%s\n\n%s' % (len(names),
                                               len(data),
                                               names, data))
            assert len(names) == len(data)
            if types is not None:
                for d, t in eqzip(data, types):
                    if t is not None:
                        asserted_of_type(d, t)
            self._data = data

        @staticmethod
        def C(*data, **kwargs):
            data = tuple(data)
            return ResultTuple(itertools.chain(data,
                                               (kwargs[n]
                                                for n in names[len(data):])))

        if has_items:
            @staticmethod
            def keys():
                return names

            def values(self):
                return self._data

            def items(self):
                return eqzip(names, self._data)

        def __setattr__(self, n, v):
            if n in ('_data',) + ('keys', 'values', 'items') * has_items:
                object.__setattr__(self, n, v)
            else:
                if n not in names:
                    raise ValueError('%r not in %r' % (n, names))
                if readonly:
                    raise ValueError('%s created as readonly' % class_name)
                self._data[names.index(n)] = v

        def __getattr__(self, n):
            if n in ('_data',) + ('keys', 'values', 'items') * has_items:
                object.__getattribute__(self, n)
            else:
                if n not in names:
                    raise AttributeError('%r not in %r' % (n, names))
                return self._data[names.index(n)]

    ResultTuple = _gen_tuple_base(BaseTuple, class_name, parent_module_name,
                                  to_str, screpr, hashable, is_iterable)
    return ResultTuple


def dictlike_tuple(class_name, has_items_is_iterable=False, to_str=None,
                   screpr=None, hashable=False, parent_module_name=None):
    class BaseTuple:

        def __init__(self, data=()):
            self._data = tuple(data)

        @staticmethod
        def C(*data):
            return ResultTuple(data)

        def __add__(self, rhs):
            self.asserted_same_type(rhs)
            return ResultTuple(self._data + rhs._data)

        def __getitem__(self, key):
            if type(key) == slice:
                return ResultTuple(self._data[key])
            else:
                return self._data[key]

        if has_items_is_iterable:
            def values(self):
                return (v for i, v in self._data)

            def items(self):
                return self._data

        # def __getnewargs__(self):
            # print('args')
            # return (self._data,)

        # def __getstate__(self):
            # print('gets')
            # return self._data

        # def __setstate__(self, state):
            # print('sets')
            # self._data = state

        def __len__(self):
            return len(self._data)

        @staticmethod
        def asserted_same_type(b):
            asserted_of_type(b, ResultTuple)
            return b

    ResultTuple = _gen_tuple_base(BaseTuple, class_name, parent_module_name,
                                  to_str, screpr, hashable,
                                  has_items_is_iterable)
    return ResultTuple
