# -*- coding: utf-8 -*-
import os.path
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict

from lacro.decorators import cached_property
from lacro.string.misc import checked_match
from lacro.string.numeric import i_maxnum_2_str, maxnum_2_strlen


def tuple_types_torepr(a, maxrec=5):
    assert maxrec >= 0
    if type(a) == tuple:
        return '(%s)' % ', '.join(tuple_types_torepr(v, maxrec - 1) for v in a)
    else:
        from lacro.string.misc import to_repr
        return to_repr(type(a), type_repr=True)


def tuple_allclose(a, b, maxrec=5):
    assert maxrec >= 0
    if type(a) == tuple or type(b) == tuple:
        if type(a) != type(b):
            return False
        if len(a) != len(b):
            return False
        return all(tuple_allclose(a[i], b[i], maxrec - 1)
                   for i in range(len(a)))
    elif hasattr(a, 'items') or hasattr(b, 'items'):
        if type(a) != type(b):
            return False
        return (tuple_allclose(tuple(a.keys()), tuple(b.keys())) and
                tuple_allclose(tuple(a.values()), tuple(b.values())))
    else:
        import numpy as np

        from lacro.math.npext import float_types, int_types
        # noinspection PyUnresolvedReferences
        c = np.result_type(type(a), type(b)).type
        if c in int_types + [np.str_]:
            return a == b
        elif c in float_types:
            return np.isclose(a, b)
        else:
            raise ValueError('Unknown type: result_type(%r, %r) = %r' %
                             (type(a), type(b), c))


def _picompress(obj):
    from pickle import dumps
    from zlib import compress
    return compress(dumps(obj, -1))


def _pidecompress(bytes):
    from pickle import loads
    from zlib import decompress
    return loads(decompress(bytes))


class _CompressedDictBase(ABC):

    @classmethod
    def from_compressed_items(cls, compressed_items):
        self = cls()
        self._dct.update(compressed_items)
        return self

    @classmethod
    def from_decompressed_items(cls, decompressed_items):
        self = cls()
        self._dct.update((k, _picompress(v)) for k, v in decompressed_items)
        return self

    def compressed_items(self):
        return self._dct.items()

    def decompressed_items(self):
        return ((k, _pidecompress(v)) for k, v in self._dct.items())

    def keys(self):
        return self._dct.keys()

    def __getitem__(self, key):
        return _pidecompress(self._dct[key])

    def __setitem__(self, key, value):
        self._dct[key] = _picompress(value)

    def __delitem__(self, key):
        del self._dct[key]

    def __repr__(self):
        return '%s.from_decompressed_items(%r)' % (
            self.__class__.__name__, list(self.decompressed_items()))

    @property
    @abstractmethod
    def _dct(self):
        pass


class CompressedDict(_CompressedDictBase):

    @cached_property
    def _dct(self):
        return dict()


class CompressedOrderedDict(_CompressedDictBase):

    @cached_property
    def _dct(self):
        return OrderedDict()


class _BaseCache(ABC):

    def __init__(self, filename, verbose, close_duplicates):
        self._verbose = verbose
        self._filename = filename
        self._close_duplicates = close_duplicates
        self._dct: Dict = None
        self._load()

    def keys(self):
        return self._dct.keys()

    def _rewrite(self, key, verbose):
        if (self._close_duplicates == 'ignore') or (key in self._dct.keys()):
            return
        sim = [k for k in self._dct.keys() if tuple_allclose(k, key)]
        if len(sim) == 0:
            return
        if self._close_duplicates == 'raise':
            raise ValueError('Found duplicates for key\n%s' %
                             '\n'.join((repr(k) + '\n' +
                                        tuple_types_torepr(key))
                                       for k in [key] + sim))
        elif self._close_duplicates == 'rewrite':
            if len(sim) > 1:
                raise ValueError('Found multiple duplicates for key %r\n%s' %
                                 (key, '\n'.join(map(repr, sim))))
            if verbose:
                print('rewriting, old: %r, new: %r' % (sim[0], key))
            self._dct[key] = self._dct[sim[0]]
            del self._dct[sim[0]]
            self._save()
        else:
            raise ValueError("close_duplicates must be either 'ignore', "
                             "'raise' or 'rewrite'")

    def get(self, key, operation, verbose=None, to_str=repr):
        from lacro.io.string import print_err
        if verbose is None:
            verbose = self._verbose
        self._rewrite(key, verbose)
        if key not in self._dct.keys():
            if verbose:
                # up to date:
                # not cached:
                # cached    :
                print_err(('not cached: %s' % (to_str(key),)) +
                          ((', filename: "%s"' % self._filename) *
                           (self._filename is not None)))
            if operation is None:
                raise KeyError(key)
            self._dct[key] = operation()
            self._save()
        elif verbose:
            print_err(('cached    : %s' % (to_str(key),)) +
                      ((', filename: "%s"' % self._filename) *
                       (self._filename is not None)))
        return self._dct[key]

    def keys_tostr(self, to_str=repr):
        return '\n'.join('%s: %r\n%s  %s' %
                         (i_maxnum_2_str(i, len(self._dct.keys())),
                          to_str(k),
                          ' ' * maxnum_2_strlen(len(self._dct.keys())),
                          tuple_types_torepr(k))
                         for i, k in enumerate(self._dct.keys()))

    def __getitem__(self, key):
        return self.get(key, operation=None)

    @abstractmethod
    def _load(self):
        pass

    @abstractmethod
    def _save(self):
        pass


class Cache(_BaseCache):

    def __init__(self, filename=None, verbose=False,
                 close_duplicates='ignore'):
        _BaseCache.__init__(self, filename, verbose, close_duplicates)

    def _load(self):
        if (self._filename is not None) and os.path.exists(self._filename):
            from lacro.io.string import load_from_pickle
            self._dct = load_from_pickle(self._filename)
        else:
            self._dct = OrderedDict()

    def _save(self):
        if self._filename is not None:
            from lacro.io.string import save_to_pickle
            save_to_pickle(self._filename, self._dct)

    def transform_items(self, func):
        self._dct = OrderedDict([(k, v) for k, v in func(self._dct.items())])
        self._save()


class CompressedCache(_BaseCache):

    def __init__(self, filename=None, verbose=False,
                 close_duplicates='ignore'):
        _BaseCache.__init__(self, filename, verbose, close_duplicates)

    def _load(self):
        if self._filename is not None:
            from lacro.io.string import load_from_pickle
            short = checked_match(r'^(.*)\.lgz$', self._filename).group(1)
            if os.path.exists(self._filename):
                self._dct = CompressedOrderedDict.from_compressed_items(
                    load_from_pickle(self._filename))
            elif os.path.exists(short):
                self._dct = CompressedOrderedDict.from_decompressed_items(
                    load_from_pickle(short).items())
                self._save()
                os.unlink(short)
            else:
                self._dct = CompressedOrderedDict()
        else:
            self._dct = CompressedOrderedDict()

    def _save(self):
        if self._filename is not None:
            from lacro.io.string import save_to_pickle
            save_to_pickle(self._filename, list(self._dct.compressed_items()))

    def transform_items(self, func):
        self._dct = CompressedOrderedDict.from_decompressed_items(
            (k, v) for k, v in func(self._dct.decompressed_items()))
        self._save()
