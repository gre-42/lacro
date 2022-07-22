# -*- coding: utf-8 -*-
import copy
from typing import Iterable, Optional, Type, Union

import numpy as np

import lacro.collections.dictlike_tuples as gen
import lacro.math.npext as ne
from lacro.array import array_of_shape, asserted_shape
from lacro.assertions import asserted_of_type
from lacro.collections import ArrayEqual, dict_union, list_union
from lacro.iterators import unique_value
from lacro.string.misc import to_repr, to_str

# ----------------------
# - column despriptors -
# ----------------------

SubsetOfType = gen.dictlike_tuple('SubsetOfType', to_str=lambda v, **k: 'S(%s)' % ', '.join(
    map(to_str, v)), screpr=ne.screpr, hashable=True, parent_module_name=__name__)
SetOfTypes = gen.dictlike_tuple('SetOfTypes', to_str=lambda v, **k: 'ST(%s)' % ', '.join(
    map(to_str, v)), screpr=ne.screpr, hashable=True, parent_module_name=__name__)

# class DictWithKeys(list):
# def __repr__(self):
# return 'K(%s)' % list.__repr__(self)


class UnknownType:
    pass


_MetaType = Optional[Union[dict, np.ndarray]]


def _meta_union(a, b) -> _MetaType:
    return dict_union(a, b, allow_duplicates='nearly')


class ColumnDescriptor:

    def __init__(self,
                 ttype,
                 basetype,
                 elemtype: Type,
                 meta: _MetaType,
                 shape1) -> None:
        self.ttype = ttype
        self.basetype = basetype
        self.elemtype = elemtype
        self.meta: _MetaType = meta
        self.shape1 = shape1

    def __to_repr__(self, **kwargs):
        return ('lacro.collections.listdict.' * kwargs['add_module'] +
                'ColumnDescriptor(%s, %s, %s, %s, %s)' % (
                    to_repr(self.ttype, **kwargs),
                    to_repr(self.basetype, **kwargs),
                    to_repr(self.elemtype, **kwargs),
                    to_repr(self.meta, **kwargs),
                    to_repr(self.shape1, **kwargs)))

    def __repr__(self):
        return to_repr(self, type_repr=True)

    def __str__(self):
        # return 'CD(ttype=%s, basetype=%s, elemtype=%s)' % (self.ttype,
        #                                                    self.basetype,
        #                                                    self.elemtype)
        # return 'CD(%s, %s, %s)' % (self.ttype, self.basetype, self.elemtype)
        # return 'CD'
        return to_str(self.ttype, type_repr=True)

    def __hash__(self):
        return hash((self.ttype, self.basetype, self.elemtype, self.meta,
                     self.shape1))

    def __eq__(self, other):
        return ((self.ttype, self.basetype, self.elemtype,
                 ArrayEqual(self.meta), self.shape1) ==
                (other.ttype, other.basetype, other.elemtype,
                 other.meta, other.shape1))

    def unify(self, other, unite_subset_of_type=False, unite_metas=False,
              k=None):
        if self.elemtype != other.elemtype:
            raise ValueError(
                'elemtypes of joined keys do not match in key %r, %s != %s' %
                (k, self.elemtype, other.elemtype))
        if self.basetype != other.basetype:
            raise ValueError(
                'basetypes of joined keys do not match in key %r, %s != %s' %
                (k, self.basetype, other.basetype))
        if self.ttype != other.ttype:
            if (unite_subset_of_type and
                is_subset_of_type(self.ttype) and
                    is_subset_of_type(other.ttype)):
                self.ttype = SubsetOfType(list_union(self.ttype,
                                                     other.ttype,
                                                     allow_duplicates=True,
                                                     order='keep'))
            else:
                raise ValueError(
                    'ttypes of joined keys do not match in key %r, %s != %s' %
                    (k, self.ttype, other.ttype))
        if self.shape1 != other.shape1:
            raise ValueError(
                'shapes of joined keys do not match in key %r, %s != %s' %
                (k, self.shape1, other.shape1))
        if np.any(self.meta != other.meta):
            if unite_metas:
                if self.meta is None:
                    self.meta = other.meta
                elif other.meta is not None:
                    if len(self.shape1) == 0:
                        self.meta = _meta_union(self.meta, other.meta)
                    elif len(self.shape1) == 1:
                        asserted_shape(self.meta, self.shape1)
                        asserted_shape(other.meta, other.shape1)
                        self.meta: _MetaType = np.array(
                            [_meta_union(asserted_of_type(r, dict),
                                         asserted_of_type(o, dict))
                             for r, o in zip(self.meta, other.meta)],
                            dtype=object)
            else:
                raise ValueError(
                    'metas of joined keys do not match in key %r, %s != %s' %
                    (k, self.meta, other.meta))

    def astype(self, type):
        if is_subset_of_type(self.ttype):
            assert self.basetype != np.ndarray
            return v2c(self.ttype, type, type, convert=True)
        else:
            return t2c(self.basetype if self.basetype == np.ndarray else type,
                       type, self.meta, self.shape1)


def unify_dspecs(*args, unite_subset_of_type=False, unite_metas=False):
    res = copy.copy(args[0])
    for a in args[1:]:
        for k in a.keys():
            if k in res.keys():
                if res[k] != a[k]:
                    res[k] = copy.copy(res[k])
                    res[k].unify(a[k], unite_subset_of_type, unite_metas, k)
            else:
                res[k] = a[k]

    return res


def v2c(values: Iterable,
        basetype=None,
        elemtype: Optional[Type] = None,
        convert=False) -> ColumnDescriptor:
    if convert:
        assert basetype is not None
        values = [basetype(v) for v in values]
    else:
        values = list(values)
    if len(set(values)) != len(values):
        raise ValueError('Column descriptor values contain duplicates')
    return ColumnDescriptor(
        ttype=SubsetOfType(values),
        basetype=(unique_value(type(v) for v in values) if basetype is None
                  else basetype),
        elemtype=(unique_value(type(v) for v in values) if elemtype is None
                  else elemtype),
        meta=None,
        shape1=())


def t2c(type: Type,
        elemtype: Optional[Type] = None,
        meta: _MetaType = None,
        shape1=()) -> ColumnDescriptor:
    if (type == np.ndarray) and (elemtype is None):
        raise ValueError('Type is ndarray, but no "elemtype" was given.')
    return ColumnDescriptor(ttype=type,
                            basetype=type,
                            elemtype=type if elemtype is None else elemtype,
                            meta=meta,
                            shape1=shape1)


def ts2c(types, *args, **kwargs) -> ColumnDescriptor:
    return t2c(SetOfTypes(types), *args, **kwargs)


def is_subset_of_type(ttype):
    assert isinstance(ttype, SubsetOfType) == (type(ttype) == SubsetOfType)
    return isinstance(ttype, SubsetOfType)
    # return hasattr(ttype, '__getitem__') and type(ttype) != type


def assure_no_subset_of_type2(dspecs, key):
    if is_subset_of_type(dspecs[key].ttype):
        raise ValueError('"%s"\'s ttype is of type "SubsetOfType".' % key)
    return dspecs[key].ttype


def assure_subset_of_type2(dspecs, key):
    if not is_subset_of_type(dspecs[key].ttype):
        raise ValueError('"%s"\'s ttype is not of type "SubsetOfType".' % key)
    return dspecs[key].ttype


def is_set_of_types(ttype):
    assert isinstance(ttype, SetOfTypes) == (type(ttype) == SetOfTypes)
    return isinstance(ttype, SetOfTypes)


def to_dtype(elemtype, k='dummy'):
    assert_elemtype_valid(elemtype, k)
    if elemtype in [np.int32, np.int64, np.float32, np.float64]:
        return elemtype
    else:
        return np.object_


def assert_elemtype_valid(elemtype, k):
    from .values import assert_valuetype_valid
    assert_valuetype_valid(elemtype, k)


def array_for_coldes(ar, nrows, coldes, key):
    return array_of_shape(ar, shape=(nrows,) + coldes.shape1,
                          dtype=to_dtype(coldes.elemtype, key),
                          msg='Key %r\n' % (key,))
