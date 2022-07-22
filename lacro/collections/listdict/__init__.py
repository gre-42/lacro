# -*- coding: utf-8 -*-

"""
This module implements something like numpy "structured arrays" or the
pandas DataFrame.
http://docs.scipy.org/doc/numpy/user/basics.rec.html
however, dictlist has the advantage of allowing for arbitrary
length arrays and combining all typical tasks (duplicate elimination,
sorting, formating output) into a single package
"""

from ._collections.cache import BaseCachedListDicts, BaseCachedReprIo
from ._collections.group import group_child_count, group_leaf_count
from ._collections.multiple import concat, hconcat, join, product
from ._collections.onetwo import (ListDict, ListGroup, ListGroup2,
                                  ReadonlyListDict, flatten_groups,
                                  unflatten_ungroup, ungroup, unique_metas)
from ._elements.column_descriptor import (ColumnDescriptor, SetOfTypes,
                                          SubsetOfType, UnknownType,
                                          assert_elemtype_valid,
                                          assure_no_subset_of_type2,
                                          assure_subset_of_type2,
                                          is_set_of_types, is_subset_of_type,
                                          t2c, to_dtype, ts2c, unify_dspecs,
                                          v2c)
from ._elements.values import (ObjectNull, assert_deep_consistent, closest_key,
                               date_distance, float_null, int_null, is_null,
                               object_null, string_null)
from ._navigate.group_saver import SaveGroupAsList, SaveGroupAsTree

__all__ = [
    'SubsetOfType', 'SetOfTypes', 'UnknownType', 'ColumnDescriptor',
    't2c', 'ts2c', 'v2c',
    'unify_dspecs',
    'is_subset_of_type', 'assure_no_subset_of_type2',
    'assure_subset_of_type2', 'is_set_of_types', 'to_dtype',
    'assert_elemtype_valid',
    'concat', 'hconcat', 'product', 'join',
    'ListDict', 'ReadonlyListDict', 'ListGroup', 'ListGroup2',
    'unique_metas', 'flatten_groups', 'unflatten_ungroup',
    'ungroup', 'group_leaf_count', 'group_child_count',
    'BaseCachedListDicts', 'BaseCachedReprIo',
    'ObjectNull', 'object_null', 'int_null', 'string_null',
    'float_null', 'is_null',
    'date_distance', 'closest_key', 'assert_deep_consistent',
    'SaveGroupAsList', 'SaveGroupAsTree']
