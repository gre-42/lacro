# -*- coding: utf-8 -*-
import copy
import csv
import itertools
import os.path
import re
import sys
from collections import OrderedDict
from typing import (Any, Callable, Dict, Hashable, Iterable, Iterator, List,
                    Optional, Sequence, Tuple, Union, cast)

import numpy as np

import lacro.math.npext as ne
from lacro.array import (array_of_shape, array_torepr, array_totuple,
                         asserted_shape, default_concatenate,
                         mask_negated_indices, repeatE)
from lacro.assertions import (asserted_is_type, asserted_of_type,
                              dict_assert_injective, list_assert_disjoint,
                              list_assert_no_duplicates, lists_assert_disjoint,
                              lists_assert_equal, set_assert_subset)
from lacro.collections import (GetattrHashableOrderedDict, HashableDict,
                               HashableOrderedDict, array_ids_join, dict_get,
                               dict_intersect, dict_union, dict_unique_value,
                               dict_unite, dict_update, items_2_str,
                               items_unique, list_intersect, list_minus,
                               list_sorted, list_union, lists_concat, maxdef)
from lacro.decorators import indexable_function
from lacro.inspext.misc import reraise
from lacro.io.string import filtered_open, open_chmod, print_err
from lacro.iterators import (eqzip, eqziplist, iterable_ids_of_change,
                             iterable_ids_of_unique, iterable_ordered_group,
                             iterables_concat, single_element, unique_value)
from lacro.path.pathver import versioned_file
from lacro.sort import argsort
from lacro.string.misc import (XhtmlText, class_attr, escape_latex,
                               is_normal_key, iterable_2_repr,
                               keys_intersect_re_keys, re_keys_2_keys, to_repr,
                               to_str, to_strp)
from lacro.string.misc import trunc as strunc
from lacro.typing import SizedIterable

from .._elements.column_descriptor import (ColumnDescriptor, SetOfTypes,
                                           SubsetOfType, UnknownType,
                                           array_for_coldes,
                                           assert_elemtype_valid,
                                           assure_no_subset_of_type2,
                                           assure_subset_of_type2,
                                           is_set_of_types, is_subset_of_type,
                                           t2c, to_dtype, unify_dspecs, v2c)
from .._elements.values import (NullComparable, assert_deep_consistent,
                                get_null, int_null, is_null, null2htmlv,
                                null2latexv, null2str, object_null,
                                string_null)
from .._navigate.group_saver import SaveGroup
from .group import (ListGroup, ListGroup2, flatten_groups, unflatten_ungroup,
                    ungroup)


class ListDict:
    # assignment

    def __init__(self,
                 ldct: Optional['ListDict'] = None,
                 *,
                 keys: Optional[SizedIterable[Hashable]] = None,
                 vals: Optional[SizedIterable[Union[np.ndarray, str]]] = None,
                 cspecs: Optional[SizedIterable[ColumnDescriptor]] = None,
                 nrows: Optional[int] = None,
                 kinds='inject',
                 vals_dirty=False,
                 assert_cons=True) -> None:
        if ldct is not None:
            # ldct.assert_keys_consistent()
            assert ((keys is None)
                    and (cspecs is None)
                    and (nrows is None)
                    and (vals is None))
            # copied below later
            keys = ldct._pdct.keys()
            vals = ldct._pdct.values()
            cspecs = ldct.dspecs.values()
            nrows = ldct.nrows
        elif vals is not None:
            assert keys is not None
            assert cspecs is not None
        else:
            raise ValueError('To construct an empty list, set '
                             'vals=[\'<empty>\',...], nrows=0')
            # assert keys is not None
            # assert cspecs is not None
            # assert nrows == 0
            # vals = [np.empty((0,) + v.shape1, dtype=to_dtype(v.elemtype, k))
            #         for k, v in eqzip(keys, cspecs)]
        assert not hasattr(cspecs, 'keys')
        # assert self.nrows is not None
        assert nrows is not None
        self._nrows: int = nrows
        self._dct: Dict[Hashable, np.ndarray] = HashableOrderedDict()
        self._dspecs: Dict[Hashable, ColumnDescriptor] = HashableOrderedDict()
        self._vals_dirty = vals_dirty
        self.add_cols(keys, cspecs, vals, kinds=kinds, assert_cons=assert_cons)
        # do not guess nrows
        # self.nrows = self._compute_length(assumed=nrows)

    @property
    def nrows(self) -> int:
        assert self._nrows is not None
        return self._nrows

    @property
    def dspecs(self):
        return self._dspecs

    @property
    def _pdct(self):
        return self._dct

    def __copy__(self):
        return self.view()

    def view(self):
        return ListDict(self)

    # def assign(self, ldct):
    #     # ldct.assert_keys_consistent()
    #     assert False
    #     self.nrows = ldct.nrows
    #     self._dct = copy.copy(ldct._pdct)
    #     self._dspecs = copy.copy(ldct.dspecs)
    #     self._vals_dirty = ldct._vals_dirty
    #     # self.assert_keys_consistent()

    def steal(self, ldct: 'ListDict'):
        self._nrows = ldct.nrows
        self._dct = ldct._pdct
        self._dspecs = ldct.dspecs
        self._vals_dirty = ldct._vals_dirty

    # meta elements
    def add_metas(self, smetas, vmetas, update=False):
        if update:
            self.assert_metas_exist(smetas.keys())
            self.assert_metas_exist(vmetas.keys())
        else:
            self.assert_metas_dont_exist(smetas.keys())
            self.assert_metas_dont_exist(vmetas.keys())
        self.assert_metas_consistent(smetas, vmetas, check_shape=False)
        for k, v in dict_union(smetas, vmetas).items():
            self.dspecs[k] = copy.copy(self.dspecs[k])
            self.dspecs[k].meta = v
        skeys, vkeys = self.svkeys(self.meta_keys())
        self.assert_metas_consistent(
            dict_intersect(self.get_metas(), skeys),
            dict_intersect(self.get_metas(), vkeys), check_shape=False)

    def added_metas(self, smetas, vmetas, update=False):
        res = self.view()
        res.add_metas(smetas, vmetas, update)
        return res

    def update_metas(self, smetas, vmetas):
        self.add_metas(smetas, vmetas, update=True)

    def updated_metas(self, smetas, vmetas):
        return self.added_metas(smetas, vmetas, update=True)

    def assert_metas_consistent(self, smetas, vmetas, check_shape=False):
        self.assert_keys_exist(smetas.keys())
        self.assert_keys_exist(vmetas.keys())
        for k, v in vmetas.items():
            if type(v) != np.ndarray:
                raise ValueError(('type(meta[%r]) == %s, but it must be '
                                  'numpy.ndarray' % (k, type(v))))
            if check_shape:
                # alternative: always check shape, and use shape[-1] instead of
                # shape[1]
                asserted_shape(self[k], (self.nrows, None),
                               msg=('Meta shape check expects 2D vkeys, '
                                    'key %r' % (k,)))
                asserted_shape(v, (self[k].shape[1],),
                               msg=('Meta shape (must be 1D) not equal '
                                    'to second dimension of value shape '
                                    '(i.e. (shape[1],)) in key %r\n' %
                                    (k,)))
            for vv in v.flat:
                asserted_of_type(vv, dict, msg='Key %r\n' % (k,))
        for k, v in smetas.items():
            asserted_of_type(v, dict, msg='Key %r\n' % (k,))

    def auto_subset_ttype(self, keys, exclude_existing=False):
        self.assert_keys_exist(keys)
        if exclude_existing:
            keys = [k for k in keys
                    if not is_subset_of_type(self.dspecs[k].ttype)]
        for k in keys:
            self.assure_no_subset_of_type(k)
        # sorted([]) instead of sorted() for cython
        self.dspecs.update(
            {k: v2c([v for v in sorted([s for s in set(self[k])
                                        if not is_null(s)])]) for k in keys})

    def auto_subset_ttyped(self, keys=None, exclude_existing=False):
        if keys is None:
            keys = self.keys()
        res = self.view()
        res.auto_subset_ttype(keys, exclude_existing)
        return res

    # columns
    def deleted_cols(self, keys, re_keys=[],
                     keys_must_exist=True, re_keys_must_exist=True):
        keys = self.keys_re_keys(keys, re_keys,
                                 keys_must_exist, re_keys_must_exist)
        return self.cols(list_minus(self.keys(), keys))

    def delete_cols(self, *args, **kwargs):
        self.steal(self.deleted_cols(*args, **kwargs))

    def cols(self,
             keys,
             re_keys=[],
             keys_must_exist=True,
             re_keys_must_exist=True) -> 'ListDict':
        # self.assert_keys_consistent()
        keys = self.keys_re_keys(keys, re_keys,
                                 keys_must_exist, re_keys_must_exist)
        dct = dict_intersect(self._pdct, keys)
        return ListDict(keys=dct.keys(), vals=dct.values(),
                        cspecs=dict_intersect(self.dspecs, keys).values(),
                        nrows=self.nrows)

    def select_cols(self,
                    keys,
                    re_keys=[],
                    keys_must_exist=True,
                    re_keys_must_exist=True) -> None:
        self.steal(self.cols(keys,
                             re_keys,
                             keys_must_exist,
                             re_keys_must_exist))

    def added_col(self,
                  key: Hashable,
                  cspec: ColumnDescriptor,
                  val: object,
                  kind: str = 'inject',
                  auto_shape1: bool = False,
                  assert_cons: bool = True,
                  prepend: bool = False) -> 'ListDict':
        res = self.view()
        res.add_col(key, cspec, val, kind, auto_shape1, assert_cons, prepend)
        return res

    def added_cols(self,
                   keys: SizedIterable[Hashable],
                   cspecs: SizedIterable[ColumnDescriptor],
                   vals: SizedIterable[np.ndarray],
                   kinds: Union[str, List[str]] = 'inject',
                   auto_shapes1: Union[bool, List[bool]] = False,
                   assert_cons=True,
                   prepend=False) -> 'ListDict':
        res = self.view()
        res.add_cols(keys, cspecs, vals, kinds, auto_shapes1,
                     assert_cons, prepend)
        return res

    def add_col(self,
                key: Hashable,
                cspec: ColumnDescriptor,
                val: object,
                kind: str = 'inject',
                auto_shape1: bool = False,
                assert_cons: bool = True,
                prepend: bool = False) -> None:
        self.add_cols([key], [cspec], [val], [kind], [auto_shape1],
                      assert_cons, prepend)

    def add_cols(self,
                 keys: SizedIterable[Hashable],
                 cspecs: SizedIterable[ColumnDescriptor],
                 vals: SizedIterable[object],
                 kinds: Union[str, List[str]] = 'inject',
                 auto_shapes1: Union[bool, List[bool]] = False,
                 assert_cons=True,
                 prepend=False) -> None:
        self.assert_keys_dont_exist(keys)
        if type(kinds) == str:
            kinds = [cast(str, kinds)] * len(keys)
        if type(auto_shapes1) == bool:
            auto_shapes1 = [cast(bool, auto_shapes1)] * len(keys)

        def auto_coldes(v, c, kind, a1):
            if a1:
                assert c.shape1 == ()
                assert 'C' not in kind
                res = copy.copy(c)
                res.shape1 = v.shape if 'R' in kind else v.shape[1:]
                return res
            else:
                return c

        cspecs = [auto_coldes(*args)
                  for args in eqzip(vals,
                                    cspecs,
                                    cast(List[str], kinds),
                                    cast(List[bool], auto_shapes1))]
        # convert "kinds" to set to remove duplicates
        set_assert_subset(['inject', 'rep_R', 'rep_C', 'rep_RC',
                           'one_R', 'one_C', 'one_RC'], set(kinds))
        vals = [
            np.empty((self.nrows,)
                     + c.shape1, dtype=to_dtype(c.elemtype, k))
            if (type(v) == str and v == '<empty>') else
            np.full((self.nrows,)
                    + c.shape1,
                    get_null(c.elemtype, key=k, unknown_is_objnull=True),
                    dtype=to_dtype(c.elemtype, k))
            if (type(v) == str and v == '<null>') else
            v if kind == 'inject' else
            repeatE(
                (array_of_shape(v, dtype=to_dtype(c.elemtype, k),
                                shape=())
                 if type(v) != np.ndarray else
                 v),
                (((self.nrows,) if 'R' in kind else (None,))
                 + (c.shape1 if 'C' in kind else (None,) * len(c.shape1))),
                pad=kind.startswith('one'),
                padval=get_null(c.elemtype, key=k, unknown_is_objnull=True))
            for kind, k, c, v in eqzip(kinds, keys, cspecs, vals)]
        dict_unite(self.dspecs, HashableOrderedDict(eqzip(keys, cspecs)))
        dict_unite(self._pdct, HashableOrderedDict(eqzip(keys, vals)))
        if prepend:
            self.sort_cols(head_keys=keys, order='keep')
        if assert_cons:
            self.assert_keys_consistent()

    def add_ldct(self, other: 'ListDict', allow_duplicates=False,
                 unite_subset_of_type=False, unite_metas=False):
        coldes = unify_dspecs(self.dspecs, other.dspecs,
                              unite_subset_of_type=unite_subset_of_type,
                              unite_metas=unite_metas)
        if allow_duplicates:
            common_keys = list_intersect(self.keys(), other.keys())
            # column descriptors are already compared above in unify_dspecs
            differing = [k for k in common_keys
                         if not ne.nan_equal(self[k], other[k])]
            if len(differing) > 0:
                raise ValueError(f'Lists differ in key {differing!r}')
            other = other.deleted_cols(common_keys)
        self._dspecs = HashableOrderedDict(
            list(coldes.items())[:len(self.keys())])
        self.add_cols(other.keys(), [coldes[k] for k in other.keys()],
                      other._pdct.values())

    def added_ldct(self, other, allow_duplicates=False,
                   unite_subset_of_type=False, unite_metas=False):
        res = self.view()
        res.add_ldct(other, allow_duplicates=allow_duplicates,
                     unite_subset_of_type=unite_subset_of_type,
                     unite_metas=unite_metas)
        return res

    def add_col_applymap(self, key, key_coldes, operation_in_row):
        self.add_col(key,
                     key_coldes,
                     array_for_coldes([operation_in_row(l)
                                       for l in self.dicts()],
                                      self.nrows, key_coldes, key))

    def added_col_applymap(self, key, key_coldes, operation_in_row):
        res = self.view()
        res.add_col_applymap(key, key_coldes, operation_in_row)
        return res

    def add_cols_applymap(self, keys, keys_coldes, operations_in_row):
        self.add_cols(
            keys, keys_coldes,
            [array_for_coldes(a, self.nrows, cd, k)
             for a, k, cd in eqzip(
                eqziplist(
                    [operations_in_row(l) for l in self.dicts()],
                    len(keys), msg='Operations, len(keys): '),
                keys, keys_coldes, msg='Operations, keys, coldes: ')])

    def added_cols_applymap(self, keys, keys_coldes, operations_in_row):
        res = self.view()
        res.add_cols_applymap(keys, keys_coldes, operations_in_row)
        return res

    def added_counter(self, key=None, return_key=False, offset=0, suffix=None,
                      prepend=False):
        if key is None:
            key = self.random_key('<id>')
        res = self.added_col(key, t2c(np.int64),
                             offset + np.arange(self.nrows),
                             prepend=prepend)
        if suffix is not None:
            res.applymap(key, lambda v: '%d%s' % (v, suffix), cspec=t2c(str))
        if return_key:
            return res, key
        else:
            return res

    def add_counter(self, *args, **kwargs):
        self.steal(self.added_counter(*args, **kwargs))

    def renamed_keys(self, old2new={}, re_old2new={}, keys_must_exist=True,
                     re_keys_must_exist=True):
        # self.assert_keys_consistent()
        if not keys_must_exist:
            old2new = dict_intersect(old2new, self.keys(), must_exist=False)
        self.assert_keys_exist(old2new.keys())

        rre = re_keys_2_keys(self.keys(), re_old2new, re_keys_must_exist)
        re_old2new = {rre[k]: v for k, v in re_old2new.items()
                      if k in rre.keys()}

        old2new = dict_union(old2new, re_old2new)
        old2new = dict_union(old2new,
                             {k: k for k in list_minus(self.keys(),
                                                       old2new.keys())})

        dict_assert_injective(old2new)

        missing_keys = set(old2new.keys()) - set(self.keys())

        if len(missing_keys) > 0:
            for k in missing_keys:
                del old2new[k]

        return ListDict(keys=[old2new[k] for k in self.keys()],
                        vals=self._pdct.values(),
                        cspecs=self.dspecs.values(), nrows=self.nrows)

    def rename_keys(self, *args, **kwargs):
        self.steal(self.renamed_keys(*args, **kwargs))

    def sorted_cols(self, order='increasing', head_keys=[], tail_keys=[]):
        new_keys = list_sorted(self.keys(), order=order,
                               head_keys=head_keys, tail_keys=tail_keys)
        return ListDict(vals=[self[k] for k in new_keys], keys=new_keys,
                        cspecs=[self.dspecs[k] for k in new_keys],
                        nrows=self.nrows)

    def sort_cols(self, order='increasing', head_keys=[], tail_keys=[]):
        self.steal(self.sorted_cols(order, head_keys, tail_keys))

    def updated_col(self, key, cold, val=None, assert_cons=True):
        if val is None:
            val = self[key]
        return self.updated_cols([key], [cold], [val], assert_cons=assert_cons)

    def update_col(self, key, cold, val=None, assert_cons=True):
        if val is None:
            val = self[key]
        self.update_cols([key], [cold], [val], assert_cons=assert_cons)

    def updated_cols(self, keys, coldes, vals=None, assert_cons=True):
        if vals is None:
            vals = self.values(keys)
        res = self.view()
        res.update_cols(keys, coldes, vals, assert_cons=assert_cons)
        return res

    def update_cols(self, keys, coldes, vals=None, assert_cons=True):
        if vals is None:
            vals = self.values(keys)
        if assert_cons:
            self.assert_keys_exist(keys)
        dict_update(self._pdct, dict(eqzip(keys, vals)))
        dict_update(self.dspecs, dict(eqzip(keys, coldes)))
        if assert_cons:
            self.assert_keys_consistent()

    def astype(self, key, type):
        # second "astype" to convert int -> np._ -> str
        return self.updated_col(key, self.dspecs[key].astype(type),
                                self[key].astype(type).astype(to_dtype(type),
                                                              copy=False))

    def applymap(self,
                 key: Optional[Hashable] = None,
                 operation: Optional[Callable[[Any], Any]] = None,
                 keys: Optional[Sequence[Hashable]] = None,
                 operation_in_row: Optional[
                     Callable[[HashableOrderedDict], Sequence[Any]]] = None,
                 operation_in_row_key: Optional[
                     Callable[[HashableOrderedDict, Hashable], Any]] = None,
                 operations_in_row: Optional[
                     Callable[[HashableOrderedDict], Sequence[Any]]] = None,
                 keys_must_exist: bool = True,
                 assert_cons: bool = True,
                 src_lst: Optional['ListDict'] = None,
                 src_key: Optional[Hashable] = None,
                 src_keys: Optional[Sequence[Hashable]] = None,
                 cspec: Optional[ColumnDescriptor] = None,
                 cspecs: Optional[Sequence[ColumnDescriptor]] = None) -> None:
        """Apply an elementwise function to specific columns of a ListDict.

        Parameters
        ----------
        key : Optional[Hashable]
            The key of the column to apply the operation to, or None.
        operation : Optional[Callable[[Any], Any]]
            The operation to apply to the column.
        keys : Optional[Sequence[Hashable]]
            The key of the columns to apply the operation to, or None.
        operation_in_row : Optional[Callable[[HashableOrderedDict], Sequence[Any]]]
            The operation to apply to to the selected columns.
        operation_in_row_key
            The operation to apply to to the selected columns, with a second
            argument giving the current column.
        operations_in_row : Optional[Callable[[HashableOrderedDict], Sequence[Any]]]
            The operation to apply to the whole column.
        keys_must_exist : bool
            Whether or not the given keys must exist. Defaults to true
        assert_cons : bool
            Whether or not to check consistency after applying the operation.
        src_lst : Optional[ListDict]
            The list to take the values from. Defaults to None, which is
            identical to self.
        src_key : Optional[Hashable]
            The source key. Defaults to None, which is identical to the
            destination key.
        src_keys : Optional[Sequence[Hashable]]
            The source keys. Defaults to None, which is identical to the
            destination keys.
        cspec : Optional[ColumnDescriptor]
            The new column descriptor. Defaults to None, which is identical to
            the old column descriptor.
        cspecs : Optional[Sequence[ColumnDescriptor]]
            The new column descriptors. Defaults to None, which is identical to
            the old column descriptors.
        Returns
        -------
            None
        """
        if ((operation is not None)
            + (operation_in_row is not None)
            + (operation_in_row_key is not None)
                + (operations_in_row is not None)) != 1:
            raise ValueError('(operation is not None) + '
                             '(operation_in_row is not None) + '
                             '(operation_in_row_key is not None) + '
                             '(operations_in_row is not None) != 1')

        if (key is not None) + (keys is not None) != 1:
            raise ValueError('(key is not None) + (keys is not None) != 1')

        if keys is None:
            keys = [key]

        if src_keys is None:
            if src_key is not None:
                src_keys = [src_key]
            else:
                src_keys = keys

        assert len(keys) == len(src_keys)

        if src_lst is None:
            src_lst = self
        assert src_lst.nrows == self.nrows
        dst_lst = self

        if not keys_must_exist:
            keys, src_keys = eqziplist(
                [(k, sk) for SK in [src_lst.keys()]
                 for k, sk in eqzip(keys, src_keys) if sk in SK], 2)

        dst_lst.assert_keys_exist(keys)
        src_lst.assert_keys_exist(src_keys)

        if operation is not None:
            def operations_in_row(l): return [operation(l[k]) for k in
                                              src_keys]

        if operation_in_row is not None:
            def operations_in_row(l): return [operation_in_row(l)] * len(keys)

        if operation_in_row_key is not None:
            def operations_in_row(l): return [operation_in_row_key(l, k)
                                              for k in src_keys]

        if cspec is not None:
            cspecs = [cspec] * len(keys)

        if cspecs is None:
            cspecs = [dst_lst.dspecs[k] for k in keys]

        ress = [[None] * self.nrows for _ in range(len(keys))]
        for r, sl in enumerate(src_lst.dicts()):
            try:
                for c, v in enumerate(operations_in_row(sl)):  # type: ignore
                    assert c < len(keys)
                    # if False:
                    #     # do not assign directly because numpy string length
                    #     # might change (and result in cropped result)
                    #     self[keys[c]][r] = v
                    # print(ress)
                    # print(self.nrows)
                    # print(range(len(keys)))
                    ress[c][r] = v
            except Exception:
                typ, value, traceback = sys.exc_info()
                reraise(ValueError(
                    'Operation failed in source row\n%s\n%s: %s' %
                    (_rowdict_2_str(sl, is_compact=False),
                     typ.__name__, value)))

                # raise ValueError('Operation failed for key %r in source row'
                #                  '\n%s\n%s: %s' % (k, _rowdict_2_str(sl),
                #                                    type(e).__name__, e))

        dst_lst.update_cols(keys, cspecs,
                            [array_for_coldes(res, self.nrows, cd, k)
                             for k, cd, res in eqzip(keys, cspecs, ress)],
                            assert_cons=assert_cons)

    def appliedmap(self, *args, **kwargs):
        res = self.view()
        res.applymap(*args, **kwargs)
        return res

    # rows
    def sorted(self,
               keys: Optional[List[Hashable]] = None,
               d2key: Optional[Callable[[dict], Hashable]] = None,
               reverse=False,
               null_small=False,
               stable=True):
        assert (keys is not None) + (d2key is not None) == 1
        if d2key is not None:
            vals = (d2key(d) for d in self.dicts())
        else:
            assert keys is not None
            vals = self.sorting_keys(keys, null_small=null_small)

        # even mergesort is not stable if reverse is enabled, because this
        # reverts identical keys => using python sort instead
        if stable:
            ids = np.array(argsort(tuple(vals), reverse=reverse),
                           dtype=np.int64)
        else:
            ids = np.argsort(
                self._sorting_keys_array(vals)[::-1 if reverse else 1])
        return self.isliced(ids)

    def sort(self,
             keys: Optional[List[Hashable]] = None,
             d2key: Optional[Callable[[dict], Hashable]] = None,
             reverse=False,
             null_small=False,
             stable=True):
        self.steal(self.sorted(keys, d2key, reverse, null_small, stable))

    # rows / cols
    @indexable_function
    def isliced(self, rows, cols=None):
        if cols is None:
            cols = self.keys()
        self.assert_keys_exist(cols)
        return ListDict(keys=cols, vals=[self[k][rows] for k in cols],
                        cspecs=dict_intersect(self.dspecs, cols).values(),
                        nrows=np.empty((self.nrows, 0))[rows, :].shape[0])

    def fsliced(self, where, cols=None, required_keys=[]):
        self.assert_keys_exist(required_keys)
        return self.isliced(np.array([where(l) for l in self.dicts()],
                                     dtype=bool), cols)

    @indexable_function
    def islice(self, where, cols=None):
        self.steal(self.isliced(where, cols))

    def fslice(self, where, cols=None, required_keys=[]):
        self.steal(self.fsliced(where, cols, required_keys))

    def separated(self, where, required_keys=[]):
        self.assert_keys_exist(required_keys)
        return [self.fsliced(where=where),
                self.fsliced(where=lambda v: not where(v))]

    def fsliced_verbose(self, where, required_keys=[],
                        error='raise'):
        good, bad = self.separated(where, required_keys)
        if bad.nrows > 0 and error != 'omit':
            msg = '%d rows are invalid\n%s' % (bad.nrows,
                                               to_str(bad.cols(required_keys),
                                                      preserve_str=False))
            if error == 'raise':
                raise ValueError(msg)
            elif error == 'warn':
                print_err(msg)
            else:
                raise ValueError('Unknown error type "%s"' % error)
        return good

    def fslice_verbose(self, *args, **kwargs):
        self.steal(self.fsliced_verbose(*args, **kwargs))

    def fsliced_key(self, key, where):
        self.assert_keys_exist([key])
        res = self.fsliced(cols=self.keys(), where=lambda l: where(l[key]))
        if is_subset_of_type(res.dspecs[key].ttype):
            # not using "v2c" to avoid overwriting basetype
            res.dspecs[key] = copy.copy(res.dspecs[key])
            res.dspecs[key].ttype = SubsetOfType(
                v for v in res.dspecs[key].ttype if where(v))
        res.assert_keys_consistent()
        self.assert_keys_consistent()
        return res

    def __iter__(self):
        raise ValueError('ListDict not iterable. Use ListDict.dicts() for a '
                         'row iterator')

    def __getitem__(self, key):
        try:
            return self._pdct[key]
        except KeyError:
            raise KeyError(
                '\n\n\n'.join(sorted(set(
                    'ListDict with keys %s does not contain key %s' %
                    (strunc(to_str_(list(self.keys())), 100), to_str_(key))
                    for to_str_ in [repr, to_str]))))

    # keys
    def keys(self) -> SizedIterable:
        return self._pdct.keys()

    def re_keys(self, re_keys, re_keys_must_exist=True):
        return keys_intersect_re_keys(self.keys(), re_keys,
                                      re_keys_must_exist)

    def keys_re_keys(self, keys, re_keys, keys_must_exist, re_keys_must_exist):
        if not keys_must_exist:
            keys = list_intersect(keys, self.keys())
        return list_union(keys, self.re_keys(re_keys, re_keys_must_exist),
                          order='keep')

    def random_key(self, prefix=''):
        return (prefix if prefix not in self.keys() else
                self.random_key(prefix + '_'))

    def meta_keys(self):
        return self.get_metas().keys()

    def assert_keys_exist(self, keys):
        set_assert_subset(self.keys(), keys, '\n', 1000,
                          msg='Keys do not exist\n')

    def assert_keys_dont_exist(self, keys):
        lists_assert_disjoint([self.keys(), keys], msg='Keys exist\n')

    def assert_metas_exist(self, keys):
        set_assert_subset(self.meta_keys(), keys, '\n', 1000,
                          msg='Metas do not exist\n')

    def assert_metas_dont_exist(self, keys):
        lists_assert_disjoint([self.meta_keys(), keys])

    def assert_numpy(self, keys):
        self.assert_keys_exist(keys)
        wrong = [k for k in keys if type(self[k]) != np.ndarray]
        if len(wrong) > 0:
            raise ValueError('Not numpy: %s. Type(s): %s' %
                             (wrong, ', '.join(type(self[k]).__name__
                                               for k in wrong)))

    def assert_is_nonnull(self, keys, kcheck=None):
        self.assert_keys_exist(keys)
        nk = [k for k in keys if np.any(self.is_any_null([k]))]
        if len(nk) > 0:
            raise ValueError(
                'Keys %s contain NULL values%s' %
                (nk,
                 '' if kcheck is None else
                 '\n%s' % '\n'.join(
                     str(v) for v in self[kcheck][self.is_any_null(keys)])))

    def assert_subset_of_type(self, keys):
        for k in keys:
            self.assure_subset_of_type(k)

    def assert_numeric(self, keys):
        for k in keys:
            self.assure_numeric(k)

    def assert_string(self, keys):
        for k in keys:
            asserted_is_type(self.dspecs[k].ttype, str)

    def assure_no_subset_of_type(self, key):
        return assure_no_subset_of_type2(self.dspecs, key)

    def assure_subset_of_type(self, key):
        return assure_subset_of_type2(self.dspecs, key)

    def assure_numeric(self, key):
        if not ne.tis_numeric(self.dspecs[key].ttype):
            raise ValueError('%r\'s ttype is not numeric. '
                             'Instead, it is of type "%s.' %
                             (key, self.dspecs[key].ttype))
        return self.dspecs[key].ttype

    def assure_elemtype_numeric(self, key):
        if not ne.tis_numeric(self.dspecs[key].elemtype):
            raise ValueError('%r\'s elemtype is not numeric. '
                             'Instead, it is of type "%s.' %
                             (key, self.dspecs[key].elemtype))
        return self.dspecs[key].elemtype

    def assert_keys_consistent(self):
        if not isinstance(self, ListDict):
            raise ValueError('Variable of type "%s" is not an instance of '
                             '"ListDict"' % type(self))
        assert type(self.dspecs) == HashableOrderedDict
        assert type(self._pdct) == HashableOrderedDict
        lists_assert_equal([list(self.dspecs.keys()),
                            list(self._pdct.keys())])
        assert list(self.dspecs.keys()) == list(self._pdct.keys())

        self.assert_numpy(self.keys())

        for k in self.keys():
            if self[k].dtype.type != np.object_:
                asserted_is_type(self[k].dtype.type, self.dspecs[k].elemtype,
                                 'Key: %r\n' % k)
            assert_elemtype_valid(self.dspecs[k].elemtype, k)
            asserted_shape(self[k], (self.nrows,) + self.dspecs[k].shape1,
                           msg='Shape error in key %r\n' % (k,))
            if is_subset_of_type(self.dspecs[k].ttype):
                for v in self.dspecs[k].ttype:
                    if type(v) != self.dspecs[k].basetype:
                        raise ValueError(
                            'Column descriptor (not necessarily the table '
                            'contents) inconsistent. Element %r of column '
                            'descriptor %r is of type "%s" and not of '
                            'basetype "%s"' % (v,
                                               k, type(v),
                                               self.dspecs[k].basetype))
        if assert_deep_consistent.get() and (not self._vals_dirty):
            for k, c in self._pdct.items():
                ttype = self.dspecs[k].ttype
                for i, v in enumerate(c):
                    if ((ttype != UnknownType)
                            and (not is_null(v, k))):
                        # print(is_null_or_undefined(v))
                        # print(v)
                        # print(type(v))
                        if is_subset_of_type(ttype):
                            if v not in ttype:
                                raise ValueError(
                                    'self[%r][%d]=%s, which is not in %s' %
                                    (k, i, strunc(repr(v), 90),
                                     ttype))
                        elif is_set_of_types(ttype):
                            if type(v) not in ttype:
                                raise ValueError(
                                    'type(self[%r][%d])=%s, which is not in %s'
                                    % (k, i, repr(type(v)), ttype))
                        else:
                            if ttype not in [np.object_, type(v)]:
                                raise ValueError(
                                    'self[%r][%d]=%s, its type is %s, which '
                                    'is not %s' % (k, i, strunc(repr(v), 90),
                                                   type(v), ttype))
        # print(self.dspecs)
        # print(self.nrows)
        self._compute_length(assumed=self.nrows)
        return self

    def assert_predicates(self, predicates, messages):
        message = '\n'.join(m(l) for l in self.dicts()
                            for p, m in eqzip(predicates, messages)
                            if not p(l))
        if len(message) != 0:
            raise ValueError(message)

    # get values

    def values(self, keys):
        self.assert_keys_exist(keys)
        return [self._pdct[k] for k in keys]

    def _npvaluesT(self, skeys, dtype):
        self.assert_keys_exist(skeys)
        [asserted_shape(self[k], (self.nrows,)) for k in skeys]
        if len(skeys) == 0:
            return np.empty(shape=(self.nrows, len(skeys)), dtype=dtype)
        else:
            return np.array(self.values(skeys), dtype=dtype).T

        # print(res.shape)
        # print(self.values(keys))
        # if len(keys) > 0:
        # does not work for strings (keeps only the first
        # character of the string)
        # res[:] = self.values(keys)
        # return res.T

    def is_any_null(self, keys=None):
        """Returns a boolean array describing which rows contain null values.

        In contrast to the cell-wise "is_null", this also checks contents of
        arrays.
        """
        if keys is None:
            keys = self.keys()
        self.assert_keys_exist(keys)
        ints = [k for k in keys if self.dspecs[k].elemtype in [np.int32,
                                                               np.int64]]
        floats = [k for k in keys if self.dspecs[k].elemtype in [np.float32,
                                                                 np.float64]]
        strs = [k for k in keys if self.dspecs[k].elemtype in [str]]
        objs = list_minus(keys, ints + floats + strs)
        # lists_assert_equal([keys, ints+floats+strs],
        #                    error='Unknown elemtype or elemtype was not '
        #                          'set.\n%s\n' % dict_intersect(
        #                              self.dspecs, keys))
        return (np.any(self.npvaluesT(ints, dtype=int) == int_null,
                       axis=1)
                | np.any(np.isnan(self.npvaluesT(floats, dtype=float)),
                       axis=1) |
                np.any((self.npvaluesT(strs, dtype=to_dtype(str))
                        == string_null), axis=1)
                | np.any((self.npvaluesT(objs, dtype=to_dtype(object)) ==
                        object_null), axis=1))

    def svkeys(self, skeys, vkeys=None):
        list_assert_no_duplicates(skeys)
        if vkeys is None:
            # there is an "ndim==2" check below
            vkeys = [k for k in skeys if self[k].ndim > 1]
            skeys = [k for k in skeys if self[k].ndim == 1]
        list_assert_disjoint(skeys, vkeys)
        return skeys, vkeys

    def npvaluesT(self, skeys=None, vkeys=None, dtype=None):
        if skeys is None:
            skeys = self.keys()
        skeys, vkeys = self.svkeys(skeys, vkeys)

        if dtype is None:
            dtype = unique_value([to_dtype(self.dspecs[k].elemtype)
                                  for k in skeys + vkeys])

        self.assert_keys_exist(skeys)
        self.assert_keys_exist(vkeys)
        list_assert_disjoint(skeys, vkeys)

        [asserted_shape(self[k], (self.nrows,)) for k in skeys]
        [asserted_shape(self[k], (self.nrows, None)) for k in vkeys]

        svals = asserted_shape(self._npvaluesT(skeys, dtype=dtype),
                               (self.nrows, len(skeys)))
        vvals = asserted_shape(
            default_concatenate(self.values(vkeys), axis=1,
                                shape=(self.nrows, 0), dtype=dtype),
            (self.nrows, None))

        # unique_value([len(s) for s in svals] + [len(s) for s in vvals])
        return asserted_shape(np.concatenate([svals, vvals], axis=1),
                              (self.nrows, len(skeys) + vvals.shape[1]))

    # get meta
    def get_metas(self):
        return OrderedDict((k, v.meta) for k, v in self.dspecs.items()
                           if v.meta is not None)

    def guess_meta(self, sguess=True, vguess=True):
        skeys, vkeys = self.svkeys(self.keys())
        smkeys, vmkeys = self.svkeys(self.meta_keys())
        if sguess:
            smissing = list_minus(skeys, smkeys)
            self.add_metas({k: {'name': k, 'type': 'scalar'}
                            for k in smissing},
                           {})
        else:
            self.assert_metas_exist(skeys)
        if vguess:
            vmissing = list_minus(vkeys, vmkeys)
            self.add_metas(
                {}, {k: np.array(
                    [{'name': '%s_%d' % (k, i), 'type': 'scalar'}
                     for i in range(
                        asserted_shape(self[k],
                                       (self.nrows, None)).shape[1])])
                     for k in vmissing})
        else:
            self.assert_metas_exist(vkeys)

    def guessed_meta(self, sguess=True, vguess=True):
        res = self.view()
        res.guess_meta(sguess, vguess)
        return res

    def meta_values(self, skeys: List[str], vkeys: List[str], kkey=None,
                    collapse=True):
        """Meta values for the specified ``skeys`` and ``vkeys``.

        Args:
            skeys: List of 1D keys
            vkeys: List of nD keys
            kkey: For instance ``'name'``. If ``None``,
                  returns the whole meta dictionary

            collapse: If ``True``, returns a 1D array of length
                      ``len(skeys) + sum(self[k].shape[1] for k in vkeys)``.
                      If ``False``, returns a 1D array of shape
                      ``len(skeys) + len(vkeys)``

        """
        self.assert_metas_exist(skeys)
        self.assert_metas_exist(vkeys)
        self.assert_metas_consistent(
            dict_intersect(self.get_metas(), skeys),
            dict_intersect(self.get_metas(), vkeys), check_shape=True)

        # outer list convert vkeys iterator to list, to concatenate with skeys.
        # inner list converts vgetk generator to list.
        def collap(L): return (list(iterables_concat(L)) if collapse else
                               list(list(l) for l in L))

        def sgetk(l): return l[kkey] if kkey is not None else l

        def vgetk(L): return (l[kkey] for l in L) if kkey is not None else L

        res = ([sgetk(self.dspecs[k].meta) for k in skeys]
               + collap(vgetk(self.dspecs[k].meta) for k in vkeys))
        return (np.array(res, dtype=object) if collapse else
                array_of_shape(res, shape=(len(skeys) + len(vkeys),),
                               dtype=object))

        # snames = [[{'name':k, 'type':'scalar'} for k in skeys]] * self.nrows
        # return np.concatenate([snames, vnames], axis=1)

    def colnames(self, skeys, vkeys, guess=True, collapse=True):
        return ((self.guessed_meta() if guess else self)
                .meta_values(skeys, vkeys, 'name', collapse=collapse))

    def assert_no_duplicates(self, keys=None):
        list_assert_no_duplicates(self.sorting_keys(keys))
        return self

    def sorting_keys(self, keys=None, null_small=False):
        if keys is None:
            keys = self.keys()
        self.assert_keys_exist(keys)
        i2key = self.row_sorting_key(keys, null_small=null_small)
        return (i2key(i) for i in range(self.nrows))

    def sorting_keys_array(self, *args, **kwargs):
        return self._sorting_keys_array(self.sorting_keys(*args, **kwargs))

    def _sorting_keys_array(self, vals):
        return array_of_shape(tuple(vals), shape=(self.nrows,), dtype=object)

    # does not do an implicit sort (might use a hashtable internally)
    def removed_duplicates(self, keys=None, preserve_order=True):
        ids = self.ids_of_unique(keys)
        if preserve_order:
            ids = np.sort(ids)
        return self.isliced(ids)

    def remove_duplicates(self, keys=None, preserve_order=True):
        self.steal(self.removed_duplicates(keys, preserve_order))

    # prefixes
    def deleted_identical_columns(self, dst2srcs={}):
        self.assert_keys_exist(dst2srcs.keys())
        self.assert_keys_exist(lists_concat(dst2srcs.values()))
        lists_assert_disjoint([dst2srcs.keys()] + list(dst2srcs.values()))

        res = self.appliedmap(keys=dst2srcs.keys(), operation_in_row_key=(
            lambda l, k: unique_value([l[k]] + [l[m] for m in dst2srcs[k]],
                                      msg='Columns are not identical.\n')))
        return res.cols(list_minus(res.keys(),
                                   lists_concat(dst2srcs.values())))

    def delete_identical_columns(self, dst2srcs={}):
        self.steal(self.deleted_identical_columns(dst2srcs))

    def renamed_prefixes(self, old2new, identical_cols_dst=[], keys=None):
        res = self.view()
        res.rename_prefixes(old2new, identical_cols_dst, keys)
        return res

    def rename_prefixes(self, old2new, identical_cols_dst=[], keys=None):
        if keys is None:
            keys = self.keys()
        keys = list(keys)
        list_assert_no_duplicates(keys)
        list_assert_no_duplicates(identical_cols_dst)
        for old, new in old2new.items():
            K = [k for k in keys if k.startswith(old)]
            if len(K) == 0:
                raise ValueError('Could not find prefix "%s" in "%s"' %
                                 (old, keys))
            dst2srcs = {d: [old + d] for d in identical_cols_dst}
            self.delete_identical_columns(dst2srcs)
            K = list_minus(K, list(iterables_concat(dst2srcs.values())),
                           must_exist=False)
            self.rename_keys({k: new + k[len(old):] for k in K})

    # returns an "HashableOrderedDict" instead of a "ListDict" because the
    # different groupings can have different lengths
    def grouped(self,
                keys: List[Hashable],
                check_is_nonnull=True,
                use_subset_ttype=False,
                auto_subset_ttype=False,
                reverse=False,
                sort_keys=True,
                return_view=False) -> Union['ListDict', ListGroup]:
        # if check:
        # self.assert_keys_consistent()
        if check_is_nonnull:
            self.assert_is_nonnull(keys)
        if auto_subset_ttype:
            self = self.auto_subset_ttyped(keys)
        if len(keys) == 0:
            return self.view() if return_view else self
        else:
            return ListGroup(
                ((v[0].v,
                  (self.isliced(VV, list_minus(self._pdct.keys(), [keys[0]]))
                   .grouped(keys[1:], check_is_nonnull=False,
                            use_subset_ttype=use_subset_ttype,
                            reverse=reverse, sort_keys=sort_keys)))
                 for v, VV in iterable_ordered_group(
                    self.sorting_keys([keys[0]]),
                    (tuple(self.coldes_sorting_keys([keys[0]]))
                     if use_subset_ttype else None),
                    reverse=reverse,
                    sort_keys=sort_keys)),
                self.dspecs[keys[0]])

    # returns an "HashableOrderedDict" instead of a "ListDict" because the
    # different groupings can have different lengths
    def grouped2(self,
                 keyss: List[List[Hashable]],
                 check_is_nonnull=True,
                 use_subset_ttype=False,
                 reverse=False,
                 sort_keys=True) -> Union['ListDict', ListGroup2]:
        # if check:
        # self.assert_keys_consistent()
        if check_is_nonnull:
            self.assert_is_nonnull(lists_concat(keyss))
        if len(keyss) == 0:
            return self
        else:
            return ListGroup2(
                ((tuple(vi.v for vi in v),
                  (self.isliced(VV, list_minus(self._pdct.keys(), keyss[0]))
                   .grouped2(keyss[1:],
                             check_is_nonnull=False,
                             use_subset_ttype=use_subset_ttype)))
                 for v, VV in iterable_ordered_group(
                    self.sorting_keys(keyss[0]),
                    (tuple(self.coldes_sorting_keys(keyss[0]))
                     if use_subset_ttype else None),
                    reverse=reverse,
                    sort_keys=sort_keys)),
                list(dict_intersect(self.dspecs, keyss[0]).values()))

    def group_and_flatten(self, keys, leaves_keys=[],
                          use_subset_ttype=False,
                          sort_keys=True, return_view=True,
                          check_is_nonnull=True):
        lists_assert_disjoint([keys, leaves_keys])

        return flatten_groups(
            self.grouped(keys + leaves_keys,
                         use_subset_ttype=use_subset_ttype,
                         sort_keys=sort_keys,
                         return_view=return_view,
                         check_is_nonnull=check_is_nonnull),
            keys,
            [self.dspecs[k] for k in keys],
            leaf_type=ListDict if len(leaves_keys) == 0 else ListGroup)

    def group_by_unique(self, keys, re_keys=[],
                        keys_must_exist=True, re_keys_must_exist=True):
        keys = self.keys_re_keys(keys, re_keys,
                                 keys_must_exist, re_keys_must_exist)
        return self.cols(keys).added_col(
            'data', t2c(GetattrHashableOrderedDict),
            array_of_shape(list(self.cols(keys).dicts()),
                           shape=(self.nrows,),
                           dtype=to_dtype(GetattrHashableOrderedDict)))

    def _joined2(self, lst1, keys, join_type, order, left_num_copies,
                 right_num_copies):
        """Internal method used by the global ``join`` method.

        Users should invoke the global ``join`` method instead.
        """

        # lst0.assert_keys_consistent()
        # lst1.assert_keys_consistent()
        self.assert_keys_exist(keys)
        lst1.assert_keys_exist(keys)
        self.assert_keys_dont_exist(set(lst1.keys()) - set(keys))

        ids0 = self.sorting_keys_array(keys)
        ids1 = lst1.sorting_keys_array(keys)

        [oid0, oid1], [did0, did1], di = array_ids_join(
            ids0, ids1, join_type, left_num_copies, right_num_copies, keys,
            duplicates_error_string=None, order=order)
        coldes = unify_dspecs(self.dspecs, lst1.dspecs)
        # res = ListDict(keys=coldes.keys(),
        #               cspecs=coldes.values(),
        #               vals=[get_null(coldes[k].elemtype)
        #                     for k in coldes.keys()],
        #               kinds='rep_RC', nrows=di)
        res = ListDict(keys=coldes.keys(),
                       cspecs=coldes.values(),
                       vals=['<empty>' for _ in coldes.keys()],
                       kinds='rep_RC',
                       nrows=di,
                       vals_dirty=True)
        for k in self.keys():
            res[k][did0] = self[k][oid0]
            if len(did0) < di and (k not in keys):
                res[k][mask_negated_indices(did0, di)] = \
                    get_null(coldes[k].elemtype, unknown_is_objnull=True)
        for k in lst1.keys():
            res[k][did1] = lst1[k][oid1]
            if len(did1) < di and (k not in keys):
                res[k][mask_negated_indices(did1, di)] = \
                    get_null(coldes[k].elemtype, unknown_is_objnull=True)
        res._vals_dirty = False
        res.assert_keys_consistent()
        return res

    def joined_re(self, re_lst, self_key, re_key, left_num_copies,
                  right_num_copies):
        from . import multiple as lds
        self.assert_keys_exist([self_key])
        re_lst.assert_keys_exist([re_key])
        self.assert_keys_dont_exist(re_lst.keys())
        self.assert_string([self_key])
        re_lst.assert_string([re_key])
        jmod = ListDict.from_dicts([dict_union(m, {self_key: c[self_key]})
                                    for m in re_lst.dicts()
                                    for c in self.dicts()
                                    if re.match(m[re_key], c[self_key])],
                                   dspecs=dict_union(re_lst.dspecs,
                                                     {self_key: t2c(str)}))

        if right_num_copies in ['1', '+']:
            missing = list_minus(re_lst[re_key], jmod[re_key])
            if len(missing) > 0:
                raise ValueError(
                    'Could not find a single occurence of "%s" in "%s"' %
                    (missing, self[self_key]))

        self, id = self.added_counter(return_key=True)
        res = lds.join([self, jmod], joined_keys=[self_key],
                       list_num_copiess=[left_num_copies, right_num_copies],
                       join_type='left')
        res = res.sorted([id])
        res = res.cols(list_minus(res.keys(), [id]))
        return res

    def minus(self, b):
        """Difference of a and b.

        Returns a slice of ``a`` that contains all rows where the values are
        different from those in ``b`` in the columns that have matching names
        in
        ``a`` and ``b``.

        ``b`` may not contain duplicate rows, i.e. ``b.assert_no_duplicates()``
        must succeed.
        """
        from . import multiple as lds
        self.assert_keys_exist(b.keys())

        if '<dummy>' in self.keys():
            raise ValueError('a has column with name "<dummy>"')

        if len(b.keys()) == 0:
            return self.isliced(np.empty(shape=(0,), dtype=int))

        b = b.updated_cols(list_intersect(self.keys(), b.keys()),
                           dict_intersect(self.dspecs, b.keys()).values())

        # compute unique a (ua) and unique b (ub)
        ua = self.cols(b.keys()).removed_duplicates()
        b.assert_no_duplicates()
        ub = b

        keys = b.keys()

        # compute unique a & b
        uab = lds.join([ua, ub], joined_keys=keys)

        uab.add_col('<dummy>', t2c(np.int64), 1, kind='rep_R')

        auab = lds.join([self, uab], joined_keys=keys,
                        join_type='left', order='left')

        return auab.isliced(auab['<dummy>'] != 1).cols(self.keys())

    # def union(a,b): same as lds.concat

    def intersection(self, b):
        from . import multiple as lds
        self.assert_keys_exist(b.keys())
        b.assert_no_duplicates()
        return lds.join([self, b], joined_keys=b.keys(), order='left')

    def removed_repetitions(self, keys, check_is_nonnull=True):
        return ungroup(self.grouped(keys, check_is_nonnull=check_is_nonnull),
                       keys,
                       self.dspecs, repeat=False)

    def remove_repetitions(self, keys, check_is_nonnull=True):
        self.steal(
            self.removed_repetitions(keys, check_is_nonnull=check_is_nonnull))

    def merged_cols(self,
                    keys: Sequence[Hashable],
                    key_sep='.',
                    value_sep='.',
                    subset_of_type=False,
                    rm_keys=True,
                    cat_strings=True,
                    res_key: Optional[Hashable] = None) -> \
            Tuple['ListDict', Optional[Hashable]]:
        self.assert_keys_exist(keys)
        if len(keys) == 0:
            return self, None
        elif len(keys) == 1:
            return self, keys[0]
        else:
            if cat_strings:
                result_type = unique_value(self.dspecs[k].basetype
                                           for k in keys)
                if result_type not in [str, XhtmlText]:
                    raise ValueError(f'Can only merge "str" and "XhtmlText"'
                                     f'columns, not "{type(result_type)}"')
                if subset_of_type:
                    coldes = v2c([result_type(value_sep.join(v))
                                  for v in itertools.product(
                        *[self.assure_subset_of_type(k)
                          for k in keys])])
                else:
                    coldes = t2c(result_type)
                if res_key is None:
                    res_key = key_sep.join(map(str, keys))
                lst = self.added_col_applymap(
                    res_key, coldes, lambda l: result_type(
                        value_sep.join(l[k] for k in keys)))
            else:
                from lacro.statistics.linmodel.covariate import LevelAnd
                if subset_of_type:
                    coldes = v2c([LevelAnd(v)
                                  for v in itertools.product(
                        *[self.assure_subset_of_type(k)
                          for k in keys])])
                else:
                    coldes = t2c(LevelAnd)
                if res_key is None:
                    res_key = LevelAnd(keys)
                lst = self.added_col_applymap(
                    res_key, coldes, lambda l: LevelAnd(l[k] for k in keys))
            if rm_keys:
                lst = lst.cols(list_minus(lst.keys(), keys))
            return lst, res_key

    def merge_cols(self,
                   keys: Sequence[Hashable],
                   key_sep='.',
                   value_sep='.',
                   subset_of_type=False,
                   rm_keys=True,
                   cat_strings=True,
                   res_key: Optional[Hashable] = None):
        lst, res_key = self.merged_cols(keys, key_sep, value_sep,
                                        subset_of_type, rm_keys,
                                        cat_strings, res_key)
        self.steal(lst)
        return res_key

    def merged_rows(self, keys, join_str, null_val=None, **str_kwargs):
        """Inverse of ``splitted``
        """
        return self.appliedmap(keys=keys,
                               cspecs=[t2c(str) for _ in range(len(keys))],
                               operations_in_row=lambda l: [
                                   (join_str.join(
                                       null2str(v, k, null_val, **str_kwargs)
                                       for v in l[k] if
                                       ((null_val is not None) or
                                        (not is_null(v)))))
                                   for k in keys])

    def contracted(self, gkeys):
        """Inverse of ``expanded``

            List:

            = == =
            a  b c
            = == =
            1  2 3
            4  9 8
            4 10 0
            = == =

            ``contracted(['a'])``

            = ======= ======
            a       b      c
            = ======= ======
            1     [2]    [3]
            4 [9, 10] [8, 0]
            = ======= ======

        """
        ckeys = list_minus(self.keys(), gkeys)
        res = (self.extrapolated_down(gkeys)
               .renamed_keys({ck: ck + '_old' for ck in ckeys})
               .reduced(gkeys,
                        ckeys,
                        [t2c(np.ndarray, elemtype=np.ndarray) for _ in ckeys],
                        lambda l: [l[ck + '_old'] for ck in ckeys]))
        return res

    def contracted_merged(self, gkeys, join_str='\n', **strjoin_kwargs):
        """First call ``contracted``, then ``merged_rows``

            List:

            = ==== ===
            a   b   c
            = ==== ===
            1  '2' '3'
            4  '9' '8'
            4 '10' '0'
            = ==== ===

            ``contracted_merged(['a'])``

            +---+-------+------+
            | a |    b  |   c  |
            +===+=======+======+
            | 1 |  '2'  |  '3' |
            +---+-------+------+
            | 4 | | '9  | | '8 |
            |   | | 10' | | 0' |
            +---+-------+------+

        """
        ckeys = list_minus(self.keys(), gkeys)
        return self.contracted(gkeys).merged_rows(ckeys, join_str=join_str,
                                                  **strjoin_kwargs)

    def splitted(self, keys, sep):
        """Inverse of ``merged_rows``
        """
        return self.appliedmap(keys=keys,
                               cspecs=[t2c(list) for _ in range(len(keys))],
                               operations_in_row=lambda l: [l[k].split(sep) for
                                                            k in keys])

    def expanded(self, keys, cspecs):
        """Inverse of ``contracted``
        """
        if self.nrows == 0:
            return self.view()

        assert len(keys) == len(cspecs)
        maxh = [maxdef((len(vs[i]) for vs in self.values(keys)), 0)
                for i in range(self.nrows)]
        nrows1 = sum(maxh)
        heights = np.concatenate([[0], np.cumsum(maxh[:-1], dtype=np.int64)])
        dspecsk = dict(eqzip(keys, cspecs))
        cspecs1 = [dspecsk[k] if k in keys else self.dspecs[k]
                   for k in self.keys()]

        def expand0(c, v):
            res = np.empty((nrows1,) + c.shape1, dtype=to_dtype(c.elemtype))
            res[mask_negated_indices(heights, nrows1)] = get_null(c.elemtype)
            res[heights] = v
            return res

        vals1 = [(np.concatenate(
            [np.concatenate(
                [v,
                 np.repeat(np.asarray(get_null(c.elemtype),
                                      dtype=to_dtype(c.elemtype, k)),
                           (maxh[i] - len(v)), 0)])
                for i, v in enumerate(self[k])])
            if k in keys else
            expand0(c, self[k])) for k, c in zip(self.keys(), cspecs1)]

        return ListDict(keys=self.keys(), cspecs=cspecs1, vals=vals1,
                        nrows=nrows1)

    def splitted_expand(self, keys=None):
        """Inverse of ``contracted_merged``
        """
        if keys is None:
            keys = self.keys()
        return self.splitted(keys, '\n').expanded(keys,
                                                  [t2c(str)] * len(keys))

    # ----------
    # - reduce -
    # ----------
    def reduced(self, keys, result_keys, result_coldes, operations=None,
                operations_in_row=None, use_subset_ttype=False,
                check_is_nonnull=True):
        assert len(result_keys) == len(result_coldes)

        assert (operations is not None) + (operations_in_row is not None) == 1

        self.assert_keys_exist(keys)
        self.assert_keys_dont_exist(result_keys)

        FG = self.group_and_flatten(keys, use_subset_ttype=use_subset_ttype,
                                    check_is_nonnull=check_is_nonnull)

        if operations is not None:
            def operations_in_row(l): return operations(l['leaves'])

        FG.add_cols_applymap(result_keys, result_coldes, operations_in_row)
        FG.delete_cols(['leaves'])
        return FG

    def reduce(self, keys, result_keys, result_coldes, operations=None,
               operations_in_row=None, use_subset_ttype=False,
               check_is_nonnull=True):
        self.steal(self.reduced(keys, result_keys, result_coldes, operations,
                                operations_in_row, use_subset_ttype,
                                check_is_nonnull))

    def interpolated(self, other, keys, result_keys, result_coldes,
                     operations_in_row, use_subset_ttype=False,
                     prefix=''):
        from . import multiple as lds
        assert len(result_keys) == len(result_coldes)

        other.assert_keys_exist(keys)
        other.assert_keys_dont_exist(result_keys)

        FG = other.group_and_flatten(keys, use_subset_ttype=use_subset_ttype)
        FG = lds.join([self, FG],
                      joined_keys=keys,
                      join_type='left',
                      order='left',
                      list_num_copiess=['0-1', '*'])

        def ops(l):
            if is_null(l['leaves']):
                return [get_null(cd.elemtype) for cd in result_coldes]
            else:
                return operations_in_row(l)

        FG.add_cols_applymap(result_keys, result_coldes, ops)
        FG.delete_cols(['leaves'])
        o2n = OrderedDict([(k, prefix + k)
                           for k in list_minus(self.keys(), keys)])
        FG.rename_keys(o2n)
        return FG

    def taken_one(self, other, keys, prefixes, operations,
                  use_subset_ttype=False):
        result_keys = list_minus(other.keys(), keys)
        p0, p1 = prefixes
        return self.interpolated(other,
                                 keys,
                                 [p1 + k for k in result_keys],
                                 [other.dspecs[k] for k in result_keys],
                                 operations,
                                 use_subset_ttype,
                                 p0)

    def take_one(self, other, keys, prefixes, operations,
                 use_subset_ttype=False):
        self.steal(self.taken_one(other, keys, prefixes, operations,
                                  use_subset_ttype))

    def binned(self, other, keys=None, null_small=False,
               my_prefix='', other_prefix='bin '):
        if keys is None:
            keys = self.keys()
        other = other.sorted(keys)
        ids = np.searchsorted(
            other.sorting_keys_array(keys, null_small=null_small),
            self.sorting_keys_array(keys, null_small=null_small))
        ids = np.minimum(other.nrows - 1, ids)
        return (self.renamed_prefixes({'': my_prefix})
                .added_ldct(other.isliced[ids]
                            .renamed_prefixes({'': other_prefix})))

    def contingency_table(self, keys, unique=None, result_key='count',
                          use_subset_ttype=False, check_is_nonnull=True):
        if unique is not None:
            lst = self.cols(keys + unique).removed_duplicates()
        else:
            lst = self
        return lst.reduced(keys, [result_key], [t2c(np.int64)],
                           lambda L: [L.nrows],
                           use_subset_ttype=use_subset_ttype,
                           check_is_nonnull=check_is_nonnull)

    def grouped_and_apply(self, keys, result_keys, result_coldes,
                          operation=None,
                          use_subset_ttype=False,
                          sort_keys=True,
                          check_is_nonnull=True,
                          leaves_keys=[],
                          operation_in_row=None):
        assert len(result_keys) == len(result_coldes)

        assert (operation is not None) + (operation_in_row is not None) == 1

        FG = self.group_and_flatten(keys, use_subset_ttype=use_subset_ttype,
                                    sort_keys=sort_keys,
                                    return_view=True,
                                    check_is_nonnull=check_is_nonnull,
                                    leaves_keys=leaves_keys)

        if operation is not None:
            def operation_in_row(l): return operation(l['leaves'])

        for l in FG.dicts():
            operation_in_row(l)
        return unflatten_ungroup(
            FG,
            dict_union(self.dspecs,
                       HashableOrderedDict(eqzip(result_keys,
                                                 result_coldes))),
            leaves_keys)

    def group_and_apply(self, keys, result_keys, result_coldes, operation,
                        use_subset_ttype=False,
                        sort_keys=True,
                        check_is_nonnull=True):
        self.steal(self.grouped_and_apply(
            keys, result_keys, result_coldes, operation,
            use_subset_ttype, sort_keys, check_is_nonnull))

    def grouped_and_add_cols(self, keys, result_keys, result_coldes,
                             operations=None, use_subset_ttype=False,
                             kinds='inject', sort_keys=True,
                             operations_in_row=None):
        assert len(result_keys) == len(result_coldes)

        assert (operations is not None) + (operations_in_row is not None) == 1

        if operations is not None:
            def operations_in_row(l): return operations(l['leaves'])

        def operation_in_row(l):
            l['leaves'].add_cols(result_keys,
                                 result_coldes,
                                 operations_in_row(l),
                                 kinds=kinds)

        return self.grouped_and_apply(keys, result_keys, result_coldes,
                                      operation_in_row=operation_in_row,
                                      use_subset_ttype=use_subset_ttype,
                                      sort_keys=sort_keys)

    def group_and_add_cols(self, *args, **kwargs):
        self.steal(self.grouped_and_add_cols(*args, **kwargs))

    def firsted(self, keys):
        return self.removed_duplicates(keys)

    def lasted(self, keys):
        return self.isliced[::-1].removed_duplicates(keys).isliced[::-1]

    def first(self, keys):
        self.steal(self.firsted(keys))

    def last(self, keys):
        self.steal(self.lasted(keys))

    def extrapolated_down(self, keys):
        if self.nrows == 0:
            return self.view()
        old = [self[k][0] for k in keys]
        for k, o in zip(keys, old):
            if is_null(o):
                raise ValueError(f'First element in key {k!r} is null')

        def getv(l):
            nonlocal old
            for i, k in enumerate(keys):
                if not is_null(l[k]):
                    old[i] = l[k]
            return old

        return self.appliedmap(keys=keys, operations_in_row=getv)

    def ids_of_unique(self, keys):
        return np.array(tuple(iterable_ids_of_unique(self.sorting_keys(keys))),
                        dtype=int)

    def ids_of_change(self, keys):
        return np.array(tuple(iterable_ids_of_change(self.sorting_keys(keys))),
                        dtype=int)

    def _do_escape(self, escape):
        if type(escape) != bool:
            self.assert_keys_exist(escape)
            return lambda k: (k in escape)
        else:
            return lambda k: escape

    def __to_xhtml__(self, *args, **kwargs):
        return self.to_xhtml(*args, **kwargs)

    def to_xhtml(self,
                 null_val='?',
                 table_class=None,
                 null_class='null',
                 td_classes={},
                 th_classes={},
                 escape=False,
                 float_fmt='{:.2e}',
                 javascript_click=False,
                 merge: Sequence[Sequence[Hashable]] = [],
                 **kwargs):
        """Returns listdict as xhtml string

        Parameters
        ----------
        null_val
        table_class
        null_class
        td_classes
        th_classes
        escape
        float_fmt
        javascript_click
            does not yet stop event propagation and does not
            distinguish between left and right click => disabled by
            default
        merge
        kwargs
            Keyword arguments passed to ``null2htmlv``
        Returns
        -------

        """
        self.assert_keys_exist(th_classes.keys())
        self.assert_keys_exist(td_classes.keys())
        do_escape = self._do_escape(escape)

        def _null2htmlv(v, k):
            return null2htmlv(v, k, null_val, null_class,
                              do_escape(k), float_fmt, **kwargs)

        if len(merge) != 0:
            lst = self.view()
            for m in merge:
                if all(mm in lst.keys() for mm in m):
                    lst.applymap(
                        keys=m,
                        cspec=t2c(XhtmlText),
                        operation_in_row_key=lambda l, k: null2htmlv(l[k], k))
                    res_key = lst.merge_cols(m, ', ', ', ')
                    lst = lst.cols([res_key] + list_minus(lst.keys(),
                                                          [res_key]))
            return lst.to_xhtml(null_val,
                                table_class,
                                null_class,
                                td_classes,
                                th_classes,
                                escape,
                                float_fmt,
                                javascript_click,
                                **kwargs)

        def js(l):
            if '__url__' in l.keys():
                if javascript_click:
                    return (' onmouseover="this.style.cursor=\'pointer\';'
                            ' if (\'mix-blend-mode\' in document.head.style)'
                            ' { this.className=\'clickable\'; }"'
                            ' onmouseout="this.className=\'\'"'
                            ' onclick="window.location = \'%s\';"'
                            % l['__url__'])
                else:
                    return (' onmouseover="'
                            'if (\'mix-blend-mode\' in document.head.style)'
                            ' { this.className=\'clickable\'; }"'
                            ' onmouseout="this.className=\'\'"')
            else:
                return ''

        def a(l, s):
            if ('__url__' in l.keys()) and (not is_null(l.__url__)):
                return f'<a href="{l.__url__}">{s}</a>'
            else:
                return s

        return '<table%s>\n%s\n</table>' % (
            class_attr(table_class),
            '\n'.join(
                ['<thead>']
                + ['<tr>\n%s\n</tr>' % '\n'.join(
                    '    <th%s>%s</th>' % (
                        class_attr(th_classes.get(k, None)),
                        _null2htmlv(k, k))
                    for k in self.keys() if is_normal_key(k))]
                + ['</thead>']
                + ['<tbody>']
                + ['<tr%s>\n%s\n</tr>' % (
                    # setting class with javascript for graceful degradation
                    js(l),
                    '\n'.join(
                        '    <td%s>%s</td>' % (
                            class_attr(td_classes.get(k, None)),
                            a(l, _null2htmlv(l[k], k)))
                        for k in self.keys() if is_normal_key(k)))
                 for l in self.dicts()] +
                ['</tbody>']))

    def to_latex(self, table_class='tabular', escape=True,
                 **kwargs):
        do_escape = self._do_escape(escape)
        # alternative: save to csv and use \pgfplotstabletypeset
        return '\\begin{%s}{ |%s| }\n%s\n\\end{%s}' % (
            table_class, '|'.join('l' for _ in self.keys()),
            '\n'.join([r'  \hline'] +
                      [r'  %s \\' % ' & '.join(escape_latex(k)
                                               for k in self.keys())]
                      + [r'  \hline']
                      + [r'  %s \\' % ' & '.join(
                          null2latexv(l[k], k, escape=do_escape(k), **kwargs)
                          for k in self.keys()) for l in self.dicts()] +
                      [r'  \hline']), table_class)

    def to_navigate(self, keys, *,
                    sort_keys=True,
                    file_loader=lambda f: f + ' (loader not implemented)',
                    save_group: SaveGroup) -> None:
        """Hierarchical navigation based on template files.

        """
        from .._navigate.saveable_group import SaveableGroup
        enl = SaveableGroup(self, keys, sort_keys, file_loader)

        def gen_index(save_group_: SaveGroup,
                      group_: ListGroup,
                      path: list):
            with save_group_ as sg:
                if len(path) > 0:
                    lst = enl.get_lst(path)
                    if lst is None:
                        return
                    sg.save_file(enl.group, keys, path, lst)
                if len(path) < len(keys):
                    for k, v in group_.items():
                        gen_index(sg, v, path + [k])

        gen_index(save_group, enl.group, path=[])

    # size
    def _compute_length(self, assumed=None):
        return dict_unique_value(dict_union(
            {k: len(self._pdct[k]) for k in self.keys()},
            ({} if assumed is None else {'<assumed>': assumed})))
        # return unique_value([len(self._pdct[k]) for k in self.keys()] +
        #                        assumed)

    # convert
    def __to_repr__(self, **kwargs):
        # return 'ListDict(%s)' % ''.join('\n  %s (%s):\n    %s' %
        #                                 (k,self.dspecs[k], v)
        #                                 for k, v in self._pdct.items())
        return ('lacro.collections.listdict.' * kwargs['add_module'] +
                'ListDict(keys=%s, vals=%s, cspecs=%s, nrows=%d)' % (
                    to_repr(list(self.keys()), **kwargs),
                    '[%s]' % ', '.join(
                        array_torepr(v, **kwargs)
                        for v in self._pdct.values()),
                    to_repr(list(self.dspecs.values()), **kwargs),
                    self.nrows))

    def __repr__(self):
        return to_repr(self)

    def __str__(self):
        return self.__tostr__()

    def __tostr__(self,
                  max_len=100,
                  float_fmt='{:.2e}',
                  show_header=True,
                  null_val='null',
                  preserve_str=True,
                  trunc=False,
                  nrows=False,
                  **kwargs):
        """Return string representation of the ListDict object

        Parameters
        ----------
        max_len : int
            Maximum column width. If ``trunc=False``, this only affects the
            whitespace padding. If ``trunc=True``, header and content are
            truncated to fit the width.
        float_fmt : str
            Format string for floating point numbers.
        show_header : str
            Whether or not to include the header.
        null_val : str
            Representation of null values.
        preserve_str : str
            If False, strings are printed with ``repr``, i.e. they are
            enclosed with single quotes.
        trunc : str
            Whether or not to truncate header and content to have a maximum
            width of max_len.
        nrows : int
            Number of rows to print
        kwargs : dict
            Additional keyword arguments passed to
            ``lacro.string.misc.to_strp``.

        Returns
        -------
            String representation of the ListDict object.

        """
        if nrows:
            return self.nrows

        def mytrunc(v):
            return strunc(v, max_len) if trunc else v

        body = {k: (['{}'.format(k)] +
                    [' '.join(null2str(v, k, null_val, float_fmt=float_fmt,
                                       preserve_str=preserve_str, **kwargs)
                              for v in self._pdct[k][i, ...].flat)
                     for i in range(self.nrows)])
                for k in self.keys()}
        body1 = {k: [[mytrunc(vv) for vv in v.split('\n')] for v in vs]
                 for k, vs in body.items()}
        maxh = [maxdef((len(vs[i]) for k, vs in body1.items()), 1)
                for i in range(self.nrows + 1)]
        body2 = {k: lists_concat([v + [''] * (maxh[i] - len(v))
                                  for i, v in enumerate(vs)])
                 for k, vs in body1.items()}
        # length of largest factor level, even if that level is not present in
        # the list
        dtyp = {k: ([null2str(v, k, null_val, float_fmt=float_fmt,
                              preserve_str=preserve_str, **kwargs)
                     for v in cspec.ttype] if is_subset_of_type(cspec.ttype)
                    else []) for k, cspec in self.dspecs.items()}
        maxw = {k: (min(
            max_len,
            max(maxdef((maxdef((len(mytrunc(vv))
                                for vv in str(v).split('\n')), 0)
                        for v in dtyp[k]), 0),
                max(len(v) for v in vs)))
            if len(vs) > 0 else 0) for k, vs in body2.items()}

        def get_padformat(k, padval):
            return '{:%s%s%d}' % (
                padval,
                '>' if self.dspecs[k].basetype in [np.float32,
                                                   np.float64,
                                                   np.int32,
                                                   np.int64] else '<',
                maxw[k])

        return '%s%s' % (
            '\n'.join((('-' if (i < maxh[0]) else ' ')
                       .join((get_padformat(k, '-' if (i < maxh[0]) else '')
                              .format(body2[k][i])) for k in self.keys()))
                      for i in range(maxh[0] * (1 - bool(show_header)),
                                     cast(int, np.sum(maxh, dtype=np.int64)))),
            ('%s(empty list)' % ('\n' if show_header else ''))
            if self.nrows == 0 else '')
        # return '\n'.join(
        #    ' '.join(' '.join('{}: {:<4}'.format(k, ('null' if is_null(v)
        #                                             else v))
        #                      for v in self._pdct[k][i,...].flat)
        #             for k in self.keys())
        #    for i in range(self.nrows))

    @property
    def info(self):
        return items_2_str([
            ('Keys', '\n'.join(('%r %s' % (k, v)
                                for k, v in self.dspecs.items()))),
            ('Rows', self.nrows)], isstr=True, width='auto')

    def __hash__(self):
        return hash((tuple(self.keys()),
                     tuple(array_totuple(v) for v in self._pdct.values()),
                     self.dspecs))

    def __eq__(self, other):
        if not isinstance(other, ListDict):
            return False
        else:
            return (((self.keys(), self.dspecs) == (other.keys(),
                                                    other.dspecs)) and
                    all(ne.nan_equal(self[k], other[k]) for k in self.keys()))

    # -----------------
    # - conversion to -
    # -----------------
    def row_sorting_key(self, keys, null_small=False):
        self.assert_keys_exist(keys)
        return lambda i: tuple(NullComparable(self[k][i], null_small)
                               for k in keys)

    def coldes_sorting_keys(self, keys, null_small=False):
        self.assert_keys_exist(keys)
        return (tuple(NullComparable(v, null_small=null_small) for v in V)
                for V in itertools.product(*[self.assure_subset_of_type(k)
                                             for k in keys]))

    def dicts(self, expanded_keys=[], null_val='<preserve>') -> \
            Iterator[GetattrHashableOrderedDict]:
        self.assert_keys_exist(expanded_keys)
        for i in range(self.nrows):
            for i1 in range(max([1] + [len(self[k][i])
                                       for k in expanded_keys])):
                yield self.row_dict(i, i1, expanded_keys, null_val)

    def tuples(self, keys):
        return eqziplist(self.values(keys), self.nrows)

    def to_dataframe(self):
        from pandas import DataFrame
        return DataFrame.from_dict(OrderedDict((k, self.to_series(k))
                                               for k in self.keys()))

    def to_series(self, key):
        from pandas import Categorical, Series
        if is_subset_of_type(self.dspecs[key].ttype):
            return Series(Categorical(self[key],
                                      categories=list(self.dspecs[key].ttype),
                                      ordered=False))
        else:
            return self[key]

    def row_dict(self, i, i1=0, expanded_keys=[], null_val='<preserve>') -> \
            GetattrHashableOrderedDict:
        def repn(v): return (
            v if ((null_val == '<preserve>') or (not is_null(v)))
            else null_val)

        return GetattrHashableOrderedDict(
            (k, repn((self[k][i][i1] if i1 < len(self[k][i]) else '')
                     if k in expanded_keys else
                     (self[k][i] if i1 == 0 else ''))) for k in self.keys())

    def row_tuple(self, i):
        return tuple(self[k][i] for k in self.keys())

    def mapping(self, old, new, injective=True, remove_duplicates=False):
        res = OrderedDict(
            items_unique(zip(*(self.cols([old, new]).removed_duplicates()
                               if remove_duplicates else
                               self).values([old, new]))))
        if injective:
            dict_assert_injective(res)
        return res

    # -------------------
    # - conversion from -
    # -------------------
    @staticmethod
    def from_dicts(dicts: SizedIterable[Dict[Hashable, object]],
                   dspecs: Optional[Dict[Hashable, object]] = None,
                   null_val: object = None,
                   allowed_duplicates: Sequence[Iterable[object]] = []) \
            -> 'ListDict':
        if dspecs is None:
            dspecs = _get_dicts_dspecs(dicts, allowed_duplicates)
        asserted_of_type(dspecs, [dict, OrderedDict,
                                  HashableOrderedDict])

        _assert_dicts_consistent(dicts, dspecs)

        def v2null(v, cd): return (v if null_val is None or v != null_val else
                                   get_null(cd.elemtype))

        vals = [array_of_shape([v2null(l[k], cd) for l in dicts],
                               shape=(len(dicts),) + cd.shape1,
                               dtype=to_dtype(cd.elemtype, k),
                               msg='Key %r\n' % (k,))
                for k, cd in dspecs.items()]
        # consistency also checked in "add_cols" inside constructor
        return ListDict(vals=vals,
                        keys=dspecs.keys(),
                        cspecs=dspecs.values(),
                        nrows=len(dicts)).assert_keys_consistent()

    @staticmethod
    def from_dataframe(dataframe, dspecs=None):
        def to_cspec(k):
            if dataframe[k].dtype.name == 'category':
                import pandas as pd
                if type(dataframe[k].cat.categories) == pd.Int64Index:
                    # dataframe[k].cat.categories contains python integers
                    return v2c(dataframe[k].cat.categories, basetype=np.int64,
                               convert=True)
                # elif False:
                #     return v2c(dataframe[k].cat.categories)
                else:
                    raise ValueError('Unknown pandas category type: %s' %
                                     type(dataframe[k].cat.categories))
            else:
                return t2c(dataframe[k].dtype.type)

        def to_value(k):
            if dataframe[k].dtype.name == 'category':
                assure_subset_of_type2(dspecs, k)
                return dataframe[k].astype(to_dtype(dspecs[k].elemtype),
                                           copy=False).values
            else:
                assure_no_subset_of_type2(dspecs, k)
                return dataframe[k].values

        if dspecs is None:
            dspecs = OrderedDict((k, to_cspec(k)) for k in dataframe.keys())
        else:
            lists_assert_equal([list(dspecs.keys()),
                                list(dataframe.keys())],
                               check_order=False)

        return ListDict(vals=[to_value(k) for k in dspecs.keys()],
                        keys=dspecs.keys(),
                        cspecs=[dspecs[k] for k in dspecs.keys()],
                        nrows=len(dataframe))

    @staticmethod
    def from_mapping(mapping: dict,
                     old: Hashable,
                     new: Hashable,
                     ocspec: ColumnDescriptor,
                     ncspec: ColumnDescriptor) -> 'ListDict':
        return ListDict(
            keys=[old, new],
            vals=[array_of_shape(
                list(mapping.keys()),
                shape=(len(mapping),),
                dtype=to_dtype(ocspec.basetype)),
                array_of_shape(
                    list(mapping.values()),
                    shape=(len(mapping),),
                    dtype=to_dtype(ncspec.basetype))],
            cspecs=[ocspec, ncspec], nrows=len(mapping))

    @staticmethod
    def from_dir(nvs: dict,
                 dirname: str,
                 name_clister: Sequence[
                     Tuple[str, Callable[[dict, str],
                                         Sequence[Tuple[object, str]]]]],
                 v2list):
        """
        traverse directory tree and set dictionary entries according to parent
        directories
        """
        if len(name_clister) == 0:
            return v2list(nvs, dirname)
        else:
            n, clister = name_clister[0]
            return lists_concat(
                [ListDict.from_dir(
                    OrderedDict(list(nvs.items())
                                + ([(n, cval)] if n != '<dummy>' else [])),
                    os.path.join(dirname, cdir),
                    name_clister[1:], v2list)
                 for cval, cdir in clister(nvs, dirname)])

    # ------
    # - io -
    # ------
    @staticmethod
    def load_csv(filename, dspecs=None, ascii_nul_replacement=None,
                 skip_comments=False, skip_empty=False, null_val=None,
                 pandas=False, fieldnames=None, **kwargs):
        # the csv docs: "... currently some issues regarding ASCII NUL
        # characters ... restrictions will be removed in the future."
        with filtered_open(filename,
                           ascii_nul_replacement=ascii_nul_replacement,
                           skip_comments=skip_comments,
                           skip_empty=skip_empty) as f:
            if pandas:
                import pandas
                df = pandas.read_csv(f, dtype=str,
                                     na_values=[get_null(str)
                                                if null_val is None
                                                else null_val],
                                     keep_default_na=False,
                                     names=fieldnames,
                                     **kwargs)
                if dspecs is None:
                    dspecs = OrderedDict((k, t2c(str))
                                         for k in df.keys())
                return ListDict.from_dataframe(
                    df.fillna(string_null),
                    dspecs=dspecs)

            reader = _MultiHdDictReader(f, fieldnames=fieldnames, **kwargs)
            # convert to list to read length. reading raises "I/O operation on
            # closed file" outside "with".
            dicts = list(reader)

            try:
                if len(dicts) == 0:
                    if dspecs is None and reader.fieldnames is None:
                        raise ValueError('File empty but neither dspecs nor '
                                         'fieldnames specified')
                if dspecs is None:
                    dspecs = OrderedDict((k, t2c(str))
                                         for k in reader.fieldnames)
                return ListDict.from_dicts(dicts, dspecs, null_val=null_val)
            except ValueError as e:
                reraise(ValueError('The following error occured while '
                                   'converting file "%s" into a ListDict.\n%s'
                                   % (filename, e)))

    def save_csv(self, filename, float_fmt='{:.2e}', split_keys=[],
                 expanded_keys=[], null_val=None, permissions=None, **kwargs):
        list_assert_disjoint(split_keys, expanded_keys)
        with open_chmod(filename, 'w', permissions=permissions,
                        optional=True, encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.keys(), **kwargs)
            writer.writeheader()
            writer.writerows(
                {k: null_val if (null_val is not None and is_null(v)) else
                 to_strp(v, float_fmt=float_fmt) for k, v in l.items()}
                for l in (self
                          .splitted(split_keys, '\n')
                          .dicts(expanded_keys=split_keys + expanded_keys)))

    @staticmethod
    def cached_dictlist(operation, filename, version, dspecs=None, mkdir=False,
                        rm_old=True):
        def op():
            res = operation() if dspecs is None else operation(dspecs)
            res.save_csv(filename)
            return res

        return versioned_file(
            op, filename, version,
            alternative_operation=lambda: ListDict.load_csv(
                filename, dspecs=dspecs),
            mkdir=mkdir, rm_old=rm_old)

    # not used for hashing because numpy can not hash floating point arrays
    # def readonly(self):
    #     res = self.view()
    #     for k in res.keys():
    #         res[k] = res[k].view()
    #         res[k].flags.writeable = False
    #     return res


class ReadonlyListDict:

    def __init__(self, lst: ListDict) -> None:
        self._lst = lst

    def __getitem__(self, key):
        res = self._lst[key].view()
        res.flags.writeable = False
        return res

    def __getattribute__(self, name):
        if name in ('keys',
                    'dspecs',
                    'sorting_keys_array',
                    'added_cols_applymap',
                    'assert_keys_exist',
                    'assert_numeric',
                    'assure_subset_of_type',
                    '_joined2'):
            return getattr(self._lst, name)
        else:
            return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name in ('_lst',):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(name)


def unique_metas(metass, keys=None):
    return np.array(
        [dict(m) for m in unique_value(
            tuple(HashableDict(c if keys is None else
                               dict_intersect(c, keys))
                  for c in R) for R in metass)],
        dtype=dict)


def _appended_collapsed_depth(
        group: Union[ListDict, ListGroup],
        collapsed_depth_key='__collapsed_depth__',
        collapsed_depth_reduction_key='__collapsed_depth_reduction__',
        rec=0,
        __collapsed_depth__=0):
    if type(group) == ListDict:
        redux = unique_value(group[collapsed_depth_reduction_key])
        assert not is_null(redux)
        assert rec - redux > 0
        return group.added_col(collapsed_depth_key,
                               t2c(np.int64),
                               (__collapsed_depth__ if redux == 0 else
                                (rec - redux)),
                               'rep_R')
    else:
        assert type(group) == ListGroup
        if len(group) > 1:
            collapsed_depth1 = rec + 1
        else:
            collapsed_depth1 = __collapsed_depth__
        return ListGroup(
            ((k, _appended_collapsed_depth(v,
                                           collapsed_depth_key,
                                           collapsed_depth_reduction_key,
                                           rec + 1,
                                           collapsed_depth1))
             for k, v in group.items()),
            group.cspec)


def _assert_dicts_consistent(lst, dspecs):
    from lacro.statistics.linmodel.covariate import Level, LevelAnd, LevelSum
    if type(lst) != list:
        raise ValueError('Variable of type "%s" is not an instance of '
                         '"DictList"' % type(lst))

    keys = dspecs.keys()
    for l in lst:
        try:
            missing_keys = set(keys) - set(l.keys())
            unexpected_keys = set(l.keys()) - set(keys)
        except Exception as e:
            raise ValueError('%s\n---\n%s\n---\n%s' %
                             (e, type(lst), strunc(repr(lst), 2000)))
        if set(l.keys()) != set(keys):
            raise ValueError(
                'Specified keys conflict with keys in row. Keys: %s\n'
                'Coldes: %s\nRow: %s\n\nMissing: %s\nUnexpected: %s' % (
                    iterable_2_repr(keys),
                    iterable_2_repr(dspecs.values()),
                    ' '.join('%s: %s' % (k, v) for k, v in l.items()),
                    iterable_2_repr(missing_keys),
                    iterable_2_repr(unexpected_keys)))
        for k in l.keys():
            asserted_of_type(
                k,
                [str, LevelAnd, LevelSum, Level],
                'Type of key (not necessarily its value) is of unknown type. '
                'All keys:\n%s\n\n' % (','.join(str(k) for k in l.keys())))


def _rowdict_2_str(row, fill_text='', is_compact=True, max_key_length=1000,
                   max_value_length=1000, head_keys=[], tail_keys=[],
                   hide_null=False, re_key='.*', value_fill_len=0,
                   float_len=5):
    import re
    keys = list_sorted(row.keys(), head_keys, tail_keys)

    if type(max_value_length) == int:
        def max_value_length(k, v=max_value_length): return v

    if type(value_fill_len) == int:
        def value_fill_len(k, v=value_fill_len): return v

    if type(float_len) == int:
        def float_len(k, v=float_len): return v

    def value_fill_len(k, v=value_fill_len): return (v(k) if v(k) >= 0 else
                                                     max_value_length(k) - v(k))

    def format_key(k):
        return ('' if max_key_length == 0 else
                '%s: ' % strunc(str(k), max_key_length))

    # format_value = lambda v,k: (('%g'%v)[:max_value_length(k)]
    #                            if ne.is_numeric(v) else
    #                            strunc(str(v),max_value_length(k)))
    # float format: %(len, signif_digits)g

    def format_value(v, k):
        return ('%d' % v if ne.is_int(v) else
                ('%%.%df' % max(1, float_len(k) - 2)) % v
                if ne.is_float(v) else
                strunc(str(v), max_value_length(k)))

    # format_value = lambda v,k: ('%3.3g'%v if ne.is_numeric(v) else
    #                            strunc(str(v),max_value_length(k)))

    return (fill_text +
            (('' if max_key_length == 0 else
              (' ' if is_compact else '\n' + fill_text))
             .join(('%%s%%%ds' % value_fill_len(k)) %
                   (format_key(k), format_value(row[k], k))
                   for k in keys if (re.match(re_key, k) and
                                     (not hide_null or
                                      not is_null(row[k], k))))))


def _get_dicts_dspecs(dicts: SizedIterable[Dict[Hashable, object]],
                      allowed_duplicates: Sequence[Iterable[object]] = []):
    if type(dicts) != list:
        raise ValueError('List not of type "list" or "DictList"\n%s\n"%s"' %
                         (type(dicts), strunc(repr(dicts), 200)))
    if len(dicts) == 0:
        raise ValueError('Guessing column descriptors requires list with at '
                         'least one entry')
    try:
        def get_type(k):
            types = set(type(dict_get(l, k)) for l in dicts)
            try:
                return SetOfTypes(
                    allowed_duplicates[list(map(set, allowed_duplicates))
                                       .index(types)])
            except ValueError:
                if len(types) > 1:
                    raise ValueError('Multiple possible types for column "%s" '
                                     'detected: "%s"\n\n%s\n\n'
                                     'Allowed duplicates: %s' %
                                     (k,
                                      iterable_2_repr(types),
                                      ' '.join(repr(l[k]) for l in dicts),
                                      allowed_duplicates))
                return single_element(types)

        return OrderedDict((k, t2c(get_type(k))) for k in dicts[0].keys())
    except Exception as e:  # KeyError as e:
        reraise(ValueError(f'{e.__class__.__name__}, {e}\n'
                           f'---\n'
                           f'dicts type: {dicts.__class__.__name__}\n'
                           f'---\n'
                           f'dicts contents: {strunc(repr(dicts), 200)}'))


class _MultiHdDictReader(csv.DictReader):

    def __init__(self, *args, fieldnames=None, **kwargs):
        if type(fieldnames) == int:
            # number of header lines
            super().__init__(*args, **kwargs)
            assert fieldnames > 0
            self._fieldnames = ['\n'.join(l) for l in eqzip(
                *[next(self.reader) for _ in range(fieldnames)])]
            self.line_num = self.reader.line_num
        else:
            super().__init__(*args, fieldnames=fieldnames, **kwargs)
