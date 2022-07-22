# -*- coding: utf-8 -*-
from typing import Hashable, Optional, Sequence

import numpy as np

from lacro.collections import dict_minus, dict_union, resolved_tree_path
from lacro.collections.listdict import string_null
from lacro.iterators import unique_value

from .._collections.group import ListGroup, ungroup
from .._collections.onetwo import ListDict, _appended_collapsed_depth
from .._elements.column_descriptor import t2c
from .naming_scheme import renaming_scheme
from .site_url import SiteUrl


class SaveableGroup:

    def __init__(self,
                 lst: ListDict,
                 keys: Sequence[Hashable],
                 sort_keys: bool,
                 file_loader) -> None:
        nsc = renaming_scheme(file_loader)
        la = lst.appliedmap(keys=keys, operation=nsc, cspec=t2c(nsc))

        def gp(l):
            return l.grouped(keys, use_subset_ttype=False, sort_keys=False)

        if sort_keys:
            la.sort(keys)
        la.add_counter('__iid__')
        la = ungroup(_appended_collapsed_depth(gp(la), __collapsed_depth__=1),
                     keys,
                     dict_union(la.dspecs,
                                {'__collapsed_depth__': t2c(np.int64),
                                 '__collapsed_id__': t2c(np.int64)}))
        la.add_col_applymap('__site_url__', t2c(SiteUrl),
                            lambda l: l[keys[l['__collapsed_depth__'] - 1]]
                            .site_url(
                            l, [l[k] for k in keys]))
        la.add_col_applymap('__url__', t2c(str),
                            lambda l: l['__site_url__'].filename)
        self._keys = keys
        self._group = gp(la)
        self._dspecs = la.dspecs

    @property
    def group(self) -> ListGroup:
        return self._group

    def get_lst(self, path: list) -> Optional[ListDict]:
        lst = resolved_tree_path(self._group, path)
        lst = ungroup(lst,
                      self._keys[len(path):],
                      dict_minus(self._dspecs, self._keys[:len(path)]))
        lpg = unique_value(len(path) > lst['__collapsed_depth__'])
        if lpg:
            return None
        # hide links in leaves
        lst.applymap('__url__',
                     operation_in_row=lambda l: string_null
                     if len(path) == l['__collapsed_depth__'] else l['__url__'])
        return lst
