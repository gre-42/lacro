# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Dict, Hashable, List, Sequence, Type, Union

from lacro.collections import (HashableOrderedDict, dict_minus, items_2_str,
                               list_minus)
from lacro.iterators import eqzip

from .._elements.column_descriptor import (ColumnDescriptor, UnknownType,
                                           array_for_coldes, is_subset_of_type,
                                           t2c)
from .._elements.values import null2str
from . import onetwo as ld


class _BaseListGroup(ABC, HashableOrderedDict):

    def __init__(self, dct, cspec_: Union[ColumnDescriptor,
                                          List[ColumnDescriptor]]) -> None:
        super().__init__(dct)
        self._cspec_ = cspec_

    def __str__(self) -> str:
        return self.__tostr__()

    @abstractmethod
    def __tostr__(self, **kwargs) -> str:
        pass

    def __repr__(self) -> str:
        return '%s(%s,\n%r)' % (self.__class__.__name__,
                                super().__repr__(),
                                self._cspec_)

    def applied(self, operation):
        return self.__class__(((k, operation(v)) for k, v in self.items()),
                              self._cspec_)


class ListGroup(_BaseListGroup):

    def __init__(self, dct, cspec: ColumnDescriptor) -> None:
        super().__init__(dct, cspec)

    @property
    def cspec(self) -> ColumnDescriptor:
        return self._cspec_

    def __tostr__(self, **kwargs) -> str:
        return items_2_str(
            ((null2str(k), v) for k, v in self.items()),
            keys=(self.cspec.ttype
                  if (self.cspec is not None and
                      is_subset_of_type(self.cspec.ttype)) else None),
            **kwargs)

    def ungroup(self, *args, **kwargs) -> 'ld.ListDict':
        return ungroup(self, *args, **kwargs)


class ListGroup2(_BaseListGroup):

    def __init__(self, dct, cspecs: List[ColumnDescriptor]) -> None:
        super().__init__(dct, cspecs)

    @property
    def cspecs(self) -> List[ColumnDescriptor]:
        return self._cspec_

    def __tostr__(self, **kwargs) -> str:
        return items_2_str(
            ('(%s)' % ', '.join(map(null2str, ks)), v)
            for ks, v in self.items())


def flatten_groups(groups,
                   keys,
                   keys_coldes,
                   leaf_type: Type[
                       Union[UnknownType,
                             'ld.ListDict',
                             ListGroup]] = UnknownType) -> 'ld.ListDict':
    from . import multiple as lds
    assert len(keys) == len(keys_coldes)
    if 'leaves' in keys:
        raise ValueError('A key of type "leaves" already exists')

    if len(keys) == 0:
        return ld.ListDict(keys=['leaves'],
                           vals=[array_for_coldes([groups], 1, t2c(leaf_type),
                                                  'leaves')],
                           cspecs=[t2c(leaf_type)], nrows=1)
    else:
        return lds.concat(
            [(flatten_groups(v, keys[1:], keys_coldes[1:], leaf_type)
              .added_col(keys[0], keys_coldes[0], k, 'rep_R'))
             for k, v in groups.items()],
            dspecs=HashableOrderedDict(
                (k, c) for k, c in eqzip(keys + ['leaves'],
                                         keys_coldes + [t2c(leaf_type)])))


def unflatten_ungroup(lst: 'ld.ListDict',
                      dspecs: Dict[Hashable, ColumnDescriptor],
                      leaves_keys=[]) -> 'ld.ListDict':
    from . import multiple as lds
    nonleaves_keys = list_minus(lst.keys(), ['leaves'])
    return lds.concat(
        [(ungroup(l['leaves'], leaves_keys,
                  dict_minus(dspecs, nonleaves_keys))
          .added_cols(nonleaves_keys, [lst.dspecs[k] for k in nonleaves_keys],
                      [l[k] for k in nonleaves_keys], 'rep_R'))
         for l in lst.dicts()], dspecs)


def ungroup(group: Union['ld.ListDict', ListGroup],
            keys: Sequence[Hashable],
            dspecs: Dict[Hashable, ColumnDescriptor],
            repeat=True) -> 'ld.ListDict':
    from . import multiple as lds
    if len(keys) == 0:
        assert type(group) == ld.ListDict
        return group
    else:
        return lds.concat(
            [(ungroup(v, keys[1:], dict_minus(dspecs, [keys[0]]), repeat)
              .added_col(keys[0], dspecs[keys[0]], k, 'rep_R'
                         if repeat else 'one_R'))
             for k, v in group.items()], dspecs)


def group_leaf_count(group: Union['ld.ListDict', ListGroup]) -> int:
    """Returns the total number of leaves

    Could also be done using ``len(flatten_groups(...))``, but
    ``flatten_groups`` group_leaf_countrequires specifying column descriptors.
    """
    assert isinstance(group, (ld.ListDict, ListGroup))
    if isinstance(group, ld.ListDict):
        return group.nrows
    else:
        return sum(group_leaf_count(v) for v in group.values())


def group_child_count(group: Union['ld.ListDict', ListGroup]) -> int:
    assert isinstance(group, (ld.ListDict, ListGroup))
    if isinstance(group, ld.ListDict):
        return group.nrows
    else:
        return len(group)
