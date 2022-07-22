# -*- coding: utf-8 -*-
import itertools
from functools import reduce
from typing import Dict, Hashable, Optional, Sequence

import numpy as np

from lacro.array import default_concatenate, repeatE
from lacro.assertions import set_assert_contains, set_assert_subset
from lacro.iterators import eqzip, eqziplist

from . import onetwo as ld


def concat(ldcts: Sequence[ld.ListDict],
           dspecs: Optional[Dict[Hashable, 'ld.ColumnDescriptor']] = None,
           unite_subset_of_type=False,
           unite_metas=False) -> ld.ListDict:
    if len(ldcts) > 0:
        # print(ldcts[0].dspecs)
        unif_coldes = ld.unify_dspecs(
            *[ldct.dspecs for ldct in ldcts],
            unite_subset_of_type=unite_subset_of_type,
            unite_metas=unite_metas)

        if dspecs is None:
            dspecs = unif_coldes
        set_assert_subset(dspecs.keys(), unif_coldes.keys())
        ld.unify_dspecs(unif_coldes, dspecs,
                        unite_subset_of_type=unite_subset_of_type,
                        unite_metas=unite_metas)
    elif dspecs is None:
        raise ValueError('No "dspecs" and empty "ldcts" given to "concat"')
    assert dspecs is not None

    return ld.ListDict(
        keys=dspecs.keys(),
        vals=[default_concatenate(
            [ldct[k] if k in ldct.keys() else
             repeatE(np.asarray(ld.get_null(cd.elemtype,
                                            unknown_is_objnull=True),
                                dtype=ld.to_dtype(cd.elemtype, k)),
                     (ldct.nrows,) + cd.shape1) for ldct in ldcts],
            axis=0,
            shape=(0,) + cd.shape1,
            dtype=ld.to_dtype(cd.elemtype, k))
            for k, cd in dspecs.items()],
        cspecs=dspecs.values(),
        nrows=sum(ldct.nrows for ldct in ldcts))


def product(lsts: Sequence[ld.ListDict]):
    return hconcat(
        [lst.isliced(np.array(r))
         for lst, r in eqzip(
            lsts,
            eqziplist(list(itertools.product(*[range(lst.nrows)
                                               for lst in lsts])),
                      len(lsts)))])
    # return concat([hconcat([l.isliced(np.array([i]))
    #                        for l,i in eqzip(lsts,r)])
    #               for r in itertools.product(*[range(lst.nrows)
    #                                            for lst in lsts])])


def hconcat(lsts: Sequence[ld.ListDict],
            allow_duplicates=False, unite_subset_of_type=False,
            unite_metas=False, keys=None) -> ld.ListDict:
    if keys is not None:
        lsts = [l.cols(keys, keys_must_exist=False) for l in lsts]
    res = reduce(
        lambda A, B: A.added_ldct(B,
                                  allow_duplicates=allow_duplicates,
                                  unite_subset_of_type=unite_subset_of_type,
                                  unite_metas=unite_metas),
        lsts[1:], lsts[0])
    if keys is not None:
        res.assert_keys_exist(keys)
    return res


def join(lsts: Sequence[ld.ListDict],
         joined_keys: Optional[Sequence[Hashable]] = None,
         join_keyss: Optional[Sequence[Sequence[Hashable]]] = None,
         where=None,
         prefixes=None,
         join_type='inner',
         order='sort',
         prefix_affinity='join_type',
         list_num_copiess=None,
         join_num_copiess=None,
         key_num_copiess=None) -> ld.ListDict:
    if len(lsts) == 0:
        raise ValueError('Received no lists to join')

    if (join_keyss is not None) + (joined_keys is not None) != 1:
        raise ValueError('Exactly one out of {join_keyss, joined_keys} '
                         'must be set')

    if joined_keys is not None:
        if len(joined_keys) == 0:
            raise ValueError('"joined_keys" is defined but empty')
        join_keyss = [joined_keys] * (len(lsts) - 1)

    if len(join_keyss) != len(lsts) - 1:
        raise ValueError('len(join_keyss) != len(lsts)-1. '
                         'len(join_keyss) = %d, len(lsts) = %d' %
                         (len(join_keyss), len(lsts)))

    if key_num_copiess is not None:
        raise ValueError('join_keyss and key_num_copiess given')

    if join_num_copiess is not None:
        if list_num_copiess is not None:
            raise ValueError('list_num_copiess and join_num_copiess given')
        if len(join_num_copiess) != len(lsts) - 1:
            raise ValueError('len(join_num_copiess) != len(lsts)-1')
        join_num_copiess = join_num_copiess
    else:
        if list_num_copiess is not None:
            if type(list_num_copiess) == str:
                list_num_copiess = [list_num_copiess] * len(lsts)
            if len(list_num_copiess) != len(lsts):
                raise ValueError('len(list_num_copiess) != len(lsts)')
            list_num_copiess = list_num_copiess
        else:
            list_num_copiess = ['*'] * len(lsts)

        join_num_copiess = list(zip(list_num_copiess[:-1],
                                    list_num_copiess[1:]))

    if prefixes is None:
        prefixes = [None] * len(lsts)

    if len(prefixes) != len(lsts):
        raise ValueError('len(prefixes) != len(lsts)')

    if type(join_type) == str:
        join_type = [join_type] * (len(lsts) - 1)

    if len(join_type) != len(lsts) - 1:
        raise ValueError('len(join_type) != len(lsts-1)')

    if type(order) == str:
        order = [order] * (len(lsts) - 1)

    if len(order) != len(lsts) - 1:
        raise ValueError('len(order) != len(lsts-1)')

    if type(prefix_affinity) == str:
        prefix_affinity = [prefix_affinity] * (len(lsts) - 1)

    if len(prefix_affinity) != len(lsts) - 1:
        raise ValueError('len(prefix_affinity) != len(lsts-1)')
    for prefix_aff in prefix_affinity:
        set_assert_contains(['join_type', 'left', 'right', 'disabled'],
                            prefix_aff)

    result = lsts[0]
    if prefixes[0] is not None:
        if len(lsts) == 1:
            if joined_keys is None:
                raise ValueError('Joining only one list '
                                 '(i.e. not doing any join) while renaming '
                                 'keys requires "joined_keys" to be set')
            keys_to_rename = set(lsts[0].keys()) - set(joined_keys)
        else:
            keys_to_rename = set(lsts[0].keys()) - set(join_keyss[0])
        result = result.renamed_keys(dict([(k, prefixes[0] + k)
                                           for k in keys_to_rename]))

    for (lst, join_keys, next_join_keys, join_tpe, ordr,
         prefix_aff, join_num_copies, prev_prefix, prefix) in eqzip(
            lsts[1:], join_keyss, join_keyss[1:] + [[]], join_type, order,
            prefix_affinity, join_num_copiess, prefixes[0:-1], prefixes[1:]):
        keys_to_rename = set(lst.keys()) - set(join_keys) - set(next_join_keys)
        if prefix is not None:  # renaming only works if type if type(key)==str
            lst = lst.renamed_keys({k: prefix + k for k in keys_to_rename})
        lnc, rnc = join_num_copies
        result = result._joined2(lst, join_keys, join_tpe, ordr, lnc, rnc)
        # list_num_copiess = ([merge_num_copiess([list_num_copiess[0],
        #                                         list_num_copiess[1]])] +
        #                     list_num_copiess)
        for pf in [prev_prefix, prefix]:
            if (pf is not None) and (prefix_aff == pf or
                                     (prefix_aff == 'join_type' and
                                      join_tpe == pf)):
                result.rename_keys({k: pf + k for k in join_keys})

    if where is not None:
        result = result.isliced(rows=where, cols=None)

    return result
