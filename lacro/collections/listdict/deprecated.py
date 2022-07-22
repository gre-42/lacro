# -*- coding: utf-8 -*-
import copy
import itertools
from collections import OrderedDict

import numpy as np

from lacro.assertions import (asserted_of_type, lists_assert_disjoint,
                              lists_assert_equal, set_assert_subset)
from lacro.collections import (Dict2Object, HashableOrderedDict,
                               dict_intersect, dict_union, dict_unite,
                               dict_updated, list_minus, lists_concat)
from lacro.iterators import eqzip
from lacro.statistics.linmodel.covariate import (Level, LevelAnd,
                                                 LevelCombinations, LevelSum)
from lacro.statistics.linmodel.covmath import simplify

from ._collections.multiple import join as mjoin
from ._collections.onetwo import ListDict
from ._elements.column_descriptor import is_subset_of_type, t2c

# returns a Dict2Object with the following variables
#	lst: the indicators
#	groups: in the same order as "keys", s.t. this works: zip(keys+['const']+[polynomials], glm_groups)
#	src_groups: "glm_groups" and all keys those groups came from, even if those keys were in subs_rmkeys
#	keys: lists_concat(lists_concat(glm_groups))
#		all keys that still exist and their newly created copies, in that order
#	src_keys: lists_concat(lists_concat(glm_src_groups))


def indicators_deprecated(lst, keys=None, subs_rmkeys=None, merge_rmkeys=None, collapse2=False, subtract_last=False, add_const=False, filter_categories=False, collapse2_ok=False, div_n=False):
    if collapse2 and not collapse2_ok:
        raise ValueError('Do not use "collapse2", use "subtract_last" instead')

    contrast_elements = OrderedDict()
    k2const = OrderedDict()
    glm_groups = {}

    # --------
    # - copy -
    # --------
    # copy for merge_cols / add_columns
    lst = copy.copy(lst)

    # ---------
    # - specs -
    # ---------
    if keys is None:
        keys = lst.keys()

    specs = [{'type': 'auto' if filter_categories else 'categ', 'name': k} if type(k) == str else k for k in keys]
    del keys

    # -------------
    # - auto subs -
    # -------------
    # done before "merge" to allow for easier error checking
    specs = [dict_updated(s, {'type': 'categ'}) if (s['type'] == 'auto' and is_subset_of_type(lst.dspecs[s['name']].ttype)) else s for s in specs]
    #specs = [dict_updated(s,{'type':'fixed'}) if (s['type']=='auto' and ne.is_numeric(lst.dspecs[s['name']].ttype)) else s for s in specs]
    specs = [dict_updated(s, {'type': 'fixed'}) if s['type'] == 'auto' else s for s in specs]
    assert(all([s['type'] != 'auto' for s in specs]))  # all([]) instead of all() s.t. "del specs" works in cython

    # ---------
    # - ncomb -
    # ---------
    specs = [dict_union(s, {'ncomb': [1, 2]}) if ((s['type'] in {'merge', 'categ'}) and ('ncomb' not in s)) else s for s in specs]
    # make sure that "ncomb" exists iff the type is in {'merge', 'categ'}
    [asserted_of_type(s['ncomb'], list) if (s['type'] in {'merge', 'categ'}) else lists_assert_disjoint([s.keys(), ['ncomb']], msg='Only "merge" and "categ" may contain an "ncomb" entry\n') for s in specs]

    # ---------------------
    # - subs before merge -
    # ---------------------
    for k, ncomb in [(s['name'], s['ncomb']) for s in specs if s['type'] == 'categ']:
        dict_unite(contrast_elements, {k: Level.C(k, LevelCombinations.C(ncomb, lst.assure_subset_of_type(k)))})

    # ---------
    # - merge -
    # ---------
    # this modifies the list in-place
    merge_src_keys = sorted(set(lists_concat([s['names'] for s in specs if s['type'] == 'merge'])))  # lists_concat([]) instead of lists_concat() s.t. "del specs" works in cython
    lists_assert_disjoint([[s['name'] for s in specs if s['type'] == 'categ'], merge_src_keys])
    # if a key is already in the "categ" specs, this will create a duplicate which will be found later in the "lists_assert_disjoint"

    def merge(names, ncomb):
        # LevelAnd must be inside (not ouside) of Level to match naming of column
        # column: (A&B).(x&y)LevelAndLevelAnd
        old_types = Level.C(LevelAnd(names), LevelAnd(LevelCombinations.C(ncomb, lst.assure_subset_of_type(k)) for k in names))
        #old_types = LevelAnd(Level.C(k,lst.assure_subset_of_type(k)) for k in names)
        res = lst.merge_cols(names, subset_of_type=True, key_sep='&', value_sep='_', rm_keys=True, cat_strings=False)
        dict_unite(contrast_elements, {res: old_types})
        return res
    specs = [s if s['type'] != 'merge' else {'name': merge(s['names'], s['ncomb']), 'onesamp':True, 'type':'categ'} for s in specs]
    # all keys that were really merged (not those that were "alone" and thus retained)
    # works because the merged key's name differs from the source names
    merge_notInSubs_keys = list_minus(merge_src_keys, [s['name'] for s in specs if s['type'] == 'categ'], must_exist=False)
    if merge_rmkeys is None:
        # do not remove keys that were added to the "categ" specs. duplicate check follows later, in "lists_assert_disjoint"
        merge_rmkeys = merge_notInSubs_keys
    set_assert_subset(merge_src_keys, merge_rmkeys)

    # --------
    # - poly -
    # --------
    poly_specs = [k for k in specs if k['type'] == 'poly']
    poly_keys = [k['name'] for k in poly_specs]

    # -----------------------------------
    # - subs_keys / subs_rmkeys / const -
    # -----------------------------------
    subs_specs = [k for k in specs if k['type'] == 'categ']
    subs_keys = [k['name'] for k in subs_specs]
    subs_onesamp = [k.get('onesamp', False) == True for k in subs_specs]
    lst.assert_subset_of_type(subs_keys)  # is also checked later in gV

    if subs_rmkeys is None:
        subs_rmkeys = subs_keys
    set_assert_subset(subs_keys, subs_rmkeys)

    gV = lst.assure_subset_of_type

    dict_unite(k2const, OrderedDict((k, gV(k)[-1] if (subtract_last and len(gV(k)) > 1) else None) for k in subs_keys))

    # ------------
    # - constant -
    # ------------

    # Why it makes sense to set "constant_key" to a sum (and not a product) of the subtracted levels
    #
    # uvw: Levels to be changed
    # o: All other constants
    # Xb = y
    # Solve for ow
    # 	dummy vars :   u       v       1
    # 	meaning    : (u-w)   (v-w)    ow
    # 	variable   :  x1      x2       c
    # XCd = y
    # Solve for ov
    # 	dummy vars :   u       w       1
    # 	meaning    : (u-v)   (w-v)    ov
    # 	calculation:  x1    c-x1-x2    c
    # 	contrast C :   1      -1       0
    # 	               0      -1       0
    # 	               0       1       1
    # Convert d -> b
    # 	b = Cd
    # 	=> ow = ov + (w-v)
    # 		=> define ow = o+w and ov=o+v

    #constant_key = simplify(LevelAnd(Level.C(k,v) for k,v in k2const.items()))
    constant_key = simplify(LevelSum((Level.C(k, v), 1) for k, v in k2const.items())) if subtract_last else None
    # print(k2const)
    # print(constant_key)
    if add_const:
        assert subtract_last
        lst.add_col(constant_key, t2c(np.int64), np.ones(lst.nrows, dtype=np.int64))
        specs = specs + [{'type': 'fixed', 'name': constant_key}]

    # --------------
    # - fixed_keys -
    # --------------
    fixed_specs = [k for k in specs if k['type'] == 'fixed']
    fixed_keys = [k['name'] for k in fixed_specs]

    # ---------
    # - kkeys -
    # ---------
    kkeys = [s['name'] for s in specs]

    # -------------------------
    # - combined sanity check -
    # -------------------------
    #print(poly_keys, std_keys, keys)
    #assert len(keys) == len(poly_keys) + len(std_keys)
    lists_assert_disjoint([poly_keys, fixed_keys, merge_src_keys])
    lists_assert_disjoint([poly_keys, subs_keys, fixed_keys, merge_notInSubs_keys])  # no "merge_src_keys" to allow for single-key, "dummy" merges
    lists_assert_equal([[k['name'] for k in kk] for kk in [specs, poly_specs + subs_specs + fixed_specs]], check_order=False)
    del specs, subs_specs, fixed_specs

    lst.assert_keys_exist(poly_keys + subs_keys + fixed_keys)

    #group_keys = std_keys

    def child_keys_to_group(keys, onesamp):
        return [[k for k in keys]] if onesamp else [[k] for k in keys]

    # --------
    # - poly -
    # --------
    # simulate "filter_categories==True" by updating "group_keys" but not "std_keys"
    if True:
        # order_2_all_keys: order -> keys (keys before renaming)
        lst.assert_numeric(poly_keys)
        for psc in poly_specs:
            orders = (range(1, -psc['order'] + 1) if psc['order'] < 0 else [psc['order']])
            if list(orders) == [1]:
                dict_unite(glm_groups, {psc['name']: [[psc['name']]]})
                dict_unite(contrast_elements, {psc['name']: psc['name']})
            else:
                levels = ['p%d' % p for p in orders]

                def order_name(p):
                    new_name = Level.C(psc['name'], 'p%d' % p)
                    lst.add_col_applymap(new_name, lst.dspecs[psc['name']], lambda l: l[psc['name']] ** p)
                    return new_name
                lst.add_cols_applymap([Level.C(psc['name'], n) for n in levels], [lst.dspecs[psc['name']]] * len(levels), lambda l: [l[psc['name']] ** p for p in orders])

                dict_unite(glm_groups, {psc['name']: [[Level.C(psc['name'], n) for n in levels]]})
                dict_unite(contrast_elements, {psc['name']: Level.C(psc['name'], LevelCombinations.C([1], levels))})

    # --------
    # - subs -
    # --------
    if True:
        # group_keys: all input keys
        # subs_keys: keys of type "SubsetOfType"

        def nV(k, v): return sum(l[k] == v for l in lst)

        def sub2(V): return V[:-1 if subtract_last else None] if (not collapse2) or len(V) > 2 else [V[0]]

        def ind_key(k, f): return (Level.C(k, LevelSum([(f, 1), (k2const[k], -1)])) if (k2const[k] is not None) else Level.C(k, f))

        def ind_value(k, v, tv): return int(tv == v) * (1 / nV(k, tv) if div_n else 1)
        ikvss = [[(ind_key(k, f), f) for f in sub2(gV(k))] for k in subs_keys]

        if len(subs_keys) > 0:
            # tv: true value (in row). could also be called "rv" (row value)
            # ikv: indicator key (e.g. "sex.f") and value (e.g. "f", the value at which the indicator variable is "1")
            # ikvs: ikv group belonging to a single key
            # ikvss: ikvs groups
            rhs = ListDict.from_dicts(
                [dict(list(eqzip(subs_keys, tvs)) + [(ikv[0], ind_value(k, ikv[1], tv)) for k, tv, ikvs in eqzip(subs_keys, tvs, ikvss) for ikv in ikvs]) for tvs in itertools.product(*[gV(k) for k in subs_keys])],
                dspecs=dict_union(dict_intersect(lst.dspecs, subs_keys),
                            OrderedDict((ikv[0], t2c(np.float64 if div_n else np.int64)) for ikvs in ikvss for ikv in ikvs)))
            lst = mjoin([lst, rhs], joined_keys=subs_keys, list_num_copiess=['1', '*'], order='left')

        # subs_rmkeys can only delete old keys ("assert_keys_exist" test above)
        dict_unite(glm_groups, {k: child_keys_to_group([ikv[0] for ikv in ikvs], osa) for k, osa, ikvs in eqzip(subs_keys, subs_onesamp, ikvss)})

    # -----------
    # - default -
    # -----------
    if True:
        dict_unite(glm_groups, dict(zip(fixed_keys, [[[k]] for k in fixed_keys])))
        # not using constant_key because
        #   1. in the case of _multiple constants, it contains a meaningless sum of levels
        #   2. in the case of a single constant, this constant is generated because of the [1] in LevelCombinations([1,..]..)
        dict_unite(contrast_elements, OrderedDict((k, k) for k in fixed_keys if (not subtract_last or k != constant_key)))

    # --------------
    # - group_keys -
    # --------------
    if True:
        # print(glm_groups)

        glm_src_groups = {k: v if [k] in v else [[k]] + v for k, v in glm_groups.items()}

        glm_groups = [glm_groups[k] for k in kkeys]
        glm_src_groups = [glm_src_groups[k] for k in kkeys]

        glm_keys = lists_concat(lists_concat(glm_groups))
        glm_src_keys = lists_concat(lists_concat(glm_src_groups))

        return Dict2Object(
            lst=lst,
            groups=HashableOrderedDict(eqzip(kkeys, glm_groups)),
            src_groups=glm_src_groups,
            keys=glm_keys,
            src_keys=glm_src_keys,
            contrast_elements=contrast_elements,
            constant_key=constant_key)
