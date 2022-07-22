#!/usr/bin/env pytest
# -*- coding: utf-8 -*-


import os.path
from collections import OrderedDict
from functools import partial
from importlib.util import find_spec

import numpy as np
import pytest

import lacro.collections.listdict.database as db
from lacro.collections.listdict import (BaseCachedListDicts, BaseCachedReprIo,
                                        ListDict, SaveGroupAsTree, closest_key,
                                        date_distance, flatten_groups,
                                        int_null, join, object_null,
                                        string_null, t2c, to_dtype,
                                        unflatten_ungroup, ungroup,
                                        unique_metas, v2c)
from lacro.inspext.app import init_pytest_suite
from lacro.path.pathabs import abspath_of_script_child
from lacro.path.pathfind import files
from lacro.path.pathmod import remove_files
from lacro.string.datetimex import date_format
from lacro.string.misc import to_repr, to_str

init_pytest_suite()


@pytest.fixture
def self():
    iv2c = partial(v2c, basetype=np.int64, convert=True)
    d0 = [{'a': 1, 'b': 2, 'c': 3},
          {'a': 4, 'b': 9, 'c': 8},
          {'a': 4, 'b': 10, 'c': 0}]
    d0n = [{'a': 1, 'b': 2, 'c': 3},
           {'a': 4, 'b': 9, 'c': 8},
           {'a': int_null, 'b': 10, 'c': int_null}]
    d0b = [{'a': 1, 'b': 2, 'c': True},
           {'a': 4, 'b': 9, 'c': False},
           {'a': 4, 'b': 10, 'c': True}]
    d0d = [{'a': 1, 'b': 2, 'c': 3, 'd': 3},
           {'a': 4, 'b': 9, 'c': 8, 'd': 4},
           {'a': 4, 'b': 10, 'c': 0, 'd': 5}]
    self.d1 = ListDict(keys=['a', 'b', 'c'],
                       vals=['<empty>', '<empty>', '<empty>'],
                       cspecs=[t2c(np.float64),
                               t2c(np.int64),
                               t2c(np.int64)],
                       nrows=0)
    self.d1n = ListDict(keys=['a', 'b', 'c'],
                        vals=['<null>', '<null>', '<null>'],
                        cspecs=[t2c(np.float64),
                                t2c(np.int64),
                                t2c(np.int64)],
                        nrows=3)
    self.d2 = ListDict.from_dicts(d0, dspecs={'a': iv2c([1, 4]),
                                              'b': t2c(np.int64),
                                              'c': t2c(np.int64)}).sorted_cols()
    self.d2n = ListDict.from_dicts(d0n, dspecs={'a': iv2c([1, 4]),
                                                'b': t2c(np.int64),
                                                'c': t2c(np.int64)}).sorted_cols()
    self.d3 = ListDict.from_dicts(d0, dspecs={'a': iv2c([1, 4]),
                                              'b': iv2c([2, 9, 10]),
                                              'c': t2c(np.int64)}).sorted_cols()
    self.d4 = ListDict.from_dicts(d0d, dspecs={'a': iv2c([1, 4]),
                                               'b': t2c(np.int64),
                                               'c': t2c(np.int64),
                                               'd': t2c(np.int64)}).sorted_cols()
    ListDict.from_dicts(d0b, dspecs={'a': iv2c([1, 4]),
                                     'b': iv2c([2, 9, 10]),
                                     'c': t2c(np.int64)})
    d4 = [{'x': 10, 'y': 12},
          {'x': 14, 'y': 19}]
    ListDict.from_dicts(d4, dspecs={'x': iv2c([10, 14]),
                                    'y': t2c(np.int64)})
    self.d6 = ListDict.from_dicts([{'x': 'f', 'y': string_null},
                                   {'x': 'g', 'y': 'g'}]).sorted_cols()
    return self


def test_0(self):
    assert str(ListDict.load_csv(
        abspath_of_script_child('multihd.csv'), fieldnames=2)) == '''\
a-b-c
d-e-f
1 2 3
x y z
4 5 6'''
    assert to_str(ListDict.load_csv(abspath_of_script_child('multihd.csv'),
                                    fieldnames=2), show_header=False) == '''\
1 2 3
x y z
4 5 6'''

    assert str(self.d1) == 'a-b-c\n(empty list)'
    assert str(self.d1n) == '''\
---a----b----c
null null null
null null null
null null null'''
    assert to_str(self.d1, show_header=False) == '(empty list)'


def test_1(self):
    assert str(ListDict(keys=[], cspecs=[], vals=[], nrows=2)) == '\n' * 2
    assert str((ListDict(keys=[], cspecs=[], vals=[], nrows=3)
                .added_col('a', t2c(np.float64), 1, kind='one_R'))) == '''\
-------a
1.00e+00
    null
    null'''
    assert str((ListDict(keys=[], cspecs=[], vals=[], nrows=3)
                .added_col('a', t2c(np.float64), 1, kind='rep_R'))) == '''\
-------a
1.00e+00
1.00e+00
1.00e+00'''

    assert to_str((ListDict(keys=[], cspecs=[], vals=[], nrows=3)
                   .added_col('a', t2c(np.float64), 1, kind='rep_R')),
                  max_len=4, trunc=True) == '''\
---a
1.00
1.00
1.00'''
    assert to_str((ListDict(keys=[], cspecs=[], vals=[], nrows=3)
                   .added_col('a', t2c(np.float64), 1, kind='rep_R')),
                  max_len=5, trunc=True) == '''\
----a
1.00e
1.00e
1.00e'''
    assert to_str((ListDict(keys=[], cspecs=[], vals=[], nrows=3)
                   .added_col('a', t2c(np.float64), 1, kind='rep_R')),
                  max_len=7, trunc=True) == '''\
------a
1...+00
1...+00
1...+00'''
    assert to_str((ListDict(keys=[], cspecs=[], vals=[], nrows=3)
                   .added_col('a', t2c(np.float64), 1, kind='rep_R')),
                  max_len=7, trunc=False) == '''\
------a
1.00e+00
1.00e+00
1.00e+00'''

    return
    if True:
        d = [
            {'a': 'y', 'b': '1'},
            {'a': '0', 'b': '0'},
            {'a': 'y', 'b': '2'},
            {'a': 'y', 'b': '6'},
            {'a': 'z', 'b': '6'}]
        D = ListDict.from_dicts(d, dspecs={'a': v2c(['0', 'y', 'z']), 'b': t2c(str)})
        D1 = ListDict.from_dicts([], dspecs={'a': v2c(['0', 'y', 'z']), 'b': t2c(str)})
        D2 = D.added_cols_applymap(['b1', 'b2'], [t2c(np.float64)] * 2,
                                   lambda l: [np.float64(l['b']) + 5,
                                              np.float64(l['b']) + 7])
        D3 = (ListDict.from_dicts(d)
              .auto_subset_ttyped()
              .added_cols_applymap(['b1', 'b2'],
                                   [t2c(np.float64)] * 2,
                                   lambda l: [np.float64(l['b']) + 5,
                                              np.float64(l['b']) + 7]))

        if False:
            print('indicators\n{}\n'.format(indicators(d2, filter_categories=True)))
            print('indicators\n{}\n'.format(indicators(d2, ['a'], filter_categories=True)))
            print('indicators\n{}\n'.format(indicators(D3, [{'type': 'merge', 'names': ['a', 'b']}], filter_categories=True, merge_rmkeys=None)))
            #print(ionly(d2, keys=['a']))
            print(indicators_interact(d2, [dict(keys=['a', {'name': 'b', 'type': 'poly', 'order': -2}]), dict(keys=[{'name': 'c', 'type': 'poly', 'order': 1}])], subtract_last=True))
        if True:
            #print(ionly(d2, keys=['a']))
            #print(Dict2Object(d2=d2, no_sublast=ionly2(d2, key='a')))
            ia = ionly2(d2, key='a', subtract_last=True)
            ib = ionly2(d2, key={'name': 'b', 'type': 'poly', 'order': -2}, subtract_last=True)
            ic = ionly2(d2, key={'name': 'c', 'type': 'poly', 'order': -2}, subtract_last=True)
            #print(Dict2Object(ia=ia, ib=ib))
            # print()
            #print(isum2([ia, ib], subtract_last=True))
            #print(interact2([ia, ib], subtract_last=True))
            #print(isum2([interact2([ia, ib],subtract_last=True),ic],subtract_last=True))
            #print(indicators_expr(d2, PredictorSum.C(PredictorInteract.C('a',{'name':'b','type':'poly','order':-2}), 'c')))
            print(indicators_expr(d2, PredictorInteract.C('a', {'name': 'b', 'type': 'poly', 'order': -2})))
            #print(indicators_expr(d2, PredictorInteract.C('a',{'name':'b','type':'poly','order':-2}), subtract_last=False, add_const=False))
        if True:
            print(indicators_interact(d2, [dict(keys=['a', {'name': 'b', 'type': 'poly', 'order': -2}])], subtract_last=True, add_const=True, hconcat_orig=False))


def test_2(self):
    return
    print(d2.cols(['a', 'b']))
    print(d2.added_col('x', t2c(int), [8, 7]).cols(['a', 'b', 'x']))


def test_3(self):
    return
    import numpy as np
    print(d2.updated_col('a', t2c(np.int64), np.array([0, 1, 5])))
    print(d2.values(['a', 'b']))
    print(d2.values([]))
    print(Slicer(d2)[0:0].values(['a', 'b']))
    print(np.array(Slicer(d2)[0:0].values(['a', 'b'])).shape)
    print(d2.applymap('a', lambda l: {1: 4, 4: 1}[l]))


def test_4(self):
    import numpy as np
    LL = self.d2.added_col('x', t2c(np.ndarray, np.int64, shape1=(2,)), np.array([[8, 7], [5, 6], [9, 1]]))
    LL2 = self.d2.added_col('x', t2c(np.ndarray, np.float64, shape1=(2,)), np.array([[8, 7], [5, np.nan], [9, 1]]))
    assert self.d2.npvaluesT([], dtype=float).shape == (3, 0)
    np.testing.assert_array_equal(self.d2.npvaluesT(['a'], dtype=float), [[ 1.], [ 4.], [ 4.]])
    np.testing.assert_array_equal(self.d2.npvaluesT(['a', 'b'], dtype=float), [[1., 2.], [4., 9.], [4., 10.]])
    np.testing.assert_array_equal(self.d2.npvaluesT(['a', 'b'], [], dtype=float), [[1., 2.], [4., 9.], [4., 10.]])
    np.testing.assert_array_equal(self.d2.npvaluesT(['a', 'b'], []), [[1., 2.], [4., 9.], [4., 10.]])
    np.testing.assert_array_equal(LL.npvaluesT(['a', 'b'], ['x'], dtype=float), [[1., 2., 8., 7.], [4., 9., 5., 6.], [4., 10., 9., 1.]])
    # auto
    np.testing.assert_array_equal(LL.npvaluesT(['a', 'b', 'x'], dtype=float), [[1., 2., 8., 7.], [4., 9., 5., 6.], [4., 10., 9., 1.]])
    #print(self.d2.added_col('x',t2c(list),np.array([[8,7],[5,6]])).npvaluesT(['a','b'], ['x']))
    # meta2T
    np.testing.assert_array_equal(LL.guessed_meta().meta_values(['a', 'b'], []), [{'type': 'scalar', 'name': 'a'}, {'type': 'scalar', 'name': 'b'}])
    np.testing.assert_array_equal(LL.guessed_meta().meta_values(['a', 'b'], [], 'name'), ['a', 'b'])
    np.testing.assert_array_equal(LL.guessed_meta().meta_values(['a', 'b'], [], 'type'), ['scalar', 'scalar'])
    np.testing.assert_array_equal(LL2.is_any_null(['a', 'b', 'x']), [False, True, False])
    # nan
    np.testing.assert_array_equal(self.d2.added_col('x', t2c(np.ndarray, str, shape1=(1,)), np.array([['a', string_null, '0']], dtype=to_dtype(str)).T).is_any_null(['x']), [False, True, False])
    np.testing.assert_array_equal(LL.added_metas({}, {'x': np.array([{'f': 1}, {'f': 3}])}).guessed_meta().meta_values(['a', 'b'], ['x']), [{'type': 'scalar', 'name': 'a'}, {'type': 'scalar', 'name': 'b'}, {'f': 1}, {'f': 3}])
    # vec name
    np.testing.assert_array_equal(LL.added_metas({}, {'x': unique_metas(np.array([[{'name': 1, 'type': 'asd'}, {'name': 3, 'type': 'dt', 'f': 5}], [{'name': 1, 'type': 'asd'}, {'name': 3, 'type': 'dt'}], [{'name': 1, 'type': 'asd'}, {'name': 3, 'type': 'dt'}]]), ['name', 'type'])}).guessed_meta().meta_values(['a', 'b'], ['x']), [{'type': 'scalar', 'name': 'a'}, {'type': 'scalar', 'name': 'b'}, {'type': 'asd', 'name': 1}, {'type': 'dt', 'name': 3}])

    np.testing.assert_array_equal(LL.added_metas({}, {'x': unique_metas(np.array([[{'name': 1, 'type': 'asd'}, {'name': 3, 'type': 'dt'}], [{'name': 1, 'type': 'asd'}, {'name': 3, 'type': 'dt'}], [{'name': 1, 'type': 'asd'}, {'name': 3, 'type': 'dt'}]]))}).guessed_meta().meta_values(['a', 'b'], ['x'], 'name'), np.array(['a', 'b', 1, 3], dtype=object))
    np.testing.assert_array_equal(LL.added_metas({}, {'x': unique_metas(np.array([[{'name': 1, 'type': 'asd'}, {'name': 3, 'type': 'dt'}], [{'name': 1, 'type': 'asd'}, {'name': 3, 'type': 'dt'}], [{'name': 1, 'type': 'asd'}, {'name': 3, 'type': 'dt'}]]), ['name', 'type'])}).guessed_meta().meta_values(['a', 'b'], ['x'], 'name'), np.array(['a', 'b', 1, 3], dtype=object))
    np.testing.assert_array_equal(LL.guessed_meta().updated_metas({}, {'x': np.array([{'f': 1, 'name': 9}, {'g': 8, 'name': 5}])}).meta_values(['a', 'b'], ['x'], 'name'), np.array(['a', 'b', 9, 5], dtype=object))
    assert to_str(LL.guessed_meta().added_ldct(LL.added_metas({}, {'x': np.array([{'f': 1, 'name': 'x_0'}, {'g': 8, 'name': 'x_1'}])}), allow_duplicates=True, unite_subset_of_type=True, unite_metas=True).get_metas(), nice_dict=True, width='auto') == '''a: name: 'a'
   type: 'scalar'
b: name: 'b'
   type: 'scalar'
c: name: 'c'
   type: 'scalar'
x: <f:    1
   name: 'x_0'
   type: 'scalar'
   
   g:    8
   name: 'x_1'
   type: 'scalar'>'''
    # guess v meta
    np.testing.assert_array_equal(LL.guessed_meta()
                                  .meta_values(['a', 'b'], ['x'], 'name'),
                                  ['a', 'b', 'x_0', 'x_1'])
    np.testing.assert_array_equal(LL.colnames([], ['x']), ['x_0', 'x_1'])
    np.testing.assert_array_equal(self.d6.npvaluesT(['y']),
                                  [['<null>'], ['g']])


def test_contracted_1(self):
    assert str(self.d2.contracted_merged(['a'])) == \
                     'a-b--c\n1 2  3\n4 9  8\n  10 0'
    assert str(self.d2n.contracted_merged(['a'])
               .splitted_expand(['b', 'c'])) == \
        '---a-b--c---\n   1 2  3   \n   4 9  8   \nnull 10 null'
    assert str(self.d2n.contracted_merged(['a'], null_val='nü')
               .splitted_expand(['b', 'c'])) == \
        '---a-b--c-\n   1 2  3 \n   4 9  8 \nnull 10 nü'
    assert str(self.d2n.fsliced(lambda l: False)
               .contracted_merged(['a'], null_val='nü')
               .splitted_expand(['b', 'c'])) == \
        'a-b-c\n(empty list)'
    assert str(self.d2.contracted_merged(['a'])
               .splitted(['b', 'c'], '\n')) == \
        'a-b-------c-----\n1 [2]     [3]   \n4 [9, 10] [8, 0]'
    assert to_str(self.d2.contracted_merged(['a'])
                  .splitted(['b', 'c'], '\n'),
                  preserve_str=False) == \
        "a-b-----------c---------\n1 ['2']       ['3']     \n4 ['9', '10'] ['8', '0']"
    assert to_str(self.d2.contracted_merged(['a'])
                  .splitted(['b', 'c'], '\n'),
                  preserve_str=False) == \
        "a-b-----------c---------\n1 ['2']       ['3']     \n4 ['9', '10'] ['8', '0']"
    assert str(self.d2.contracted(['a'])) == \
        'a-b-------c----\n1 [2]     [3]  \n4 [ 9 10] [8 0]'


def test_contracted_2(self):
    assert to_str(self.d6.sorted(['y']).contracted(['y']),
                  preserve_str=False) == \
        "y---x---------\n'g' <'g', 'f'>"

    assert str(self.d2.contracted(['a']).expanded(
        ['b', 'c'], [t2c(np.int64), t2c(np.int64)])) == \
        '---a--b-c\n   1  2 3\n   4  9 8\nnull 10 0'


def test_6(self):
    assert str(self.d6.merged_cols(['x', 'y'], res_key='asd')[0]) == \
                     'asd-----\nf.<null>\ng.g     '
    assert str(self.d6.merged_cols(['x', 'y'])[0]) == \
        'x.y-----\nf.<null>\ng.g     '
    assert str(self.d6.merged_cols(['x', 'y'], rm_keys=False)[0]) == \
        'x-y----x.y-----\nf null f.<null>\ng g    g.g     '
    L = self.d6.view()
    L.merge_cols(['x', 'y'])
    assert str(L) == 'x.y-----\nf.<null>\ng.g     '


def test_7(self):
    assert to_str(self.d2.astype('a', np.float64),
                  preserve_str=False) == \
                     '-------a--b-c\n1.00e+00  2 3\n4.00e+00  9 8\n' \
                     '4.00e+00 10 0'
    assert to_str(self.d2.astype('a', str),
                  preserve_str=False) == \
        "a----b-c\n'1'  2 3\n'4'  9 8\n'4' 10 0"


def test_8(self):
    class MM:

        def __repr__(self):
            return 'MM()'
    assert str(self.d6.added_col('n', t2c(np.ndarray, str, shape1=(1,)), np.array([['a', string_null]], to_dtype(str)).T).npvaluesT(['n'])) == "[['a']\n ['<null>']]"
    assert str(self.d6.added_col('n', t2c(np.ndarray, str, shape1=(1,)), np.array([['a', string_null]], to_dtype(str)).T)['n'].shape) == '(2, 1)'
    assert str(self.d6) == 'x-y---\nf null\ng g   '
    assert str(self.d6['y'].shape) == '(2,)'
    assert str(self.d6.npvaluesT(['y'])) == "[['<null>']\n ['g']]"
    assert str(self.d6.is_any_null(['x', 'y'])) == '[ True False]'

    assert str(self.d6.grouped('x', use_subset_ttype=False)) == \
        'f:  y---\n    null\ng:  y\n    g'
    assert str(self.d6.grouped('y', use_subset_ttype=False,
                               check_is_nonnull=False)) == \
        'g:  x\n    g\nnull: \n    x\n    f'
    assert str(self.d6.added_col('h', t2c(MM), np.array([MM(), object_null])).grouped('h', use_subset_ttype=False, check_is_nonnull=False)) == 'MM(): \n    x-y---\n    f null\nnull: \n    x-y\n    g g'


def test_9(self):
    assert str(self.d6.grouped(['y'], check_is_nonnull=False)) == 'g:  x\n    g\nnull: \n    x\n    f'
    assert str(self.d6.grouped(['y'], check_is_nonnull=False, sort_keys=False)) == 'null: \n    x\n    f\ng:  x\n    g'


def test_to_latex(self):
    assert self.d6.to_xhtml() == '''\
<table>
<thead>
<tr>
    <th>x</th>
    <th>y</th>
</tr>
</thead>
<tbody>
<tr>
    <td>f</td>
    <td><span class="null">?</span></td>
</tr>
<tr>
    <td>g</td>
    <td>g</td>
</tr>
</tbody>
</table>'''
    assert self.d6.to_latex() == '\\begin{tabular}{ |l|l| }\n  \\hline\n  x & y \\\\\n  \\hline\n  f & null \\\\\n  g & g \\\\\n  \\hline\n\\end{tabular}'
    assert db.to_latex(OrderedDict([('d6', self.d6), ('d2', self.d2)])) == '\\documentclass[a4paper]{article}\n\n\\usepackage[utf8]{inputenc}\n\\usepackage{amsmath}\n\n\\begin{document}\n\n\\section*{d6}\n\\begin{tabular}{ |l|l| }\n  \\hline\n  x & y \\\\\n  \\hline\n  f & null \\\\\n  g & g \\\\\n  \\hline\n\\end{tabular}\n\n\\section*{d2}\n\\begin{tabular}{ |l|l|l| }\n  \\hline\n  a & b & c \\\\\n  \\hline\n  1 & 2 & 3 \\\\\n  4 & 9 & 8 \\\\\n  4 & 10 & 0 \\\\\n  \\hline\n\\end{tabular}\n\n\\end{document}\n'


def test_save_pdf(self, tmpdir):
    j = partial(os.path.join, str(tmpdir))
    db.save_pdf(j('asd.pdf'), OrderedDict([('d6', self.d6), ('d2', self.d2)]))
    remove_files([j('asd.pdf')])
    db.save_pdf(j('asd.pdf'), OrderedDict([('d6', ListDict.from_dicts([{'a': r'\12&34{}äöüß§$/%!"'}])), ('d2', self.d2)]))  # ´`
    remove_files([j('asd.pdf')])


def test_11(self):
    #print(d2.grouped(['a'], use_subset_ttype=True))
    #print(d3.grouped(['b'], use_subset_ttype=True))
    #print(d3.grouped(['a','b'], use_subset_ttype=True))
    L = self.d3

    assert str(L.grouped(['a', 'b'], use_subset_ttype=True)) == \
        '''\
1:  2:  c
        3
    9:  c
        (empty list)
    10: c
        (empty list)
4:  2:  c
        (empty list)
    9:  c
        8
    10: c
        0'''
    assert to_str(L.grouped(['a', 'b'], use_subset_ttype=True),
                  nrows=True) == '''\
1:  2:  1
    9:  0
    10: 0
4:  2:  0
    9:  1
    10: 1'''
    assert str(L.contingency_table(['a', 'b'],
                                   use_subset_ttype=True)) == '''\
a--b-count
1  2     1
1  9     0
1 10     0
4  2     0
4  9     1
4 10     1'''
    assert ungroup(L.grouped(['a', 'b'], use_subset_ttype=True),
                   ['a', 'b'], L.dspecs) == L
    assert str(flatten_groups(L.grouped(['a', 'b'],
                                        use_subset_ttype=True),
                              ['a'], [t2c(np.int64)])) == '''\
a-leaves----------
1 2:  c           
      3           
  9:  c           
      (empty list)
  10: c           
      (empty list)
4 2:  c           
      (empty list)
  9:  c           
      8           
  10: c           
      0           '''
    assert str(L.group_and_flatten(['a', 'b'], use_subset_ttype=True)) == '''\
a--b-leaves------
1  2 c           
     3           
1  9 c           
     (empty list)
1 10 c           
     (empty list)
4  2 c           
     (empty list)
4  9 c           
     8           
4 10 c           
     0           '''
    assert str(L.auto_subset_ttyped(['c']).group_and_flatten(['a', 'c'], use_subset_ttype=True)) == '''\
a-c-leaves------
1 0 -b          
    (empty list)
1 3 -b          
     2          
1 8 -b          
    (empty list)
4 0 -b          
    10          
4 3 -b          
    (empty list)
4 8 -b          
     9          '''
    assert str(L.auto_subset_ttyped(['c']).group_and_flatten(['a', 'c'], use_subset_ttype=False)) == '''\
a-c-leaves
1 3 -b    
     2    
4 0 -b    
    10    
4 8 -b    
     9    '''
    assert str(L.group_and_flatten(['c'], use_subset_ttype=False)) == '''\
c-leaves
0 a--b  
  4 10  
3 a--b  
  1  2  
8 a--b  
  4  9  '''
    assert str(unflatten_ungroup(L.group_and_flatten(['c'], use_subset_ttype=False), L.dspecs)) == '''\
a--b-c
4 10 0
1  2 3
4  9 8'''
    assert str(L.grouped2([['a'], ['b']],
                          use_subset_ttype=True)) == '''\
(1): 
    (2): 
        c
        3
    (9): 
        c
        (empty list)
    (10): 
        c
        (empty list)
(4): 
    (2): 
        c
        (empty list)
    (9): 
        c
        8
    (10): 
        c
        0'''
    assert str(L.grouped2([['a', 'b']], use_subset_ttype=True)) == \
        '''\
(1, 2): 
    c
    3
(1, 9): 
    c
    (empty list)
(1, 10): 
    c
    (empty list)
(4, 2): 
    c
    (empty list)
(4, 9): 
    c
    8
(4, 10): 
    c
    0'''
    assert str(L.grouped2([['a', 'b']], use_subset_ttype=False)) == \
        '''\
(1, 2): 
    c
    3
(4, 9): 
    c
    8
(4, 10): 
    c
    0'''


def test_12(self):
    return
    if False:
        l2.add_ldct(l5)
        print(l2)
    if False:
        #print(dict_intersect(d2.dct, ['a','b']))
        # print(d2.cols(['a','b']).dct)
        # print(d2.cols(['b','a']).dct)
        print(d2.cols(['a', 'b']))
        print(d2.cols(['b', 'a']))
        print(d2.cols(['a', 'b']).dspecs)
        print(d2.cols(['b', 'a']).dspecs)
    if True:
        # print(type(d2.dct))
        print(d2.added_col('g', t2c(np.int64), 3, 'rep_R'))
        print(d2.added_col('g', t2c(np.int64), 3, 'rep_RC'))
        print(d2.added_col('g', t2c(np.ndarray, np.int64, shape1=(3,)), 3,
                           'rep_RC'))
    if True:
        #L = d3
        sett = list(set([
            d3.group_and_flatten(['c'], use_subset_ttype=False),
            d3.group_and_flatten(['c'], use_subset_ttype=False)]))
        assert len(sett) == 1
        # print(sett)
        # print(sett[0]==sett[1])
        # print(hash(sett[0])==hash(sett[1]))
        #print(hash(d3.group_and_flatten(['c'], skip_empty=True, auto_subset_ttype=True)))
        #print(hash(L.group_and_flatten(['c'], skip_empty=True, auto_subset_ttype=True)))
        #print(hash(L.group_and_flatten(['c'], skip_empty=True, auto_subset_ttype=True)))
        #print(hash(L.group_and_flatten(['c'], skip_empty=True, auto_subset_ttype=True).dspecs))
        #print(hash(L.group_and_flatten(['c'], skip_empty=True, auto_subset_ttype=True).dspecs))
        # d = [
        #L.group_and_flatten(['c'], skip_empty=True, auto_subset_ttype=True).dspecs,
        # L.group_and_flatten(['c'], skip_empty=True, auto_subset_ttype=True).dspecs]
        #print([hash(v) for v in d])
        # ipy().embed()


def test_13(self):
    assert to_repr(
        self.d3.reduced(
            ['a'], ['sum(b)', 'sum(c)'], [t2c(np.int64), t2c(np.int64)],
            lambda L: [np.sum(L['b']), np.sum(L['c'])]), type_repr=True) == \
        '''ListDict(keys=['a', 'sum(b)', 'sum(c)'], vals=[array([1, 4]), array([ 2, 19]), array([3, 8])], cspecs=[ColumnDescriptor(SubsetOfType.C(int64(1), int64(4)), int64, int64, None, ()), ColumnDescriptor(int64, int64, int64, None, ()), ColumnDescriptor(int64, int64, int64, None, ())], nrows=2)'''
    LR = self.d3.view()
    LR.reduce(['a'], ['sum(b)', 'sum(c)'], [t2c(np.int64), t2c(np.int64)],
              lambda L: [np.sum(L['b']), np.sum(L['c'])])
    assert LR == self.d3.reduced(['a'], ['sum(b)', 'sum(c)'],
                                 [t2c(np.int64), t2c(np.int64)],
                                 lambda L: [np.sum(L['b']),
                                            np.sum(L['c'])])
    assert str(self.d3.grouped_and_add_cols(
        ['a'], ['ct1', 'ct2'], [t2c(np.int64), t2c(np.int64)],
        lambda L: [L['b'] + 2, np.arange(len(L['c'])) + 5])) == '''\
a--b-c-ct1-ct2
1  2 3   4   5
4  9 8  11   5
4 10 0  12   6'''
    assert str(self.d3.grouped_and_add_cols(
        [], ['ct1', 'ct2'], [t2c(np.int64), t2c(np.int64)],
        lambda L: [L['b'] + 2, np.arange(len(L['c'])) + 5])) == '''\
a--b-c-ct1-ct2
1  2 3   4   5
4  9 8  11   6
4 10 0  12   7'''


def test_14(self):
    assert self.d2.info == '''\
Keys: 'a' S(1, 4)
      'b' int64
      'c' int64
Rows: 3'''


def test_15(self):
    assert str(self.d6.sorted(['y'], null_small=True)) == '''\
x-y---
f null
g g   '''
    assert str(self.d6.sorted(['y'], null_small=False)) == '''\
x-y---
g g   
f null'''


@pytest.mark.parametrize('nparents', [0, 1, None])
def test_16(self, tmpdir, nparents):
    (self.d4
     .added_col('__collapsed_depth_reduction__', t2c(np.int64), 0, 'rep_R')
     .to_navigate(['c', 'b', 'a', 'd'],
                  save_group=SaveGroupAsTree(str(tmpdir), nparents=nparents)))
    remove_files(files(str(tmpdir), return_abs_path=True))


@pytest.mark.skipif(find_spec('pandas') is None, reason='pandas not installed')
def test_17(self):
    assert str(self.d4.to_dataframe()) == '''\
   a   b  c  d
0  1   2  3  3
1  4   9  8  4
2  4  10  0  5'''
    assert str(ListDict.from_dataframe(self.d4.to_dataframe())) == \
        str(self.d4)
    assert ListDict.from_dataframe(self.d4.to_dataframe()) == self.d4
    assert ListDict.from_dataframe(self.d4.to_dataframe(), self.d4.dspecs) == \
        self.d4


def test_18(self):
    assert str(self.d2.fsliced_key('a', lambda v: v != 1)) == '''\
a--b-c
4  9 8
4 10 0'''
    assert to_str(self.d2.fsliced_key('a', lambda v: v != 1)
                  .dspecs) == \
        "{'a': S(4), 'b': int64, 'c': int64}"
    assert str(self.d2.isliced([0, -1])) == '''\
a--b-c
1  2 3
4 10 0'''
    assert str(self.d2.isliced[0::2]) == '''\
a--b-c
1  2 3
4 10 0'''
    assert str(ListDict.isliced(self.d2, [0, -1])) == '''\
a--b-c
1  2 3
4 10 0'''


def test_19(self):
    dj0 = ListDict.from_dicts(
        [{'a': 1, 'b': 2, 'c': 3},
         {'a': 4, 'b': 9, 'c': 8},
         {'a': 4, 'b': 10, 'c': 0}],
        dspecs={'a': t2c(np.int64), 'b': t2c(np.int64), 'c': t2c(np.int64)})
    dj1 = ListDict.from_dicts(
        [{'a': 4, 'b': 9, 'x': 8},
         {'a': 1, 'b': 2, 'x': 3}],
        dspecs={'a': t2c(np.int64), 'b': t2c(np.int64), 'x': t2c(np.int64)})
    dj0.sort_cols()
    dj1.sort_cols()
    assert str(dj0.sorted_cols(order='decreasing')
               .renamed_keys({'a': 'a_new'})) == '''\
c--b-a_new
3  2     1
8  9     4
0 10     4'''
    assert str(dj0._joined2(lst1=dj1, keys=['a', 'b'],
                            join_type='outer',
                            left_num_copies='*',
                            right_num_copies='*',
                            order='left')) == '''\
a--b-c----x
1  2 3    3
4  9 8    8
4 10 0 null'''
    assert str(dj0._joined2(lst1=dj1, keys=['a', 'b'],
                            join_type='outer',
                            left_num_copies='*',
                            right_num_copies='*',
                            order='right')) == '''\
a--b-c----x
4  9 8    8
1  2 3    3
4 10 0 null'''
    assert str(join([dj0, dj1],
                    joined_keys=['a', 'b'],
                    prefixes=['first ', 'second '])) == '''\
a-b-first c-second x
1 2       3        3
4 9       8        8'''
    assert str(join([dj0, dj1],
                    joined_keys=['a', 'b'],
                    prefixes=['first ', 'second '],
                    join_type='left')) == '''\
a--b-first c-second x
1  2       3        3
4  9       8        8
4 10       0     null'''
    assert str(join([dj0, dj1],
                    joined_keys=['a', 'b'],
                    prefixes=['first ', 'second '],
                    join_type='right')) == '''\
a-b-first c-second x
1 2       3        3
4 9       8        8'''


def test_20(self):
    dj0 = ListDict.from_dicts(
        [{'a': '1', 'c': 3},
         {'a': '4', 'c': 8},
         {'a': '4-', 'c': 0}],
        dspecs={'a': t2c(str), 'c': t2c(np.int64)})
    dj1 = ListDict.from_dicts(
        [{'ar': '4', 'x': 8},
         {'ar': '1', 'x': 3}],
        dspecs={'ar': t2c(str), 'x': t2c(np.int64)})
    dj0.sort_cols()
    dj1.sort_cols()
    assert str(dj0.joined_re(dj1, 'a', 'ar',
                         left_num_copies='1',
                         right_num_copies='1')) == \
        '''\
a--c-ar-x
1  3 1  3
4  8 4  8
4- 0 4  8'''


def test_21(self):
    assert str(self.d2.binned(self.d2)) == '''\
a--b-c-bin a-bin b-bin c
1  2 3     1     2     3
4  9 8     4     9     8
4 10 0     4    10     0'''
    assert str(self.d2.binned(self.d2.isliced[:2])) == '''\
a--b-c-bin a-bin b-bin c
1  2 3     1     2     3
4  9 8     4     9     8
4 10 0     4     9     8'''
    assert str(self.d2n.binned(self.d2)) == '''\
---a--b----c-bin a-bin b-bin c
   1  2    3     1     2     3
   4  9    8     4     9     8
null 10 null     4    10     0'''
    assert str(self.d2.binned(self.d2n)) == '''\
a--b-c-bin a-bin b-bin c
1  2 3     1     2     3
4  9 8     4     9     8
4 10 0  null    10  null'''
    assert str(self.d2.binned(self.d2n, my_prefix='my ',
                              other_prefix='other ')) == '''\
my a-my b-my c-other a-other b-other c
   1    2    3       1       2       3
   4    9    8       4       9       8
   4   10    0    null      10    null'''


def test_22(self):
    assert str(self.d2.interpolated(self.d2, list(self.d2.keys()),
                                 ['ia'], [t2c(np.int64)],
                                 lambda L: [5])) == \
        '''\
a--b-c-ia
1  2 3  5
4  9 8  5
4 10 0  5'''
    assert str(self.d2.interpolated(self.d2.isliced[:2], list(self.d2.keys()),
                                 ['ia'], [t2c(np.int64)],
                                 lambda L: [5])) == \
        '''\
a--b-c---ia
1  2 3    5
4  9 8    5
4 10 0 null'''


def test_23(self):
    D = self.d2.added_col('p a', t2c(np.int64), self.d2['a'])
    assert D.renamed_prefixes({'p ': ''}, ['a']) == self.d2
    assert D.renamed_prefixes({'p ': 'x'}, ['a']) == self.d2
    assert str(D.renamed_prefixes({'p ': 'x '})) == '''\
a--b-c-x a
1  2 3   1
4  9 8   4
4 10 0   4'''


def test_24(self):
    assert str(self.d2) == '''\
a--b-c
1  2 3
4  9 8
4 10 0'''
    assert str(self.d2.firsted(['a'])) == '''\
a-b-c
1 2 3
4 9 8'''
    assert str(self.d2.lasted(['a'])) == '''\
a--b-c
1  2 3
4 10 0'''


def test_25(self):
    d0 = ListDict.from_dicts(
        [{'a': '1', 'b': '2', 'c': '2009-08-07'},
         {'a': '4', 'b': '9', 'c': '2010-09-08'},
         {'a': '4', 'b': '10', 'c': '2011-10-09'}]).sorted_cols()
    d1 = ListDict.from_dicts(
        [{'a': '4', 'b': '9', 'c': '2010-09-09'},
         {'a': '4', 'b': '9', 'c': '2011-10-09'}]).sorted_cols()
    assert str(d0.taken_one(d1, ['a'], ['x ', 'other '],
                           closest_key('c', date_distance(date_format)))) == \
        '''\
a-x b-x c--------other b-other c---
1 2   2009-08-07 null    null      
4 9   2010-09-08 9       2010-09-09
4 10  2011-10-09 9       2011-10-09'''
    assert str(
        d0.interpolated(d1, ['a', 'b'], ['other c'], [t2c(str)],
                        closest_key('c', date_distance(date_format)))) == \
        '''\
a-b--c----------other c---
1 2  2009-08-07 null      
4 9  2010-09-08 2010-09-09
4 10 2011-10-09 null      '''


def test_cache(self, tmpdir):
    j = partial(os.path.join, str(tmpdir))

    class CachedReprIo(BaseCachedReprIo):

        def __init__(self):
            BaseCachedReprIo.__init__(self, j('cached_'), 'A')

        @BaseCachedReprIo.cached_property
        def a(self):
            return 5

    cr = CachedReprIo()
    assert cr.a == 5
    assert cr.a == 5
    remove_files(map(j, 'cached_a.py cached_a.py_doneA cached_a.blob3'.split()))

    remove_files(map(j, 'cached_a.csv cached_a.csv_doneA'.split()), force=True)

    class CachedReprIo(BaseCachedListDicts):

        def __init__(self):
            BaseCachedListDicts.__init__(self, j('cached_'), 'A')

        @BaseCachedListDicts.cached_property
        def a(self1):
            return self.d2.sorted_cols()

    cr = CachedReprIo()
    assert str(cr.a) == '''\
a--b-c
1  2 3
4  9 8
4 10 0'''
    assert str(cr.a) == '''\
a-b--c
1 2  3
4 9  8
4 10 0'''
    remove_files(map(j, 'cached_a.csv cached_a.csv_doneA'.split()))


def test_27(self):
    return
    if False:
        #print(d3.applied('a', lambda v: string_null if v==4 else v))
        print(added_columns_apply(d3, ['ix', 'fy', 'bz'], [t2c(int), t2c(float), t2c(bool)], lambda l: [string_null if l['a'] == 4 else l['a'], string_null if l['a'] == 4 else 1.0 * l['a'], string_null if l['b'] == 9 else False]))
    if True:
        print(d3)
        print(d3.sorted(['c']))
        print(d3.sorted(['c'], reverse=True))
        print(d3.added_col('g', t2c(str), np.array(['b', 'a', string_null], dtype=to_dtype(str))).sorted(['g'], null_small=True))
        print(d3.added_col('g', t2c(str), np.array(['b', 'a', string_null], dtype=to_dtype(str))).sorted(['g'], null_small=False))
    if False:
        # print(d6['x'].dtype)
        print(d6.appliedmap('x', lambda v: ',' + v + '__*'))
    if False:
        print(d2)
        print(d2.removed_duplicates(['a']))
        print(d2.removed_duplicates(['a'], preserve_order=False))
    if False:
        print(d2.added_cols_applymap(['x', 'y'], [t2c(np.int64), t2c(np.int64)], lambda l: [2 * l['a'], l['a'] // 2]))
    if True:
        print(concat([d3, d5]))
        print(concat([d3, d6]))
        print(d3b)
        print(ListDict(keys=['a', 'b'], cspecs=[t2c(np.int64), t2c(np.ndarray, np.int32, shape1=(2,))], vals=[1, np.int32(2)], kinds='rep_RC', nrows=4))
    if True:
        print(d6.added_col_applymap('j', t2c(str), lambda l: l['x'] + '__'))
        print(d6.added_cols_applymap(['j', 'h'], [t2c(str), t2c(str)], lambda l: [l['x'] + '_j_', l['x'] + '_h']))
    if True:
        jd1 = [
            {'a': 'y', 'b': '1'},
            {'a': '0', 'b': '0'},
            {'a': 'y', 'b': '2'},
            {'a': 'y', 'b': '6'},
            {'a': 'z', 'b': '6'}]
        jd2 = [
            {'a': 'y', 'c': '1'},
            # {'a':'0','c':'0'},
            # {'a':'y','c':'2'},
            # {'a':'y','c':'6'},
            {'a': 'z', 'c': '6'}]
        jD1 = ListDict.from_dicts(jd1, dspecs={'a': v2c(['0', 'y', 'z']), 'b': t2c(str)})
        jD2 = ListDict.from_dicts(jd2, dspecs={'a': v2c(['0', 'y', 'z']), 'c': t2c(str)})

        #print(join2(d6.added_col_apply('j', t2c(str), lambda l:l['x']+'__'), d6.added_col_apply('k', t2c(str), lambda l:l['x']+'__'), keys=['x','y'], join_type='inner', left_num_copies='1', right_num_copies='1', order='sort'))

        print('l\n', jD1)
        print('r\n', jD2)
        print()
        print('sort\n', jD1._joined2(jD2, keys=['a'], join_type='outer', left_num_copies='0-1', right_num_copies='+', order='sort'), '\n')
        print('left\n', jD1._joined2(jD2, keys=['a'], join_type='outer', left_num_copies='0-1', right_num_copies='+', order='left'), '\n')
        print('right\n', jD1._joined2(jD2, keys=['a'], join_type='outer', left_num_copies='0-1', right_num_copies='+', order='right'), '\n')
    if True:
        print(ListDict.from_dicts([{'a': 3, 'b': 6}, {'a': 4, 'b': 9}], {'a': t2c(np.int64), 'b': t2c(np.int64)}))
        #print(ListDict.from_dicts([{'a':3,'b':6},{'a':4,'b':9,'c':7}], {'a':t2c(np.int64), 'b':t2c(np.int64)}))
        print(ListDict.from_dicts([{'a': '3', 'b': '6'}, {'a': '4', 'b': 'f'}]))
    if False:
        print(ListDict.load_csv('asd.csv'))
        print(ListDict.load_csv('asd.csv', ascii_nul_replacement='f'))
    if False:
        print(d2)
        save_csv('asd.csv', d2)
        print(ListDict.load_csv('asd.csv', dspecs=d2.dspecs))
        print(ListDict.load_csv('asd.csv'))
    if False:
        list_assert_no_duplicates(list(d2.dicts()))
        list_assert_no_duplicates(list(concat([d2, d2]).dicts()))
    if True:
        print(d2.renamed_keys({'a': 'g'}))
        print(d2.added_col('l', d2.dspecs['a'], d2['a']).deleted_identical_columns({'a': ['l']}))
        # print(d2.added_col('l',d2.dspecs['a'],d2['a']).deleted_identical_columns({'b':['l']}))
    if True:
        a = ListDict.from_dicts([{'a': 1, 'b': 2, 'c': 3},
                {'a': 4, 'b': 9, 'c': 8},
            {'a': 4, 'b': 10, 'c': 0}], dspecs={'a': v2c([1, 4], np.int64, convert=True), 'b': t2c(np.int64), 'c': t2c(np.int64)})
        b = ListDict.from_dicts([{'a': 1, 'b': 2},
                {'a': 4, 'b': 9}], dspecs={'a': v2c([1, 4], np.int64, convert=True), 'b': t2c(np.int64)})
        b1 = ListDict.from_dicts([{'a': 1, 'b': 2}], dspecs={'a': v2c([1, 4], np.int64, convert=True), 'b': t2c(np.int64)})
        print(a)
        print(minus(a, b))
        print(minus(a, b1))
        print(intersection(a, b))
        print(intersection(a, b1))
    if False:
        class MM:
            pass
        print(np.array([1, 2, 3]) == object_null)
        print(np.array([1, 2, 3]) == object_null)
        print(np.array([1, 2, np.array([1, 3])], dtype=object) == 4)
        print(np.array([1, 2, 3], dtype=object) == 4)
        print(object_null == np.array([1, 2, 3]))
        print(d2.is_any_null())
        print(d2.added_col('h', t2c(MM), np.array([MM(), MM(), object_null])).is_any_null())
    if True:
        b = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -215.31674889, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -260.04599697, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 212.91774464, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 88.46921145, 0.0, 0.0, 129.53557348, 0.0, 58.53328043, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -64.73967055]
        print('python')
        print(' '.join(str(i) for i, v in sorted(enumerate(b), key=lambda iv: iv[1])))
        print('default')
        print(' '.join(str(i) for i in np.argsort(b)))
        print('quicksort')
        print(' '.join(str(i) for i in np.argsort(b, kind='quicksort')))
        print('mergesort')
        print(' '.join(str(i) for i in np.argsort(b, kind='mergesort')))
        print('heapsort')
        print(' '.join(str(i) for i in np.argsort(b, kind='heapsort')))
    if True:
        print(d2.added_col('j', t2c(np.ndarray, np.int64), np.array([[1, 2, 3]]).T, auto_shape1=True))
        #print(list(d2.added_col('j', t2c(np.ndarray,np.int64), np.array([[1,2,3]]).T, auto_shape1=True).dicts(expanded_keys='j')))

    if True:
        d0 = [{'a': 1, 'b': 2},
        {'a': 4, 'b': 9},
            {'a': 4, 'b': 10}]
        d1 = ListDict.from_dicts([], dspecs={'a': t2c(np.float64), 'b': t2c(np.int64)})
        d2 = ListDict.from_dicts(d0, dspecs={'a': v2c([1, 4], np.int64, convert=True), 'b': t2c(np.int64)})
        d3 = ListDict.from_dicts(d0, dspecs={'a': v2c([1, 4], np.int64, convert=True), 'b': v2c([2, 9, 10], np.int64, convert=True)})
        d4 = [{'x': 10, 'y': 12},
        {'x': 14, 'y': 19}]
        d5 = ListDict.from_dicts(d4, dspecs={'x': v2c([10, 14], np.int64, convert=True), 'y': t2c(np.int64)})

        print(concat([d2, d2], dspecs=d2.dspecs))
        print(concat([d2, d2, d5]))
        print(hconcat([d2.isliced(np.array([0, 2])), d5]))
        print(hconcat([d2.isliced(np.array([0, 2])), d5, d5], allow_duplicates=True))
        print(product([d2, d5]))
    if True:
        a = Dict2Object(d3=d3, d5=d5)
        b = Dict2Object(d3=d3, d5=d5, gg=d5)
        print(a)
        print(db.concat([a, a]))
        # print(db.concat([a,a,b]))
        #print(d3, d5)
        #print(concat([d3, d5]))
        #print(concat([d3, d6]))
        # print(d3b)
