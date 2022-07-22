# -*- coding: utf-8 -*-
from typing import Hashable, Optional, Sequence

from lacro.assertions import set_assert_contains
from lacro.collections import Dict2Object, resolved_tree_path
from lacro.decorators import cached_property
from lacro.iterators import first_element, unique_value

from .._collections import onetwo as ld


def a_hi(href, text, selected, enable_anchor):
    set_assert_contains(['direct', 'indirect', 'deselected'], selected)
    if selected == 'direct':
        return f'<span class="{selected}">{text}</span>'
    else:
        fragment = '#selected' if enable_anchor else ''
        return f'<a class="{selected}" href="{href}{fragment}">{text}</a>'


def expanding_li(ls, selected):
    return ([f'<li>  {s}</li>' for s in ls] if selected else
            ['<li>'] + [f'  {s}' for s in ls] + ['</li>'])


def txt_hi(selected, enable_anchor):
    set_assert_contains(['direct', 'indirect', 'deselected'], selected)
    return (('<span id="selected"></span>'
             if enable_anchor and selected == 'direct' else ''))


class Value:

    def __init__(self, n, l, lpath, ref, style):
        self._n = n
        self._l = l
        self._lpath = lpath
        self._ref = ref
        self._style = style

    @property
    def filename(self):
        return self._n.site_url(self._l, self._lpath + [self._n]).filename


class Toc:

    def __init__(self, n, L, lpath, ref, style):
        self._n = n
        self._L = L
        self._lpath = lpath
        self._ref = ref
        self._style = style

    @cached_property
    def first_item(self):
        return self._first_item(self._ref.npath, match=True)

    def _first_item(self, mpath, match):
        if len(self._lpath) == self._ref.npath:
            # self._L should ideally have length 1 (group keys fully describe
            # items)
            assert type(self._L) == ld.ListDict
            assert self._L.nrows >= 1
            return mpath, self._lpath, self._L.row_dict(0)
        else:
            assert type(self._L) == ld.ListGroup

            if (match and
                len(self._lpath) - 1 < self._ref.ipath and
                    self._ref.path[len(self._lpath)] in self._L.keys()):
                n1 = self._ref.path[len(self._lpath)]
            else:
                n1 = first_element(self._L.keys())
                if match:
                    mpath = len(self._lpath)
                    match = False
            return Toc(n1, self._L[n1], self._lpath + [n1], self._ref,
                       self._style)._first_item(mpath, match)

    @property
    def filename(self):
        mpath, lpath, l = self.first_item
        return self._n.site_url(
            l,
            lpath[:1 + min(max(len(self._lpath) - 1, self._ref.ipath),
                           l['__collapsed_depth__'] - 1, mpath - 1)]).filename

    def to_xhtml_ul(self, max_depth):
        lis = '\n'.join(
            Headings(n,
                     L,
                     self._lpath,
                     self._ref,
                     self._style).to_xhtml_elements(max_depth)
            for n, L in self._L.items())
        if lis.strip('\n') == '':
            return ''
        else:
            return f'<ul>{lis}\n</ul>\n'


class Headings:

    def __init__(self, n, L, lpath, ref, style):
        self._n = n
        self._L = L
        self._lpath = lpath
        self._ref = ref
        self._style = style
        if (len(self._lpath) < len(self._ref.path) and
                self._n == self._ref.path[len(self._lpath)]):
            if len(self._lpath) == self._ref.ipath:
                self._selected = 'direct'
            else:
                self._selected = 'indirect'
        else:
            self._selected = 'deselected'
        self._k = dict(
            text=self._title,
            enable_anchor=self._style.enable_anchor)
        self._indentation_width = 4

    def to_xhtml_elements(self, max_depth):
        if not self._n.user_contents('visible', True):
            return ''
        elif len(self._lpath) == self._ref.npath - 1:
            return self._leaf_to_xhtml_elements(max_depth)
        else:
            return self._node_to_xhtml_elements(max_depth)

    @property
    def _title(self):
        return self._n.user_contents('title')

    @cached_property
    def _toc(self):
        return Toc(self._n,
                   self._L,
                   self._lpath + [self._n],
                   self._ref,
                   self._style)

    @property
    def _toc_ref_self(self):
        """
        toc if ref=self
        """
        return Toc(self._n,
                   self._L,
                   self._lpath + [self._n],
                   self._ref.updated(dict(ipath=len(self._lpath))),
                   self._style)

    def _draw_nodes(self, collapsed_depth, max_depth):
        if max_depth is not None:
            collapsed_depth = min(collapsed_depth, max_depth)
        return len(self._lpath) < collapsed_depth

    def _node_to_xhtml_elements(self, max_depth):
        # navigate up in hierarchy
        if not self._draw_nodes(self._toc.first_item[2]['__collapsed_depth__'],
                                max_depth):
            return ''
        return '\n    %s<li>%s%s</li>' % (
            ' ' * self._indentation_width * len(self._lpath),
            txt_hi(selected=self._selected,
                   enable_anchor=self._style.enable_anchor) +
            a_hi(href=(self._toc_ref_self.filename
                       if (self._selected == 'indirect') else
                       self._toc.filename),
                 selected=self._selected,
                 **self._k),
            (self._toc.to_xhtml_ul(max_depth)
             if (self._selected != 'deselected')
             else ''))

    def _leaf_to_xhtml_elements(self, max_depth):
        # L should ideally have length 1 (group keys fully describe items)
        assert type(self._L) == ld.ListDict
        assert self._L.nrows >= 1
        if not self._draw_nodes(unique_value(self._L['__collapsed_depth__']),
                                max_depth):
            return ''
        return ''.join(
            '\n    %s' % s for s in expanding_li(
                # using txt_hi because iid="selected" must be on the left,
                # otherwise scrolls to the right when indented
                [' ' * self._indentation_width * (len(self._lpath) - 1) +
                 txt_hi(selected=self._selected,
                        enable_anchor=self._style.enable_anchor) +
                 a_hi(href=Value(self._n, l, self._lpath, self._ref,
                                 self._style).filename,
                      selected=self._selected,
                      **self._k)
                 for l in self._L.dicts()], selected=True))


def generate_toc(group: 'ld.ListGroup',
                 keys: Sequence[Hashable],
                 path: Sequence[Hashable],
                 nparents: Optional[int],
                 enable_anchor: bool) -> Toc:
    assert len(path) <= len(keys)
    set_assert_contains([True, False], enable_anchor)
    npath = len(keys)

    nskipped = (0 if nparents is None else
                max(0, len(path) - 1 - nparents))
    lpath = path[:nskipped]
    return Toc(path[-1],
               resolved_tree_path(group, lpath),
               lpath,
               ref=Dict2Object(ipath=len(path) - 1,
                               path=path,
                               npath=npath),
               style=Dict2Object(enable_anchor=enable_anchor))
