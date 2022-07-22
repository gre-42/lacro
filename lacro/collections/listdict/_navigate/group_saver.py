# -*- coding: utf-8 -*-
import os.path
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from lacro.collections import dict_union
from lacro.io.string import load_string_from_file, save_string_to_file
from lacro.path.pathabs import abspath_of_script_child
from lacro.string.misc import F1, sub_count

from .._collections import onetwo as ld
from .._navigate.naming_scheme import NamingScheme


class SaveGroup(ABC):

    @abstractmethod
    def __enter__(self) -> 'SaveGroupWorker':
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class SaveGroupWorker(SaveGroup):

    @abstractmethod
    def save_file(self,
                  group: 'ld.ListGroup',
                  keys: list,
                  path: List[NamingScheme],
                  lst: 'ld.ListDict') -> None:
        pass


def f1_sub(contents: str,
           dct: Dict[str, str],
           path: List[NamingScheme],
           subs: List[str]) -> str:
    contents = F1(contents, dct)
    for o, n in subs:
        contents = sub_count(o, n, contents)
    subs1 = path[-1].user_contents('sub')
    if isinstance(subs1, ld.ListDict):
        for o, n in subs1.mapping('old', 'new',
                                  injective=False).items():
            contents = sub_count(o, n, contents)
    else:
        assert subs1 == ''
    return contents


class SaveGroupAsTree(SaveGroupWorker):

    def __init__(self,
                 destination_dirname: str,
                 gen_dct=lambda path, lst, user_contents: dict(
                     title='title',
                     contents=lst.to_xhtml(escape=True)),
                 template_filename=abspath_of_script_child(
                     '..', 'templates', 'index.xhtml'),
                 enable_anchor=False,
                 subs: List[str] = [],
                 nparents=None) -> None:
        self._destination_dirname = destination_dirname
        self._template_filename = template_filename
        self._gen_dct = gen_dct
        self._enable_anchor = enable_anchor
        self._subs = subs
        self._nparents = nparents

    def __enter__(self) -> SaveGroupWorker:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def save_file(self,
                  group: 'ld.ListGroup',
                  keys: list,
                  path: List[NamingScheme],
                  lst: 'ld.ListDict') -> None:
        from .._navigate.toc import generate_toc

        toc = generate_toc(group,
                           keys,
                           path,
                           self._nparents,
                           self._enable_anchor)

        dct = dict_union(
            dict(toc=toc.to_xhtml_ul(max_depth=None)),
            self._gen_dct(path, lst, path[-1].user_contents))
        save_string_to_file(
            os.path.join(self._destination_dirname, toc.filename),
            f1_sub(load_string_from_file(self._template_filename),
                   dct,
                   path,
                   self._subs))


class SaveGroupAsList(SaveGroup):

    def __init__(self,
                 destination_dirname: str,
                 join_template_filename=abspath_of_script_child(
                     '..', 'templates', 'join.xhtml'),
                 index_template_filename=abspath_of_script_child(
                     '..', 'templates', 'index.xhtml'),
                 gen_join_dct=lambda path, lst, user_contents: dict(
                     title='title',
                     contents=lst.to_xhtml(escape=True)),
                 index_dct=dict(title='title'),
                 subs=[],
                 nparents: Optional[int] = None,
                 enable_anchor: bool = False,
                 single_file: bool = False) -> None:
        self._gen_dct = gen_join_dct
        self._index_dct = index_dct
        self._subs = subs
        self._destination_dirname = destination_dirname
        self._index_template_filename = index_template_filename
        self._join_template_filename = join_template_filename
        self._nparents = nparents
        self._enable_anchor = enable_anchor
        self._single_file = single_file
        self._worker = None

    def __enter__(self) -> SaveGroupWorker:
        assert self._worker is None, 'enter called multiple times'
        self._worker = SaveGroupAsListWorker(self, rec=0)
        return self._worker

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._single_file:
            contents = self._worker._contents()
            save_string_to_file(
                os.path.join(self._destination_dirname, 'index.xhtml'),
                F1(load_string_from_file(self._index_template_filename),
                   dict_union(self._index_dct,
                              dict(contents=contents))))


class SaveGroupAsListWorker(SaveGroupWorker):

    def __init__(self, sgl: SaveGroupAsList, rec: int) -> None:
        self._sgl = sgl
        self._children = []
        self._dct = None
        self._path = None
        self._toc = None
        self._rec = rec

    def __call__(self, group: 'ld.ListGroup') -> SaveGroup:
        return self

    def __enter__(self) -> SaveGroup:
        child = SaveGroupAsListWorker(self._sgl, rec=self._rec + 1)
        self._children.append(child)
        return child

    def __exit__(self, exc_type, exc_val, exc_tb):
        if (not self._sgl._single_file) and self._rec == 1:
            contents = self._contents(header_rec=1)
            save_string_to_file(
                os.path.join(self._sgl._destination_dirname,
                             self._toc.filename),
                F1(load_string_from_file(self._sgl._index_template_filename),
                   dict_union(self._sgl._index_dct,
                              dict(toc=self._toc.to_xhtml_ul(max_depth=1),
                                   contents=contents))))

    def _contents(self, header_rec) -> str:
        subsections = ''.join(c._contents(header_rec + 1)
                              for c in self._children)
        if self._dct is None:
            return subsections
        return f1_sub(
            load_string_from_file(self._sgl._join_template_filename),
            dict_union(self._dct, dict(subsections=subsections,
                                       section_title_tag=f'h{header_rec}')),
            self._path,
            self._sgl._subs)

    def save_file(self,
                  group: 'ld.ListGroup',
                  keys: list,
                  path: List[NamingScheme],
                  lst: 'ld.ListDict') -> None:
        from .._navigate.toc import generate_toc
        assert self._dct is None, 'save_file called multiple times'
        self._dct = self._sgl._gen_dct(path, lst, path[-1].user_contents)
        self._path = path
        self._toc = generate_toc(group,
                                 keys,
                                 path,
                                 self._sgl._nparents,
                                 self._sgl._enable_anchor)
