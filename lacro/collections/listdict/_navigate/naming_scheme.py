# -*- coding: utf-8 -*-
from functools import total_ordering
from typing import Callable, Union

from lacro.collections import FunctionChain
from lacro.decorators import cached_property
from lacro.io.string import cached_file_loader, load_file_if_uri
from lacro.string.misc import XhtmlText, capitalize, to_xhtml

from .site_url import SiteUrl


@total_ordering
class NamingScheme:

    def __init__(self, n):
        self._n = n

    def __lt__(self, other):
        assert type(self) == type(other)
        return self._value < other._value

    def __eq__(self, other):
        return type(self) == type(other) and self._value == other._value

    def __hash__(self):
        return hash(self._value)

    def __str__(self):
        return self._title

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self._n)

    def __to_xhtml__(self, **kwargs):
        if isinstance(self._n, XhtmlText):
            return self._title
        else:
            return to_xhtml(self._title, **kwargs)

    def site_url(self, l, lpath) -> SiteUrl:
        uri_path = self.user_contents('uri_path', None)
        return SiteUrl(uri_path, l, lpath)

    def user_contents(self, name, default: object = ''):
        if name == 'title':
            return to_xhtml(self._n, to_str=FunctionChain([str, capitalize]))
        else:
            return default

    @cached_property
    def _title(self):
        return self.user_contents('title')

    @cached_property
    def _counter(self):
        return self.user_contents('counter', 1)

    @cached_property
    def _value(self):
        return self._counter, self._title


def renaming_scheme(file_loader: Union[str, Callable[[str], str]]):
    file_loader = cached_file_loader(file_loader)

    class RenamingScheme(NamingScheme):

        @load_file_if_uri(file_loader)
        def user_contents(self, name, default=''):
            if type(self._n) == dict:
                return self._n.get(name, default)
            else:
                return NamingScheme.user_contents(self, name, default)

    return RenamingScheme
