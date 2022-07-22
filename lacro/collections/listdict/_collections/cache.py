# -*- coding: utf-8 -*-
from functools import update_wrapper

from .onetwo import ListDict


class BaseCachedFunctions:

    def __init__(self, format, version, cacher):
        self._format = format
        self._version = version
        self._cacher = cacher

    class cached_property:

        def __init__(self, func) -> None:
            self._func = func
            update_wrapper(self, func)  # type: ignore

        def __get__(self, obj, objtype):
            if obj is None:
                return self

            return obj._cacher(self._func.__get__(obj, objtype),
                               obj._format.format(name=self._func.__name__),
                               obj._version)


class BaseCachedListDicts(BaseCachedFunctions):

    def __init__(self, prefix, version):
        BaseCachedFunctions.__init__(self, '%s{name}.csv' % prefix, version,
                                     ListDict.cached_dictlist)


class BaseCachedReprIo(BaseCachedFunctions):

    def __init__(self, prefix, version):
        from lacro.path.pathver import cached_repr_io
        BaseCachedFunctions.__init__(self, '%s{name}.py' % prefix, version,
                                     cached_repr_io)
