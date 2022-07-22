# -*- coding: utf-8 -*-
import inspect
from typing import NoReturn


def func_defaults(func):
    return dict((v.name, v.default)
                for v in inspect.signature(func).parameters.values()
                if v.default != v.empty)


def pop_func_defaults(func, kwargs):
    return {k: kwargs.pop(k) for k in (set(func_defaults(func).keys()) &
                                       set(kwargs.keys()))}


def reraise(exception) -> NoReturn:
    import sys
    et, ev, tb = sys.exc_info()
    raise exception.with_traceback(tb) from None
    # raise exception, None, c
    # e.__traceback__ = c
    # raise exception


def parent_vars(nparent=2):
    from lacro.collections import dict_union
    stack = inspect.stack(0)
    psta = stack[nparent][0]
    return dict_union(psta.f_globals, psta.f_locals, allow_duplicates=True)


def getattrs(obj, attrs):
    return [getattr(obj, attr) for attr in attrs]
