# -*- coding: utf-8 -*-
import time
from functools import partial, update_wrapper, wraps
from inspect import BoundArguments, Parameter, signature
from typing import Callable


def _cached_property_name(name):
    return '_cached_' + name


def _is_cached_property_name(name):
    return name.startswith('_cached_')


def _cached_get_name(name):
    return '_cache_' + name


class CachedProperties:

    def __init__(self, obj):
        self._obj = obj

    @property
    def names(self):
        return [d for d in dir(self._obj) if _is_cached_property_name(d)]

    def cached(self, names):
        return [getattr(self._obj, name) for name in names
                if hasattr(self._obj, _cached_property_name(name))]

    def delete(self, names=None):
        if names is None:
            for d in self.names:
                delattr(self._obj, d)
        else:
            for name in names:
                if hasattr(self._obj, _cached_property_name(name)):
                    delattr(self._obj, _cached_property_name(name))


class cached_property:

    def __init__(self, func: Callable) -> None:
        self._func = func
        update_wrapper(self, func)

    def __get__(self, obj, objtype):
        if obj is None:
            return self
        if not hasattr(obj, _cached_property_name(self._func.__name__)):
            setattr(obj,
                    _cached_property_name(self._func.__name__),
                    self._func.__get__(obj, objtype)())
        return getattr(obj, _cached_property_name(self._func.__name__))


class CachedGet:

    def __init__(self, func):
        self._func = func
        update_wrapper(self, func)

    def __get__(self, obj, objtype):
        if obj is None:
            return self
        if not hasattr(obj, _cached_get_name(self._func.__name__)):
            setattr(obj,
                    _cached_get_name(self._func.__name__),
                    self.__class__(self._func.__get__(obj, objtype)))
        return getattr(obj, _cached_get_name(self._func.__name__))


class cached_function(CachedGet):

    def __init__(self, func):
        super().__init__(func)
        self._cache = None
        self._signature = signature(self._func)
        if any(v.kind == Parameter.VAR_KEYWORD
               for v in self._signature.parameters.values()):
            raise TypeError('Keyword-only arguments are not supported by '
                            'cached_function')

    def __call__(self, *args, **kwargs):
        bound_arguments: BoundArguments = self._signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        assert len(bound_arguments.kwargs) == 0
        if self._cache is None:
            from lacro.collections import Cache
            self._cache = Cache()
        return self._cache.get(bound_arguments.args,
                               lambda: self._func(*args, **kwargs))


def ignore_unknown_arguments(func):
    sig = signature(func)

    has_var_keyword = any(v.kind == Parameter.VAR_KEYWORD
                          for v in sig.parameters.values())
    has_var_positional = any(v.kind == Parameter.VAR_POSITIONAL
                             for v in sig.parameters.values())
    num_positionals = sum(1 for v in sig.parameters.values()
                          if v.kind in {Parameter.POSITIONAL_ONLY,
                                        Parameter.POSITIONAL_OR_KEYWORD})

    @wraps(func)
    def func_(*args, **kwargs):
        return func(*(args if has_var_positional
                      else (tuple(args)[:num_positionals])),
                    **(kwargs if has_var_keyword else
                       {k: v for k, v in kwargs.items()
                        if k in sig.parameters.keys()}))
    return func_


def for_all_public_methods(decorator, excluded=[]):
    """Apply decorator to all public callables except those explicitly
    excluded.

    Parameters
    ----------
    decorator : callable
        The decorator to apply
    excluded : list
        A list of method names to which the decorator should not be applied.

    References
    ----------
        http://stackoverflow.com/questions/6307761/how-can-i-decorate-all-functions-of-a-class-without-typing-it-over-and-over-for

    """
    def decorate(cls):
        for attr in dir(cls):
            if ((not attr.startswith('_')) and
                    callable(getattr(cls, attr)) and
                    (attr not in excluded)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate


def timefunc(description, f, enabled=True):
    if enabled:
        start = time.time()
        print('%s starting' % description)
    res = f()
    if enabled:
        print('%s end time: %f' % (description, time.time() - start))
    return res


def timefunc_smooth(description=None, alpha=0.95):
    class timefunc_smooth_(CachedGet):

        def __init__(self, func):
            super().__init__(func)
            self._smo_ldiff = None
            self._smo_gdiff = None
            self._prev_end = None

        def __call__(self, *args, **kwargs):
            start = time.time()
            res = self._func(*args, **kwargs)
            end = time.time()
            ldiff = end - start
            gdiff = None if self._prev_end is None else end - self._prev_end
            self._prev_end = end
            self._smo_ldiff = (ldiff if self._smo_ldiff is None else
                               alpha * self._smo_ldiff + (1 - alpha) * ldiff)
            self._smo_gdiff = (gdiff if (gdiff is None or
                                         self._smo_gdiff is None) else
                               alpha * self._smo_gdiff + (1 - alpha) * gdiff)
            print('%s %e%s' % (
                self._func.__name__ if description is None else description,
                self._smo_ldiff,
                '' if self._smo_gdiff is None else (' %e' % self._smo_gdiff)))
            return res
    return timefunc_smooth_


class indexable_function:
    """
    see also ``Func2List``
    """

    def __init__(self, function):
        self._function = function
        update_wrapper(self, function)

    def __get__(self, obj, objtype):
        if obj is None:
            return self

        return indexable_function(self._function.__get__(obj, objtype))

    def __getitem__(self, id):
        if type(id) == tuple:
            return self._function(*id)
        else:
            return self._function(id)

    def __call__(self, *args, **kwargs):
        return self._function(*args, **kwargs)


def fixed_point_iteration_(f, x, *args, **kwargs):
    y = f(x, *args, **kwargs)
    while y != x:
        x = y
        y = f(x, *args, **kwargs)
    return y


def fixed_point_iteration(f):
    return partial(fixed_point_iteration_, f)
