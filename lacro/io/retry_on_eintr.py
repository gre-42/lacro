# -*- coding: utf-8 -*-
import os
from functools import wraps

from lacro.io.encext import eopen, eos_fdopen
from lacro.iterators import is_iterable


def retry_function(function, ignored_types=[str, bytes], decorate=False):
    def func(*args, **kwargs):
        while True:
            # https://www.python.org/dev/peps/pep-0475/
            try:
                res = function(*args, **kwargs)
                if decorate:
                    return retry_any(res, ignored_types)
                else:
                    return res
            except InterruptedError:
                continue
            # except IOError as e:
                # if e.errno == errno.EINTR:
                # continue
                # else:
                # raise
    return func


def retry_attributes(obj, ignored_types=[str, bytes]):
    class class_attribute:

        def __init__(self, name):
            self._name = name

        def __get__(self, obj1, objtype):
            attr = retry_function(getattr, ignored_types,
                                  decorate=False)(obj.__class__, self._name)

            def func(*args, **kwargs):
                if (len(args) > 0) and (args[0].__class__.__name__ ==
                                        'RetryClass'):
                    args = (args[0].__class__._opened,) + tuple(args[1:])
                if obj1 is None:
                    return attr(*args, **kwargs)
                else:
                    return attr(obj1.__class__._opened, *args, **kwargs)
            return retry_any(func if callable(attr) else attr, ignored_types)

    def obj_attribute(self, name):
        if name in ('__class__',):
            assert object.__getattribute__(self, name) == RetryClass
            return object.__getattribute__(self, name)
        assert name != '_opened', ('use obj.__class__._opened '
                                   'instead of obj._opened')
        assert self.__class__._opened == obj

        return retry_function(getattr, ignored_types,
                              decorate=True)(obj, name)

    RetryClass = type(
        'RetryClass', (),
        dict([(k, class_attribute(k)) for k in dir(obj.__class__)
              if k not in ('__init__', '__new__', '__del__',
                           '__getattribute__', '__qualname__', '__class__')] +
             [('__getattribute__', obj_attribute), ('_opened', obj)]))
    return RetryClass()


def retry_any(obj, ignored_types=[str, bytes]):
    if type(obj) in ignored_types:
        return obj
    assert (callable(obj) + is_iterable(obj)) <= 1
    if callable(obj):
        return retry_function(obj, ignored_types, decorate=True)
    if is_iterable(obj):
        return retry_attributes(obj, ignored_types)
    return obj


@retry_any
@wraps(eopen)
def retry_open(*args, **kwargs):
    return eopen(*args, **kwargs)


@retry_any
@wraps(os.open)
def os_retry_open(*args, **kwargs):
    return os.open(*args, **kwargs)


@retry_any
@wraps(eos_fdopen)
def os_retry_fdopen(*args, **kwargs):
    return eos_fdopen(*args, **kwargs)
