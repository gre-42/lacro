# -*- coding: utf-8 -*-
from functools import wraps


def _eopen(file, mode='r', buffering=-1, encoding=None,
           errors=None, newline=None, closefd=True, opener=None, _opener=None):
    import re
    if not re.match(r'^[rwa]b?\+?$', mode):
        raise ValueError('Unknown mode')
    if ('b' not in mode) and (encoding is None):
        raise ValueError('Text mode requires encoding')
    return _opener(file, mode, buffering, encoding, errors, newline, closefd,
                   opener)


@wraps(_eopen)
def eopen(*args, **kwargs):
    """
    A wrapper for `open` that raises an error if a file is opened in text
    mode without explicitly specifying a codec.
    """
    return _eopen(*args, _opener=open, **kwargs)


@wraps(_eopen)
def eos_fdopen(*args, **kwargs):
    """
    A wrapper for `os.fopen` that raises an error if a file is opened in text
    mode without explicitly specifying a codec.
    """
    import os
    return _eopen(*args, _opener=os.fdopen, **kwargs)
