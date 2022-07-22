# -*- coding: utf-8 -*-
import re
from functools import partial
from typing import Callable, Iterable, cast

_find_unsafe = re.compile(r'[^\w@%+=:,./-]', re.ASCII).search


def _quote(s: str, *, q) -> str:
    """Return a shell-escaped version of the string *s*.

    Same as shlex.quote except for the 'q' parameter.
    """
    n = {"'": '"', '"': "'"}[q]
    if not s:
        return q + q
    if _find_unsafe(s) is None:
        return s

    # use single quotes, and put single quotes into double quotes
    # the string $'b is then quoted as '$'"'"'b'
    return q + s.replace(q, q + n + q + n + q) + q


def _join(lst: Iterable[str], *, q, sep) -> str:
    return sep.join(map(cast(Callable[[str], str], partial(_quote, q=q)), lst))


def qquote(s: str) -> str:
    return _quote(s, q="'")


def dquote(s: str) -> str:
    return _quote(s, q='"')


def qjoin(lst: Iterable[str], sep=' ') -> str:
    return _join(lst, q="'", sep=sep)


def djoin(lst: Iterable[str], sep=' ') -> str:
    return _join(lst, q='"', sep=sep)
