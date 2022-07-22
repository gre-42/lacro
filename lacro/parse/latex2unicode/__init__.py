# -*- coding: utf-8 -*-
"""
Sources:
    http://en.wikipedia.org/wiki/Unicode_subscripts_and_superscripts
    https://github.com/vikhyat/latex-to-unicode
      lib/data.rb
        generated from
          lib/data/*
      http://vikhyat.net/projects/latex_to_unicode/
"""
from typing import Callable

from lacro.collections import map_recursive
from lacro.decorators import cached_function, cached_property
from lacro.inspext.misc import reraise
from lacro.io.string import load_strings_from_file
from lacro.iterators import eqzip
from lacro.path.pathabs import abspath_of_script_child
from lacro.sort import dict_lensorted
from lacro.string.misc import replace_all


@cached_function
def _frac_dict():
    return dict(eqzip(
        [(1, 2), (1, 4), (3, 4), (1, 7), (1, 9), (1, 10), (1, 3), (2, 3),
         (1, 5), (2, 5), (3, 5), (4, 5), (1, 6), (5, 6), (1, 8), (3, 8),
         (5, 8), (7, 8), (1, 0), (0, 3)],
        '½¼¾⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞⅟↉'))


def frac(a, b):
    import re
    a, b = [x if re.match(r'^[\d\w]+$', x) else '(%s)' % x
            for x in [a, b]]
    try:
        return _frac_dict()[(int(a), int(b))]
    except (ValueError, KeyError):
        return '%s/%s' % (a, b)


def sqrt(v):
    return '√' + v


def combined(c):
    """
    http://www.unicode.org/charts/PDF/U0300.pdf
    http://en.wikipedia.org/wiki/Strikethrough#Unicode
    """
    return lambda v: ''.join(vv + chr(c) for vv in v)


@cached_function
def _tabular_function_items(filename):
    return [l.split(' ') for l in load_strings_from_file(
        abspath_of_script_child('data', filename))]


class _TabularFunction:

    def __init__(self, filename, inverted):
        self._filename = filename
        self._inverted = inverted

    def items(self):
        return _tabular_function_items(self._filename)

    @cached_property
    def _dict(self):
        # split() treats nbsp='\u00A0' identical to whitespace
        # => using split(' ')
        lines = self.items()
        try:
            return dict_lensorted(l[::-1 if self._inverted else 1]
                                  for l in lines)
        except ValueError as e:
            reraise(ValueError(
                'Could not load file "%s"\n%s\n'
                'File contents:\n%s' %
                (abspath_of_script_child('data',
                                         self._filename), e,
                 '\n'.join(map(repr, lines)))))

    def __call__(self, s):
        return replace_all(s, self._dict)


_tabular_function_names = ('symbols bb bf cal frak it mono updown subscripts '
                           'superscripts'.split())
for func_name in _tabular_function_names:
    globals()[func_name] = _TabularFunction(func_name, inverted=False)
    globals()['i' + func_name] = _TabularFunction(func_name, inverted=True)

symbols: Callable[[str], str]
bb: Callable[[str], str]
bf: Callable[[str], str]
cal: Callable[[str], str]
frak: Callable[[str], str]
it: Callable[[str], str]
mono: Callable[[str], str]
updown: Callable[[str], str]
subscripts: Callable[[str], str]
superscripts: Callable[[str], str]

isymbols: Callable[[str], str]
ibb: Callable[[str], str]
ibf: Callable[[str], str]
ical: Callable[[str], str]
ifrak: Callable[[str], str]
iit: Callable[[str], str]
imono: Callable[[str], str]
iupdown: Callable[[str], str]
isubscripts: Callable[[str], str]
isuperscripts: Callable[[str], str]


def latex2unicode(s):
    def operation(s):
        from lacro.parse import macro_expansion
        return macro_expansion.expand(
            symbols(s).replace('\\\\', '\n'),
            {r'\bb': bb,
             r'\bf': bf,
             r'\cal': cal,
             r'\frak': frak,
             r'\it': it,
             r'\mono': mono,
             r'\updown': updown,
             '_': subscripts,
             '^': superscripts,
             r'\shst': combined(0x335),  # short stroke
             r'\st': combined(0x336),  # long stroke
             r'\shso': combined(0x337),  # short solidus
             r'\so': combined(0x338),  # long solidus
             r'\hat': combined(0x302),
             r'\tilde': combined(0x303),
             r'\bar': combined(0x304),
             r'\overline': combined(0x305),
             r'\undersim': combined(0x330),
             r'\underminus': combined(0x320),
             r'\underbar': combined(0x331),
             r'\underline': combined(0x332),
             r'\mathring': combined(0x030a),
             r'\sqrt': sqrt,
             r'\frac': frac})

    return map_recursive(operation, s)


def unicode2latex(s):
    for f in _tabular_function_names:
        # updown converts e.g. "d"<->"p", which means "normal" characters
        # would be changed
        if f != 'updown':
            s = globals()['i' + f](s)
    return s
