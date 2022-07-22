# -*- coding: utf-8 -*-

import inspect
import os.path
import re
from collections import OrderedDict
from typing import Dict, Iterable

from lacro.assertions import (asserted_of_type, dict_assert_function,
                              dict_assert_inverse_surjective,
                              set_assert_contains)
from lacro.decorators import ignore_unknown_arguments
from lacro.string.numeric import shorten_float_str


def trunc(text, max_len):
    if max_len < 7:
        return text[:max_len]
    else:
        m = '..'
        b = max_len // 2 - len(m) // 2
        remain = max_len - b - len(m)
        e = len(text) - remain
        return text if len(text) <= max_len else text[:b] + m + text[e:]


def nice_typenames(st):
    return re.sub("<(type|class) '(.*?)'>", r'\2', st)


def with_module(module, name, add_module):
    return (('%s.%s' % (module, name)) if (add_module
                                           and (module != 'builtins')) else
            name)


def _joined_collection_repr(strs, ms):
    return ms % _newline_join(strs, ', ', ',\n\n')


def _joined_collection_str(strs, ms):
    return ms % _newline_join(strs, ', ', '\n\n')


def to_repr(v, **kwargs):
    def func(type_repr=False, add_module=False, sort_dict=False, **kwargs1):
        if inspect.isclass(v):
            if type_repr:
                return with_module(v.__module__, v.__name__, add_module)
            else:
                return repr(v)
        else:
            from lacro.stdext import HashableOrderedDict

            def tor(vv):
                return to_repr(vv, **kwargs)

            if type(v) in (list, tuple):
                strs = [tor(e) for e in v]
                ms = {list: '[%s]', tuple: ('(%s)' if len(v) != 1 else
                                            '(%s,)')}[type(v)]
                return _joined_collection_repr(strs, ms)
            elif type(v) == dict:
                from lacro.collections import object_order_key
                strs = ['%s: %s' % (tor(k), tor(vv)) for k, vv in
                        (sorted(v.items(),
                                key=lambda kv: object_order_key(kv[0]))
                         if sort_dict else v.items())]
                ms = '{%s}'
                return _joined_collection_repr(strs, ms)
            elif type(v) in (OrderedDict, HashableOrderedDict):
                strs = [tor(kv) for kv in v.items()]
                ms = '%s([%%s])' % v.__class__.__name__
                return _joined_collection_repr(strs, ms)
            elif type(v) == str:
                return repr(v)
            elif hasattr(v, '__to_repr__'):
                # do not use **kwargs to pass default values
                return v.__to_repr__(type_repr=type_repr,
                                     add_module=add_module,
                                     sort_dict=sort_dict, **kwargs1)
            else:
                return with_module(v.__class__.__module__, repr(v), add_module)
    return func(**kwargs)


def to_str(v, **kwargs):
    def func(nice_dict=False, preserve_str=False, sort_dict=False,
             float_fmt=None, shorten=False):
        import numpy as np

        from lacro.stdext import (GetattrHashableOrderedDict,
                                  HashableOrderedDict, dict_2_str)

        def tos(vv):
            return to_str(vv, **kwargs)

        if (type(v) == np.ndarray) and (v.dtype.type != np.object_):
            return str(v)
        if type(v) in (list, tuple, np.ndarray):
            strs = [tos(e) for e in v]
            ms = {list: '[%s]', tuple: '(%s)', np.ndarray: '<%s>'}[type(v)]
            return _joined_collection_str(strs, ms)
        elif type(v) in (dict, OrderedDict, HashableOrderedDict,
                         GetattrHashableOrderedDict):
            if nice_dict:
                return dict_2_str(v, **kwargs)
            else:
                return '{%s}' % ', '.join(
                    '%s: %s' % (tos(k), tos(vv))
                    for k, vv in (sorted(v.items())
                                  if (sort_dict and type(v) == dict) else
                                  v.items()))
        if (type(v) in (np.float32, np.float64)) and (float_fmt is not None):
            def sho(f): return shorten_float_str(f) if shorten else f

            return sho(float_fmt.format(v))
        elif isinstance(v, str):
            if not preserve_str:
                return repr(v)
            else:
                return v
        elif inspect.isclass(v):
            return to_repr(v, **kwargs)
        elif hasattr(v, '__tostr__'):
            return v.__tostr__(**kwargs)
        else:
            return str(v)
    return ignore_unknown_arguments(func)(**kwargs)


def to_strp(*args, preserve_str=True, **kwargs):
    return to_str(*args, preserve_str=preserve_str, **kwargs)


class XhtmlText(str):

    def __repr__(self):
        return '%s(%s)' % (__class__.__name__, str.__repr__(self))


def to_xhtml(v, to_str=to_strp, **kwargs):
    if type(v) == XhtmlText:
        return v
    elif hasattr(v, '__to_xhtml__'):
        return XhtmlText(v.__to_xhtml__(**kwargs))
    else:
        return XhtmlText(escape_xml_text(to_str(v, **kwargs)))


def _newline_join(iterable: Iterable[str], j0: str, j1: str) -> str:
    """
    join iterabe ``iterable`` by ``j0`` if the previous element contained
    a newline, else join by ``j1``
    """
    res = ''
    for i, v in enumerate(iterable):
        if i != 0:
            # noinspection PyUnboundLocalVariable
            res += j1 if '\n' in o else j0
        res += v
        o = v
    return res


def indented(text, width=2, char=' ', skip_empty=False):
    return indentedl(text.split('\n'), width, char, skip_empty)


def indentedl(text, width=2, char=' ', skip_empty=False):
    if not callable(width):
        def width(i, w=width): return w
    return '\n'.join(s if skip_empty and s == '' else
                     '%s%s' % (char * width(i), s) for i, s in enumerate(text))


box_chars_1 = {
    'ul': '┌',
    'lr': '┘',
    'ur': '┐',
    'll': '└',
    'cx': '│',
    'xc': '─'}


def textbox(text, char='#'):
    if type(char) == str:
        char = {
            'ul': char,
            'lr': char,
            'ur': char,
            'll': char,
            'cx': char,
            'xc': char}
    lines = text.split('\n')
    fill = max(map(len, lines))
    return ('{char[ul]}{0:{char[xc]}<{fill}}{char[ur]}\n{1:}'
            '{char[ll]}{0:{char[xc]}<{fill}}{char[lr]}').format(
                '',
                ''.join('{char[cx]} {text:<{fill}} {char[cx]}\n'.format(
                    char=char,
                    fill=fill,
                    text=text)
                    for text in lines),
                fill=fill + 2,
                char=char)


def textline(text, length, char='='):
    return (char * ((length - len(text) - 2) // 2)
            + (' ' + text + ' ' if text != '' else char * 2)
            + char * ((length - len(text) - 1) // 2))


def strunc(text, max_len):
    return trunc(str(text), max_len)


def unique_substrings(strings):
    if len(strings) == 0:
        return strings
    min_len = min(len(s) for s in strings)
    L = None
    for l in reversed(range(min_len)):
        if all(s.startswith(strings[0][:l]) for s in strings):
            L = l
            break

    R = None
    for r in reversed(range(min_len)):
        if all(s.endswith(strings[0][-r - 1:]) for s in strings):
            R = -r - 1
            break
    # print(L,R)
    return [s[L:R] for s in strings]


def to_camelcase(s, split_chars='_'):
    return ''.join('%s%s' % (v[0:1].upper(), v[1:].lower())
                   for v in s.split(split_chars))


# http://aitech.ac.jp/~ckelly/midi/help/caps.html
english_lower_words = (
    'a an the '  # articles
    'and but or nor '  # conjunctions
    # prepositions that are less than five letters long
    'at by for from in into of off on onto out over to up with '
    'as aka').split()  # (only if it is followed by a noun)


def capitalize(s, lowered=english_lower_words):
    return re.sub(r'(\w+)', lambda g: (g.group(1).lower()
                                       if (g.group(1) in lowered)
                                       else (g.group(1)[0].upper() +
                                             g.group(1)[1:])), s)


def lowerize(s, keep=[], capitalized=[]):
    res = re.sub(r'(\w+)', lambda g: (g.group(1) if (g.group(1) in keep) else
                                      g.group(1)[0].upper() + g.group(1)[1:]
                                      if (g.group(1) in capitalized) else
                                      g.group(1).lower()), s)
    return res[0].upper() + res[1:]


def added_line_numbers(st):
    from lacro.string.numeric import maxnum_2_strlen
    lst = st.split('\n')
    l10 = maxnum_2_strlen(len(lst))
    return '\n'.join(('%%%dd: %%s' % l10) % (1 + i, l)
                     for i, l in enumerate(lst))


def L(s, dct=None, udct={}):
    from lacro.collections import dict_union
    from lacro.inspext.misc import parent_vars
    if dct is None:
        dct = parent_vars()
    dct = dict_union(dct, udct)
    if type(s) == list:
        return [L(v, dct) for v in s]
    else:
        try:
            return s % dct
        except ValueError as e:
            raise ValueError('Can not format string "%s" using dict "%s".'
                             '\n%s' % (trunc(s, 200), trunc(str(dct), 200), e))


def _F(s, dct, udct, nparent, args, kwargs):
    from lacro.collections import dict_union, map_recursive
    from lacro.inspext.misc import parent_vars
    from lacro.string.unitstr import unit_formatter
    for a in args:
        if type(a) == dict:
            raise ValueError('dct used as positional argument')
    if dct is None:
        dct = parent_vars(nparent)
    dct = dict_union(dct, udct, kwargs)

    def operation(s):
        asserted_of_type(s, str)
        assert type(s) == str
        try:
            return unit_formatter.format(s, *args, **dct)
            # return s.format(*args, **dct)
        except KeyError as e:
            raise ValueError('KeyError in string format. Key: %s. '
                             'Format keys: \n%s' %
                             (e, iterable_2_repr(sorted(dct.keys()))))
            # raise ValueError('Could not find key %s in keys \n%s' %
            #  (e, trunc(str(sorted(dct.keys())), 200)))

    return map_recursive(operation, s)


def FF(*args, **kwargs):
    args = list(args)
    s = args.pop(0)
    dct = kwargs.pop('dct', None)
    udct = kwargs.pop('udct', {})
    return _F(s, dct=dct, udct=udct, nparent=3, args=args, kwargs=kwargs)


def F0(*args, **kwargs):
    args = list(args)
    s = args.pop(0)
    udct = kwargs.pop('udct', {})
    return _F(s, dct={}, udct=udct, nparent=3, args=args, kwargs=kwargs)


# raises an exception if the dictionary contains unused variables


def F1(s: str, dct: Dict[str, str]) -> str:
    return replace_all(s, {'{%s}' % k: v for k, v in dct.items()},
                       must_exist=True)


def old_format_to_new(s):
    s = re.sub(r'%\((\w+)\)s', r'{\1}', s)
    s = re.sub(r'%\((\w+)\)(.)', r'{\1:\2}', s)
    return s


def checked_getenv(name, default='__raise__', type=None):
    # alternative implementation: return dict_get(os.environ, name)
    from os import getenv
    res = getenv(name)
    if res is None:
        if default is not '__raise__':
            return default
        else:
            raise ValueError('Could not find environment variable "%s"' %
                             name)
    elif type is not None:
        try:
            return type(res)
        except ValueError:
            from lacro.inspext.misc import reraise
            reraise(ValueError('Could not convert environment variable "%s" '
                               'to type "%s"' % (name, type.__name__)))
    else:
        return res


def _checked_match(method, pattern, string, flags, msg,
                   print_pattern, print_string):
    res = method(pattern, string, flags)
    if res is None:
        raise ValueError('%sCould not find pattern%s' % (msg, ''.join(
            (['\n\nPattern\n%r' % pattern] if print_pattern else [])
            + (['\n\nString\n%s' % string] if print_string else []))))
    return res


def checked_match(pattern, string, flags=0, msg='', print_pattern=True,
                  print_string=True):
    return _checked_match(re.match, pattern, string, flags, msg,
                          print_pattern, print_string)


def checked_search(pattern, string, flags=0, msg='', print_pattern=True,
                   print_string=True):
    return _checked_match(re.search, pattern, string, flags, msg,
                          print_pattern, print_string)


def match_single(*args, **kwargs):
    res = re.match(*args, **kwargs)
    if res is None:
        return None
    else:
        assert len(res.groups()) == 1
        return res.group(1)


def rename_prefix(s, old, new, assure_exists=True):
    if assure_exists and not s.startswith(old):
        raise ValueError('"%s" does not have prefix "%s"' % (s, old))
    return re.sub('^%s(.*)$' % old, r'%s\1' % new, s)


def rstripstr(s, right, must_exist=False):
    return checked_match('(.*?)(?:%s)%s$' % (re.escape(right), ''
                                             if must_exist else '?'),
                         s).group(1)


def replace_all(text, dic, must_exist=False, whole_word=False):
    if type(dic) in [tuple, list]:
        dic = OrderedDict(dic)
    if must_exist:
        for i in dic.keys():
            if ((whole_word and (not re.match(r'.*\b%s\b.*' % i, text)))
                    or ((not whole_word) and (i not in text))):
                raise ValueError('Could not find a single occurence '
                                 'of "%s"' % i)

    for i, j in dic.items():
        text = (re.sub(r'\b%s\b' % i, j, text) if whole_word else
                text.replace(i, j))
    return text


def replace_all_re(text, dic):
    for i, j in dic.items():
        text = re.sub(i, j, text)
    return text


def class_attr(cla):
    return f' class="{cla}"' if cla is not None else ''


def attr_val(name, val):
    return f' {name}="{val}"' if val is not None else ''


def joined_attr_vals(dct):
    return ' '.join(s for s in (attr_val(n, v) for n, v in sorted(dct.items()))
                    if s != '')


def opt_if_nonnull(name_values):
    return [('%s %s' % nv if type(nv) == tuple else '%s' % nv)
            for nv in name_values if type(nv) != tuple or nv[1] is not None]


def tree_2_str(tree, stop=[str], width=1, rec=0, maxrec=10):
    if rec > maxrec:
        return ('<stopping because current recursion depth of %d is larger '
                'than maximum recursion depth of %d>' % (rec, maxrec))

    if hasattr(tree, 'items'):
        from lacro.collections import items_2_str
        return type(tree).__name__ + '\n' + indented(
            items_2_str(((k, tree_2_str(v, stop, width, rec + 1, maxrec))
                         for k, v in tree.items()), width), width, ' ')
    else:
        from lacro.iterators import is_iterable
        if is_iterable(tree) and (type(tree) not in stop):
            return type(tree).__name__ + '\n' + indented(
                iterable_2_str([tree_2_str(v, stop, width, rec + 1, maxrec)
                                for v in tree], '\n'), width, ' ')
        else:
            return str(tree)  # type(tree).__name__+' '+


def text_sliding_window(text, known_words, length, ignore_case=False,
                        preserve='space', filter=lambda v: True):
    set_assert_contains(['space', 'space+newline', 'none'], preserve)
    known_regs = [re.compile(w, (re.IGNORECASE if ignore_case else 0))
                  for w in known_words]
    import numpy as np
    from scipy.ndimage.morphology import binary_dilation
    words = np.array(re.findall(r'(\w+|[^\w]+)' if preserve != 'none' else
                                r'\w+', text), dtype=object)
    known = [any(r.match(w) for r in known_regs) for w in words]
    known = binary_dilation(known, np.ones((length * 2 if preserve != 'none'
                                            else length,)))
    # return ' '.join(w for w in words[known])

    from scipy.ndimage import label
    labl, num_features = label(known, structure=np.ones((3,)))
    return '\n'.join(l for l in (('' if preserve != 'none' else
                                  ' ').join(w.replace('\n', ' ')
                                            if preserve == 'space' else
                                            w for w in words[labl == n + 1])
                                 for n in range(num_features)) if filter(l))


def re_colors(text, word2color, ignore_case=False, search=False):
    r2k = re_keys_2_keys(re.findall(r'\w+', text), word2color.keys(),
                         keys_must_exist=False, is_unique=False,
                         ignore_case=ignore_case, search=search)
    return {w: word2color[r] for r, W in r2k.items() for w in W}


color_styles = dict(
    bold='\033[1m',
    ok_blue='\033[94m',
    ok_green='\033[92m',
    black='\033[30m',
    blue='\033[34m',
    green='\033[32m',
    cyan='\033[36m',
    red='\033[31m',
    purple='\033[35m',
    brown='\033[33m',
    light_gray='\033[37m',
    dark_gray='\033[1;30m',
    light_blue='\033[1;34m',
    light_green='\033[1;32m',
    light_cyan='\033[1;36m',
    light_red='\033[1;31m',
    light_purple='\033[1;35m',
    yellow='\033[1;33m',
    white='\033[1;37m',

    bg_black='\033[40m',
    bg_blue='\033[44m',
    bg_green='\033[42m',
    bg_cyan='\033[46m',
    bg_red='\033[41m',
    bg_purple='\033[45m',
    bg_brown='\033[43m',
    bg_light_gray='\033[47m')


def colored(text, styles):
    from lacro.collections import dict_get
    endc = '\033[0m'

    if type(styles) == str:
        styles = [styles]

    asserted_of_type(styles, [list, tuple])
    return ''.join(dict_get(color_styles, s) for s in styles) + text + endc


def colored_words(text, word2color, ignore_case=False):
    if ignore_case:
        word2color = {w.lower(): c for w, c in word2color.items()}

    def lo(w): return w.lower() if ignore_case else w

    return re.sub(r'(\w+)', lambda t: (colored(t.group(1),
                                               word2color[lo(t.group(1))])
                                       if (lo(t.group(1)) in word2color.keys())
                                       else t.group(1)), text)


def re_keys_2_keys(keys, re_keys, keys_must_exist=True, is_unique=True,
                   pattern='re_keys', ignore_case=False, search=False):
    match = re.search if search else re.match
    set_assert_contains(['re_keys', 'keys'], pattern)
    Ks = OrderedDict((rk, [k for k in keys
                           if (match(rk, k, (re.IGNORECASE
                                             if ignore_case else 0))
                               if pattern == 're_keys' else
                               match(k, rk, (re.IGNORECASE
                                             if ignore_case else 0)))])
                     for rk in re_keys)
    if not keys_must_exist:
        Ks = OrderedDict((rk, K) for rk, K in Ks.items() if len(K) > 0)
    if is_unique:
        dict_assert_function(Ks)
        return OrderedDict((rk, K[0]) for rk, K in Ks.items())
    else:
        dict_assert_inverse_surjective(Ks, msg='Available keys: %s\n' %
                                               iterable_2_repr(keys))
        return Ks


def keys_intersect_re_keys(keys, re_keys, keys_must_exist=True,
                           pattern='re_keys'):
    from lacro.collections import lists_union
    return lists_union(re_keys_2_keys(keys, re_keys, keys_must_exist,
                                      is_unique=False,
                                      pattern=pattern).values(),
                       allow_duplicates=True, order='keep')


def hyphenation_cat(v, known):
    return re.sub(''.join('(%s)?' % k.replace('-', '') for k in known),
                  lambda g: '\u00AD'.join(hyphenation(vv, known)
                                          for vv in g.groups()
                                          if vv is not None), v)


def hyphenation(v, known):
    return replace_all(v, {k.replace('-', ''): k.replace('-', '\u00AD')
                           for k in known})


def hyphenation_all(v):
    return '\u00AD'.join(v)


def mywrap(s, *args, **kwargs):
    from textwrap import TextWrapper

    class MyTextwrap(TextWrapper):

        def _split(self, text):
            return [] if len(text) == 0 else [c for c in text.split('\u00AD')]

    return '\n'.join(MyTextwrap(*args, **kwargs).fill(s1)
                     for s1 in s.split('\n'))


# html / xml


def html_to_xml(st):
    """
    http://changelog.ca/log/2006/06/12/making_nbsp_work_with_xml_rss_and_atom
    "&#160;" instead of "&nbsp;"

    see also: html.escape / html.unescape
    """
    return st.replace('&nbsp;', '&#160;')


def escape_xml_text(st):
    """
    see also: html.escape / html.unescape
    """
    return st.replace('&', '&#38;').replace('<', '&#60;').replace('>', '&#62;')


# def escape_latex(st, only_text):
#     return st if (only_text and ('$' in st)) else st.replace('_','\_')


def escape_latex(st, only_text=False):
    """
    http://stackoverflow.com/questions/16259923/how-can-i-escape-latex-special-characters-inside-django-templates
    """
    CHARS = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless',
        '>': r'\textgreater',
        '¹': '$^{1}$',
        '²': '$^{2}$',
        '³': '$^{3}$',
        'β': r'$\beta$',
    }

    return (st if (only_text and ('$' in st)) else
            (''.join(CHARS.get(char, char) for char in st)))


def exp_latex(st):
    s, a, b, se, e = checked_match(r'^([+-]?)(\d+)\.(\d+)e([+-])0+(\d+)$',
                                   st).groups()
    se = se.strip('+')
    if (se, e) != ('', '0'):
        return s + a + '.' + b + r'$\cdot$10$^{\text{' + se + e + '}}$'
    else:
        return s + a + '.' + b


def assert_valid_regex(v):
    try:
        re.compile(v)
    except Exception as e:
        raise ValueError('Not a valid regex: %s\n%s' % (v, e))


def iterable_2_str(lst, delimiter=', ', to_str=str):
    return delimiter.join(map(to_str, lst))


def iterable_2_repr(*args, **kwargs):
    return iterable_2_str(*args, to_str=repr, **kwargs)


def paste(strings, delimiter=' ', assure_equal_length=True):
    """
    Unix paste
    """
    liness = [string.split('\n') for string in strings]
    if assure_equal_length:
        le = [len(lines) for lines in liness]
        if min(le) != max(le):
            raise ValueError('received lists of unequal length (%d != %d)' %
                             (min(le), max(le)))
    return '\n'.join(delimiter.join(t) for t in zip(*liness))


def assert_no_envvars(st):
    if os.path.expandvars(st) != st:
        raise ValueError('String contains environment variables in line(s)\n%s'
                         % '\n'.join('%3d: %s' % (i, l)
                                     for i, l in enumerate(st.split('\n'))
                                     if l.find('$') != -1))
    return st


class PrintInc:

    def __init__(self, eol='\n', xhtml=False, classes={'code': 'code'}):
        self.s = ''
        self._eol = eol
        self._pre = False
        self._xhtml = xhtml
        self._classes = classes

    def __enter__(self):
        self._pre = False
        return self

    def __exit__(self, type, value, traceback):
        self.finish()

    def print(self, v, end=None, pre=False, escape=True):
        if self._xhtml:
            if pre and not self._pre:
                # '<pre>'
                self.s += '<div%s>' % class_attr(self._classes['code'])
            if self._pre and not pre:
                self.s += '</div>'  # '<pre/>'
            self._pre = pre
        self.s += ((to_xhtml(v) if escape else v)
                   + (end if end is not None else self._eol))

    def finish(self):
        self.print('', end='', pre=False)


def map_or_None(f, s):
    assert type(s) == str
    if s == 'None':
        return None
    else:
        return f(s)


def sub_count(pattern: str, repl: str, string: str) -> str:
    """Pattern substitution with f-string syntax and an additional counter argument.

    Source:
    http://stackoverflow.com/a/16762094/2292832

    Parameters
    ----------
    pattern : str
        Regular expression pattern.
    repl : str
        Replacement string to be inserted, in f-string style ({0}, {1}, {a}).
    string : str
        String to apply the function to.

    Returns
    -------
    str
        The replaced string.

    """
    counter = 0

    def _sub(match):
        nonlocal counter
        counter += 1
        return repl.format(*match.groups(''), counter=counter,
                           **match.groupdict())

    return re.sub(pattern, _sub, string)


def clean_using_dictionary(noisy, dictionary):
    from difflib import SequenceMatcher as SM

    import numpy as np
    return dictionary[np.argmax([max(SM(None, n, d).ratio() for n in noisy)
                                 for d in dictionary])]


def is_normal_key(s):
    return not re.match('^__.*__$', s)
