# -*- coding: utf-8 -*-

import re

from lacro.inspext.misc import reraise
from lacro.string.misc import checked_match


def _parse_rec(string, rec, verbose, printc, multiarg):
    """
    in: a{b{c}}{d}
    out: (a,b{c},d)
    """
    argssA = []
    while True:
        if verbose:
            print(printc * rec + repr(string))
        start, sep, string = checked_match('^([^{}]*)([{}])?(.*)$', string,
                                           re.DOTALL).groups()
        if sep == '{':
            if verbose:
                print(printc * rec + '<-')
            # necessary to support f{a}{b}
            # support for empty function names excludes multiarg
            if (not multiarg) or (start != ''):
                argssA.append((start,))
            argss, string, dlevel = _parse_rec(string, rec + 1, verbose,
                                               printc, multiarg)
            if not dlevel:
                raise ValueError('Expected "}" before "%s"' % string)
            if verbose:
                print(printc * rec + 'child ' + str(argss))
            if len(argssA) == 0:
                raise ValueError('Empty brackets, like {abc}, require '
                                 'multiarg=False')
            argssA[-1] = argssA[-1] + (argss,)
        else:
            if verbose:
                print(printc * rec + '->')
            if start != '':
                argssA.append(start)
            return argssA, string, (sep == '}')


def _parse(string, verbose=False, printc='  ', multiarg=True):
    argss, string, dlevel = _parse_rec(string, 0, verbose, printc, multiarg)
    if string != '':
        raise ValueError('Did not expect "}" before "%s"' % string)
    if dlevel:
        raise ValueError('String ended with too many "}"')
    return argss


def expand(string, funcs, multiarg=True):
    if multiarg and ('' in funcs.keys()):
        raise ValueError('Empty function names can not be used with multiarg')
    func_keys = '|'.join(re.escape(k) for k in sorted(funcs.keys()))

    def expand2_(dom):
        if type(dom) == str:
            return dom
        elif type(dom) == list:
            return ''.join(map(expand2_, dom))
        elif type(dom) == tuple:
            name, vals = dom[0], dom[1:]
            # '.*? *' instead of '.*? *' to allow for empty string while still
            # taking all whitespaces
            #   '.*?': 'y{Y1} {Y2}' -> 'y{Y1}{Y2}'
            #   '.*? *': 'y{Y1} {Y2}' -> 'y{Y1}{Y2}'

            # '^(.*?\n*)$'
            #   In : re.match('^(.*?)$', 'a\n', re.DOTALL).groups()
            #   Out: ('a',)

            #   In : re.match('^(.*)$', 'a\n', re.DOTALL).groups()
            #   Out: ('a\n',)

            #   In : re.match('^(.*?\n*)$', 'a\n', re.DOTALL).groups()
            #   Out: ('a\n',)

            start, fname = checked_match('^(.*?[\n ]*)(%s) *$' % func_keys,
                                         name, re.DOTALL).groups()
            args = '<unassigned>'  # assign a default value for error message
            try:
                args = [expand2_(v) for v in vals]
                return start + funcs[fname](*args)
            except Exception as e:
                reraise(ValueError('Error executing function "%s" with '
                                   'arguments %r\n%s' % (fname, args, e)))
    return ''.join(expand2_(_parse(string, multiarg=multiarg)))
