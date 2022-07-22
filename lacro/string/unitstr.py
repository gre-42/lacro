# -*- coding: utf-8 -*-
import decimal
import re
import string
from typing import List, Union


def float_to_decimal(f):
    """
    Convert a floating point number to a Decimal with no loss of information
    http://docs.python.org/library/decimal.html#decimal-faq
    """
    n, d = f.as_integer_ratio()
    numerator, denominator = decimal.Decimal(n), decimal.Decimal(d)
    ctx = decimal.Context(prec=60)
    result = ctx.divide(numerator, denominator)
    while ctx.flags[decimal.Inexact]:
        ctx.flags[decimal.Inexact] = False
        ctx.prec *= 2
        result = ctx.divide(numerator, denominator)
    return result


def float_2_str_signif(number, sigfig):
    """
    http://stackoverflow.com/questions/2663612/nicely-representing-a-floating-point-number-in-python/2663623#2663623
    """
    import numpy as np
    if not np.isfinite(number):
        return str(number)
    assert sigfig > 0
    try:
        d = decimal.Decimal(number)
    except TypeError:
        d = float_to_decimal(float(number))
    sign, digits, exponent = d.as_tuple()
    if len(digits) < sigfig:
        digits = list(digits)
        digits.extend([0] * (sigfig - len(digits)))
    shift = d.adjusted()
    result = int(''.join(map(str, digits[:sigfig])))
    # Round the result
    if len(digits) > sigfig and digits[sigfig] >= 5:
        result += 1
    result = list(str(result))
    # Rounding can change the length of result
    # If so, adjust shift
    shift += len(result) - sigfig
    # reset len of result to sigfig
    result = result[:sigfig]
    if shift >= sigfig - 1:
        # Tack more zeros on the end
        result.extend(['0'] * (shift - sigfig + 1))
    elif 0 <= shift:
        # Place the decimal point in between digits
        result.insert(shift + 1, '.')
    else:
        # Tack zeros on the front
        assert shift < 0
        result = ['0.'] + ['0'] * (-shift - 1) + result
    if sign:
        result.insert(0, '-')
    return ''.join(result)


def precision_for_length(f, max_len):
    for d in reversed(range(10)):
        res = '{:.{}f}'.format(f, d)
        # print(res)
        if len(res) <= max_len:
            # print('b')
            return res


class NotInRangeError(ValueError):
    pass


# ---------
# - units -
# ---------


def quantity_to_symbol_mult(quantity, tex=False, force_char=False,
                            percent=False, exp=False):
    import numpy as np
    nosym1 = ' ' if force_char else ''
    if percent:
        if not np.isfinite(quantity):
            return {'symbol': nosym1, 'mult': 1}
        if quantity == 0:
            return {'symbol': nosym1, 'mult': 1}
        if abs(quantity) < 1e-3 and exp:
            raise NotInRangeError()
        if abs(quantity) < 1e-2:
            return {'symbol': '‰', 'mult': 1e+3}
        return {'symbol': '%', 'mult': 1e+2}
    else:
        if not np.isfinite(quantity):
            return {'symbol': nosym1, 'mult': 1}
        if quantity == 0:
            return {'symbol': nosym1, 'mult': 1}
        if abs(quantity) < 1e-3:
            return {'symbol': (r'{\boldmath$\mu$}'
                               if tex else 'μ'), 'mult': 1e+6}
        if abs(quantity) < 1e0:
            return {'symbol': 'm', 'mult': 1e+3}
        if abs(quantity) < 1e3:
            return {'symbol': nosym1, 'mult': 1e0}
        if abs(quantity) < 1e6:
            return {'symbol': 'k', 'mult': 1e-3}
        if abs(quantity) < 1e9:
            return {'symbol': 'M', 'mult': 1e-6}
        if abs(quantity) < 1e12:
            return {'symbol': 'G', 'mult': 1e-9}
        if abs(quantity) < 1e15:
            return {'symbol': 'T', 'mult': 1e-12}
    return {'symbol': '?', 'mult': 1e0}


def abc_float2str(f, round=True, signif=None, commalen=None):
    if commalen is not None:
        return precision_for_length(f, commalen)
    elif signif is not None:
        return float_2_str_signif(f, signif)
    else:
        import math
        if round and (abs(math.fmod(10 * f, 10)) < 1e-1 or abs(f) >= 100):
            # return str(round2i(f))
            return '%0.0f' % f
        else:
            return '%0.1f' % f


def abc_quantity2str(v, width=0, round=True, signif=None, commalen=None,
                     exp=False, **kwargs):
    try:
        sm = quantity_to_symbol_mult(v, exp=exp, **kwargs)
        return (abc_float2str(v * sm['mult'], round=round, signif=signif,
                              commalen=commalen) + sm['symbol']).rjust(width)
    except NotInRangeError:
        lenn0 = (signif - 1
                 if signif is not None else commalen
                 if commalen is not None else width)
        lenn = max(0, lenn0 - 6 if v < 1 else lenn0 - 5)
        # TODO: subtract length of longest symbol
        # print(lenn, '{:%d.%de}'%(width,lenn))
        return ('{:%d.%d%s}' % (width, lenn, exp)).format(v)


class UnitFormatted:

    def __init__(self, value: Union[str, float, List[object]]) -> None:
        self._value = value

    def __format__(self, format_spec):
        gr = re.match(r'^(\d*)(%?)([Ee]?)H(t?)(\d*)([fso]?)(c?)$', format_spec)
        if gr is not None:
            width, p, e, t, sig, fs, c = gr.groups()
            if width is None:
                width = 0
            if fs == 's' and sig == '':
                raise ValueError('"s" given, but no significance specified')
            if fs == 'o' and sig == '':
                raise ValueError('"o" given, but no length specified')
            return abc_quantity2str(
                self._value,
                width=0 if width == '' else int(width),
                tex=(t != ''),
                force_char=(c != ''),
                percent=(p != ''),
                round=(fs == ''),
                signif=(int(sig) if fs == 's' else None),
                commalen=(int(sig) if fs == 'o' else None),
                exp=(False if e == '' else e))
        if format_spec.endswith('nN'):
            return ('' if self._value is None else
                    self._value.__format__(format_spec[:-2]))
        if format_spec.startswith('j'):
            # print(format_spec[1:])
            # return ' '.join(self.format_field(v, format_spec[1:])
            #                 for v in value)
            return format_spec[1:].join(map(str, self._value))
        # {var:n value}
        for a, b, true in ((r'\?', r'\|', True), (r'\|', r'\?', False)):
            gr = re.match(f'^{a}(.*?)(?:{b}(.*))?$', format_spec)
            if gr is not None:
                if bool(self._value) is true:
                    res = gr.group(1)
                else:
                    res = '' if gr.group(2) is None else gr.group(2)
                return res
                # return res.replace('$',str(value))
        gr = re.match('^/(.+)$', format_spec)
        if gr is not None:
            from lacro.path.pathrel import cwd_relpath
            return cwd_relpath(self._value, gr.group(1))
        try:
            return self._value.__format__(format_spec)
        except TypeError as ex:
            raise TypeError(f'{ex}\nType(value): {type(self._value)!r}\n'
                            f'Format spec: {format_spec}')


class UnitFormatter(string.Formatter):

    def format(*args, **kwargs):
        # allow for arbitrarily named arguments (also "self" and
        # "format_string")
        return args[0].vformat(args[1], args[2:], kwargs)
        # return super().format(self, format_string, args, kwargs)

    def vformat(self, format_string, args, kwargs):
        # allow "del args[0]" by converting "args" to "list"
        # WARNING: this allows for mixing automatic numbering and manual
        # numbering, which the default str.format method does not support
        args = list(args)
        used_args = set()
        result = self._vformat(format_string, args, kwargs, used_args,
                               recursion_depth=10)
        if type(result) == tuple:
            # API change somewhere between 3.4.1 and 3.5.1
            assert len(result) == 2
            result = result[0]
        self.check_unused_args(used_args, args, kwargs)
        return result

    def format_field(self, value, format_spec):
        # print('value', value, 'spec', format_spec)
        return UnitFormatted(value).__format__(format_spec)

    def get_value(self, key, args, kwargs):
        # print('v', key, args, kwargs)
        if key == '':
            v = args[0]
            del args[0]
            return v
        else:
            return super().get_value(key, args, kwargs)

    def get_field(self, field_name, args, kwargs):
        # print('k', field_name, args, kwargs)
        return super().get_field(field_name, args, kwargs)


unit_formatter = UnitFormatter()
