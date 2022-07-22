# -*- coding: utf-8 -*-

import math
import re
from typing import cast

from simpleeval import simple_eval


def align_decimal(number, left_pad=7, right_pad=2, fillchar=''):
    """
    Format a number in a way that will align decimal points.
    http://stackoverflow.com/a/11035185/2292832
    http://stackoverflow.com/a/9549302/2292832
    """
    return ("{part[0]:>%s%s}{part[1]:%s}{part[2]:<%s}" %
            (fillchar, left_pad, '1' * (right_pad > 0),
             right_pad)).format(part=('%g' % number).partition('.'))


def under_g(v):
    return ('%g' % v).replace('.', '-')


def shorten_float_str(f):
    from lacro.string.misc import checked_match
    return ''.join(checked_match(r'^(\d*)(\.)(\d*)(e)(?:\+)?(\-)?(?:0*)(\d+)$',
                                 f).groups(''))


def int_string_order_key(s):
    return [((0, int(i)) if s == '' else (1, s))
            for i, s in re.findall(r'(?:(\d+)|([^\d]+))', s)]


def float_string_order_key(s):
    return [((0, float(f)) if s == '' else (1, s))
            for f, s in re.findall(r'(?:(\d+(?:\.\d+)?)|([^\d]+))', s)]


def i_maxnum_2_str(i, maxnum):
    return ('{:%dd}' % (maxnum_2_strlen(maxnum),)).format(i)


def g_maxnum_2_str(f, maxnum, fillchar=''):
    return align_decimal(f, maxnum_2_strlen(maxnum), right_pad=0,
                         fillchar=fillchar)


def ints_to_nicefrac(a, b):
    from fractions import Fraction
    return str(Fraction(a, b))


def float_to_nicefrac(f):
    from fractions import Fraction
    return str(Fraction(f).limit_denominator())


def icalculate(s):
    result = simple_eval(s)
    if type(result) != int:
        raise ValueError('Result of "%s" is not of type integer' % s)
    return result


def fcalculate(s):
    result = simple_eval(s)
    if type(result) not in [int, float]:
        raise ValueError('Result of "%s" is neither integer nor float' % s)
    return result


def maxnum_2_strlen(i: int) -> int:
    # i==0 => log10(1+0) exists
    # i==10 => log10(10)==1 => log10(1+10)>1
    return cast(int, math.ceil(math.log10(1 + i)))


def str2bool(s):
    from lacro.collections import dict_get
    return dict_get({'True': True, 'False': False},
                    s,
                    msg='Could not convert string to bool\n')


def _bin(x, width):
    """
    http://stackoverflow.com/questions/187273/base-2-binary-representation-using-python
    see ``also builtins.bin``
    """
    return ''.join(str((x >> i) & 1) for i in range(width - 1, -1, -1))
