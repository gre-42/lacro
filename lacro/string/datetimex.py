# -*- coding: utf-8 -*-
import re
from datetime import datetime, timedelta

from lacro.collections import dict_minus_val

date_format = '%Y-%m-%d'
datetime_format = '%Y-%m-%d %H:%M:%S'


def datetime_from_string(s):
    def gi(g, i): return int(g.group(i + 1))

    try:
        g = re.match(r'^(\d\d)/(\d\d)/(\d\d\d\d)$', s)
        if g is not None:
            return datetime(year=gi(g, 2),
                            month=gi(g, 0),
                            day=gi(g, 1))
        g = re.match(
            r'^(\d\d\d\d)-(\d\d)-(\d\d)( (\d\d):(\d\d):(\d\d)(.(\d))?)?$', s)
        if g is not None:
            other = dict_minus_val(dict(
                hour=g.group(5),
                minute=g.group(6),
                second=g.group(7),
                # subsec = g.group(9),
            ))
            return datetime(year=gi(g, 0),
                            month=gi(g, 1),
                            day=gi(g, 2),
                            **{k: int(v) for k, v in other.items()})
        g = re.match(r'^(\d\d).(\d\d).(\d\d\d\d)$', s)
        if g is not None:
            return datetime(year=gi(g, 2),
                            month=gi(g, 1),
                            day=gi(g, 0))
        raise ValueError('Unknown format')
    except ValueError as e:
        raise ValueError('Could not parse date string "%s"\n%s' % (s, e))


def age(birthdate, now):
    """
    http://stackoverflow.com/a/4828842/2292832
    """
    return (now - birthdate) / timedelta(days=365.2425)
