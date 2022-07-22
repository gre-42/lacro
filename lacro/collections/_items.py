# -*- coding: utf-8 -*-
from lacro.assertions import list_assert_no_duplicates


def items_unique(items):
    res = list(items)
    list_assert_no_duplicates([k for k, v in res])
    return res


def _newline_str(s, width, char, alen):
    from lacro.string.misc import indentedl
    res = str(s).split('\n')
    if alen > width:
        return '\n' + indentedl(res, width, char)
    else:
        return indentedl(res, lambda i: width - alen if i == 0 else width,
                         char)


def items_2_str(items, isstr=False, width=4, colon=True, keys=None, **kwargs):
    def vs(v):
        from lacro.string.misc import to_str
        return v if isstr else to_str(v, width=width, colon=colon, **kwargs)
    ks = str
    sep = ': ' if colon else ' '
    if keys is None:
        items = tuple(items)
        keys = [k for k, v in items]
    if width == 'auto':
        items = list(items)
        if len(items) == 0:
            width1 = None
        else:
            width1 = max(len(ks(k)) + len(sep) for k in keys)
    else:
        width1 = width
    return '\n'.join('%s%s%s' % (ks(k), sep, _newline_str(vs(v), width1, ' ',
                                                          len(ks(k)) +
                                                          len(sep)))
                     for k, v in items)
