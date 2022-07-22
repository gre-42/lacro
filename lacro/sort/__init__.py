# -*- coding: utf-8 -*-

from collections import OrderedDict

from lacro.collections import inverted_injective_dict


def lowercase_sorted(lst):
    """
    lowercase sort
    """
    return sorted(lst, key=lambda s: s.lower())


def sort_ids_like(dest, src):
    """
    src[sort_ids_like(dest, src)] == dest
    """
    d = inverted_injective_dict(dict(enumerate(src)))
    return [d[k] for k in dest]


def argsort(lst, reverse=False):
    """
    lst[argsort(lst)] == sorted(lst)
    """
    # http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    # by unutbu
    return sorted(range(len(lst)), key=lst.__getitem__, reverse=reverse)
    # return [i for i,s in sorted(enumerate(lst), key=lambda iv:iv[1])]


def dict_lensorted(dct):
    return OrderedDict(sorted(dct.items() if hasattr(dct, 'items') else
                              dct, key=lambda kv: len(kv[0]),
                              reverse=True))


def ranking(lst):
    return argsort(argsort(lst))
