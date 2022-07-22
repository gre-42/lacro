# -*- coding: utf-8 -*-
import os.path

from .pathabs import abspath_sl


def child_relpath(child, parent, assure_is_child=True):
    if not child.startswith(parent):
        if assure_is_child:
            raise ValueError(
                'Path "%s" is not a subdirectory of "%s"' % (child, parent))
        return child
    else:
        return os.path.relpath(child, parent)


def cwd_relpath(child, parent):
    if os.path.isabs(child):
        return child
    elif os.path.isabs(parent):
        return abspath_sl(child)
    else:
        return os.path.relpath(child, parent)


def rdirname(filename):
    res = os.path.dirname(filename)
    return '.' if res == '' else res
