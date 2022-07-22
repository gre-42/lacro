# -*- coding: utf-8 -*-
from typing import Sequence

from lacro.io.string import print_err

from . import naming_scheme as ns


class SiteUrl:

    def __init__(self, uri_path, l, lpath: Sequence['ns.NamingScheme']):
        self._uri_path = uri_path
        self._l = l
        self._lpath = lpath

    def _get_uri_path(self, max_depth):
        if self._uri_path is None:
            lpath = self._lpath[:max_depth]
            path = [p.user_contents('filestep') for p in lpath]
            if any(path):
                if not all(path):
                    print_err('Inconsistent filesteps: %s\n%s' %
                              (path, [p._n for p in lpath]))
                    return 'error%d_%d' % (len(lpath) - 1, self._l['__iid__'])
                else:
                    return '-'.join(path)
            else:
                return 'index%d_%d' % (len(lpath) - 1, self._l['__iid__'])
        else:
            return self._uri_path

    @property
    def filename(self):
        uri_path = self._get_uri_path(self._l['__collapsed_depth__'])
        return uri_path + '.xhtml'

    def child_filename(self, child_name):
        uri_path = self._get_uri_path(None)
        return uri_path + '-' + child_name

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self._uri_path)
