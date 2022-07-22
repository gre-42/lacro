# -*- coding: utf-8 -*-
import os.path
import re
from collections import OrderedDict
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from lacro.io.string import load_string_from_file, save_string_to_file
from lacro.parse.html import absolutified, get_base, get_encoding
from lacro.path.pathmod import escape_filename


def urlopen_string(url, absolutify=False, errors='strict',
                   default_encoding=None, unescape=False, **kwargs):
    resource = urlopen(url, **kwargs)
    if re.match('^text/(?:html(?:; charset=.*$)?|bibliography$)',
                resource.headers['content-type']):
        pass
    elif resource.headers['content-type'] == 'application/pdf':
        return 'content-type = application/pdf'
    else:
        raise ValueError('Unknown content type: "%s"' %
                         resource.headers['content-type'])
    contents = resource.read()
    encoding = resource.headers.get_content_charset()
    if encoding is None:
        encoding = get_encoding(contents, default=default_encoding)
    res = contents.decode(encoding, errors=errors)
    if absolutify:
        try:
            base = get_base(res)
        except ValueError:
            base = resource.geturl()
        res = absolutified(res, base)
    if unescape:
        from html import unescape as unescape_
        res = unescape_(res)
    return res


def urlopen_mozilla(url, **kwargs):
    return urlopen_string(Request(url, headers={'User-agent': 'Mozilla/5.0'}),
                          **kwargs)


def google_url(**kwargs):
    return ('https://www.google.de/search?%s' %
            urlencode(OrderedDict(sorted(kwargs.items()))))


def google_search(**kwargs):
    return urlopen_mozilla(google_url(**kwargs))


class CachedUrlopen:
    def __init__(self, dirname, verbose=False, unescape=False, **kwargs):
        self.dirname = dirname
        self.verbose = verbose
        self.kwargs = kwargs
        self.unescape = unescape

    def __call__(self, url):
        gfile = os.path.join(self.dirname, escape_filename(url) + '.html')
        if not os.path.exists(gfile):
            if self.verbose:
                print('not cached:', gfile)
                print('url:', url)
            save_string_to_file(gfile, urlopen_mozilla(url, **self.kwargs))
        else:
            if self.verbose:
                print('cached:', gfile)
        res = load_string_from_file(gfile)
        if self.unescape:
            from html import unescape
            res = unescape(res)
        return res
