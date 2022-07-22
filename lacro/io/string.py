# -*- coding: utf-8 -*-
import json
import os
import pickle
import re
import sys
import threading
from functools import partial
from typing import Any, Callable, Union

from lacro.inspext.misc import reraise
from lacro.io.retry_on_eintr import os_retry_fdopen, os_retry_open, retry_open

_print_tlock = threading.Lock()


def save_string_to_file(filename, st):
    with retry_open(filename, 'w', encoding='utf-8') as fil:
        fil.write(st)


def save_strings_to_file(filename, st):
    save_string_to_file(filename, '\n'.join(st))


def load_string_from_file(filename, default=None, **kwargs):
    try:
        with open_nul(filename, **kwargs) as fil:
            result = fil.read()
    except IOError:
        if default is not None:
            return default
        else:
            raise
    return result


def load_strings_from_file(filename, rm_trailing_newline=True, **kwargs):
    # this resulted in the \n character not being removed
    # result = [line for line in fil]
    if rm_trailing_newline:
        result = (load_string_from_file(filename, **kwargs)
                  .rstrip('\n')
                  .split('\n'))
    else:
        result = load_string_from_file(filename, **kwargs).split('\n')
    return result


def filtered_open(filename, rm_trailing_newline=False, skip_comments=False,
                  skip_empty=False, ascii_nul_replacement=None):
    from io import StringIO
    return StringIO('\n'.join(l for l in load_strings_from_file(
        filename,
        rm_trailing_newline=rm_trailing_newline,
        ascii_nul_replacement=ascii_nul_replacement)
        if ((not skip_empty or l != '') and
            (not skip_comments or not l.startswith('#')))))


def open_nul(filename, ascii_nul_replacement=None):
    if ascii_nul_replacement is not None:
        from io import StringIO
        with open_optional(filename, 'rb') as f:
            st = f.read()
        return StringIO(st.decode('utf-8').replace('\x00',
                                                   ascii_nul_replacement))
    else:
        return open_optional(filename, 'r', encoding='utf-8')


def open_chmod(filename, rw, permissions, optional=False, encoding=None):
    def opener(fn, mo, encoding):
        assert 'a' not in rw
        if permissions is None:
            return retry_open(filename, rw, encoding=encoding)
        else:
            return os_retry_fdopen(
                os_retry_open(filename,
                              os.O_WRONLY * ('r' not in rw) |
                              os.O_RDONLY * ('w' not in rw) |
                              os.O_RDWR * ('w' in rw and 'r' in rw) |
                              (os.O_CREAT | os.O_TRUNC) * ('+' in rw or
                                                           'w' in rw),
                              int(permissions, 8)), rw, encoding=encoding)
    return (open_optional(filename, rw, opener, encoding) if optional else
            opener(filename, rw, encoding))


def open_optional(filename, mode, opener=retry_open, encoding=None):
    if type(filename) == str:
        return opener(filename, mode, encoding=encoding)
    else:
        return filename


def open_rw(file, mode, encoding=None):
    """
    'r+' does not create nonexistent files
    'w+' erases all file contents and read() returns ''
    'a+' always appends, even if seek(0) is called
    Is not safe w.r.t. race conditions
    """
    if mode == 'rw+' and not os.path.exists(file):
        return retry_open(file, 'a+', encoding=encoding)
    else:
        return retry_open(file, 'r+', encoding=encoding)


def open_timestap_file(f):
    from datetime import datetime
    res = retry_open(f, 'w', encoding='utf-8')
    res.write(datetime.now().strftime('%Y-%m-%d %H:%M,%S') + '\n')
    return res


def mem_file_from_bytes(st):
    """
    Gets around "StringIO instance has no attribute 'fileno'" error
    """
    import tempfile
    f = tempfile.SpooledTemporaryFile()
    f.write(st)
    f.seek(0)
    return f


def print_file_contents(filename, message, lineStart='#\t', to_err=False):
    myp = sys.stderr if to_err else sys.stdout
    global _print_tlock
    with _print_tlock:
        with retry_open(filename, 'r', encoding='utf-8') as fil:
            myp.write('\n')
            myp.write('### %(message)s | '
                      'filename %(filename)s | '
                      'contents follow ###\n' % locals())
            myp.write('\n'.join(lineStart + line.strip('\n') for line in fil) +
                      '\n')
        myp.flush()


def print_err(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def nice_print(s, file=sys.stdout, **kwargs):
    from lacro.string.misc import to_str
    print(to_str(s, nice_dict=True, **kwargs), file=file)


def p(s, **kwargs):
    nice_print(s, **kwargs)


def pe(s, **kwargs):
    nice_print(s, file=sys.stderr, **kwargs)


def printed(x, to_str=str, enabled=True):
    if enabled:
        print(to_str(x))
    return x


def rprinted(x):
    print(repr(x))
    return x


def pshort(obj):
    from lacro.string.misc import trunc
    print(trunc(str(obj), 200))


class _MyPickler(pickle.Pickler):

    def persistent_id(self, obj):
        from inspect import isclass
        if isclass(obj):
            return (obj.__module__, obj.__name__)
        else:
            return None


class _MyUnpickler(pickle.Unpickler):

    def persistent_load(self, obj_id):
        module, func = obj_id
        from importlib import import_module
        return getattr(import_module(module), func)

    def find_class(self, module, name):
        if (module, name) == ('cat.containers.gen_types', 'construct_f'):
            module, name = ('lacro.collections.dictlike_tuples',
                            '_construct_f')
        return super().find_class(module, name)


class TransportPickle:

    def __init__(self):
        import tempfile
        self.file = tempfile.NamedTemporaryFile(mode='rb', suffix='.blob',
                                                delete=False)
        self.file.close()

    def load(self):
        if os.path.getsize(self.file.name) == 0:
            raise ValueError('Pickle file "%s" was never written' %
                             self.file.name)
        with retry_open(self.file.name, 'rb') as f:
            return pickle.load(f)

    def __str__(self):
        return self.file.name


def save_to_pickle(filename, obj):
    with retry_open(filename, 'wb') as f:
        p = _MyPickler(f, -1)
        p.dump(obj)


def load_from_pickle(filename):
    try:
        # maybe Unpickler does not support "__get__"
        with retry_open(filename, 'rb') as f:
            from io import BytesIO
            v = f.read()
            with BytesIO(v) as f1:
                u = _MyUnpickler(f1)
                return u.load()
    except Exception as e:
        reraise(ValueError('Could not load pickle file "%s"\n%s' % (filename, e)))


def load_from_json(filename: str) -> Any:
    with retry_open(filename, encoding='utf-8') as f:
        return json.load(f)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_to_json(filename: str, obj: object) -> None:
    with retry_open(filename, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=4, cls=NumpyEncoder)


def tprint(*args, **kwargs):
    # threading.activeCount()
    txt0 = '%d:' % ([t.ident for t in threading.enumerate()]
                    .index(threading.current_thread().ident), )
    global _print_tlock
    with _print_tlock:
        print(txt0, *args, **kwargs)


def ttprint(text):
    global _print_tlock
    with _print_tlock:
        print('%d: %s' % (threading.current_thread().ident, text))


def stop_on_print(txt, skip_matches=False):
    if hasattr(sys.stdout, 'stdout'):
        print('Skipping "stop_on_print" because it was already called')
        return
    print('halting on "%s"' % txt)

    class Printer(object):

        def __init__(self):
            self.stdout = sys.stdout

        def write(self, s):
            match_found = (txt in s)
            if not (skip_matches and match_found):
                self.stdout.write('%s' % s)
            # traceback.print_stack(file=self.stdout)
            if match_found:
                raise ValueError('Detected print statement containing "%s"' %
                                 txt)

        def flush(self):
            self.stdout.flush()

    sys.stdout = Printer()


class StdoutReader:
    def __init__(self, stderr_to_stdout=False):
        self._stderr_to_stdout = stderr_to_stdout

    def __enter__(self):
        import sys
        from io import StringIO
        self._stdout_orig = sys.stdout
        sys.stdout = StringIO()
        if self._stderr_to_stdout:
            self._stderr_orig = sys.stderr
            sys.stderr = sys.stdout
        return self

    def __exit__(self, type, value, traceback):
        self.string = sys.stdout.getvalue()
        sys.stdout = self._stdout_orig
        if self._stderr_to_stdout:
            sys.stderr = self._stderr_orig


def cached_file_loader(file_loader):
    if type(file_loader) == str:
        from lacro.decorators import cached_function

        @cached_function
        def func(f):
            path = file_loader
            if f.endswith('.csv'):
                import lacro.collections.listdict as ld
                return ld.ListDict.load_csv(os.path.join(path, f))
            else:
                return load_string_from_file(os.path.join(path, f))
        return func
    else:
        return file_loader


def load_file_if_uri(file_loader: Union[str, Callable[[str], str]]):
    file_loader = cached_file_loader(file_loader)

    class LoadFileIfUri:

        def __init__(self, function):
            self._function = function

        def __get__(self, obj, objtype):
            if obj is None:
                return self

            return LoadFileIfUri(partial(self._function, obj))

        def __call__(self, *args, **kwargs):
            res = self._function(*args, **kwargs)
            if isinstance(res, str):
                g = re.match('^file://(.*)$', res)
                if g is None:
                    return res
                else:
                    return file_loader(g.group(1))
            else:
                return res
    return LoadFileIfUri
