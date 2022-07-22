# -*- coding: utf-8 -*-
import errno
import os.path
import re

# -------------
# - construct -
# -------------


def escape_filename(fn):
    return re.sub('_+', '_', re.sub(r'[\\:\'"äöüß ,/()\]\[{\}|]', '_', fn))


def path_join_base(path, subdir):
    assert_basename(subdir)
    return os.path.join(path, subdir)


def assert_basename(path):
    if os.path.basename(path) != path:
        raise ValueError('"%s" is not a basename' % path)


def mkdir_p(path, flags='p'):
    if flags == '':
        os.mkdir(path)
    elif flags == 'p':
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST:
                pass
            else:
                raise
    else:
        raise ValueError('Unknown flags: %s' % flags)


# def add_filename_suffix(file, suffix):
#     fe = file.split('.', maxsplit=1)
#     if len(fe) == 1:
#         return fe[0] + suffix
#     else
#         return fe[0] + suffix + '.' + fe[1]


def file_extension(f):
    return os.path.splitext(f)[1][1:]


def path_wo_extension(f):
    return os.path.splitext(f)[0]


def remove_files(files, force=False):
    for f in files:
        try:
            # same as os.unlink
            os.remove(f)
        except FileNotFoundError:
            if force:
                pass
            else:
                raise


def remove_dirs(dirs):
    for d in dirs:
        os.rmdir(d)


def removes(files, dirs):
    remove_files(files)
    remove_dirs(dirs)


def remove_recursive(pathes):
    """
    like shutil.rmtree, but also works for files and symbolic links
    """
    for path in pathes:
        if os.path.islink(path) or os.path.isfile(path):
            os.remove(path)
        else:
            import shutil
            shutil.rmtree(path)


def included_parent_dirs_sorted_relative(pathes):
    res = []
    for p in pathes:
        assert p.find('\\') == -1
        assert not p.startswith('/')
        while len(p) != 0:
            res.append(p)
            # dirname('/')=='/', but this is not allowed anyway
            p = os.path.dirname(p)
            # p = p[:max(0,p.rfind('/'))]
    return sorted(list(set(res)))


def included_parent_dirs(pathes):
    res = []
    for p in pathes:
        while os.path.dirname(p) != p:
            res.append(p)
            p = os.path.dirname(p)
    return res
