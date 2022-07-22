# -*- coding: utf-8 -*-
import glob
import itertools
import os.path
import re

# --------
# - find -
# --------


def file_chain(func):
    def _func(files, *args, chain=False, **kwargs):
        if chain:
            return itertools.chain.from_iterable(func(f, *args, **kwargs)
                                                 if os.path.isdir(f) else [f]
                                                 for f in files)
        else:
            return func(files, *args, **kwargs)
    return _func


def subdirs_and_files(parent_dir, regex='.*', return_abs_path=False):
    try:
        # http://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory
        # http://stackoverflow.com/questions/229186/os-walk-without-digging-into-directories-below
        root, dirs, files = next(os.walk(parent_dir))
        df = dirs + files
        df = [d for d in df if re.match(regex, d)]
        if return_abs_path:
            df = [os.path.join(parent_dir, d) for d in df]
        return df
    except StopIteration:
        raise IOError('Could not list subdirs and files of "%s"' % parent_dir)


def subdirs(parent_dir, regex='.*', return_abs_path=False):
    try:
        # http://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory
        # http://stackoverflow.com/questions/229186/os-walk-without-digging-into-directories-below
        root, dirs, files = next(os.walk(parent_dir))
        dirs = [d for d in dirs if re.match(regex, d)]
        if return_abs_path:
            dirs = [os.path.join(parent_dir, d) for d in dirs]
        return dirs
    except StopIteration:
        raise IOError('Could not list subdirs of "%s"' % parent_dir)


@file_chain
def files(parent_dir, regex='.*', return_abs_path=False, **kwargs):
    r"""
    example: files('.', regex='.*\.py$')
    """
    try:
        # http://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory
        # http://stackoverflow.com/questions/229186/os-walk-without-digging-into-directories-below
        root, dirs, files = next(os.walk(parent_dir))
        files = [f for f in files if re.match(regex, f, **kwargs)]
        if return_abs_path:
            files = [os.path.join(parent_dir, f) for f in files]
        return files
    except StopIteration:
        raise IOError('Could not list files of "%s"' % parent_dir)
    # if extension == '':
    #     extension = '*'
    # return glob.glob(os.path.join(config.source_path, extension))

    # filter(lambda x: x.endswith('m'), os.listdir('.'))
    # os.path.expandvars
    # source_files = glob.glob(config.source_path + '*.nii.gz')


@file_chain
def all_files(parent_dir, regex='.*'):
    return (os.path.join(root, f)
            for root, dirnames, filenames in os.walk(parent_dir)
            for f in filenames if re.match(regex, f))


def glob_ids(filename, nids=-1):
    res = sorted(glob.glob(filename))
    # noinspection PyChainedComparisons
    if (nids > 0 and len(res) != nids) or (nids < 0 and len(res) < -nids):
        raise ValueError('Could not find%s %d element(s), found %d: "%s"%s' %
                         (' at least' if (nids < 0) else '', abs(nids),
                          len(res), filename,
                          '\n%s' % '\n'.join('  %s' % r for r in res)
                          if len(res) > 0 else ''))
    return res


def glob_id(filename, id=0, nids=1):
    """
    returns the "id's" filename of at least "-nids" or exactly "nids" files
    "id" can also be negative to count backwards
    """
    assert id < abs(nids) if id > 0 else abs(id) <= abs(nids)
    return glob_ids(filename, nids)[id]


def resolve_base_glob(dir, gbase, id=0, nids=1):
    return os.path.basename(glob_id(os.path.join(dir, gbase), id, nids))
