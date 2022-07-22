# -*- coding: utf-8 -*-
import inspect
import os.path


def script_dir(nparent=1):
    [frame,
     filename,
     line_number,
     function_name,
     lines,
     index] = inspect.getouterframes(inspect.currentframe())[nparent]
    return os.path.dirname(os.path.realpath(filename))


def line_number(nparent=1):
    [frame,
     filename,
     line_number,
     function_name,
     lines,
     index] = inspect.getouterframes(inspect.currentframe())[nparent]
    return line_number


def cwd_sl():
    from subprocess import check_output
    return check_output(['bash', '-c', 'pwd']).decode('utf-8').rstrip('\n')


def abspath_of_script_child(*filenames, nparent=1):
    return os.path.normpath(os.path.join(script_dir(nparent + 1), *filenames))


def abspath_sl(basename):
    return os.path.normpath(os.path.join(cwd_sl(), basename))


def safe_normpath(path, parent_dir=None, recursion=0):
    if parent_dir is None:
        parent_dir, path = os.path.splitdrive(path)

    def joi(p): return (
        p if parent_dir is None else
        (('.' if parent_dir == '' else parent_dir)
         if p == '.' else
         os.path.join('' if parent_dir == '.' else parent_dir, p)))

    if path.endswith(os.sep):
        return safe_normpath(path[:-1], recursion=recursion + 1)
    #print(path, '|', os.path.dirname(path), '|', os.path.basename(path))
    assert recursion < 60
    if path in ['.', '']:
        res = '.'
    else:
        ltail = path.split(os.sep)
        if ltail[0] == '':  # absolute path
            ltail[0] = os.sep
            # return safe_normpath(os.sep.join(ltail[1:]),
            # os.path.join(parent_dir,''), recursion=recursion+1)
        if not os.path.lexists(joi(ltail[0])):
            raise ValueError('Path "%s" does not exist' % joi(ltail[0]))
        assert len(ltail) > 0
        if len(ltail) == 1:
            if ltail[0] == '..':
                raise ValueError(
                    'Cannot go above parent directory "%s"' % joi('.'))
            to_par = False
        else:
            to_par = (ltail[1] == '..')
        # print(to_par)
        if to_par and os.path.islink(joi(ltail[0])):
            raise ValueError('Cannot normalize path "%s", because it accesses '
                             'the parent directory of symbolic link "%s"' %
                             (joi(path), joi(ltail[0])))
        if to_par:
            res = safe_normpath(os.sep.join(
                ltail[2:]), parent_dir, recursion=recursion + 1)
        else:
            res = safe_normpath(os.sep.join(ltail[1:]), joi(
                ltail[0]), recursion=recursion + 1)
    if res != os.path.normpath(path):
        # ipy().embed()
        pass
        #raise ValueError('parent: "%s", path: "%s", safe_normpath: "%s", normpath: "%s"' % (parent_dir, path, res, os.path.normpath(path)))
    return joi(res)


def readlinkabs(l, skip_nonlinks=False):
    """
    Return an absolute path for the destination 
    of a symlink
    """
    if not skip_nonlinks or os.path.islink(l):
        assert os.path.islink(l)
        p = os.readlink(l)
        if os.path.isabs(p):
            return p
        return os.path.join(os.path.dirname(l), p)
    else:
        return l
