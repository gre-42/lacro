# -*- coding: utf-8 -*-
import os.path

from lacro.dsync.named_lock import locked_function
from lacro.io.string import print_err
from lacro.path.shlext import dquote
from lacro.verbosity import verbosity


def versioned_path_uptodate(filename, version, function_name=None):
    isup = os.path.exists('%s_done%s' % (filename, version))
    if verbosity.value >= 1:
        print_err(('' if function_name is None else function_name + ' ') +
                  {True: 'up to date: ', False: 'outdated: '}[isup] + filename)
    return isup


def versioned_path(operation, pathname, version='', function_name='file',
                   rm_old=True, mkdir=True, assert_out_of_date=False,
                   alternative_operation=lambda: None, ndelete_retries=0,
                   is_file=False, force_update=False, locked=True):
    if is_file:
        return versioned_file(operation, pathname, version, function_name,
                              rm_old, mkdir, assert_out_of_date,
                              alternative_operation, force_update, locked)
    else:
        return versioned_directory(operation, pathname, version, function_name,
                                   rm_old, mkdir, ndelete_retries,
                                   assert_out_of_date, alternative_operation,
                                   force_update, locked)


def versioned_file(operation, filename, version='', function_name='file',
                   rm_old=True, mkdir=True, assert_out_of_date=False,
                   alternative_operation=lambda: None, force_update=False,
                   locked=True):
    def operation1():
        if (not force_update) and versioned_path_uptodate(filename, version,
                                                          function_name):
            if assert_out_of_date:
                raise ValueError('"%s" up to date, but asserted out of date' %
                                 filename)
            return alternative_operation()
        else:
            from lacro.run_in import bash
            dirname = os.path.dirname(os.path.abspath(filename))
            bash.run([
                f'rm -f {dquote(filename)}_done*',
                (f'rm -f {dquote(filename)}' if rm_old is True
                 else
                 f'rm -rf {dquote(dirname)}' if rm_old == 'dir'
                 else ''),
                f'mkdir -p {dquote(dirname)}' if mkdir else ''])

            res = operation()
            bash.run(['touch %s_done%s' % (filename, version)])
            return res
    return locked_function(operation1, filename, locked)()


def versioned_directory(operation, dirname, version='', function_name='dir',
                        rm_old=True, mkdir=True, ndelete_retries=0,
                        assert_out_of_date=False,
                        alternative_operation=lambda: None, force_update=False,
                        locked=True):
    def operation1():
        if (not force_update) and versioned_path_uptodate(dirname, version,
                                                          function_name):
            if assert_out_of_date:
                raise ValueError(
                    '"%s" up to date, but asserted out of date' % dirname)
            return alternative_operation()
        else:
            from lacro.run_in import bash
            bash.run([
                f'rm -f {dquote(dirname)}_done*',
                f'rm -rf {dquote(dirname)}' if rm_old else '',
                f'mkdir -p {dquote(dirname)}' if mkdir else ''],
                nretries=ndelete_retries)

            res = operation()
            bash.run(['touch %s_done%s' % (dirname, version)])
            return res
    return locked_function(operation1, dirname, locked)()


def cached_repr_io(operation, pathname, version, function_name='pickle',
                   rm_old=True, mkdir=False, alternative_operation=lambda x: x,
                   force_update=False, is_file=True):
    from lacro.io import repr_io
    filename = pathname if is_file else os.path.join(pathname, 'pickle.py')

    def pickle_operation():
        res = operation()
        repr_io.save(filename, res)
        # with retry_open(filename, 'wb') as f:
        # pickle.dump(res, f, -1)
        return res

    def pickle_alternative_operation():
        return alternative_operation(repr_io.load_blob(filename))
        # with retry_open(filename, 'rb') as f:
        # return alternative_operation(pickle.load(f))

    return versioned_path(
        pickle_operation, pathname, version, function_name,
        alternative_operation=pickle_alternative_operation, rm_old=rm_old,
        mkdir=mkdir, force_update=force_update, is_file=is_file)
