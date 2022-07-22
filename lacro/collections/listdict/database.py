# -*- coding: utf-8 -*-
from lacro.assertions import lists_assert_equal
from lacro.collections import Dict2Object
from lacro.io.string import save_string_to_file
from lacro.path.pathrel import rdirname
from lacro.string.misc import escape_latex, rstripstr

from ._collections import multiple as ld


def concat(dbs, **kwargs):
    assert len(dbs) > 0

    keys = dbs[0].keys()
    lists_assert_equal((db.keys() for db in dbs), check_order=False)
    return Dict2Object({k: ld.concat([getattr(db, k) for db in dbs],
                                     **kwargs) for k in keys})


def to_latex(dbs, longtable=False, utf8=True, amsmath=True, graphicx=False,
             margin='', prelude='', fontsize='', clearpage=False, **kwargs):
    inc = '\n'.join(
        [r'\usepackage[utf8]{inputenc}'] * utf8 +
        [r'\usepackage{amsmath}'] * amsmath +
        [r'\usepackage{longtable}'] * longtable +
        [r'\usepackage{graphicx}'] * graphicx +
        [r'\usepackage[margin=%s]{geometry}' % margin] * (margin != '') +
        [prelude] * (prelude != ''))

    fs = '\n'.join(
        [fontsize] * (fontsize != ''))

    return r'''\documentclass[a4paper]{article}

%s

\begin{document}
%s
%s

\end{document}
''' % (inc, fs, ('\n\n\\clearpage\n\n' if clearpage else '\n\n').join(
        '\\section*{%s}\n%s' % (
            escape_latex(n),
            lst.to_latex(table_class={True: 'longtable',
                                      False: 'tabular'}[longtable], **kwargs))
        for n, lst in dbs.items()))


def save_latex(filename, dbs, **kwargs):
    save_string_to_file(filename, to_latex(dbs, **kwargs))


def save_pdf(filename, dbs, verbose=False, keep_tmp=False, latexmk=False,
             skip_pdf=False, keep_latex=False, **kwargs):
    from os import unlink
    from os.path import basename, exists
    from subprocess import CalledProcessError

    from lacro.run_in.system import noninteractive
    base = rstripstr(filename, '.pdf', must_exist=True)
    save_latex(base + '.tex', dbs, **kwargs)
    if not skip_pdf:
        # jobname did not work for absolute pathes, using cwd instead
        if exists(filename):  # make sure latexmk is run
            unlink(filename)
        cmd = (['latexmk', basename(base), '-pdf'] if latexmk else
               ['pdflatex', basename(base)])
        try:
            noninteractive(cmd, return_stdout=not verbose,
                           stderr_to_stdout=True, wdir=rdirname(filename))
        except CalledProcessError as e:
            raise ValueError('Could not save database as pdf\n%s\n%s' %
                             (e, e.output))
        # noninteractive(['pdflatex', '-jobname=%s'%base, base],
        #               return_stdout=not verbose)
        for f in ((['.aux', '.log', '.fls', '.fdb_latexmk'] if latexmk else
                   ['.aux', '.log']) * (not keep_tmp) +
                  ['.tex'] * (not keep_latex)):
            unlink(base + f)


def save_xls(filename, dbs):
    """
    http://stackoverflow.com/questions/14225676/save-list-of-dataframes-to-multisheet-excel-spreadsheet
    sheet names not working in open office calc
    """
    from pandas import ExcelWriter

    writer = ExcelWriter(filename)
    for n, df in dbs.items():
        df.pandas().to_excel(writer, n, index=False)
    writer.save()
