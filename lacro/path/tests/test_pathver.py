#!/usr/bin/env pytest
# -*- coding: utf-8 -*-

import os.path
import re
from functools import partial

from lacro.inspext.app import init_pytest_suite
from lacro.io.string import StdoutReader
from lacro.path.pathmod import removes
from lacro.path.pathver import versioned_directory, versioned_file

init_pytest_suite()


def test_pathver(tmpdir):
    j = partial(os.path.join, str(tmpdir))
    with StdoutReader(stderr_to_stdout=True) as so:
        versioned_file(lambda: print('printing file'),
                       j('hello1_file'), 'A')
        versioned_directory(lambda: print('printing dir'),
                            j('hello1_dir'), 'A')

        versioned_file(lambda: print('printing file'),
                       j('hello1_file'), 'A', locked=False)
        versioned_directory(lambda: print('printing dir'),
                            j('hello1_dir'), 'A', locked=False)

        removes(files=[j('hello1_file_doneA'), j('hello1_dir_doneA')],
                dirs=[j('hello1_dir')])

    assert re.match(
        fr'''^file outdated: {tmpdir}/hello1_file

### Executing tmp Bash script \| filename .*\.sh \| contents follow ###
#	#!/bin/bash -eu
#	rm -f {tmpdir}/hello1_file_done\*
#	rm -f {tmpdir}/hello1_file
#	mkdir -p .*
printing file

### Executing tmp Bash script \| filename .*\.sh \| contents follow ###
#	#!/bin/bash -eu
#	touch {tmpdir}/hello1_file_doneA
dir outdated: {tmpdir}/hello1_dir

### Executing tmp Bash script \| filename .*\.sh \| contents follow ###
#	#!/bin/bash -eu
#	rm -f {tmpdir}/hello1_dir_done\*
#	rm -rf {tmpdir}/hello1_dir
#	mkdir -p {tmpdir}/hello1_dir
printing dir

### Executing tmp Bash script \| filename .*\.sh \| contents follow ###
#	#!/bin/bash -eu
#	touch {tmpdir}/hello1_dir_doneA
file up to date: {tmpdir}/hello1_file
dir up to date: {tmpdir}/hello1_dir
$''', so.string) is not None
