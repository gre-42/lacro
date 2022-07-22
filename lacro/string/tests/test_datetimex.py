#!/usr/bin/env pytest
# -*- coding: utf-8 -*-

import pytest

from lacro.inspext.app import init_pytest_suite
from lacro.string.datetimex import (date_format, datetime_format,
                                     datetime_from_string)

init_pytest_suite()


def test_16():
    assert datetime_from_string('24.05.2010') \
                     .strftime(date_format) == '2010-05-24'
    assert datetime_from_string('05/24/2010') \
        .strftime(date_format) == '2010-05-24'
    assert datetime_from_string('2010-05-24') \
        .strftime(date_format) == '2010-05-24'
    with pytest.raises(ValueError):
        datetime_from_string('2010-05.24')
    assert datetime_from_string('2012-07-10 15:06:27.0') \
        .strftime(datetime_format) == '2012-07-10 15:06:27'
