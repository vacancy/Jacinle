#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : env.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/10/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import os
import sys
from jacinle.cli.keyboard import str2bool
from .cache import cached_result

__all__ = ['jac_getenv', 'jac_is_verbose', 'jac_is_debug']


def jac_getenv(name, default=None, type=None, prefix=None):
    if prefix is None:
        prefix = 'JAC_'

    value = os.getenv((prefix + name).upper(), default)

    if value is None:
        return None

    if type is None:
        return value
    elif type == 'bool':
        return str2bool(value)
    else:
        return type(value)


@cached_result
def jac_get_dashdebug_arg():
    # Return True if there is a '-debug' or '--debug' arg in the argv.
    for value in sys.argv:
        if value in ('-debug', '--debug'):
            return True
    return False


@cached_result
def jac_is_verbose(default='n', prefix=None):
    return jac_getenv('verbose', default, type='bool', prefix=prefix)


@cached_result
def jac_is_debug(default='n', prefix=None):
    return jac_get_dashdebug_arg() or jac_getenv('debug', default, type='bool', prefix=prefix)

