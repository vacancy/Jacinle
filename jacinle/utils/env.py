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
from typing import Any, Optional, Union
from jacinle.cli.keyboard import str2bool
from .cache import cached_result

__all__ = ['jac_getenv', 'jac_is_verbose', 'jac_is_debug']


def jac_getenv(name: str, default: Any = None, type: Optional[Union[str, type]] = None, prefix: str = None) -> Any:
    """Get the environment variable with the given name.

    Args:
        name: the name of the environment variable.
        default: the default value if the environment variable is not set.
        type: the type of the environment variable. If not given, the type of the default value will be used.
            It supports a special type, denoted by a string ``'bool'``, indicating that the value
            will be converted to a boolean value ('true'/'false', 'yes'/'no', '1'/'0').
        prefix: the prefix of the environment variable. If not given, the prefix will be ``JAC_``.

    Returns:
        the value of the environment variable.
    """
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
    """Return if the ``--debug`` argument is given in the command line."""
    # Return True if there is a '-debug' or '--debug' arg in the argv.
    for value in sys.argv:
        if value in ('-debug', '--debug'):
            return True
    return False


@cached_result
def jac_is_verbose(default='n', prefix=None):
    """Return if the verbose mode is enabled. This is controlled by the environment variable ``JAC_VERBOSE``."""
    return jac_getenv('verbose', default, type='bool', prefix=prefix)


@cached_result
def jac_is_debug(default='n', prefix=None):
    """Return if the debug mode is enabled. This is controlled by the environment variable ``JAC_DEBUG`` or the ``--debug`` argument in the command line."""
    return jac_get_dashdebug_arg() or jac_getenv('debug', default, type='bool', prefix=prefix)

