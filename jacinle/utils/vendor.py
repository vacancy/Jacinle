#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : vendor.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/25/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import importlib
import functools
from jacinle.logging import get_logger

logger = get_logger(__file__)

__all__ = ['has_vendor', 'requires_vendors']


def has_vendor(vendor):
    """
    Determine if a module has a python module.

    Args:
        vendor: (str): write your description
    """
    try:
        importlib.import_module(vendor)
    except ImportError as e:
        return False
    return True


def requires_vendors(*vendors):
    """
    Decorator to check if any vendors are available.

    Args:
        vendors: (str): write your description
    """
    def wrapper(func):
        """
        Decorator to import a function to a class.

        Args:
            func: (callable): write your description
        """
        checked_vendors = False

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            """
            Decorator to use a local module.

            Args:
            """
            nonlocal checked_vendors
            if not checked_vendors:
                for v in vendors:
                    try:
                        importlib.import_module(v)
                    except ImportError as e:
                        raise ImportError('Cannot import {}. Make sure you have it as a vendor for Jacinle.'.format(v)) from e
                checked_vendors = True

            return func(*args, **kwargs)

        return wrapped
    return wrapper

