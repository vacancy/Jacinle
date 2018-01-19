# -*- coding: utf-8 -*-
# File   : defaults.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/28/17
#
# This file is part of Jacinle.

import contextlib
import functools

from .naming import class_name_of_method

__all__ = ['defaults_manager']


class DefaultsManager(object):
    def __init__(self):
        self._defaults = {}

    def wrap_custom_as_default(self, custom_method):
        identifier = class_name_of_method(custom_method)

        custom_method = contextlib.contextmanager(custom_method)

        @contextlib.contextmanager
        @functools.wraps(custom_method)
        def wrapped_func(slf, *args, **kwargs):
            backup = self._defaults.get(identifier, None)
            self._defaults[identifier] = slf
            with custom_method(slf, *args, **kwargs):
                yield
            self._defaults[identifier] = backup

        return wrapped_func

    def gen_get_default(self, cls):
        identifier = class_name_of_method(cls.as_default)

        def get_default(default=None):
            return self._defaults.get(identifier, default)
        return get_default

    def set_default(self, cls, default):
        identifier = class_name_of_method(cls.as_default)
        self._defaults[identifier] = default


defaults_manager = DefaultsManager()
