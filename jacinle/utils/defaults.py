#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : defaults.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/28/2017
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import threading
import contextlib
import functools

from .meta import decorator_with_optional_args
from .naming import class_name_of_method

__all__ = ['defaults_manager', 'wrap_custom_as_default', 'gen_get_default', 'gen_set_default']


class DefaultsManager(object):
    def __init__(self):
        self._is_local = dict()

        self._defaults_global = dict()
        self._defaults_local = threading.local()

    @decorator_with_optional_args(is_method=True)
    def wrap_custom_as_default(self, *, is_local=False):
        def wrapper(meth):
            identifier = class_name_of_method(meth)
            meth = contextlib.contextmanager(meth)
            self._is_local[identifier] = is_local
            defaults = self._get_defaults_registry(identifier)

            @contextlib.contextmanager
            @functools.wraps(meth)
            def wrapped_func(slf, *args, **kwargs):
                backup = defaults.get(identifier, None)
                defaults[identifier] = slf
                with meth(slf, *args, **kwargs):
                    yield
                defaults[identifier] = backup

            return wrapped_func
        return wrapper

    def gen_get_default(self, cls, default_getter=None):
        identifier = class_name_of_method(cls.as_default)

        def get_default(default=None):
            if default is None and default_getter is not None:
                default = default_getter()

            # NB(Jiayuan Mao): cannot use .get(identifier, default), because after calling as_default, the current
            #     default will be set to None.
            val = self._get_defaults_registry(identifier).get(identifier, None)
            if val is None:
                val = default
            return val
        return get_default

    def gen_set_default(self, cls):
        identifier = class_name_of_method(cls.as_default)

        def set_default(default):
            self._get_defaults_registry(identifier)[identifier] = default
        return set_default

    def set_default(self, cls, default):
        identifier = class_name_of_method(cls.as_default)
        self._get_defaults_registry(identifier)[identifier] = default

    def _get_defaults_registry(self, identifier):
        is_local = self._is_local.get(identifier, False)
        if is_local:
            if not hasattr(self._defaults_local, 'defaults'):
                self._defaults_local.defaults = dict()
            defaults = self._defaults_local.defaults
        else:
            defaults = self._defaults_global
        return defaults


defaults_manager = DefaultsManager()
wrap_custom_as_default = defaults_manager.wrap_custom_as_default
gen_get_default = defaults_manager.gen_get_default
gen_set_default = defaults_manager.gen_set_default
