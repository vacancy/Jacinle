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
import inspect
import contextlib
import functools

from .meta import decorator_with_optional_args
from .naming import class_name_of_method

__all__ = [
    'defaults_manager', 'wrap_custom_as_default', 'gen_get_default', 'gen_set_default',
    'option_context',
    'ARGDEF', 'default_args'
]


class DefaultsManager(object):
    def __init__(self):
        """
        Initialize the thread.

        Args:
            self: (todo): write your description
        """
        self._is_local = dict()

        self._defaults_global = dict()
        self._defaults_local = threading.local()

    @decorator_with_optional_args(is_method=True)
    def wrap_custom_as_default(self, *, is_local=False):
        """
        Creates decorator which wraps a custom method.

        Args:
            self: (todo): write your description
            is_local: (bool): write your description
        """
        def wrapper(meth):
            """
            Creates a context manager decorator.

            Args:
                meth: (str): write your description
            """
            identifier = class_name_of_method(meth)
            meth = contextlib.contextmanager(meth)
            self._is_local[identifier] = is_local
            defaults = self._get_defaults_registry(identifier)

            @contextlib.contextmanager
            @functools.wraps(meth)
            def wrapped_func(slf, *args, **kwargs):
                """
                Wraps a function that wraps a function call.

                Args:
                    slf: (todo): write your description
                """
                backup = defaults.get(identifier, None)
                defaults[identifier] = slf
                with meth(slf, *args, **kwargs):
                    yield
                defaults[identifier] = backup

            return wrapped_func
        return wrapper

    def gen_get_default(self, cls, default_getter=None):
        """
        Return the default getter. getter.

        Args:
            self: (todo): write your description
            cls: (todo): write your description
            default_getter: (str): write your description
        """
        identifier = class_name_of_method(cls.as_default)

        def get_default(default=None):
            """
            Return the default value for the given default.

            Args:
                default: (todo): write your description
            """
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
        """
        Generate a default set of class_name.

        Args:
            self: (todo): write your description
            cls: (todo): write your description
        """
        identifier = class_name_of_method(cls.as_default)

        def set_default(default):
            """
            Sets the default identifier.

            Args:
                default: (dict): write your description
            """
            self._get_defaults_registry(identifier)[identifier] = default
        return set_default

    def set_default(self, cls, default):
        """
        Set the default value of the default.

        Args:
            self: (todo): write your description
            cls: (todo): write your description
            default: (dict): write your description
        """
        identifier = class_name_of_method(cls.as_default)
        self._get_defaults_registry(identifier)[identifier] = default

    def _get_defaults_registry(self, identifier):
        """
        Return the default value for the given identifier.

        Args:
            self: (todo): write your description
            identifier: (str): write your description
        """
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


class _LocalObjectSimulator(object):
    __slots__ = ['ctx']


def option_context(name, is_local=True, **kwargs):
    """
    Creates a context that will set the context.

    Args:
        name: (str): write your description
        is_local: (bool): write your description
    """
    class OptionContext(object):
        def __init__(self, **init_kwargs):
            """
            Initialize this class.

            Args:
                self: (todo): write your description
                init_kwargs: (dict): write your description
            """
            for k, v in kwargs.items():
                setattr(self, k, v)
            if hasattr(self.__class__, 'current_context') and self.__class__.current_context.ctx is not None:
                c = self.__class__.get_default()
                for k in kwargs:
                    setattr(self, k, getattr(c, k))
            for k, v in init_kwargs.items():
                assert k in kwargs
                setattr(self, k, v)

        @classmethod
        def get_option(cls, name):
            """
            Get the value of an option.

            Args:
                cls: (todo): write your description
                name: (str): write your description
            """
            getattr(cls.get_default(), name)

        @classmethod
        def set_default_option(cls, name, value):
            """
            Sets the default value of an option.

            Args:
                cls: (todo): write your description
                name: (str): write your description
                value: (todo): write your description
            """
            cls._create_default_context()
            setattr(cls.default_context.ctx, name, value)

        @classmethod
        def get_default(cls):
            """
            Return the default context.

            Args:
                cls: (todo): write your description
            """
            cls._create_current_context()
            if cls.current_context.ctx is not None:
                return cls.current_context.ctx
            else:
                cls._create_default_context()
                return cls.default_context.ctx

        @contextlib.contextmanager
        def as_default(self):
            """
            Creates the current context.

            Args:
                self: (todo): write your description
            """
            self.__class__._create_current_context()
            backup = self.__class__.current_context.ctx
            self.__class__.current_context.ctx = self
            yield
            self.__class__.current_context.ctx = backup

        @classmethod
        def _create_default_context(cls):
            """
            Create a default context.

            Args:
                cls: (callable): write your description
            """
            if hasattr(cls, 'default_context'):
                return

            if is_local:
                cls.default_context = threading.local()
            else:
                cls.default_context = _LocalObjectSimulator()
            cls.default_context.ctx = cls(**kwargs)

        @classmethod
        def _create_current_context(cls):
            """
            Create the current context.

            Args:
                cls: (callable): write your description
            """
            if hasattr(cls, 'current_context'):
                return

            if is_local:
                cls.current_context = threading.local()
            else:
                cls.current_context = _LocalObjectSimulator()
            cls.current_context.ctx = None

    OptionContext.__name__ = name

    return OptionContext


ARGDEF = object()


def default_args(func):
    """
    Decorator that accepts a function arguments.

    Args:
        func: (callable): write your description
    """
    def wrapper(func):
        """
        Decorator for functions that require a function is_default signature.

        Args:
            func: (callable): write your description
        """
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            """
            Decorator toilizing a function with kwargs.

            Args:
            """
            bounded = sig.bind(*args, **kwargs)
            bounded.apply_defaults()

            for k, v in bounded.arguments.items():
                if v is ARGDEF:
                    if k in sig.parameters:
                        default_value = sig.parameters[k].default
                        bounded.arguments[k] = default_value

            return func(*bounded.args, **bounded.kwargs)

        return wrapped
    return wrapper(func)

