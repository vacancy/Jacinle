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
from .inspect import class_name_of_method

__all__ = [
    'defaults_manager', 'wrap_custom_as_default', 'gen_get_default', 'gen_set_default',
    'option_context', 'FileOptions',
    'ARGDEF', 'default_args'
]


class DefaultsManager(object):
    """Defaults manager can be used to create program or thread-level registries.
    One of the typical use case is that you can create an instance of a specific class, and then set it as the default,
    and then get this instance from elsewhere.

    For example::

        >>> class Storage(object):
        ...     def __init__(self, value):
        ...         self.value = value

        >>> storage = Storage(1)
        >>> set_defualt_storage(storage)
        >>> get_default_storage()  # now you can call this elsewhere.

    Another important feature supported by this default manager is that it allows you to have "nested" default registries.

    For example::

        >>> get_default_storage().value  # -> 1
        >>> with Stoage(2).as_default():
        ...     get_default_storage().value  # -> 2
        ...     with Storage(3).as_default():
        ...         get_default_storage().value  # -> 3
        ...     get_default_storage().value  # -> 2

    Similar features have been used commonly in TensorFlow, e.g., tf.Session, tf.Graph.

    To create a class with a default registry, use the following:

    .. code-block:: python

        class Storage(object):
            def __init__(self, value):
                self.value = value

            @defaults_manager.wrap_custom_as_default(is_local=True)
            def as_default(self):  # this is a contextmanager
                yield

        get_default_storage = defaults_manager.gen_get_default(Storage)
        set_default_storage = defaults_manager.gen_set_default(Storage)
    """

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


class _LocalObjectSimulator(object):
    __slots__ = ['ctx']


def option_context(name, is_local=True, **kwargs):
    class OptionContext(object):
        def __init__(self, **init_kwargs):
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
            """Get the option value of the current context."""
            getattr(cls.get_default(), name)

        @classmethod
        def set_default_option(cls, name, value):
            """Set the option value for the current context."""
            cls._create_default_context()
            setattr(cls.default_context.ctx, name, value)

        @classmethod
        def get_default(cls):
            """Get the current option context."""
            cls._create_current_context()
            if cls.current_context.ctx is not None:
                return cls.current_context.ctx
            else:
                cls._create_default_context()
                return cls.default_context.ctx

        @contextlib.contextmanager
        def as_default(self):
            """Make this option context the current context. It will overwrite the current option values."""
            self.__class__._create_current_context()
            backup = self.__class__.current_context.ctx
            self.__class__.current_context.ctx = self
            yield
            self.__class__.current_context.ctx = backup

        @classmethod
        def _create_default_context(cls):
            if hasattr(cls, 'default_context'):
                return

            if is_local:
                cls.default_context = threading.local()
            else:
                cls.default_context = _LocalObjectSimulator()
            cls.default_context.ctx = cls(**kwargs)

        @classmethod
        def _create_current_context(cls):
            if hasattr(cls, 'current_context'):
                return

            if is_local:
                cls.current_context = threading.local()
            else:
                cls.current_context = _LocalObjectSimulator()
            cls.current_context.ctx = None

    OptionContext.__name__ = name

    return OptionContext


class FileOptions(object):
    """A class that stores options in a single file.

    Example:
        .. code-block:: python

            # file: my_module.py
            options = FileOptions(__file__, number_to_add=1)

            def my_func(x: int) -> int:
                return x + options.number_to_add

            # file: my_script.py
            import my_module
            my_module.options.set(number_to_add=2)
            my_module.my_func(1)  # returns 3

    """

    def __init__(self, __file__, **init_kwargs):
        self.__file__ = __file__
        for k, v in init_kwargs.items():
            setattr(self, k, v)

    def set(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f'{k} is not an option for file "{self.__file__}". Available options are: {", ".join(self.__dict__.keys())}.')
            setattr(self, k, v)


ARGDEF = object()
"""A special value to indicate that the default value of an argument will be determined in a deferred manner. See :func:`default_args`."""


def default_args(func):
    """A helper function handles the case of "fall-through" default arguments. Suppose we have two functions:
    ``f`` and ``g``, and ``f`` calls ``g``. ``g`` has a default argument ``x``, e.g., ``x=1``.
    In many cases, we do not want to specify the default value of ``x`` in ``f``. One way to do this is to
    use ``None`` as the default value of ``x`` in ``f``, and then check if ``x`` is ``None`` in ``g``. However
    this does not handle cases where ``x`` can be ``None`` in other cases. It also requires additional
    checks in ``g``. With this decorator, we can simply write ``x=ARGDEF`` in ``f``, and then ``x`` will
    be set to ``1`` in ``g``.

    Example:
        .. code-block:: python

            def f(x=ARGDEF):
                g(x)

            @default_args
            def g(x=1):
                print(x)

            f()  # prints 1
            f(2)  # prints 2

    """
    def wrapper(func):
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
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

