#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : environ_v2.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/17/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import functools
import contextlib
from jacinle.utils.printing import kvprint
from jacinle.utils.meta import dict_deep_kv, run_once

__all__ = ['configs', 'def_configs', 'def_configs_func', 'set_configs', 'set_configs_func', 'StrictG']

ENABLE_CONFIG_AUTOINIT = False
ENABLE_DEF_CONFIG = False


@contextlib.contextmanager
def set_configs():
    """
    Set the global configurations.

    Args:
    """
    global ENABLE_CONFIG_AUTOINIT
    assert not ENABLE_CONFIG_AUTOINIT
    ENABLE_CONFIG_AUTOINIT = True
    yield
    assert ENABLE_CONFIG_AUTOINIT
    ENABLE_CONFIG_AUTOINIT = False


@contextlib.contextmanager
def def_configs():
    """
    Context manager to global configuration options.

    Args:
    """
    global ENABLE_DEF_CONFIG
    with set_configs():
        assert not ENABLE_DEF_CONFIG
        ENABLE_DEF_CONFIG = True
        yield
        assert ENABLE_DEF_CONFIG
        ENABLE_DEF_CONFIG = False


def set_configs_func(func):
    """
    Decorator to set a configs.

    Args:
        func: (todo): write your description
    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        """
        Decorator that will be used function.

        Args:
        """
        with set_configs():
            func(*args, **kwargs)
    return wrapped


def def_configs_func(func):
    """
    Decorator for configs configuration function.

    Args:
        func: (callable): write your description
    """
    @functools.wraps(func)
    @run_once
    def wrapped(*args, **kwargs):
        """
        Decorator to add a function to the wrapped function.

        Args:
        """
        with def_configs():
            func(*args, **kwargs)
    return wrapped


class StrictG(dict):
    def __init__(self, *args, **kwargs):
        """
        Initialize the object.

        Args:
            self: (todo): write your description
        """
        super().__init__(*args, **kwargs)
        self['__defined_kvs__'] = dict()
        self['__defined_info__'] = dict()

    def __getattr__(self, k):
        """
        Returns the value of a attribute.

        Args:
            self: (todo): write your description
            k: (str): write your description
        """
        if k not in self:
            if ENABLE_CONFIG_AUTOINIT:
                self[k] = StrictG()
            else:
                raise AttributeError
        return self[k]

    def __setattr__(self, k, v):
        """
        Set an attribute of - value pair.

        Args:
            self: (todo): write your description
            k: (todo): write your description
            v: (todo): write your description
        """
        if ENABLE_DEF_CONFIG:
            if k in self.__defined_kvs__ or (k in self and isinstance(self[k], StrictG)):
                raise AttributeError('Key "{}" has already been implicitly or explicitly defined.'.format(k))
            self.__defined_kvs__[k] = v
            self.setdefault(k, v)
            self.validate(k)
        else:
            self[k] = v
            self.validate(k)

    def def_(self, name, type=None, choices=None, default=None, help=help):
        """
        Validate the given type.

        Args:
            self: (todo): write your description
            name: (str): write your description
            type: (todo): write your description
            choices: (list): write your description
            default: (todo): write your description
            help: (todo): write your description
            help: (todo): write your description
        """
        if name in self.__defined_info__ or (name in self and isinstance(self[name], StrictG)):
            raise AttributeError('Key "{}" has already been implicitly or explicitly defined.'.format(name))

        self.__defined_info__[name] = {
            'type': type,
            'choices': choices,
            'help': help
        }
        if default is not None:
            self.setdefault(default)

        if name in self:
            self.validate(name)

    def validate(self, name):
        """
        Validate the field is valid

        Args:
            self: (todo): write your description
            name: (str): write your description
        """
        if name in self.__defined_info__:
            info = self.__defined_info__[name]

            value = self[name]
            if info['type'] is not None:
                if not isinstance(value, info['type']):
                    raise TypeError('Key "{}" does not satisfy the type constraint: {}, got {}.'.format(name, info['type'], type(value)))
            if info['choices'] is not None:
                if value not in info['choices']:
                    raise TypeError('Key "{}" should be one of the: {}, got {}.'.format(name, info['choices'], value))

    def find_undefined_values(self, global_prefix='configs'):
        """
        Return a list of dims in global_prefix.

        Args:
            self: (todo): write your description
            global_prefix: (str): write your description
        """
        undefined = list()
        def dfs(d, prefix):
            """
            Dfs a dictionary keys in d.

            Args:
                d: (dict): write your description
                prefix: (str): write your description
            """
            for k, v in d.items():
                if isinstance(v, StrictG):
                    dfs(v, prefix + '.' + k)
                else:
                    if k not in d.__defined_kvs__ and not (k.startswith('__') and k.endswith('__')):
                        undefined.append(prefix + '.' + k)
        try:
            dfs(self, global_prefix)
        finally:
            del dfs

        return undefined

    def format(self, sep=': ', end='\n'):
        """
        Return a dict todo as a dict.

        Args:
            self: (todo): write your description
            sep: (todo): write your description
            end: (int): write your description
        """
        return kvformat(dict(dict_deep_kv(self)), sep=sep, end=end)

    def print(self, sep=': ', end='\n', file=None):
        """
        Pretty print method to print out a dictionary.

        Args:
            self: (todo): write your description
            sep: (str): write your description
            end: (int): write your description
            file: (str): write your description
        """
        return kvprint(dict(dict_deep_kv(self)), sep=sep, end=end, file=file)


configs = StrictG()

