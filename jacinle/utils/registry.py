#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : registry.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/29/16
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import collections
import threading

from .context import EmptyContext

__all__ = ['Registry', 'DefaultRegistry', 'RegistryGroup', 'CallbackRegistry', 'LockRegistry']


class Registry(object):
    __FALLBACK_KEY__ = '__fallback__'

    _registry = None

    def __init__(self):
        """
        Initialize the registry.

        Args:
            self: (todo): write your description
        """
        self._init_registry()

    def _init_registry(self):
        """
        Initialize the registry.

        Args:
            self: (todo): write your description
        """
        self._registry = dict()

    @property
    def fallback(self):
        """
        Returns the fallback of the registry.

        Args:
            self: (todo): write your description
        """
        return self._registry.get(self.__FALLBACK_KEY__, None)

    def set_fallback(self, value):
        """
        Sets the fallback value.

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        self._registry[self.__FALLBACK_KEY__] = value
        return self

    def register(self, entry, value):
        """
        Register a new entry.

        Args:
            self: (todo): write your description
            entry: (str): write your description
            value: (todo): write your description
        """
        self._registry[entry] = value
        return self

    def unregister(self, entry):
        """
        Unregister the given entry.

        Args:
            self: (todo): write your description
            entry: (str): write your description
        """
        return self._registry.pop(entry, None)

    def has(self, entry):
        """
        Returns true if the given entry.

        Args:
            self: (todo): write your description
            entry: (todo): write your description
        """
        return entry in self._registry

    def lookup(self, entry, fallback=True, default=None):
        """
        Return the default value for the given entry.

        Args:
            self: (todo): write your description
            entry: (todo): write your description
            fallback: (todo): write your description
            default: (todo): write your description
        """
        if fallback:
            fallback_value = self._registry.get(self.__FALLBACK_KEY__, default)
        else:
            fallback_value = default
        return self._registry.get(entry, fallback_value)

    def keys(self):
        """
        Returns a list of all keys.

        Args:
            self: (todo): write your description
        """
        return list(self._registry.keys())

    def items(self):
        """
        Returns a list items.

        Args:
            self: (todo): write your description
        """
        return list(self._registry.items())


class DefaultRegistry(Registry):
    __base_class__ = dict

    def _init_registry(self):
        """
        Initialize the base class.

        Args:
            self: (todo): write your description
        """
        base_class = type(self).__base_class__
        self._registry = collections.defaultdict(base_class)

    def lookup(self, entry, fallback=False, default=None):
        """
        Returns the default value for the given entry.

        Args:
            self: (todo): write your description
            entry: (todo): write your description
            fallback: (todo): write your description
            default: (todo): write your description
        """
        assert fallback is False and default is None
        return self._registry[entry]

    def __getitem__(self, item):
        """
        Return the item from the cache.

        Args:
            self: (todo): write your description
            item: (str): write your description
        """
        return self.lookup(item)


class RegistryGroup(object):
    __base_class__ = Registry

    def __init__(self):
        """
        !

        Args:
            self: (todo): write your description
        """
        self._init_registry_group()

    def _init_registry_group(self):
        """
        Initialize the registry group.

        Args:
            self: (todo): write your description
        """
        base_class = type(self).__base_class__
        self._registries = collections.defaultdict(base_class)

    def __getitem__(self, item):
        """
        Return the item from the cache.

        Args:
            self: (todo): write your description
            item: (str): write your description
        """
        return self._registries[item]

    def register(self, registry_name, entry, value, **kwargs):
        """
        Register an entry to the registry.

        Args:
            self: (todo): write your description
            registry_name: (str): write your description
            entry: (str): write your description
            value: (todo): write your description
        """
        return self._registries[registry_name].register(entry, value, **kwargs)

    def lookup(self, registry_name, entry, fallback=True, default=None):
        """
        Returns the default value of an entry.

        Args:
            self: (todo): write your description
            registry_name: (str): write your description
            entry: (todo): write your description
            fallback: (todo): write your description
            default: (todo): write your description
        """
        return self._registries[registry_name].lookup(entry, fallback=fallback, default=default)


class CallbackRegistry(Registry):
    """
    A callable manager utils.

    If there exists a super callback, it will block all callbacks.
    A super callback will receive the called name as its first argument.

    Then the dispatcher will try to call the callback by name.
    If such name does not exists, a fallback callback will be called.

    The fallback callback will also receive the called name as its first argument.

    Examples:

    >>> registry = CallbackRegistry()
    >>> callback_func = print
    >>> registry.register('name', callback_func)  # register a callback.
    >>> registry.dispatch('name', 'arg1', 'arg2', kwarg1='kwarg1')  # dispatch.
    """
    def __init__(self):
        """
        Initialize the state

        Args:
            self: (todo): write your description
        """
        super().__init__()
        self._super_callback = None

    @property
    def super_callback(self):
        """
        Returns the super callback.

        Args:
            self: (todo): write your description
        """
        return self._super_callback

    def set_super_callback(self, callback):
        """
        Set the super callback.

        Args:
            self: (todo): write your description
            callback: (todo): write your description
        """
        self._super_callback = callback
        return self

    @property
    def fallback_callback(self):
        """
        Returns the fallbackback_callback

        Args:
            self: (todo): write your description
        """
        return self.fallback

    def set_fallback_callback(self, callback):
        """
        Sets the fallback callback.

        Args:
            self: (todo): write your description
            callback: (todo): write your description
        """
        return self.set_fallback(callback)

    def dispatch(self, name, *args, **kwargs):
        """
        This decorator sets the callback.

        Args:
            self: (todo): write your description
            name: (str): write your description
        """
        if self._super_callback is not None:
            return self._super_callback(self, name, *args, **kwargs)
        return self.dispatch_direct(name, *args)

    def dispatch_direct(self, name, *args, **kwargs):
        """Dispatch by name, ignoring the super callback."""
        callback = self.lookup(name, fallback=False)
        if callback is None:
            if self.fallback_callback is None:
                raise ValueError('Unknown callback entry: "{}".'.format(name))
            return self.fallback_callback(self, name, *args, **kwargs)
        return callback(*args, **kwargs)


class LockRegistry(DefaultRegistry):
    __base_class__ = threading.Lock

    def synchronized(self, entry, activate=True):
        """
        Look up an entry in the entry hasentry.

        Args:
            self: (todo): write your description
            entry: (todo): write your description
            activate: (todo): write your description
        """
        if activate:
            return self.lookup(entry)
        return EmptyContext()
