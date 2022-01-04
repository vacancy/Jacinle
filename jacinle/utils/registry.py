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
import uuid

from .context import EmptyContext

__all__ = [
    'Registry', 'DefaultRegistry', 'RegistryGroup',
    'CallbackRegistry', 'LockRegistry',
    'subclass_registry_point',
    'SimpleEventRegistry',
    'EventRegistry', 'EventRegistryGroup',
    'register_event', 'trigger_event'
]


class Registry(object):
    __FALLBACK_KEY__ = '__fallback__'

    _registry = None

    def __init__(self):
        self._init_registry()

    def _init_registry(self):
        self._registry = dict()

    @property
    def fallback(self):
        return self._registry.get(self.__FALLBACK_KEY__, None)

    def set_fallback(self, value):
        self._registry[self.__FALLBACK_KEY__] = value
        return self

    def register(self, entry, value):
        self._registry[entry] = value
        return self

    def unregister(self, entry):
        return self._registry.pop(entry, None)

    def has(self, entry):
        return entry in self._registry

    def lookup(self, entry, fallback=True, default=None):
        if fallback:
            fallback_value = self._registry.get(self.__FALLBACK_KEY__, default)
        else:
            fallback_value = default
        return self._registry.get(entry, fallback_value)

    def keys(self):
        return list(self._registry.keys())

    def items(self):
        return list(self._registry.items())


class DefaultRegistry(Registry):
    __base_class__ = dict

    def _init_registry(self):
        base_class = type(self).__base_class__
        self._registry = collections.defaultdict(base_class)

    def lookup(self, entry, fallback=False, default=None):
        assert fallback is False and default is None
        return self._registry[entry]

    def __getitem__(self, item):
        return self.lookup(item)


class RegistryGroup(object):
    __base_class__ = Registry

    def __init__(self):
        self._init_registry_group()

    def _init_registry_group(self):
        base_class = type(self).__base_class__
        self._registries = collections.defaultdict(base_class)

    def __getitem__(self, item):
        return self._registries[item]

    def register(self, registry_name, entry, value, **kwargs):
        return self._registries[registry_name].register(entry, value, **kwargs)

    def lookup(self, registry_name, entry, fallback=True, default=None):
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
        super().__init__()
        self._super_callback = None

    @property
    def super_callback(self):
        return self._super_callback

    def set_super_callback(self, callback):
        self._super_callback = callback
        return self

    @property
    def fallback_callback(self):
        return self.fallback

    def set_fallback_callback(self, callback):
        return self.set_fallback(callback)

    def dispatch(self, name, *args, **kwargs):
        if self._super_callback is not None:
            return self._super_callback(self, name, *args, **kwargs)
        return self.dispatch_direct(name, *args, **kwargs)

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
        if activate:
            return self.lookup(entry)
        return EmptyContext()


def subclass_registry_point(wrapped):
    def register_subclass(cls, identifier):
        def wrapper(subcls):
            cls.SUBCLASS_REGISTRY.register(identifier, subcls)
            return subcls
        return wrapper

    def available_subclasses(cls):
        return cls.SUBCLASS_REGISTRY.keys()

    def instantiate(cls, identifier, *args, **kwargs):
        return cls.SUBCLASS_REGISTRY.dispatch(identifier, *args, **kwargs)

    wrapped.SUBCLASS_REGISTRY = CallbackRegistry()
    wrapped.register_subclass = classmethod(register_subclass)
    wrapped.available_subclasses = classmethod(available_subclasses)
    wrapped.instantiate = classmethod(instantiate)
    return wrapped


class SimpleEventRegistry(object):
    def __init__(self, allowed_events=None):
        self._allowed_events = allowed_events
        self._events = collections.defaultdict(list)

    def register(self, event, callback):
        if self._allowed_events is not None:
            assert event in self._allowed_events
        self._events[event].append(callback)

    def trigger(self, event, *args, **kwargs):
        if self._allowed_events is not None:
            assert event in self._allowed_events
        for f in self._events[event]:
            f(*args, **kwargs)


class EventRegistry(Registry):
    DEF_PRIORITY = 10

    def _init_registry(self):
        self._registry = collections.defaultdict(  # entry
            lambda: collections.defaultdict(  # priority
                collections.OrderedDict,  # sub-key
            )
        )

    def register(self, entry, callback, priority=DEF_PRIORITY, subkey=None):
        if subkey is None:
            subkey = uuid.uuid4()
        self._registry[entry][priority][subkey] = callback
        return subkey

    def unregister(self, entry, key=None, priority=DEF_PRIORITY):
        if key is None:
            return self._registry.pop(entry, None)

        entries = self._registry[entry][priority]
        if callable(entry):
            for k, v in list(entries.items()):
                return entries.pop(k)
            return None
        return entries.pop(key, None)

    def lookup(self, entry, key=None, priority=DEF_PRIORITY, default=None):
        if key is None:
            return self._registry.get(key, default)

        return self._registry[entry][priority].get(key, default)

    def trigger(self, entry, *args, **kwargs):
        self.trigger_args(entry, args, kwargs)

    def trigger_args(self, entry, args, kwargs):
        group = self._registry[entry]
        entry_lists = [group[k] for k in sorted(group.keys())]

        for entries in entry_lists:
            for entry in entries:
                entry(*args, **kwargs)


class EventRegistryGroup(RegistryGroup):
    __base_class__ = EventRegistry

    def lookup(self, registry_name, entry, key=None, **kwargs):
        return self._registries[registry_name].lookup(entry, key=key, **kwargs)

    def trigger(self, registry_name, entry, *args, **kwargs):
        return self._registries[registry_name].trigger(entry, *args, **kwargs)


default_event_registry_group = EventRegistryGroup()
register_event = default_event_registry_group.register
trigger_event = default_event_registry_group.trigger
