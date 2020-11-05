#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : registry.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import collections
import uuid

from jacinle.utils.registry import Registry, RegistryGroup

__all__ = [
    'SimpleEventRegistry',
    'EventRegistry', 'EventRegistryGroup',
    'register_event', 'trigger_event'
]


class SimpleEventRegistry(object):
    def __init__(self, allowed_events=None):
        """
        Initialize events.

        Args:
            self: (todo): write your description
            allowed_events: (bool): write your description
        """
        self._allowed_events = allowed_events
        self._events = collections.defaultdict(list)

    def register(self, event, callback):
        """
        Register a callback implementation :.

        Args:
            self: (todo): write your description
            event: (str): write your description
            callback: (todo): write your description
        """
        if self._allowed_events is not None:
            assert event in self._allowed_events
        self._events[event].append(callback)

    def trigger(self, event, *args, **kwargs):
        """
        Triggers all registered callbacks.

        Args:
            self: (todo): write your description
            event: (todo): write your description
        """
        if self._allowed_events is not None:
            assert event in self._allowed_events
        for f in self._events[event]:
            f(*args, **kwargs)


class EventRegistry(Registry):
    DEF_PRIORITY = 10

    def _init_registry(self):
        """
        Initialize the registry.

        Args:
            self: (todo): write your description
        """
        self._registry = collections.defaultdict(  # entry
            lambda: collections.defaultdict(  # priority
                collections.OrderedDict,  # sub-key
            )
        )

    def register(self, entry, callback, priority=DEF_PRIORITY, subkey=None):
        """
        Register a new subkey.

        Args:
            self: (todo): write your description
            entry: (str): write your description
            callback: (todo): write your description
            priority: (int): write your description
            DEF_PRIORITY: (todo): write your description
            subkey: (str): write your description
        """
        if subkey is None:
            subkey = uuid.uuid4()
        self._registry[entry][priority][subkey] = callback
        return subkey

    def unregister(self, entry, key=None, priority=DEF_PRIORITY):
        """
        Unregister a previously registered with the given entry.

        Args:
            self: (todo): write your description
            entry: (str): write your description
            key: (str): write your description
            priority: (int): write your description
            DEF_PRIORITY: (todo): write your description
        """
        if key is None:
            return self._registry.pop(entry, None)

        entries = self._registry[entry][priority]
        if callable(entry):
            for k, v in list(entries.items()):
                return entries.pop(k)
            return None
        return entries.pop(key, None)

    def lookup(self, entry, key=None, priority=DEF_PRIORITY, default=None):
        """
        Returns the value for the given entry.

        Args:
            self: (todo): write your description
            entry: (todo): write your description
            key: (str): write your description
            priority: (int): write your description
            DEF_PRIORITY: (int): write your description
            default: (todo): write your description
        """
        if key is None:
            return self._registry.get(key, default)

        return self._registry[entry][priority].get(key, default)

    def trigger(self, entry, *args, **kwargs):
        """
        Triggers an entry.

        Args:
            self: (todo): write your description
            entry: (todo): write your description
        """
        self.trigger_args(entry, args, kwargs)

    def trigger_args(self, entry, args, kwargs):
        """
        Trigger an entry point.

        Args:
            self: (todo): write your description
            entry: (todo): write your description
        """
        group = self._registry[entry]
        entry_lists = [group[k] for k in sorted(group.keys())]

        for entries in entry_lists:
            for entry in entries:
                entry(*args, **kwargs)


class EventRegistryGroup(RegistryGroup):
    __base_class__ = EventRegistry

    def lookup(self, registry_name, entry, key=None, **kwargs):
        """
        Look up a value from the registry.

        Args:
            self: (todo): write your description
            registry_name: (str): write your description
            entry: (todo): write your description
            key: (str): write your description
        """
        return self._registries[registry_name].lookup(entry, key=key, **kwargs)

    def trigger(self, registry_name, entry, *args, **kwargs):
        """
        Triggers.

        Args:
            self: (todo): write your description
            registry_name: (str): write your description
            entry: (todo): write your description
        """
        return self._registries[registry_name].trigger(entry, *args, **kwargs)


default_event_registry_group = EventRegistryGroup()
register_event = default_event_registry_group.register
trigger_event = default_event_registry_group.trigger
