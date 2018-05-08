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
