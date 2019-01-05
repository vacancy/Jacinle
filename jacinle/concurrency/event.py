#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : event.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/22/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import threading
import multiprocessing
import functools

__all__ = [
    'MPLibExtension', 'instantiate_mplib_ext',
    'MTBooleanEvent', 'MPBooleanEvent',
    'MTOrEvent', 'MPOrEvent',
    'MTCoordinatorEvent'
]


class MPLibExtension(object):
    __mplib__ = threading


def instantiate_mplib_ext(base_class):
    class MultiThreadingImpl(base_class):
        __name__ = 'MT' + base_class.__name__
        __mplib__ = threading

    class MultiProcessingImpl(base_class):
        __name__ = 'MP' + base_class.__name__
        __mplib__ = multiprocessing

    return MultiThreadingImpl, MultiProcessingImpl


class BooleanEvent(MPLibExtension):
    def __init__(self):
        self._t = type(self).__mplib__.Event()
        self._f = type(self).__mplib__.Event()
        self._t.clear()
        self._f.set()
        self._lock = type(self).__mplib__.Lock()

    def is_true(self):
        with self._lock:
            return self._t.is_set()

    def is_false(self):
        with self._lock:
            return self._f.is_set()

    def set(self):
        with self._lock:
            self._t.set()
            self._f.clear()

    def clear(self):
        with self._lock:
            self._t.clear()
            self._f.set()

    def wait(self, predicate=True, timeout=None):
        target = self._t if predicate else self._f
        return target.wait(timeout)

    def wait_true(self, timeout=None):
        return self.wait(True, timeout=timeout)

    def wait_false(self, timeout=None):
        return self.wait(False, timeout=timeout)

    def set_true(self):
        self.set()

    def set_false(self):
        self.clear()

    def value(self):
        return self.is_true()


MTBooleanEvent, MPBooleanEvent = instantiate_mplib_ext(BooleanEvent)


def _or_event_set(self):
    self._set()
    self.changed()


def _or_event_clear(self):
    self._clear()
    self.changed()


def _orify(e, changed_callback):
    e._set = e.set
    e._clear = e.clear
    e.changed = changed_callback
    e.set = lambda: _or_event_set(e)
    e.clear = lambda: _or_event_clear(e)


def OrEvent(*events, mplib=threading):
    """Waiting on several events together.
    http://stackoverflow.com/questions/12317940/python-threading-can-i-sleep-on-two-threading-events-simultaneously"""

    or_event = mplib.Event()

    def changed():
        bools = [e.is_set() for e in events]
        if any(bools):
            or_event.set()
        else:
            or_event.clear()

    for e in events:
        _orify(e, changed)
    changed()
    return or_event


MTOrEvent = functools.partial(OrEvent, mplib=threading)
MPOrEvent = functools.partial(OrEvent, mplib=multiprocessing)


class MTCoordinatorEvent(object):
    def __init__(self, nr_workers):
        self._event = threading.Event()
        self._queue = queue.Queue()
        self._nr_workers = nr_workers

    def broadcast(self):
        self._event.set()
        for i in range(self._nr_workers):
            self._queue.get()
        self._event.clear()

    def wait(self):
        self._event.wait()
        self._queue.put(1)

    def check(self):
        rc = self._event.is_set()
        if rc:
            self._queue.put(1)
        return rc

