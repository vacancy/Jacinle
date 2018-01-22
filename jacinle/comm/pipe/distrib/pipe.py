# -*- coding: utf-8 -*-
# File   : pipe.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 22/01/2018
#
# This file is part of Jacinle.

import queue

from jacinle.utils.meta import notnone_property

__all__ = ['InputPipe', 'OutputPipe']


class PipeBase(object):
    def __init__(self, direction, name, bufsize):
        self._direction = direction
        self._name = name
        self._controller = None
        self._queue = queue.Queue(maxsize=bufsize)

    @property
    def direction(self):
        return self._direction

    @property
    def name(self):
        return self._name

    @notnone_property
    def controller(self):
        return self._controller

    def set_controller(self, controller):
        self._controller = controller

    def put(self, data):
        self._queue.put(data)

    def put_nowait(self, data):
        try:
            self._queue.put_nowait(data)
            return True
        except queue.Full:
            return False

    def get(self):
        return self._queue.get()

    def get_nowait(self):
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def empty(self):
        return self._queue.empty()

    def full(self):
        return self._queue.full()


class InputPipe(PipeBase):
    def __init__(self, name, bufsize=10):
        super().__init__('IN', name, bufsize)


class OutputPipe(PipeBase):
    def __init__(self, name, bufsize=10):
        super().__init__('OUT', name, bufsize)
