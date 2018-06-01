#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pipe.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/22/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import queue
import uuid

from jacinle.utils.meta import notnone_property
from .controller import BroadcastMessage

__all__ = ['DistribInputPipe', 'DistribOutputPipe']


class PipeBase(object):
    def __init__(self, direction, name, bufsize, identifier=None):
        self._direction = direction
        self._name = name
        self._controller = None
        self._queue = queue.Queue(maxsize=bufsize)
        self._identifier = identifier or uuid.uuid4().hex

    @property
    def direction(self):
        return self._direction

    @property
    def name(self):
        return self._name

    @property
    def identifier(self):
        return self._identifier

    @notnone_property
    def controller(self):
        return self._controller

    @property
    def raw_queue(self):
        return self._queue

    def set_controller(self, controller):
        self._controller = controller

    def put(self, data):
        self._queue.put(self._wrap_send_message(data))

    def put_nowait(self, data):
        try:
            self._queue.put_nowait(self._wrap_send_message(data))
            return True
        except queue.Full:
            return False

    def get(self):
        return self._unwrap_recv_message(self._queue.get())

    def get_nowait(self):
        try:
            return self._unwrap_recv_message(self._queue.get_nowait())
        except queue.Empty:
            return None

    def empty(self):
        return self._queue.empty()

    def full(self):
        return self._queue.full()

    def _wrap_send_message(self, data):
        raise NotImplementedError()

    def _unwrap_recv_message(self, data):
        raise NotImplementedError()


class DistribBroadcastPipeBase(PipeBase):
    def __init__(self, direction, name, bufsize=10):
        super().__init__(direction, name, bufsize)

    def _unwrap_recv_message(self, data):
        return data.payload

    def _wrap_send_message(self, data):
        return BroadcastMessage(self.identifier, data)


class DistribInputPipe(DistribBroadcastPipeBase):
    def __init__(self, name, bufsize=10):
        super().__init__('IN', name, bufsize)


class DistribOutputPipe(DistribBroadcastPipeBase):
    def __init__(self, name, bufsize=10):
        super().__init__('OUT', name, bufsize)
