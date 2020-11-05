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
        """
        Initialize a queue.

        Args:
            self: (todo): write your description
            direction: (todo): write your description
            name: (str): write your description
            bufsize: (int): write your description
            identifier: (todo): write your description
        """
        self._direction = direction
        self._name = name
        self._controller = None
        self._queue = queue.Queue(maxsize=bufsize)
        self._identifier = identifier or uuid.uuid4().hex

    @property
    def direction(self):
        """
        Returns the direction of the direction.

        Args:
            self: (todo): write your description
        """
        return self._direction

    @property
    def name(self):
        """
        The name of the name

        Args:
            self: (todo): write your description
        """
        return self._name

    @property
    def identifier(self):
        """
        The identifier : the identifier

        Args:
            self: (todo): write your description
        """
        return self._identifier

    @notnone_property
    def controller(self):
        """
        Returns the controller controller.

        Args:
            self: (todo): write your description
        """
        return self._controller

    @property
    def raw_queue(self):
        """
        The queue : class.

        Args:
            self: (todo): write your description
        """
        return self._queue

    def set_controller(self, controller):
        """
        Sets the controller.

        Args:
            self: (todo): write your description
            controller: (todo): write your description
        """
        self._controller = controller

    def put(self, data):
        """
        Pushes data to the queue.

        Args:
            self: (todo): write your description
            data: (array): write your description
        """
        self._queue.put(self._wrap_send_message(data))

    def put_nowait(self, data):
        """
        Put data to the queue.

        Args:
            self: (todo): write your description
            data: (array): write your description
        """
        try:
            self._queue.put_nowait(self._wrap_send_message(data))
            return True
        except queue.Full:
            return False

    def get(self):
        """
        Get the next message from the queue.

        Args:
            self: (todo): write your description
        """
        return self._unwrap_recv_message(self._queue.get())

    def get_nowait(self):
        """
        Get the next message from the queue.

        Args:
            self: (todo): write your description
        """
        try:
            return self._unwrap_recv_message(self._queue.get_nowait())
        except queue.Empty:
            return None

    def empty(self):
        """
        Return the queue.

        Args:
            self: (todo): write your description
        """
        return self._queue.empty()

    def full(self):
        """
        Return the full queue of the queue.

        Args:
            self: (todo): write your description
        """
        return self._queue.full()

    def _wrap_send_message(self, data):
        """
        Wraps the message. message.

        Args:
            self: (todo): write your description
            data: (todo): write your description
        """
        raise NotImplementedError()

    def _unwrap_recv_message(self, data):
        """
        Unwrap a message

        Args:
            self: (todo): write your description
            data: (array): write your description
        """
        raise NotImplementedError()


class DistribBroadcastPipeBase(PipeBase):
    def __init__(self, direction, name, bufsize=10):
        """
        Initialize a new buffer.

        Args:
            self: (todo): write your description
            direction: (todo): write your description
            name: (str): write your description
            bufsize: (int): write your description
        """
        super().__init__(direction, name, bufsize)

    def _unwrap_recv_message(self, data):
        """
        Unwrap a message

        Args:
            self: (todo): write your description
            data: (array): write your description
        """
        return data.payload

    def _wrap_send_message(self, data):
        """
        Convert a raw message to a raw bytes.

        Args:
            self: (todo): write your description
            data: (todo): write your description
        """
        return BroadcastMessage(self.identifier, data)


class DistribInputPipe(DistribBroadcastPipeBase):
    def __init__(self, name, bufsize=10):
        """
        Initialize the buffer.

        Args:
            self: (todo): write your description
            name: (str): write your description
            bufsize: (int): write your description
        """
        super().__init__('IN', name, bufsize)


class DistribOutputPipe(DistribBroadcastPipeBase):
    def __init__(self, name, bufsize=10):
        """
        Initialize the buffer.

        Args:
            self: (todo): write your description
            name: (str): write your description
            bufsize: (int): write your description
        """
        super().__init__('OUT', name, bufsize)
