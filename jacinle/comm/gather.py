#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : gather.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/22/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import contextlib
import queue
import threading

import zmq

from jacinle.concurrency.packing import loadb, dumpb
from jacinle.concurrency.zmq_utils import get_addr, bind_to_random_ipc, graceful_close
from jacinle.utils.meta import notnone_property

__all__ = ['GatherOutputPipe', 'GatherInputPipe', 'make_gather_pair']

GATHER_HWM = 2


class GatherInputPipe(object):
    def __init__(self, name, mode='tcp'):
        """
        Initialize the socket.

        Args:
            self: (todo): write your description
            name: (str): write your description
            mode: (todo): write your description
        """
        self._name = name
        self._mode = mode
        self._conn_info = None

        self._context = zmq.Context()
        self._sock = self._context.socket(zmq.PULL)
        self._sock.set_hwm(GATHER_HWM)

    @notnone_property
    def conn_info(self):
        """
        Return the connection info.

        Args:
            self: (todo): write your description
        """
        return self._conn_info

    def initialize(self):
        """
        Initialize the connection.

        Args:
            self: (todo): write your description
        """
        if self._conn_info is not None:
            return

        if self._mode == 'tcp':
            port = self._sock.bind_to_random_port('tcp://*')
            self._conn_info = 'tcp://{}:{}'.format(get_addr(), port)
        elif self._mode == 'ipc':
            self._conn_info = bind_to_random_ipc(self._sock, self._name)

    def finalize(self):
        """
        Finalize the socket.

        Args:
            self: (todo): write your description
        """
        graceful_close(self._sock)
        self._context.term()

    @contextlib.contextmanager
    def activate(self):
        """
        A context manager which this context manager.

        Args:
            self: (todo): write your description
        """
        self.initialize()
        try:
            yield
        finally:
            self.finalize()

    def recv(self):
        """
        Receive a message from the socket.

        Args:
            self: (todo): write your description
        """
        try:
            return loadb(self._sock.recv(copy=False).bytes)
        except zmq.ContextTerminated:
            pass


class GatherOutputPipe(object):
    def __init__(self, conn_info, send_qsize=10):
        """
        Initialize a connection pool.

        Args:
            self: (todo): write your description
            conn_info: (todo): write your description
            send_qsize: (int): write your description
        """
        self._conn_info = conn_info
        self._send_qsize = send_qsize

        self._context = None
        self._sock = None
        self._send_queue = None
        self._send_thread = None

    def initialize(self):
        """
        Initialize the socket.

        Args:
            self: (todo): write your description
        """
        self._context = zmq.Context()
        self._sock = self._context.socket(zmq.PUSH)
        self._sock.set_hwm(GATHER_HWM)
        self._sock.connect(self._conn_info)

        self._send_queue = queue.Queue(maxsize=self._send_qsize)
        self._send_thread = threading.Thread(target=self.mainloop_send, daemon=True)
        self._send_thread.start()

    def finalize(self):
        """
        Finalize the socket.

        Args:
            self: (todo): write your description
        """
        graceful_close(self._sock)
        self._context.term()

    @contextlib.contextmanager
    def activate(self):
        """
        A context manager which this context manager.

        Args:
            self: (todo): write your description
        """
        self.initialize()
        try:
            yield
        finally:
            self.finalize()

    def mainloop_send(self):
        """
        The main loop.

        Args:
            self: (todo): write your description
        """
        try:
            while True:
                job = self._send_queue.get()
                self._sock.send(dumpb(job), copy=False)
        except zmq.ContextTerminated:
            pass

    def send(self, payload):
        """
        Send a message to the queue.

        Args:
            self: (todo): write your description
            payload: (dict): write your description
        """
        self._send_queue.put(payload)
        return self


def make_gather_pair(name, nr_workers=None, mode='tcp', send_qsize=10):
    """
    Gathers a pair

    Args:
        name: (str): write your description
        nr_workers: (todo): write your description
        mode: (str): write your description
        send_qsize: (int): write your description
    """
    pull = GatherInputPipe(name, mode=mode)
    pull.initialize()
    nr_pushs = nr_workers or 1
    pushs = [GatherOutputPipe(pull.conn_info, send_qsize=send_qsize) for _ in range(nr_pushs)]

    if nr_workers is None:
        return pull, pushs[0]
    return pull, pushs
