#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : broadcast.py
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

__all__ = ['BroadcastOutputPipe', 'BroadcastInputPipe', 'make_broadcast_pair']

BROADCAST_HWM = 2


class BroadcastOutputPipe(object):
    def __init__(self, name, send_qsize=10, mode='tcp'):
        self._name = name
        self._send_qsize = send_qsize
        self._mode = mode
        self._conn_info = None

        self._context = zmq.Context()
        self._sock = self._context.socket(zmq.PUB)
        self._sock.set_hwm(BROADCAST_HWM)
        self._send_queue = None
        self._send_thread = None

    @notnone_property
    def conn_info(self):
        return self._conn_info

    def initialize(self):
        if self._conn_info is not None:
            return

        if self._mode == 'tcp':
            port = self._sock.bind_to_random_port('tcp://*')
            self._conn_info = 'tcp://{}:{}'.format(get_addr(), port)
        elif self._mode == 'ipc':
            self._conn_info = bind_to_random_ipc(self._sock, self._name)

        self._send_queue = queue.Queue(maxsize=self._send_qsize)
        self._send_thread = threading.Thread(target=self.mainloop_send, daemon=True)
        self._send_thread.start()

    def finalize(self):
        graceful_close(self._sock)
        self._context.term()

    @contextlib.contextmanager
    def activate(self):
        self.initialize()
        try:
            yield
        finally:
            self.finalize()

    def mainloop_send(self):
        try:
            while True:
                job = self._send_queue.get()
                self._sock.send(dumpb(job), copy=False)
        except zmq.ContextTerminated:
            pass

    def send(self, payload):
        self._send_queue.put(payload)
        return self


class BroadcastInputPipe(object):
    def __init__(self, conn_info):
        self._conn_info = conn_info

        self._context = None
        self._sock = None

    def initialize(self):
        self._context = zmq.Context()
        self._sock = self._context.socket(zmq.SUB)
        self._sock.set_hwm(BROADCAST_HWM)
        self._sock.connect(self._conn_info)
        self._sock.setsockopt(zmq.SUBSCRIBE, b'')

    def finalize(self):
        graceful_close(self._sock)
        self._context.term()

    @contextlib.contextmanager
    def activate(self):
        self.initialize()
        try:
            yield
        finally:
            self.finalize()

    def recv(self):
        try:
            return loadb(self._sock.recv(copy=False).bytes)
        except zmq.ContextTerminated:
            pass


def make_broadcast_pair(name, nr_workers=None, mode='tcp', send_qsize=10):
    push = BroadcastOutputPipe(name, send_qsize=send_qsize, mode=mode)
    push.initialize()
    nr_pulls = nr_workers or 1
    pulls = [BroadcastInputPipe(push.conn_info) for _ in range(nr_pulls)]

    if nr_workers is None:
        return push, pulls[0]
    return push, pulls
