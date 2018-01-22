# -*- coding: utf-8 -*-
# File   : pushpull.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 22/01/2018
#
# This file is part of Jacinle.

import zmq
import threading
import queue
import contextlib

from jacinle.utils.meta import notnone_property

from ..packing import loadb, dumpb
from ..zmq_utils import get_addr, bind_to_random_ipc, graceful_close

__all__ = ['PushPipe', 'PullPipe', 'make_push_pair']


class PullPipe(object):
    def __init__(self, name, mode='tcp'):
        self._name = name
        self._mode = mode
        self._conn_info = None

        self._context = zmq.Context()
        self._sock = self._context.socket(zmq.PULL)
        self._sock.set_hwm(2)

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


class PushPipe(object):
    def __init__(self, conn_info, send_qsize=10):
        self._conn_info = conn_info
        self._send_qsize = send_qsize

        self._context = None
        self._sock = None
        self._send_queue = None
        self._send_thread = None

    def initialize(self):
        self._context = zmq.Context()
        self._sock = self._context.socket(zmq.PUSH)
        self._sock.set_hwm(2)
        self._sock.connect(self._conn_info)
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


def make_push_pair(name, nr_workers=None, mode='tcp', send_qsize=10):
    pull = PullPipe(name, mode=mode)
    pull.initialize()
    nr_pushs = nr_workers or 1
    pushs = [PushPipe(pull.conn_info, send_qsize=send_qsize) for _ in range(nr_pushs)]

    if nr_workers is None:
        return pull, pushs[0]
    return pull, pushs
