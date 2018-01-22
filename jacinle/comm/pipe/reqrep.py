# -*- coding: utf-8 -*-
# File   : reqrep.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 22/01/2018
#
# This file is part of Jacinle.

import zmq
import threading
import queue
import contextlib
import collections

from jacinle.logging import get_logger
from jacinle.utils.meta import notnone_property
from jacinle.utils.registry import CallbackRegistry

from ..packing import dumpb, loadb
from ..zmq_utils import get_addr, bind_to_random_ipc, graceful_close

logger = get_logger(__file__)

__all__ = ['QueryMessage', 'RepPipe', 'ReqPipe']

QueryMessage = collections.namedtuple('QueryMessage', ['identifier', 'payload'])


class RepPipe(object):
    def __init__(self, name, send_qsize=0, mode='ipc'):
        self._name = name
        self._conn_info = None

        self._context_lock = threading.Lock()
        self._context = zmq.Context()
        self._tosock = self._context.socket(zmq.ROUTER)
        self._frsock = self._context.socket(zmq.PULL)
        self._tosock.set_hwm(10)
        self._frsock.set_hwm(10)
        self._dispatcher = CallbackRegistry()

        self._send_queue = queue.Queue(maxsize=send_qsize)
        self._rcv_thread = None
        self._snd_thread = None
        self._mode = mode
        assert mode in ('ipc', 'tcp')

    @property
    def dispatcher(self):
        return self._dispatcher

    @notnone_property
    def conn_info(self):
        return self._conn_info

    def initialize(self):
        self._conn_info = []
        if self._mode == 'tcp':
            port = self._frsock.bind_to_random_port('tcp://*')
            self._conn_info.append('tcp://{}:{}'.format(get_addr(), port))
            port = self._tosock.bind_to_random_port('tcp://*')
            self._conn_info.append('tcp://{}:{}'.format(get_addr(), port))
        elif self._mode == 'ipc':
            self._conn_info.append(bind_to_random_ipc(self._frsock, self._name + '-c2s-'))
            self._conn_info.append(bind_to_random_ipc(self._tosock, self._name + '-s2c-'))

        self._rcv_thread = threading.Thread(target=self.mainloop_recv, daemon=True)
        self._rcv_thread.start()
        self._snd_thread = threading.Thread(target=self.mainloop_send, daemon=True)
        self._snd_thread.start()

    def finalize(self):
        graceful_close(self._tosock)
        graceful_close(self._frsock)
        self._context.term()

    @contextlib.contextmanager
    def activate(self):
        self.initialize()
        try:
            yield
        finally:
            self.finalize()

    def mainloop_recv(self):
        try:
            while True:
                if self._frsock.closed:
                    break

                msg = loadb(self._frsock.recv(copy=False).bytes)
                identifier, type, payload = msg
                self._dispatcher.dispatch(type, self, identifier, payload)
        except zmq.ContextTerminated:
            pass
        except zmq.ZMQError as e:
            if self._tosock.closed:
                logger.warning('Recv socket closed unexpectedly.')
            else:
                raise e

    def mainloop_send(self):
        try:
            while True:
                if self._tosock.closed:
                    break

                job = self._send_queue.get()
                self._tosock.send_multipart([job.identifier, dumpb(job.payload)], copy=False)
        except zmq.ContextTerminated:
            pass
        except zmq.ZMQError as e:
            if self._tosock.closed:
                logger.warning('Send socket closed unexpectedly.')
            else:
                raise e

    def send(self, identifier, msg):
        self._send_queue.put(QueryMessage(identifier, msg))


class ReqPipe(object):
    def __init__(self, name, conn_info):
        self._name = name
        self._conn_info = conn_info
        self._context = None
        self._tosock = None
        self._frsock = None

    @property
    def identity(self):
        return self._name.encode('utf-8')

    def initialize(self):
        self._context = zmq.Context()
        self._tosock = self._context.socket(zmq.PUSH)
        self._frsock = self._context.socket(zmq.DEALER)
        self._tosock.setsockopt(zmq.IDENTITY, self.identity)
        self._frsock.setsockopt(zmq.IDENTITY, self.identity)
        self._tosock.set_hwm(2)
        self._tosock.connect(self._conn_info[0])
        self._frsock.connect(self._conn_info[1])

    def finalize(self):
        graceful_close(self._frsock)
        graceful_close(self._tosock)
        self._context.term()

    @contextlib.contextmanager
    def activate(self):
        self.initialize()
        try:
            yield
        finally:
            self.finalize()

    def query(self, type, inp, do_recv=True):
        self._tosock.send(dumpb((self.identity, type, inp)), copy=False)
        if do_recv:
            out = loadb(self._frsock.recv(copy=False).bytes)
            return out
