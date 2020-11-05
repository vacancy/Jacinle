#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cs.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/22/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import collections
import contextlib
import queue
import threading

import zmq

from jacinle.concurrency.packing import dumpb, loadb
from jacinle.concurrency.zmq_utils import get_addr, bind_to_random_ipc, graceful_close
from jacinle.logging import get_logger
from jacinle.utils.meta import notnone_property
from jacinle.utils.registry import CallbackRegistry

logger = get_logger(__file__)

__all__ = ['ServerPipe', 'ClientPipe', 'make_cs_pair']

_QueryMessage = collections.namedtuple('QueryMessage', ['identifier', 'payload'])


class ServerPipe(object):
    def __init__(self, name, send_qsize=0, mode='tcp'):
        """
        Initialize a thread.

        Args:
            self: (todo): write your description
            name: (str): write your description
            send_qsize: (int): write your description
            mode: (todo): write your description
        """
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
        """
        Dispatcher.

        Args:
            self: (todo): write your description
        """
        return self._dispatcher

    @notnone_property
    def conn_info(self):
        """
        Return the connection info.

        Args:
            self: (todo): write your description
        """
        return self._conn_info

    def initialize(self, tcp_port=None):
        """
        Initialize the tcp connection.

        Args:
            self: (todo): write your description
            tcp_port: (array): write your description
        """
        self._conn_info = []
        if self._mode == 'tcp':
            if tcp_port is not None:
                port = tcp_port[0]
                self._frsock.bind('tcp://*:{}'.format(port))
            else:
                port = self._frsock.bind_to_random_port('tcp://*')
            self._conn_info.append('tcp://{}:{}'.format(get_addr(), port))
            if tcp_port is not None:
                port = tcp_port[1]
                self._tosock.bind('tcp://*:{}'.format(port))
            else:
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
        """
        Finalize the socket.

        Args:
            self: (todo): write your description
        """
        graceful_close(self._tosock)
        graceful_close(self._frsock)
        self._context.term()

    @contextlib.contextmanager
    def activate(self, tcp_port=None):
        """
        A context manager.

        Args:
            self: (todo): write your description
            tcp_port: (int): write your description
        """
        self.initialize(tcp_port=tcp_port)
        try:
            yield
        finally:
            self.finalize()

    def mainloop_recv(self):
        """
        The main loop.

        Args:
            self: (todo): write your description
        """
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
        """
        The main loop.

        Args:
            self: (todo): write your description
        """
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
        """
        Send a message to the queue.

        Args:
            self: (todo): write your description
            identifier: (todo): write your description
            msg: (str): write your description
        """
        self._send_queue.put(_QueryMessage(identifier, msg))


class ClientPipe(object):
    def __init__(self, name, conn_info):
        """
        Initialize a connection.

        Args:
            self: (todo): write your description
            name: (str): write your description
            conn_info: (todo): write your description
        """
        self._name = name
        self._conn_info = conn_info
        self._context = None
        self._tosock = None
        self._frsock = None

    @property
    def identity(self):
        """
        Return the identity.

        Args:
            self: (todo): write your description
        """
        return self._name.encode('utf-8')

    def initialize(self):
        """
        Initialize the socket.

        Args:
            self: (todo): write your description
        """
        self._context = zmq.Context()
        self._tosock = self._context.socket(zmq.PUSH)
        self._frsock = self._context.socket(zmq.DEALER)
        self._tosock.setsockopt(zmq.IDENTITY, self.identity)
        self._frsock.setsockopt(zmq.IDENTITY, self.identity)
        self._tosock.set_hwm(2)
        self._tosock.connect(self._conn_info[0])
        self._frsock.connect(self._conn_info[1])

    def finalize(self):
        """
        Finalize the socket.

        Args:
            self: (todo): write your description
        """
        graceful_close(self._frsock)
        graceful_close(self._tosock)
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

    def query(self, type, inp=None, do_recv=True):
        """
        Send a query to the server.

        Args:
            self: (todo): write your description
            type: (str): write your description
            inp: (todo): write your description
            do_recv: (todo): write your description
        """
        self._tosock.send(dumpb((self.identity, type, inp)), copy=False)
        if do_recv:
            return self.recv()

    def recv(self):
        """
        Receive a message from the socket.

        Args:
            self: (todo): write your description
        """
        out = loadb(self._frsock.recv(copy=False).bytes)
        return out


def make_cs_pair(name, nr_clients=None, mode='tcp', send_qsize=10):
    """
    Make a csfs pair.

    Args:
        name: (str): write your description
        nr_clients: (todo): write your description
        mode: (todo): write your description
        send_qsize: (int): write your description
    """
    rep = ServerPipe(name + '-rep', mode=mode, send_qsize=send_qsize)
    rep.initialize()
    nr_reqs = nr_clients or 1
    reqs = [ClientPipe(name + '-req-' + str(i), rep.conn_info) for i in range(nr_reqs)]

    if nr_clients is None:
        return rep, reqs[0]
    return rep, reqs
