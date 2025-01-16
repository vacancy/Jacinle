#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cs_simple.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/26/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Simplified communication module for client-server communication using REP-REQ pattern."""

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


class SimpleServerPipe(object):
    def __init__(self, name: str, mode: str = 'tcp'):
        assert mode in ('ipc', 'tcp')
        self._name = name
        self._conn_info = None
        self._mode = mode

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._message_counter = 0
        self._dispatcher = CallbackRegistry()

    @property
    def dispatcher(self):
        return self._dispatcher

    @property
    def conn_info(self):
        assert self._conn_info is not None, 'The pipe is not initialized yet.'
        return self._conn_info

    def initialize(self, tcp_port=None, ipc_port=None):
        if self._mode == 'tcp':
            if tcp_port is None:
                port = self._socket.bind_to_random_port('tcp://*')
            else:
                if isinstance(tcp_port, (int, str)):
                    tcp_port = tcp_port
                elif isinstance(tcp_port, (tuple, list)):
                    tcp_port = tcp_port[0]
                else:
                    raise ValueError('Invalid tcp_port: {}.'.format(tcp_port))
                self._socket.bind('tcp://*:{}'.format(tcp_port))
                port = tcp_port
            self._conn_info = 'tcp://{}:{}'.format(get_addr(), port)
            print('ServerPipe initialized: {}.'.format(self._conn_info))
        else:
            if ipc_port is None:
                ipc_port = bind_to_random_ipc(self._socket, self._name)
            else:
                if len(ipc_port) == 2:
                    ipc_port = ipc_port[0]
                self._socket.bind('ipc://{}'.format(ipc_port))
            self._conn_info = 'ipc://{}'.format(ipc_port)

        logger.info('ServerPipe initialized: {}.'.format(self._conn_info))

    def finalize(self):
        graceful_close(self._socket)
        self._context.term()

    @contextlib.contextmanager
    def activate(self, tcp_port=None, ipc_port=None):
        self.initialize(tcp_port, ipc_port)
        try:
            yield
        finally:
            self.finalize()

    def serve_forever(self):
        while True:
            type, message = loadb(self._socket.recv())
            self._message_counter += 1
            self._dispatcher.dispatch(type, self, self._message_counter, message)

    def send(self, identifier, response):
        assert self._message_counter == identifier
        self._message_counter += 1
        self._socket.send(dumpb(response))


class SimpleClientPipe(object):
    def __init__(self, name: str, conn_info):
        self._name = name
        self._conn_info = conn_info
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._initialized = False

        if isinstance(self._conn_info, (tuple, list)):
            self._conn_info = conn_info[0]

    def initialize(self, timeout=None):
        rv = self._socket.connect(self._conn_info)
        self._initialized = True
        return True

    def finalize(self):
        if self._initialized:
            graceful_close(self._socket)
            self._context.term()
            self._initialized = False

    @contextlib.contextmanager
    def activate(self, timeout=None):
        self.initialize()
        try:
            yield
        finally:
            self.finalize()

    def query(self, type, message=None, do_recv=True):
        self._socket.send(dumpb((type, message)))
        if do_recv:
            return self.recv()

    def recv(self):
        return loadb(self._socket.recv())

