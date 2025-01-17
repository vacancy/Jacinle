#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : service_name_server.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/16/2025
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import time
import threading

from jacinle.utils.env import jac_getenv
from jacinle.comm.service import Service, SocketClient

__all__ = ['SimpleNameServer', 'SimpleNameServerClient', 'sns_register', 'sns_get', 'sns_has']


class SimpleNameServer(Service):
    DEFAULT_PORT = 11103

    def __init__(self):
        super().__init__()
        self.mutex = threading.Lock()
        self.name2objects = dict()
        self.name2last_heartbeat = dict()

    def heartbeat(self, name, obj):
        print('Heartbeat from {}.'.format(name))
        with self.mutex:
            self.name2objects[name] = obj
            self.name2last_heartbeat[name] = time.time()

    def get(self, name):
        with self.mutex:
            return self.name2objects.get(name, None)

    def has(self, name):
        with self.mutex:
            return name in self.name2objects

    def cleaner(self):
        while True:
            time.sleep(30)
            with self.mutex:
                now = time.time()
                for name, last_heartbeat in list(self.name2last_heartbeat.items()):
                    if now - last_heartbeat > 60:
                        print('Removing {} due to timeout.'.format(name))
                        del self.name2objects[name]
                        del self.name2last_heartbeat[name]

    def call(self, func_name, *args, **kwargs):
        if func_name == 'heartbeat':
            return self.heartbeat(*args, **kwargs)
        elif func_name == 'get':
            return self.get(*args, **kwargs)
        elif func_name == 'has':
            return self.has(*args, **kwargs)
        else:
            raise NotImplementedError('Unknown function name: {}.'.format(func_name))

    def serve_socket(self, name='jacinle/nameserver', tcp_port=DEFAULT_PORT, use_simple=True):
        threading.Thread(target=self.cleaner, args=tuple(), daemon=True).start()
        super().serve_socket(name, tcp_port, use_simple=True)


class SimpleNameServerClient(SocketClient):
    def __init__(self, host='localhost', port=11103):
        conn_info = 'tcp://{}:{}'.format(host, port)
        super().__init__('jacinle/nameserver::client', conn_info, use_simple=True)
        self.initialize(auto_close=True)

    def heartbeat(self, name, obj):
        self.call('heartbeat', name, obj)

    def register(self, name, obj):
        def thread():
            while True:
                self.heartbeat(name, obj)
                time.sleep(10)

        threading.Thread(target=thread, args=tuple(), daemon=True).start()

    def get(self, name):
        return self.call('get', name)

    def has(self, name):
        return self.call('has', name)


_default_name_server_client = None


def get_default_name_server_client():
    global _default_name_server_client

    if _default_name_server_client is None:
        host = jac_getenv('SNS_HOST', 'localhost')
        port = jac_getenv('SNS_PORT', SimpleNameServer.DEFAULT_PORT)
        _default_name_server_client = SimpleNameServerClient(host, port)

    return _default_name_server_client


def sns_register(name, obj):
    get_default_name_server_client().register(name, obj)


def sns_get(name):
    return get_default_name_server_client().get(name)


def sns_has(name):
    return get_default_name_server_client().has(name)

