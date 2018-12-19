#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : service.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import uuid

from jacinle.logging import get_logger
from jacinle.comm.cs import ServerPipe, ClientPipe

logger = get_logger(__file__)

__all__ = ['Service', 'SocketServer', 'SocketClient']


class Service(object):
    def __init__(self, configs=None):
        self.configs = configs

    def serve_socket(self, name=None, spec=None):
        if name is None:
            name = self.__class__.__name__

        return SocketServer(self, name, spec=spec).serve()

    def initialize(self):
        pass

    def call(self, feed_dict):
        raise NotImplementedError()

    def finalize(self):
        pass


class SocketServer(object):
    def __init__(self, service, name, spec):
        self.service = service
        self.name = name
        self.spec = spec

        self.identifier = self.name + '-server-' + uuid.uuid4().hex
        self.server = ServerPipe(self.identifier)
        self.server.dispatcher.register('get_identifier', self.call_get_identifier)
        self.server.dispatcher.register('get_conn_info', self.call_get_conn_info)
        self.server.dispatcher.register('get_spec', self.call_get_spec)
        self.server.dispatcher.register('get_configs', self.call_get_configs)
        self.server.dispatcher.register('query', self.call_query)

    def serve(self):
        with self.server.activate():
            logger.info('Server started.')
            logger.info('  Name:       {}'.format(self.name))
            logger.info('  Identifier: {}'.format(self.identifier))
            logger.info('  Conn info:  {} {}'.format(*self.conn_info))
            while True:
                import time; time.sleep(1)

    @property
    def conn_info(self):
        return self.server.conn_info

    def call_get_identifier(self, pipe, identifier, inp):
        pipe.send(identifier, self.identifier)

    def call_get_conn_info(self, pipe, identifier, inp):
        pipe.send(identifier, self.conn_info)

    def call_get_spec(self, pipe, identifier, inp):
        pipe.send(identifier, self.sepc)

    def call_get_configs(self, pipe, identifier, inp):
        pipe.send(identifier, self.service.configs)

    def call_query(self, pipe, identifier, feed_dict):
        logger.info('Received query from: {}.'.format(identifier))
        output_dict = self.service.call(feed_dict)
        pipe.send(identifier, output_dict)


class SocketClient(object):
    def __init__(self, name, conn_info):
        self.name = name
        self.identifier = self.name + '-client-' + uuid.uuid4().hex
        self.conn_info = conn_info

        self.client = ClientPipe(self.identifier, conn_info=self.conn_info)

    def activate(self):
        logger.info('Client started.')
        logger.info('  Name:       {}'.format(self.name))
        logger.info('  Identifier: {}'.format(self.identifier))
        logger.info('  Conn info:  {}'.format(self.conn_info))
        return self.client.activate()

    def get_server_identifier(self):
        return self.client.query('get_identifier')

    def get_client_identifier(self):
        return self.identifier

    def get_server_conn_info(self):
        return self.client.query('get_conn_info')

    def get_spec(self):
        return self.client.query('get_spec')

    def call(self, feed_dict):
        return self.client.query('query', feed_dict)

