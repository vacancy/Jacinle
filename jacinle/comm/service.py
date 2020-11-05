#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : service.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import sys
import uuid
import inspect
import contextlib

from jacinle.logging import get_logger
from jacinle.utils.printing import kvformat
from jacinle.utils.exception import format_exc

from .cs import ServerPipe, ClientPipe
from .echo import EchoToPipe, echo_from_pipe

logger = get_logger(__file__)

__all__ = ['Service', 'SocketServer', 'SocketClient']


class Service(object):
    def __init__(self, configs=None, spec=None):
        """
        Initialize the configuration.

        Args:
            self: (todo): write your description
            configs: (dict): write your description
            spec: (todo): write your description
        """
        self.configs = configs
        self.spec = spec

    def serve_socket(self, name=None, tcp_port=None):
        """
        Starts a socket.

        Args:
            self: (todo): write your description
            name: (str): write your description
            tcp_port: (int): write your description
        """
        if name is None:
            name = self.__class__.__name__

        return SocketServer(self, name, tcp_port=tcp_port).serve()

    def initialize(self):
        """
        Initialize the next callable object

        Args:
            self: (todo): write your description
        """
        pass

    def call(self, *args, **kwargs):
        """
        Calls the given arguments.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError()

    def finalize(self):
        """
        Finalizes the finalize.

        Args:
            self: (todo): write your description
        """
        pass


class ServiceException(object):
    def __init__(self, remote_message):
        """
        Initialize a new message.

        Args:
            self: (todo): write your description
            remote_message: (todo): write your description
        """
        self.remote_message = remote_message

    def __repr__(self):
        """
        Return a human - readable representation of this object.

        Args:
            self: (todo): write your description
        """
        return 'Service exception: ' + self.remote_message


class SocketServer(object):
    def __init__(self, service, name, tcp_port=None):
        """
        Initialize a service.

        Args:
            self: (todo): write your description
            service: (todo): write your description
            name: (str): write your description
            tcp_port: (int): write your description
        """
        self.service = service
        self.name = name
        self.tcp_port = tcp_port

        self.identifier = self.name + '-server-' + uuid.uuid4().hex
        self.server = ServerPipe(self.identifier)
        self.server.dispatcher.register('get_name', self.call_get_name)
        self.server.dispatcher.register('get_identifier', self.call_get_identifier)
        self.server.dispatcher.register('get_conn_info', self.call_get_conn_info)
        self.server.dispatcher.register('get_spec', self.call_get_spec)
        self.server.dispatcher.register('get_configs', self.call_get_configs)
        self.server.dispatcher.register('get_signature', self.call_get_signature)
        self.server.dispatcher.register('query', self.call_query)

    def serve(self):
        """
        Starts a server.

        Args:
            self: (todo): write your description
        """
        with self.server.activate(tcp_port=self.tcp_port):
            logger.info('Server started.')
            logger.info('  Name:       {}'.format(self.name))
            logger.info('  Identifier: {}'.format(self.identifier))
            logger.info('  Conn info:  {} {}'.format(*self.conn_info))
            while True:
                import time; time.sleep(1)

    @property
    def conn_info(self):
        """
        Return the connection info.

        Args:
            self: (todo): write your description
        """
        return self.server.conn_info

    def call_get_name(self, pipe, identifier, inp):
        """
        Call the given pipe.

        Args:
            self: (todo): write your description
            pipe: (todo): write your description
            identifier: (str): write your description
            inp: (int): write your description
        """
        pipe.send(identifier, self.name)

    def call_get_identifier(self, pipe, identifier, inp):
        """
        Calls a pipe.

        Args:
            self: (todo): write your description
            pipe: (str): write your description
            identifier: (str): write your description
            inp: (str): write your description
        """
        pipe.send(identifier, self.identifier)

    def call_get_conn_info(self, pipe, identifier, inp):
        """
        Get a connection info.

        Args:
            self: (todo): write your description
            pipe: (str): write your description
            identifier: (str): write your description
            inp: (todo): write your description
        """
        pipe.send(identifier, self.conn_info)

    def call_get_spec(self, pipe, identifier, inp):
        """
        Calls the given service.

        Args:
            self: (todo): write your description
            pipe: (str): write your description
            identifier: (str): write your description
            inp: (str): write your description
        """
        pipe.send(identifier, self.service.sepc)

    def call_get_configs(self, pipe, identifier, inp):
        """
        Call a list of service configurations.

        Args:
            self: (todo): write your description
            pipe: (str): write your description
            identifier: (str): write your description
            inp: (str): write your description
        """
        pipe.send(identifier, self.service.configs)

    def call_get_signature(self, pipe, identifier, inp):
        """
        Call a call signature.

        Args:
            self: (todo): write your description
            pipe: (todo): write your description
            identifier: (str): write your description
            inp: (todo): write your description
        """
        pipe.send(identifier, repr(inspect.getfullargspec(self.service.call)))

    def call_query(self, pipe, identifier, feed_dict):
        """
        Send a query to the service.

        Args:
            self: (todo): write your description
            pipe: (todo): write your description
            identifier: (str): write your description
            feed_dict: (dict): write your description
        """
        logger.info('Received query from: {}.'.format(identifier))
        try:
            if feed_dict['echo']:
                with EchoToPipe(pipe, identifier).activate():
                    output_dict = self.service.call(*feed_dict['args'], **feed_dict['kwargs'])
            else:
                output_dict = self.service.call(*feed_dict['args'], **feed_dict['kwargs'])
        except:
            output_dict = ServiceException(format_exc(sys.exc_info()))
        pipe.send(identifier, output_dict)


class SocketClient(object):
    def __init__(self, name, conn_info):
        """
        Initialize a client.

        Args:
            self: (todo): write your description
            name: (str): write your description
            conn_info: (todo): write your description
        """
        self.name = name
        self.identifier = self.name + '-client-' + uuid.uuid4().hex
        self.conn_info = conn_info

        self.client = ClientPipe(self.identifier, conn_info=self.conn_info)

    def initialize(self):
        """
        Initialize the client.

        Args:
            self: (todo): write your description
        """
        self.client.initialize()
        logger.info('Client started.')
        logger.info('  Name:              {}'.format(self.name))
        logger.info('  Identifier:        {}'.format(self.identifier))
        logger.info('  Conn info:         {}'.format(self.conn_info))
        logger.info('  Server name:       {}'.format(self.get_server_name()))
        logger.info('  Server identifier: {}'.format(self.get_server_identifier()))
        logger.info('  Server signaature: {}'.format(self.get_signature()))
        configs = self.get_configs()
        if configs is not None:
            logger.info('  Server configs:\n' + kvformat(configs, indent=2))

    def finalize(self):
        """
        Finalize the client.

        Args:
            self: (todo): write your description
        """
        self.client.finalize()

    @contextlib.contextmanager
    def activate(self):
        """
        A context manager to a new context.

        Args:
            self: (todo): write your description
        """
        try:
            self.initialize()
            yield
        finally:
            self.finalize()

    def get_server_name(self):
        """
        Get the server name.

        Args:
            self: (todo): write your description
        """
        return self.client.query('get_name')

    def get_server_identifier(self):
        """
        Gets the server identifier.

        Args:
            self: (todo): write your description
        """
        return self.client.query('get_identifier')

    def get_client_identifier(self):
        """
        Gets : attribute

        Args:
            self: (todo): write your description
        """
        return self.identifier

    def get_server_conn_info(self):
        """
        Return the server info.

        Args:
            self: (todo): write your description
        """
        return self.client.query('get_conn_info')

    def get_spec(self):
        """
        Return the spec for the spec.

        Args:
            self: (todo): write your description
        """
        return self.client.query('get_spec')

    def get_configs(self):
        """
        Get the configs.

        Args:
            self: (todo): write your description
        """
        return self.client.query('get_configs')

    def get_signature(self):
        """
        Returns the signature.

        Args:
            self: (todo): write your description
        """
        return self.client.query('get_signature')

    def call(self, *args, echo=True, **kwargs):
        """
        Calls a remote server

        Args:
            self: (todo): write your description
            echo: (todo): write your description
        """
        self.client.query('query', {'args': args, 'kwargs': kwargs, 'echo': echo}, do_recv=False)
        if echo:
            echo_from_pipe(self.client)
        output = self.client.recv()
        if isinstance(output, ServiceException):
            raise RuntimeError(repr(output))
        return output

