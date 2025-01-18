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
import atexit
from typing import List

from jacinle.logging import get_logger
from jacinle.utils.printing import kvformat
from jacinle.utils.exception import format_exc

from .cs import ServerPipe, ClientPipe
from .cs_simple import SimpleServerPipe, SimpleClientPipe
from .echo import EchoToPipe, echo_from_pipe

logger = get_logger(__file__)

__all__ = ['Service', 'SocketServer', 'SocketClient']


class Service(object):
    def __init__(self, configs=None, spec=None):
        self.configs = configs
        self.spec = spec

    def serve_socket(self, name=None, tcp_port=None, use_simple=False, register_name_server=False, verbose=True):
        if name is None:
            name = self.__class__.__name__

        return SocketServer(self, name, tcp_port=tcp_port, use_simple=use_simple, register_name_server=register_name_server, verbose=verbose).serve()

    def initialize(self):
        pass

    def call(self, *args, **kwargs):
        raise NotImplementedError()

    def finalize(self):
        pass


class ServiceException(object):
    def __init__(self, remote_message):
        self.remote_message = remote_message

    def __repr__(self):
        return 'Service exception: ' + self.remote_message


class ServiceGeneratorStart(object):
    def __init__(self, token):
        self.token = token


class ServiceGeneratorEnd(object):
    pass


class SocketServer(object):
    def __init__(self, service, name, tcp_port=None, ipc_port=None, use_simple: bool = False, register_name_server: bool = False, verbose: bool = True):
        self.service = service
        self.name = name
        self.tcp_port = tcp_port
        self.ipc_port = ipc_port
        self.mode = 'tcp'
        if self.ipc_port is not None:
            self.mode = 'ipc'
        self.verbose = verbose

        self.use_simple = use_simple
        self.register_name_server = register_name_server
        self.identifier = self.name + '-server-' + uuid.uuid4().hex
        if use_simple:
            self.server = SimpleServerPipe(self.identifier, mode=self.mode)
        else:
            self.server = ServerPipe(self.identifier, mode=self.mode)
        self.server.dispatcher.register('get_name', self.call_get_name)
        self.server.dispatcher.register('get_identifier', self.call_get_identifier)
        self.server.dispatcher.register('get_conn_info', self.call_get_conn_info)
        self.server.dispatcher.register('get_spec', self.call_get_spec)
        self.server.dispatcher.register('get_configs', self.call_get_configs)
        self.server.dispatcher.register('get_signature', self.call_get_signature)
        self.server.dispatcher.register('query', self.call_query)
        self.server.dispatcher.register('generator_init', self.call_generator_init)
        self.server.dispatcher.register('generator_next', self.call_generator_next)
        self.server.dispatcher.register('generator_deinit', self.call_generator_deinit)

        self.generator_states = dict()

    def serve(self):
        with self.server.activate(tcp_port=self.tcp_port, ipc_port=self.ipc_port):
            if self.verbose:
                print('Server started.')
                print('  Name:       {}'.format(self.name))
                print('  Identifier: {}'.format(self.identifier))
                print('  Conn info:  {}'.format(self.conn_info))

            if self.register_name_server:
                from jacinle.comm.service_name_server import sns_register
                sns_register(self.name, {'conn_info': self.conn_info})

            if self.use_simple:
                self.server.serve_forever()
            else:
                while True:
                    import time; time.sleep(1)

    @contextlib.contextmanager
    def activate(self):
        with self.server.activate(tcp_port=self.tcp_port, ipc_port=self.ipc_port):
            logger.info('Server started.')
            logger.info('  Name:       {}'.format(self.name))
            logger.info('  Identifier: {}'.format(self.identifier))
            logger.info('  Conn info:  {}'.format(self.conn_info))
            yield

    @property
    def conn_info(self) -> List[str]:
        return self.server.conn_info

    def call_get_use_simple(self, pipe, identifier, inp):
        pipe.send(identifier, self.use_simple)

    def call_get_name(self, pipe, identifier, inp):
        pipe.send(identifier, self.name)

    def call_get_identifier(self, pipe, identifier, inp):
        pipe.send(identifier, self.identifier)

    def call_get_conn_info(self, pipe, identifier, inp):
        pipe.send(identifier, self.conn_info)

    def call_get_spec(self, pipe, identifier, inp):
        pipe.send(identifier, self.service.spec)

    def call_get_configs(self, pipe, identifier, inp):
        pipe.send(identifier, self.service.configs)

    def call_get_signature(self, pipe, identifier, inp):
        pipe.send(identifier, repr(inspect.getfullargspec(self.service.call)))

    def call_query(self, pipe, identifier, feed_dict):
        if self.verbose:
            print('Received query from: {}.'.format(identifier))
        try:
            if feed_dict['echo']:
                with EchoToPipe(pipe, identifier).activate():
                    output_dict = self.service.call(*feed_dict['args'], **feed_dict['kwargs'])
            else:
                output_dict = self.service.call(*feed_dict['args'], **feed_dict['kwargs'])
        except:
            output_dict = ServiceException(format_exc(sys.exc_info()))
        pipe.send(identifier, output_dict)

    def call_generator_init(self, pipe, identifier, feed_dict):
        logger.info('Received generator_init from: {}.'.format(identifier))
        token = uuid.uuid4().hex
        try:
            if feed_dict['echo']:
                with EchoToPipe(pipe, identifier).activate():
                    generator = self.service.call(*feed_dict['args'], **feed_dict['kwargs'])
                    self.generator_states[token] = {
                        'generator': generator,
                        'echo': True
                    }
                pipe.send(identifier, ServiceGeneratorStart(token))
            else:
                generator = self.service.call(*feed_dict['args'], **feed_dict['kwargs'])
                self.generator_states[token] = {
                    'generator': generator,
                    'echo': False
                }
                pipe.send(identifier, ServiceGeneratorStart(token))
        except:
            output_dict = ServiceException(format_exc(sys.exc_info()))
            pipe.send(identifier, output_dict)

    def call_generator_next(self, pipe, identifier, feed_dict):
        logger.info('Received generator_next from: {}.'.format(identifier))
        token = feed_dict['token']
        state = self.generator_states[token]
        try:
            if state['echo']:
                with EchoToPipe(pipe, identifier).activate():
                    output = next(state['generator'])
            else:
                output = next(state['generator'])
            pipe.send(identifier, output)
        except StopIteration:
            pipe.send(identifier, ServiceGeneratorEnd())
        except:
            output_dict = ServiceException(format_exc(sys.exc_info()))
            pipe.send(identifier, output_dict)

    def call_generator_deinit(self, pipe, identifier, feed_dict):
        logger.info('Received generator_deinit from: {}.'.format(identifier))
        token = feed_dict['token']
        state = self.generator_states[token]
        try:
            if state['echo']:
                with EchoToPipe(pipe, identifier).activate():
                    state['generator'].close()
            else:
                state['generator'].close()
            pipe.send(identifier, ServiceGeneratorEnd())
        except:
            output_dict = ServiceException(format_exc(sys.exc_info()))
            pipe.send(identifier, output_dict)


class SocketClient(object):
    def __init__(self, name, conn_info=None, echo=True, use_simple=False, use_name_server: bool = False, verbose: bool = True):
        self.name = name
        self.identifier = self.name + '-client-' + uuid.uuid4().hex
        self.verbose = verbose

        if use_name_server:
            if conn_info is None:
                from jacinle.comm.service_name_server import sns_get
                sns_info = sns_get(name)
                if sns_info is None:
                    raise ValueError('Name server does not have the information of the service: {}.'.format(name))
                conn_info = sns_info['conn_info']
            else:
                logger.warning('conn_info is provided, ignoring the name server.')
        else:
            if conn_info is None:
                raise ValueError('conn_info must be provided if not using name server.')

        self.conn_info = conn_info

        self.use_simple = use_simple
        self.echo = not use_simple and echo

        if self.use_simple:
            self.client = SimpleClientPipe(self.identifier, conn_info=self.conn_info)
        else:
            self.client = ClientPipe(self.identifier, conn_info=self.conn_info)
        self._initialized = False

    def initialize(self, auto_close=False, timeout=None):
        success = self.client.initialize(timeout)

        if not success:
            self.client.finalize()
            return False

        if self.verbose:
            logger.info('Client started.')
            logger.info('  Name:              {}'.format(self.name))
            logger.info('  Identifier:        {}'.format(self.identifier))
            logger.info('  Conn info:         {}'.format(self.conn_info))
            logger.info('  Server name:       {}'.format(self.get_server_name()))
            logger.info('  Server identifier: {}'.format(self.get_server_identifier()))
            logger.info('  Server signature:  {}'.format(self.get_signature()))
            configs = self.get_configs()
            if configs is not None:
                logger.info('  Server configs: {}'.format(configs))
        self._initialized = True

        if auto_close:
            atexit.register(self.finalize)

        return True

    def finalize(self):
        self.client.finalize()
        self._initialized = False

    @property
    def initialized(self):
        return self._initialized

    @contextlib.contextmanager
    def activate(self):
        try:
            self.initialize()
            yield
        finally:
            self.finalize()

    def get_use_simple(self):
        return self.client.query('get_use_simple')

    def get_server_name(self):
        return self.client.query('get_name')

    def get_server_identifier(self):
        return self.client.query('get_identifier')

    def get_client_identifier(self):
        return self.identifier

    def get_server_conn_info(self):
        return self.client.query('get_conn_info')

    def get_spec(self):
        return self.client.query('get_spec')

    def get_configs(self):
        return self.client.query('get_configs')

    def get_signature(self):
        return self.client.query('get_signature')

    def call(self, *args, echo=None, **kwargs):
        if echo is None:
            echo = self.echo
        if echo and self.use_simple:
            raise RuntimeError('Echo is not supported in simple mode.')
        self.client.query('query', {'args': args, 'kwargs': kwargs, 'echo': echo}, do_recv=False)
        if echo:
            echo_from_pipe(self.client)
        output = self.client.recv()
        if isinstance(output, ServiceException):
            raise RuntimeError(repr(output))
        return output

    def call_generator(self, *args, echo=False, **kwargs):
        if echo is None:
            echo = self.echo
        if echo and self.use_simple:
            raise RuntimeError('Echo is not supported in simple mode.')

        self.client.query('generator_init', {'args': args, 'kwargs': kwargs, 'echo': echo}, do_recv=False)
        if echo:
            echo_from_pipe(self.client)
        output = self.client.recv()
        if isinstance(output, ServiceException):
            raise RuntimeError(repr(output))
        if not isinstance(output, ServiceGeneratorStart):
            raise RuntimeError('Invalid generator start token.')

        token = output.token
        try:
            while True:
                self.client.query('generator_next', {'token': token}, do_recv=False)
                if echo:
                    echo_from_pipe(self.client)
                output = self.client.recv()
                if isinstance(output, ServiceException):
                    raise RuntimeError(repr(output))
                if isinstance(output, ServiceGeneratorEnd):
                    break
                yield output
        finally:
            self.client.query('generator_deinit', {'token': token}, do_recv=False)
            if echo:
                echo_from_pipe(self.client)
            self.client.recv()

    def __getattr__(self, name):
        def _call(*args, **kwargs):
            return self.call(name, *args, **kwargs)
        return _call

