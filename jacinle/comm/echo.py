#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : echo.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import threading
import contextlib
import sys

from jacinle.utils.printing import PrintToStringContext

__all__ = ['EchoToPipe', 'echo_from_pipe']


class StringQueue(object):
    """Adapted from: http://code.activestate.com/recipes/426060-a-queue-for-string-data-which-looks-like-a-file-ob/"""

    def __init__(self):
        """
        Initialize the buffers.

        Args:
            self: (todo): write your description
        """
        self.l_buffer = []
        self.s_buffer = ""
        self.lock = threading.Lock()

    def write(self, data):
        """
        Writes data to the transport.

        Args:
            self: (todo): write your description
            data: (todo): write your description
        """
        self.l_buffer.append(data)

    def flush(self):
        """
        Flush the cache entries.

        Args:
            self: (todo): write your description
        """
        pass

    def _build_str(self):
        """
        Build a string representation of the buffer.

        Args:
            self: (todo): write your description
        """
        with self.lock:
            new_string = ''.join(self.l_buffer)
            self.s_buffer = ''.join((self.s_buffer, new_string))
            self.l_buffer = []

    def __len__(self):
        """
        Returns the length of the buffer

        Args:
            self: (todo): write your description
        """
        with self.lock:
            return sum(len(i) for i in self.l_buffer) + len(self.s_buffer)

    def read(self, count=None):
        """
        Read up to count of bytes from the stream.

        Args:
            self: (todo): write your description
            count: (int): write your description
        """
        if count is None or count > len(self.s_buffer):
            self._build_str()

        if count is None:
            result, self.s_buffer = self.s_buffer, ''
            return result

        if count > len(self.s_buffer):
            return ''
        else:
            result = self.s_buffer[:count]
            self.s_buffer = self.s_buffer[len(result):]
            return result


class EchoMessage(object):
    def __init__(self, source, message):
        """
        Initialize a new source.

        Args:
            self: (todo): write your description
            source: (str): write your description
            message: (str): write your description
        """
        self.source = source
        self.message = message


class EndEcho(object):
    pass


class EchoToPipe(object):
    def __init__(self, pipe, identifier):
        """
        Initialize a new pipe.

        Args:
            self: (todo): write your description
            pipe: (str): write your description
            identifier: (todo): write your description
        """
        self.pipe = pipe
        self.identifier = identifier
        self.echo_thread = None
        self.stop_event = threading.Event()

        self.out = StringQueue()
        self.err = StringQueue()

        self.out_ctx = PrintToStringContext(target='STDOUT', stream=self.out)
        self.err_ctx = PrintToStringContext(target='STDERR', stream=self.err)

    def echo(self):
        """
        Echo a message.

        Args:
            self: (todo): write your description
        """
        to_close = False
        while True:
            msg = self.out.read()
            if len(msg) > 0:
                self.pipe.send(self.identifier, EchoMessage(1, msg))
            msg = self.err.read()
            if len(msg) > 0:
                self.pipe.send(self.identifier, EchoMessage(2, msg))

            if to_close:
                self.pipe.send(self.identifier, EndEcho())
                break

            if self.stop_event.wait(0.1):
                to_close = True

    def initialize(self):
        """
        Initialize the thread.

        Args:
            self: (todo): write your description
        """
        self.echo_thread = threading.Thread(target=self.echo)
        self.echo_thread.start()

    def finalize(self):
        """
        Finalize the stream.

        Args:
            self: (todo): write your description
        """
        self.stop_event.set()
        self.echo_thread.join()

    @contextlib.contextmanager
    def activate(self):
        """
        A context manager which the context.

        Args:
            self: (todo): write your description
        """
        try:
            self.initialize()
            with self.out_ctx, self.err_ctx:
                yield
        finally:
            self.finalize()


def echo_from_pipe(pipe):
    """
    Echo a message.

    Args:
        pipe: (todo): write your description
    """
    count = 0
    while True:
        msg = pipe.recv()
        if isinstance(msg, EchoMessage):
            count += 1
            if msg.source == 1:
                sys.stdout.write(msg.message)
            elif msg.source == 2:
                sys.stderr.write(msg.message)
        elif isinstance(msg, EndEcho):
            if count:
                sys.stdout.write('\r')
            break
        else:
            raise ValueError('Unknwon echo: {}.'.format(msg))

