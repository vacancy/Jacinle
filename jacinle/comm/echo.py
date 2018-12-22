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
        self.l_buffer = []
        self.s_buffer = ""
        self.lock = threading.Lock()

    def write(self, data):
        self.l_buffer.append(data)

    def flush(self):
        pass

    def _build_str(self):
        with self.lock:
            new_string = ''.join(self.l_buffer)
            self.s_buffer = ''.join((self.s_buffer, new_string))
            self.l_buffer = []

    def __len__(self):
        with self.lock:
            return sum(len(i) for i in self.l_buffer) + len(self.s_buffer)

    def read(self, count=None):
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
        self.source = source
        self.message = message


class EndEcho(object):
    pass


class EchoToPipe(object):
    def __init__(self, pipe, identifier):
        self.pipe = pipe
        self.identifier = identifier
        self.echo_thread = None
        self.stop_event = threading.Event()

        self.out = StringQueue()
        self.err = StringQueue()

        self.out_ctx = PrintToStringContext(target='STDOUT', stream=self.out)
        self.err_ctx = PrintToStringContext(target='STDERR', stream=self.err)

    def echo(self):
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
        self.echo_thread = threading.Thread(target=self.echo)
        self.echo_thread.start()

    def finalize(self):
        self.stop_event.set()
        self.echo_thread.join()

    @contextlib.contextmanager
    def activate(self):
        try:
            self.initialize()
            with self.out_ctx, self.err_ctx:
                yield
        finally:
            self.finalize()


def echo_from_pipe(pipe):
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

