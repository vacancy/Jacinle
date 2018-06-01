#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : zmq_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/22/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import zmq
import socket
import uuid
import json

from jacinle.utils.network import get_local_addr_v2
from jacinle.concurrency.packing import loadb, dumpb


json_dumpb = lambda x: json.dumps(x).encode('utf-8')
json_loadb = lambda x: json.loads(x.decode('utf-8'))

get_addr = get_local_addr_v2


def router_recv_json(sock, flag=zmq.NOBLOCK, loader=json_loadb):
    try:
        identifier, delim, *payload = sock.recv_multipart(flag)
        return [identifier] + list(map(lambda x: loader(x), payload))
    except zmq.error.ZMQError:
        return None, None


def router_send_json(sock, identifier, *payloads, flag=0, dumper=json_dumpb):
    try:
        buf = [identifier, b'']
        buf.extend(map(lambda x: dumper(x), payloads))
        sock.send_multipart(buf, flags=flag)
    except zmq.error.ZMQError:
        return False
    return True


def req_recv_json(sock, flag=0, loader=json_loadb):
    try:
        response = sock.recv_multipart(flag)
        response = list(map(lambda x: loader(x), response))
        return response[0] if len(response) == 1 else response
    except zmq.error.ZMQError:
        return None


def req_send_json(sock, *payloads, flag=0, dumper=json_dumpb):
    buf = []
    buf.extend(map(lambda x: dumper(x), payloads))
    try:
        sock.send_multipart(buf, flag)
    except zmq.error.ZMQError:
        return False
    return True


def iter_recv(meth, sock):
    while True:
        res = meth(sock, flag=zmq.NOBLOCK)
        succ = res[0] is not None if isinstance(res, (tuple, list)) else res is not None
        if succ:
            yield res
        else:
            break


def req_send_and_recv(sock, *payloads):
    req_send_json(sock, *payloads)
    return req_recv_json(sock)


def push_pyobj(sock, data, flag=zmq.NOBLOCK):
    try:
        sock.send(dumpb(data), flag, copy=False)
    except zmq.error.ZMQError:
        return False
    return True


def pull_pyobj(sock, flag=zmq.NOBLOCK):
    try:
        response = loadb(sock.recv(flag, copy=False).bytes)
        return response
    except zmq.error.ZMQError:
        return None


def bind_to_random_ipc(sock, name):
    name = name + uuid.uuid4().hex[:8]
    conn = 'ipc:///tmp/{}'.format(name)
    sock.bind(conn)
    return conn


def uid():
    return socket.gethostname() + '/' + uuid.uuid4().hex


def graceful_close(sock):
    if sock is None:
        return
    sock.setsockopt(zmq.LINGER, 0)
    sock.close()
