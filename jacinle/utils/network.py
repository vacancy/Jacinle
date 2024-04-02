#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : network.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/06/2017
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import socket
import errno

__all__ = ['get_local_addr', 'get_local_addr_v1', 'get_local_addr_v2', 'get_free_port', 'get_free_port_from']


def get_local_addr_v1() -> str:
    """Get the local IP address of the machine. This is the old version of get_local_addr.

    Returns:
        the local IP address.
    """
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return '127.0.0.1'


def get_local_addr_v2() -> str:
    """Get the local IP address of the machine. This is the new version of get_local_addr.

    Returns:
        the local IP address.
    """
    try:
        return _get_local_addr_v2_impl()
    except Exception:
        # fallback to get_local_addrv1
        return get_local_addr_v1()


# http://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib
def _get_local_addr_v2_impl() -> str:
    resolve = [ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1]
    if len(resolve):
        return resolve[0]

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    addr = s.getsockname()[0]
    s.close()
    return addr


def get_local_addr() -> str:
    """Get the local IP address of the machine.

    Returns:
        the local IP address.
    """
    return get_local_addr_v2()


def _check_port_usage(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
    except socket.error as e:
        if e.errno == errno.EADDRINUSE:
            return False
        return False
    s.close()
    return True


def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def get_free_port_from(start_port):
    for port in range(start_port, 65536):
        if _check_port_usage(port):
            return port
    return None

