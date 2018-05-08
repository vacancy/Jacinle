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

__all__ = ['get_local_addr', 'get_local_addr_v1', 'get_local_addr_v2']


def get_local_addr_v1():
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return '127.0.0.1'


def get_local_addr_v2():
    try:
        return _get_local_addr_v2_impl()
    except Exception:
        # fallback to get_local_addrv1
        return get_local_addr_v1()


# http://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib
def _get_local_addr_v2_impl():
    resolve = [ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1]
    if len(resolve):
        return resolve[0]

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    addr = s.getsockname()[0]
    s.close()
    return addr


get_local_addr = get_local_addr_v2
