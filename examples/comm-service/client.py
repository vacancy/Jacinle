#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : client.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/19/2017
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import time

from jacinle.comm.service import SocketClient


def main():
    client = SocketClient('client', ['tcp://127.0.0.1:31001', 'tcp://127.0.0.1:31002'])
    with client.activate():
        inp = dict(a=1, b=2)
        out = client.call(inp)
        print('Success: input={}, output={}'.format(inp, out))
        time.sleep(1)


if __name__ == '__main__':
    main()
