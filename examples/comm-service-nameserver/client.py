#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : client.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/11/2025
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from jacinle.comm.service import SocketClient


def main():
    client = SocketClient('my-service/add', use_name_server=True, use_simple=True, verbose=False)
    client.initialize()

    print(client.add(1, 2))


if __name__ == '__main__':
    main()

