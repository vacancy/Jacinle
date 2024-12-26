#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : server.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/19/2017
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from jacinle.cli.argument import JacArgumentParser
from jacinle.comm.service import Service


class MyService(Service):
    def call(self, inp):
        out = inp['a'] + inp['b']
        print('Server string: {} {}.'.format(inp['a'], inp['b']))
        return dict(out=out)


def main():
    parser = JacArgumentParser()
    parser.add_argument('--simple', action='store_true')
    args = parser.parse_args()

    s = MyService()
    s.serve_socket(tcp_port=[31001, 31002], use_simple=args.simple)


if __name__ == '__main__':
    main()
