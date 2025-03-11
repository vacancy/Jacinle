#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : server.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/11/2025
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from jacinle.comm.service import Service


class MyService(Service):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.endpoints = dict()

    def serve_socket(self):
        super().serve_socket(self.name, use_simple=True, register_name_server=True, verbose=True)

    def register_endpoint(self, name, func):
        self.endpoints[name] = func

    def call(self, name, *args, **kwargs):
        return self.endpoints[name](*args, **kwargs)


def main():
    def add(a, b):
        return a + b

    service = MyService('my-service/add')
    service.register_endpoint('add', add)
    service.serve_socket()


if __name__ == '__main__':
    main()

