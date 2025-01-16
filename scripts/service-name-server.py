#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : service-name-server.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2017
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import argparse

from jacinle.comm.service_name_server import SimpleNameServer
from jacinle.logging import get_logger

logger = get_logger(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--port', dest='port', default=SimpleNameServer.DEFAULT_PORT)
args = parser.parse_args()


if __name__ == '__main__':
    logger.critical('Starting name server at {}.'.format(args.port))
    SimpleNameServer().serve_socket(tcp_port=args.port)

