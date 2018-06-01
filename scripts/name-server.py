#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : name-server.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2017
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import argparse

from jacinle.comm.name_server import _configs as configs
from jacinle.comm.name_server import run_name_server
from jacinle.logging import get_logger

logger = get_logger(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--host', dest='host', default=configs.NS_CTL_HOST)
parser.add_argument('-p', '--port', dest='port', default=configs.NS_CTL_PORT)
parser.add_argument('--protocal', dest='protocal', default=configs.NS_CTL_PROTOCAL)
args = parser.parse_args()


if __name__ == '__main__':
    logger.critical('Starting name server at {}://{}:{}.'.format(args.protocal, args.host, args.port))
    run_name_server(host=args.host, port=args.port, protocal=args.protocal)
