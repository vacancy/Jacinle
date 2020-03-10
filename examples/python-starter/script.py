#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : script.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/25/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import six
import os.path as osp

import jacinle.io as io

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger
from jacinle.utils.tqdm import tqdm, tqdm_gofor, get_current_tqdm

logger = get_logger(__file__)

parser = JacArgumentParser()
args = parser.parse_args()


def main():
    from IPython import embed; embed()


if __name__ == '__main__':
    main()

