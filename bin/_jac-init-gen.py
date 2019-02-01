#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : _jac-init-gen.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/31/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import os
import os.path as osp
import sys
import tempfile
import jacinle.io as io
from jacinle.logging import get_logger

logger = get_logger(__file__)


def load_vendors(config, f):
    if 'vendors' not in config:
        return

    for k, v in config['vendors'].items():
        assert 'root' in v, '"root" not found in vendor: {}.'.format(k)

        logger.info('Loading vendor: {}.'.format(k))
        print('export PYTHONPATH={}:$PYTHONPATH'.format(v['root']), file=f)


def main():
    f = tempfile.NamedTemporaryFile('w', delete=False)

    wd = os.getcwd()
    yml_filename = osp.join(wd, 'jacinle.yml')
    if osp.isfile(osp.join(wd, 'jacinle.yml')):
        logger.critical('Loading jacinle config: {}.'.format(osp.abspath(yml_filename)))
        config = io.load(yml_filename)
        load_vendors(config, f)

    f.close()
    print(f.name)


if __name__ == '__main__':
    main()

