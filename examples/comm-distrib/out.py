#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : out.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/03/2017
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import time

import numpy as np

from jacinle.comm.distrib import DistribOutputPipe, control


def main():
    q = DistribOutputPipe('jacinle.test')
    with control(pipes=[q]):
        while True:
            data = {'msg': 'hello', 'current': time.time(), 'data': np.zeros(shape=(128, 224, 224, 3), dtype='float32')}
            print('RFlow sending', data['current'])
            q.put(data)


if __name__ == '__main__':
    main()
