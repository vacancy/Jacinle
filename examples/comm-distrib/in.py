#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : in.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/03/2017
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import time

from jacinle.comm.distrib import DistribInputPipe, control
from jacinle.utils.printing import stprint


def main():
    q = DistribInputPipe('jacinle.test')
    with control(pipes=[q]):
        for i in range(10):
            stprint(q.get())
            time.sleep(1)


if __name__ == '__main__':
    main()
