#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : in_benchmark.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/05/2017
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import time
import itertools

from threading import Thread

from jacinle.comm.distrib import control, DistribInputPipe


counter = itertools.count()


def recv_thread():
    q = DistribInputPipe('jacinle.test')
    with control(pipes=[q]):
        while True:
            q.get()
            next(counter)


def main():
    current = next(counter)
    prob_interval = 1

    Thread(target=recv_thread, daemon=True).start()

    while True:
        previous = current
        current = next(counter)
        now = time.strftime('%Y-%m-%d %H:%M:%S')
        nr_packs = current - previous - 1
        pps = nr_packs / prob_interval
        print('RFlow benchmark: timestamp={}, pps={}.'.format(now, pps))
        time.sleep(prob_interval)


if __name__ == '__main__':
    main()
