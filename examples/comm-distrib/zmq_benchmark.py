#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : zmq_benchmark.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/16/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import itertools
import time
from threading import Thread
import multiprocessing as mp

import numpy as np

from jacinle.comm.gather import make_gather_pair
from jacinle.utils.meta import map_exec_method


counter = itertools.count()


def recv_thread(q):
    while True:
        q.recv()
        next(counter)


def mainloop_pull(q):
    current = next(counter)
    prob_interval = 1

    Thread(target=recv_thread, args=(q, ), daemon=True).start()

    while True:
        previous = current
        current = next(counter)
        now = time.strftime('%Y-%m-%d %H:%M:%S')
        nr_packs = current - previous - 1
        pps = nr_packs / prob_interval
        print('RFlow benchmark: timestamp={}, pps={}.'.format(now, pps))
        time.sleep(prob_interval)


def mainloop_push(worker_id, pipe):
    print('Initialized: worker_id=#{}.'.format(worker_id))
    with pipe.activate():
        while True:
            msg = {'msg': 'hello', 'current': time.time(), 'data': np.zeros(shape=(128, 224, 224, 3), dtype='float32')}
            pipe.send(msg)
            # print('Sent: msg={}.'.format(msg))


def main():
    pull, pushs = make_gather_pair('jaincle-test', nr_workers=1, mode='tcp')
    push_procs = [mp.Process(target=mainloop_push, args=(i, p)) for i, p in enumerate(pushs)]
    map_exec_method('start', push_procs)

    mainloop_pull(pull)


if __name__ == '__main__':
    main()
