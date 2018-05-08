#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : gather.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/16/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import time
import multiprocessing as mp

from jacinle.comm.gather import make_gather_pair
from jacinle.utils.meta import map_exec_method


def mainloop_pull(pipe):
    with pipe.activate():
        while True:
            msg = pipe.recv()
            print('Received: worker_id=#{}, msg={}.'.format(msg['worker_id'], msg))


def mainloop_push(worker_id, pipe):
    print('Initialized: worker_id=#{}.'.format(worker_id))
    with pipe.activate():
        while True:
            msg = dict(text='Hello world!', time=time.strftime('%H:%M:%S'), worker_id=worker_id)
            pipe.send(msg)
            print('Sent: msg={}.'.format(msg))
            time.sleep(1)


def main():
    pull, pushs = make_gather_pair('jaincle-test', nr_workers=4, mode='ipc')
    push_procs = [mp.Process(target=mainloop_push, args=(i, p)) for i, p in enumerate(pushs)]
    map_exec_method('start', push_procs)

    mainloop_pull(pull)


if __name__ == '__main__':
    main()
