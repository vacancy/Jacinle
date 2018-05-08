#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : broadcast.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/16/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import time
import multiprocessing as mp

from jacinle.comm.broadcast import make_broadcast_pair
from jacinle.utils.meta import map_exec_method


def mainloop_push(pipe):
    with pipe.activate():
        while True:
            msg = dict(text='Hello world!', time=time.strftime('%H:%M:%S'))
            pipe.send(msg)
            print('Sent: msg={}.'.format(msg))
            time.sleep(1)


def mainloop_pull(worker_id, pipe):
    print('Initialized: worker_id=#{}.'.format(worker_id))
    try:
        with pipe.activate():
            while True:
                msg = pipe.recv()
                print('Received: worker_id=#{}, msg={}.'.format(worker_id, msg))
    except:
        import traceback
        traceback.print_exc()
        raise


def main():
    push, pulls = make_broadcast_pair('jaincle-test', nr_workers=1, mode='ipc')
    pull_procs = [mp.Process(target=mainloop_pull, args=(i, p)) for i, p in enumerate(pulls)]
    map_exec_method('start', pull_procs)

    mainloop_push(push)


if __name__ == '__main__':
    main()
