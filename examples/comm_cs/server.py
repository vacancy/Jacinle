# -*- coding:utf8 -*-
# File   : server.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/19/17
# 
# This file is part of Jacinle.

import time
from jacinle.comm.cs import ServerPipe


def answer(pipe, identifier, inp):
    out = inp['a'] + inp['b']
    pipe.send(identifier, dict(out=out))


def main():
    server = ServerPipe('server')
    server.dispatcher.register('calc', answer)
    with server.activate():
        print('Client command:')
        print('jac-run client.py', *server.conn_info)
        while True:
            time.sleep(1)


if __name__ == '__main__':
    main()
