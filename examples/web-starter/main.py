#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : main.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/23/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from jacinle.logging import get_logger
from jacweb.web import make_app

import tornado.ioloop

logger = get_logger(__file__)


def main():
    app = make_app([
        'app.index'
    ], {
        'gzip': True,
        'debug': False,
        'xsrf_cookies': True,

        'static_path': 'static',
        'template_path': 'app/templates',

        "cookie_secret": "20f42d0ae6548e88cf9788e725b298bd",
        "session_secret": "3cdcb1f00803b6e78ab50b466a40b9977db396840c28307f428b25e2277f1bcc",
        "frontend_secret": "asdjikfh98234uf9pidwja09f9adsjp9fd6840c28307f428b25e2277f1bcc",

        "cookie_prefix": 'jac_',

        # 'session_engine': 'off',
        'session_engine': 'memcached',
        "session_cookie_prefix": 'jac_sess_',
        "session_memcached_prefix": 'jac_sess_',
        "session_timeout": 60*30,

        "memcached_host": "127.0.0.1",
        "memcached_port": "11211",
    })
    app.listen(8081, xheaders=True)

    logger.critical('Mainloop started.')
    loop = tornado.ioloop.IOLoop.current()
    loop.start()


if __name__ == '__main__':
    main()

