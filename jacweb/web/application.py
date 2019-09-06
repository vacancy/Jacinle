#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : application.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/23/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from tornado.web import Application, RequestHandler
from copy import deepcopy

from jacinle.logging import get_logger
from jacinle.utils.imp import load_module

import time

logger = get_logger(__file__)

__all__ = ['route', 'make_app', 'get_app', 'JacApplication', 'JacRequestHandler']

__handlers__ = []
__application__ = None
__default_frontend_salt__ = '|v<zj!#PGnxgc])Sd_=i|8,?,7><s5+Ck_IWcbC D:@4a~(1bEA=M+MacyI2udaz'


def route(regex):
    global __handlers__

    def wrap_class(cls):
        __handlers__.append((regex, cls))
        return cls
    return wrap_class


def make_app(modules, settings):
    global __application__, __handlers__
    assert __application__ is None

    for m in modules:
        load_module(m)
    __application__ = JacApplication(__handlers__, **settings)
    return __application__


def get_app():
    global __application__
    return __application__


try:
    _ARG_DEFAULT = RequestHandler._ARG_DEFAULT
except:
    from tornado.web import _ARG_DEFAULT


class JacRequestHandler(RequestHandler):
    def __init__(self, *args, **kwargs):
        self.session = None
        self.__start_time = None
        self.__end_time = None

        super(JacRequestHandler, self).__init__(*args, **kwargs)

    def initialize(self):
        if self.application.session_enabled:
            self.session = self.application.get_session(self)
        self.__start_time = time.time()

    def save_session(self):
        if self.application.session_enabled:
            self.session.save()

    def get_template_namespace(self):
        dct = super(JacRequestHandler, self).get_template_namespace()
        dct['url'] = self.request.uri
        dct['url_root'] = self.application.settings.get('url_root', '/')
        dct['is_localhost'] = self.application.settings.get('is_localhost', False)
        dct['frontend_secret'] = self.application.settings.get('frontend_secret', __default_frontend_salt__)
        return dct

    def finish(self, *args, **kwargs):
        self.on_body_finish()
        super().finish(*args, **kwargs)

    def on_body_finish(self):
        self.save_session()

    def on_finish(self):
        self.__end_time = time.time()
        logger.info('{} query {} executed for {:.3f} ms'.format(self.request.remote_ip, self.request.uri, (self.__end_time - self.__start_time) * 1000))
        super().on_finish()

    def get_query_argument(self, name, default=_ARG_DEFAULT, strip=True, type=None, danger_set=None):
        return self._get_argument(name, default, self.request.arguments, strip, type, danger_set)

    def get_body_argument(self, name, default=_ARG_DEFAULT, strip=True, type=None, danger_set=None):
        return self._get_argument(name, default, self.request.body_arguments, strip, type, danger_set)

    def get_argument(self, name, default=_ARG_DEFAULT, strip=True, type=None, danger_set=None):
        return self._get_argument(name, default, self.request.arguments, strip, type, danger_set)

    def _get_argument(self, name, default, source, strip=True, type=None, danger_set=None):
        raw = super()._get_argument(name, default, source, strip)
        if type is int:
            return self._assert_int(raw)
        elif type is str:
            return self._assert_safe_string(raw, danger_set)
        elif type is None:
            return raw
        else:
            raise NotImplementedError('Unknown argument type: {}.'.format(type))

    def _assert_int(self, v):
        try:
            v = int(v)
        except ValueError:
            raise JacRequestException('Invalid data submitted.')
        return v

    def _assert_safe_string(self, s, danger_set):
        if danger_set is None:
            danger_set = '`#\'"<>'
        try:
            s = str(s)
            for i in s:
                if i in danger_set:
                    raise ValueError
        except ValueError:
            raise JacRequestException('Invalid data submitted.')
        return s


class JacApplication(Application):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.settings['session_engine'] == 'off':
            self.session_enabled = False
            self.session_manager = None
        elif self.settings['session_engine'] == 'memcached':
            from jacweb.session.memcached import MemcachedSessionManager
            self.session_enabled = True
            self.session_manager = MemcachedSessionManager(
                secret=self.settings['session_secret'],
                memcache_host=self.settings['memcached_host'],
                memcache_port=self.settings['memcached_port'],
                timeout=self.settings['session_timeout'],
                cookie_prefix=self.settings['session_cookie_prefix'],
                memcached_prefix=self.settings['session_memcached_prefix']
            )
            logger.critical('Initializing the session manager using memcached: {}.'.format(self.session_manager.memcache.full_addr))
        else:
            raise ValueError('Unknown session engine: {}.'.format(self.settings['session_engine']))

    def get_session(self, request_handler):
        assert self.session_manager is not None, 'Session manager not initialized.'
        return self.session_manager.new(request_handler)


class JacRequestException(Exception):
    pass

