#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : memcached.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/23/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import collections

from jacinle.storage.kv.memcached import MemcachedKVStore
from .session import SessionManagerBase

__all__ = ['MemcachedSessionManager', 'MemcachedSessionIdentifier']


class MemcachedSessionIdentifier(collections.namedtuple('_MemcachedSessionIdentifier', ['session_id', 'hmac_key'])):
    pass


class MemcachedSessionManager(SessionManagerBase):
    def __init__(self, secret, memcache_host, memcache_port, timeout, cookie_prefix='jac_ses_', memcached_prefix='jac_sess_'):
        super().__init__(secret)

        self.memcache = MemcachedKVStore(memcache_host, memcache_port)
        self.session_timeout = timeout
        self.cookie_prefix = cookie_prefix
        self.memcached_prefix = memcached_prefix

        assert self.memcache.available

    def get(self, request_handler):
        session_id = hmac_key = None
        if request_handler is not None:
            session_id = request_handler.get_secure_cookie(self.cookie_prefix + 'session_id').decode('utf8')
            hmac_key = request_handler.get_secure_cookie(self.cookie_prefix + 'verification').decode('utf8')

        if session_id is None:
            session_exists = False
            session_id = self._generate_id()
            hmac_key = self._generate_hmac(session_id)
        else:
            session_exists = True

        data = {}
        if session_exists and hmac_key == self._generate_hmac(session_id):
            data = self.memcache.get(self.memcached_prefix + session_id, data, refresh=True, refresh_timeout=self.session_timeout)
        return MemcachedSessionIdentifier(session_id, hmac_key), data

    def set(self, request_handler, identifier, data):
        request_handler.set_secure_cookie(self.cookie_prefix + 'session_id', identifier.session_id.encode('utf8'))
        request_handler.set_secure_cookie(self.cookie_prefix + 'verification', identifier.hmac_key.encode('utf8'))
        self.memcache.put(self.memcached_prefix + identifier.session_id, data, replace=True, timeout=self.session_timeout)

