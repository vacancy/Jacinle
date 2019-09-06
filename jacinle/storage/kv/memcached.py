#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : memcached.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import lmdb
import pickle

try:
    import memcache
except ImportError:
    from jacinle.logging import get_logger
    logger = get_logger(__file__)
    logger.warning('Cannot import memcache. MemcachedKVStore is unavailable.')
    memcache = None

from .kv import KVStoreBase

_loads = pickle.loads
_dumps = pickle.dumps
_encode = lambda s: s.encode('utf8')


class MemcachedKVStore(KVStoreBase):
    def __init__(self, addr, port, readonly=False):
        super().__init__(readonly)
        self.available = memcache is not None

        self.addr = addr
        self.port = port
        self.full_addr = '{}:{}'.format(addr, port)

        self._connection = None

    @property
    def connection(self):
        if self._connection is None:
            self._connection = memcache.Client([self.full_addr], debug=0)
        return self._connection

    def _has(self, key):
        key = _encode(key)
        try:
            data = self.connection.get(key)
            if data is not None:
                return True
            return False
        except IOError:
            return False

    def _get(self, key, default=None, refresh=False, refresh_timeout=0):
        key = _encode(key)
        try:
            data = self.connection.get(key)
            if data is not None:
                if refresh:
                    self.connection.replace(key, data, refresh_timeout, 1)
                return _loads(data)
            else:
                return default
        except IOError:
            return default

    def _put(self, key, value, replace=False, timeout=0):
        key = _encode(key)
        # TODO(Jiayuan Mao @ 10/23): implement replace.
        assert replace, 'Not implemented.'
        self.connection.set(key, _dumps(value), timeout, 1)

    def _erase(self, key):
        key = _encode(key)
        self.connection.delete(key)

