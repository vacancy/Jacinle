# -*- coding: utf-8 -*-
# File   : mem.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 19/01/2018
#
# This file is part of Jacinle.

from jacinle.utils.context import EmptyContext
from .kv import KVStoreBase


class MemKVStore(KVStoreBase):
    def __init__(self, readonly=False):
        super().__init__(readonly=readonly)
        self._store = dict()

    def _get(self, key, default):
        return self._store.get(key, default)

    def _put(self, key, value, replace):
        if not replace:
            self._store.setdefault(key, value)
        else:
            self._store[key] = value

    def _transaction(self, *args, **kwargs):
        return EmptyContext()

    def _keys(self):
        return self._store.keys()
