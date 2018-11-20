#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : mem.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from .kv import KVStoreBase


class MemKVStore(KVStoreBase):
    def __init__(self, readonly=False):
        super().__init__(readonly=readonly)
        self._store = dict()

    def _has(self, key):
        return key in self._store

    def _get(self, key, default):
        return self._store.get(key, default)

    def _put(self, key, value, replace):
        if not replace:
            self._store.setdefault(key, value)
        else:
            self._store[key] = value

    def _erase(self, key):
        return self._store.pop(key)

    def _keys(self):
        return self._store.keys()
