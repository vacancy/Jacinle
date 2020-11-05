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
        """
        Initialize the store.

        Args:
            self: (todo): write your description
            readonly: (bool): write your description
        """
        super().__init__(readonly=readonly)
        self._store = dict()

    def _has(self, key):
        """
        Return true if the given key is in the cache.

        Args:
            self: (todo): write your description
            key: (str): write your description
        """
        return key in self._store

    def _get(self, key, default):
        """
        Return the value of a key.

        Args:
            self: (todo): write your description
            key: (str): write your description
            default: (todo): write your description
        """
        return self._store.get(key, default)

    def _put(self, key, value, replace):
        """
        Stores the value of value in the cache.

        Args:
            self: (todo): write your description
            key: (str): write your description
            value: (todo): write your description
            replace: (bool): write your description
        """
        if not replace:
            self._store.setdefault(key, value)
        else:
            self._store[key] = value

    def _erase(self, key):
        """
        Erases the given key from the cache.

        Args:
            self: (todo): write your description
            key: (str): write your description
        """
        return self._store.pop(key)

    def _keys(self):
        """
        Return all keys in the store

        Args:
            self: (todo): write your description
        """
        return self._store.keys()
