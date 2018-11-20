#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : lmdb.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import os
import lmdb
import pickle

from .kv import KVStoreBase
from jacinle.utils.cache import cached_property
from jacinle.utils.container import OrderedSet

__all__ = ['LMDBKVStore']

_loads = pickle.loads
_dumps = pickle.dumps


class LMDBKVStore(KVStoreBase):
    _key_charset = 'utf8'
    _magic_key = b'__keys__'

    def __init__(self, lmdb_path, readonly=True, keys=None):
        super().__init__(readonly=readonly)
        self._open(lmdb_path, readonly=readonly, keys=keys)

    def _open(self, lmdb_path, readonly, keys):
        self._lmdb = lmdb.open(lmdb_path,
                               subdir=os.path.isdir(lmdb_path),
                               readonly=readonly,
                               lock=False,
                               readahead=False,
                               map_size=1099511627776 * 2,
                               max_readers=100)
        self._lmdb_keys = None
        if keys is not None:
            self._lmdb_keys = OrderedSet(keys)
        self._is_dirty = False

    @cached_property
    def txn(self):
        return self._lmdb.begin(write=not self.readonly)

    def _has(self, key):
        self._load_lmdb_keys()
        return key in self._lmdb_keys

    def _get(self, key, default):
        value = self.txn.get(key.encode(self._key_charset), default=default)
        value = _loads(value)
        return value

    def _put(self, key, value, replace=False):
        self._is_dirty = True
        self._load_lmdb_keys()
        self._lmdb_keys.append(key)
        return self.txn.put(key.encode(self._key_charset), _dumps(value), overwrite=replace)

    def _erase(self, key):
        self._is_dirty = True
        self._load_lmdb_keys()
        self._lmdb_keys.remove(key)
        return self.txn.pop(key.encode(self._key_charset))

    def _transaction(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_trace):
        if exc_type:
            self.txn.abort()
        else:
            self.txn.put(self._magic_key, _dumps(self._lmdb_keys.as_list()))
            self.txn.commit()

    def _keys(self):
        self._load_lmdb_keys(True)
        return self._lmdb_keys

    def _load_lmdb_keys(self, assert_exist=False):
        if self._lmdb_keys is None:
            keys = self.txn.get(self._magic_key, None)
            if assert_exist:
                assert keys is not None, 'LMDBKVStore does not support __keys__ access'
            if keys is None:
                self._lmdb_keys = OrderedSet()
            else:
                self._lmdb_keys = OrderedSet(_loads(keys))

