# -*- coding: utf-8 -*-
# File   : lmdb.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 19/01/2018
#
# This file is part of Jacinle.


from .kv import KVStoreBase
from jacinle.utils.cache import cached_property

import os
import lmdb
import pickle

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
        self._lmdb_keys = keys
        self._is_dirty = False

    @cached_property
    def txn(self):
        return self._lmdb.begin(write=not self.readonly)

    def _get(self, key, default):
        value = self.txn.get(key.encode(self._key_charset), default=default)
        value = _loads(value)
        return value

    def _put(self, key, value, replace=False):
        self._is_dirty = True
        if self._lmdb_keys is None:
            self._lmdb_keys = []
        # TODO(MJY):: test whehter the key already exists
        self._lmdb_keys.append(key)
        return self.txn.put(key.encode(self._key_charset), _dumps(value), overwrite=replace)

    def _transaction(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_trace):
        if exc_type:
            self.txn.abort()
        else:
            self.txn.put(self._magic_key, _dumps(self._lmdb_keys))
            self.txn.commit()

    def _keys(self):
        if self._lmdb_keys is None:
            self._lmdb_keys = self.txn.get(self._magic_key, None)
            assert self._lmdb_keys is not None, 'LMDBKVStore does not support __keys__ access'
            self._lmdb_keys = _loads(self._lmdb_keys)
        return self._lmdb_keys
