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
from typing import Any, Optional, Iterable

from .base import KVStoreBase
from jacinle.logging import get_logger
from jacinle.utils.cache import cached_property
from jacinle.utils.container import OrderedSet

__all__ = ['LMDBKVStore']

_logger = get_logger(__file__)
_loads = pickle.loads
_dumps = pickle.dumps


class LMDBKVStore(KVStoreBase):
    """A LMDB-based key-value store. This class also supports sub-databases.

    See ``examples/kv-lmdb`` for more usage.

    Basic usage with a single database:
        .. code-block:: python

            kv = LMDBKVStore('/tmp/test_1.lmdb', readonly=False)

            with kv.transaction():
                kv['a'] = 1
                kv['b'] = 2

            assert 'a' in kv and kv['a'] == 1
            assert 'b' in kv and kv['b'] == 2
            assert 'c' not in kv

            for k in kv.keys():
                print(k, kv[k])
    """

    _key_charset = 'utf8'
    _magic_key = b'__keys__'
    _magic_main_databse = b'__main_database__'

    def __init__(self, lmdb_path: str, readonly: bool = True, max_dbs: int = 0, keys: Optional[Iterable[Any]] = None):
        """Initialize the LMDBKVStore.

        Args:
            lmdb_path: the path to the LMDB file. By default, this path is a directory.
            readonly: whether to open the LMDB in readonly mode.
            max_dbs: the maximum number of sub-databases. 0 means the single-database mode.
            keys: the keys in the main database. For new databases, use None.
        """
        super().__init__(readonly=readonly)
        self._max_dbs = max_dbs
        self._open(lmdb_path, readonly=readonly, keys=keys, max_dbs=max_dbs)

    def _open(self, lmdb_path, readonly, keys, max_dbs):
        self._lmdb = lmdb.open(
            lmdb_path,
            subdir=os.path.isdir(lmdb_path) or not os.path.exists(lmdb_path),
            readonly=readonly,
            lock=False,
            readahead=False,
            map_size=1099511627776 * 2,
            max_readers=100,
            max_dbs=max_dbs
        )
        self._subdbs = dict()
        self._lmdb_keys = dict()
        self._is_dirty = dict()

        if keys is not None:
            self._lmdb_keys[self._magic_main_databse] = OrderedSet(keys)
            self._is_dirty[self._magic_main_databse] = True

        self._is_in_transaction = False

    @cached_property
    def txn(self):
        """The current LMDB transaction."""
        return self._lmdb.begin(write=not self.readonly)

    @property
    def lmdb(self):
        """The LMDB environment."""
        return self._lmdb

    def get_subdb(self, name=None):
        """Get a sub-database by name."""
        if name is None:
            return None
        if name not in self._subdbs:
            self._subdbs[name] = self._lmdb.open_db(name.encode(self._key_charset), txn=self.txn)
        return self._subdbs[name]

    def set_dirty(self, db=None):
        """Set the dirty flag for the given sub-database."""
        self._is_dirty[db] = True

    def _has(self, key, db=None):
        self._load_lmdb_keys(db)
        return key in self._lmdb_keys[db]

    def _get(self, key, default, db=None):
        value = self.txn.get(key.encode(self._key_charset), default=default, db=self.get_subdb(db))
        value = _loads(value)
        return value

    def _put(self, key, value, replace=False, db=None):
        assert self._is_in_transaction, 'LMDBKVStore does not support put outside transaction.'
        self._load_lmdb_keys(db)
        self._is_dirty[db] = True
        self._lmdb_keys[db].append(key)
        return self.txn.put(key.encode(self._key_charset), _dumps(value), overwrite=replace, db=self.get_subdb(db))

    def _erase(self, key, db=None):
        assert self._is_in_transaction, 'LMDBKVStore does not support erase outside transaction.'
        self._load_lmdb_keys(db)
        self._is_dirty[db] = True
        self._lmdb_keys.remove(key)
        return self.txn.pop(key.encode(self._key_charset), db=self.get_subdb(db))

    def _keys(self, db=None):
        self._load_lmdb_keys(db)
        return self._lmdb_keys[db]

    def _transaction(self, *args, **kwargs):
        return self

    def __enter__(self):
        self._is_in_transaction = True
        return self

    def __exit__(self, exc_type, exc_value, exc_trace):
        self._is_in_transaction = False
        if exc_type:
            self.txn.abort()
        else:
            for k, v in self._is_dirty.items():
                if v:
                    self.txn.put(self._magic_key, _dumps(self._lmdb_keys[k].as_list()), db=self.get_subdb(k))
                    self._is_dirty[k] = False
            self.txn.commit()

        self.txn = self._lmdb.begin(write=not self.readonly)

    def _load_lmdb_keys(self, db=None, assert_exist=False):
        if db in self._lmdb_keys:
            return
        dbobj = self.get_subdb(db)

        keys = self.txn.get(self._magic_key, None, db=dbobj)
        if assert_exist:
            assert keys is not None, 'LMDBKVStore does not support __keys__ access'

        if keys is None:
            # build the list of keys by traversing the databse
            keys = OrderedSet()
            with self.txn.cursor(db=dbobj) as cursor:
                for k, _ in cursor:
                    keys.append(k.decode(self._key_charset))
            self._lmdb_keys[db] = keys
            self._is_dirty[db] = True

            if db is None and self._max_dbs > 0 and len(keys) > 0:
                _logger.warning(
                    'LMDBKVStore is trying to build a list of keys of an non-empty main database. '
                    'This means that the underlying lmdb databse was not created by this class. '
                    'Because LMDB internally stores sub databases as links in the main database, this operation will'
                    'create a union of all keys in the main database and the names of the sub databases. '
                    'In most cases this is not what you want.'
                    'You can use kv._lmdb_keys[None].remove() to remove the names of the sub databases, and then call kv.set_dirty() to trigger auto-commit.'
                )
        else:
            self._lmdb_keys[db] = OrderedSet(_loads(keys))

