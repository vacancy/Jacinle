#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : base.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from typing import Any, Iterable
from jacinle.utils.context import EmptyContext


class KVStoreBase(object):
    """The base class for all key-value stores."""

    def __init__(self, readonly: bool = False):
        """Initialize the KVStore.

        Args:
            readonly: If True, the KVStore is readonly.
        """
        self.__readonly = readonly

    @property
    def readonly(self):
        """Whether the KVStore is readonly."""
        return self.__readonly

    def has(self, key, **kwargs) -> bool:
        """Whether the key exists in the KVStore."""
        return self._has(key, **kwargs)

    def get(self, key, default=None, **kwargs):
        """Get the value of the key.

        Args:
            key: the key.
            default: the default value if the key does not exist.
        """
        return self._get(key, default=default, **kwargs)

    def put(self, key, value, replace: bool = True, **kwargs):
        """Put the value of the key. If the key already exists, the value will be replaced if replace is True.

        Args:
            key: the key.
            value: the value.
            replace: whether to replace the value if the key already exists.
        """
        assert not self.readonly, 'KVStore is readonly: {}.'.format(self)
        return self._put(key, value, replace=replace, **kwargs)

    def update(self, key, value, **kwargs):
        """Update the value of the key. If the key does not exist, the value will be put.

        Args:
            key: the key.
            value: the value.
        """
        kwargs['replace'] = True
        self.put(key, value, **kwargs)

    def erase(self, key, **kwargs):
        """Erase the key from the KVStore.

        Args:
            key: the key.
        """
        assert not self.readonly, 'KVStore is readonly: {}.'.format(self)
        return self._erase(key, **kwargs)

    def transaction(self, *args, **kwargs):
        """Create a transaction context."""
        return self._transaction(*args, **kwargs)

    def keys(self, **kwargs) -> Iterable[Any]:
        """Get all keys in the KVStore."""
        return self._keys(**kwargs)

    def __contains__(self, key):
        return self.has(key)

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        return self.put(key, value)

    def __delitem__(self, key):
        return self.erase(key)

    def _has(self, key, **kwargs):
        raise NotImplementedError('KVStore {} does not support has.'.format(self.__class__.__name__))

    def _get(self, key, default, **kwargs):
        raise NotImplementedError('KVStore {} does not support get.'.format(self.__class__.__name__))

    def _put(self, key, value, replace, **kwargs):
        raise NotImplementedError('KVStore {} does not support put.'.format(self.__class__.__name__))

    def _erase(self, key, **kwargs):
        raise NotImplementedError('KVStore {} does not support erase.'.format(self.__class__.__name__))

    def _keys(self, **kwargs):
        raise NotImplementedError('KVStore {} does not support keys access.'.format(self.__class__.__name__))

    def _transaction(self, *args, **kwargs):
        return EmptyContext()
