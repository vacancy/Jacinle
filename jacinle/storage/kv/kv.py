#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : kv.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from jacinle.utils.context import EmptyContext


class KVStoreBase(object):
    def __init__(self, readonly=False):
        """
        Initialize the readonly.

        Args:
            self: (todo): write your description
            readonly: (bool): write your description
        """
        self.__readonly = readonly

    @property
    def readonly(self):
        """
        Reads the readonly.

        Args:
            self: (todo): write your description
        """
        return self.__readonly

    def has(self, key, **kwargs):
        """
        Returns true if key exists.

        Args:
            self: (todo): write your description
            key: (str): write your description
        """
        return self._has(key, **kwargs)

    def get(self, key, default=None, **kwargs):
        """
        Retrieves value of a key.

        Args:
            self: (todo): write your description
            key: (todo): write your description
            default: (todo): write your description
        """
        return self._get(key, default=default, **kwargs)

    def put(self, key, value, replace=True, **kwargs):
        """
        Store the value of a key.

        Args:
            self: (todo): write your description
            key: (str): write your description
            value: (todo): write your description
            replace: (bool): write your description
        """
        assert not self.readonly, 'KVStore is readonly: {}.'.format(self)
        return self._put(key, value, replace=replace, **kwargs)

    def update(self, key, value, **kwargs):
        """
        Update the value in the value of a.

        Args:
            self: (todo): write your description
            key: (str): write your description
            value: (todo): write your description
        """
        kwargs['replace'] = True
        self.put(key, value, **kwargs)

    def erase(self, key, **kwargs):
        """
        Erase the lock.

        Args:
            self: (todo): write your description
            key: (str): write your description
        """
        assert not self.readonly, 'KVStore is readonly: {}.'.format(self)
        return self._erase(key, **kwargs)

    def __contains__(self, key):
        """
        Determine whether the given key is contained in this set.

        Args:
            self: (todo): write your description
            key: (todo): write your description
        """
        return self.has(key)

    def __getitem__(self, key):
        """
        Returns the value of a cache.

        Args:
            self: (dict): write your description
            key: (str): write your description
        """
        return self.get(key)

    def __setitem__(self, key, value):
        """
        Sets the value of a key.

        Args:
            self: (todo): write your description
            key: (str): write your description
            value: (str): write your description
        """
        return self.set(key, value)

    def __delitem__(self, key):
        """
        Remove an item from the cache.

        Args:
            self: (todo): write your description
            key: (str): write your description
        """
        return self.erase(key)

    def transaction(self, *args, **kwargs):
        """
        Wraps a transaction.

        Args:
            self: (todo): write your description
        """
        return self._transaction(*args, **kwargs)

    def keys(self):
        """
        Returns a list of all keys in this is_keys.

        Args:
            self: (todo): write your description
        """
        return self._keys()

    def _has(self, key):
        """
        Returns true if the given key exists.

        Args:
            self: (todo): write your description
            key: (str): write your description
        """
        raise NotImplementedError('KVStore {} does not support has.'.format(self.__class__.__name__))

    def _get(self, key, default):
        """
        Return the value of a given key.

        Args:
            self: (todo): write your description
            key: (str): write your description
            default: (todo): write your description
        """
        raise NotImplementedError('KVStore {} does not support get.'.format(self.__class__.__name__))

    def _put(self, key, value, replace):
        """
        Stores the given key.

        Args:
            self: (todo): write your description
            key: (str): write your description
            value: (todo): write your description
            replace: (bool): write your description
        """
        raise NotImplementedError('KVStore {} does not support put.'.format(self.__class__.__name__))

    def _erase(self, key):
        """
        Erase the given key.

        Args:
            self: (todo): write your description
            key: (str): write your description
        """
        raise NotImplementedError('KVStore {} does not support erase.'.format(self.__class__.__name__))

    def _transaction(self, *args, **kwargs):
        """
        Decorator that wraps a transaction.

        Args:
            self: (todo): write your description
        """
        return EmptyContext()

    def _keys(self):
        """
        Returns a list of the keys.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError('KVStore {} does not support keys access.'.format(self.__class__.__name__))

