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
        self.__readonly = readonly

    @property
    def readonly(self):
        return self.__readonly

    def has(self, key, **kwargs):
        return self._has(key, **kwargs)

    def get(self, key, default=None, **kwargs):
        return self._get(key, default=default, **kwargs)

    def put(self, key, value, replace=True, **kwargs):
        assert not self.readonly, 'KVStore is readonly: {}.'.format(self)
        return self._put(key, value, replace=replace, **kwargs)

    def update(self, key, value, **kwargs):
        kwargs['replace'] = True
        self.put(key, value, **kwargs)

    def erase(self, key, **kwargs):
        assert not self.readonly, 'KVStore is readonly: {}.'.format(self)
        return self._erase(key, **kwargs)

    def __contains__(self, key):
        return self.has(key)

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        return self.set(key, value)

    def __delitem__(self, key):
        return self.erase(key)

    def transaction(self, *args, **kwargs):
        return self._transaction(*args, **kwargs)

    def keys(self):
        return self._keys()

    def _has(self, key):
        raise NotImplementedError('KVStore {} does not support has.'.format(self.__class__.__name__))

    def _get(self, key, default):
        raise NotImplementedError('KVStore {} does not support get.'.format(self.__class__.__name__))

    def _put(self, key, value, replace):
        raise NotImplementedError('KVStore {} does not support put.'.format(self.__class__.__name__))

    def _erase(self, key):
        raise NotImplementedError('KVStore {} does not support erase.'.format(self.__class__.__name__))

    def _transaction(self, *args, **kwargs):
        return EmptyContext()

    def _keys(self):
        raise NotImplementedError('KVStore {} does not support keys access.'.format(self.__class__.__name__))

