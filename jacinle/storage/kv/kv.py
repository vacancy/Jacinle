# -*- coding: utf-8 -*-
# File   : kv.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 19/01/2018
#
# This file is part of Jacinle.


class KVStoreBase(object):
    def __init__(self, readonly=False):
        self.__readonly = readonly

    @property
    def readonly(self):
        return self.__readonly

    def get(self, key, default=None):
        return self._get(key, default=default)

    def put(self, key, value, replace=True):
        assert not self.readonly, 'KVStore is readonly: {}.'.format(self)
        return self._put(key, value, replace=replace)

    def transaction(self, *args, **kwargs):
        return self._transaction(*args, **kwargs)

    def keys(self):
        return self._keys()

    def _get(self, key, default):
        raise NotImplementedError()

    def _put(self, key, value, replace):
        raise NotImplementedError()

    def _transaction(self, *args, **kwargs):
        raise NotImplementedError()

    def _keys(self):
        assert False, 'KVStore does not support keys access.'

