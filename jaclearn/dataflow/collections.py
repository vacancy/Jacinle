#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : collections.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/22/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import numpy as np

from jacinle.utils.argument import UniqueValueGetter

from .dataflow import SimpleDataFlowBase, ProxyDataFlowBase, RandomizedDataFlowBase

__all__ = [
    'DictDataFlowProxy', 'EmptyDictDataFlow',
    'QueueDataFlow', 'PoolDataFlow',
    'ListOfArrayDataFlow', 'DictOfArrayDataFlow',
    'DictToBatchDataFlow',
    'KVStoreDataFlow', 'KVStoreRandomSampleDataFlow',
    'PoolRandomSampleDataFlow', 'LOARandomSampleDataFlow', 'DOARandomSampleDataFlow'
]


class DictDataFlowProxy(ProxyDataFlowBase):
    def __init__(self, keys, iterable):
        """
        Recursively iterable keys.

        Args:
            self: (todo): write your description
            keys: (array): write your description
            iterable: (todo): write your description
        """
        super().__init__(iterable)
        self._keys = keys
        self._iterable = iterable

    def _gen(self):
        """
        Generate an iterator over ( key value.

        Args:
            self: (todo): write your description
        """
        for v in self._iterable:
            assert len(self._keys) == len(v), 'DictDataFlowAdapter: length mismatched'
            yield dict(zip(self._keys, v))


class EmptyDictDataFlow(SimpleDataFlowBase):
    def _gen(self):
        """
        Generate a generator that yields a generator.

        Args:
            self: (todo): write your description
        """
        while True:
            yield {}


class QueueDataFlow(SimpleDataFlowBase):
    def __init__(self, queue):
        """
        Initialize the queue.

        Args:
            self: (todo): write your description
            queue: (todo): write your description
        """
        self._queue = queue

    def _gen(self):
        """
        Generator that yields the queue.

        Args:
            self: (todo): write your description
        """
        while True:
            yield self._queue.get()


class PoolDataFlow(SimpleDataFlowBase):
    def __init__(self, pool):
        """
        Initialize the pool.

        Args:
            self: (todo): write your description
            pool: (todo): write your description
        """
        self._pool = pool
        self._length = len(self._pool)

    def _gen(self):
        """
        Generate an iterable.

        Args:
            self: (todo): write your description
        """
        for i in range(self._length):
            yield self._pool[i]

    def _len(self):
        """
        Returns the length of the field.

        Args:
            self: (todo): write your description
        """
        return self._length


class ListOfArrayDataFlow(SimpleDataFlowBase):
    def __init__(self, loa):
        """
        Initialize the vector.

        Args:
            self: (todo): write your description
            loa: (todo): write your description
        """
        self._loa = loa

        uvg = UniqueValueGetter('ListOfArrayDataFlow length consistency check failed')
        for i in self._loa:
            uvg.set(len(i))
        self._length = uvg.get()

    def _gen(self):
        """
        Generator : class :.

        Args:
            self: (todo): write your description
        """
        for i in range(self._length):
            yield [l[i] for l in self._loa]

    def _len(self):
        """
        Returns the length of the field.

        Args:
            self: (todo): write your description
        """
        return self._length


def DictOfArrayDataFlow(doa):
    """
    Return a new dstream of the given dict.

    Args:
        doa: (todo): write your description
    """
    keys = doa.keys()
    values = [doa[k] for k in keys]
    return DictDataFlowProxy(keys, ListOfArrayDataFlow(values))


class DictToBatchDataFlow(ProxyDataFlowBase):
    def __init__(self, iterable, excludes=None):
        """
        Initialize the iterable.

        Args:
            self: (todo): write your description
            iterable: (todo): write your description
            excludes: (list): write your description
        """
        super().__init__(iterable)
        self._excludes = set(excludes) if excludes is not None else set()

    def _gen(self):
        """
        Unwraps all items from the array.

        Args:
            self: (todo): write your description
        """
        for item in self.unwrapped:
            for k, v in item.items():
                if k not in self._excludes:
                    item[k] = np.array(v)[np.newaxis]


class KVStoreDataFlow(SimpleDataFlowBase):
    def __init__(self, kv_getter):
        """
        Initialize the kvstore.

        Args:
            self: (todo): write your description
            kv_getter: (todo): write your description
        """
        self._kv_getter = kv_getter
        self._kvstore = None
        self._keys = None

    def _initialize(self):
        """
        Initialize the kvstore.

        Args:
            self: (todo): write your description
        """
        super()._initialize()
        self._kvstore = self._kv_getter()
        self._keys = list(self._kvstore.keys())

    def _gen(self):
        """
        Return an iterator over all the keys.

        Args:
            self: (todo): write your description
        """
        for k in self._keys:
            yield self._kvstore.get(k)


class KVStoreRandomSampleDataFlow(RandomizedDataFlowBase):
    def __init__(self, kv_getter, seed=None):
        """
        Initialize kvstore.

        Args:
            self: (todo): write your description
            kv_getter: (todo): write your description
            seed: (int): write your description
        """
        super().__init__(seed=seed)
        self._kv_getter = kv_getter
        self._kvstore = None
        self._keys = None
        self._nr_keys = None

    def _initialize(self):
        """
        Initialize kvstore.

        Args:
            self: (todo): write your description
        """
        super()._initialize()
        self._kvstore = self._kv_getter()
        self._keys = list(self._kvstore.keys())
        self._nr_keys = len(self._keys)

    def _gen(self):
        """
        Generates over all - > iterator.

        Args:
            self: (todo): write your description
        """
        while True:
            k = self._keys[self._rng.choice(self._nr_keys)]
            yield self._kvstore.get(k)


class PoolRandomSampleDataFlow(RandomizedDataFlowBase):
    _pool = None

    def __init__(self, pool, seed=None):
        """
        Initialize the pool.

        Args:
            self: (todo): write your description
            pool: (todo): write your description
            seed: (int): write your description
        """
        super().__init__(seed=seed)
        self._pool = pool
        self._length = len(self._pool)

    def _gen(self):
        """
        Generate a list of lists.

        Args:
            self: (todo): write your description
        """
        while True:
            self._rng.shuffle_list(self._pool)
            for i in range(self._length):
                yield self._pool[i]


class LOARandomSampleDataFlow(RandomizedDataFlowBase):
    _loa = None
    _length = None

    def __init__(self, loa, seed=None):
        """
        Initialize the seed.

        Args:
            self: (todo): write your description
            loa: (todo): write your description
            seed: (int): write your description
        """
        super().__init__(seed=seed)
        self._set_loa(loa)

    def _set_loa(self, loa):
        """
        Calculate the alpha.

        Args:
            self: (todo): write your description
            loa: (str): write your description
        """
        self._loa = loa
        uvg = UniqueValueGetter('LOARandomSampleDataFlow length consistency check failed')
        for i in self._loa:
            uvg.set(len(i))
        self._length = uvg.get()

    def _gen(self):
        """
        Generator that yields all rngs.

        Args:
            self: (todo): write your description
        """
        while True:
            state = self._rng.get_state()
            for item in self._loa:
                self._rng.set_state(state)
                self._rng.shuffle(item)
            for i in range(self._length):
                yield [l[i] for l in self._loa]


def DOARandomSampleDataFlow(doa, seed=None):
    """
    Creates a new dstream object.

    Args:
        doa: (todo): write your description
        seed: (int): write your description
    """
    keys = doa.keys()
    values = [doa[k] for k in keys]
    return DictDataFlowProxy(keys, LOARandomSampleDataFlow(values, seed=seed))


class RandomRepeatDataFlow(RandomizedDataFlowBase):
    def __init__(self, source, nr_repeat, cache_size, block=False, seed=None):
        """
        Initialize the cache.

        Args:
            self: (todo): write your description
            source: (str): write your description
            nr_repeat: (int): write your description
            cache_size: (int): write your description
            block: (todo): write your description
            seed: (int): write your description
        """
        super().__init__(seed=seed)
        self._source = source
        self._nr_repeat = nr_repeat
        self._cache_size = cache_size
        self._block = block

    def _gen(self):
        """
        Yields a generator.

        Args:
            self: (todo): write your description
        """
        it = iter(self._source)
        while True:
            data = []
            for i in range(self._cache_size):
                d = next(it)
                data.append(d)
                if not self._block:
                    yield d

            nr_repeat = self._nr_repeat if self._block else self._nr_repeat - 1
            for i in range(nr_repeat * self._cache_size):
                idx = self._rng.randint(len(data))
                yield data[idx]
