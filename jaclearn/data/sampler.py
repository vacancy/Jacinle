#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : sampler.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from jacinle.random.rng import gen_rng
from jacnp.indexing import index_select_batch

__all__ = ['EpochBatchSampler', 'SimpleBatchSampler']


class _SizedGenerator(object):
    def __init__(self, generator, length):
        """
        Initialize the generator.

        Args:
            self: (todo): write your description
            generator: (todo): write your description
            length: (int): write your description
        """
        self._generator = generator
        self._length = length

    def __iter__(self):
        """
        Iterate over the iterator.

        Args:
            self: (todo): write your description
        """
        for i in self._generator:
            yield i

    def __len__(self):
        """
        Returns the length of the field.

        Args:
            self: (todo): write your description
        """
        return self._length


class RenamedDictSamplerBase(object):
    def _gen_renamed(self, data, keys, renames=None):
        """
        Yields the key.

        Args:
            self: (todo): write your description
            data: (todo): write your description
            keys: (list): write your description
            renames: (str): write your description
        """
        if renames is None:
            return self._gen(data, keys)

        assert len(renames) == len(keys)
        for v in self._gen(data, keys):
            yield {k1: v[k2] for k1, k2 in zip(renames, keys)}

    def _gen(self, data, keys):
        """
        Generate data.

        Args:
            self: (todo): write your description
            data: (todo): write your description
            keys: (list): write your description
        """
        raise NotImplementedError()


class EpochBatchSampler(RenamedDictSamplerBase):
    def __init__(self, batch_size, epoch_size, rng=None):
        """
        Initialize the _rng.

        Args:
            self: (todo): write your description
            batch_size: (int): write your description
            epoch_size: (int): write your description
            rng: (int): write your description
        """
        self._batch_size = batch_size
        self._epoch_size = epoch_size
        self._rng = rng or gen_rng()

    def _gen(self, data, keys):
        """
        Generate an iterator over the given keys.

        Args:
            self: (todo): write your description
            data: (todo): write your description
            keys: (list): write your description
        """
        n = len(data[keys[0]])
        for i in range(self._epoch_size):
            this_idx = self._rng.randint(n, size=self._batch_size)
            this = {k: index_select_batch(data[k], this_idx) for k in keys}
            yield this

    def __call__(self, data, keys, renames=None):
        """
        Call a call call.

        Args:
            self: (todo): write your description
            data: (todo): write your description
            keys: (list): write your description
            renames: (str): write your description
        """
        return _SizedGenerator(self._gen_renamed(data, keys, renames), self._epoch_size)


class SimpleBatchSampler(RenamedDictSamplerBase):
    def __init__(self, batch_size, nr_repeat, rng=None):
        """
        Initialize a batch.

        Args:
            self: (todo): write your description
            batch_size: (int): write your description
            nr_repeat: (int): write your description
            rng: (int): write your description
        """
        self._batch_size = batch_size
        self._nr_repeat = nr_repeat
        self._rng = rng or gen_rng()

    def _gen(self, data, keys):
        """
        Yields chunks of the given keys.

        Args:
            self: (todo): write your description
            data: (todo): write your description
            keys: (list): write your description
        """
        n = len(data[keys[0]])

        for i in range(self._nr_repeat):
            idx = self._rng.permutation(n)
            for j in range(n // self._batch_size):
                this_idx = idx[j * self._batch_size:j * self._batch_size + self._batch_size]
                this = {k: index_select_batch(data[k], this_idx) for k in keys}
                yield this

    def _len(self, data, keys):
        """
        Return the number of rows in the dataset.

        Args:
            self: (todo): write your description
            data: (todo): write your description
            keys: (list): write your description
        """
        n = len(data[keys[0]])
        return self._nr_repeat * (n // self._batch_size)

    def __call__(self, data, keys, renames=None):
        """
        Calls call call.

        Args:
            self: (todo): write your description
            data: (todo): write your description
            keys: (list): write your description
            renames: (str): write your description
        """
        return _SizedGenerator(self._gen_renamed(data, keys, renames), self._len(data, keys))
