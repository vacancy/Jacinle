#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dataflow.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/22/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import collections

from jacinle.logging import get_logger
from jacinle.random import gen_rng

logger = get_logger(__file__)

__all__ = ['DataFlowBase', 'SimpleDataFlowBase', 'ProxyDataFlowBase', 'AdvancedDataFlowBase', 'RandomizedDataFlowBase']


class DataFlowBase(object):
    pass


collections.Iterator.register(DataFlowBase)


class SimpleDataFlowBase(DataFlowBase):
    __initialized = False

    def _initialize(self):
        """
        Initialize the next callable.

        Args:
            self: (todo): write your description
        """
        pass

    def _reset(self):
        """
        Reset the state.

        Args:
            self: (todo): write your description
        """
        pass

    def _gen(self):
        """
        Generate a generator.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError()

    def _finalize(self):
        """
        Finalize the underlyingize.

        Args:
            self: (todo): write your description
        """
        pass

    def _len(self):
        """
        Returns the number of bytes in bytes.

        Args:
            self: (todo): write your description
        """
        return None

    def __len__(self):
        """
        Returns the length of the field.

        Args:
            self: (todo): write your description
        """
        try:
            return self._len()
        except TypeError:
            return None

    def __iter__(self):
        """
        Iterate over all the data.

        Args:
            self: (todo): write your description
        """
        if not self.__initialized:
            self._initialize()
            self.__initialized = True
        self._reset()
        try:
            for v in self._gen():
                yield v
        except Exception as e:
            logger.exception('{} got exception {} during iter: {}.'.format(type(self), type(e), e))
        finally:
            self._finalize()


class ProxyDataFlowBase(SimpleDataFlowBase):
    def __init__(self, other):
        """
        Determine the other.

        Args:
            self: (todo): write your description
            other: (todo): write your description
        """
        self._unwrapped = other

    @property
    def unwrapped(self):
        """
        Returns the wrapped wrapped function.

        Args:
            self: (todo): write your description
        """
        return self._unwrapped

    def _gen(self):
        """
        Iterate over all the elements of the list.

        Args:
            self: (todo): write your description
        """
        for item in self._unwrapped:
            yield item

    def _len(self):
        """
        Returns the length of the field.

        Args:
            self: (todo): write your description
        """
        return len(self._unwrapped)


class AdvancedDataFlowBase(DataFlowBase):
    def __init__(self):
        """
        Initialize the instance.

        Args:
            self: (todo): write your description
        """
        self._is_first_iter = True

    def __len__(self):
        """
        Returns the number of rows in the queue.

        Args:
            self: (todo): write your description
        """
        return self._count()

    def __iter__(self):
        """
        Return an iterable. iterable.

        Args:
            self: (todo): write your description
        """
        self._initialize()
        self._is_first_iter = True
        return self

    def __next__(self):
        """
        Returns the next result.

        Args:
            self: (todo): write your description
        """
        if not self._is_first_iter:
            if self._have_next():
                self._move_next()
            else:
                self._finalize()
                raise StopIteration()
        else:
            self._is_first_iter = False
        result = self._get()
        return result

    def _initialize(self):
        """
        Initialize the given value.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError()

    def _finalize(self):
        """
        Finalize the underlyingize.

        Args:
            self: (todo): write your description
        """
        pass

    def _get(self):
        """
        Returns the result of the result.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError()

    def _count(self):
        """
        Return the number of occurrences of this collection.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError()

    def _move_next(self):
        """
        Move the next result.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError()

    def _have_next(self):
        """
        Returns the next result.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError()


class RandomizedDataFlowBase(SimpleDataFlowBase):
    _rng = None

    def __init__(self, seed=None):
        """
        Initialize the seed.

        Args:
            self: (todo): write your description
            seed: (int): write your description
        """
        self._seed = seed

    def _initialize(self):
        """
        Initialize the rng.

        Args:
            self: (todo): write your description
        """
        self._rng = gen_rng(seed=self._seed)
