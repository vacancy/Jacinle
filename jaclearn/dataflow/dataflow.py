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
        pass

    def _reset(self):
        pass

    def _gen(self):
        raise NotImplementedError()

    def _finalize(self):
        pass

    def _len(self):
        return None

    def __len__(self):
        try:
            return self._len()
        except TypeError:
            return None

    def __iter__(self):
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
        self._unwrapped = other

    @property
    def unwrapped(self):
        return self._unwrapped

    def _gen(self):
        for item in self._unwrapped:
            yield item

    def _len(self):
        return len(self._unwrapped)


class AdvancedDataFlowBase(DataFlowBase):
    def __init__(self):
        self._is_first_iter = True

    def __len__(self):
        return self._count()

    def __iter__(self):
        self._initialize()
        self._is_first_iter = True
        return self

    def __next__(self):
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
        raise NotImplementedError()

    def _finalize(self):
        pass

    def _get(self):
        raise NotImplementedError()

    def _count(self):
        raise NotImplementedError()

    def _move_next(self):
        raise NotImplementedError()

    def _have_next(self):
        raise NotImplementedError()


class RandomizedDataFlowBase(SimpleDataFlowBase):
    _rng = None

    def __init__(self, seed=None):
        self._seed = seed

    def _initialize(self):
        self._rng = gen_rng(seed=self._seed)
