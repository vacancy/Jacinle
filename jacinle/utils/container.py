#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : container.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import copy
import collections

from .printing import kvformat, kvprint

__all__ = ['G', 'g', 'GView', 'SlotAttrObject', 'OrderedSet']


class G(dict):
    def __getattr__(self, k):
        if k not in self:
            raise AttributeError
        return self[k]

    def __setattr__(self, k, v):
        self[k] = v

    def __delattr__(self, k):
        del self[k]

    def format(self, sep=': ', end='\n'):
        return kvformat(self, sep=sep, end=end)

    def print(self, sep=': ', end='\n', file=None):
        return kvprint(self, sep=sep, end=end, file=file)


g = G()


class GView(object):
    def __init__(self, dict_=None):
        if dict_ is None:
            dict_ = dict()
        object.__setattr__(self, '_dict', dict_)

    def __getattr__(self, k):
        if k not in self.raw():
            raise AttributeError
        return self.raw()[k]

    def __setattr__(self, k, v):
        self.raw()[k] = v

    def __delattr__(self, k):
        del self.raw()[k]

    def __getitem__(self, k):
        return self.raw()[k]

    def __setitem__(self, k, v):
        self.raw()[k] = v

    def __delitem__(self, k):
        del self.raw()[k]

    def __contains__(self, k):
        return k in self.raw()

    def __iter__(self):
        return iter(self.raw().items())

    def raw(self):
        return object.__getattribute__(self, '_dict')

    def update(self, other):
        self.raw().update(other)

    def copy(self):
        return GView(self.raw().copy())

    def format(self, sep=': ', end='\n'):
        return kvformat(self.raw(), sep=sep, end=end)

    def print(self, sep=': ', end='\n', file=None):
        return kvprint(self.raw(), sep=sep, end=end, file=file)


class SlotAttrObject(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __setattr__(self, k, v):
        assert not k.startswith('_')
        if k not in type(self).__dict__:
            # do not use hasattr; since it may result in infinite recursion
            raise AttributeError(
                '{}: could not set non-existing attribute {}'.format(
                    self, k))
        cvt = getattr(type(self), '_convert_{}'.format(k), None)
        if cvt is not None:
            v = cvt(v)
        super().__setattr__(k, v)

    def clone(self):
        return copy.deepcopy(self)


class OrderedSet(object):
    def __init__(self, initial_list=None):
        if initial_list is not None:
            self._dict = collections.OrderedDict([(v, True) for v in initial_list])
        else:
            self._dict = collections.OrderedDict()

    def append(self, value):
        self._dict[value] = True

    def remove(self, value):
        del self._dict[value]

    def __contains__(self, value):
        return value in self._dict

    def __iter__(self):
        return self._dict.keys()

    def as_list(self):
        return list(self._dict.keys())

