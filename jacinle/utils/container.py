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
from typing import Any, Optional, Iterable

from jacinle.utils.meta import repr_from_str
from jacinle.utils.printing import kvformat, kvprint

__all__ = ['G', 'g', 'GView', 'SlotAttrObject', 'OrderedSet']


class G(dict):
    """A simple container that wraps a dict and provides attribute access to the dict.

    Example:
        >>> g = G()
        >>> g.a = 1
        >>> g['b'] = 2

    """
    def __getattr__(self, k):
        if k not in self:
            raise AttributeError
        return self[k]

    def __setattr__(self, k, v):
        self[k] = v

    def __delattr__(self, k):
        del self[k]

    def format(self, sep=': ', end='\n'):
        """Format the dict as a string using :func:`jacinle.utils.printing.kvformat`."""
        return kvformat(self, sep=sep, end=end)

    def print(self, sep=': ', end='\n', file=None):
        """Print the dict using :func:`jacinle.utils.printing.kvprint`."""
        return kvprint(self, sep=sep, end=end, file=file)


g = G()
"""A simple global dict-like object."""


class GView(object):
    """A simple container that wraps a dict and provides attribute access to the dict.
    In contrast to :class:`G`, this class wraps around an existing dict."""

    def __init__(self, dict_=None):
        """Initialize the container."""
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
        """Return the underlying :class:`dict`."""
        return object.__getattribute__(self, '_dict')

    def update(self, other):
        """Update the underlying dict with another dict."""
        self.raw().update(other)

    def keys(self):
        """Return the keys of the underlying dict."""
        return self.raw().keys()

    def values(self):
        """Return the values of the underlying dict."""
        return self.raw().values()

    def items(self):
        """Iterate over the items of the underlying dict."""
        return self.raw().items()

    def copy(self):
        """Return a shallow copy of the underlying dict."""
        return GView(self.raw().copy())

    def format(self, sep=': ', end='\n'):
        """Format the dict as a string using :func:`jacinle.utils.printing.kvformat`."""
        return kvformat(self.raw(), sep=sep, end=end)

    def print(self, sep=': ', end='\n', file=None):
        """Print the dict using :func:`jacinle.utils.printing.kvprint`."""
        return kvprint(self.raw(), sep=sep, end=end, file=file)

    def __str__(self):
        return self.format()

    __repr__ = repr_from_str


class SlotAttrObject(object):
    """Create a object that allows only a fixed set of attributes to be set."""

    def __init__(self, **kwargs):
        """Initialize the object.

        Args:
            kwargs: the attributes to be set.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    def update(self, **kwargs):
        """Update the attributes of the object."""
        for k, v in kwargs.items():
            setattr(self, k, v)

    def clone(self, deep: bool = False):
        """Return a shallow or a deep copy of the object."""
        if deep:
            return copy.deepcopy(self)
        return copy.copy(self)

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


class OrderedSet(object):
    """A set that keeps the order of the elements."""

    def __init__(self, initial_list: Optional[Iterable[Any]] = None):
        """Initialize the set.

        Args:
            initial_list: the initial list of elements.
        """
        if initial_list is not None:
            self._dict = collections.OrderedDict([(v, True) for v in initial_list])
        else:
            self._dict = collections.OrderedDict()

    def as_list(self):
        """Return the elements as a list."""
        return list(self._dict.keys())

    def append(self, value):
        """Append an element to the set."""
        self._dict[value] = True

    def remove(self, value):
        """Remove an element from the set."""
        del self._dict[value]

    def __contains__(self, value):
        return value in self._dict

    def __iter__(self):
        yield from self._dict.keys()
