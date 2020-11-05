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
        """
        Returns the value of a given attribute.

        Args:
            self: (todo): write your description
            k: (str): write your description
        """
        if k not in self:
            raise AttributeError
        return self[k]

    def __setattr__(self, k, v):
        """
        Set the given attribute of k.

        Args:
            self: (todo): write your description
            k: (todo): write your description
            v: (todo): write your description
        """
        self[k] = v

    def __delattr__(self, k):
        """
        Remove an attribute from a key.

        Args:
            self: (todo): write your description
            k: (todo): write your description
        """
        del self[k]

    def format(self, sep=': ', end='\n'):
        """
        Formats the formatted string.

        Args:
            self: (todo): write your description
            sep: (todo): write your description
            end: (int): write your description
        """
        return kvformat(self, sep=sep, end=end)

    def print(self, sep=': ', end='\n', file=None):
        """
        Print the kv - thumbnails.

        Args:
            self: (todo): write your description
            sep: (str): write your description
            end: (int): write your description
            file: (str): write your description
        """
        return kvprint(self, sep=sep, end=end, file=file)


g = G()


class GView(object):
    def __init__(self, dict_=None):
        """
        Initialize the object.

        Args:
            self: (todo): write your description
            dict_: (todo): write your description
        """
        if dict_ is None:
            dict_ = dict()
        object.__setattr__(self, '_dict', dict_)

    def __getattr__(self, k):
        """
        Get the value of a raw attribute.

        Args:
            self: (todo): write your description
            k: (str): write your description
        """
        if k not in self.raw():
            raise AttributeError
        return self.raw()[k]

    def __setattr__(self, k, v):
        """
        Set the value of a key.

        Args:
            self: (todo): write your description
            k: (todo): write your description
            v: (todo): write your description
        """
        self.raw()[k] = v

    def __delattr__(self, k):
        """
        Remove an attribute from the dictionary.

        Args:
            self: (todo): write your description
            k: (todo): write your description
        """
        del self.raw()[k]

    def __getitem__(self, k):
        """
        Return the first item from the queue.

        Args:
            self: (todo): write your description
            k: (todo): write your description
        """
        return self.raw()[k]

    def __setitem__(self, k, v):
        """
        Set the value of a key.

        Args:
            self: (todo): write your description
            k: (todo): write your description
            v: (todo): write your description
        """
        self.raw()[k] = v

    def __delitem__(self, k):
        """
        Remove an item from the queue.

        Args:
            self: (todo): write your description
            k: (todo): write your description
        """
        del self.raw()[k]

    def __contains__(self, k):
        """
        Returns true if k is contained in k.

        Args:
            self: (todo): write your description
            k: (str): write your description
        """
        return k in self.raw()

    def __iter__(self):
        """
        Returns an iterator over the raw () method.

        Args:
            self: (todo): write your description
        """
        return iter(self.raw().items())

    def raw(self):
        """
        Returns the raw object.

        Args:
            self: (todo): write your description
        """
        return object.__getattribute__(self, '_dict')

    def update(self, other):
        """
        Update the set with the other.

        Args:
            self: (todo): write your description
            other: (todo): write your description
        """
        self.raw().update(other)

    def copy(self):
        """
        Returns a copy of the current view.

        Args:
            self: (todo): write your description
        """
        return GView(self.raw().copy())

    def format(self, sep=': ', end='\n'):
        """
        Return a formatted string as a string.

        Args:
            self: (todo): write your description
            sep: (todo): write your description
            end: (int): write your description
        """
        return kvformat(self.raw(), sep=sep, end=end)

    def print(self, sep=': ', end='\n', file=None):
        """
        Print the contents of the segment representation.

        Args:
            self: (todo): write your description
            sep: (str): write your description
            end: (int): write your description
            file: (str): write your description
        """
        return kvprint(self.raw(), sep=sep, end=end, file=file)


class SlotAttrObject(object):
    def __init__(self, **kwargs):
        """
        Initialize an attribute.

        Args:
            self: (todo): write your description
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    def update(self, **kwargs):
        """
        Updates the object s attributes from the model.

        Args:
            self: (todo): write your description
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __setattr__(self, k, v):
        """
        Set the attribute of the object.

        Args:
            self: (todo): write your description
            k: (str): write your description
            v: (todo): write your description
        """
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
        """
        Returns a copy of this object

        Args:
            self: (todo): write your description
        """
        return copy.deepcopy(self)


class OrderedSet(object):
    def __init__(self, initial_list=None):
        """
        Initialize the object.

        Args:
            self: (todo): write your description
            initial_list: (list): write your description
        """
        if initial_list is not None:
            self._dict = collections.OrderedDict([(v, True) for v in initial_list])
        else:
            self._dict = collections.OrderedDict()

    def append(self, value):
        """
        Append a new value to the list.

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        self._dict[value] = True

    def remove(self, value):
        """
        Remove a value from the dictionary.

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        del self._dict[value]

    def __contains__(self, value):
        """
        Returns true if the value is contained in this dict.

        Args:
            self: (todo): write your description
            value: (str): write your description
        """
        return value in self._dict

    def __iter__(self):
        """
        Return an iterator over all the keys.

        Args:
            self: (todo): write your description
        """
        return self._dict.keys()

    def as_list(self):
        """
        Return a list as a list.

        Args:
            self: (todo): write your description
        """
        return list(self._dict.keys())

