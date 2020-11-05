#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : traversal.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/04/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""
Tree traversal uilties.
"""

import jacinle.random as random
from jacinle.utils.enum import JacEnum

__all__ = ['TraversalOrder', 'traversal']


class TraversalOrder(JacEnum):
    PRE = 'pre'
    POST = 'post'


def traversal(root, order='pre'):
    """
    Iterate over nodes in - order.

    Args:
        root: (todo): write your description
        order: (int): write your description
    """
    order = TraversalOrder.from_string(order)

    def dfs(x):
        """
        Iterate over all dfs by default order.

        Args:
            x: (todo): write your description
        """
        if order is TraversalOrder.PRE:
            yield x
        for c in x.children:
            yield from dfs(c)
        if order is TraversalOrder.POST:
            yield x

    return dfs(root)


def _shuffled(a):
    """
    Return a copy of a list.

    Args:
        a: (array): write your description
    """
    a = a.copy()
    random.shuffle_list(a)
    return a


def random_traversal(root, order='pre'):
    """
    Generate a multi - order.

    Args:
        root: (todo): write your description
        order: (int): write your description
    """
    order = TraversalOrder.from_string(order)

    def dfs(x):
        """
        Iterate over all nodes in a dfs order.

        Args:
            x: (todo): write your description
        """
        if order is TraversalOrder.PRE:
            yield x
        for c in _shuffled(x.children):
            yield from dfs(c)
        if order is TraversalOrder.POST:
            yield x

    return dfs(root)


def is_binary_tree(root):
    """
    Returns true if a binary is binary.

    Args:
        root: (todo): write your description
    """
    for x in traversal(root):
        if not (x.is_leaf or x.nr_children == 2):
            return False
    return True


class BinaryTraversalOrder(JacEnum):
    LR = 'lr'
    NLR = 'nlr'
    LNR = 'lnr'
    LRN = 'lrn'


def binary_traversal(root, order='lnr'):
    """
    Returns a generator that yields a generator of root.

    Args:
        root: (todo): write your description
        order: (int): write your description
    """
    order = BinaryTraversalOrder.from_string(order)

    def dfs(x):
        """
        Iterate over all dfs in x.

        Args:
            x: (todo): write your description
        """
        if x.is_leaf:
            yield x
        else:
            for x in order.val:
                if x == 'l':
                    yield from dfs(x.lson)
                elif x == 'n':
                    yield x
                elif x == 'r':
                    yield from dfs(x.rson)

    return dfs(root)

