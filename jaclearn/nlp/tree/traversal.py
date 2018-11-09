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
    order = TraversalOrder.from_string(order)

    def dfs(x):
        if order is TraversalOrder.PRE:
            yield x
        for c in x.children:
            yield from dfs(c)
        if order is TraversalOrder.POST:
            yield x

    return dfs(root)


def _shuffled(a):
    a = a.copy()
    random.shuffle_list(a)
    return a


def random_traversal(root, order='pre'):
    order = TraversalOrder.from_string(order)

    def dfs(x):
        if order is TraversalOrder.PRE:
            yield x
        for c in _shuffled(x.children):
            yield from dfs(c)
        if order is TraversalOrder.POST:
            yield x

    return dfs(root)


def is_binary_tree(root):
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
    order = BinaryTraversalOrder.from_string(order)

    def dfs(x):
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

