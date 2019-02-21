#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : node.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/04/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""
The definition for tree Nodes.
"""

from copy import deepcopy

__all__ = ['Node']


class Node(object):
    def __init__(self, vtype, etype=None):
        self.vtype = vtype
        self.etype = etype
        self.children = []
        self.father = None
        self.sibling_ind = -1

    @property
    def nr_children(self):
        return len(self.children)

    @property
    def size(self):
        return 1 + sum(c.size for c in self.children)

    @property
    def nr_leaves(self):
        if self.is_leaf:
            return 1
        return sum(c.nr_leaves for c in self.children)

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def lson(self):
        assert len(self.children) == 2
        return self.childre[0]

    @property
    def rson(self):
        assert len(self.children) == 2
        return self.children[1]

    @property
    def depth(self):
        """
        Depth is defined as the number of nodes on the maximum distance with the root of nodes + 1.
        (Thus a single nodes will have depth 1.)
        """
        if self.is_leaf:
            return 1
        return max([c.depth for c in self.children]) + 1

    @property
    def breadth(self):
        """
        Breadth is defined as the maximum number of children of nodes in the tree.
        """
        if self.is_leaf:
            return 1
        return max(max([c.breath for c in self.children]), len(self.children))

    def clone(self):
        return deepcopy(self)

    def insert_child(self, pos, node):
        node.father = self
        self.children.insert(pos, node)
        self._refresh_sibling_inds()

        return node

    def remove_child(self, node):
        assert self.children[node.sibling_ind] == node
        self.children.remove(node)
        self._refresh_sibling_inds()

        rv = node.father, node.sibling_ind
        node.father = None
        node.sibling_ind = -1
        return rv

    def append_child(self, node):
        node.father = self
        node.sibling_ind = len(self.children)
        self.children.append(node)

        return node

    def attach(self, father, sibling_ind=-1):
        """
        Attach to a new father.
        """
        if sibling_ind == -1:
            return father.append_child(self)
        return father.insert_child(sibling_ind, self)

    def detach(self):
        """
        Detach from the father.
        """
        return self.father.remove_child(self)

    def _refresh_sibling_inds(self):
        for i, c in enumerate(self.children):
            c.sibling_ind = i

    def __str_node__(self):
        if self.etype is not None:
            return 'VType: {} EType: {}'.format(self.vtype, self.etype)
        return 'VType: {}'.format(self.vtype)

    def __str__(self):
        results = [self.__str_node__()]
        for c in self.children:
            lines = str(c).split('\n')
            results.extend(['  ' + l for l in lines])
        return '\n'.join(results)

    def __repr__(self):
        return str(self)

