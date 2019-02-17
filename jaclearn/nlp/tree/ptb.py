#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : ptb.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/04/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""
PTB-formed constituency trees.
"""

__all__ = ['PTBNode']

import six

from .node import Node
from .traversal import traversal


class PTBNode(Node):
    def __init__(self, vtype, token=None, index=-1):
        super().__init__(vtype, None)
        self.token = token
        self.index = index

    @property
    def leftmost_index(self):
        return self.index if self.is_leaf else self.children[0].leftmost_index

    @property
    def rightmost_index(self):
        return self.index if self.is_leaf else self.children[-1].rightmost_index

    @classmethod
    def from_string(cls, encoding, incl_vtype=True, default_vtype=None):
        if isinstance(encoding, six.string_types):
            steps = encoding.split()
        else:
            assert isinstance(encoding, (tuple, list))
            steps = encoding

        stack = []
        word_id = 0
        for s in steps:
            if s == '(' or s.startswith('('):
                while s.startswith('('):
                    stack.append('(')
                    s = s[1:]
                if len(s) > 0:
                    stack.append(s)
            elif s == ')' or s.endswith(')'):
                nr_right = 0
                while s.endswith(')'):
                    nr_right += 1
                    s = s[:-1]
                if len(s) > 0:
                    stack.append(s)

                for i in range(nr_right):
                    poped = []
                    while True:
                        x = stack.pop()
                        if isinstance(x, six.string_types) and x == '(':
                            break
                        poped.append(x)
                    poped = poped[::-1]

                    if incl_vtype:
                        if len(poped) == 2 and isinstance(poped[1], six.string_types):  # is leaf
                            stack.append(cls(poped[0], poped[1]))
                        else:
                            node = cls(poped[0])
                            for x in poped[1:]:
                                node.append_child(x)
                            stack.append(node)
                    else:
                        if len(poped) == 1 and isinstance(poped[0], six.string_types):  # is leaf
                            stack.append(cls(default_vtype, poped[0]))
                        else:
                            node = cls(default_vtype)
                            for x in poped:
                                if isinstance(x, six.string_types):
                                    x = cls(default_vtype, x)
                                node.append_child(x)
                            stack.append(node)
            else:
                stack.append(s)

        if len(stack) != 1:
            raise ValueError('Invalid PTB encoding.')

        return stack[0]

    def to_string(self, to_string=True, compressed=True, vtype=True):
        if not to_string:
            compressed = False

        def dfs(node):
            if compressed:
                if node.is_leaf:
                    if vtype:
                        yield '({} {})'.format(node.vtype, node.token)
                    else:
                        yield '({})'.format(node.token)
                else:
                    if vtype:
                        yield '({} '.format(node.vtype)
                    else:
                        yield '('
                    for i, x in enumerate(node.children):
                        if i != 0:
                            yield ' '
                        yield from dfs(x)
                    yield ')'
            else:
                yield '('
                if node.is_leaf:
                    if vtype:
                        yield node.vtype
                    yield node.token
                else:
                    if vtype:
                        yield node.vtype
                    for x in node.children:
                        yield from dfs(x)
                yield ')'

        s = list(dfs(self))
        if not to_string:
            return s

        if compressed:
            return ''.join(s)
        return ' '.join(s)

    def to_sentence(self, to_string=True):
        def dfs():
            for node in traversal(self, 'pre'):
                if node.is_leaf:
                    yield node.token
        if not to_string:
            return list(dfs())
        return ' '.join(list(dfs()))

    def __str_node__(self):
        if self.is_leaf:
            return 'VType: {} Token: {}'.format(self.vtype, self.token)
        return 'VType: {}'.format(self.vtype)

    def assign_index(self, start_index=0):
        if not self.is_leaf and self.token is not None:
            raise ValueError('Cannot assign index for trees with non-leaf tokens.')

        if self.is_leaf:
            self.index = start_index
            return self.index + 1

        for c in self.children:
            start_index = c.assign_index(start_index)
        return start_index

