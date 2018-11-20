#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : constituency.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/04/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""
Constituency Tree.
"""

import jacinle.random as random
from jacinle.utils.enum import JacEnum

from .ptb import PTBNode
from .traversal import traversal

TEMP_NODE = '<TEMP>'


def _new_temp_node(token=None):
    return PTBNode(TEMP_NODE, token)


def binarize_tree(tree):
    def dc(root, children):
        n = len(children)
        if n == 1:
            return children[0]
        lhs = children[:n // 2]
        rhs = children[n // 2:]

        for part in [lhs, rhs]:
            if len(part) == 1:
                root.append_child(part[0])
            else:
                imm = _new_temp_node()
                imm.attach(root)
                dc(imm, part)

    def dfs(node):
        for x in node.children:
            dfs(x)

        n = len(node.children)
        if n == 0:
            pass
        elif n == 1:
            y, z = node, node.children[0]
            x, sibling_ind = y.detach()
            z.detach()
            z.vtype = y.vtype
            if x is None:
                node = z
            else:
                z.attach(x, sibling_ind)
        elif n == 2:
            pass
        else:
            children = node.children.copy()
            for x in children:
                x.detach()
            dc(node, children)
        return node

    return dfs(tree.clone())


def make_balanced_binary_tree(sequence):
    root = _new_temp_node()
    for x in sequence:
        _new_temp_node(x).attach(root)
    return binarize_tree(root)



class StepMaskSelectionMode(JacEnum):
    FIRST = 'first'
    RANDOM = 'random'


def compose_bianry_tree_step_masks(tree, selection='first'):
    selection = StepMaskSelectionMode.from_string(selection)
    nodes = list(traversal(tree, 'pre'))
    clean_nodes = {x for x in nodes if x.is_leaf}
    ever_clean_nodes = clean_nodes.copy()

    answer = []

    while len(clean_nodes) > 1:
        # all allowed nodes
        allowed = {x: i for i, x in enumerate(nodes) if (
            x not in ever_clean_nodes and
            all(map(lambda y: y in clean_nodes, x.children))
        )}

        # project it to
        allowed_projected = {x for x in clean_nodes if (
            x.sibling_ind == 0 and x.father in allowed
        )}

        ordered_clean_nodes = [x for x in nodes if x in clean_nodes]
        clean_nodes_indices = {x: i for i, x in enumerate(ordered_clean_nodes)}

        if selection is StepMaskSelectionMode.FIRST:
            selected = nodes[min(allowed.values())]
        elif selection is StepMaskSelectionMode.RANDOM:
            selected = random.choice_list(list(allowed))
        else:
            raise ValueError('Unknown StepMaskSelectionMode: {}.'.format(selection))

        mask_allowed_projected = [1 if x in allowed_projected else 0 for x in ordered_clean_nodes]
        assert len(selected.children) == 2

        # sanity check.
        lson = clean_nodes_indices[selected.children[0]]
        rson = clean_nodes_indices[selected.children[1]]
        assert lson + 1 == rson

        clean_nodes.difference_update(selected.children)
        clean_nodes.add(selected)
        ever_clean_nodes.add(selected)

        answer.append((lson, mask_allowed_projected))

    return answer

