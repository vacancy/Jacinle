#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : embedding_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/24/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import numpy as np

import jacinle.random as rand
from .constant import EBD_UNKNOWN, EBD_ALL_ZEROS


def make_element2idx(elements_to_embed, add_all_zeros=False, add_unknown=False):
    elements_to_embed = sorted(elements_to_embed)
    element2idx = {EBD_ALL_ZEROS: 0} if add_all_zeros else {}
    element2idx.update({el: idx for idx, el in enumerate(elements_to_embed, start=len(element2idx))})
    if add_unknown:
        element2idx[EBD_UNKNOWN] = len(element2idx)
    return element2idx


def init_random(elements_to_embed, embedding_size, add_all_zeros=False, add_unknown=False):
    """
    Initialize a random embedding matrix for a collection of elements. Elements are sorted in order to ensure
    the same mapping from indices to elements each time.

    :param elements_to_embed: collection of elements to construct the embedding matrix for
    :param embedding_size: size of the embedding
    :param add_all_zeros: add a all_zero embedding at index 0
    :param add_unknown: add unknown embedding at the last index
    :return: an embedding matrix and a dictionary mapping elements to rows in the matrix
    """
    element2idx = make_element2idx(elements_to_embed, add_all_zeros=add_all_zeros, add_unknown=add_unknown)

    embeddings = rand.random((len(element2idx), embedding_size)).astype('f')
    if add_all_zeros:
        embeddings[0] = np.zeros([embedding_size])

    return embeddings, element2idx
