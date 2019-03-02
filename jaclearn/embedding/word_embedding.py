#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : word_embedding.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/24/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

# This file is adapted from from UKP Lab's project.
# https://github.com/UKPLab/emnlp2017-relation-extraction
# Original copyrights:
# Copyright (C) 2016 UKP lab
#
# Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
#
# Embeddings and vocabulary utility methods

import re
import numpy as np

from jacinle.logging import get_logger

from .constant import EBD_ALL_ZEROS, EBD_UNKNOWN

logger = get_logger(__file__)

special_tokens = {"&ndash;": "–", "&mdash;": "—", "@card@": "0"}


def load_word_index(path, filter=None, format='glove'):
    """
    Loads only the word index from the embeddings file

    @return word to index dictionary
    """
    assert format == 'glove'

    word2idx = {}  # Maps a word to the index in the embeddings matrix

    with open(path, 'r') as fIn:
        idx = 1
        for line in fIn:
            split = line.strip().split(' ')
            if filter is not None and split[0] not in filter:
                continue
            word2idx[split[0]] = idx
            idx += 1

    word2idx[EBD_ALL_ZEROS] = 0
    word2idx[EBD_UNKNOWN] = idx

    return word2idx


def load(path, word_index_only=False, filter=None, format='glove'):
    """
    Loads pre-trained embeddings from the specified path.
    """

    assert format in ('glove', 'fasttext')

    if word_index_only:
        return load_word_index(path, filter=filter, format=format)

    word2idx = {}  # Maps a word to the index in the embeddings matrix
    embeddings = []
    embedding_size = None

    with open(path, 'r', encoding='utf-8') as fIn:
        idx = 1
        for lineno, line in enumerate(fIn):
            if format == 'fasttext':
                if lineno == 0:
                    continue

            try:
                split = line.strip().split(' ')

                if filter is not None and split[0] not in filter:
                    continue

                val = np.array([float(num) for num in split[1:]], dtype='float32')

                if embedding_size is None:
                    embedding_size = len(val)
                else:
                    if embedding_size != len(val):
                        logger.warning('Skip invalid entry (vector length): Line#{}.'.format(lineno))
                        continue

                embeddings.append(val)
                word2idx[split[0]] = idx
                idx += 1
            except ValueError:
                logger.warning('Skip invalid entry (encoding): Line#{}.'.format(lineno))
                # 840D GloVe file has some encoding errors...
                continue

    word2idx[EBD_ALL_ZEROS] = 0
    embeddings.insert(0, np.zeros(embedding_size, dtype='float32'))

    # rare words
    unknown_emb = np.average(np.array(embeddings[-101:]), axis=0)
    embeddings.append(unknown_emb)
    word2idx[EBD_UNKNOWN] = idx
    idx += 1

    return np.array(embeddings, dtype='float32'), word2idx


def map(word, word2idx):
    """
    Get the word index for the given word. Maps all numbers to 0, lowercases if necessary.

    :param word: the word in question
    :param word2idx: dictionary constructed from an embeddings file
    :return: integer index of the word
    """
    unknown_idx = word2idx[EBD_UNKNOWN]
    word = word.strip()
    if word in word2idx:
        return word2idx[word]
    elif word.lower() in word2idx:
        return word2idx[word.lower()]
    elif word in special_tokens:
        return word2idx[special_tokens[word]]
    trimmed = re.sub("(^\W|\W$)", "", word)
    if trimmed in word2idx:
        return word2idx[trimmed]
    elif trimmed.lower() in word2idx:
        return word2idx[trimmed.lower()]
    no_digits = re.sub("([0-9][0-9.,]*)", '0', word)
    if no_digits in word2idx:
        return word2idx[no_digits]
    return unknown_idx


def map_sequence(word_sequence, word2idx):
    """
    Get embedding indices for the given word sequence.

    :param word_sequence: sequence of words to process
    :param word2idx: dictionary of word mapped to their embedding indices
    :return: a sequence of embedding indices
    """
    return [map(word, word2idx) for word in word_sequence]
