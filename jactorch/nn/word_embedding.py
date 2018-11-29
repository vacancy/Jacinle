#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : word_embedding.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 06/15/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['WordEmbedding']


class WordEmbedding(nn.Module):
    def __init__(self, word_embeddings, nr_extra_words, fake=False):
        super().__init__()
        self.nr_words = word_embeddings.shape[0]
        self.nr_extra_words = nr_extra_words
        self.nr_tot_words = self.nr_words + self.nr_extra_words
        self.embedding_dim = word_embeddings.shape[1]
        self.fake = fake

        self.impl = nn.Embedding(self.nr_tot_words, self.embedding_dim, padding_idx=0)
        if not fake:
            self.word_embeddings = nn.Parameter(torch.tensor(word_embeddings))
            self.word_embeddings.requires_grad = False
            self.extra_word_embeddings = nn.Parameter(torch.zeros(nr_extra_words, self.embedding_dim,
                dtype=self.word_embeddings.dtype, device=self.word_embeddings.device))
            self.extra_word_embeddings.requires_grad = True
            self.impl.weight = nn.Parameter(torch.cat((self.word_embeddings, self.extra_word_embeddings), dim=0))

        self.reset_parameters()

    def reset_parameters(self):
        if not self.fake:
            self.extra_word_embeddings.data.normal_(
                self.word_embeddings.data.mean(),
                self.word_embeddings.data.std()
            )

    @property
    def weight(self):
        return self.impl.weight

    def forward(self, words):
        return self.impl(words)

