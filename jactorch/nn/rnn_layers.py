#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : rnn_layers.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/21/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn

from jactorch.functional.indexing import index_one_hot_ellipsis
from jactorch.nn.rnn_utils import rnn_with_length
from jactorch.utils.meta import as_tensor

__all__ = ['RNNLayer', 'LSTMLayer', 'GRULayer']


# TODO(Jiayuan Mao @ 04/21): support rnn_cell as input.
class RNNLayerBase(nn.Module):
    """Basic RNN layer. Will be inherited by concreate implementations."""

    def __init__(self, input_dim, hidden_dim, nr_layers,
            bias=True, batch_first=True, dropout=0, bidirectional=False):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nr_layers = nr_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.rnn = type(self).__rnn_class__(input_dim, hidden_dim, nr_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        self.reset_parameters()

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def reset_parameters(self):
        for name, weight in self.rnn.named_parameters():
            if name.startswith('weight'):
                nn.init.orthogonal_(weight)
            else:
                assert name.startswith('bias')
                weight.data.zero_()

    def forward(self, input, input_lengths, sorted=False):
        initial_states = self.zero_state(input)
        rnn_output, last_output = rnn_with_length(self.rnn, input, input_lengths, initial_states, batch_first=self.batch_first, sorted=sorted)
        return rnn_output, self.extract_last_output(last_output)

    def zero_state(self, input):
        batch_dim = 0 if self.batch_first else 1
        batch_size = input.size(batch_dim)
        hidden_size = self.rnn.hidden_size
        nr_layers = self.rnn.num_layers * (int(self.rnn.bidirectional) + 1)
        state_shape = (nr_layers, batch_size, self.rnn.hidden_size)

        storage = as_tensor(input)
        gen = lambda: torch.zeros(*state_shape, device=input.device)
        if self.state_is_tuple:
            return (gen(), gen())
        return gen()

    def extract_last_output(self, rnn_last_output):
        if self.rnn.bidirectional:
            extract = lambda x: torch.cat((x[-2], x[-1]), dim=-1)
        else:
            extract = lambda x: x[-1]
        if type(rnn_last_output) is tuple:
            return tuple(map(extract, rnn_last_output))
        return extract(rnn_last_output)

    @property
    def state_is_tuple(self):
        return 'lstm' in type(self.rnn).__name__.lower()


class RNNLayer(RNNLayerBase):
    __rnn_class__ = nn.RNN


class LSTMLayer(RNNLayerBase):
    __rnn_class__ = nn.LSTM


class GRULayer(RNNLayerBase):
    __rnn_class__ = nn.GRU

