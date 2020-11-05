#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : neural_logic.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/28/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch.nn as nn

from jacinle.utils.enum import JacEnum
from jactorch.nn.cnn import MLPLayer
from jactorch.nn.quickaccess import get_activation

__all__ = ['NeuralLogicInferenceMethod', 'NeuralLogicInferenceBase', 'NeuralLogicInference', 'NeuralLogitsInference']


class NeuralLogicInferenceMethod(JacEnum):
    SKIP = 'skip'
    MLP = 'mlp'


class NeuralLogicInferenceBase(nn.Module):
    def __init__(self, model, input_dim, output_dim, hidden_dim):
        """
        Initialize the model.

        Args:
            self: (todo): write your description
            model: (todo): write your description
            input_dim: (int): write your description
            output_dim: (int): write your description
            hidden_dim: (int): write your description
        """
        super().__init__()
        self.method = NeuralLogicInferenceMethod.from_string(model)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        if self.method is NeuralLogicInferenceMethod.MLP:
            self.layer = nn.Sequential(MLPLayer(input_dim, output_dim, hidden_dim))
        else:
            raise NotImplementedError('Unknown logic inference method: {}.'.format(self.method))

    def forward(self, input):
        """
        R forward forward.

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
        if self.method is NeuralLogicInferenceMethod.SKIP:
            return input

        input_size = input.size()[:-1]
        input_channel = input.size(-1)

        f = input.view(-1, input_channel)
        f = self.layer(f)
        f = f.view(*input_size, -1)
        return f

    def get_output_dim(self, input_dim):
        """
        Returns the dimensions of a dimension.

        Args:
            self: (todo): write your description
            input_dim: (todo): write your description
        """
        if self.method is NeuralLogicInferenceMethod.SKIP:
            return input_dim
        return self.output_dim


class NeuralLogicInference(NeuralLogicInferenceBase):
    def __init__(self, model, input_dim, output_dim, hidden_dim, activation='sigmoid'):
        """
        Initialize the model.

        Args:
            self: (todo): write your description
            model: (todo): write your description
            input_dim: (int): write your description
            output_dim: (int): write your description
            hidden_dim: (int): write your description
            activation: (str): write your description
        """
        super().__init__(model, input_dim, output_dim, hidden_dim)

        if self.method is NeuralLogicInferenceMethod.MLP:
            self.layer.add_module(str(len(self.layer)), get_activation(activation))
        else:
            raise NotImplementedError('Unknown logic inference method: {}.'.format(self.method))


class NeuralLogitsInference(NeuralLogicInferenceBase):
    pass

