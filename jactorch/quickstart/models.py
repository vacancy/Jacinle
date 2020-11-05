#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : models.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/27/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch.nn as nn

from jactorch.nn.cnn.layers import MLPLayer

__all__ = ['MLPModel', 'MLPRegressionModel', 'MLPClassificationModel', 'LinearRegressionModel', 'LinearClassificationModel']


class ModelIOKeysMixin(object):
    def _get_input(self, feed_dict):
        """
        Gets the input for the given feed.

        Args:
            self: (todo): write your description
            feed_dict: (str): write your description
        """
        return feed_dict['input']

    def _get_label(self, feed_dict):
        """
        Return the label for a feed.

        Args:
            self: (todo): write your description
            feed_dict: (dict): write your description
        """
        return feed_dict['label']

    def _compose_output(self, value):
        """
        Compose output value as a dictionary.

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        return dict(pred=value)


class MLPModel(MLPLayer):
    pass


class MLPRegressionModel(MLPModel, ModelIOKeysMixin):
    def __init__(self, input_dim, output_dim, hidden_dims, batch_norm=None, dropout=None, activation='relu'):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            input_dim: (int): write your description
            output_dim: (int): write your description
            hidden_dims: (int): write your description
            batch_norm: (str): write your description
            dropout: (str): write your description
            activation: (str): write your description
        """
        super().__init__(input_dim, output_dim, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.loss = nn.MSELoss()

    def forward(self, feed_dict):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            feed_dict: (dict): write your description
        """
        pred = super().forward(self._get_input(feed_dict))
        if self.training:
            loss = self.loss(pred, self._get_label(feed_dict))
            return loss, dict(), dict()
        else:
            return self._compose_output(pred)


class MLPClassificationModel(MLPModel, ModelIOKeysMixin):
    def __init__(self, input_dim, nr_classes, hidden_dims, batch_norm=None, dropout=None, activation='relu'):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            input_dim: (int): write your description
            nr_classes: (todo): write your description
            hidden_dims: (int): write your description
            batch_norm: (str): write your description
            dropout: (str): write your description
            activation: (str): write your description
        """
        super().__init__(input_dim, nr_classes, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.softmax = nn.Softmax()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, feed_dict):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            feed_dict: (dict): write your description
        """
        logits = super().forward(self._get_input(feed_dict))
        if self.training:
            loss = self.loss(logits, self._get_label(feed_dict))
            return loss, dict(), dict()
        else:
            return self._compose_output(self.softmax(logits))

    def _compose_output(self, value):
        """
        Composes output as a dictionary.

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        _, pred = value.max(dim=1)
        return dict(prob=value, pred=pred)


class LinearRegressionModel(MLPRegressionModel):
    def __init__(self, input_dim, output_dim):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            input_dim: (int): write your description
            output_dim: (int): write your description
        """
        super().__init__(input_dim, output_dim, [])


class LinearClassificationModel(MLPClassificationModel):
    def __init__(self, input_dim, nr_classes):
        """
        Initialize the inputs.

        Args:
            self: (todo): write your description
            input_dim: (int): write your description
            nr_classes: (todo): write your description
        """
        super().__init__(input_dim, nr_classes, [])
