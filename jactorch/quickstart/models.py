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
        return feed_dict['input']

    def _get_label(self, feed_dict):
        return feed_dict['label']

    def _compose_output(self, value):
        return dict(pred=value)


class MLPModel(MLPLayer):
    pass


class MLPRegressionModel(MLPModel, ModelIOKeysMixin):
    def __init__(self, input_dim, output_dim, hidden_dims, batch_norm=None, dropout=None, activation='relu'):
        super().__init__(input_dim, output_dim, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.loss = nn.MSELoss()

    def forward(self, feed_dict):
        pred = super().forward(self._get_input(feed_dict))
        if self.training:
            loss = self.loss(pred, self._get_label(feed_dict))
            return loss, dict(), dict()
        else:
            return self._compose_output(pred)


class MLPClassificationModel(MLPModel, ModelIOKeysMixin):
    def __init__(self, input_dim, nr_classes, hidden_dims, batch_norm=None, dropout=None, activation='relu'):
        super().__init__(input_dim, nr_classes, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.softmax = nn.Softmax()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, feed_dict):
        logits = super().forward(self._get_input(feed_dict))
        if self.training:
            loss = self.loss(logits, self._get_label(feed_dict))
            return loss, dict(), dict()
        else:
            return self._compose_output(self.softmax(logits))

    def _compose_output(self, value):
        _, pred = value.max(dim=1)
        return dict(prob=value, pred=pred)


class LinearRegressionModel(MLPRegressionModel):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim, [])


class LinearClassificationModel(MLPClassificationModel):
    def __init__(self, input_dim, nr_classes):
        super().__init__(input_dim, nr_classes, [])
