#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : context.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/31/2021
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch.nn as nn
from jactorch.io import load_weights
from jactorch.utils.meta import as_tensor

__all__ = ['NNEnv']


class NNEnv(object):
    """A basic environment that wraps around a nn.Module. This Env supports basic utility functions such as loading a checkpoint."""

    def __init__(self, model: nn.Module):
        """Initialize the environment.

        Args:
            model: the model to be wrapped.
        """
        self._model = model

    @property
    def model(self):
        """Get the model."""
        return self._model

    @property
    def model_unwrapped(self):
        """Get the model, but unwrap the DataParallel if necessary."""
        model = self.model
        if isinstance(model, nn.DataParallel):
            model = model.module
        return model

    def load_weights(self, filename: str, **kwargs):
        """Load weights from a checkpoint file.

        Args:
            filename: the checkpoint file.
        """
        return load_weights(self._model, filename, **kwargs)

    def forward(self, *args, cast_tensor=False, **kwargs):
        """Forward the model. Roughly equivalent to ``self.model(*args, **kwargs)``.

        Args:
            cast_tensor: whether to cast inputs to tensors.
        """
        if cast_tensor:
            args = as_tensor(args)
            kwargs = as_tensor(kwargs)
        outputs = self._model(*args, **kwargs)
        return outputs

