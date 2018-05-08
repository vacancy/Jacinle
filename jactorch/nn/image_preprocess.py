#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : image_preprocess.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/04/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import numpy as np
import torch.nn as nn

from jactorch.graph.variable import new_var_with

__all__ = ['ImageNetNormalizer']


class ImageNetNormalizer(nn.Module):
    v_imagenet_means = np.array([0.485, 0.456, 0.406], dtype='float32').reshape(1, 3, 1, 1)
    v_imagenet_stds = np.array([0.229, 0.224, 0.225], dtype='float32').reshape(1, 3, 1, 1)

    def forward(self, x):
        imagenet_means = new_var_with(x, self.v_imagenet_means)
        imagenet_stds = new_var_with(x, self.v_imagenet_stds)

        x /= 255.
        x -= imagenet_means
        x /= imagenet_stds
        return x
