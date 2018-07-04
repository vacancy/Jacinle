#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test-torch-conv.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/28/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import unittest

import torch
import jactorch.nn as jacnn
from jactorch.utils.meta import as_variable


class TestTorchConv(unittest.TestCase):
    def _test_conv_size(self, in_tensor, conv, out_size, **kwargs):
        self.assertEqual(conv(in_tensor, **kwargs).size(), out_size)

    def test_conv2d(self):
        in_tensor = as_variable(torch.randn(16, 8, 4, 2))
        self._test_conv_size(in_tensor, jacnn.Conv2d(8, 16, 3, padding=1), (16, 16, 4, 2))
        self._test_conv_size(in_tensor, jacnn.Conv2d(8, 16, 2, stride=2, padding_mode='valid'), (16, 16, 2, 1))
        self._test_conv_size(in_tensor, jacnn.Conv2d(8, 16, 3, padding_mode='same'), (16, 16, 4, 2))
        self._test_conv_size(in_tensor, jacnn.Conv2d(8, 16, (5, 3), padding_mode='same', border_mode='replicate'), (16, 16, 4, 2))
        self._test_conv_size(in_tensor, jacnn.Conv2d(8, 16, (5, 3), padding_mode='same', border_mode='reflect'), (16, 16, 4, 2))

    def test_deconv2d(self):
        in_tensor = as_variable(torch.randn(16, 8, 4, 2))
        self._test_conv_size(in_tensor, jacnn.ConvTranspose2d(8, 16, 3, padding_mode='same'), (16, 16, 4, 2))
        self._test_conv_size(in_tensor, jacnn.ConvTranspose2d(8, 16, 3, 2, padding_mode='same'), (16, 16, 8, 4), scale_factor=2)

    def test_resize_conv2d(self):
        in_tensor = as_variable(torch.randn(16, 8, 4, 2))
        self._test_conv_size(in_tensor, jacnn.ResizeConv2d(8, 16, 3, scale_factor=2), (16, 16, 8, 4))

    def test_deconv2d(self):
        in_tensor = as_variable(torch.randn(16, 8, 4, 2))
        self._test_conv_size(in_tensor, jacnn.Deconv2dLayer(8, 16, 3, scale_factor=2), (16, 16, 8, 4))
        self._test_conv_size(in_tensor, jacnn.Deconv2dLayer(8, 16, 3, scale_factor=2, algo='convtranspose'), (16, 16, 8, 4))


if __name__ == '__main__':
    unittest.main()