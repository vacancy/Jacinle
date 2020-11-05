#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test-torch-inferencer.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import unittest

import numpy as np
import torch.nn as nn

from jactorch.quickstart.inference import ModelInferencer, AsyncModelInferencer, BatchedAsyncModelInferencer


class SimpleModel(nn.Module):
    def forward(self, feed_dict):
        """
        Takes a dictionary of keys for a dictionary

        Args:
            self: (todo): write your description
            feed_dict: (dict): write your description
        """
        return feed_dict['input'] + 1


class TestTorchInferencer(unittest.TestCase):
    def test_basic_inference(self):
        """
        Test the test test.

        Args:
            self: (todo): write your description
        """
        inferencer = ModelInferencer(SimpleModel())
        with inferencer.activate():
            result = inferencer.inference(dict(input=np.zeros(1, dtype='float32')))
        self.assertEqual(float(result), 1)

    def test_async_inference(self):
        """
        Run inference.

        Args:
            self: (todo): write your description
        """
        inferencer = AsyncModelInferencer(SimpleModel())
        results = []
        with inferencer.activate():
            for i in range(16):
                results.append(inferencer.inference(dict(input=np.zeros(1, dtype='float32') + i)))
        for i, r in enumerate(results):
            self.assertEqual(float(r.get_result()), 1 + i)

    def test_batched_async_inference(self):
        """
        Test ifference inference.

        Args:
            self: (todo): write your description
        """
        inferencer = BatchedAsyncModelInferencer(SimpleModel())
        results = []
        with inferencer.activate():
            for i in range(16):
                results.append(inferencer.inference(dict(input=np.zeros(1, dtype='float32') + i)))
        for i, r in enumerate(results):
            self.assertEqual(float(r.get_result()), 1 + i)


if __name__ == '__main__':
    unittest.main()
