#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : unittest.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/27/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import unittest

import numpy as np
from torch.autograd import Variable


def _as_numpy(v):
    if isinstance(v, np.ndarray):
        return v
    if isinstance(v, Variable):
        v = v.data
    return v.cpu().numpy()


class TorchTestCase(unittest.TestCase):
    def assertTensorClose(self, a, b, atol=1e-3, rtol=1e-3):
        npa, npb = _as_numpy(a), _as_numpy(b)
        self.assertTrue(
                np.allclose(npa, npb, atol=atol),
                'Tensor close check failed\n{}\n{}\nadiff={}, rdiff={}'.format(
                    a, b, np.abs(npa - npb).max(), np.abs((npa - npb) / np.fmax(npa, 1e-5)).max()
                )
        )
