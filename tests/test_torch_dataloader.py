# -*- coding: utf-8 -*-
# File   : test_torch_dataloader.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/03/2018
#
# This file is part of Jacinle.

import time
import unittest

import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

import jacinle.random as random
from jactorch.data.dataloader import JacDataLoader
from jactorch.utils.meta import as_tensor, as_float


class _FakeDataset(Dataset):
    _value = None

    def __len__(self):
        return 2

    def __getitem__(self, index):
        if self._value is None:
            self._value = random.rand()
        time.sleep(0.1)
        return as_tensor(np.array([self._value]))


def _my_init_func(worker_id, msg):
    print('Worker #{}: {}. (Seed: {})'.format(worker_id, msg, random.get_state()[1].std()))


class TestTorchDataLoader(unittest.TestCase):
    def test_torch_dataloader(self):
        ds = _FakeDataset()
        dl = DataLoader(ds, num_workers=2)
        res = list(dl)
        self.assertEqual(as_float(res[0]), as_float(res[1]))

    def test_jac_dataloader(self):
        ds = _FakeDataset()
        dl = JacDataLoader(ds, num_workers=2, init_func=_my_init_func, init_args=[('hello', ), ('world', )])
        res = list(dl)
        self.assertNotEqual(as_float(res[0]), as_float(res[1]))


if __name__ == '__main__':
    unittest.main()
