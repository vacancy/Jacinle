#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test-torch-trainer.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/28/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import unittest

import numpy as np
from torch.utils.data import DataLoader

import jacinle.random as random
from jactorch.quickstart.train import ModelTrainer
from jactorch.quickstart.models import LinearClassificationModel, MLPClassificationModel


def _make_data_loader(nr_data):
    def _generate_fake_data():
        values = []
        for i in range(nr_data):
            input = random.normal(size=(2, )).astype('float32')
            label = int(input.sum() > 0)
            values.append(dict(input=input, label=label))
        return values

    data = _generate_fake_data()
    return DataLoader(data, batch_size=8, shuffle=True)


def _eval_accuracy(fd, od):
    return {'accuracy': np.equal(od['pred'], fd['label']).astype('float32').mean()}


class TestTorchTrainer(unittest.TestCase):
    def test_train_linear(self):
        model = LinearClassificationModel(2, 2)
        self._test_model(model)

    def test_train_mlp(self):
        model = MLPClassificationModel(2, 2, [10], dropout=True)
        self._test_model(model)

    def _test_model(self, model):
        trainer = ModelTrainer(model, 'Adam', lr=0.1)
        trainer.train(_make_data_loader(128), 50, print_interval=10)
        result = trainer.validate(_make_data_loader(128), _eval_accuracy)
        self.assertGreater(result['accuracy'], 0.98)


if __name__ == '__main__':
    unittest.main()
