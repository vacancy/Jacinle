#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : train.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/27/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import time

import torch

from jacinle.utils.meter import GroupMeters
from jactorch.optim.quickaccess import get_optimizer
from jactorch.utils.meta import as_numpy, as_float, as_tensor
from jacinle.logging import get_logger

logger = get_logger(__file__)

__all__ = ['ModelTrainer']


class ModelTrainer(object):
    def __init__(self, model, optimizer, lr=0.01, weight_decay=0, **opt_kwargs):
        optimizer = get_optimizer(optimizer, model, lr=lr, weight_decay=weight_decay, **opt_kwargs)
        self._model = model
        self._optimizer = optimizer

    def train_step(self, feed_dict, meters=None):
        assert self._model.training
        feed_dict = as_tensor(feed_dict)

        self._optimizer.zero_grad()
        loss, monitors, output_dict = self._model(feed_dict)
        loss.backward()
        self._optimizer.step()

        loss, monitors = map(as_float, [loss, monitors])
        if meters is not None:
            meters.update(loss=loss)
            meters.update(monitors)

        return as_float(loss)

    def train_epoch(self, data_loader, meters=None):
        if meters is None:
            meters = GroupMeters()

        self._model.train()
        end = time.time()
        for fd in data_loader:
            data_time = time.time() - end; end = time.time()
            self.train_step(fd, meters=meters)
            step_time = time.time() - end; end = time.time()
            meters.update({'time/data': data_time, 'time/step': step_time})
        return meters

    def train(self, data_loader, nr_epochs, verbose=True, meters=None, early_stop=None, print_interval=1):
        if meters is None:
            meters = GroupMeters()

        for epoch in range(1, 1 + nr_epochs):
            meters.reset()
            self.train_epoch(data_loader, meters=meters)
            if verbose and epoch % print_interval == 0:
                caption = 'Epoch: {}:'.format(epoch)
                logger.info(meters.format_simple(caption))
            if early_stop is not None:
                flag = early_stop(self._model)
                if flag:
                    break

    def validate_step(self, feed_dict, metric, meters=None):
        feed_dict_np = as_numpy(feed_dict)
        feed_dict = as_tensor(feed_dict)
        with torch.no_grad():
            output_dict = self._model(feed_dict)
        output_dict_np = as_numpy(output_dict)
        result = as_float(metric(feed_dict_np, output_dict_np))
        if meters is not None:
            meters.update(result)
        return result

    def validate(self, data_loader, metric, meters=None):
        if meters is None:
            meters = GroupMeters()

        self._model.eval()
        end = time.time()
        for fd in data_loader:
            data_time = time.time() - end; end = time.time()
            self.validate_step(fd, metric, meters=meters)
            step_time = time.time() - end; end = time.time()
            meters.update({'time/data': data_time, 'time/step': step_time})

        return meters.avg
