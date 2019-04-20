#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : env.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/09/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import os.path as osp
import time

import torch
import torch.nn as nn

from jacinle.event.registry import SimpleEventRegistry
from jacinle.logging import get_logger
from jactorch.io import load_weights
from jactorch.utils.meta import as_tensor, as_float, as_cpu

logger = get_logger(__file__)

__all__ = ['TrainerEnv']


def default_reduce_func(k, v):
    return v.mean()


class TrainerEnv(object):
    def __init__(self, model, optimizer):
        self._model = model
        self._optimizer = optimizer

        self._train_loader = None
        self._validation_loader = None
        self._event_manager = SimpleEventRegistry({
            'epoch:before', 'epoch:after',
            'step:before', 'step:after',
            'forward:before', 'forward:after',
            'backward:before', 'backward:after',
        })

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    def register_event(self, name, callback):
        logger.info('Register trainer event: name={}, callback={}.'.format(name, callback.__module__ + '.' + callback.__name__))
        self._event_manager.register(name, callback)

    def trigger_event(self, name, *args, **kwargs):
        self._event_manager.trigger(name, *args, **kwargs)

    def save_checkpoint(self, filename, extra=None):
        # Hack the data parallel.
        model = self._model
        if isinstance(model, nn.DataParallel):
            model = model.module

        state = {
            'model': as_cpu(model.state_dict()),
            'optimizer': as_cpu(self._optimizer.state_dict()),
            'extra': extra
        }
        try:
            torch.save(state, filename)
            logger.info('Checkpoint saved: "{}".'.format(filename))
        except Exception:
            logger.exception('Error occurred when dump checkpoint "{}".'.format(filename))

    def load_checkpoint(self, filename):
        if osp.isfile(filename):
            model = self._model
            if isinstance(model, nn.DataParallel):
                model = model.module

            try:
                checkpoint = torch.load(filename)
                model.load_state_dict(checkpoint['model'])
                self._optimizer.load_state_dict(checkpoint['optimizer'])
                logger.critical('Checkpoint loaded: {}.'.format(filename))
                return checkpoint['extra']
            except Exception:
                logger.exception('Error occurred when load checkpoint "{}".'.format(filename))
        else:
            logger.warning('No checkpoint found at specified position: "{}".'.format(filename))
        return None

    def load_weights(self, filename, **kwargs):
        return load_weights(self._model, filename, **kwargs)

    def set_learning_rate(self, lr):
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def decay_learning_rate(self, decay):
        for param_group in self._optimizer.param_groups:
            param_group['lr'] *= decay

    def step(self, feed_dict, reduce_func=default_reduce_func, cast_tensor=False):
        assert self._model.training, 'Step a evaluation-mode model.'

        self.trigger_event('step:before', self)

        if cast_tensor:
            feed_dict = as_tensor(feed_dict)

        begin = time.time()

        self.trigger_event('forward:before', self, feed_dict)
        loss, monitors, output_dict = self._model(feed_dict)
        self.trigger_event('forward:after', self, feed_dict, loss, monitors, output_dict)

        loss = reduce_func('loss', loss)
        monitors = {k: reduce_func(k, v) for k, v in monitors.items()}

        loss_f = as_float(loss)
        monitors_f = as_float(monitors)

        self._optimizer.zero_grad()
        self.trigger_event('backward:before', self, feed_dict, loss, monitors, output_dict)
        if loss.requires_grad:
            loss.backward()
        self.trigger_event('backward:after', self, feed_dict, loss, monitors, output_dict)
        if loss.requires_grad:
            self._optimizer.step()

        end = time.time()

        self.trigger_event('step:after', self)

        return loss_f, monitors_f, output_dict, {'time/gpu': end - begin}

    def evaluate(self, feed_dict, cast_tensor=False):
        assert not self._model.training, 'Evaluating a training-mode model.'
        begin = time.time()
        if cast_tensor:
            feed_dict = as_tensor(feed_dict)
        with torch.no_grad():
            output_dict = self._model(feed_dict)
        end = time.time()

        return output_dict, dict(gpu_time=end - begin)
