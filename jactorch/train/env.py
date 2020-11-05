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
from jactorch.io import load_weights, state_dict, load_state_dict
from jactorch.utils.meta import as_tensor, as_float, as_cpu

logger = get_logger(__file__)

__all__ = ['TrainerEnv']


def cuda_time(sync=True):
    """
    Cuda time of the current task.

    Args:
        sync: (todo): write your description
    """
    if sync:
        torch.cuda.synchronize()
    return time.time()


def default_reduce_func(k, v):
    """
    Reduce function for a function.

    Args:
        k: (todo): write your description
        v: (array): write your description
    """
    if torch.is_tensor(v):
        return v.mean()
    return v


class TrainerEnv(object):
    def __init__(self, model, optimizer):
        """
        Initialize the optimizer.

        Args:
            self: (todo): write your description
            model: (todo): write your description
            optimizer: (todo): write your description
        """
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
        """
        Return the model instance.

        Args:
            self: (todo): write your description
        """
        return self._model

    @property
    def model_unwrapped(self):
        """
        Unwraps model instances.

        Args:
            self: (todo): write your description
        """
        model = self.model
        if isinstance(model, nn.DataParallel):
            model = model.module
        return model

    @property
    def optimizer(self):
        """
        Return the optimizer.

        Args:
            self: (todo): write your description
        """
        return self._optimizer

    def register_event(self, name, callback):
        """
        Register a new event.

        Args:
            self: (todo): write your description
            name: (str): write your description
            callback: (todo): write your description
        """
        logger.info('Register trainer event: name={}, callback={}.'.format(name, callback.__module__ + '.' + callback.__name__))
        self._event_manager.register(name, callback)

    def trigger_event(self, name, *args, **kwargs):
        """
        Triggers an event.

        Args:
            self: (todo): write your description
            name: (str): write your description
        """
        self._event_manager.trigger(name, *args, **kwargs)

    def save_checkpoint(self, filename, extra=None):
        """
        Save the model to disk.

        Args:
            self: (todo): write your description
            filename: (str): write your description
            extra: (dict): write your description
        """
        # Hack the data parallel.
        model = self._model

        state = {
            'model': state_dict(model, cpu=True),
            'optimizer': as_cpu(self._optimizer.state_dict()),
            'extra': extra
        }
        try:
            torch.save(state, filename)
            logger.info('Checkpoint saved: "{}".'.format(filename))
        except Exception:
            logger.exception('Error occurred when dump checkpoint "{}".'.format(filename))

    def load_checkpoint(self, filename):
        """
        Load checkpoint from file.

        Args:
            self: (todo): write your description
            filename: (str): write your description
        """
        if osp.isfile(filename):
            model = self._model
            if isinstance(model, nn.DataParallel):
                model = model.module

            try:
                checkpoint = torch.load(filename)
                load_state_dict(model, checkpoint['model'])
                self._optimizer.load_state_dict(checkpoint['optimizer'])
                logger.critical('Checkpoint loaded: {}.'.format(filename))
                return checkpoint['extra']
            except Exception:
                logger.exception('Error occurred when load checkpoint "{}".'.format(filename))
        else:
            logger.warning('No checkpoint found at specified position: "{}".'.format(filename))
        return None

    def load_weights(self, filename, **kwargs):
        """
        Load model from file.

        Args:
            self: (todo): write your description
            filename: (str): write your description
        """
        return load_weights(self._model, filename, **kwargs)

    def set_learning_rate(self, lr):
        """
        Set learning rate of the optimizer.

        Args:
            self: (float): write your description
            lr: (float): write your description
        """
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def decay_learning_rate(self, decay):
        """
        Decay learning rate.

        Args:
            self: (todo): write your description
            decay: (todo): write your description
        """
        for param_group in self._optimizer.param_groups:
            param_group['lr'] *= decay

    def step(self, feed_dict, grad_clip=0., reduce_func=default_reduce_func, cast_tensor=False, measure_time=False):
        """
        Perform a single optimization step.

        Args:
            self: (todo): write your description
            feed_dict: (dict): write your description
            grad_clip: (array): write your description
            reduce_func: (todo): write your description
            default_reduce_func: (todo): write your description
            cast_tensor: (todo): write your description
            measure_time: (bool): write your description
        """
        if hasattr(self.model, 'train_step'):
            return self.model.train_step(self.optimizer, feed_dict)

        assert self._model.training, 'Step a evaluation-mode model.'
        extra = dict()

        self.trigger_event('step:before', self)

        if cast_tensor:
            feed_dict = as_tensor(feed_dict)

        if measure_time:
            end_time = cuda_time()

        self.trigger_event('forward:before', self, feed_dict)
        loss, monitors, output_dict = self._model(feed_dict)
        self.trigger_event('forward:after', self, feed_dict, loss, monitors, output_dict)

        if measure_time:
            extra['time/forward'] = cuda_time() - end_time
            end_time = cuda_time(False)

        loss = reduce_func('loss', loss)
        monitors = {k: reduce_func(k, v) for k, v in monitors.items()}

        loss_f = as_float(loss)
        monitors_f = as_float(monitors)

        if measure_time:
            extra['time/loss'] = cuda_time() - end_time
            end_time = cuda_time(False)

        self._optimizer.zero_grad()
        self.trigger_event('backward:before', self, feed_dict, loss, monitors, output_dict)
        if loss.requires_grad:
            loss.backward()
            if grad_clip > 0:
                from torch.nn.utils.clip_grad import clip_grad_norm_
                clip_grad_norm_(self.model.parameters(), grad_clip)

        if measure_time:
            extra['time/backward'] = cuda_time() - end_time
            end_time = cuda_time(False)

        self.trigger_event('backward:after', self, feed_dict, loss, monitors, output_dict)
        if loss.requires_grad:
            self._optimizer.step()

        if measure_time:
            extra['time/optimize'] = cuda_time() - end_time
            end_time = cuda_time(False)

        self.trigger_event('step:after', self)

        return loss_f, monitors_f, output_dict, extra

    def evaluate(self, feed_dict, cast_tensor=False):
        """
        Evaluate the model.

        Args:
            self: (todo): write your description
            feed_dict: (dict): write your description
            cast_tensor: (todo): write your description
        """
        assert not self._model.training, 'Evaluating a training-mode model.'
        begin = time.time()
        if cast_tensor:
            feed_dict = as_tensor(feed_dict)
        with torch.no_grad():
            output_dict = self._model(feed_dict)
        end = time.time()

        return output_dict, dict(gpu_time=end - begin)
