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

from jacinle.logging import get_logger
from jacinle.utils.registry import SimpleEventRegistry

from jactorch.graph.nn_env import NNEnv
from jactorch.io import state_dict, load_state_dict
from jactorch.utils.meta import as_tensor, as_float, as_cpu
from .utils import set_learning_rate, decay_learning_rate

logger = get_logger(__file__)

__all__ = ['TrainerEnv']


def cuda_time(sync: bool = True) -> float:
    """Return the current time in seconds, with CUDA synchronization.

    Args:
        sync: if True, synchronize the CUDA stream before taking the time.

    Returns:
        the current time in seconds.
    """
    if sync:
        torch.cuda.synchronize()
    return time.time()


def default_reduce_func(k, v):
    """Default reduce function for the TrainerEnv."""
    if torch.is_tensor(v):
        return v.mean()
    return v


class TrainerEnv(NNEnv):
    def __init__(self, model: nn.Module, optimizer):
        super().__init__(model)
        self._optimizer = optimizer

        self._train_loader = None
        self._validation_loader = None
        self._event_manager = SimpleEventRegistry({
            'epoch:before', 'epoch:after',
            'step:before', 'step:after',
            'forward:before', 'forward:after',
            'backward:before', 'backward:after',
        })

        self._init_event_triggers()
        self.__prepared = False

    @property
    def optimizer(self):
        return self._optimizer

    def _init_event_triggers(self):
        for key in self._event_manager.allowed_events:
            name, be = key.split(':')
            callback_name = f'on_{name}_{be}'
            if hasattr(self.model_unwrapped, callback_name):
                callback = getattr(self.model_unwrapped, callback_name)
                self.register_event(key, callback)

    def register_event(self, name, callback):
        logger.info('Register trainer event: name={}, callback={}.'.format(name, callback.__module__ + '.' + callback.__name__))
        self._event_manager.register(name, callback)

    def trigger_event(self, name, *args, **kwargs):
        self._event_manager.trigger(name, *args, **kwargs)

    def save_checkpoint(self, filename, extra=None):
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

    def load_checkpoint(self, filename, **kwargs):
        if osp.isfile(filename):
            model = self._model
            if isinstance(model, nn.DataParallel):
                model = model.module

            try:
                checkpoint = torch.load(filename)
                load_state_dict(model, checkpoint['model'], **kwargs)
                self._optimizer.load_state_dict(checkpoint['optimizer'])
                logger.critical('Checkpoint loaded: {}.'.format(filename))
                return checkpoint['extra']
            except Exception:
                logger.exception('Error occurred when load checkpoint "{}".'.format(filename))
        else:
            logger.warning('No checkpoint found at specified position: "{}".'.format(filename))
        return None

    def set_learning_rate(self, lr):
        set_learning_rate(self._optimizer, lr)

    def decay_learning_rate(self, decay):
        decay_learning_rate(self._optimizer, decay)

    def prepare(self):
        self.__prepared = True

        assert self._model.training, 'Step a evaluation-mode model.'
        self.trigger_event('step:before', self)
        self._optimizer.zero_grad()

    def update(self, feed_dict, loss, monitors, output_dict, grad_clip=0., reduce_func=default_reduce_func, measure_time=False, extra=None):
        assert self.__prepared, 'Two consecutive call of TrainerEnv.update()'
        self.__prepared = False

        if extra is None:
            extra = dict()

        loss = reduce_func('loss', loss)
        monitors = {k: reduce_func(k, v) for k, v in monitors.items()}

        loss_f = as_float(loss)
        monitors_f = as_float(monitors)

        if measure_time:
            extra['time/loss'] = cuda_time() - end_time
            end_time = cuda_time(False)

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

    def step(self, feed_dict, grad_clip=0., reduce_func=default_reduce_func, cast_tensor=False, measure_time=False):
        if hasattr(self.model, 'train_step'):
            try:
                return self.model.train_step(
                    self.optimizer, feed_dict,
                    grad_clip=grad_clip, reduce_func=reduce_func, cast_tensor=False
                )
            except NotImplementedError:
                pass

        extra = dict()

        self.prepare()

        if measure_time:
            end_time = cuda_time()

        if cast_tensor:
            feed_dict = as_tensor(feed_dict)

        self.trigger_event('forward:before', self, feed_dict)
        loss, monitors, output_dict = self._model(feed_dict)
        self.trigger_event('forward:after', self, feed_dict, loss, monitors, output_dict)

        if measure_time:
            extra['time/forward'] = cuda_time() - end_time
            end_time = cuda_time(False)

        return self.update(feed_dict, loss, monitors, output_dict, grad_clip=grad_clip, reduce_func=reduce_func, measure_time=measure_time, extra=extra)

    def evaluate(self, feed_dict, cast_tensor=False):
        assert not self._model.training, 'Evaluating a training-mode model.'
        begin = time.time()
        if cast_tensor:
            feed_dict = as_tensor(feed_dict)
        with torch.no_grad():
            output_dict = self._model(feed_dict)
        end = time.time()

        return output_dict, dict(gpu_time=end - begin)

