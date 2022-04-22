#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : context.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/31/2021
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import functools
import jactorch.train.monitor as monitor
from jacinle.utils.container import GView
from jacinle.utils.defaults import wrap_custom_as_default, gen_get_default

__all__ = ['ForwardContext', 'get_forward_context']


def _wrap_monitor_function(function):
    @functools.wraps(function)
    def new_function(self, *args, **kwargs):
        self.monitors.update(function(*args, **kwargs))
        return self
    return new_function


class ForwardContext(object):
    def __init__(self, training, *, loss=0, monitors=None, output_dict=None):
        self.training = training
        self.loss = loss
        self.monitors = GView(monitors)
        self.output_dict = GView(output_dict)
        self.hyperparameters = dict()

    def set_hyperparameter(self, key, value):
        self.hyperparameters[key] = value

    def get_hyperparameter(self, key, default=None):
        return self.hyperparameters.get(key, default=default)

    def add_loss(self, loss, key=None, accumulate=True):
        if float(accumulate) > 0:
            self.loss = self.loss + loss * float(accumulate)

        if key is not None:
            if f'loss/{key}' in self.monitors:
                self.monitors[f'loss/{key}'] += loss
            else:
                self.monitors[f'loss/{key}'] = loss
        return self

    def add_accuracy(self, accuracy, key):
        self.monitors[f'accuracy/{key}'] = accuracy
        return self

    def add_output(self, output, key):
        self.output_dict[key] = output
        return self

    def update_monitors(self, monitors):
        self.monitors.update(monitors)
        return self

    def update_mo(self, monitors, output_dict):
        self.monitors.update(monitors)
        self.output_dict.update(output_dict)
        return self

    binary_classification_accuracy = _wrap_monitor_function(monitor.binary_classification_accuracy)
    classification_accuracy = _wrap_monitor_function(monitor.classification_accuracy)
    regression_accuracy = _wrap_monitor_function(monitor.regression_accuracy)
    monitor_rms = _wrap_monitor_function(monitor.monitor_rms)
    monitor_param_saturation = _wrap_monitor_function(monitor.monitor_param_saturation)
    monitor_param_rms = _wrap_monitor_function(monitor.monitor_param_rms)
    monitor_param_gradrms = _wrap_monitor_function(monitor.monitor_param_gradrms)
    monitor_param_gradrms_ratio = _wrap_monitor_function(monitor.monitor_param_gradrms_ratio)

    @wrap_custom_as_default(is_local=True)
    def as_default(self) -> 'ForwardContext':
        yield self

    def finalize(self):
        if self.training:
            return self.loss, self.monitors, self.output_dict
        else:
            self.output_dict.monitors = self.monitors
            return self.output_dict


def get_forward_context() -> ForwardContext: ...

get_forward_context = gen_get_default(ForwardContext)
