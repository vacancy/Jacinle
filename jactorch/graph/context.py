#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : context.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/31/2021
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""Context manager used in a forward pass. It supports accessing to global variables and monitors
in different modules. See :class:`ForwardContext` for more details."""

import functools
from typing import Any, Optional, Union, Dict

import torch
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
    """A context manager that serves as a global variable for the forward pass. It supports
    accessing to global variables in different modules.

    Example:
        .. code-block:: python

            with ForwardContext(training=True) as ctx:
                ctx.add_loss(loss)
                ctx.add_accuracy(accuracy, 'acc')
                ctx.add_output(output, 'output')

                # In a different file, you can access the context by:
                ctx = get_forward_context()
                ctx.add_loss(some_other_loss)

            loss, monitors, output_dict = ctx.finalize()
    """

    def __init__(self, training: bool, *, loss: float = 0, monitors: Optional[Dict] = None, output_dict: Optional[Dict] = None):
        """Initialize the context.

        Args:
            training: whether the forward pass is in training mode.
            loss: the initial loss.
            monitors: the initial monitors.
            output_dict: the initial output dictionary.
        """
        self.training = training
        self.loss = loss
        self.monitors = GView(monitors)
        self.output_dict = GView(output_dict)
        self.hyperparameters = dict()

    loss: Union[float, torch.Tensor]
    """The current loss."""

    monitors: GView
    """The current monitors."""

    output_dict: GView
    """The current output dictionary."""

    def set_hyperparameter(self, key: str, value: Any):
        """Set a hyperparameter for the forward pass.

        Args:
            key: the key of the hyperparameter.
            value: the value of the hyperparameter.
        """
        self.hyperparameters[key] = value

    def get_hyperparameter(self, key: str, default: Any = None) -> Any:
        """Get a hyperparameter for the forward pass.

        Args:
            key: the key of the hyperparameter.
            default: the default value of the hyperparameter.

        Returns:
            the value of the hyperparameter.
        """
        return self.hyperparameters.get(key, default=default)

    def add_loss(self, loss: Union[float, torch.Tensor], key: Optional[str] = None, accumulate=True) -> 'ForwardContext':
        """Add a (sub) loss to the context.

        Args:
            loss: the sub-loss to add.
            key: the name of the loss. If None, the loss will not be monitored.
            accumulate: whether to accumulate the loss in the final loss. This value can either be a boolean or a
                float number. If it is a boolean, it indicates whether to accumulate the loss.
                If it is a float number, it indicates the weight of the loss.

        Returns:
            self.
        """
        if float(accumulate) > 0:
            self.loss = self.loss + loss * float(accumulate)

        if key is not None:
            if f'loss/{key}' in self.monitors:
                self.monitors[f'loss/{key}'] += float(loss)
            else:
                self.monitors[f'loss/{key}'] = float(loss)
        return self

    def add_accuracy(self, accuracy: Union[float, torch.Tensor], key: str) -> 'ForwardContext':
        """Add an accuracy to the context.

        Args:
            accuracy: the accuracy to add.
            key: the name of the accuracy.

        Returns:
            self.
        """
        self.monitors[f'accuracy/{key}'] = float(accuracy)
        return self

    def add_output(self, output: Any, key: str) -> 'ForwardContext':
        """Add an output to the context.

        Args:
            output: the output to add.
            key: the name of the output.

        Returns:
            self.
        """
        self.output_dict[key] = output
        return self

    def update_monitors(self, monitors: Dict[str, Union[float, torch.Tensor]]):
        """Update the monitors in the context.

        Args:
            monitors: the monitors to update.
        """
        self.monitors.update(monitors)
        return self

    def update_mo(self, monitors: Dict[str, Union[float, torch.Tensor]], output_dict: Dict[str, Any]):
        """Update the monitors and output dictionary in the context."""
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
        """Set the context as the default context."""
        yield self

    def finalize(self):
        """Finalize the context and return the loss, monitors, and output dictionary."""
        if self.training:
            return self.loss, self.monitors, self.output_dict
        else:
            self.output_dict.monitors = self.monitors
            return self.output_dict


_get_forward_context = gen_get_default(ForwardContext)


def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    return _get_forward_context()
