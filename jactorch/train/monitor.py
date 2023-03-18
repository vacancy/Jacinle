#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : monitor.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/12/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from typing import Optional, Dict

import torch
import torch.nn.functional as F
from jactorch.utils.meta import as_float
from jactorch.utils.grad import no_grad_func

__all__ = [
    'binary_classification_accuracy',
    'binary_classification_accuracy_4',
    'classification_accuracy',
    'regression_accuracy',
    'monitor_rms',
    'monitor_param_saturation',
    'monitor_param_rms',
    'monitor_param_gradrms',
    'monitor_param_gradrms_ratio'
]


@no_grad_func
def binary_classification_accuracy(pred: torch.Tensor, label: torch.Tensor, name: str = '', saturation: bool = True) -> Dict[str, float]:
    r"""Compute the accuracy of binary classification.

    Args:
        pred: the prediction, of the same shape as ``label``.
        label: the label, of the same shape as ``pred``.
        name: the name of this monitor.
        saturation: whether to check the saturation of the prediction. Saturation
            is defined as :math:`1 - \min(pred, 1 - pred)`

    Returns:
        a dict of monitor values.
    """
    if name != '':
        name = '/' + name
    prefix = 'accuracy' + name
    pred = pred.view(-1)  # Binary accuracy
    label = label.view(-1)
    acc = label.float().eq((pred > 0.5).float())
    if saturation:
        sat = 1 - (pred - (pred > 0.5).float()).abs()
        return {
            prefix: as_float(acc.float().mean()),
            prefix + '/saturation/mean': as_float(sat.mean()),
            prefix + '/saturation/min': as_float(sat.min())
        }
    return {prefix: as_float(acc.float().mean())}


@no_grad_func
def binary_classification_accuracy_4(pred: torch.Tensor, label: torch.Tensor, name: str = '') -> Dict[str, float]:
    if name != '':
        name = '/' + name

    prefix = 'accuracy' + name
    pred = pred.view(-1)  # Binary accuracy
    label = label.view(-1)
    numel = pred.numel()

    gt_0_pred_0 = ((label < 0.5) & (pred < 0.5)).sum() / numel
    gt_0_pred_1 = ((label < 0.5) & (pred > 0.5)).sum() / numel
    gt_1_pred_0 = ((label > 0.5) & (pred < 0.5)).sum() / numel
    gt_1_pred_1 = ((label > 0.5) & (pred > 0.5)).sum() / numel

    return {
        prefix + '/gt_0_pred_0': as_float(gt_0_pred_0),
        prefix + '/gt_0_pred_1': as_float(gt_0_pred_1),
        prefix + '/gt_1_pred_0': as_float(gt_1_pred_0),
        prefix + '/gt_1_pred_1': as_float(gt_1_pred_1),
    }


@no_grad_func
def classification_accuracy(pred: torch.Tensor, label: torch.Tensor, name: str = '') -> Dict[str, float]:
    r"""Compute the accuracy of N-way classification.

    Args:
        pred: the prediction, of the same shape as ``label``.
        label: the label, of the same shape as ``pred``.
        name: the name of this monitor.

    Returns:
        a dict of monitor values.
    """
    if name != '':
        name = '/' + name
    prefix = 'accuracy' + name
    pred = pred.view(-1)  # Binary accuracy
    label = label.view(-1)
    acc = label.float().eq((pred).float())
    return {prefix: as_float(acc.float().mean())}


@no_grad_func
def regression_accuracy(pred: torch.Tensor, label: torch.Tensor, name: str = '') -> Dict[str, float]:
    r"""Compute the accuracy of regression.

    Args:
        pred: the prediction, of the same shape as ``label``.
        label: the label, of the same shape as ``pred``.
        name: the name of this monitor.

    Returns:
        a dict of monitor values.
    """
    if name != '':
        name = '/' + name
    prefix = 'accuracy' + name
    pred = pred.view(-1)  # Binary accuracy
    label = label.view(-1)
    diff = pred - label
    return {
        prefix + '/l1': as_float(diff.abs().mean()),
        prefix + '/l2': as_float(0.5 * diff.pow(2).mean())
    }


def _rms(p):
    return as_float((p ** 2).mean() ** 0.5)


@no_grad_func
def monitor_rms(_dict: Optional[Dict[str, torch.Tensor]], **values: torch.Tensor) -> Dict[str, float]:
    """Monitor the RMS of the given values. This function takes either a dict or multiple keyword arguments.

    Args:
        _dict: a dict of values.
        **values: multiple keyword arguments.

    Returns:
        a dict of monitor values.
    """
    values.update(_dict)
    monitors = {}
    for name, p in values.items():
        monitors['rms/' + name] = _rms(p)
    return monitors


@no_grad_func
def monitor_param_saturation(model: torch.nn.Module) -> Dict[str, float]:
    """Monitor the saturation of the parameters of the given model.

    Args:
        model: the model to monitor.

    Returns:
        a dict of monitor values.
    """
    monitors = {}
    for name, p in model.named_parameters():
        p = F.sigmoid(p)
        sat = 1 - (p - (p > 0.5).float()).abs()
        monitors['sat/' + name] = sat
    return monitors


@no_grad_func
def monitor_param_rms(model: torch.nn.Module) -> Dict[str, float]:
    """Monitor the RMS of the parameters of the given model.

    Args:
        model: the model to monitor.

    Returns:
        a dict of monitor values.
    """
    monitors = {}
    for name, p in model.named_parameters():
        monitors['param/rms/' + name] = _rms(p)
    return monitors


@no_grad_func
def monitor_param_gradrms(model: torch.nn.Module) -> Dict[str, float]:
    """Monitor the RMS of the gradients of the parameters of the given model.

    Args:
        model: the model to monitor.

    Returns:
        a dict of monitor values.
    """
    monitors = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            monitors['param/gradrms/' + name] = _rms(p.grad)
    return monitors


@no_grad_func
def monitor_param_gradrms_ratio(model: torch.nn.Module) -> Dict[str, float]:
    """Monitor the ratio of the RMS of the gradients of the parameters of the given model.

    Args:
        model: the model to monitor.

    Returns:
        a dict of monitor values.
    """
    monitors = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            monitors['param/gradrmsratio/' + name] = _rms(p.grad) / max(_rms(p), 1e-8)
    return monitors
