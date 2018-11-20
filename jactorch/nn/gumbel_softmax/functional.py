#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : functional.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/01/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch
import torch.nn.functional as F
from jacinle.utils.enum import JacEnum

from jactorch.functional import set_index_one_hot_, one_hot_nd
from jactorch.nn.mask.functional import masked_softmax

__all__ = ['greedy_softmax', 'gumbel_softmax', 'SoftmaxImplmentation', 'general_softmax']


def _sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))


def _gumbel_softmax_sample(logits, dim=-1, tau=1, eps=1e-10, mask=None):
    """
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = _sample_gumbel(logits.size(), eps=eps, out=logits.new())
    y = logits + gumbel_noise
    return masked_softmax(y / tau, mask, dim=dim)


def greedy_softmax(logits, dim=-1, mask=None):
    # TODO(Jiayuan Mao @ 07/29): add support for dim != -1.
    assert dim == -1, 'Greedy softmax support only dim=-1'
    if mask is not None:
        probs = masked_softmax(logits, mask=mask, dim=dim)
    else:
        probs = logits  # we only need to take the max
    one_hot = one_hot_nd(probs.max(dim)[1], logits.size(dim))
    return one_hot


def gumbel_softmax(logits, dim=-1, tau=1, hard=False, mask=None, eps=1e-10):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.

    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        dim: along which dim the softmax is performed
        tau: non-negative scalar temperature
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        eps: eps

    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probability distribution that sums to 1 across classes

    Based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """

    y_soft = _gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        with torch.no_grad():
            _, k = y_soft.max(dim=dim)
            # this bit is based on
            # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
            y_hard = torch.zeros_like(logits)
            y_hard.requires_grad = False
            set_index_one_hot_(y_hard, dim, k, 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = (y_hard - y_soft).detach() + y_soft
    else:
        y = y_soft
    return y


class SoftmaxImplmentation(JacEnum):
    STANDARD = 'standard'
    GUMBEL = 'gumbel'
    GUMBEL_HARD = 'gumbel_hard'


def general_softmax(logits, dim=-1, tau=1, impl='standard', mask=None, training=False):
    impl = SoftmaxImplmentation.from_string(impl)
    if impl is SoftmaxImplmentation.STANDARD:
        return masked_softmax(logits / tau, dim=dim)
    elif impl in (SoftmaxImplmentation.GUMBEL, SoftmaxImplmentation.GUMBEL_HARD):
        if not training:
            # no need to use logits / tau
            return greedy_softmax(logits, dim=dim, mask=mask)
        if impl is SoftmaxImplmentation.GUMBEL:
            return gumbel_softmax(logits, dim=dim, tau=tau, hard=False, mask=mask)
        else:
            return gumbel_softmax(logits, dim=dim, tau=tau, hard=True, mask=mask)

