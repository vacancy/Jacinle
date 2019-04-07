#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : linalg.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/04/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

__all__ = ['normalize']


def normalize(tensor, p=2, dim=-1, eps=1e-8):
    r"""
    Normalize the input along a specific dimension.

    .. math::
        tensor = \frac{tensor}{\max(\lVert tensor \rVert_p, \epsilon)}

    Args:
        tensor (Tensor): input.
        p (int): the exponent value in the norm formulation. Default: 2.
        dim (int): the dimension of the normalization.
        eps (float): eps for numerical stability.

    Returns:
        Tensor: normalized input.

    """
    return tensor / tensor.norm(p, dim=dim, keepdim=True).clamp(min=eps)
