#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cluster.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/25/2022
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""Clustering functions."""

import torch
from jacinle.utils.vendor import requires_vendors

__all__ = ['kmeans']


@requires_vendors('kmeans_pytorch')
def kmeans(data: torch.Tensor, nr_clusters: int, nr_iterations: int = 20, distance: str = 'euclidean', device=None, verbose=False):
    if device is None:
        device = data.device

    from kmeans_pytorch import kmeans
    return kmeans(X=data, num_clusters=nr_clusters, distance=distance, device=device, tqdm_flag=verbose, iter_limit=nr_iterations)

