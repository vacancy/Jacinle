#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : geometry.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/04/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch
import jactorch
import torch.nn.functional as F

__all__ = ['gen_voronoi']


def gen_voronoi(centers, height, width):
    range_y = torch.arange(height, device=centers.device)
    range_x = torch.arange(width, device=centers.device)
    y, x = jactorch.meshgrid(range_y, range_x, dim=0)
    y, x = y.reshape(-1), x.reshape(-1)
    coords = torch.stack([y, x], dim=1).float()
    coords, centers = jactorch.meshgrid(coords, centers, dim=0)
    dis = (coords[:, :, 0] - centers[:, :, 1]) ** 2 + (coords[:, :, 1] - centers[:, :, 0]) ** 2
    assignment = dis.argmin(1)
    return dis.view((height, width, -1)), assignment.view((height, width))

