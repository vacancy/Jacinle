#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : data_parallel.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/04/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch.cuda as cuda
import torch.nn as nn
from torch.nn.parallel.data_parallel import DataParallel

from jactorch.cuda.copy import async_copy_to

from .dict_gather import data_parallel_dict_gather
from .replicate import replicate
from .replication_callback import exec_data_parallel_replication_callback
from .user_scattered import use_user_scattered

__all__ = ['JacDataParallel', 'UserScatteredJacDataParallel']


class JacDataParallel(DataParallel):
    def __init__(self, module,
                 device_ids=None, output_device=None, dim=0,
                 allow_replication_callback=True,
                 user_scattered=False, use_scatter_stream=True,
                 scatter_func=None,
                 use_dict_gather=True, dict_gather_layout=None,
                 persistent=False, copy_parameters=False, copy_buffers=True):

        super(DataParallel, self).__init__()
        if device_ids is None:
            device_ids = list(range(cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]

        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])

        self.allow_replication_callback = allow_replication_callback

        self.user_scattered = user_scattered
        self.use_scatter_stream = use_scatter_stream
        self.scatter_func = scatter_func

        self.use_dict_gather = use_dict_gather
        self.dict_gather_layout = dict_gather_layout

        self.persistent = persistent
        self.copy_parameters = copy_parameters
        self.copy_buffers = copy_buffers
        self.replicas = nn.ModuleList()

    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            inputs = async_copy_to(inputs, 0)
            kwargs = async_copy_to(kwargs, 0)
            return self.module(*inputs[0], **kwargs[0])

        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def scatter(self, inputs, kwargs, device_ids):
        if self.scatter_func is not None:
            return self.scatter_func(inputs, kwargs, device_ids, dim=self.dim)
        elif self.user_scattered:
            return use_user_scattered(inputs, kwargs, device_ids, use_stream=self.use_scatter_stream)
        return super().scatter(inputs, kwargs, device_ids)

    def gather(self, outputs, output_device):
        if self.use_dict_gather:
            return data_parallel_dict_gather(self, outputs, output_device, layout=self.dict_gather_layout)
        return super().gather(outputs, output_device)

    def replicate(self, module, device_ids):
        if self.persistent or len(self.replicas) == 0:
            if not self.persistent:
                modules = super().replicate(module, device_ids)
            else:
                modules = replicate(
                    module, device_ids, copy_parameters=self.copy_parameters, copy_buffers=self.copy_buffers
                )
            if self.allow_replication_callback:
                exec_data_parallel_replication_callback(modules)
            if self.persistent:
                self.replicas.extend(modules)
        else:
            modules = list(self.replicas)
        return modules


class UserScatteredJacDataParallel(JacDataParallel):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('user_scattered', True)
        super().__init__(*args, **kwargs)
