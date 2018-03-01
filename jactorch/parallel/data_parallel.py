# -*- coding: utf-8 -*-
# File   : data_parallel.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/02/2018
# 
# This file is part of Jacinle.

from jactorch.cuda.copy import async_copy_to

from .dict_gather import DictGatherDataParallel
from .replication_callback import ReplicationCallbackDataParallel
from .user_scattered import UserScatteredDataParallel

__all__ = ['JacDataParallel', 'UserScatteredJacDataParallel']


class JacDataParallel(DictGatherDataParallel, ReplicationCallbackDataParallel):
    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            inputs = async_copy_to(inputs, 0)
            kwargs = async_copy_to(kwargs, 0)
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)


class UserScatteredJacDataParallel(JacDataParallel, UserScatteredDataParallel):
    pass
