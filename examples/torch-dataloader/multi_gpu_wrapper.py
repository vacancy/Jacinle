#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : multi_gpu_wrapper.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/24/2020
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import random
import torch
import torch.cuda as cuda
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
import jacinle
import jactorch


class MyDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        length = random.randint(5, 10)
        return {
            'x': torch.rand(length),
            'y': torch.rand(length)
        }

    def __len__(self):
        return 128


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(5))  # a dummy parameter.

    def forward(self, feed_dict):
        outputs = dict(
            z=feed_dict['x'] + feed_dict['y'],
            devices=(str(feed_dict['x'].device), str(feed_dict['y'].device))
        )
        if self.training:
            monitors = dict(z_sum=outputs['z'].sum(), z_max=outputs['z'].max())
            loss = monitors['z_sum']
            return loss, monitors, outputs
        else:
            return outputs


import argparse
args = argparse.Namespace()
args.use_gpu = True
if args.use_gpu:
    nr_devs = cuda.device_count()
    assert nr_devs > 0, 'No GPU device available'
    args.gpus = [i for i in range(nr_devs)]
    args.gpu_parallel = (nr_devs > 1)


def main():
    dataset = MyDataset()
    from jactorch.data.dataloader import JacDataLoader, JacDataLoaderMultiGPUWrapper
    from jactorch.data.collate import VarLengthCollateV3
    dataloader = JacDataLoader(dataset, batch_size=8, collate_fn=VarLengthCollateV3({'x': 'concat', 'y': 'concat'}), shuffle=True, drop_last=True, num_workers=0)
    dataloader = JacDataLoaderMultiGPUWrapper(dataloader, args.gpus)

    from jactorch.parallel import JacDataParallel
    model = MyModel()
    model = JacDataParallel(model, user_scattered=True, dict_gather_layout={'z': 'concat', 'devices': 'skip'})
    model.cuda()
    optimizer = optim.SGD(model.parameters(), 1e-4)

    from jactorch.train.env import TrainerEnv, default_reduce_func
    env = TrainerEnv(model, optimizer)

    # the reduce func only changes the behavior of reduction on the loss function and the monitors.
    def custom_reduce_func(k, v):
        if '_max' in k:
            return v.max()
        elif '_sum' in k:
            return v.sum()
        else:
            return default_reduce_func(k, v)

    feed_dict = next(iter(dataloader))
    loss, monitors, outputs, _ = env.step(feed_dict, reduce_func=custom_reduce_func)

    # feed_dict is a List[Dict], where each dict contain 4 keys: x, y, x_length, and y_length.
    # The length of the list is the number of GPUs.
    # All x's and y's are concatenated along the first dimension (the batch dimension).
    # All {x,y}_lengths are int-typed tensors, recording the length for each item in the batch (thus of size [batch_size]).
    jacinle.stprint(feed_dict)
    # outputs is a dict, which gathers all outputs across all gpus.
    # You can specify the gathering method via dict_gather_layout.
    # For a value to "concat", it will output the concatenation of all tensors across all gpus.
    # An auxiliary tensor: z_length will be added. It is int64-typed, of size [nr_devs], which records the size of dim0
    # for all tensors.
    # If you want to have the maximal control of the outputs, specify 'skip'. In this case, it outputs List[Tuple[str]].
    jacinle.stprint(outputs)
    jacinle.stprint(monitors)


if __name__ == '__main__':
    main()
