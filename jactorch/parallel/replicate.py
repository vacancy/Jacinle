#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : replicate.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch.cuda.comm as comm


def replicate(network, devices, copy_parameters=False, copy_buffers=False):
    devices = tuple(devices)
    num_replicas = len(devices)

    params = list(network.parameters())
    param_indices = {param: idx for idx, param in enumerate(params)}
    if not copy_parameters:
        from torch.nn.parallel._functions import Broadcast

        param_copies = Broadcast(devices)(*params)
        if len(params) > 0:
            param_copies = [param_copies[i:i + len(params)]
                            for i in range(0, len(param_copies), len(params))]
    else:
        param_copies = _copy_parameters(params, devices)

    buffers = list(network._all_buffers())
    buffer_indices = {buf: idx for idx, buf in enumerate(buffers)}
    if not copy_buffers:
        buffer_copies = comm.broadcast_coalesced(buffers, devices)
    else:
        buffer_copies = _copy_parameters(buffers, devices)

    modules = list(network.modules())
    module_copies = [[] for device in devices]
    module_indices = {}

    for i, module in enumerate(modules):
        module_indices[module] = i
        for j in range(num_replicas):
            replica = module.__new__(type(module))
            replica.__dict__ = module.__dict__.copy()
            replica._parameters = replica._parameters.copy()
            replica._buffers = replica._buffers.copy()
            replica._modules = replica._modules.copy()
            module_copies[j].append(replica)

    for i, module in enumerate(modules):
        for key, child in module._modules.items():
            if child is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._modules[key] = None
            else:
                module_idx = module_indices[child]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._modules[key] = module_copies[j][module_idx]
        for key, param in module._parameters.items():
            if param is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._parameters[key] = None
            else:
                param_idx = param_indices[param]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._parameters[key] = param_copies[j][param_idx]
        for key, buf in module._buffers.items():
            if buf is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._buffers[key] = None
            else:
                buffer_idx = buffer_indices[buf]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._buffers[key] = buffer_copies[j][buffer_idx]

    return [module_copies[j][0] for j in range(num_replicas)]


def _copy_parameters(params, devices):
    results = []
    for i, d in devices:
        if i == 0:
            results.append(params.copy())
        else:
            results.append([p.cuda(d) for p in params])
    return results
