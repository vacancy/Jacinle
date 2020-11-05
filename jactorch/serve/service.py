#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : service.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch.backends.cudnn as cudnn

import jactorch.io as torchio

from jacinle.logging import get_logger
from jacinle.comm.service import Service

logger = get_logger(__file__)


class TorchService(Service):
    def load_model(self, desc, load, use_gpu, gpu_parallel=False, gpu_ids=None, cudnn_benchmark=False, args=None, **kwargs):
        """
        Load a model from disk.

        Args:
            self: (str): write your description
            desc: (str): write your description
            load: (str): write your description
            use_gpu: (bool): write your description
            gpu_parallel: (str): write your description
            gpu_ids: (str): write your description
            cudnn_benchmark: (str): write your description
        """
        model = desc.make_model(args, **kwargs)

        if use_gpu:
            model.cuda()
            if gpu_parallel:
                from jactorch.parallel import JacDataParallel
                model = JacDataParallel(model, device_ids=gpu_ids).cuda()
            cudnn.benchmark = cudnn_benchmark

        if torchio.load_weights(model, load):
            logger.critical('Loaded weights from pretrained model: "{}".'.format(load))
        model.eval()

        return model

