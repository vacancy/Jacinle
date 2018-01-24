# -*- coding: utf-8 -*-
# File   : io.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 24/01/2018
# 
# This file is part of Jacinle.

import numpy as np
import torch

import jacinle.io as io
from jacinle.logging import get_logger

logger = get_logger(__file__)


def load_weights(model, filename):
    if osp.isfile(filename):
        try:
            weights = io.load(filename)

            # Hack for checkpoint.
            if 'model' in weights and 'optimizer' in weights:
                weights = weights['model']

            # Build the tensors.
            for k, v in weights.items():
                if isinstance(v, np.ndarray):
                    weights[k] = torch.from_numpy(v)

            try:
                model.load_state_dict(weights)
            except KeyError as e:
                logger.warning('Unexpected or missing weights found: {}.'.format(str(e)))
            logger.critical('Weights loaded: {}.'.format(filename))
            return True
        except Exception as e:
            logger.exception('Error occurred when load weights {}.'.format(filename))
    else:
        logger.warning('No weights file found at specified position: {}.'.format(filename))
    return None
