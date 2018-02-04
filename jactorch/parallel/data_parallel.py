# -*- coding: utf-8 -*-
# File   : data_parallel.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/02/2018
# 
# This file is part of Jacinle.

from .dict_gather import DictGatherDataParallel
from .replication_callback import ReplicationCallbackDataParallel
from .user_scattered import UserScatteredDataParallel

__all__ = ['JacDataParallel', 'UserScatteredJacDataParallel']


class JacDataParallel(DictGatherDataParallel, ReplicationCallbackDataParallel):
    pass


class UserScatteredJacDataParallel(DictGatherDataParallel,
                                   ReplicationCallbackDataParallel,
                                   UserScatteredDataParallel):

    pass
