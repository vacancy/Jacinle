# -*- coding:utf8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com 
# 
# This file is part of Atari-Simulator.


from .controller import *
from .pack import *


__quit_action__ = 32767


def automake(name, *args, **kwargs):
    if name.startswith('gym.'):
        name = name[4:]
        from ..envs.gym import GymRLEnv
        return GymRLEnv(name, *args, **kwargs)
    elif name.startswith('jacinle.'):
        raise NotImplementedError()

        name = name[8:]
        from .. import custom
        assert hasattr(custom, name), 'Custom RLEnviron {} not found.'.format(name)
        return getattr(custom, name)(*args, **kwargs)
