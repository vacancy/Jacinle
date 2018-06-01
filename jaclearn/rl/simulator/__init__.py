#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/23/2017
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.


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
