#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : init.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/01/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.


def register_rng():
    from jacinle.random.rng import global_rng_registry, global_rng_state_registry

    try:
        import torch
        # This will also automatically initialize cuda seeds.
        global_rng_registry.register('torch', lambda: torch.manual_seed)
        # TODO(Jiayuan Mao @ 2023/01/17): get and set cuda random seeds (when available).
        global_rng_state_registry.register('torch', lambda: (torch.get_rng_state, torch.set_rng_state))

    except ImportError:
        pass


def init_main():
    register_rng()
