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
    from jacinle.random.rng import global_rng_registry

    try:
        import torch
        # This will also automatically initialize cuda seeds.
        global_rng_registry.register('torch', lambda: torch.manual_seed)
    except ImportError:
        pass


def init_main():
    register_rng()
