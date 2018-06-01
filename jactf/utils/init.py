#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : init.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/01/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import os


def tune_tensorflow():
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'  # issue#9339
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '3'   # use more warm-up


def register_rng():
    import tensorflow
    from jacinle.random.rng import global_rng_registry
    global_rng_registry.register('tf', lambda: tensorflow.set_random_seed)


def init_main():
    tune_tensorflow()
    register_rng()
