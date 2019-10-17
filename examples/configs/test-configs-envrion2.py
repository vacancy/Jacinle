#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test-configs-envrion2.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/17/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from jacinle.config.environ_v2 import configs, def_configs, set_configs

with set_configs():
    configs.model.hidden_dim = 10
    configs.train.learning_rate = 10
    configs.train.beta = 10

with def_configs():
    configs.model.hidden_dim = 1
    configs.train.learning_rate = 10
    configs.train.weight_decay = 0


configs.print()
print('Undefined configs:')
print(*configs.find_undefined_values(), sep='\n')
