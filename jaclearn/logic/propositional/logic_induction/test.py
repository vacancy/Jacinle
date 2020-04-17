#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/16/2020
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import numpy as np
import logic_induction

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype='uint8')
outputs = np.array([[0], [1], [1], [0]], dtype='uint8')
input_names = ['x', 'y']
print(logic_induction.search(inputs, outputs, input_names))

