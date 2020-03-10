#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/09/2020
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch

if torch.__version__ < '0.3.1':
    from .dataloader_torch030 import *
else:
    from .dataloader import *
