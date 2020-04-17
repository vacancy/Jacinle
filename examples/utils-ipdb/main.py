#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : main.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/11/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from jacinle.utils.debug import decorate_exception_hook


@decorate_exception_hook
def some_func(x):
    return 1 / x


some_func(0)
