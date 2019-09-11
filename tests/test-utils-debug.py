#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test-utils-debug.py
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


if __name__ == '__main__':
    some_func(0)

