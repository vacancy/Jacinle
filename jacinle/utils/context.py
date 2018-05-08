#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : context.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/31/16
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

__all__ = ['EmptyContext']


class EmptyContext(object):
    def __enter__(self):
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        return
