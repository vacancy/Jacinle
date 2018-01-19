# -*- coding: utf-8 -*-
# File   : context.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/31/16
#
# This file is part of Jacinle.

__all__ = ['EmptyContext']


class EmptyContext(object):
    def __enter__(self):
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        return
