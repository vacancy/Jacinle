#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : exception.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

__all__ = ['format_exc']


def format_exc(ei):
    import io
    import traceback

    sio = io.StringIO()
    tb = ei[2]
    # See issues #9427, #1553375. Commented out for now.
    # if getattr(self, 'fullstack', False):
    #     traceback.print_stack(tb.tb_frame.f_back, file=sio)
    traceback.print_exception(ei[0], ei[1], tb, None, sio)
    s = sio.getvalue()
    sio.close()
    if s[-1:] == "\n":
        s = s[:-1]
    return s

