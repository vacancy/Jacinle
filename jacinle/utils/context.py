#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : context.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/31/16
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

__all__ = ['EmptyContext', 'KeyboardInterruptContext', 'SaverContext']


class EmptyContext(object):
    def __enter__(self):
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        return


class KeyboardInterruptContext(object):
    def __enter__(self):
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(exc_val, KeyboardInterrupt):
            return True


class SaverContext(object):
    """
    Save some information before entering the context. Restore the information after.
    Inspired by: https://github.com/caelan/pybullet-planning/blob/master/pybullet_tools/utils.py
    """

    def __enter__(self):
        self.save()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()

    def save(self):
        pass

    def restore(self):
        raise NotImplementedError()

