#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : inspect.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/05/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""
Code inspection tools.
"""

import inspect
from . import _inspect_mate

__all__ = ['class_name', 'func_name', 'method_name', 'class_name_of_method', 'bind_args']


def class_name(instance_or_class):
    if isinstance(instance_or_class, type):
        return func_name(instance_or_class)
    return func_name(instance_or_class.__class__)


def func_name(func):
    return func.__module__ + '.' + func.__qualname__


def method_name(method):
    assert '.' in method.__qualname__, '"{}" is not a method.'.format(repr(method))
    return func_name(method)


def class_name_of_method(method):
    name = method_name(method)
    return name[:name.rfind('.')]


def bind_args(sig, *args, **kwargs):
    bounded = sig.bind(*args, **kwargs)
    bounded.apply_defaults()

    return bounded

