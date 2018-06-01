#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : naming.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2017
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

__all__ = ['class_name', 'func_name', 'method_name', 'class_name_of_method']


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
