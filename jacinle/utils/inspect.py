#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : inspect.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/05/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from typing import Any, Callable, Iterable

"""Code inspection tools."""

__all__ = ['class_name', 'func_name', 'method_name', 'class_name_of_method', 'bind_args', 'get_subclasses']


def class_name(instance_or_class: Any) -> str:
    """Get the class name of an instance or a class object.

    Args:
        instance_or_class: an instance or a class object.

    Returns:
        the class name of the instance or the class object.
    """
    if isinstance(instance_or_class, type):
        return func_name(instance_or_class)
    return func_name(instance_or_class.__class__)


def func_name(func: Callable) -> str:
    """Get a full name of a function, including the module name.

    Args:
        func: a function.

    Returns:
        the full name of the function.
    """
    return func.__module__ + '.' + func.__qualname__


def method_name(method: Callable) -> str:
    """Get a full name of a method, including the module name and the class name."""
    assert '.' in method.__qualname__, '"{}" is not a method.'.format(repr(method))
    return func_name(method)


def class_name_of_method(method: Callable) -> str:
    """Get the class name of a method."""
    name = method_name(method)
    return name[:name.rfind('.')]


def bind_args(sig, *args, **kwargs):
    """Bind arguments to a signature."""
    bounded = sig.bind(*args, **kwargs)
    bounded.apply_defaults()

    return bounded


def get_subclasses(cls: type) -> Iterable[type]:
    """Get all subclasses of a class."""
    for subclass in cls.__subclasses__():
        yield from get_subclasses(subclass)
        yield subclass

