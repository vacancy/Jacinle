#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : imp.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/29/16
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import importlib
import os
import sys


__all__ = [
    'load_module', 'load_module_filename', 'load_source',
    'tuple_to_classname', 'classname_to_tuple',
    'load_class', 'module_vars_as_dict'
]


def load_module(module_name):
    """
    Load a module from a module name.

    Args:
        module_name: (str): write your description
    """
    module = importlib.import_module(module_name)
    return module


def load_module_filename(module_filename):
    """
    Loads the module name of a module.

    Args:
        module_filename: (str): write your description
    """
    assert module_filename.endswith('.py')
    realpath = os.path.realpath(module_filename)
    pos = realpath.rfind('/')
    dirname, module_name = realpath[0:pos], realpath[pos+1:-3]
    sys.path.insert(0, dirname)
    module = load_module(module_name)
    del sys.path[0]
    return module


def load_source(filename, name=None):
    """
    Load a source file.

    Args:
        filename: (str): write your description
        name: (str): write your description
    """
    import imp

    if name is None:
        basename = os.path.basename(filename)
        if basename.endswith('.py'):
            basename = basename[:-3]
        name = basename.replace('.', '_')

    return imp.load_source(name, filename)


def tuple_to_classname(t):
    """
    Convert a classname to a classname.

    Args:
        t: (todo): write your description
    """
    assert len(t) == 2, ('Only tuple with length 2 (module name, class name) can be converted to classname, '
            'got {}, {}.'.format(t, len(t)))

    return '.'.join(t)


def classname_to_tuple(classname):
    """
    Convert a classname to a classname.

    Args:
        classname: (str): write your description
    """
    pos = classname.rfind('.')
    if pos == -1:
        return '', classname
    else:
        return classname[0:pos], classname[pos + 1:]


def load_class(classname, exit_on_error=True):
    """
    Load a class from the given class.

    Args:
        classname: (str): write your description
        exit_on_error: (bool): write your description
    """
    if isinstance(classname, str):
        classname = classname_to_tuple(classname)
    elif isinstance(classname, tuple):
        assert len(classname) == 2, 'Classname should be tuple of length 2 or a single string.'
    module_name, clz_name = classname
    try:
        module = load_module(module_name)
        clz = getattr(module, clz_name)
        return clz
    except ImportError as e:
        print('Cannot import {}.{}, with original error message: {}.'.format(module_name, clz_name, str(e)))
        if exit_on_error:
            exit(1)
    return None


def module_vars_as_dict(module):
    """
    Convert a dict into a module

    Args:
        module: (todo): write your description
    """
    res = {}
    for k in dir(module):
        if not k.startswith('__'):
            res[k] = getattr(module, k)
    return res
