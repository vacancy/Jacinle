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
from typing import Any, Optional, Union, Tuple, Dict
from importlib.machinery import SourceFileLoader

try:
    from typing import ModuleType
except ImportError:
    ModuleType = Any  # Workaround for older versions of Python.


__all__ = [
    'load_module', 'load_module_filename', 'load_source',
    'tuple_to_classname', 'classname_to_tuple',
    'load_class', 'module_vars_as_dict'
]


def load_module(module_name: str) -> ModuleType:
    """Import a module by its module name (e.g., jacinle.utils.imp).

    Args:
        module_name: the name of the module.

    Returns:
        the imported module.
    """
    module = importlib.import_module(module_name)
    return module


def load_module_filename(module_filename: str) -> ModuleType:
    """Import a module by its filename (e.g., /Users/jiayuan/Projects/Jacinle/jacinle/utils/imp.py).

    Args:
        module_filename: the filename of the module.

    Returns:
        the imported module.
    """
    assert module_filename.endswith('.py')
    realpath = os.path.realpath(module_filename)
    pos = realpath.rfind('/')
    dirname, module_name = realpath[0:pos], realpath[pos + 1:-3]
    sys.path.insert(0, dirname)
    module = load_module(module_name)
    del sys.path[0]
    return module


def load_source(filename: str, name: Optional[str] = None) -> ModuleType:
    """Load a source file as a module.

    Args:
        filename: the filename of the source file.
        name: the name of the module.

    Returns:
        the loaded module.
    """
    if name is None:
        basename = os.path.basename(filename)
        if basename.endswith('.py'):
            basename = basename[:-3]
        name = basename.replace('.', '_')

    return SourceFileLoader(name, filename).load_module()


def tuple_to_classname(t: Tuple[str, str]) -> str:
    """Convert a module-class tuple to a string.

    Args:
        t: the module-class tuple. E.g., ('jacinle.utils.imp', 'load_module').

    Returns:
        the string representation of the module-class tuple. E.g., 'jacinle.utils.imp.load_module'.
    """
    assert len(t) == 2, (
        'Only tuple with length 2 (module name, class name) can be converted to classname, '
        'got {}, {}.'.format(t, len(t))
    )

    return '.'.join(t)


def classname_to_tuple(classname: str) -> Tuple[str, str]:
    """Convert a string to a module-class tuple.

    Args:
        classname: the string representation of the module-class tuple. E.g., 'jacinle.utils.imp.load_module'.

    Returns:
        the module-class tuple. E.g., ('jacinle.utils.imp', 'load_module').
    """
    pos = classname.rfind('.')
    if pos == -1:
        return '', classname
    else:
        return classname[0:pos], classname[pos + 1:]


def load_class(classname: Union[str, Tuple[str, str]], exit_on_error: bool = True):
    """Load a class by its classname (including module).

    Args:
        classname: the classname of the class.
        exit_on_error: whether to exit the program when the class cannot be loaded.

    Returns:
        the loaded class.
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


def module_vars_as_dict(module: ModuleType) -> Dict[str, Any]:
    """Get all variables in a module as a dictionary.

    Args:
        module: the module.

    Returns:
        a dictionary of all variables in the module.
    """

    res = {}
    for k in dir(module):
        if not k.startswith('__'):
            res[k] = getattr(module, k)
    return res

