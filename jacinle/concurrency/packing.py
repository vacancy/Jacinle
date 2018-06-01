#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : packing.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/22/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import os
import functools

from jacinle.utils.enum import JacEnum
from jacinle.utils.registry import RegistryGroup, CallbackRegistry

__all__ = [
    'check_pickle', 'loadb_pickle', 'dumpb_pickle',
    'check_msgpack', 'loadb_msgpack', 'dumpb_msgpack',
    'check_pyarrow', 'loadb_pyarrow', 'dumpb_pyarrow',
    'loadb', 'dumpb',
    'get_available_backends', 'get_default_backend', 'set_default_backend'
]

import pickle

loadb_pickle = pickle.loads
dumpb_pickle = pickle.dumps

try:
    import msgpack
    import msgpack_numpy

    msgpack_numpy.patch()
    dumpb_msgpack = functools.partial(msgpack.dumps, use_bin_type=True)
    loadb_msgpack = msgpack.loads
except ImportError:
    dumpb_msgpack = loadb_msgpack = None


try:
    import pyarrow

    dumpb_pyarrow = lambda obj: pyarrow.serialize(obj).to_buffer()
    loadb_pyarrow = lambda buffer: pyarrow.deserialize(buffer)
except ImportError:
    dumpb_pyarrow = loadb_pyarrow = None


class _PackingFunctionRegistryGroup(RegistryGroup):
    __base_class__ = CallbackRegistry

    def dispatch(self, registry_name, entry, *args, **kwargs):
        return self[registry_name].dispatch(entry, *args, **kwargs)


_packing_function_registry = _PackingFunctionRegistryGroup()


def check_pickle():
    return True


def check_msgpack():
    return dumpb_msgpack is not None


def check_pyarrow():
    return dumpb_pyarrow is not None


class _PackingBackend(JacEnum):
    PICKLE = 'pickle'
    MSGPACK = 'msgpack'
    PYARROW = 'pyarrow'


_packing_function_registry.register('check', _PackingBackend.PICKLE, lambda: True)
_packing_function_registry.register('check', _PackingBackend.MSGPACK, check_msgpack)
_packing_function_registry.register('check', _PackingBackend.PYARROW, check_pyarrow)


_packing_function_registry.register('loadb', _PackingBackend.PICKLE, loadb_pickle)
_packing_function_registry.register('dumpb', _PackingBackend.PICKLE, dumpb_pickle)

_packing_function_registry.register('loadb', _PackingBackend.MSGPACK, loadb_msgpack)
_packing_function_registry.register('dumpb', _PackingBackend.MSGPACK, dumpb_msgpack)

_packing_function_registry.register('loadb', _PackingBackend.PYARROW, loadb_pyarrow)
_packing_function_registry.register('dumpb', _PackingBackend.PYARROW, dumpb_pyarrow)

_default_packing_backend = _PackingBackend.PICKLE


def get_default_backend():
    return _default_packing_backend.name


def get_available_backends():
    return [obj.name for obj in _PackingBackend.choice_objs()
            if _packing_function_registry.dispatch('check', obj)]


def set_default_backend(backend):
    global _default_packing_backend
    _default_packing_backend = _PackingBackend.from_string(backend)
    assert _default_packing_backend.name in get_available_backends(), (
        'Unsupported backend on your machine: "{}".'.format(_default_packing_backend.name))


def loadb(bstr, *args, backend=None, **kwargs):
    backend = backend or _default_packing_backend
    return _packing_function_registry.dispatch('loadb', backend, bstr, *args, **kwargs)


def dumpb(obj, *args, backend=None, **kwargs):
    backend = backend or _default_packing_backend
    return _packing_function_registry.dispatch('dumpb', backend, obj, *args, **kwargs)


def _initialize_backend():
    set_default_backend(os.getenv('JAC_PACKING_BACKEND', _PackingBackend.PICKLE))


_initialize_backend()
