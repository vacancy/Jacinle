#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : fs.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2017
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.


import os
import os.path as osp
import glob
import shutil
import six
import contextlib

import pickle
import gzip
import numpy as np

from jacinle.logging import get_logger
from jacinle.utils.registry import RegistryGroup, CallbackRegistry

from .common import get_ext

logger = get_logger(__file__)

__all__ = [
    'as_file_descriptor', 'fs_verbose', 'set_fs_verbose',
    'open', 'open_txt', 'open_h5', 'open_gz',
    'load', 'load_txt', 'load_h5', 'load_pkl', 'load_pklgz', 'load_npy', 'load_npz', 'load_pth',
    'dump', 'dump_pkl', 'dump_pklgz', 'dump_npy', 'dump_npz', 'dump_pth',
    'link', 'mkdir', 'remove', 'locate_newest_file', 'io_function_registry'
]

sys_open = open


def as_file_descriptor(fd_or_fname, mode='r'):
    if type(fd_or_fname) is str:
        return sys_open(fd_or_fname, mode)
    return fd_or_fname


def open_h5(file, mode, **kwargs):
    import h5py
    return h5py.File(file, mode, **kwargs)


def open_txt(file, mode, **kwargs):
    return sys_open(file, mode, **kwargs)


def open_gz(file, mode):
    return gzip.open(file, mode)


def load_pkl(file, **kwargs):
    with as_file_descriptor(file, 'rb') as f:
        try:
            return pickle.load(f, **kwargs)
        except UnicodeDecodeError:
            if 'encoding' in kwargs:
                raise
            return pickle.load(f, encoding='latin1', **kwargs)


def load_pklgz(file, **kwargs):
    with open_gz(file, 'rb') as f:
        return load_pkl(f)


def load_h5(file, **kwargs):
    return open_h5(file, 'r', **kwargs)


def load_txt(file, **kwargs):
    with sys_open(file, 'r', **kwargs) as f:
        return f.readlines()


def load_npy(file, **kwargs):
    return np.load(file, **kwargs)


def load_npz(file, **kwargs):
    return np.load(file, **kwargs)


def load_pth(file, **kwargs):
    import torch
    return torch.load(file, **kwargs)


def dump_pkl(file, obj, **kwargs):
    with as_file_descriptor(file, 'wb') as f:
        return pickle.dump(obj, f, **kwargs)


def dump_pklgz(file, obj, **kwargs):
    with open_gz(file, 'wb') as f:
        return pickle.dump(obj, f)


def dump_npy(file, obj, **kwargs):
    return np.save(file, obj)


def dump_npz(file, obj, **kwargs):
    return np.savez(file, obj)


def dump_pth(file, obj, **kwargs):
    import torch
    return torch.save(obj, file)


class _IOFunctionRegistryGroup(RegistryGroup):
    __base_class__ = CallbackRegistry

    def dispatch(self, registry_name, file, *args, **kwargs):
        entry = get_ext(file)
        callback = self.lookup(registry_name, entry, fallback=True, default=_default_io_fallback)
        return callback(file, *args, **kwargs)


def _default_io_fallback(file, *args, **kwargs):
    raise ValueError('Unknown file extension: "{}".'.format(file))


io_function_registry = _IOFunctionRegistryGroup()
io_function_registry.register('open', '.txt', open_txt)
io_function_registry.register('open', '.h5', open_h5)
io_function_registry.register('open', '.gz', open_gz)
io_function_registry.register('open', '__fallback__', sys_open)

io_function_registry.register('load', '.pkl',   load_pkl)
io_function_registry.register('load', '.pklgz', load_pklgz)
io_function_registry.register('load', '.txt',   load_txt)
io_function_registry.register('load', '.h5',    load_h5)
io_function_registry.register('load', '.npy',   load_npy)
io_function_registry.register('load', '.npz',   load_npz)
io_function_registry.register('load', '.pth',   load_pth)

io_function_registry.register('dump', '.pkl',   dump_pkl)
io_function_registry.register('dump', '.pklgz', dump_pklgz)
io_function_registry.register('dump', '.npy',   dump_npy)
io_function_registry.register('dump', '.npz',   dump_npz)
io_function_registry.register('dump', '.pth',   dump_pth)


_fs_verbose = False


@contextlib.contextmanager
def fs_verbose(mode=True):
    global _fs_verbose

    _fs_verbose, mode = mode, _fs_verbose
    yield
    _fs_verbose = mode


def set_fs_verbose(mode):
    global _fs_verbose
    _fs_verbose = mode


def open(file, mode, **kwargs):
    if _fs_verbose and isinstance(file, six.string_types):
        logger.info('Opening file: "{}", mode={}.'.format(file, mode))
    return io_function_registry.dispatch('open', file, mode, **kwargs)


def load(file, **kwargs):
    if _fs_verbose and isinstance(file, six.string_types):
        logger.info('Loading data from file: "{}".'.format(file))
    return io_function_registry.dispatch('load', file, **kwargs)


def dump(file, obj, **kwargs):
    if _fs_verbose and isinstance(file, six.string_types):
        logger.info('Dumping data to file: "{}".'.format(file))
    return io_function_registry.dispatch('dump', file, obj, **kwargs)


def link(path_origin, *paths, use_relative_path=True):
    for item in paths:
        if os.path.exists(item):
            os.remove(item)
        if use_relative_path:
            src_path = os.path.relpath(path_origin, start=os.path.dirname(item))
        else:
            src_path = path_origin
        os.symlink(src_path, item)


def mkdir(path):
    return os.makedirs(path, exist_ok=True)


def remove(file):
    if osp.exists(file):
        if osp.isdir(file):
            shutil.rmtree(file, ignore_errors=True)
        if osp.isfile(file):
            os.remove(file)


def locate_newest_file(dirname, pattern):
    assert osp.isdir(dirname)
    fs = glob.glob(osp.join(dirname, pattern))
    return max(fs, key=osp.getmtime)
