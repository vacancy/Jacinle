# -*- coding: utf-8 -*-
# File   : fs.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/19/17
#
# This file is part of Jacinle.


import os
import os.path as osp
import glob

import pickle
import gzip
import numpy as np

from jacinle.utils.registry import RegistryGroup, CallbackRegistry

from .common import get_ext

__all__ = [
    'as_file_descriptor',
    'open', 'open_h5', 'open_gz',
    'load', 'load_h5', 'load_pkl', 'load_pklgz', 'load_npy', 'load_npz', 'load_pth',
    'save', 'save_pkl', 'save_pklgz', 'save_npy', 'save_npz', 'save_pth',
    'link', 'mkdir', 'locate_newest_file'
]

sys_open = open


def as_file_descriptor(fd_or_fname, mode='r'):
    if type(fd_or_fname) is str:
        return sys_open(fd_or_fname, mode)
    return fd_or_fname


def open_h5(file, mode, **kwargs):
    import h5py
    return h5py.File(file, mode, **kwargs)


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


def load_npy(file, **kwargs):
    return np.load(file, **kwargs)


def load_npz(file, **kwargs):
    return np.load(file, **kwargs)


def load_pth(file, **kwargs):
    import torch
    return torch.load(file, **kwargs)


def save_pkl(file, obj, **kwargs):
    with as_file_descriptor(file, 'wb') as f:
        return pickle.dump(obj, f, **kwargs)


def save_pklgz(file, obj, **kwargs):
    with open_gz(file, 'wb') as f:
        return pickle.dump(obj, f)


def save_npy(file, obj, **kwargs):
    return np.save(file, obj)


def save_npz(file, obj, **kwargs):
    return np.savez(file, obj)


def save_pth(file, obj, **kwargs):
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
io_function_registry.register('open', '.h5', open_h5)
io_function_registry.register('open', '.gz', open_gz)
io_function_registry.register('open', '__fallback__', sys_open)

io_function_registry.register('load', '.pkl',   load_pkl)
io_function_registry.register('load', '.pklgz', load_pklgz)
io_function_registry.register('load', '.h5',    load_h5)
io_function_registry.register('load', '.npy',   load_npy)
io_function_registry.register('load', '.npz',   load_npz)
io_function_registry.register('load', '.pth',   load_pth)

io_function_registry.register('save', '.pkl',   save_pkl)
io_function_registry.register('save', '.pklgz', save_pklgz)
io_function_registry.register('save', '.npy',   save_npy)
io_function_registry.register('save', '.npz',   save_npz)
io_function_registry.register('save', '.pth',   save_pth)


def open(file, mode, **kwargs):
    return io_function_registry.dispatch('open', file, mode, **kwargs)


def load(file, **kwargs):
    return io_function_registry.dispatch('load', file, **kwargs)


def save(file, obj, **kwargs):
    return io_function_registry.dispatch('save', file, obj, **kwargs)


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


def locate_newest_file(dirname, pattern):
    assert osp.isdir(dirname)
    fs = glob.glob(osp.join(dirname, pattern))
    return max(fs, key=osp.getmtime)
