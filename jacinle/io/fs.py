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
import io
import contextlib
import tempfile as tempfile_lib
from typing import Optional, Union, List

import pickle
import gzip
import numpy as np

from jacinle.logging import get_logger
from jacinle.utils.enum import JacEnum
from jacinle.utils.filelock import FileLock
from jacinle.utils.registry import RegistryGroup, CallbackRegistry

from .common import get_ext

logger = get_logger(__file__)

__all__ = [
    'as_file_descriptor', 'fs_verbose', 'set_fs_verbose',
    'open', 'open_txt', 'open_h5', 'open_gz',
    'load', 'load_txt', 'load_h5', 'load_pkl', 'load_pklgz', 'load_npy', 'load_npz', 'load_mat', 'load_pth',
    'dump', 'dump_pkl', 'dump_pklgz', 'dump_npy', 'dump_npz', 'dump_mat', 'dump_pth',
    'safe_dump',
    'link', 'mkdir', 'lsdir', 'remove', 'locate_newest_file', 'tempfile',
    'io_function_registry'
]

sys_open = open


def as_file_descriptor(fd_or_fname: Union[str, io.IOBase], mode: str = 'r') -> io.IOBase:
    """Convert a file descriptor or a file name to a file descriptor.

    Args:
        fd_or_fname: a file descriptor or a file name.
        mode: the mode to open the file if `fd_or_fname` is a file name.

    Returns:
        a file descriptor.
    """
    if type(fd_or_fname) is str:
        return sys_open(fd_or_fname, mode)
    return fd_or_fname


def open_h5(filename: str, mode: str, **kwargs):
    """Open a HDF5 file."""
    import h5py
    return h5py.File(filename, mode, **kwargs)


def open_txt(filename, mode, **kwargs):
    """Open a text file."""
    return sys_open(filename, mode, **kwargs)


def open_gz(filename, mode):
    """Open a gzip file."""
    return gzip.open(filename, mode)


def load_pkl(fd_or_filename: Union[str, io.IOBase], **kwargs):
    """Load a pickle file."""
    with as_file_descriptor(fd_or_filename, 'rb') as f:
        try:
            return pickle.load(f, **kwargs)
        except UnicodeDecodeError:
            if 'encoding' in kwargs:
                raise
            return pickle.load(f, encoding='latin1', **kwargs)


def load_pklgz(filename: str, **kwargs):
    """Load a gziped pickle file."""
    with open_gz(filename, 'rb') as f:
        return load_pkl(f)


def load_h5(filename: str, **kwargs):
    """Load a HDF5 file."""
    return open_h5(filename, 'r', **kwargs)


def load_txt(filename, **kwargs):
    """Load a text file."""
    with sys_open(filename, 'r', **kwargs) as f:
        return f.readlines()


def load_npy(fd_or_filename: Union[str, io.IOBase], **kwargs):
    """Load a npy numpy file."""
    return np.load(fd_or_filename, **kwargs)


def load_npz(fd_or_filename: Union[str, io.IOBase], **kwargs):
    """Load a npz numpy file."""
    return np.load(fd_or_filename, **kwargs)


def load_mat(filename: str, **kwargs):
    """Load a matlab file."""
    import scipy.io as sio
    return sio.loadmat(filename, **kwargs)


def load_pth(filename, **kwargs):
    """Load a PyTorch file."""
    import torch
    return torch.load(filename, **kwargs)


def dump_pkl(fd_or_filename: Union[str, io.IOBase], obj, **kwargs):
    """Dump a pickle file."""
    with as_file_descriptor(fd_or_filename, 'wb') as f:
        return pickle.dump(obj, f, **kwargs)


def dump_pklgz(filename: str, obj, **kwargs):
    """Dump a gziped pickle file."""
    with open_gz(filename, 'wb') as f:
        return pickle.dump(obj, f)


def dump_npy(filename: str, obj, **kwargs):
    """Dump a npy numpy file."""
    return np.save(filename, obj)


def dump_npz(filename: str, obj, **kwargs):
    """Dump a npz numpy file."""
    return np.savez(filename, obj)


def dump_mat(filename, obj, **kwargs):
    """Dump a matlab file."""
    import scipy.io as sio
    return sio.savemat(filename, obj, **kwargs)


def dump_pth(filename, obj, **kwargs):
    """Dump a PyTorch file."""
    import torch
    return torch.save(obj, filename)


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
io_function_registry.register('load', '.mat',   load_mat)
io_function_registry.register('load', '.pth',   load_pth)

io_function_registry.register('dump', '.pkl',   dump_pkl)
io_function_registry.register('dump', '.pklgz', dump_pklgz)
io_function_registry.register('dump', '.npy',   dump_npy)
io_function_registry.register('dump', '.npz',   dump_npz)
io_function_registry.register('dump', '.mat',   dump_mat)
io_function_registry.register('dump', '.pth',   dump_pth)


_fs_verbose = False


@contextlib.contextmanager
def fs_verbose(mode=True):
    """A context manager to enable/disable verbose mode in file system operations."""
    global _fs_verbose

    _fs_verbose, mode = mode, _fs_verbose
    yield
    _fs_verbose = mode


def set_fs_verbose(mode: bool = True):
    """Enable/disable verbose mode in file system operations."""
    global _fs_verbose
    _fs_verbose = mode


def open(filename: str, mode: str, **kwargs):
    """Open a file."""
    if _fs_verbose and isinstance(filename, str):
        logger.info('Opening file: "{}", mode={}.'.format(filename, mode))
    return io_function_registry.dispatch('open', filename, mode, **kwargs)


def load(filename: str, **kwargs):
    """Load a file with automatic file type detection."""
    if _fs_verbose and isinstance(filename, str):
        logger.info('Loading data from file: "{}".'.format(filename))
    return io_function_registry.dispatch('load', filename, **kwargs)


def dump(filename, obj, **kwargs):
    """Dump a file with automatic file type detection."""
    if _fs_verbose and isinstance(filename, str):
        logger.info('Dumping data to file: "{}".'.format(filename))
    return io_function_registry.dispatch('dump', filename, obj, **kwargs)


def safe_dump(filename: str, data, use_lock=True, use_temp=True, lock_timeout=10) -> bool:
    """Dump data to a file in a safe way. Basically, it will dump the data to a temporary file and
    then move it to the target file. This is to avoid the case that the target file is corrupted
    when the program is interrupted during the dumping process. It also supports file locking to
    avoid the case that multiple processes are dumping data to the same file at the same time.

    Args:
        filename: the target file name.
        data: the data to be dumped.
        use_lock: whether to use file locking.
        use_temp: whether to use a temporary file.
        lock_timeout: the timeout for file locking.

    Returns:
        If uses temp file, return True if the data is dumped to the temp file successfully, otherwise False.
        If not use temp file, return the result of the dump operation.
    """
    temp_fname = 'temp.' + filename
    lock_fname = 'lock.' + filename

    def safe_dump_inner():
        if use_temp:
            dump(temp_fname, data)
            os.replace(temp_fname, filename)
            return True
        else:
            return dump(temp_fname, data)

    if use_lock:
        with FileLock(lock_fname, lock_timeout) as flock:
            if flock.is_locked:
                return safe_dump_inner()
            else:
                logger.warning('Cannot lock the file: {}.'.format(filename))
                return False
    else:
        return safe_dump_inner()


def link(path_origin: str, *paths: str, use_relative_path=True):
    """Create a symbolic link to a file or directory.

    Args:
        path_origin: the original file or directory.
        paths: the symbolic links to be created.
        use_relative_path: whether to use relative path.
    """
    for item in paths:
        if os.path.exists(item):
            os.remove(item)
        if use_relative_path:
            src_path = os.path.relpath(path_origin, start=os.path.dirname(item))
        else:
            src_path = path_origin
        os.symlink(src_path, item)


def mkdir(path):
    """Create a directory if it does not exist without raising errors when the directory already exists."""
    return os.makedirs(path, exist_ok=True)


class LSDirectoryReturnType(JacEnum):
    BASE = 'base'
    NAME = 'name'
    REL = 'rel'
    FULL = 'full'
    REAL = 'real'


def lsdir(dirname: str, pattern: Optional[str] = None, return_type: Union[str, LSDirectoryReturnType] = 'full', sort: bool = True) -> List[str]:
    """List all files in a directory.

    Args:
        dirname: the directory name.
        pattern: the file name pattern in glob format.
        return_type: the return type. Can be one of the following:
            'base': return the base name of the file.
            'name': return the file name.
            'rel': return the relative path of the file.
            'full': return the full path of the file.
            'real': return the real path of the file.
        sort: whether to sort the file names.

    Returns:
        a list of file names.
    """
    if sort:
        return sorted(lsdir(dirname, pattern, return_type=return_type, sort=False))

    assert '*' in dirname or '?' in dirname or osp.isdir(dirname)

    return_type = LSDirectoryReturnType.from_string(return_type)
    if pattern is not None:
        files = glob.glob(osp.join(dirname, pattern), recursive=True)
    elif '*' in dirname:
        files = glob.glob(dirname)
    else:
        files = os.listdir(dirname)

    if return_type is LSDirectoryReturnType.BASE:
        return [osp.basename(f) for f in files]
    elif return_type is LSDirectoryReturnType.NAME:
        return [osp.splitext(osp.basename(f))[0] for f in files]
    elif return_type is LSDirectoryReturnType.REL:
        assert '*' not in dirname and '?' not in dirname, 'Cannot use * or ? for relative paths.'
        return [osp.relpath(f, dirname) for f in files]
    elif return_type is LSDirectoryReturnType.FULL:
        return files
    elif return_type is LSDirectoryReturnType.REAL:
        return [osp.realpath(osp.join(dirname, f)) for f in files]
    else:
        raise ValueError('Unknown lsdir return type: {}.'.format(return_type))


def remove(file_or_dirname: str):
    """Remove a file or directory."""
    if osp.exists(file_or_dirname):
        if osp.isdir(file_or_dirname):
            shutil.rmtree(file_or_dirname, ignore_errors=True)
        if osp.isfile(file_or_dirname):
            os.remove(file_or_dirname)


def locate_newest_file(dirname: str, pattern: str) -> Optional[str]:
    """Locate the newest file in a directory. If there is no file matching the pattern, return None.

    Args:
        dirname: the directory name.
        pattern: the file name pattern in glob format.

    Returns:
        the full path of the newest file.
    """
    fs = lsdir(dirname, pattern, return_type='full')
    if len(fs) == 0:
        return None
    return max(fs, key=osp.getmtime)


@contextlib.contextmanager
def tempfile(mode: str = 'w+b', suffix: str = '', prefix: str = 'tmp'):
    """A context manager that creates a temporary file and deletes it after use.

    Example:
        .. code-block:: python

            with tempfile() as f:
                f.write(b'hello world')
                f.seek(0)
                print(f.read())

    Args:
        mode: the mode to open the file.
        suffix: the suffix of the file name.
        prefix: the prefix of the file name.
    """
    f = tempfile_lib.NamedTemporaryFile(mode, suffix=suffix, prefix=prefix, delete=False)
    yield f
    os.unlink(f.name)

