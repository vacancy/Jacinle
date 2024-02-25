#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : network.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2017
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import os
import os.path as osp
import hashlib
import urllib

from .common import fsize_format
from jacinle.utils.tqdm import tqdm_pbar

__all__ = ['download', 'check_integrity']


def download(url: str, dirname: str, cli: bool = True, filename: str = None, md5: str = None):
    """
    Download URL to a directory. Will figure out the filename automatically from URL.
    Will figure out the filename automatically from URL, if not given.

    Args:
        url: URL to download.
        dirname: directory to download to.
        cli: whether to use CLI progress bar.
        filename: filename to save to.
        md5: md5 hash to check.

    Source: https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/utils/fs.py
    """

    if cli:
        from jacinle.cli.keyboard import maybe_mkdir
        maybe_mkdir(dirname)
    else:
        assert osp.isdir(dirname)

    filename = filename or url.split('/')[-1]
    path = os.path.join(dirname, filename)

    def hook(t):
        last_b = [0]

        def inner(b, bsize, tsize=None):
            if tsize is not None:
                t.total = tsize
            t.update((b - last_b[0]) * bsize)
            last_b[0] = b
        return inner
    try:
        with tqdm_pbar(unit='B', unit_scale=True, miniters=1, desc=filename) as pbar:
            path, _ = urllib.request.urlretrieve(url, path, reporthook=hook(pbar))

        statinfo = os.stat(path)
        size = statinfo.st_size
    except Exception:
        print('Failed to download {}.'.format(url))
        raise
    assert size > 0, "Download an empty file!"
    print('Successfully downloaded ' + filename + " " + fsize_format(size) + '.')

    if md5 is not None:
        assert check_integrity(path, md5), 'Integrity check for {} failed'.format(path)

    return path


def check_integrity(filename: str, md5: str) -> bool:
    """Check data integrity using md5 hashing.

    Args:
        filename: path to the file.
        md5: md5 hash to check.
    """
    if not os.path.isfile(filename):
        return False
    md5o = hashlib.md5()
    with open(filename, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True
