#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : travis.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/04/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import os.path as osp
import glob
import subprocess

from typing import Optional, Sequence
from jacinle.logging import get_logger

logger = get_logger(__file__)

__all__ = ['auto_travis']


def auto_travis(filename: str, required_files: Optional[Sequence[str]] = None, use_glob: bool = True):
    """A simple function to automatically run the ``./travis.sh`` script in the
    current directory. The most common usage is to put the following code in
    your ``__init__.py`` file and add ``auto_travis(__file__)`` to the end of
    the file.

    Args:
        filename: the ``__file__`` variable of the current module.
        required_files: a list of required files to check. If not provided, use ``['**/*.so']``.
        use_glob: whether to use glob to check the required files.
    """
    if required_files is None:
        required_files = ['**/*.so']

    dirname = osp.dirname(filename)

    compiled = True
    for fname in required_files:
        fname = osp.join(dirname, fname)
        if not osp.exists(fname):
            if not use_glob or len(glob.glob(osp.join(dirname, fname), recursive=True)) == 0:
                compiled = False

    if compiled:
        logger.critical('Loading c extension from: "{}".'.format(osp.realpath(dirname)))
    else:
        logger.critical('Compiling c extension at: "{}".'.format(osp.realpath(dirname)))
        subprocess.check_call(['./travis.sh'], cwd=dirname)
