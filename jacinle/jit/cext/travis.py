#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : travis.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/04/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import sys
import os.path as osp
import glob
import subprocess

from typing import Optional, Sequence
from jacinle.logging import get_logger

logger = get_logger(__file__)

__all__ = ['auto_travis']


def auto_travis(filename: str, required_files: Optional[Sequence[str]] = None, required_imports: Optional[Sequence[str]] = None, use_glob: bool = True, force_recompile: bool = False):
    """A simple function to automatically run the ``./travis.sh`` script in the
    current directory. The most common usage is to put the following code in
    your ``__init__.py`` file and add ``auto_travis(__file__)`` to the end of
    the file.

    Args:
        filename: the ``__file__`` variable of the current module.
        required_files: a list of required files to check. If not provided, use ``['**/*.so']``.
        required_imports: a list of required modules to import. If not provided, use required_files to check.
        use_glob: whether to use glob to check the required files.
    """
    compiled = True
    dirname = osp.dirname(filename)

    if required_imports is not None:
        sys.path.insert(0, dirname)
        if isinstance(required_imports, str):
            required_imports = [required_imports]
        for module in required_imports:
            try:
                __import__(module)
            except ImportError:
                compiled = False
                break
        sys.path.pop(0)
    else:
        # Use required_files to check.
        if required_files is None:
            required_files = ['**/*.so']

        for fname in required_files:
            fname = osp.join(dirname, fname)
            if not osp.exists(fname):
                if not use_glob or len(glob.glob(osp.join(dirname, fname), recursive=True)) == 0:
                    compiled = False

    if compiled and not force_recompile:
        logger.critical('Loading c extension from: "{}".'.format(osp.realpath(dirname)))
    else:
        logger.critical('Compiling c extension at: "{}".'.format(osp.realpath(dirname)))
        subprocess.check_call(['./travis.sh'], cwd=dirname)
