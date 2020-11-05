#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : git.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import subprocess


def get_git_revision_hash(short=False):
    """
    Returns the sha1 hash of the git repo.

    Args:
        short: (str): write your description
    """
    if short:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()


def get_git_uncommitted_files():
    """
    Returns a list of all git files. git repo.

    Args:
    """
    return subprocess.check_output(['git', 'status', '--porcelain']).decode('utf-8').strip().split('\n')
