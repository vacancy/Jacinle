# -*- coding:utf-8 -*-
# File   : git.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 18/01/2018
# 
# This file is part of Jacinle.

import subprocess


def get_git_revision_hash(short=False):
    if short:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()


def get_git_uncommitted_files():
    return subprocess.check_output(['git', 'status', '--porcelain']).strip().split('\n')
