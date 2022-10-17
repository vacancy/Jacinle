#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : git.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import glob
import os.path as osp
import subprocess


def git_current_tracking_remote():
    try:
        string = subprocess.check_output(['git', 'rev-parse', '--symbolic-full-name', '--abbrev-ref', '@{u}']).decode('utf-8').strip()
        return string.split('/')[0]
    except subprocess.CalledProcessError:
        return None


def git_remote_url(remote_identifier=None):
    if remote_identifier is None:
        remote_identifier = git_current_tracking_remote()
    try:
        string = subprocess.check_output(['git', 'remote', 'get-url', str(remote_identifier)]).decode('utf-8').strip()
        return string
    except subprocess.CalledProcessError:
        return None


def git_recent_logs(revision_hash, n=5):
    try:
        string = subprocess.check_output(['git', '--no-pager', 'log', str(revision_hash), '-n', str(n)]).decode('utf-8').strip()
        return string
    except subprocess.CalledProcessError:
        return None


def git_revision_hash(short=False):
    try:
        if short:
            return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except subprocess.CalledProcessError:
        return None


def git_uncommitted_files():
    try:
        files = subprocess.check_output(['git', 'status', '--porcelain']).decode('utf-8').strip().split('\n')
        files = [f.strip() for f in files if len(f.strip()) > 0]
        return files
    except subprocess.CalledProcessError:
        return []


def git_root():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--show-cdup']).decode('utf-8').strip()
    except subprocess.CalledProcessError:
        return None


LARGE_FILE_THRESH = 128 * 1024  # 128 kb


def git_status_full():
    try:
        fmt = subprocess.check_output(['git', 'status', '-vv']).decode('utf-8').strip() + '\n'
        fmt += '--------------------------------------------------\n'
        fmt += 'Changes not tracked:\n'
    except subprocess.CalledProcessError:
        return 'git status failed.'

    for filename in git_uncommitted_files():
        if filename.startswith('?? '):
            fname = filename[3:]
            fmt += _git_diff_no_index(osp.join(git_root(), fname))
    return fmt


def _git_diff_no_index(fname):
    if osp.isdir(fname):
        fmt = f'{fname} is a directory.\n'
        for x in sorted(glob.glob(osp.join(fname, '**', '*'), recursive=True)):
            if osp.isfile(x):
                fmt += _git_diff_no_index(x)
        return fmt
    if osp.getsize(fname) > LARGE_FILE_THRESH:
        return f'{fname} is too large to be diffed.\n'
    else:
        return subprocess.run(['git', '--no-pager', 'diff', '--no-index', '/dev/null', fname], stdout=subprocess.PIPE, check=False).stdout.decode('utf-8').strip() + '\n'


def git_guard(force=False):
    uncommitted_files = git_uncommitted_files()
    if len(uncommitted_files) > 0:
        from jacinle.logging import get_logger
        from jacinle.cli.keyboard import yes_or_no
        logger = get_logger(__file__)

        logger.warning('Uncommited changes at the current repo:\n  ' + '\n  '.join(uncommitted_files))
        if force:
            if not yes_or_no('Are you sure you want to continue?', default='no'):
                exit(1)
        logger.info(git_status_full())

