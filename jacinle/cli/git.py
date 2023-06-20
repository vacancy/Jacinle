#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : git.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""Utility functions to get git information of the current directory."""

import glob
import os.path as osp
import subprocess
from typing import Optional, List


def git_current_tracking_remote() -> str:
    """Get the current tracking remote.

    Returns:
        the name of the current tracking remote.
    """
    try:
        string = subprocess.check_output(['git', 'rev-parse', '--symbolic-full-name', '--abbrev-ref', '@{u}']).decode('utf-8').strip()
        return string.split('/')[0]
    except subprocess.CalledProcessError:
        return None


def git_remote_url(remote_identifier: Optional[str] = None) -> str:
    """Get the URL of the remote.

    Args:
        remote_identifier: the identifier of the remote. If None, use the current tracking remote.

    Returns:
        the URL of the remote.
    """
    if remote_identifier is None:
        remote_identifier = git_current_tracking_remote()
    try:
        string = subprocess.check_output(['git', 'remote', 'get-url', str(remote_identifier)]).decode('utf-8').strip()
        return string
    except (subprocess.CalledProcessError, TypeError):
        # when the returned value is None there will be a type error.
        return None


def git_recent_logs(revision_hash: str, n: int = 5) -> str:
    """Get the recent logs of the given revision hash.

    Args:
        revision_hash: the revision hash.
        n: the number of logs to be returned.

    Returns:
        the recent logs as a single string.
    """
    try:
        string = subprocess.check_output(['git', '--no-pager', 'log', str(revision_hash), '-n', str(n)]).decode('utf-8').strip()
        return string
    except subprocess.CalledProcessError:
        return None


def git_revision_hash(short: bool = False) -> str:
    """Get the current revision hash.

    Args:
        short: whether to use the short version of the hash.

    Returns:
        the current revision hash.
    """
    try:
        if short:
            return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except subprocess.CalledProcessError:
        return None


def git_uncommitted_files() -> List[str]:
    """Get the list of uncommitted files.

    Returns:
        the list of uncommitted files.
    """
    try:
        files = subprocess.check_output(['git', 'status', '--porcelain']).decode('utf-8').strip().split('\n')
        files = [f.strip() for f in files if len(f.strip()) > 0]
        return files
    except subprocess.CalledProcessError:
        return []


def git_root() -> str:
    """Get the root directory of the git repo.

    Returns:
        the root directory of the git repo.
    """
    try:
        return subprocess.check_output(['git', 'rev-parse', '--show-cdup']).decode('utf-8').strip()
    except subprocess.CalledProcessError:
        return None


LARGE_FILE_THRESH = 128 * 1024  # 128 kb


def git_status_full():
    """Get the full status of the current git repo. This includes the content of the untracked files and the diff of the uncommitted files.
    Note that when the file is too large (larger than 128 kb), its content will not be shown.

    Returns:
        a single string containing the full status of the current git repo.
    """
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
        try:
            return subprocess.run(['git', '--no-pager', 'diff', '--no-index', '/dev/null', fname], stdout=subprocess.PIPE, check=False).stdout.decode('utf-8').strip() + '\n'
        except Exception as e:
            return f'{fname} failed in git-diff. Exception: {e}'


def git_guard(force: bool = False):
    """A utility function to guard the current git repo. It will check whether there are uncommitted files.

    - When ``force`` is False, it will print a warning message including the list of uncommitted files and the diff of the uncommitted files.
    - When ``force`` is True, it will ask a confirmation from the user. If the user confirms, it will return True. Otherwise, it will terminate the program.
    """
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

