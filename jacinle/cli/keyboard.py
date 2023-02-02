#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : keyboard.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""Uiltity functions to parse keyboard inputs."""

import os
import sys
import os.path as osp
from typing import Optional

__all__ = ['str2bool', 'yn2bool', 'str2bool_long', 'yes_or_no', 'maybe_mkdir']


def str2bool(s: str) -> bool:
    """Convert a string to boolean value.

    Args:
        s: the string to be converted.

    Returns:
        True if the string is "yes", "true", "y", "t", "1";
        False if the string is "no", "false", "n", "f", "0";
        otherwise, raise ValueError.
    """
    if s.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif s.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('str2bool is undefined for: "{}".'.format(s))


def yn2bool(s: str) -> bool:
    """Convert a string to boolean value.

    Args:
        s: the string to be converted.

    Returns:
        True if the string is "y" or "yes";
        False if the string is "n" or "no";
        otherwise, raise ValueError.
    """
    if s.lower() in ('yes', 'y'):
        return True
    elif s.lower() in ('no', 'n'):
        return False
    else:
        raise ValueError('yn2bool is undefined for: "{}".'.format(s))


def str2bool_long(s: str) -> bool:
    """Convert a string to boolean value.

    Args:
        s: the string to be converted.

    Returns:
        True if the string is "yes", "true";
        False if the string is "no", "false";
        otherwise, raise ValueError.
    """
    if s.lower() in ('yes', 'true'):
        return True
    elif s.lower() in ('no', 'false'):
        return False
    else:
        raise ValueError('str2bool_long is undefined for: "{}".'.format(s))


def yes_or_no(question: str, default: Optional[str] = "yes") -> bool:
    """Ask a yes/no question via input() and return their answer.

    Args:
        question: the question to be asked.
        default: the default answer. It must be "yes" (the default), "no" or None (meaning that an answer is required from the user).
    """

    valid = {
        "yes": True, "y": True, "ye": True,
        "no": False, "n": False,
        "default": None, "def": None, "d": None
    }

    quiet = os.getenv('JAC_QUIET', '')
    if quiet != '':
        quiet = quiet.lower()
        assert quiet in valid, 'Invalid JAC_QUIET environ: {}.'.format(quiet)
        choice = valid[quiet]
        sys.stdout.write('Jacinle Quiet run:\n\tQuestion: {}\n\tChoice: {}\n'.format(question, 'Default' if choice is None else 'Yes' if choice else 'No'))
        return choice if choice is not None else default

    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("Invalid default answer: '%s'." % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def maybe_mkdir(dirname):
    """Make a directory if it does not exist.

    Args:
        dirname: the directory to be created.
    """
    if osp.isdir(dirname):
        return
    if osp.isfile(dirname):
        return

    import jacinle.io as io
    if yes_or_no('Creating directory "{}"?'.format(dirname)):
        io.mkdir(dirname)

