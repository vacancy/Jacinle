#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : clipboard.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/12/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Utility functions for clipboard operations."""

import platform
import numpy as np
import subprocess
import tempfile
import os

__all__ = ['copy_to_clipboard', 'copy_to_clipboard_image', 'paste_from_clipboard']


def copy_to_clipboard(text: str) -> None:
    raise NotImplementedError()


def copy_to_clipboard_image(img: np.ndarray, is_rgb: bool = True) -> None:
    raise NotImplementedError()


def paste_from_clipboard() -> str:
    raise NotImplementedError()


def assert_cv2_available():
    try:
        import cv2
    except ImportError:
        raise ImportError('cv2 is required for this function.')
    return cv2


if platform.system() == 'Darwin':
    def copy_to_clipboard(text: str) -> None:
        subprocess.run('pbcopy', universal_newlines=True, input=text)

    def copy_to_clipboard_image(img: np.ndarray, is_rgb: bool = True) -> None:
        cv2 = assert_cv2_available()

        if is_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        tmpfile = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(tmpfile.name, img)
        copy_to_clipboard_image_path(tmpfile.name)
        os.unlink(tmpfile.name)

    def copy_to_clipboard_image_path(path: str) -> None:
        subprocess.run(['osascript', '-e', f'set the clipboard to (read (POSIX file "{path}") as JPEG picture)'])

    def paste_from_clipboard() -> str:
        return subprocess.run('pbpaste', universal_newlines=True, stdout=subprocess.PIPE).stdout
elif platform.system() == 'Linux':
    def copy_to_clipboard(text: str) -> None:
        subprocess.run(['xclip', '-selection', 'clipboard'], universal_newlines=True, input=text)

    def copy_to_clipboard_image(img: np.ndarray, is_rgb: bool = True) -> None:
        cv2 = assert_cv2_available()

        if is_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        tmpfile = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        cv2.imwrite(tmpfile.name, img)
        subprocess.run(['xclip', '-selection', 'clipboard', '-t', 'image/png', '-i', tmpfile.name])
        os.unlink(tmpfile.name)

    def paste_from_clipboard() -> str:
        return subprocess.run(['xclip', '-selection', 'clipboard', '-o'], universal_newlines=True, stdout=subprocess.PIPE).stdout
else:
    pass
