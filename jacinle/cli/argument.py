# -*- coding: utf-8 -*-
# File   : argument.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2018
# 
# This file is part of Jacinle.

import os.path as osp
import argparse

from .device import DeviceNameFormat, parse_and_set_devices
from .keyboard import str2bool


class JacArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register('type', 'bool', _type_bool)
        self.register('type', 'checked_file', _type_checked_file)
        self.register('type', 'checked_dir', _type_checked_dir)
        self.register('type', 'ensured_dir', _type_ensured_dir)
        self.register('action', 'set_device', SetDeviceAction)


def _type_bool(string):
    try:
        return str2bool(string)
    except ValueError:
        raise argparse.ArgumentTypeError()


def _type_checked_file(string):
    if not osp.isfile(string):
        raise argparse.ArgumentTypeError('Check file existence failed: "{}".'.format(string))
    return string


def _type_checked_dir(string):
    if not osp.isdir(string):
        raise argparse.ArgumentTypeError('Check directory existence failed: "{}".'.format(string))
    return string


def _type_ensured_dir(string):
    if not osp.isdir(string):
        # TODO(Jiayuan Mao @ 05/08): change to a Y/N question.
        import jacinle.io as io
        io.mkdir(string)
    return string


class SetDeviceAction(argparse.Action):
    def __init__(self, option_strings, dest, format='int', set_device=True, nargs=None, const=None, default=None,
                 type=None, choices=None, required=False, help=None, metavar=None):

        DeviceNameFormat.assert_valid(format)
        self.format = format
        self.set_device = set_device

        super().__init__(option_strings=option_strings, dest=dest, nargs=nargs, const=const, default=default,
                         type=type, choices=choices, required=required, help=help, metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, parse_and_set_devices(values, self.format, self.set_device))
