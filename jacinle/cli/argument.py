#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : argument.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import os.path as osp
import argparse
import json

from jacinle.logging import get_logger
from jacinle.utils.enum import JacEnum
from jacinle.utils.printing import print_to
from jacinle.utils.container import G

from .device import DeviceNameFormat, parse_and_set_devices
from .keyboard import str2bool, yn2bool

__all__ = ['JacArgumentParser']

logger = get_logger(__file__)


class JacArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        """
        Initialize the class.

        Args:
            self: (todo): write your description
        """
        kwargs.setdefault('fromfile_prefix_chars', '@')
        kwargs.setdefault('formatter_class', argparse.ArgumentDefaultsHelpFormatter)
        super().__init__(*args, **kwargs)

        self.register('type', 'bool', _type_bool)
        self.register('type', 'checked_file', _type_checked_file)
        self.register('type', 'checked_dir', _type_checked_dir)
        self.register('type', 'ensured_dir', _type_ensured_dir)
        self.register('type', 'kv', _type_kv)
        self.register('action', 'set_device', SetDeviceAction)
        self.register('action', 'as_enum', AsEnumAction)


def _type_bool(string):
    """
    Convert a string to bool.

    Args:
        string: (str): write your description
    """
    try:
        return str2bool(string)
    except ValueError:
        raise argparse.ArgumentTypeError()


def _type_checked_file(string):
    """
    Checks if the given string is a string.

    Args:
        string: (str): write your description
    """
    if not osp.isfile(string):
        raise argparse.ArgumentTypeError('Check file existence failed: "{}".'.format(string))
    return string


def _type_checked_dir(string):
    """
    Checks if the given string is a valid checked.

    Args:
        string: (str): write your description
    """
    if not osp.isdir(string):
        raise argparse.ArgumentTypeError('Check directory existence failed: "{}".'.format(string))
    return string


def _type_ensured_dir(string):
    """
    Make a directory if it exists.

    Args:
        string: (str): write your description
    """
    if not osp.isdir(string):
        # TODO(Jiayuan Mao @ 05/08): change to a Y/N question.
        import jacinle.io as io
        io.mkdir(string)
    return string


class _KV(object):
    def __init__(self, string):
        """
        Parse a string.

        Args:
            self: (todo): write your description
            string: (str): write your description
        """
        self.string = string

        if len(self.string) > 0:
            kvs = list(string.split(';'))
        else:
            kvs = []

        for i, kv in enumerate(kvs):
            k, v = kv.split('=')

            try:
                bool_value = yn2bool(v)
                v = bool_value
                kvs[i] = (k, v)
                continue
            except ValueError:
                pass

            if v.startswith('"') or v.startswith("'"):
                assert v.endswith('"') or v.endswith("'")
                v = v[1:-1]

            try:
                v = float(v)
                if int(v) == v:
                    v = int(v)
            except:
                pass

            kvs[i] = (k, v)

        self.kvs = kvs

    def apply(self, configs):
        """
        Apply config values to configs.

        Args:
            self: (todo): write your description
            configs: (dict): write your description
        """
        with print_to(logger.info):
            print('Applying KVs:')
            for k, v in self.kvs:
                print('  kv.{} = {}'.format(k, v))
                keys = k.split('.')
                current = configs
                for k in keys[:-1]:
                    current = current.setdefault(k, G())
                current[keys[-1]] = v

    def __jsonify__(self):
        """
        Return the json representation of this object.

        Args:
            self: (todo): write your description
        """
        return json.dumps(self.kvs)


def _type_kv(string):
    """
    In the format of:
        --configs "data.int_or_float=int_value; data.string='string_value'"
    """

    return _KV(string)


class SetDeviceAction(argparse.Action):
    def __init__(self, option_strings, dest, format='int', set_device=True, nargs=None, const=None, default=None,
                 type=None, choices=None, required=False, help=None, metavar=None):
        """
        Initialize the command.

        Args:
            self: (todo): write your description
            option_strings: (str): write your description
            dest: (str): write your description
            format: (str): write your description
            set_device: (todo): write your description
            nargs: (todo): write your description
            const: (list): write your description
            default: (str): write your description
            type: (str): write your description
            choices: (todo): write your description
            required: (todo): write your description
            help: (todo): write your description
            metavar: (todo): write your description
        """

        DeviceNameFormat.assert_valid(format)
        self.format = format
        self.set_device = set_device

        super().__init__(option_strings=option_strings, dest=dest, nargs=nargs, const=const, default=default,
                         type=type, choices=choices, required=required, help=help, metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        """
        Call parser.

        Args:
            self: (todo): write your description
            parser: (todo): write your description
            namespace: (str): write your description
            values: (array): write your description
            option_string: (str): write your description
        """
        setattr(namespace, self.dest, parse_and_set_devices(values, self.format, self.set_device))


class AsEnumAction(argparse.Action):
    def __init__(self, option_strings, dest, type, nargs=None, const=None, default=None, choices=None,
                 required=False, help=None, metavar=None):
        """
        Initialize a enum option.

        Args:
            self: (todo): write your description
            option_strings: (str): write your description
            dest: (str): write your description
            type: (str): write your description
            nargs: (todo): write your description
            const: (list): write your description
            default: (str): write your description
            choices: (todo): write your description
            required: (todo): write your description
            help: (todo): write your description
            metavar: (todo): write your description
        """

        assert issubclass(type, JacEnum)

        self.enum_type = type
        if choices is None:
            choices = type.choice_values()
        if default is not None:
            default = self.enum_type.from_string(default)

        super().__init__(option_strings=option_strings, dest=dest, nargs=nargs, const=const, default=default,
                         type=None, choices=choices, required=required, help=help, metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        """
        Call the given parser.

        Args:
            self: (todo): write your description
            parser: (todo): write your description
            namespace: (str): write your description
            values: (array): write your description
            option_string: (str): write your description
        """
        if isinstance(values, (tuple, list)):
            setattr(namespace, self.dest, tuple(map(self.enum_type.from_string, values)))
        else:
            setattr(namespace, self.dest, self.enum_type.from_string(values))

