#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : enum.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import six
import enum

__all__ = ['JacEnum']


class JacEnum(enum.Enum):
    """A customized enumeration class, adding helper functions for string-based argument parsing."""

    @classmethod
    def from_string(cls, value):
        value = _canonize_enum_value(value, True)
        if isinstance(value, six.string_types) and hasattr(cls, value):
            return getattr(cls, value)
        value = _canonize_enum_value(value)
        return cls(value)

    @classmethod
    def type_name(cls):
        return cls.__name__

    @classmethod
    def choice_names(cls):
        return list(filter(lambda x: not x.startswith('_'), dir(cls)))

    @classmethod
    def choice_objs(cls):
        return [getattr(cls, name) for name in cls.choice_names()]

    @classmethod
    def choice_values(cls):
        return [getattr(cls, name).value for name in cls.choice_names()]

    @classmethod
    def is_valid(cls, value):
        value = _canonize_enum_value(value)
        return value in cls.choice_values()

    @classmethod
    def assert_valid(cls, value):
        assert cls.is_valid(value), 'Invalid {}: "{}". Supported choices: {}.'.format(
            cls.type_name(), value, ','.join(cls.choice_values()))

    def __jsonify__(self):
        return self.value


def _canonize_enum_value(value, cap=False):
    if isinstance(value, six.string_types):
        if cap:
            value = value.upper()
        else:
            value = value.lower()
    return value

