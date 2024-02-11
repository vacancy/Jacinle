#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : enum.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import enum
from typing import Union

__all__ = ['JacEnum']


class JacEnum(enum.Enum):
    """A customized enumeration class, adding helper functions for string-based argument parsing."""

    @classmethod
    def from_string(cls, value: Union[str, 'JacEnum']) -> 'JacEnum':
        if isinstance(value, JacEnum):
            return value
        value = _canonize_enum_value(value, True)
        if isinstance(value, str) and hasattr(cls, value):
            return getattr(cls, value)
        value = _canonize_enum_value(value)
        return cls(value)

    @classmethod
    def type_name(cls):
        """Return the type name of the enum."""
        return cls.__name__

    @classmethod
    def choice_names(cls):
        """Returns the list of the name of all possible choices."""
        return list(filter(lambda x: not x.startswith('_'), dir(cls)))

    @classmethod
    def choice_objs(cls):
        """Returns the list of the object of all possible choices."""
        return [getattr(cls, name) for name in cls.choice_names()]

    @classmethod
    def choice_values(cls):
        """Returns the list of the value of all possible choices."""
        return [getattr(cls, name).value for name in cls.choice_names()]

    @classmethod
    def is_valid(cls, value):
        """Check if the value is a valid choice."""
        value = _canonize_enum_value(value)
        return value in cls.choice_values()

    @classmethod
    def assert_valid(cls, value):
        """Assert if the value is a valid choice."""
        assert cls.is_valid(value), 'Invalid {}: "{}". Supported choices: {}.'.format(
            cls.type_name(), value, ','.join(cls.choice_values())
        )

    def __jsonify__(self):
        return self.value


def _canonize_enum_value(value, cap=False):
    if isinstance(value, str):
        if cap:
            value = value.upper()
        else:
            value = value.lower()
    return value

