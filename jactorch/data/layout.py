#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : layout.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/09/2020
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import six
from typing import Union, Optional
from jacinle.utils.enum import JacEnum

__all__ = ['DataLayoutType', 'DataLayoutSpec', 'DataLayout', 'skip', 'concat', 'pad', 'pad2d', 'padimage']


class DataLayoutType(JacEnum):
    SKIP = 'skip'
    CONCAT = 'concat'
    PAD = 'pad'
    PAD2D = 'pad2d'
    PADIMAGE = 'padimage'


class DataLayoutSpec(object):
    def __init__(self, type, **kwargs):
        self.__dict__.update(kwargs)
        self.type = DataLayoutType.from_string(type)


def skip():
    return DataLayoutSpec('skip')


def concat():
    return DataLayoutSpec('concat')


def pad(fill=0):
    return DataLayoutSpec('pad', fill=fill)


def pad2d(fill=0):
    return DataLayoutSpec('pad2d', fill=fill)


def padimage(fill=0):
    return DataLayoutSpec('pad2d', fill=fill)


_type_to_constructor = {
    'skip': skip,
    'concat': concat,
    'pad': pad,
    'pad2d': pad2d,
    'padimage': padimage
}


class DataLayout(object):
    def __init__(self, layout: Optional[dict] = None):
        self.layout = dict()

        if layout is not None:
            for k, v in layout.items():
                self.decl(k, v)

    def decl(self, key: str, spec: Union[str, DataLayoutSpec]):
        if isinstance(spec, six.string_types):
            self.layout[key] = _type_to_constructor[spec]()
        else:
            self.layout[key] = spec

    def __contains__(self, key: str):
        return key in self.layout

    def __getitem__(self, key: str):
        return self.layout.get(key, None)

