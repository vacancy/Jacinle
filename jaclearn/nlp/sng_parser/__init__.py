#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/04/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from jacinle.utils.vendor import has_vendor, requires_vendors

__all__ = ['tprint', 'Parser', 'get_default_parser', 'parse']


if has_vendor('sng_parser'):
    from sng_parser import tprint, Parser, get_default_parser, parse
else:
    from jacinle.utils.meta import make_dummy_func
    tprint = requires_vendors('sng_parser')(make_dummy_func())
    Parser = requires_vendors('sng_parser')(make_dummy_func())
    get_default_parser = requires_vendors('sng_parser')(make_dummy_func())
    parse = requires_vendors('sng_parser')(make_dummy_func())

