#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/16/2020
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from typing import Optional, Union, List
import numpy as np

from jacinle.jit.cext import auto_travis
auto_travis(__file__)

from .logic_induction import (
    search as csearch,
    LogicFormTypePy as LogicFormType,
    LogicFormOutputFormatPy as LogicFormOutputFormat
)

__all__ = ['search']


def search(
    inputs: np.ndarray,
    outputs: np.ndarray,
    input_names: List[str],
    type: Optional[Union[str, LogicFormType]] = 'GENERAL',
    output_format: Optional[Union[str, LogicFormOutputFormat]] = 'DEFAULT',
    depth: Optional[int] = 4,
    coverage: Optional[float] = 0.99
):
    return csearch(inputs, outputs, input_names, type=type, output_format=output_format, depth=depth, coverage=coverage)


