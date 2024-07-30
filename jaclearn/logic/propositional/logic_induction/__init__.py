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

try:
    from .logic_induction import (
        search as csearch,
        LogicFormTypePy as LogicFormType,
        LogicFormOutputFormatPy as LogicFormOutputFormat
    )
except ImportError:
    auto_travis(__file__, force_recompile=True)
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
) -> List[str]:
    """Search for logic forms inductively.

    Example:
        .. code-block:: python

            import numpy as np
            from jaclearn.logic.propositional.logic_induction import search

            inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype='uint8')
            outputs = np.array([[0], [1], [1], [0]], dtype='uint8')
            input_names = ['x', 'y']
            print(search(inputs, outputs, input_names))  # ['((x OR y) AND (NOT x OR NOT y))']

    Args:
        inputs: the input data, with shape (N, D).
        outputs: the output data, with shape (N, C).
        input_names: the names of the input variables, a list of length D.
        type: the type of the logic form (options are 'GENERAL', 'CONJUNCTION', 'DISJUNCTION').
        output_format: the output format of the logic form (options are 'DEFAULT', 'LISP').
        depth: the maximum depth of the logic form.
        coverage: the minimum coverage of the logic form.

    Returns:
        a list of strings, of length C.
    """
    return csearch(inputs, outputs, input_names, type=type, output_format=output_format, depth=depth, coverage=coverage)

