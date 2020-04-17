#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : logic_induction.pyx
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/16/2020
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

# distutils: language = c++

cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from logic_induction cimport LogicInductionConfig
from jacinle.utils.enum import JacEnum


class LogicFormTypePy(JacEnum):
    CONJUNCTION = 1
    DISJUNCTION = 2
    GENERAL = 3


class LogicFormOutputFormatPy(JacEnum):
    DEFAULT = 1
    LISP = 2


def search(
    np.ndarray[np.uint8_t, ndim=2, mode="c"] inputs,
    np.ndarray[np.uint8_t, ndim=2, mode="c"] outputs,
    input_names,
    type='GENERAL',
    output_format='DEFAULT',
    size_t depth=4,
    double coverage=0.99
):
    assert inputs.shape[0] == outputs.shape[0]
    assert outputs.shape[1] == 1

    config = new LogicInductionConfig()
    config.type = LogicFormTypePy.from_string(type).value
    config.output_format = LogicFormOutputFormatPy.from_string(output_format).value
    config.nr_examples = inputs.shape[0]
    config.nr_input_variables = inputs.shape[1]
    config.nr_output_variables = outputs.shape[1]
    config.depth = depth
    config.coverage = coverage

    context = new LogicInductionContext()
    context.config = config
    context.inputs = &inputs[0, 0]
    context.outputs = &outputs[0, 0]
    context.input_names = [s.encode('utf8') for s in input_names]

    induction = new LogicInduction(config, context)

    try:
        return induction.search().decode('utf8')
    finally:
        del config
        del context
        del induction
