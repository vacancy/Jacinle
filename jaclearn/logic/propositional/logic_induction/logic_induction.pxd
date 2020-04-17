#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : logic_induction.pxd
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/16/2020
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "csrc/logic_induction.h":
    cdef enum LogicFormType:
        CONJUNCTION_TYPE, DISJUNCTION_TYPE, GENERAL_TYPE

    cdef enum LogicFormOutputFormat:
        DEFAULT_FORMAT, LISP_FORMAT

    cdef cppclass LogicInductionConfig:
        LogicFormType type
        LogicFormOutputFormat output_format
        size_t nr_examples
        size_t nr_input_variables
        size_t nr_output_variables
        size_t depth
        double coverage

    cdef cppclass LogicInductionContext:
        LogicInductionConfig *config
        unsigned char *inputs
        unsigned char *outputs
        vector[string] input_names

    cdef cppclass LogicInduction:
        LogicInduction(LogicInductionConfig *, LogicInductionContext *) except +
        string search()

