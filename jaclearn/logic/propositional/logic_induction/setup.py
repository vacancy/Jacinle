#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : setup.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/16/2020
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from setuptools import setup, Extension
import numpy as np

# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"

ext_modules = [
    Extension(
        'logic_induction',
        sources=['csrc/logic_induction_impl.cc', 'logic_induction.pyx'],
        include_dirs = [np.get_include(), 'src'],
        language='c++',
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c++14', '-Ofast'],
    )
]

if __name__ == '__main__':
    setup(
        name='logic_induction',
        packages=['logic_induction'],
        version='1.0',
        ext_modules= ext_modules
    )


