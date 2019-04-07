#! /bin/bash
# File   : travis.sh
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

python3 setup.py build_ext --inplace
rm -rf build

