#! /bin/bash
#
# travis.sh
# Copyright (C) 2019 Jiayuan Mao <maojiayuan@gmail.com>
#
# Distributed under terms of the MIT license.
#

cd torch_sampling
CFLAGS=-stdlib=libc++ python setup.py build_ext --inplace
cd ..

