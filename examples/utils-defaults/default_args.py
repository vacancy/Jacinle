#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : default_args.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/18/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from jacinle.utils.defaults import ARGDEF, default_args


@default_args
def inner(a, b, *args, c='default_c', d='default_d'):
    print(a, b, c, d, args)


def outer(b, *args, c, d=ARGDEF):
    inner('called by outer', b, *args, c=c, d=d)


if __name__ == '__main__':
    from IPython import embed; embed()
    outer('value_b', c='value_c')

