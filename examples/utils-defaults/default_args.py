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
    """
    Print the inner inner distance between two arguments.

    Args:
        a: (array): write your description
        b: (array): write your description
        c: (array): write your description
        d: (array): write your description
    """
    print(a, b, c, d, args)


def outer(b, *args, c, d=ARGDEF):
    """
    R returns the outer outer interval )

    Args:
        b: (todo): write your description
        c: (todo): write your description
        d: (todo): write your description
        ARGDEF: (todo): write your description
    """
    inner('called by outer', b, *args, c=c, d=d)


if __name__ == '__main__':
    from IPython import embed; embed()
    outer('value_b', c='value_c')

