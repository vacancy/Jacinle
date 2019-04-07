#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : jit_example.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/06/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import numpy as np
from jacinle.jit.cython import jit_cython


@jit_cython(force_update=True)
def foo(x: 'long') -> 'long':
    return x + 1


@jit_cython(force_update=True)
def cython_loop(n):
    s: 'long' = 0
    i: 'long'

    for i in range(n):
        s += foo(i)
    return s

def python_loop(n):
    s = 0
    for i in range(n):
        s += i + 1
    return s


@jit_cython(force_update=True, boundscheck=False, wraparound=False)
def cython_np_sum(a: 'np.ndarray[np.float32_t, ndim=1]'):
    i: int = 0
    n: int = a.shape[0]
    s: float = 0

    for i in range(n):
        s += a[i]

    return s


def python_np_sum(a):
    s = 0
    for i in range(a.shape[0]):
        s += a[i]

    return s


def main_timeit():
    import timeit

    print(cython_loop)
    print('Mean time: {:.3f} ms'.format(1000 * timeit.timeit('cython_loop(100000)', number=100, globals=globals())))
    print('Answer:', cython_loop(100000))
    print(python_loop)
    print('Mean time: {:.3f} ms'.format(1000 * timeit.timeit('python_loop(100000)', number=100, globals=globals())))
    print('Answer:', python_loop(100000))

    global arr
    arr = np.random.random(size=100000).astype('float32')
    print(cython_np_sum)
    print('Mean time: {:.3f} ms'.format(1000 * timeit.timeit('cython_np_sum(arr)', number=100, globals=globals())))
    print('Answer:', cython_np_sum(arr))
    print(python_np_sum)
    print('Mean time: {:.3f} ms'.format(1000 * timeit.timeit('python_np_sum(arr)', number=100, globals=globals())))
    print('Answer:', python_np_sum(arr))
    del arr


if __name__ == '__main__':
    main_timeit()

