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


@jit_cython(force_update=True, boundscheck=False, wraparound=False)
def cython_np_sum_2d_v1(a: 'np.ndarray[np.float32_t, ndim=2]'):
    i: int = 0
    n: int = a.shape[0]
    m: int = a.shape[1]
    b = np.zeros(n, dtype=np.float32)

    for i in range(n):
        b[i] = cython_np_sum(a[i])

    return b


@jit_cython(force_update=True, boundscheck=False, wraparound=False)
def cython_np_sum_2d_v1_5(a: 'np.ndarray[np.float32_t, ndim=2]'):
    i: int = 0
    j: int = 0
    n: int = a.shape[0]
    m: int = a.shape[1]
    b: 'np.ndarray[np.float32_t, ndim=1]' = np.zeros(n, dtype=np.float32)
    c: 'np.ndarray[np.float32_t, ndim=1]'
    s: float = 0

    for i in range(n):
        c = a[i]
        for j in range(m):
            b[i] += c[j];

    return b


@jit_cython(force_update=True, boundscheck=False, wraparound=False)
def cython_np_sum_2d_v2(a: 'np.ndarray[np.float32_t, ndim=2]'):
    i: int = 0
    j: int = 0
    n: int = a.shape[0]
    m: int = a.shape[1]
    b: 'np.ndarray[np.float32_t, ndim=1]' = np.zeros(n, dtype=np.float32)
    s: float = 0

    for i in range(n):
        for j in range(m):
            b[i] += a[i, j];

    return b


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

    arr = np.random.random(size=(1000, 1000)).astype('float32')
    print(cython_np_sum_2d_v1)
    print('Mean time: {:.3f} ms'.format(1000 * timeit.timeit('cython_np_sum_2d_v1(arr)', number=100, globals=globals())))
    print(cython_np_sum_2d_v1_5)
    print('Mean time: {:.3f} ms'.format(1000 * timeit.timeit('cython_np_sum_2d_v1_5(arr)', number=100, globals=globals())))
    print(cython_np_sum_2d_v2)
    print('Mean time: {:.3f} ms'.format(1000 * timeit.timeit('cython_np_sum_2d_v2(arr)', number=100, globals=globals())))

    del arr


if __name__ == '__main__':
    main_timeit()

