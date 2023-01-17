#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : lmdb-single.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/17/2023
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from jacinle.storage.kv.lmdb import LMDBKVStore


def main():
    kv = LMDBKVStore('/tmp/test_1.lmdb', readonly=False)

    with kv.transaction():
        kv['a'] = 1
        kv['b'] = 2

    assert 'a' in kv and kv['a'] == 1
    assert 'b' in kv and kv['b'] == 2
    assert 'c' not in kv

    for k in kv.keys():
        print(k, kv[k])


if __name__ == '__main__':
    main()

