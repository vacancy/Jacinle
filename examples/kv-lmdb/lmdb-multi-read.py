#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : lmdb-multi-read.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/17/2023
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from jacinle.storage.kv.lmdb import LMDBKVStore


def main():
    kv = LMDBKVStore('/tmp/test_2.lmdb', readonly=True, max_dbs=128)

    assert 'a' in kv and kv['a'] == 1
    assert 'b' in kv and kv['b'] == 2
    assert 'c' not in kv
    assert 'd' not in kv

    assert kv.has('c', db='sub') and kv.get('c', db='sub') == 3
    assert kv.has('d', db='sub') and kv.get('d', db='sub') == 4

    for k in kv.keys():
        print(k, kv[k])

    for k in kv.keys(db='sub'):
        print('sub', k, kv.get(k, db='sub'))


if __name__ == '__main__':
    main()

