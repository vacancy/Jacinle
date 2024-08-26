#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : inspect-file.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/16/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import jacinle
import jacinle.io as io

parser = jacinle.JacArgumentParser()
parser.add_argument('filename', nargs='+')
parser.add_argument('--stprint', action='store_true')
parser.add_argument('--stprint-depth', type=int, default=3)
args = parser.parse_args()


def main():
    for i, filename in enumerate(args.filename):
        globals()[f'f{i + 1}'] = io.load(filename)

    if args.stprint:
        for i in range(len(args.filename)):
            print(f'File {i + 1}:')
            jacinle.stprint(globals()[f'f{i + 1}'], max_depth=args.stprint_depth)
    else:
        for i in range(len(args.filename)):
            print(f'File {i + 1}: {args.filename[i]} loaded as `f{i + 1}`')
        print()
        from IPython import embed; embed()


if __name__ == '__main__':
    main()

