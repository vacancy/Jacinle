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
args = parser.parse_args()


def main():
    for i, filename in enumerate(args.filename):
        globals()[f'f{i + 1}'] = io.load(filename)

    from IPython import embed; embed()


if __name__ == '__main__':
    main()

