#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : example.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/22/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import os.path as osp
from jaclearn.nlp.graph.dependency_visualizer import visualize_simple_svg


def main():
    svg = visualize_simple_svg('a b c', [(0, 1, 'tag1'), (1, 2)])

    from jaclearn.nlp.graph.dependency_visualizer.templates import TPL_PAGE
    with open('./index.html', 'w') as f:
        f.write(TPL_PAGE.format(lang='en', dir='ltr', content=svg))
    print(osp.realpath('./index.html'))


if __name__ == '__main__':
    main()
