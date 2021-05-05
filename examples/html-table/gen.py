#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : gen.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/05/2021
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import imageio
from PIL import Image
from jaclearn.visualize.html_table import HTMLTableColumnDesc, HTMLTableVisualizer

a = Image.fromarray(imageio.imread('imageio:chelsea.png'))
b = Image.fromarray(imageio.imread('imageio:astronaut.png'))


vis = HTMLTableVisualizer('generated', 'Visualization')
with vis.html():
    with vis.table('Table Name', [
        HTMLTableColumnDesc(identifier='column1', name='ID', type='code'),
        HTMLTableColumnDesc(identifier='column2', name='Image', type='image'),
        HTMLTableColumnDesc(identifier='column3', name='Frames (Image)', type='frames'),
        HTMLTableColumnDesc(identifier='column4', name='Frames (Text)', type='frames'),
        HTMLTableColumnDesc(identifier='column5', name='Additional Information', type='code')
    ]):
        vis.row(column1=1, column2=a, column3=[
            {'image': a, 'info': 'imageio:chelsea.png'},
            {'image': b, 'info': 'imageio:astronaut.png'},
        ], column4=[
            {'text': '12345', 'info': 'text info 1'},
            {'text': '67890', 'info': 'text info 2'},
        ], column5='Some string')


