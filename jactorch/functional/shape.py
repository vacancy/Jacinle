# -*- coding: utf-8 -*-
# File   : shape.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 25/01/2018
# 
# This file is part of Jacinle.


def broadcast(tensor, dim, size):
    assert tensor.size(dim) == 1
    shape = tuple(tensor.size())
    return tensor.expand(shape[:dim] + (size, ) + shape[dim+1:])
