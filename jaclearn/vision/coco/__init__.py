#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/20/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.


from jacinle.jit.cext import auto_travis
auto_travis(__file__)

from .pycocotools.coco import COCO
from .pycocotools.cocoeval import COCOeval
