#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : uid.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/21/2020
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import time
import uuid

__all__ = ['gen_time_string', 'gen_uuid4']


def gen_time_string():
    return time.strftime('%Y-%m-%d-%H-%M-%S')


def gen_uuid4():
    return uuid.uuid4().hex

