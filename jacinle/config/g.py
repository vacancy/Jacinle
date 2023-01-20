#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : g.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/12/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""A simple global dict-like objects.

Example:

    .. code-block:: python

        from jacinle.config.g import g

        g.configfile = 'config.yaml'
        g.project_name = 'Jacinle'
        g.project_version = 1

.. rubric:: Variables

.. autosummary::
    :toctree:

    g

"""

from jacinle.utils.container import g

__all__ = ['g']

