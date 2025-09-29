#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : setup.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/28/2022
#
# This file is part of SceneGraphParser.
# Distributed under terms of the MIT license.

import os
import os.path as osp
from setuptools import setup, find_packages

find_packages()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
requirements="""Cython
pkgconfig
tabulate
ipdb
pyyaml
tqdm
pyzmq
numpy
scipy
scikit-learn
matplotlib""".split()

setup(
    name='jacinle',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html

    description='Personal Python Toolbox',
    long_description=long_description,
    long_description_content_type="text/markdown",

    install_requires=requirements,

    # The project's main homepage.
    url='',

    # Author details
    author='',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=["bin", "docs", "vendors", "scripts", "tests"]),
)

