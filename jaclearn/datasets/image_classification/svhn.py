#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : svhn.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import os

import numpy as np

from jacinle.io.network import download

__all__ = ['load_svhn']


svhn_web_address = {
    'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
              "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
    'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
             "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
    'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
              "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]
}


def load_svhn(data_dir, extra=False):
    from scipy.io import loadmat

    all_set_keys = list(svhn_web_address.keys())
    if not extra:
        all_set_keys = all_set_keys[:2]

    all_sets = []

    for subset in all_set_keys:
        data_addr, data_file, data_hash = svhn_web_address[subset]

        dataset = os.path.join(data_dir, data_file)

        if not os.path.isfile(dataset):
            download(data_addr, data_dir, data_file, md5=data_hash)

        mat = loadmat(dataset)
        mat['X'] = np.transpose(mat['X'], [3, 0, 1, 2])
        all_sets.append((np.ascontiguousarray(mat['X']), mat['y']))

    return tuple(all_sets)
