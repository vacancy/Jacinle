#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cifar.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import os.path as osp
import functools

import pickle
import tarfile
import numpy as np

from jacinle.io.network import download

__all__ = ['load_cifar', 'load_cifar10', 'load_cifar100']


cifar_web_address = 'http://www.cs.toronto.edu/~kriz/'


def _read_cifar(filenames, cls):
    image = []
    label = []
    for fname in filenames:
        with open(fname, 'rb') as f:
            raw_dict = pickle.load(f, encoding='latin1')
        raw_data = raw_dict['data']
        label.extend(raw_dict['labels' if cls == 10 else 'fine_labels'])
        for x in raw_data:
            x = x.reshape(3, 32, 32)
            x = np.transpose(x, [1, 2, 0])
            image.append(x)
    return np.array(image), np.array(label)


def load_cifar(data_dir, nr_classes=10):
    assert nr_classes in (10, 100)

    data_file = 'cifar-{}-python.tar.gz'.format(nr_classes)
    origin = cifar_web_address + data_file
    dataset = osp.join(data_dir, data_file)
    if nr_classes == 10:
        folder_name = 'cifar-10-batches-py'
        filenames = ['data_batch_{}'.format(i) for i in range(1, 6)]
        filenames.append('test_batch')
    else:
        folder_name = 'cifar-100-python'
        filenames = ['train', 'test']

    if not osp.isdir(osp.join(data_dir, folder_name)):
        if not osp.isfile(dataset):
            download(origin, data_dir, data_file)
        tarfile.open(dataset, 'r:gz').extractall(data_dir)

    filenames = list(map(lambda x: osp.join(data_dir, folder_name, x), filenames))

    train_set = _read_cifar(filenames[:-1], nr_classes)
    test_set = _read_cifar([filenames[-1]], nr_classes)

    return train_set, test_set


load_cifar10 = functools.partial(load_cifar, nr_classes=10)
load_cifar100 = functools.partial(load_cifar, nr_classes=100)
