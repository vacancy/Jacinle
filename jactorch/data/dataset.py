#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dataset.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/08/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from torch.utils.data.dataset import Dataset

__all__ = ['ProxyDataset', 'ListDataset']


class ProxyDataset(Dataset):
    def __init__(self, base_dataset):
        self._base_dataset = base_dataset

    @property
    def base_dataset(self):
        return self._base_dataset

    def __getitem__(self, item):
        return self.base_dataset[item]

    def __len__(self):
        return len(self.base_dataset)


class ListDataset(Dataset):
    def __init__(self, list):
        self.list = list

    def __getitem__(self, item):
        return self.list[item]

    def __len__(self):
        return len(self.list)
