# -*- coding: utf-8 -*-
# File   : dataset.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/03/2018
# 
# This file is part of Jacinle.

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
