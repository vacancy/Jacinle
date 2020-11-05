#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dataset.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/08/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import random
import itertools

from jacinle.logging import get_logger

logger = get_logger(__file__)

__all__ = ['IterableDatasetMixin', 'ProxyDataset', 'ListDataset', 'FilterableDatasetUnwrapped', 'FilterableDatasetView']


class Dataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        """
        Get an item from the given index.

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        raise NotImplementedError

    def __len__(self):
        """
        Returns the number of bytes.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError

    def __add__(self, other):
        """
        Add another dataset objects.

        Args:
            self: (todo): write your description
            other: (todo): write your description
        """
        from torch.utils.data.dataset import ConcatDataset
        return ConcatDataset([self, other])


class IterableDatasetMixin(object):
    def __iter__(self):
        """
        Iterate over the iterator items.

        Args:
            self: (todo): write your description
        """
        for i in range(len(self)):
            yield i, self[i]


class ProxyDataset(Dataset):
    """
    A proxy dataset base class for wrapping a base dataset.
    """
    def __init__(self, base_dataset):
        """

        Args:
            base_dataset (Dataset): the base dataset.

        """
        self._base_dataset = base_dataset

    @property
    def base_dataset(self):
        """
        Return the base_dataset.

        Args:
            self: (todo): write your description
        """
        return self._base_dataset

    def __getitem__(self, item):
        """
        Return the item from the dataset.

        Args:
            self: (todo): write your description
            item: (str): write your description
        """
        return self.base_dataset[item]

    def __len__(self):
        """
        Returns the length of the dataset.

        Args:
            self: (todo): write your description
        """
        return len(self.base_dataset)


class ListDataset(Dataset):
    """
    Wraps a list into a pytorch Dataset.
    """
    def __init__(self, list):
        """

        Args:
            list (list[Any]): the list of data.

        """
        self.list = list

    def __getitem__(self, item):
        """
        Return the value of item

        Args:
            self: (todo): write your description
            item: (str): write your description
        """
        return self.list[item]

    def __len__(self):
        """
        Returns the length of the list.

        Args:
            self: (todo): write your description
        """
        return len(self.list)


class FilterableDatasetUnwrapped(Dataset, IterableDatasetMixin):
    """
    A filterable dataset. User can call various `filter_*` operations to obtain a subset of the dataset.
    """
    def __init__(self):
        """
        Initialize metainfo

        Args:
            self: (todo): write your description
        """
        super().__init__()
        self.metainfo_cache = dict()

    def get_metainfo(self, index):
        """
        Get the metainfo for the given index.

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        if index not in self.metainfo_cache:
            self.metainfo_cache[index] = self._get_metainfo(index)
        return self.metainfo_cache[index]

    def _get_metainfo(self, index):
        """
        Returns the metainfo at index.

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        raise NotImplementedError()


class FilterableDatasetView(FilterableDatasetUnwrapped):
    def __init__(self, owner_dataset, indices=None, filter_name=None, filter_func=None):
        """
        Args:
            owner_dataset (Dataset): the original dataset.
            indices (List[int]): a list of indices that was filterred out.
            filter_name (str): human-friendly name for the filter.
            filter_func (Callable): just for tracking.
        """

        super().__init__()
        self.owner_dataset = owner_dataset
        self.indices = indices
        self._filter_name = filter_name
        self._filter_func = filter_func

    @property
    def unwrapped(self):
        """
        Unwraps_dataset.

        Args:
            self: (todo): write your description
        """
        if self.indices is not None:
            return self.owner_dataset.unwrapped
        return self.owner_dataset

    @property
    def filter_name(self):
        """
        Returns the filter name.

        Args:
            self: (todo): write your description
        """
        return self._filter_name if self._filter_name is not None else '<anonymous>'

    @property
    def full_filter_name(self):
        """
        Return the name of the dataset.

        Args:
            self: (todo): write your description
        """
        if self.indices is not None:
            return self.owner_dataset.full_filter_name + '/' + self.filter_name
        return '<original>'

    @property
    def filter_func(self):
        """
        Returns a new filter function.

        Args:
            self: (todo): write your description
        """
        return self._filter_func

    def collect(self, key_func):
        """
        Return a dictionary of the metric values

        Args:
            self: (todo): write your description
            key_func: (todo): write your description
        """
        return {key_func(self.get_metainfo(i)) for i in range(len(self))}

    def filter(self, filter_func, filter_name=None):
        """
        Return a filter by filter_func.

        Args:
            self: (todo): write your description
            filter_func: (todo): write your description
            filter_name: (str): write your description
        """
        indices = []
        for i in range(len(self)):
            metainfo = self.get_metainfo(i)
            if filter_func(metainfo):
                indices.append(i)
        if len(indices) == 0:
            raise ValueError('Filter results in an empty dataset.')
        return type(self)(self, indices, filter_name, filter_func)

    def random_trim_length(self, length):
        """
        Returns a random length of length length.

        Args:
            self: (todo): write your description
            length: (int): write your description
        """
        assert length < len(self)
        logger.info('Randomly trim the dataset: #samples = {}.'.format(length))
        indices = list(random.choice(len(self), size=length, replace=False))
        return type(self)(self, indices=indices, filter_name='randomtrim[{}]'.format(length))

    def trim_length(self, length):
        """
        Trim the length of the array.

        Args:
            self: (todo): write your description
            length: (int): write your description
        """
        if type(length) is float and 0 < length <= 1:
            length = int(len(self) * length)
        assert length < len(self)
        logger.info('Trim the dataset: #samples = {}.'.format(length))
        return type(self)(self, indices=list(range(0, length)), filter_name='trim[{}]'.format(length))

    def trim_range(self, begin, end=None):
        """
        Trim the range between start and end.

        Args:
            self: (todo): write your description
            begin: (float): write your description
            end: (todo): write your description
        """
        if end is None:
            end = len(self)
        assert end <= len(self)
        logger.info('Trim the dataset: #samples = {}.'.format(end - begin))
        return type(self)(self, indices=list(range(begin, end)), filter_name='trimrange[{}:{}]'.format(begin, end))

    def split_trainval(self, split):
        """
        Splits the dataset into multiple indices.

        Args:
            self: (todo): write your description
            split: (todo): write your description
        """
        if isinstance(split, float) and 0 < split < 1:
            split = int(len(self) * split)
        split = int(split)

        assert 0 < split < len(self)
        nr_train = split
        nr_val = len(self) - nr_train
        logger.info('Split the dataset: #training samples = {}, #validation samples = {}.'.format(nr_train, nr_val))
        return (
                type(self)(self, indices=list(range(0, split)), filter_name='train'),
                type(self)(self, indices=list(range(split, len(self))), filter_name='val')
        )

    def split_kfold(self, k):
        """
        Generate k into k into k into k into k into k into k - sized indexes.

        Args:
            self: (todo): write your description
            k: (todo): write your description
        """
        assert len(self) % k == 0
        block = len(self) // k

        for i in range(k):
            yield (
                    type(self)(self, indices=list(range(0, i * block)) + list(range((i + 1) * block, len(self))), filter_name='fold{}[train]'.format(i + 1)),
                    type(self)(self, indices=list(range(i * block, (i + 1) * block)), filter_name='fold{}[val]'.format(i + 1))
            )

    def repeat(self, nr_repeats):
        """
        Return a new list of the elements.

        Args:
            self: (todo): write your description
            nr_repeats: (todo): write your description
        """
        indices = list(itertools.chain(*[range(len(self)) for _ in range(nr_repeats)]))
        return type(self)(self, indices=indices, filter_name='repeat[{}]'.format(nr_repeats))

    def __getitem__(self, index):
        """
        Return the index for the given index.

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        if self.indices is None:
            return self.owner_dataset[index]
        return self.owner_dataset[self.indices[index]]

    def __len__(self):
        """
        Returns the number of the indices.

        Args:
            self: (todo): write your description
        """
        if self.indices is None:
            return len(self.owner_dataset)
        return len(self.indices)

    def get_metainfo(self, index):
        """
        Return the metric for the given index.

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        if self.indices is None:
            return self.owner_dataset.get_metainfo(index)
        return self.owner_dataset.get_metainfo(self.indices[index])

