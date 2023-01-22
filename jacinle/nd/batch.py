#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : batch.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/18/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import collections
from typing import Any, Sequence, List
import numpy as np

__all__ = ['batchify', 'unbatchify']


def batchify(inputs: Sequence[Any]) -> Any:
    """Recursively combine a list of inputs into a batch. This function handles tuples, lists, dicts, and numpy arrays.

    Examples:
        >>> batchify([np.array([1, 2, 3]), np.array([4, 5, 6])])
        array([[1, 2, 3],
                [4, 5, 6]])
        >>> batchify([
        ...     {'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6])},
        ...     {'a': np.array([7, 8, 9]), 'b': np.array([10, 11, 12])},
        ... ])
        {'a': array([[1, 2, 3],
                    [7, 8, 9]]),
        'b': array([[ 4,  5,  6],
                    [10, 11, 12]])}

    Args:
        inputs: a list of inputs to be batched.

    Returns:
        a batched input.
    """
    first = inputs[0]
    if isinstance(first, (tuple, list, collections.UserList)):
        return [batchify([ele[i] for ele in inputs]) for i in range(len(first))]
    elif isinstance(first, (collections.Mapping, collections.UserDict)):
        return {k: batchify([ele[k] for ele in inputs]) for k in first}
    return np.stack(inputs)


def unbatchify(inputs: Any) -> List[Any]:
    """Recursively split a batch into a list of inputs. This function handles tuples, lists, dicts, and numpy arrays.
    This function is the inverse of :func:`batchify`.

    Example:
        >>> unbatchify(np.array([[1, 2, 3], [4, 5, 6]]))
        [array([1, 2, 3]), array([4, 5, 6])]
        >>> unbatchify({'a': np.array([[1, 2, 3], [4, 5, 6]]), 'b': np.array([[7, 8, 9], [10, 11, 12]])})
        [{'a': array([1, 2, 3]), 'b': array([7, 8, 9])}, {'a': array([4, 5, 6]), 'b': array([10, 11, 12])}]

    Args:
        inputs: a batched input.

    Returns:
        a list of inputs.
    """
    if isinstance(inputs, (tuple, list, collections.UserList)):
        outputs = [unbatchify(e) for e in inputs]
        return list(map(list, zip(*outputs)))
    elif isinstance(inputs, (collections.Mapping, collections.UserDict)):
        outputs = {k: unbatchify(v) for k, v in inputs.items()}
        first = outputs[0]
        return [{k: outputs[k][i] for k in inputs} for i in range(len(first))]
    return list(inputs)
