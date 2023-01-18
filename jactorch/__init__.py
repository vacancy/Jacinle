#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/24/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""Jacinle PyTorch functions and modules.

.. rubric:: Contexts

.. autosummary::

    ForwardContext
    NNEnv
    TrainerEnv

    get_forward_context

.. rubric:: IO

.. autosummary::

    state_dict
    load_state_dict
    load_weights

.. rubric:: Parameter Filtering and Grouping

.. autosummary::

    find_parameters
    filter_parameters
    exclude_parameters
    compose_param_groups
    param_group
    mark_freezed
    mark_unfreezed
    detach_modules

.. rubric:: Data Structures and Helpful Functions

All of the following functions accepts an arbitrary Python data structure as inputs (e.g., tuples, lists, dictionaries).
They will recursively traverse the data structure and apply the function to each element.

.. autosummary::

    async_copy_to
    as_tensor
    as_variable
    as_numpy
    as_float
    as_cuda
    as_cpu
    as_detached

.. rubric:: Arithmetics

.. autosummary::

    atanh
    logit
    log_sigmoid
    tstat
    soft_amax
    soft_amin

.. rubric:: Clustering

.. autosummary::

    kmeans

.. rubric:: Gradient

.. autosummary::

    grad_multi
    zero_grad
    no_grad_func

.. rubric:: Indexing

.. autosummary::

    batched_index_select
    index_nonzero
    index_one_hot
    index_one_hot_ellipsis
    inverse_permutation
    leftmost_nonzero
    one_hot
    one_hot_dim
    one_hot_nd
    reversed
    rightmost_nonzero
    set_index_one_hot_

.. rubric:: Kernel

.. autosummary::

    cosine_distance
    dot
    inverse_distance

.. rubric:: Linear Algebra

.. autosummary::

    normalize

.. rubric:: Log-Linear

.. autosummary::

    batch_logmatmulexp
    log1mexp
    logaddexp
    logits_and
    logits_or
    logmatmulexp
    logsumexp

.. rubric:: Masking

.. autosummary::

    length2mask
    length_masked_reversed
    length_masked_softmax
    mask_meshgrid
    masked_average
    masked_softmax

.. rubric:: Ranges

.. autosummary::

    meshgrid
    meshgrid_exclude_self

.. rubric:: Probability

.. autosummary::

    check_prob_normalization
    normalize_prob

.. rubric:: Quantization

.. autosummary::

    quantize
    randomized_quantize

.. rubric:: Sampling

.. autosummary::

    sample_bernoulli
    sample_multinomial


.. rubric:: Shape

.. autosummary::

    add_dim
    add_dim_as_except
    broadcast
    broadcast_as_except
    concat_shape
    flatten
    flatten2
    force_view
    move_dim
    repeat
    repeat_times

.. rubric:: Modules
"""

try:
    from .cuda.copy import async_copy_to
    from .functional.arith import  atanh, logit, log_sigmoid, tstat, soft_amax, soft_amin
    from .functional.clustering import kmeans
    from .functional.grad import grad_multi, zero_grad
    from .functional.indexing import batched_index_select, index_nonzero, index_one_hot, index_one_hot_ellipsis, inverse_permutation, leftmost_nonzero, one_hot, one_hot_dim, one_hot_nd, reversed, rightmost_nonzero, set_index_one_hot_
    from .functional.kernel import cosine_distance, dot, inverse_distance
    from .functional.linalg import normalize
    from .functional.loglinear import batch_logmatmulexp, log1mexp, logaddexp, logits_and, logits_or, logmatmulexp, logsumexp
    from .functional.masking import length2mask, length_masked_reversed, length_masked_softmax, mask_meshgrid, masked_average, masked_softmax
    from .functional.range import meshgrid, meshgrid_exclude_self
    from .functional.probability import check_prob_normalization, normalize_prob
    from .functional.quantization import quantize, randomized_quantize
    from .functional.sampling import sample_bernoulli, sample_multinomial
    from .functional.shape import add_dim, add_dim_as_except, broadcast, broadcast_as_except, concat_shape, flatten, flatten2, force_view, move_dim, repeat, repeat_times

    from .graph.context import ForwardContext, get_forward_context
    from .graph.nn_env import NNEnv
    from .graph.parameter import find_parameters, filter_parameters, exclude_parameters, compose_param_groups, param_group, mark_freezed, mark_unfreezed, detach_modules

    from .train.env import TrainerEnv

    from .io import state_dict, load_state_dict, load_weights

    from .utils.meta import as_tensor, as_variable, as_numpy, as_float, as_cuda, as_cpu, as_detached
    from .utils.grad import no_grad_func

    from . import data
    from . import nn
    from . import models
    from . import optim
    from . import parallel
    from . import train

except ImportError:
    from jacinle.logging import get_logger
    logger = get_logger(__file__)
    logger.exception('Import error is raised during initializing the jactorch package. Please make sure that the torch '
                     'package is correctly installed')

from jactorch.utils.init import init_main

init_main()

del init_main
