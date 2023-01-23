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

    ~jactorch.graph.context.ForwardContext
    ~jactorch.graph.nn_env.NNEnv
    ~jactorch.train.env.TrainerEnv

    ~jactorch.graph.context.get_forward_context

.. rubric:: IO

.. autosummary::

    ~jactorch.io.state_dict
    ~jactorch.io.load_state_dict
    ~jactorch.io.load_weights

.. rubric:: Parameter Filtering and Grouping

.. autosummary::

    ~jactorch.graph.parameter.find_parameters
    ~jactorch.graph.parameter.filter_parameters
    ~jactorch.graph.parameter.exclude_parameters
    ~jactorch.graph.parameter.compose_param_groups
    ~jactorch.graph.parameter.param_group
    ~jactorch.graph.parameter.mark_freezed
    ~jactorch.graph.parameter.mark_unfreezed
    ~jactorch.graph.parameter.detach_modules

.. rubric:: Data Structures and Helpful Functions

All of the following functions accepts an arbitrary Python data structure as inputs (e.g., tuples, lists, dictionaries).
They will recursively traverse the data structure and apply the function to each element.

.. autosummary::

    ~jactorch.cuda.copy.async_copy_to
    ~jactorch.utils.meta.as_tensor
    ~jactorch.utils.meta.as_variable
    ~jactorch.utils.meta.as_numpy
    ~jactorch.utils.meta.as_float
    ~jactorch.utils.meta.as_cuda
    ~jactorch.utils.meta.as_cpu
    ~jactorch.utils.meta.as_detached

.. rubric:: Arithmetics

.. autosummary::

    ~jactorch.functional.arith.atanh
    ~jactorch.functional.arith.logit
    ~jactorch.functional.arith.log_sigmoid
    ~jactorch.functional.arith.tstat
    ~jactorch.functional.arith.soft_amax
    ~jactorch.functional.arith.soft_amin

.. rubric:: Clustering

.. autosummary::

    ~jactorch.functional.clustering.kmeans

.. rubric:: Gradient

.. autosummary::

    ~jactorch.functional.grad.grad_multi
    ~jactorch.functional.grad.zero_grad
    ~jactorch.utils.grad.no_grad_func

.. rubric:: Indexing

.. autosummary::

    ~jactorch.functional.indexing.batched_index_select
    ~jactorch.functional.indexing.index_nonzero
    ~jactorch.functional.indexing.index_one_hot
    ~jactorch.functional.indexing.index_one_hot_ellipsis
    ~jactorch.functional.indexing.inverse_permutation
    ~jactorch.functional.indexing.leftmost_nonzero
    ~jactorch.functional.indexing.one_hot
    ~jactorch.functional.indexing.one_hot_dim
    ~jactorch.functional.indexing.one_hot_nd
    ~jactorch.functional.indexing.reversed
    ~jactorch.functional.indexing.rightmost_nonzero
    ~jactorch.functional.indexing.set_index_one_hot_

.. rubric:: Kernel

.. autosummary::

    ~jactorch.functional.kernel.cosine_distance
    ~jactorch.functional.kernel.dot
    ~jactorch.functional.kernel.inverse_distance

.. rubric:: Linear Algebra

.. autosummary::

    ~jactorch.functional.linalg.normalize

.. rubric:: Log-Linear

.. autosummary::

    ~jactorch.functional.loglinear.batch_logmatmulexp
    ~jactorch.functional.loglinear.log1mexp
    ~jactorch.functional.loglinear.logaddexp
    ~jactorch.functional.loglinear.logits_and
    ~jactorch.functional.loglinear.logits_or
    ~jactorch.functional.loglinear.logmatmulexp
    ~jactorch.functional.loglinear.logsumexp

.. rubric:: Masking

.. autosummary::

    ~jactorch.functional.masking.length2mask
    ~jactorch.functional.masking.length_masked_reversed
    ~jactorch.functional.masking.length_masked_softmax
    ~jactorch.functional.masking.mask_meshgrid
    ~jactorch.functional.masking.masked_average
    ~jactorch.functional.masking.masked_softmax

.. rubric:: Ranges

.. autosummary::

    ~jactorch.functional.range.meshgrid
    ~jactorch.functional.range.meshgrid_exclude_self

.. rubric:: Probability

.. autosummary::

    ~jactorch.functional.probability.check_prob_normalization
    ~jactorch.functional.probability.normalize_prob

.. rubric:: Quantization

.. autosummary::

    ~jactorch.functional.quantization.quantize
    ~jactorch.functional.quantization.randomized_quantize

.. rubric:: Sampling

.. autosummary::

    ~jactorch.functional.sampling.sample_bernoulli
    ~jactorch.functional.sampling.sample_multinomial


.. rubric:: Shape

.. autosummary::

    ~jactorch.functional.shape.add_dim
    ~jactorch.functional.shape.add_dim_as_except
    ~jactorch.functional.shape.broadcast
    ~jactorch.functional.shape.broadcast_as_except
    ~jactorch.functional.shape.concat_shape
    ~jactorch.functional.shape.flatten
    ~jactorch.functional.shape.flatten2
    ~jactorch.functional.shape.force_view
    ~jactorch.functional.shape.move_dim
    ~jactorch.functional.shape.repeat
    ~jactorch.functional.shape.repeat_times
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
    logger.exception(
        'Import error is raised during initializing the jactorch package. Please make sure that the torch '
        'package is correctly installed'
    )

from jactorch.utils.init import init_main

init_main()

del init_main
