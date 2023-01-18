#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : rng.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import os
import contextlib
import random as sys_random

import numpy as np
import numpy.random as npr

from typing import Optional
from jacinle.utils.defaults import defaults_manager
from jacinle.utils.registry import Registry

__all__ = ['JacRandomState', 'get_default_rng', 'gen_seed', 'gen_rng', 'reset_global_seed', 'with_seed', 'seed']


class JacRandomState(npr.RandomState):
    def choice_list(self, list_, size=1, replace=False, p=None):
        """Efficiently draw an element from an list, if the rng is given, use it instead of the system one."""
        if size == 1:
            if type(list_) in (list, tuple):
                return list_[self.choice(len(list_), p=p)]
            return self.choice(list_, p=p)
        else:
            if type(list_) in (list, tuple):
                inds = self.choice(len(list_), size=size, replace=replace, p=p)
                return [list_[i] for i in inds]
            return self.choice(list_, size=size, replace=replace, p=p)

    def shuffle_list(self, list_):
        if type(list_) is list:
            sys_random.shuffle(list_, random=self.random_sample)
        else:
            self.shuffle(list_)

    def shuffle_multi(self, *arrs):
        length = len(arrs[0])
        for a in arrs:
            assert len(a) == length, 'non-compatible length when shuffling multiple arrays'

        inds = np.arange(length)
        self.shuffle(inds)
        return tuple(map(lambda x: x[inds], arrs))

    @defaults_manager.wrap_custom_as_default(is_local=True)
    def as_default(self):
        yield self


_rng = JacRandomState()


get_default_rng = defaults_manager.gen_get_default(JacRandomState, default_getter=lambda: _rng)


def gen_seed() -> int:
    """Generate a random seed (a.k.a. a random integer in [0, 2^32))."""
    return get_default_rng().randint(4294967296)


def gen_rng(seed: Optional[int] = None) -> JacRandomState:
    """Generate a random number generator with the given seed."""
    return JacRandomState(seed)


global_rng_registry = Registry()
global_rng_registry.register('jacinle', lambda: _rng.seed)
global_rng_registry.register('numpy', lambda: npr.seed)
global_rng_registry.register('sys', lambda: sys_random.seed)

global_rng_state_registry = Registry()
global_rng_state_registry.register('jacinle', lambda: (_rng.get_state, _rng.set_state))
global_rng_state_registry.register('numpy', lambda: (npr.get_state, npr.set_state))
global_rng_state_registry.register('sys', lambda: (sys_random.getstate, sys_random.setstate))


def reset_global_seed(seed: Optional[int] = None, verbose: bool = False) -> int:
    """Reset the global seed for all random number generators.

    Args:
        seed: the seed to use. If None, a random seed will be generated.
        verbose: whether to print the seed.

    Returns:
        the seed used.
    """
    if seed is None:
        seed = gen_seed()
    for k, seed_getter in global_rng_registry.items():
        if verbose:
            from jacinle.logging import get_logger
            logger = get_logger(__file__)
            logger.critical('Reset random seed for: {} (pid={}, seed={}).'.format(k, os.getpid(), seed))
        seed_getter()(seed)
    return seed


def seed(seed):
    reset_global_seed(seed)


@contextlib.contextmanager
def with_seed(seed: Optional[int] = None, verbose: bool = False):
    """A context manager that sets the global seed to the given value, and restores it after the context.
    Note that when the given seed is None, this function will not do anything.

    Example::

        with with_seed(123):
            # do something

    Args:
        seed: the seed to set.
    """
    if seed is None:
        yield
        return

    states = dict()
    setter_functions = dict()

    for k in global_rng_state_registry.keys():
        if global_rng_registry.has(k) and global_rng_state_registry.has(k):
            seed_func = global_rng_registry.lookup(k)()
            state_getter, state_setter = global_rng_state_registry.lookup(k)()

            states[k] = state_getter()
            setter_functions[k] = state_setter
            seed_func(seed)

            if verbose:
                from jacinle.logging import get_logger
                logger = get_logger(__file__)
                logger.critical('Reset random seed for: {} (pid={}, seed={}).'.format(k, os.getpid(), seed))

    yield

    for k, state in states.items():
        setter_functions[k](state)


def _initialize_global_seed():
    seed = os.environ.get('JACINLE_GLOBAL_SEED', None)
    if seed is not None:
        seed = int(seed)
        reset_global_seed(seed, verbose=True)


_initialize_global_seed()

