#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : _lazy_bind.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import functools

from .rng import get_default_rng, _rng

__all__ = [
    # Utility functions
    'random_sample',        # Uniformly distributed floats over ``[0, 1)``.
    'bytes',                # Uniformly distributed random bytes.
    'random_integers',      # Uniformly distributed integers in a given range.
    'permutation',          # Randomly permute a sequence / generate a random sequence.
    'shuffle',              # Randomly permute a sequence in place.
    'seed',                 # Seed the random number generator.
    'choice',               # Random sample from 1-D array.

    # Compatibility functions
    'rand',                 # Uniformly distributed values.
    'randn',                # Normally distributed values.
    'randint',              # Uniformly distributed integers in a given range.

    # Univariate distributions
    'beta',                 # Beta distribution over ``[0, 1]``.
    'binomial',             # Binomial distribution.
    'chisquare',            # :math:`\\chi^2` distribution.
    'exponential',          # Exponential distribution.
    'f',                    # F (Fisher-Snedecor) distribution.
    'gamma',                # Gamma distribution.
    'geometric',            # Geometric distribution.
    'gumbel',               # Gumbel distribution.
    'hypergeometric',       # Hypergeometric distribution.
    'laplace',              # Laplace distribution.
    'logistic',             # Logistic distribution.
    'lognormal',            # Log-normal distribution.
    'logseries',            # Logarithmic series distribution.
    'negative_binomial',    # Negative binomial distribution.
    'noncentral_chisquare', # Non-central chi-square distribution.
    'noncentral_f',         # Non-central F distribution.
    'normal',               # Normal / Gaussian distribution.
    'pareto',               # Pareto distribution.
    'poisson',              # Poisson distribution.
    'power',                # Power distribution.
    'rayleigh',             # Rayleigh distribution.
    'triangular',           # Triangular distribution.
    'uniform',              # Uniform distribution.
    'vonmises',             # Von Mises circular distribution.
    'wald',                 # Wald (inverse Gaussian) distribution.
    'weibull',              # Weibull distribution.
    'zipf',                 # Zipf's distribution over ranked data.

    # Multivariate distributions
    'dirichlet',            # Multivariate generalization of Beta distribution.
    'multinomial',          # Multivariate generalization of the binomial distribution.
    'multivariate_normal',  # Multivariate generalization of the normal distribution.

    # Standard distributions
    'standard_cauchy',      # Standard Cauchy-Lorentz distribution.
    'standard_exponential', # Standard exponential distribution.
    'standard_gamma',       # Standard Gamma distribution.
    'standard_normal',      # Standard normal distribution.
    'standard_t',           # Standard Student's t-distribution.

    # Internal functions
    'get_state',            # Get tuple representing internal state of generator.
    'set_state',            # Set state of generator.

    # Customized utility functions
    'choice_list',          # Choose an element from a python list.
    'shuffle_list',         # Shuffle an python list.
    'shuffle_multi',        # Shuffle multiple arrays simultaneously.
]


def _lazy_bind(instance_func):
    func_name = instance_func.__name__

    @functools.wraps(instance_func)
    def wrapped(*args, **kwargs):
        func = getattr(get_default_rng(), func_name)
        return func(*args, **kwargs)

    return wrapped


# Utility functions
random_sample = _lazy_bind(_rng.random_sample)
bytes = _lazy_bind(_rng.bytes)
random_integers = _lazy_bind(_rng.random_integers)
permutation = _lazy_bind(_rng.permutation)
shuffle = _lazy_bind(_rng.shuffle)
seed = _lazy_bind(_rng.seed)
choice = _lazy_bind(_rng.choice)
# Compatibility functions
rand = _lazy_bind(_rng.rand)
randn = _lazy_bind(_rng.randn)
randint = _lazy_bind(_rng.randint)
# Univariate distributions
beta = _lazy_bind(_rng.beta)
binomial = _lazy_bind(_rng.binomial)
chisquare = _lazy_bind(_rng.chisquare)
exponential = _lazy_bind(_rng.exponential)
f = _lazy_bind(_rng.f)
gamma = _lazy_bind(_rng.gamma)
geometric = _lazy_bind(_rng.geometric)
gumbel = _lazy_bind(_rng.gumbel)
hypergeometric = _lazy_bind(_rng.hypergeometric)
laplace = _lazy_bind(_rng.laplace)
logistic = _lazy_bind(_rng.logistic)
lognormal = _lazy_bind(_rng.lognormal)
logseries = _lazy_bind(_rng.logseries)
negative_binomial = _lazy_bind(_rng.negative_binomial)
noncentral_chisquare = _lazy_bind(_rng.noncentral_chisquare)
noncentral_f = _lazy_bind(_rng.noncentral_f)
normal = _lazy_bind(_rng.normal)
pareto = _lazy_bind(_rng.pareto)
poisson = _lazy_bind(_rng.poisson)
power = _lazy_bind(_rng.power)
rayleigh = _lazy_bind(_rng.rayleigh)
triangular = _lazy_bind(_rng.triangular)
uniform = _lazy_bind(_rng.uniform)
vonmises = _lazy_bind(_rng.vonmises)
wald = _lazy_bind(_rng.wald)
weibull = _lazy_bind(_rng.weibull)
zipf = _lazy_bind(_rng.zipf)
# Multivariate distributions
dirichlet = _lazy_bind(_rng.dirichlet)
multinomial = _lazy_bind(_rng.multinomial)
multivariate_normal = _lazy_bind(_rng.multivariate_normal)
# Standard distributions
standard_cauchy = _lazy_bind(_rng.standard_cauchy)
standard_exponential = _lazy_bind(_rng.standard_exponential)
standard_gamma = _lazy_bind(_rng.standard_gamma)
standard_normal = _lazy_bind(_rng.standard_normal)
standard_t = _lazy_bind(_rng.standard_t)
# Internal functions
get_state = _lazy_bind(_rng.get_state)
set_state = _lazy_bind(_rng.set_state)
# Customized utility functions
choice_list = _lazy_bind(_rng.choice_list)
shuffle_list = _lazy_bind(_rng.shuffle_list)
shuffle_multi = _lazy_bind(_rng.shuffle_multi)

# Some aliases:
ranf = random = sample = random_sample
__all__.extend(['ranf', 'random', 'sample'])
