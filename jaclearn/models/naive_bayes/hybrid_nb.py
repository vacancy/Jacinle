#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : hybrid_nb.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/26/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import sklearn.naive_bayes as nb

from jacinle.logging import get_logger
from jacinle.utils.enum import JacEnum

logger = get_logger(__file__)

__all__ = ['NaiveBayesianDistribution', 'HybridNB']


class NaiveBayesianDistribution(JacEnum):
    GAUSSIAN = 'gaussian'
    MULTINOMIAL = 'multinomial'
    BERNOULLI = 'bernoulli'


class HybridNB(object):
    def __init__(self, distributions, weights=None, **kwargs):
        self.models = []
        for dist in distributions:
            dist = NaiveBayesianDistribution.from_string(dist)
            if dist is NaiveBayesianDistribution.GAUSSIAN:
                model = nb.GaussianNB(**kwargs)
            elif dist is NaiveBayesianDistribution.MULTINOMIAL:
                model = nb.MultinomialNB(**kwargs)
            elif dist is NaiveBayesianDistribution.BERNOULLI:
                model = nb.BernoulliNB(**kwargs)
            else:
                raise ValueError('Unknown distribution: {}.'.format(dist))
            kwargs['fit_prior'] = False  # Except the first model.
            self.models.append(model)

        self.weights = weights

    def fit(self, xs, y, verbose=True):
        assert len(xs) == len(self.models)
        for x, model in zip(xs, self.models):
            if verbose:
                logger.info('Fitting model: {}.'.format(repr(model)))
            model.fit(x, y)

    def predict(self, xs, verbose=True):
        if self.weights is not None:
            raise NotImplementedError('HybridNB.weights is not supported.')

        log_prob = 0
        for x, model in zip(xs, self.models):
            if verbose:
                logger.info('Predicting using model: {}.'.format(repr(model)))
            log_prob += model.predict_log_proba(x)
        return log_prob.argmax(axis=1)

