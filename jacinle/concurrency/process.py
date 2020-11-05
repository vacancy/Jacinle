#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : process.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import multiprocessing
import os

from jacinle.config.environ import update_env
from jacinle.logging import get_logger
from jacinle.random.rng import gen_seed, reset_global_seed

logger = get_logger(__file__)

__all__ = ['JacProcess']


class JacProcess(multiprocessing.Process):
    def __init__(self, *args, extra_env=None, seed=None, **kwargs):
        """
        Initialize the environment.

        Args:
            self: (todo): write your description
            extra_env: (str): write your description
            seed: (int): write your description
        """
        super().__init__(*args, **kwargs)
        if seed is None:
            seed = gen_seed()
        self._extra_env = extra_env
        self._seed = seed

    def run(self):
        """
        Runs the environment.

        Args:
            self: (todo): write your description
        """
        if self._extra_env is not None:
            update_env(self._extra_env)
        reset_global_seed(self._seed)
        logger.critical('JacEnvBox pid={} (ppid={}) rng_seed={}.'.format(os.getpid(), os.getppid(), self._seed))
        super().run()

    def __call__(self):
        """
        Call the call.

        Args:
            self: (todo): write your description
        """
        self.start()
        self.join()
