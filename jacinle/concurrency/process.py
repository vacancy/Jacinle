# -*- coding: utf-8 -*-
# File   : process.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 19/01/2018
#
# This file is part of Jacinle.

import multiprocessing
import os

from jacinle.config.environ import update_env
from jacinle.logging import get_logger
from jacinle.random.rng import gen_seed, reset_global_rng

logger = get_logger(__file__)


class JacProcess(multiprocessing.Process):
    def __init__(self, *args, extra_env=None, seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        if seed is None:
            seed = gen_seed()
        self._extra_env = extra_env
        self._seed = seed

    def run(self):
        if self._extra_env is not None:
            update_env(self._extra_env)
        reset_global_rng(self._seed)
        logger.critical('JacEnvBox pid={} (ppid={}) rng_seed={}.'.format(os.getpid(), os.getppid(), self._seed))
        super().run()

    def __call__(self):
        self.start()
        self.join()
