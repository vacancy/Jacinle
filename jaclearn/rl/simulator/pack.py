#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pack.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/23/2017
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from jacinle.utils.imp import module_vars_as_dict


class Pack(object):
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.steps = []
        self.is_ended = False

        self.__last_observation = None

    def reset(self, observation):
        self.__last_observation = observation

    def step(self, action, observation, reward, done, info=None):
        assert not self.is_ended

        last_observation = self.__last_observation

        if done:
            self.is_ended = True
            self.__last_observation = None
        else:
            self.__last_observation = observation

        record = dict(
            action=action,
            observation=last_observation,
            reward=reward,
            info=info)
        self.steps.append(record)

        if done:
            record = dict(action=None, observation=observation, reward=None, info=None)
            self.steps.append(record)

    def make_pickleable(self):
        return dict(
            cfg=module_vars_as_dict(self.cfg) if self.cfg is not None else None,
            steps=self.steps,
            is_ended=self.is_ended
        )
