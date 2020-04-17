#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : gym_recorder.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/16/2020
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import os.path as osp
import mujoco_py
from copy import deepcopy

import jacinle.io as io
import jaclearn.math.rotation as R
from jacinle.utils.enum import JacEnum


class MujocoObjectType(JacEnum):
    BODY = 'body'
    GEOM = 'geom'
    SITE = 'site'


class MujocoGymRecorder(object):
    def __init__(self, gym_environ):
        self.gym_environ = gym_environ
        self.sim = gym_environ.sim
        self.gym_states = list()

        self.object_names = dict()
        self.object_poses = list()

        self.reset()

    def _init_object_names(self):
        for obj_type in MujocoObjectType.choice_objs():
            all_names = getattr(self.sim.model, obj_type.value + '_names')
            self.object_names[obj_type.value] = deepcopy(all_names)

    def reset(self):
        self._init_object_names()
        self.gym_states = list()
        self.object_poses = list()
        self.step()

    def step(self):
        self.gym_states.append(self.sim.get_state())
        new_object_poses = dict()
        for obj_type, names in self.object_names.items():
            new_object_poses[obj_type] = this_type = {n: dict() for n in names}
            for value_name in ['xpos', 'xquat', 'xvelp', 'xvelr']:
                if obj_type in ('geom', 'site') and value_name == 'xquat':
                    for i, (name, value) in enumerate(zip(names, getattr(self.sim.data, obj_type + '_' + 'xmat'))):
                        this_type[name][value_name] = R.mat2quat(value.reshape(3, 3)).tolist()
                else:
                    for i, (name, value) in enumerate(zip(names, getattr(self.sim.data, obj_type + '_' + value_name))):
                        this_type[name][value_name] = value.tolist()
        self.object_poses.append(new_object_poses)

    def dump(self, save_dir):
        io.mkdir(save_dir)
        io.dump(osp.join(save_dir, 'mj_states.pkl'), self.gym_states)
        io.dump(osp.join(save_dir, 'objects.json'), dict(
            names=self.object_names,
            poses=self.object_poses
        ))

    def hook(self):
        old_reset = self.gym_environ.reset
        old_step = self.gym_environ.step
        def reset(*args, **kwargs):
            retval = old_reset(*args, **kwargs)
            self.reset()
            return retval

        def step(*args, **kwargs):
            retval = old_step(*args, **kwargs)
            self.step()
            return retval

        self.gym_environ.reset = reset
        self.gym_environ.step = step
        return self

