#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : environ.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import contextlib

from copy import deepcopy

from jacinle.utils.meta import dict_deep_keys, dict_deep_update


__all__ = ['env', 'load_env', 'has_env', 'get_env', 'set_env', 'with_env']


class Environ(object):
    __env_ext__ = '.env.pkl'

    def __init__(self, envs=None):
        self.envs = dict()
        if envs is not None:
            self.load(envs)

    def load(self, env_spec, incremental=False):
        new_envs = self.__get_envs_from_spec(env_spec)
        if incremental:
            dict_deep_update(self.envs, new_envs)
        else:
            self.envs = deepcopy(new_envs)
        return self

    def update(self, env_spec):
        return self.load(env_spec, incremental=True)

    def dump(self, path, prefix=None):
        raise NotImplementedError('Not supported yet: "Environ.dump".')

    def as_dict(self):
        return deepcopy(self.envs)

    def as_dict_ref(self):
        return self.envs

    def clone(self):
        new_env = Environ()
        new_env.envs = deepcopy(self.envs)
        return new_env

    def keys(self, is_flattened=True):
        if is_flattened:
            return dict_deep_keys(self.envs)
        return list(self.envs.keys())

    def has(self, key):
        """
        Check whether a key is in current env object.
        :param key: the key.
        :return: True if the provided key is in current env object.
        """
        return self.get(key, None) is not None

    def get(self, key, default=None):
        """
        Get a value of a environment provided a key. You can provide a default value, but this value will not affect
        the env object.
        :param key: the key, note that dict of dict can (should) be imploded by ``.''.
        :param default: if the given key is not found in current env object, the default value will be returned.
        :return: the value if the env contains the given key, otherwise the default value provided.
        """
        subkeys = key.split('.')
        current = self.envs
        for subkey in subkeys[0:-1]:
            if subkey not in current:
                current[subkey] = dict()
            current = current[subkey]
        if subkeys[-1] in current:
            return current[subkeys[-1]]
        elif default is None:
            return default
        else:
            current[subkeys[-1]] = default
            return default

    def set(self, key, value=None, do_inc=False, do_replace=True, inc_default=0):
        """
        Set an environment value by key-value pair.
        :param key: the key, note that dict of dict can (should) be imploded by ``.''.
        :param value: the value.
        :param do_inc: if True, will perform += instead of =
        :param do_replace: if True, will set the value regardless of its original value
        :param inc_default: the default value for the do_inc operation
        :return: self
        """
        subkeys = key.split('.')
        current = self.envs
        for subkey in subkeys[0:-1]:
            if subkey not in current:
                current[subkey] = dict()
            current = current[subkey]
        if do_inc:
            if subkeys[-1] not in current:
                current[subkeys[-1]] = inc_default
            current[subkeys[-1]] += value
        elif do_replace or subkeys[-1] not in current:
            current[subkeys[-1]] = value
        return self

    def set_default(self, key, default=None):
        """
        Set an environment value by key-value pair. If the key already exists, it will not be overwritten.
        :param key: the key, note that dict of dict can (should) be imploded by ``.''.
        :param default: the ``default'' value.
        :return: self
        """
        self.set(key, default, do_replace=False)

    def inc(self, key, inc=1, default=0):
        """
        Increase the environment value provided a key.
        :param key: the key, note that dict of dict can (should) be imploded by ``.''.
        :param inc: the number to be increased,
        :param default: the default value of the accumulator.
        :return:
        """
        self.set(key, inc, do_inc=True, inc_default=default)
        return self

    def __contains__(self, item):
        return self.has(item)

    def __getitem__(self, item):
        return self.get(item, None)

    def __setitem__(self, key, value):
        self.set(key, value)
        return value

    def __get_envs_from_spec(self, env_spec):
        if isinstance(env_spec, str) and env_spec.endswith(self.__env_ext__):
            raise NotImplementedError('Not implemented loading method.')
        elif isinstance(env_spec, dict):
            return env_spec
        elif isinstance(env_spec, object) and (hasattr(env_spec, 'envs') or hasattr(env_spec, '__envs__')):
            return getattr(env_spec, 'envs', None) or getattr(env_spec, '__envs__')
        else:
            raise TypeError('unsupported env spec: {}.'.format(env_spec))


env = Environ()

load_env = env.load
update_env = env.update
has_env = env.has
get_env = env.get
set_env = env.set


@contextlib.contextmanager
def with_env(env_spec, incremental=True):
    if not incremental:
        backup = env.as_dict_ref()
    else:
        backup = env.as_dict()

    env.load(env_spec, incremental=incremental)
    yield

    env.envs = backup
