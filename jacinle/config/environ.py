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
from typing import Any

from jacinle.utils.meta import dict_deep_keys, dict_deep_update


__all__ = ['env', 'load_env', 'has_env', 'get_env', 'set_env', 'with_env']


class Environ(object):
    """A global environment object."""

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

    def has(self, key: str) -> bool:
        """Check whether a key is in current env object.

        Args:
            key: the key.

        Returns:
            True if the key is in current env object, otherwise False.
        """
        return self.get(key, None) is not None

    def get(self, key, default=None):
        """Get a value of a environment provided a key. You can provide a default value, but this value will not affect the env object.

        Args:
            key: the key. Dict of dict can (should) be imploded by ``.``.
            default: the default value.

        Returns:
            The value if the env contains the given key, otherwise the default value provided.
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

    def set(self, key: str, value: Any = None, do_inc: bool = False, do_replace: bool = True, inc_default: Any = 0):
        """Set an environment value by key-value pair.

        Args:
            key: the key, note that dict of dict can (should) be imploded by ``.''.
            value: the value.
            do_inc: whether to increase the value.
            do_replace: whether to replace the value if the key already exists.
            inc_default: the default value of the accumulator.

        Returns:
            self
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

    def set_default(self, key: str, default: Any = None):
        """Set an environment value by key-value pair. If the key already exists, it will not be overwritten.

        Args:
            key: the key, note that dict of dict can (should) be imploded by ``.''.
            default: the default value.

        Returns:
            self

        :param key: the key, note that dict of dict can (should) be imploded by ``.''.
        :param default: the ``default'' value.
        :return: self
        """
        self.set(key, default, do_replace=False)

    def inc(self, key: str, inc: Any = 1, default: Any = 0):
        """Increase the environment value provided a key.

        Args:
            key: the key.
            inc: the increment.
            default: the default value.

        Returns:
            self
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

