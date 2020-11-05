#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : matching.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/02/2017
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.


import fnmatch
import re
import collections

__all__ = ['NameMatcher', 'IENameMatcher']


class NameMatcher(object):
    def __init__(self, rules=None):
        """
        Initialize the rules.

        Args:
            self: (todo): write your description
            rules: (dict): write your description
        """
        if rules is None:
            self._rules = []
        elif isinstance(rules, dict):
            self._rules = list(rules.items())
        else:
            assert isinstance(rules, collections.Iterable)
            self._rules = list(rules)

        self._map = {}
        self._compiled_rules = []
        self._compiled = False

        self._matched = []
        self._unused = set()
        self._last_stat = None

    @property
    def rules(self):
        """
        Returns a : class

        Args:
            self: (todo): write your description
        """
        return self._rules

    def map(self):
        """
        Return a new map with the map.

        Args:
            self: (todo): write your description
        """
        assert self._compiled
        return self._map

    def append_rule(self, rule):
        """
        Add a new rule.

        Args:
            self: (todo): write your description
            rule: (todo): write your description
        """
        self._rules.append(tuple(rule))

    def insert_rule(self, index, rule):
        """
        Insert a new rule.

        Args:
            self: (todo): write your description
            index: (int): write your description
            rule: (todo): write your description
        """
        self._rules.insert(index, rule)

    def pop_rule(self, index=None):
        """
        Pop a rule from the index.

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        self._rules.pop(index)

    def begin(self, *, force_compile=False):
        """
        Begin the rules.

        Args:
            self: (todo): write your description
            force_compile: (str): write your description
        """
        if not self._compiled or force_compile:
            self.compile()
        self._matched = []
        self._unused = set(range(len(self._compiled_rules)))

    def end(self):
        """
        The end of this rule.

        Args:
            self: (todo): write your description
        """
        return self._matched, {self._compiled_rules[i][0] for i in self._unused}

    def match(self, k):
        """
        Matches a pattern matches.

        Args:
            self: (todo): write your description
            k: (todo): write your description
        """
        for i, (r, p, v) in enumerate(self._compiled_rules):
            if p.match(k):
                if i in self._unused:
                    self._unused.remove(i)
                self._matched.append((k, r, v))
                return v
        return None

    def compile(self):
        """
        Compile rules. rules

        Args:
            self: (todo): write your description
        """
        self._map = dict()
        self._compiled_rules = []

        for r, v in self._rules:
            self._map[r] = v
            p = fnmatch.translate(r)
            p = re.compile(p, flags=re.IGNORECASE)
            self._compiled_rules.append((r, p, v))
        self._compiled = True

    def __enter__(self):
        """
        Returns the first call.

        Args:
            self: (todo): write your description
        """
        self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Called when an exception.

        Args:
            self: (todo): write your description
            exc_type: (todo): write your description
            exc_val: (todo): write your description
            exc_tb: (todo): write your description
        """
        self._last_stat = self.end()

    def get_last_stat(self):
        """
        Return the last modification_stat_stat.

        Args:
            self: (todo): write your description
        """
        return self._last_stat



class IENameMatcher(object):
    def __init__(self, include=None, exclude=None):
        """
        Initialize a variable statements.

        Args:
            self: (todo): write your description
            include: (todo): write your description
            exclude: (todo): write your description
        """
        if include is None:
            self.include = None
        else:
            self.include = NameMatcher([(i, True) for i in include])

        if exclude is None:
            self.exclude = None
        else:
            self.exclude = NameMatcher([(e, True) for e in exclude])
        self._last_stat = None

    def begin(self):
        """
        Begin the database.

        Args:
            self: (todo): write your description
        """
        if self.include is not None:
            self.include.begin()
        if self.exclude is not None:
            self.exclude.begin()
        self._last_stat = (set(), set())

    def end(self):
        """
        Ends the last document.

        Args:
            self: (todo): write your description
        """
        if self.include is not None:
            self.include.end()
        if self.exclude is not None:
            self.exclude.end()

        if len(self._last_stat[0]) < len(self._last_stat[1]):
            self._last_stat = ('included', self._last_stat[0])
        else:
            self._last_stat = ('excluded', self._last_stat[1])

    def match(self, k):
        """
        Returns true if k matches pattern.

        Args:
            self: (todo): write your description
            k: (todo): write your description
        """
        if self.include is None:
            ret = True
        else:
            ret = bool(self.include.match(k))

        if self.exclude is not None:
            ret = ret and not bool(self.exclude.match(k))

        if ret:
            self._last_stat[0].add(k)
        else:
            self._last_stat[1].add(k)
        return ret

    def __enter__(self):
        """
        Returns the first call.

        Args:
            self: (todo): write your description
        """
        self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Called when an exception.

        Args:
            self: (todo): write your description
            exc_type: (todo): write your description
            exc_val: (todo): write your description
            exc_tb: (todo): write your description
        """
        self.end()

    def get_last_stat(self):
        """
        Return the last modification_stat_stat.

        Args:
            self: (todo): write your description
        """
        return self._last_stat
