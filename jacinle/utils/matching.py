#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : matching.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/02/2017
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""Functions to match names using glob patterns."""

import fnmatch
import re
import collections
from typing import Any, Optional, Union, Iterable, Tuple, List, Dict

__all__ = ['NameMatcher', 'IENameMatcher']


class NameMatcher(object):
    """A name matcher based on a set of glob patterns.

    The rule set is a list of (pattern, value) pairs. The pattern is a glob pattern, and the value is the value to be returned when the pattern matches.

    Example:
        .. code-block:: python

            matcher = NameMatcher({'*.jpg': 'image', '*.png': 'image', '*.txt': 'text'})
            with matcher:
                matcher.match('a.jpg')  # 'image'
                matcher.match('a.png')  # 'image'

            matched, unused = matcher.get_last_stat()  # Return a tuple of (matched values, unmatched patterns).
            print(matched)  # [('a.jpg', '*.jpg', 'image'), ('a.png', '*.png', 'image')]
            print(unused)  # {'*.txt'}
    """

    def __init__(self, rules: Optional[Union[Iterable[Tuple[str, Any]], Dict[str, Any]]] = None):
        """Initialize the name matcher.

        Args:
            rules: A list of (pattern, value) pairs, or a dict of {pattern: value}.
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
    def rules(self) -> List[Tuple[str, Any]]:
        """Get the rules."""
        return self._rules

    def map(self) -> Dict[str, Any]:
        """Get the map of {pattern: value}."""
        assert self._compiled
        return self._map

    def append_rule(self, rule: Tuple[str, Any]):
        """Append a rule to the rule set. The rule is a (pattern, value) pair.

        Args:
            rule: the rule to be appended.
        """
        self._rules.append(tuple(rule))

    def insert_rule(self, index: int, rule: Tuple[str, Any]):
        """Insert a rule to the rule set at a given position (priority). The rule is a (pattern, value) pair."""
        self._rules.insert(index, rule)

    def pop_rule(self, index=None):
        """Pop a rule from the rule set.

        Args:
            index: the index of the rule to be popped. If None, the last rule will be popped.
        """
        self._rules.pop(index)

    def begin(self, *, force_compile=False):
        """Begin a matching session."""
        if not self._compiled or force_compile:
            self.compile()
        self._matched = []
        self._unused = set(range(len(self._compiled_rules)))

    def end(self):
        """End a matching session, which returns a tuple of (matched values, unmatched patterns). See the docstring of :class:`NameMatcher` for more details."""
        return self._matched, {self._compiled_rules[i][0] for i in self._unused}

    def match(self, k: str) -> Optional[Any]:
        """Match a name against the rule set. Return the value if matched, otherwise return None.

        Args:
            k: the name to be matched.

        Returns:
            The value if matched, otherwise None.
        """
        for i, (r, p, v) in enumerate(self._compiled_rules):
            if p.match(k):
                if i in self._unused:
                    self._unused.remove(i)
                self._matched.append((k, r, v))
                return v
        return None

    def compile(self):
        """Compile the rule set."""
        self._map = dict()
        self._compiled_rules = []

        for r, v in self._rules:
            self._map[r] = v
            p = fnmatch.translate(r)
            p = re.compile(p, flags=re.IGNORECASE)
            self._compiled_rules.append((r, p, v))
        self._compiled = True

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._last_stat = self.end()

    def get_last_stat(self):
        """Get the last matching session's result."""
        return self._last_stat


class IENameMatcher(object):
    """A name matcher based on two sets of glob patterns: one for inclusion and one for exclusion.

    - When ``include`` is None, ``exclude`` is not None, the matcher will match all names that are not excluded.
    - When ``include`` is not None, ``exclude`` is None, the matcher will match all names that are included.
    - When ``include`` is not None, ``exclude`` is not None, the matcher will match all names that are included and not excluded.
        The ``exclude`` rule set has higher priority than the ``include`` rule set.

    Example:
        .. code-block:: python

            matcher = IENameMatcher(include=['*.jpg', '*.png'], exclude=['*.bak.png'])
            with matcher:
                matcher.match('a.jpg')  # True
                matcher.match('a.png')  # True
                matcher.match('a.bak.png')  # False
                matcher.match('a.txt')  # False
                matcher.match('a.bak.txt')  # False

            stat_type, things = matcher.get_last_stat()
            print(stat_type)  # 'exclude'
            # Everything that has been rejected.
            print(things)  # ['a.bak.png', 'a.txt', 'a.bak.txt']
    """

    def __init__(self, include: Optional[Iterable[str]], exclude: Optional[Iterable[str]]):
        """Initialize the name matcher.

        Args:
            include: a list of glob patterns for inclusion.
            exclude: a list of glob patterns for exclusion.
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
        """Begin a matching session."""
        if self.include is not None:
            self.include.begin()
        if self.exclude is not None:
            self.exclude.begin()
        self._last_stat = (set(), set())

    def end(self):
        """End a matching session, which returns a tuple of ``(stat_type, things)`` See the docstring of :class:`IENameMatcher`."""
        if self.include is not None:
            self.include.end()
        if self.exclude is not None:
            self.exclude.end()

        if len(self._last_stat[0]) < len(self._last_stat[1]):
            self._last_stat = ('included', self._last_stat[0])
        else:
            self._last_stat = ('excluded', self._last_stat[1])

    def match(self, k: str) -> bool:
        """Match a name against the rule set. Return True if matched, otherwise return False."""
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
        self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()

    def get_last_stat(self):
        """Get the last matching session's result."""
        return self._last_stat
