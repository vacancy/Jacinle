#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : extract_rule.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/16/2022
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

from jacinle.utils.enum import JacEnum
from typing import Sequence, List, Mapping, Any, Optional, Union
from sklearn.tree import _tree, DecisionTreeClassifier

__all__ = ['DecisionRuleFormat', 'AtomicDecisionRule', 'DecisionRule', 'extract_rule']


class DecisionRuleFormat(JacEnum):
    PYTHON = 'python'
    LISP = 'lisp'


class AtomicDecisionRule(object):
    def __init__(self, variable: Any, threshold: Optional[float], right_branch: Optional[bool] = False):
        """
        Instantiate an atomic decision rule. That is, by comparing a variable with a threshold.

        - left branch: `variable <= threshold`
        - right branch: `variable > threshold`

        Args:
            variable (str): the name for the variable.
            threshold (Optional[float]): the threshold. If set to none, indicates that the variable is a boolean variable.
            right_branch (Optional[bool]): the boolean indicator.
        """

        self.variable = variable
        self.threshold = threshold
        self.right_branch = right_branch

    def format(self, format: Union[DecisionRuleFormat, str]):
        """
        Format an atomic decision rule into a string.

        Args:
            format (Union[DecisionRuleFormat, str]): the format.
        """
        format = DecisionRuleFormat.from_string(format)

        is_bool = (self.threshold is None)
        if not is_bool:
            if format is DecisionRuleFormat.PYTHON:
                return f'{self.variable} <= {self.threshold}' if not self.right_branch else f'{self.variable} <= {self.threshold}'
            elif format is DecisionRuleFormat.LISP:
                return f'(leq {self.variable} {self.threshold})' if not self.right_branch else f'(ge {self.variable} {self.threshold})'
            else:
                raise ValueError('Unknown format: {}.'.format(format))
        else:
            if format is DecisionRuleFormat.PYTHON:
                return f'not {self.variable}' if not self.right_branch else f'{self.variable}'
            elif format is DecisionRuleFormat.LISP:
                if self.variable.startswith('(') and self.variable.endswith(')'):
                    return f'(not {self.variable})' if not self.right_branch else f'{self.variable}'

                return f'(not ({self.variable}))' if not self.right_branch else f'({self.variable})'
            else:
                raise ValueError('Unknown format: {}.'.format(format))

    def __str__(self):
        return self.format(DecisionRuleFormat.PYTHON)

    __repr__ = __str__


class DecisionRule(object):
    """
    A decision rule is a pair of (DNF, label).
    """
    def __init__(self, clauses: List[List[AtomicDecisionRule]], label: Any, probabilities: Optional[List[float]] = None):
        """
        Instantiate a decision formula.

        Args:
            clauses (Sequence[Sequence[AtomicDecisionRule]]): the DNF, represented as a two-level nested list.
            label (Any): the output label.

        """
        self.clauses = clauses
        self.label = label
        self.probabilities = probabilities

    def format_clause(self, format: Union[DecisionRuleFormat, str]):
        format = DecisionRuleFormat.from_string(format)

        if format is DecisionRuleFormat.PYTHON:
            assert self.probabilities is None, 'Python format does not support probability.'
            if len(self.clauses) == 1:
                return ('( ' + ' and '.join([
                    atom.format(format) for atom in self.clauses[0]
                ]) + ' )')
            return '(\n  ' + ' or\n  '.join([
                ('( ' + ' and '.join([
                    atom.format(format) for atom in clause
                ]) + ' )')
            for clause in self.clauses]) + '\n)'
        elif format is DecisionRuleFormat.LISP:
            if self.probabilities is not None:
                probabilities_str = [
                    f'{p:.4f} '
                    for p in self.probabilities
                ]
                if len(self.clauses) == 1:
                    return (f'{probabilities_str[0]}(and ' + ' '.join([
                        atom.format(format) for atom in self.clauses[0]
                    ]) + ')')
                return '(or\n  ' + '\n  '.join([
                    (f'{pstr}(and ' + ' '.join([
                        atom.format(format) for atom in clause
                    ]) + ')')
                for pstr, clause in zip(probabilities_str, self.clauses)]) + '\n)'
            else:
                if len(self.clauses) == 1:
                    return ('(and ' + ' '.join([
                        atom.format(format) for atom in self.clauses[0]
                    ]) + ')')
                return '(or\n  ' + '\n  '.join([
                    ('(and ' + ' '.join([
                        atom.format(format) for atom in clause
                    ]) + ')')
                for clause in self.clauses]) + '\n)'
        else:
            raise ValueError('Unknown format: {}.'.format(format))

    def __str__(self):
        return self.format_clause(DecisionRuleFormat.PYTHON)

    __repr__ = __str__


def extract_rule(
    decision_tree: DecisionTreeClassifier,
    feature_names: Sequence[Any],
    boolean_input: Optional[bool] = True,
    boolean_output: Optional[bool] = False,
    multi_output: bool = False,
) -> Mapping[Any, DecisionRule]:
    """
    Extract logic rules (DNF) from a trained DecisionTreeClassifier.

    Args:
        decision_tree (DecisionTreeClassifier): a pre-trained decision tree.
        feature_names (Sequence[str]): a list of strings for the features.
        boolean_input (Optional[bool]): whether the input features are boolean or not.
        boolean_output (Optional[bool]): whether the output features are boolean or not.
        multi_output (bool): whether the output features are multi-valued or not. In this case, all possible values
            at a leaf node will be registered.

    Returns:
        rule_dict (Mapping[Any, DecisionRule]): A mapping from label (strings, integers...) to the corresponding rule.
    """
    rule_dict = dict()
    true_probabilities = list()

    inner = decision_tree.tree_
    classes = decision_tree.classes_

    if boolean_output:
        assert set(classes).issubset([False, True])
        if True not in list(classes):
            return dict()
        tindex = list(classes).index(True)
        rule_dict[True] = list()

    def dfs(node, rule):
        if inner.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[inner.feature[node]]
            threshold = inner.threshold[node] if not boolean_input else None # not used for binary features.

            rule_neg = rule.copy()
            rule_neg.append(AtomicDecisionRule(name, threshold, False))
            dfs(inner.children_left[node], rule_neg)
            rule_pos = rule.copy()
            rule_pos.append(AtomicDecisionRule(name, threshold, True))
            dfs(inner.children_right[node], rule_pos)
        else:
            if boolean_output:
                if multi_output:
                    for index in inner.value[node][0].nonzero()[0]:
                        val = bool(decision_tree.classes_[index])
                        rule_dict.setdefault(val, list()).append(rule)
                else:
                    if inner.value[node][0, tindex] > 0:
                        val = inner.value[node][0, tindex] / inner.value[node].sum()
                        rule_dict[True].append(rule)
                        true_probabilities.append(val)
            else:
                if multi_output:
                    for index in inner.value[node][0].nonzero()[0]:
                        val = int(decision_tree.classes_[index])
                        rule_dict.setdefault(val, list()).append(rule)
                else:
                    val = decision_tree.classes_[inner.value[node].argmax()]
                    rule_dict.setdefault(val, list()).append(rule)


    try:
        dfs(0, [])

        if boolean_output:
            rule_dict[True] = DecisionRule(rule_dict[True], True, probabilities=true_probabilities)
        else:
            for k, v in rule_dict.items():
                rule_dict[k] = DecisionRule(v, k)
        return rule_dict
    finally:
        del dfs

