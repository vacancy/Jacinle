#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test-learn-ptb.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/04/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import unittest

import jaclearn.nlp.tree as tree
import jaclearn.nlp.tree.constituency as constituency

PTB_TESTCASE = '(S (NP (NP (DT The) (NN time)) (PP (IN for) (NP (NN action)))) (VP (VBZ is) (ADVP (RB now))) (. .))'


class TestPTBFormat(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ptb_load(self):
        t = tree.PTBNode.from_string(PTB_TESTCASE)
        self.assertEqual(t.to_string(), PTB_TESTCASE)

        binarized = constituency.binarize_tree(t)
        balanced = constituency.make_balanced_binary_tree(t.to_sentence(False))

        self.assertEqual(t.to_sentence(), binarized.to_sentence())
        self.assertEqual(t.to_sentence(), balanced.to_sentence())

    def test_ptb_mask(self):
        t = tree.PTBNode.from_string(PTB_TESTCASE)
        t = constituency.binarize_tree(t)
        tokens = [tree.PTBNode('', x) for x in t.to_sentence(False)]

        answer = constituency.compose_bianry_tree_step_masks(t)
        for x, _ in answer:
            imm = tree.PTBNode('')
            tokens[x].attach(imm)
            tokens[x + 1].attach(imm)
            tokens = tokens[:x] + [imm] + tokens[x+2:]

        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].to_string(vtype=False), t.to_string(vtype=False))


if __name__ == '__main__':
    unittest.main()

