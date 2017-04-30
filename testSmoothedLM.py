#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project: NLP
Created on 28.01.17
@author: Eric Lief
"""


import unittest
from smoothedLM import *


class TestSmoothedLanguageModel(unittest.TestCase):

    def setUp(self):
        tokens = []
        with open("unittest.txt", encoding='iso-8859-2') as f:
            for line in f:
                if line != '\n':
                    tokens.append(line.strip())
        self.lm = SmoothedLM(tokens)

        tokens = []
        with open("unittest2.txt", encoding='iso-8859-2') as f:
            for line in f:
                if line != '\n':
                    tokens.append(line.strip())
        self.lm2 = SmoothedLM(tokens)

    def test_tokenSize(self):
        self.assertEqual(len(self.lm.tokens), 17)
        self.assertEqual(len(self.lm2.tokens), 34)

    def test_vocabSize(self):
        self.assertEqual(self.lm.V, 12)
        self.assertEqual(self.lm2.V, 12)

    def test__p__(self):

        L = [.25, .25, .25, .25]
        self.assertAlmostEqual(self.lm._p_("not", "not", "not", L), .035539216)
        self.assertAlmostEqual(self.lm._p_("I", "am", "Sam", L), .300245098)
        self.assertAlmostEqual(self.lm._p_("<s>", "<s>", "I", L), .43995098)

        L = [.1, .2, .3, .4]
        self.assertAlmostEqual(self.lm._p_("not", "not", "not", L), .020098039)
        self.assertAlmostEqual(self.lm._p_("I", "am", "Sam", L), .381862745)
        self.assertAlmostEqual(self.lm._p_("<s>", "<s>", "I", L), .593627451)


if __name__ == "__main__":
    """
    To run unittest:

    python -m unittest -v languageModel.py

    """
    unittest.main()

