#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project: NLP
Created on 26.12.16
@author: Eric Lief
"""


import unittest
from languageModel import *


tokens = []
with open("unittest.txt", encoding='iso-8859-2') as f:
    for line in f:
        if line != '\n':
            tokens.append(line.strip())
model = LanguageModel(tokens)
model.printGrams()

tokens = []
with open("unittest2.txt", encoding='iso-8859-2') as f:
    for line in f:
        if line != '\n':
            tokens.append(line.strip())
model = LanguageModel(tokens)
model.printGrams()


class TestLanguageModel(unittest.TestCase):
    """
    To run unittest:

    python -m unittest -v languageModel.py

    """

    def setUp(self):
        tokens = []
        with open("unittest.txt", encoding='iso-8859-2') as f:
            for line in f:
                if line != '\n':
                    tokens.append(line.strip())
        self.lm = LanguageModel(tokens)
        tokens = []
        with open("unittest2.txt", encoding='iso-8859-2') as f:
            for line in f:
                if line != '\n':
                    tokens.append(line.strip())
        self.lm2 = LanguageModel(tokens)

    def test_tokenSize(self):
        self.assertEqual(len(self.lm.tokens), 17)
        self.assertEqual(len(self.lm2.tokens), 34)

    def test_vocabSize(self):
        self.assertEqual(self.lm.V, 12)
        self.assertEqual(self.lm2.V, 12)

    def test_unigramSize(self):
        self.assertEqual(sum(self.lm.unigrams.values()), 19)
        self.assertEqual(sum(self.lm2.unigrams.values()), 36)

    def test_bigramSize(self):
        self.assertEqual(sum(self.lm.bigrams.values()), 19)
        self.assertEqual(sum(self.lm2.bigrams.values()), 36)

    def test_trigramSize(self):
        self.assertEqual(sum(self.lm.trigrams.values()), 19)
        self.assertEqual(sum(self.lm2.trigrams.values()), 36)

    def test_p0(self):
        self.assertAlmostEqual(self.lm.p0("Sam"), 1/12)
        self.assertAlmostEqual(self.lm.p0("I"), 1/12)
        self.assertAlmostEqual(self.lm2.p0("Sam"), 1/12)
        self.assertAlmostEqual(self.lm2.p0("I"), 1/12)

    def test_p1(self):
        self.assertAlmostEqual(self.lm.p1("I"), 3/17)
        self.assertAlmostEqual(self.lm.p1("am"), 2/17)
        self.assertAlmostEqual(self.lm.p1("green"), 1/17)
        self.assertAlmostEqual(self.lm.p1("."), 3/17)
        self.assertAlmostEqual(self.lm.p1("<s>"), 2/17)
        self.assertAlmostEqual(self.lm2.p1("I"), 6/34)
        self.assertAlmostEqual(self.lm2.p1("am"), 4/34)
        self.assertAlmostEqual(self.lm2.p1("green"), 2/34)
        self.assertAlmostEqual(self.lm2.p1("."), 6/34)
        self.assertAlmostEqual(self.lm2.p1("<s>"), 2/34)

    def test_p2(self):
        self.assertAlmostEqual(self.lm.p2(".", "I"), 1/3)
        self.assertAlmostEqual(self.lm.p2("I", "am"), 2/3)
        self.assertAlmostEqual(self.lm.p2("Sam", "I"), 1/2)
        self.assertAlmostEqual(self.lm.p2(".", "</s>"), 1/3)
        self.assertAlmostEqual(self.lm.p2("am", "Sam"), 1/2)
        self.assertAlmostEqual(self.lm.p2("<s>", "I"), 1/2)
        self.assertAlmostEqual(self.lm.p2("ham", "."), 1/1)
        self.assertAlmostEqual(self.lm2.p2(".", "I"), 3/6)
        self.assertAlmostEqual(self.lm2.p2("I", "am"), 4/6)
        self.assertAlmostEqual(self.lm2.p2("Sam", "I"), 2/4)
        self.assertAlmostEqual(self.lm2.p2(".", "</s>"), 1/6)
        self.assertAlmostEqual(self.lm2.p2("ham", "."), 2/2)

    def test_p3(self):
        self.assertAlmostEqual(self.lm.p3("<s>", "<s>", "I"), 1/1)
        self.assertAlmostEqual(self.lm.p3("<s>", "I", "am"), 1/1)
        self.assertAlmostEqual(self.lm.p3("I", "am", "Sam"), 1/2)
        self.assertAlmostEqual(self.lm.p3("and", "ham", "."), 1/1)
        self.assertAlmostEqual(self.lm.p3("ham", ".", "</s>"), 1/1)
        self.assertAlmostEqual(self.lm.p3("not", "not", "not"), 0)
        self.assertAlmostEqual(self.lm.p3(".", "</s>", "</s>"), 1/1)
        self.assertAlmostEqual(self.lm2.p3("<s>", "<s>", "I"), 1/1)
        self.assertAlmostEqual(self.lm2.p3("<s>", "I", "am"), 1/1)
        self.assertAlmostEqual(self.lm2.p3("I", "am", "Sam"), 2/4)
        self.assertAlmostEqual(self.lm2.p3("and", "ham", "."), 2/2)
        self.assertAlmostEqual(self.lm2.p3("ham", ".", "</s>"), 1/2)
        self.assertAlmostEqual(self.lm2.p3(".", "</s>", "</s>"), 1/1)


if __name__ == "__main__":
    unittest.main()