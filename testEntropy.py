#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project: NLP
Created on 26.12.16
@author: Eric Lief
"""

import unittest
from entropy import *

tokens = []
with open("unittest.txt", encoding='iso-8859-2') as f:
    for line in f:
        if line != '\n':
            tokens.append(line.strip())
model = Entropy(tokens)
# model.printGrams()
model.charSet()
for char in model.printCharFreq():
    print(char)

tokens = []
with open("unittest2.txt", encoding='iso-8859-2') as f:
    for line in f:
        if line != '\n':
            tokens.append(line.strip())
model = Entropy(tokens)
for char in model.printCharFreq():
    print(char)


# model.printGrams()

class TestEntropy(unittest.TestCase):
    """
    To run unittest:

    python -m unittest -v languagemodel.py

    """

    def setUp(self):
        tokens = []
        with open("unittest.txt", encoding='iso-8859-2') as f:
            for line in f:
                if line != '\n':
                    tokens.append(line.strip())
        self.model1 = Entropy(tokens)

        tokens = []
        with open("unittest2.txt", encoding='iso-8859-2') as f:
            for line in f:
                if line != '\n':
                    tokens.append(line.strip())
        self.model2 = Entropy(tokens)

    # Entropy (H) of text
    def test_h(self):
        self.assertAlmostEqual(self.model1.h(), 3.656001116)
        self.assertAlmostEqual(self.model2.h(), 3.533209185)

    # Conditional entropy (H) of text
    def test_h2(self):
        self.assertAlmostEqual(self.model1.h2(), .794692647)
        self.assertAlmostEqual(self.model2.h2(), .713666544)

    # Perplexity (G) of text
    def test_g(self):
        self.assertAlmostEqual(self.model1.g(), 12.60567193)
        self.assertAlmostEqual(self.model2.g(), 11.57715761)

    # Conditional perplexity (G) of text
    def test_g2(self):
        self.assertAlmostEqual(self.model1.g2(), 1.734707771)
        self.assertAlmostEqual(self.model2.g2(), 1.639966726)

    # Character/letter inventory
    def test_chars(self):
        self.assertEqual(self.model1.charSet(),
                         ['I', 'a', 'm', 'S', '.', 'd', 'o', 'n', 't', 'l', 'i', 'k', 'e', 'g', 'r', 's', 'h'])
        self.assertEqual(self.model2.charSet(),
                         ['I', 'a', 'm', 'S', '.', 'd', 'o', 'n', 't', 'l', 'i', 'k', 'e', 'g', 'r', 's', 'h'])

    def test_charCount(self):
        self.assertEqual(self.model1.charCount(), 40)
        self.assertEqual(self.model2.charCount(), 80)

    def test_charFreq(self):
        self.assertEqual(self.model1.charFreq()['a'], 6 / 40)
        self.assertEqual(self.model2.charFreq()['a'], 12 / 80)
        self.assertEqual(self.model1.charFreq()['I'], 3 / 40)
        self.assertEqual(self.model2.charFreq()['I'], 6 / 80)
        self.assertEqual(self.model1.charFreq()['S'], 2 / 40)
        self.assertEqual(self.model2.charFreq()['S'], 4 / 80)
        self.assertEqual(self.model1.charFreq()['e'], 4 / 40)
        self.assertEqual(self.model2.charFreq()['e'], 8 / 80)
        try:
            self.assertEqual(self.model1.charFreq()['z'], 0)
            self.assertEqual(self.model2.charFreq()['z'], 0)
        except KeyError:
            pass


if __name__ == "__main__":
    unittest.main()
