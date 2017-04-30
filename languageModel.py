#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 11:10:26 2016

@author: mambo
"""

from collections import defaultdict


class LanguageModel:
    """
    This class holds all gram distribution data,
    as well as the methods to calculate probabilities
    and setter and getter methods.
    """
    def __init__(self, tokens):
        self._tokens = list(tokens)
        self._trigrams = self.findTrigrams()
        self._bigrams = self.findBigrams()
        self._unigrams = self.findUnigrams()
        self._T = len(self._tokens)		        # text size
        self._V = len(self._unigrams) 	        # vocabulary size


    @property
    def T(self):
        return self._T

    @property
    def V(self):
        return self._V

    @property
    def tokens(self):
        """
        Get method for tokens.
        :return:
                list of tokens
        """
        return self._tokens

    @property
    def unigrams(self):
        """
        Get method for unigrams
        :return:
                dictionary of unigram
                counts.
        """
        return self._unigrams

    @property
    def bigrams(self):
        """
        Get method for bigrams
        :return:
                dictionary of bigram
                counts.
        """
        return self._bigrams

    @property
    def trigrams(self):
        """
        Get method for unigrams
        :return:
                dictionary of trigram
                counts.
        """
        return self._trigrams

    def findUnigrams(self):
        """
        Helper method for computing unigram
        counts
        :return:
                dictionary of unigrams
        """
        unigrams = defaultdict(int)
        for first, second in self._bigrams:
            unigrams[first] += self._bigrams[(first, second)]
        return dict(unigrams)

    def findBigrams(self):
        """
        Helper method for computing bigram
        counts
        :return:
                dictionary of bigrams
        """
        bigrams = defaultdict(int)
        for first, second, third in self._trigrams:
            bigrams[(first, second)] += self._trigrams[(first, second, third)]
        return dict(bigrams)

    def findTrigrams(self):
        """
        Helper method for computing trigram
        counts
        :return:
                dictionary of trigrams
        """
        trigrams = defaultdict(int)
        first = '<s>'
        second = '<s>'
        for third in self._tokens:
            trigrams[(first, second, third)] += 1
            first = second
            second = third

        # process end of text
        trigrams[(self._tokens[-1], '</s>', '</s>')] = 1   # insert final ending symbol
        trigrams[(self._tokens[-2], self._tokens[-1], '</s>')] = 1   # insert final ending symbol

        return dict(trigrams)

    def p0(self, w=None):
            """
            zero-gram-based probability
            :param w:
                        word: string
            :return:
                        probability: float
            """

            return 1 / self._V

    # need to except ZeroDivisionError
    def p1(self, w):
        """
        Unigram-based probability
        :param w:
                    word: string
        :return:
                    probability: float
        """
        try:
            p = self._unigrams[w] / self._T
        except ZeroDivisionError:
            return 0
        except KeyError:
            return 0
        return p

    def p2(self, w1, w2):
        """
        Bigram-based conditional
        probability
        P(w2 | w1) = C(w1, w2)
                   = C(w1):

        :param w1:
                    word: string
        :param w2:
                    word: string
        :return:
                    probability: float
        """
        try:
            p = self._bigrams[(w1, w2)] / self._unigrams[w1]
        except ZeroDivisionError:
            return 0
        except KeyError:
            return 0
        return p

    def p3(self, w1, w2, w3):
        """
        Trigram-based conditional
        probability:

        P(w3 | w1, w2) = C (w1, w2, w3)
                         C (w1, w2)
        :param w1:
                    word: string
        :param w2:
                    word: string
        :param w3:
                    word: string
        :return:
                    probability: float
        """
        try:
            p = self._trigrams[(w1, w2, w3)] / self._bigrams[(w1, w2)]
        except ZeroDivisionError:
            return 0
        except KeyError:
            return 0
        return p

    def printGrams(self):
        print("token size {}\n".format(self.T))
        print("vocab size {}\n".format(self.V))
        print("unigrams of size {}\n{}".format(sum(self.unigrams.values()), self.unigrams))
        print("bigrams of size {}\n{}".format(sum(self.bigrams.values()), self.bigrams))
        print("trigrams of size {}\n{}".format(sum(self.trigrams.values()), self.trigrams))

