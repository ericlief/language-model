#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project: NLP
Created on 23.12.16
@author: Eric Lief
"""

import random
import math
import languageModel as lm
from collections import defaultdict


class Entropy:
    def __init__(self, tokens):
        self._model = lm.LanguageModel(tokens)  # wrapped language model
        self._charSet = self.charSet()          # set of chars in text
        self._charFreq = self.charFreq()        # dictionary of char frequency
        self._charCount = self.charCount()      # total chars in text

    @property
    def charFreq(self):
        """
        A property for list of chars in text
        :return:
                  dictionary of char frequency in document
        """
        return self._charFreq

    @property
    def charCount(self):
        """
        A property for chars in text
        :return:
                 count of chars in document
        """
        return self._charCount

    def wordCount(self):
        """
        Get number of words in text.
        :return:
                    number of words
        """
        return len(self._model.tokens)

    def charSet(self):
        """
        Get set of characters in document.

        :return:
                    list of letters/chars
        """

        chars = []
        for word in self._model.tokens:
            letters = list(word)            # split word in chars
            for letter in letters:
                if letter not in chars:     # add to set
                    chars.append(letter)
        return chars

    def charCount(self):
        """
        Get count of letters in document.

        :return:
                    int
        """

        count = 0
        for word in self._model.tokens:
            for char in word:
                count += 1
        return count

    def charFreq(self):
        """
        Get frequency of letters in document.

        :return:
                    string generator
        """

        chars = defaultdict(int)
        for word in self._model.tokens:
            letters = list(word)
            for letter in letters:
                chars[letter] += 1
        total = sum(chars.values())
        return {k: (v / total) for k, v in chars.items()}

    def mostFrequentWords(self):
        """
        Sort words by frequency
        :return:
                    list of (word, freq) pairs
        """

        return [(w, c) for c, w in sorted([(f, w ) for w, f in self._model.unigrams.items()], reverse=True)]

    def mostFrequentChars(self):
        """
        Sort chars by frequency
        :return:
                    list of (char, freq) pairs
        """

        return [(ch, c) for c, ch in sorted([(f, ch) for ch, f in self._charFreq.items()], reverse=True)]

    def hapax(self):
        """
        Get words which only occur once (hapax legomena)
        :return:
                    list of hapax
        """

        return [w for (w, c) in self._model.unigrams.items() if c == 1]

    def uniqueBigrams(self):
        """
        Get bigrams which only occur once (hapax legomena)
        :return:
                    list of hapax
        """

        return [b for (b, c) in self._model.bigrams.items() if c == 1]

    def printCharFreq(self):
        """
        Print frequency of letters in document.

        :return:
                    string generator
        """

        return ("%f\t%s" % (f, ch) for f, ch in sorted([(v, k) for k, v in self._charFreq.items()], reverse=True))

    def messupChars(self, prob):
        """
        Randomize characters in document
        :param prob:
                        the likelihood with which to mess up text
        :return:
                        list of randomized tokens

        """

        messedUp = []                       # list of new randomized tokens
        size = len(self._charSet)           # size of character inventory
        for word in self._model.tokens:
            newWord = ''                                    # this will be the new word
            for i in range(len(word)):                      # iterate through chars in word
                rd = random.random()                        # random float [0,1]
                if rd < prob:                               # if random less than likelihood
                    random.seed()                           # reseed
                    rd = random.random()                    # new random number to map to set
                    ch = self._charSet[math.floor(rd * size)]     # select random char from inventory
                else:
                    ch = word[i]                            # else, do nothing
                newWord += ch                               # concatenate char to new word
            messedUp.append(newWord)
        return messedUp

    def messupWords(self, prob):
        """
        Randomize words in document
        :param prob:
                         the likelihood with which to mess up text
        :return:
                         list of randomized tokens

        """
        random.seed()                           # reseed
        messedUp = []                           # list of new randomized tokens
        words = list(self._model.unigrams)      # set of words in text
        size = len(words)                       # number of words in text
        for word in self._model.tokens:         # iterate throught tokens (text)
            rd = random.random()                # random number [0, 1]
            #print(rd, end='\t')
            if rd < prob:                       # mess up with given prob
                random.seed()                   # reseed
                rd = random.random()            # new random number to map to set
                i = math.floor(rd * size)       # index in set to grab word
                #print(i)
                newWord = words[i]              # the messed up word
            else:
                newWord = word                  # or else no change
            messedUp.append(newWord)            # add word to list
        return messedUp

    def h(self):
        """
        Entropy of text:

        H(W) = sum( P(w) log P(w) ))

        :return:
                entropy: float
        """
        result = 0
        for w in self._model.unigrams:
            p = self._model.p1(w)
            result += p * math.log2(p)
        return -result

    def h2(self):
        """
        Conditional entropy of bigrams in text:

        H(W2 | W1) = sum(sum( P(w1, w2) log P(w2 | w1) ))

        bigrams.items:
        bigram[0]   bigram[1]
        (w1, w2) : count

        :return:
                    entropy
        """
        result = 0
        for bigram in self._model.bigrams.items():
            w1 = bigram[0][0]       # (w1, w2), count
            w2 = bigram[0][1]       # not used
            count = bigram[1]
            #joint_prob = count / self._model.T
            joint_prob = count / (self._model.T + 1)    # add one to text size to account for start/end symbols
            cond_prob = count / self._model.unigrams[w1]
            result += joint_prob * math.log2(cond_prob)
        return -result

    def g(self):
        """
        Perplexity of text

            G(x) = 2^H(x)

        :return:
                perplexity
        """

        return math.pow(2, self.h())

    def g2(self):
        """
        Conditional perplexity of text

                G(x) = 2^H(x)

        :return:
                conditional perplexity

        """

        return math.pow(2, self.h2())


