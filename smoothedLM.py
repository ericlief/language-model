#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project: NLP
Created on 28.12.16
@author: Eric Lief
"""

from languageModel import *
import numpy as np


class SmoothedLM:
    """
    This class holds the training, test and heldout data,
    in order to facilitate the training of the smoothing
    parameters and subsequent testing.
    """

    def __init__(self, train, heldout, test):
        self._trainModel = LanguageModel(train)         # training model
        self._testModel = LanguageModel(test)           # test model
        self._heldoutModel = LanguageModel(heldout)     # heldout model used for smoothing params
        self._lambdas = [0, 0, 0, 0]                    # smoothing params (lambdas)

    def getLambdas(self):
        """
        Returns lambdas of model comoputed
        with heldout data
        :return: lambdas
        """
        return self._lambdas

    def EM(self, L, e):
        """
        Expectation-Maximization algorithm, which
        computes smoothing params.
        :param L: lambda vector
        :param e: tolerance used to test convergence
        :return:
        """
        i = 1                   # iteration
        while True:
            nextL = 4 * [0]     # next lambda

            # E Step (Step 1)
            # Iterate through all occurring trigrams
            # in the heldout data (H), i.e. minimizing
            # log likelihood
            counts = [0, 0, 0, 0]
            ratio = [0, 0, 0, 0]
            for (w1, w2, w3), cnt in self._heldoutModel.trigrams.items():
                ratio[3] = L[3] * self._trainModel.p3(w1, w2, w3) / self._p_(w1, w2, w3, L)  # ratio of p3/p' to model distribution function
                ratio[2] = L[2] * self._trainModel.p2(w2, w3) / self._p_(w1, w2, w3, L)
                ratio[1] = L[1] * self._trainModel.p1(w3) / self._p_(w1, w2, w3, L)
                ratio[0] = L[0] * self._trainModel.p0() / self._p_(w1, w2, w3, L)

                # M-step (Step 2)
                # Calculate expected counts of lambdas, i.e. weight, taking
                # into account the number of occurrences of each trigram (cnt)
                for j in range(len(L)):
                    counts[j] += cnt * ratio[j]                 # weight of lambda in whole equation (count)

            # Update values for parameters given current distribution
            for k in range(len(L)):
                total = np.sum(counts)
                nextL[k] = counts[k] / total        # next lambda

            # Check if lambda values have converged
            converged = True
            for l in range(len(L)):
                if np.abs(nextL[l] - L[l]) > e:     # tolerance = e
                    converged = False
            L = nextL

            # Return values if lambdas have converged
            if converged:
                break

            i += 1          # increment iteration counter

        self._lambdas = list(L)     # copy lambdas passed by reference

        return list(L)

    def _p_(self, w1, w2, w3, L):
        """
        Computes the smoothed (weighted) probability P' using
        the distribution calculated for the language model
        using training data
        :param w1: w_i-2
        :param w2: w_1-1
        :param w3: w_i
        :param L: lambdas
        :return: P'
        """
        result = L[3] * self._trainModel.p3(w1, w2, w3) + L[2] * self._trainModel.p2(w2, w3) \
               + L[1] * self._trainModel.p1(w3) + L[0] * self._trainModel.p0()
        return result

    def crossEntropy(self, L):
        """
        Compute the cross entropy H or negative
        log likelihood of test data using
        the new smoothed language model.

        :param L: lambdas
        :return: H
        """

        # Iterate through trigrams in model
        H = 0  # cross entropy
        for (w1, w2, w3), cnt in self._testModel.trigrams.items():
            # log likelihood/cross-entropy
            H += cnt * np.log2(self._p_(w1, w2, w3, L))  # this is p' (model), multiplied by the count of the trigrams

        H *= -1.0 / self._testModel.T  # per word cross entropy/negative log likelihood
        return H

    def boostL3(self, x, L):
        """
        Add x% to lambdas (boost)
        :param x: % to boost
        :param L: lambdas
        :return: new lambdas
        """

        # Boost trigram lambda (L3) by 100x%:
        # L3 = L3 + x * (1 - L3)
        # Then discount the factor x*(1-L3)
        # from the remaining three lambdas equally
        factor = x * (1 - L[3])
        newL3 = L[3] + factor

        # Scale lambdas now
        s = L[0] + L[1] + L[2] + newL3
        L[3] = newL3 / s
        L[2] = L[2] / s
        L[1] = L[1] / s
        L[0] = L[0] / s

        return L

    def discountL3(self, x, L):
        """
        Discount x% to lambdas (boost)
        :param x: % to boost
        :param L: lambdas
        :return: new lambdas
        """

        # Discount trigram lambda (L3) by 100x%:
        # L3 = x * L3
        # Then boost remaining Lambdas proportionally
        newL3 = x * L[3]
        s = L[0] + L[1] + L[2] + newL3

        # Scale lambdas now
        L[3] = newL3 / s
        L[2] = L[2] / s
        L[1] = L[1] / s
        L[0] = L[0] / s

        return L

    def coverageGraph(self):
        """
        Calculate percentage of words in test data which
        have already been seen in the training data.
        :return: coverage
        """
        count = 0
        for word in self._testModel.unigrams:
            if word in self._trainModel.unigrams:
                count += 1
        return count / self._testModel.V


