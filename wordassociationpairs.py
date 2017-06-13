#!usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project: Code
File: wordpairs
Created on 12.6.17
@author: Eric Lief
"""

from collections import Counter, defaultdict
import io
import sys
import numpy as np

class WordAssociationPairs:

    def __init__(self, tokens):
        self._tokens = tokens
        self._word_counts = Counter(tokens)
        self._T = len(tokens)
        #self.counts_adj_pairs = defaultdict(lambda: defaultdict(lambda: 0))
        #self.pmi_adj_pairs = defaultdict(lambda: defaultdict(lambda: 0.0))

        # Get counts for all adjacent word pairs
        self.count_adjacent_word_pairs()

        # Calculate pointwise mutual information (PMI) for adjacent word pairs
        self.pointwise_mutual_information_adj_pairs()

        # Get counts for all distant word pairs
        self.count_distant_word_pairs()

        # Calculate pointwise mutual information (PMI) for distant word pairs
        self.pointwise_mutual_information_dist_pairs()

    def count_adjacent_word_pairs(self):
        self.counts_adj_pairs = defaultdict(lambda: defaultdict(lambda: 0))
        w1 = self._tokens[0]
        for w2 in self._tokens[1:]:
            self.counts_adj_pairs[w1][w2] += 1
            w1 = w2

    def pointwise_mutual_information_adj_pairs(self):
        self.pmi_adj_pairs = defaultdict(lambda: defaultdict(lambda: 0.0))
        for w1, seconds in self.counts_adj_pairs.items():
            for w2, count in seconds.items():
                c_w1 = self._word_counts[w1]
                c_w2 = self._word_counts[w2]
                if c_w1 + c_w2 < 20:
                    continue
                pmi = np.log2(count * self._T / (c_w1 * c_w2))
                self.pmi_adj_pairs[w1][w2] = pmi
        print('done with adj pairs')

    def count_distant_word_pairs(self):
        print('getting dist counts')
        self.counts_dist_pairs = defaultdict(lambda: defaultdict(lambda: 0))
        for i, w1 in enumerate(self._tokens):
            # Get pairs to the right
            dist = 0
            for w2 in self._tokens[i:]:     # skip adj word?
                if 1 < dist <= 50:
                    self.counts_dist_pairs[w1][w2] += 1
                    # print('dist ', dist)
                    # print(w1, w2)
                dist += 1

            # Get pairs to left
            # tokens = self._tokens[]
            dist = 0
            # print(self._tokens[i-2::-1])
            for w2 in self._tokens[i::-1]:
                if 1 < dist <= 50:
                    self.counts_dist_pairs[w1][w2] += 1
                    # print('dist ', dist)
                    # print(w1, w2)
                dist += 1
                # print(i, w1, j, w2)
        #print(self.counts_dist_pairs)
        print('done w counts')

    def pointwise_mutual_information_dist_pairs(self):
        self.pmi_dist_pairs = defaultdict(lambda: defaultdict(lambda: 0.0))
        print('getting pmi ')
        for w1, seconds in self.counts_dist_pairs.items():
            for w2, count in seconds.items():
                c_w1 = self._word_counts[w1]
                c_w2 = self._word_counts[w2]
                if c_w1 + c_w2 < 20:
                    continue
                pmi = np.log2(count * self._T / (c_w1 * c_w2))
                self.pmi_dist_pairs[w1][w2] = pmi
                print(w1,w2,pmi)

    @property
    def T(self):
        return self.T

    @property
    def word_counts(self, w):
        return self._word_counts[w]

    # @property
    # def counts_adj_pairs(self, w1, w2):
    #     return self.counts_adj_pairs[w1][w2]
    #
    # @counts_adj_pairs.setter
    # def counts_adj_pairs(self, w1, w2, count):
    #     self.counts_adj_pairs[w1][w2] = count
    #
    # @property
    # def pmi_adj_pairs(self, w1, w2):
    #     return self.pmi_adj_pairs[w1, w2]
    #
    # @pmi_adj_pairs.setter
    # def pmi_adj_pairs(self, w1, w2 , pmi):
    #     self.pmi_adj_pairs[w1][w2] = pmi


if __name__ == "__main__":

    # Test
    stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    tokens = []
    for line in stream:
        if line != '\n':
            tokens.append(line.strip())

    model = WordAssociationPairs(tokens)
    # print(model.counts_adj_pairs)
    # print(model.pmi_adj_pairs)

    out = "adj.txt"
    with open(out, 'w') as f:
        for w1, seconds in model.pmi_adj_pairs.items():
            for w2, count in seconds.items():
                f.write(str(count) + '\t' + w1 + '\t' + w2 + '\t' + '\n')
    out = "dist.txt"
    with open(out, 'w') as f:
        for w1, seconds in model.pmi_dist_pairs.items():
            for w2, count in seconds.items():
                f.write(str(count) + '\t' + w1 + '\t' + w2 + '\t' + '\n')


