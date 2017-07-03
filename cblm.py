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
import math


class ClassBasedLM:
    def __init__(self, tokens):
        self._tokens = tokens               # tokens from whole text
        self._word_counts = Counter(tokens) # unigram word counts
        self._T = len(tokens)               # this is the size of the whole text
        self._bg_uniq_8k = None             # this is the set of unique bigrams from 8k word text

        # Initialize n-gram data structures: count tables, etc.
        self.init_n_grams(tokens)
        self._V = len(self._unigram_rhs_counts) # size of vocab

        # Uncomment for pointwise MI

        # # Get counts for all adjacent word pairs
        # self.count_adjacent_word_pairs()
        #
        # # Calculate pointwise mutual information (PMI) for adjacent word pairs
        # self.pointwise_mutual_information_adj_pairs()
        #
        # # Get counts for all distant word pairs
        # self.count_distant_word_pairs()
        #
        # # Calculate pointwise mutual information (PMI) for distant word pairs
        # self.pointwise_mutual_information_dist_pairs()

    def init_n_grams(self, tokens):
        """
        Helper method for computing unigram
        and bigrams counts.
        """
        # Add start and end symbols

        tokens = tokens[:8000]
        tokens = ['<s>'] + tokens + ['</s>']

        # Initialize bigram counts table
        B = 0
        bigram_counts = defaultdict(lambda: defaultdict(lambda: 0))
        left = tokens[0]
        for right in tokens[1:]:
            bigram_counts[left][right] += 1
            left = right
            B += 1
        self._bigram_counts = bigram_counts

        # Initialize unigram counts dictionaries
        # Build from bigram counts
        unigram_lhs_counts = defaultdict(lambda: 0)
        unigram_rhs_counts = defaultdict(lambda: 0)
        unigram_counts = defaultdict(lambda: 0)
        U = 0
        for left in bigram_counts:
            for right in bigram_counts[left]:
                unigram_lhs_counts[left] += bigram_counts[left][right]
                unigram_rhs_counts[right] += bigram_counts[left][right]
                unigram_counts[left] += bigram_counts[left][right]
                U += 1
        self._unigram_lhs_counts = unigram_lhs_counts
        self._unigram_rhs_counts = unigram_rhs_counts
        self._unigram_counts = unigram_counts
        L = sum(unigram_lhs_counts.values())
        R = sum(unigram_rhs_counts.values())

        # print(L, R, L==R)
        self._U = L
        self._B = B

        # Initialize bigram counts table for first 8000 words
        words_8k = tokens[:8000]
        word_count = Counter(words_8k)
        word_count_10x = [(w, c) for w, c in word_count.items() if c >= 10]  # [('man', 3), ('the', 22),  ...]
        # word_count_10x = sorted(word_count_10x, key=lambda x: x[1], reverse=True)       # [('the', 22), ('man', 3), ...]
        words_10x = [w for w, c in word_count_10x]  # [('the', 'man', ...]
        word_to_class = {w: i for i, w in enumerate(words_10x)}  # {'the': 0, 'man': 1, ...}
        class_to_word = {i: set([w]) for i, w in enumerate(words_10x)}  # {0: {'the',...}, 1: {'man',...}, ...}

        # Construct table of bigram class counts
        # QUESTION: Do we need to add a start/end symbol here?
        n = len(words_10x)
        class_counts_table = [n * [0] for x in range(n)]
        for i in range(n):
            for j in range(n):
                left = words_10x[i]
                right = words_10x[j]
                if bigram_counts.get(left):
                    if bigram_counts.get(left).get(right):  # should exist, but just in case...
                        count = bigram_counts.get(left).get(right)
                        class_counts_table[i][j] = count  # set class bigram count
                        # class_to_word[i].append(left)
                        # class_to_word[j].append(right)

        # Arrays storing unigram class counts
        unigram_class_counts_lhs = n * [0]
        unigram_class_counts_rhs = n * [0]
        for i in range(n):
            cl = words_10x[i]  # class/word
            if unigram_lhs_counts.get(cl):
                count = unigram_lhs_counts[cl]
                unigram_class_counts_lhs[i] = count
            if unigram_rhs_counts.get(cl):
                count = unigram_rhs_counts[cl]
                unigram_class_counts_rhs[i] = count

        # This is the set of classes, starting as {0,1,2,...},
        # We will subtract classes after each merge
        classes = list(range(n))
        # print(classes)
        print('\nunigram class counts\n')
        print(classes)
        print(unigram_class_counts_lhs)
        print(unigram_class_counts_rhs)
        print('lhs\n', unigram_lhs_counts)
        print('rhs\n', unigram_rhs_counts)

        # Set instance variables
        self._word_count_10x = word_count_10x
        self._words_10x = words_10x
        self._class_counts_table = class_counts_table
        self._class_to_word = class_to_word
        self._word_to_class = word_to_class
        self._classes = classes
        self._unigram_class_counts_lhs = unigram_class_counts_lhs
        self._unigram_class_counts_rhs = unigram_class_counts_rhs

        # TODO
        non_zeroes = defaultdict(list)

    def mi(self):  # latest: get bigrams of 8k text
        counts = self._class_counts_table
        n = len(counts[0])

        T = self._T
        B = self._B
        U = self._U
        tokens = self._tokens[:8000]
        bg = list(zip(tokens, tokens[1:]))
        bg_uniq_8k = set(bg)
        self._bg_uniq_8k = bg_uniq_8k
        print(bg_uniq_8k)

        mi = 0
        for (left, right) in bg_uniq_8k:
            # for right in self.bigram_counts[left]:
            # print(left, right, self.bigram_counts[left][right])
            c_l_r = self._bigram_counts[left][right]
            c_l = self._unigram_counts[left]
            c_r = self._unigram_counts[right]
            # c_l = unig[left]
            # c_r = unig[right]
            mi += c_l_r / float(B) * np.log2(B * c_l_r / ((float(c_l) * c_r)))
            # self.pmi_adj_pairs[left][right] = pmi

        print('done with MI', mi, B)
        return mi

    def merge(self, stop=15):
        mi = self.mi()  # initial mi
        n = len(self._classes)
        table_losses = [n * [0] for x in range(n)]

        # Iterate until stop condition is reached
        # while len(classes) > stop:
        # for i in range(1):
        #     print('iter=', i)
        #     # classes = self._classes


        iter = 1
        min_loss_a_b = math.inf
        min_loss_pair = None  # argmin  (a,b)
        while len(self._classes) >= stop:

            print('iter=', iter)
            print('classes remaining')
            print(self._classes)

            classes = list(self._classes)
            # Get the first class A to merge
            for a in classes:

                s_k_a = self.sum_k(a)
                print('a=', a, 'sk_a=', s_k_a)
                # min_loss_a_b = math.inf
                # min_loss_b = None
                for b in classes:

                    # print('b=', b)
                    if self._class_counts_table[a][b] == 0:
                        # print('bg=0, skipping')
                        continue

                    s_k_b = self.sum_k(b)

                    # print('sk_b=', s_k_b)

                    # Compute sums of columns and rows to be subtracted.
                    # These are all class bigrams containing a and those
                    # containing b. This is the subtraction subterm.
                    loss_a_b = s_k_a + s_k_b - self.quality(a, b) - self.quality(b, a) - self.add(a, b)

                    # print('loss=', loss_a_b)

                    if loss_a_b < min_loss_a_b:
                        min_loss_a_b = loss_a_b
                        min_loss_pair = (a, b)

                        print('got min pair ', min_loss_pair)

                    print(a, b, loss_a_b)

            print('merging', self._words_10x[min_loss_pair[0]], self._words_10x[min_loss_pair[1]])
            print('min loss for ', min_loss_pair, min_loss_a_b)
            print('removing ', min_loss_pair[1])
            print(self._classes)

            self._classes.remove(min_loss_pair[1])
            iter += 1

            print("STOP")
            print('classes left', self._classes)

            break

            # table_losses[a][b]

            # n = le    `n(self._words_10x)
            #
            # for a in self._class_counts_table:
            #     for b in self._class_counts_table:
            #         q =
            #
            #
            # for i in range(n-1, 0, -1):
            #     for m in range(n):
            #         left = self._words_10x[i]
            #         right = self._words_10x[j]
            #         if bigram_counts.get(left):
            #             if bigram_counts.get(left).get(right):  # should exist, but just in case...
            #                 count = bigram_counts.get(left).get(right)
            #                 class_counts[i][j] = count  # set class bigram count
            #                 # class_to_word[i].append(left)
            #                 # class_to_word[j].append(right)

            # def init_l_k(self, mi, a, b):
            #     return self.subtract(a, b) + self.add(a, b)

    def add(self, a, b):
        s = 0
        T = self._B

        # Sum the quality for all class pairs for future merge of a+b
        i = self._words_10x[a]
        j = self._words_10x[b]
        # c_lhs_ab = self.unigram_lhs_counts[l] + self.unigram_rhs_counts[r]
        # c_rhs_ab = self.unigram_class_counts_rhs[a] + self.unigram_class_counts_rhs[b]
        c_lhs_ab = self.unigram_class_counts_lhs[a] + self.unigram_class_counts_lhs[b]
        c_rhs_ab = self.unigram_class_counts_rhs[a] + self.unigram_class_counts_rhs[b]

        # print('a=', a, 'b=', b, 'c_lhs/rhs_ab', c_lhs_ab, c_rhs_ab)

        for t in self._tokens[:8000]:

            # This is a+b on the left-hand-side (lhs)
            c_ab_t = self._bigram_counts[i][t] + self._bigram_counts[j][t]
            c_rhs_t = self.unigram_rhs_counts[t]

            # print('c=', c, 'c_ab_c', c_ab_c, 'c_rhc_c', c_rhs_c)

            if c_ab_t != 0:
                # print('ab-c: zero found, skipping')
                s += c_ab_t / float(T) * np.log2(T * c_ab_t / ((float(c_lhs_ab) * c_rhs_t)))

            # This is a+b on the right-hand-side (rhs)
            c_t_ab = self._bigram_counts[t][a] + self._bigram_counts[t][b]
            c_lhs_t = self.unigram_lhs_counts[t]

            # print('c_c_ab', c_c_ab, 'c_lhs_c', c_lhs_c)
            if c_t_ab != 0:
                s += c_t_ab / float(T) * np.log2(T * c_t_ab / ((float(c_lhs_t) * c_rhs_ab)))

        # This is a+b on both sides: (a+b)(a+b)
        c_ab_ab = self._class_counts_table[a][a] + self._class_counts_table[a][b] + \
                  self._class_counts_table[b][a] + self._class_counts_table[b][b]

        # print('c_ab_ab', c_ab_ab)

        if c_ab_ab != 0:
            # print('ab-ab: zero found, skipping')
            s += c_ab_ab / float(T) * np.log2(T * c_ab_ab / ((float(c_lhs_ab) * c_rhs_ab)))

        # print('add done', s)

        return s

    def sum_k(self, a):
        """
        Compute sum of quality (mi contribution)
        of all pairs (l, r) containing
        Class A. We do so by iterating through the columns and rows
        of the matrix.
        These are all class bigrams containing a.
        :param a:
        :return: sum
        """

        # Sum the quality for all class pairs containing A
        s = 0
        i = self._words_10x[a]
        for t in self._tokens[:8000]:
            # count = self._class_counts_table[b][a]
            s += self.quality(t, i)

        for t in self._tokens[:8000]:
            # count = self._class_counts_table[b][a]
            s += self.quality(i, t)

        # subtract double counted intersection
        s -= self.quality(i, i)

        # print('sum ', s)

        return s

    def quality(self, a, b):
        T = self._B
        # print(T)
        q = 0
        # c_l_r = self._bigram_counts[left][right]
        c_l_r = self._bigram_counts[a][b]
        # left = self._words_10x[a]
        # right = self._words_10x[b]
        # c_l = self._unigram_lhs_counts[left]
        # c_r = self.unigram_rhs_counts[right]
        c_l = self._unigram_lhs_counts[a]
        c_r = self._unigram_rhs_counts[b]
        # print(c_l_r, '/', c_l, '*', c_r)

        if c_l_r != 0:
            q = c_l_r / float(T) * np.log2(T * c_l_r / ((float(c_l) * c_r)))

        # print('quality of ', a, b, q)

        return q

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
                if c_w1 < 10 or c_w2 < 10:
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
            for w2 in self._tokens[i:]:  # skip adj word?
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
        # print(self.counts_dist_pairs)
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
                print(w1, w2, pmi)

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
    def unigram_lhs_counts(self):
        """
        Get method for unigrams
        :return:
                dictionary of unigram
                counts.
        """
        return self._unigram_lhs_counts

    @property
    def unigram_rhs_counts(self):
        """
        Get method for unigrams
        :return:
                dictionary of unigram
                counts.
        """
        return self._unigram_rhs_counts

    @property
    def unigram_class_counts_lhs(self):
        """
        Get method for unigrams
        :return:
                dictionary of unigram
                counts.
        """
        return self._unigram_class_counts_lhs

    @property
    def unigram_class_counts_rhs(self):
        """
        Get method for unigrams
        :return:
                dictionary of unigram
                counts.
        """
        return self._unigram_class_counts_rhs

    @property
    def bigram_counts(self):
        """
        Get method for bigrams
        :return:
                dictionary of bigram
                counts.
        """
        return self._bigram_counts

    @property
    def class_counts(self):
        """
        Get method for bigrams
        :return:
                dictionary of bigram
                counts.
        """
        return self._class_counts_table

    @property
    def word_counts(self, w):
        return self._word_counts[w]

    @property
    def words_10x(self):
        return self._words_10x

    @property
    def class_to_word(self):
        return self._class_to_word

    @property
    def word_to_class(self, word):
        return self._word_to_class[word]

    def print_class_counts_table(self, n):
        for i in range(n):
            for j in range(n):
                print(self._class_counts_table[i][j], end=' ')
            print()


if __name__ == "__main__":
    # Test
    stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    tokens = []
    for line in stream:
        if line != '\n':
            tokens.append(line.strip())

    model = ClassBasedLM(tokens)

    # print(model.words_10x)
    # print(model.bigram_counts)
    # print(model._unigram_counts)
    # print(model.words_10x)

    # print(model.class_to_word[3])
    # print(model.class_to_word[10])
    # print(model.class_counts[3][10])
    # print(model.bigram_counts['and']['I'])
    #
    # print(model.class_counts[3][10]==model.bigram_counts['and']['I'])
    # print(model.words_10x[0])
    # print(model.words_10x[1])
    # print(model.bigram_counts['the']['I'])
    # print(model.class_counts[1][10]==model.bigram_counts['the']['I'])
    # print(model.bigram_counts['the'])

    # print(len(model.class_counts[0]))
    # print(model.class_counts)

    model.print_class_counts_table(20)
    # print('mi=', model.mi2())

    # print('init mi=', model.mi())

    model.merge(15)
    print(model.words_10x)
    # print(model._classes)


