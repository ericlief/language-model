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
        self._tokens = tokens
        self._word_counts = Counter(tokens)
        # self._bigrams = self.findBigrams()
        # self._unigrams = self.findUnigrams()

        self._T = len(tokens)

        #self.counts_adj_pairs = defaultdict(lambda: defaultdict(lambda: 0))
        #self.pmi_adj_pairs = defaultdict(lambda: defaultdict(lambda: 0.0))

        # Initialize n-gram data structures: count tables, etc.
        self.init_n_grams(tokens)
        self._V = len(self._unigram_rhs_counts)

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
        # n = len(tokens)

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
        print(L, R, L==R)
        self._U = L
        self._B = B

        # Initialize bigram class counts table
        # class_counts_table = defaultdict(lambda: defaultdict(lambda: 0))

        # word_to_class = {}

        # Initialize bigram counts table for first 8000 words

        # bigram_counts_8k = defaultdict(lambda: defaultdict(lambda: 0))
        # word_count = Counter(tokens[:10])

        # Wrong, compute 10X times in the 8k tokens, not whole text
        # words_10x = [token for token in words_8k if unigram_lhs_counts[token] >= 10]  # use right hand since no ending symbol inserted?

        # words_8k = tokens
        words_8k = tokens[:8000]

        # bg = list(zip(tokens, tokens[1:]))
        # bg_c = Counter(bg)
        # print(bg_c)
        # mi = 0
        # # unig = Counter(tokens)
        # B = len(bg)
        # T = len(tokens)
        # for (left,right), count in bg_c.items():
        #
        #     # for right in self.bigram_counts[left]:
        #         # print(left, right, self.bigram_counts[left][right])
        #     c_l_r = count
        #     # c_l = self._unigram_counts[left]
        #     # c_r = self._unigram_counts[right]
        #     c_l = unig[left]
        #     c_r = unig[right]
        #     mi += c_l_r / float(B) * np.log2(B * c_l_r / ((float(c_l) * c_r)))
        #     # self.pmi_adj_pairs[left][right] = pmi
        #
        # print('done with MI', mi, B)

        word_count = Counter(words_8k)
        word_count_10x = [(w, c) for w, c in word_count.items() if c >= 2]              # [('man', 3), ('the', 22),  ...]
        word_count_10x = sorted(word_count_10x, key=lambda x: x[1], reverse=True)       # [('the', 22), ('man', 3), ...]
        words_10x = [w for w, c in word_count_10x ]                                     # [('the', 'man', ...]
        word_to_class = {w: i for i, w in enumerate(words_10x)}                         # {'the': 0, 'man': 1, ...}
        class_to_word = {i: set([w]) for i, w in enumerate(words_10x)}                  # {0: {'the',...}, 1: {'man',...}, ...}


        # Construct table of bigram class counts
        # QUESTION: Do we need to add a start/end symbol here?
        n = len(words_10x)
        class_counts_table = [n*[0] for x in range(n)]
        for i in range(n):
            for j in range(n):
                left = words_10x[i]
                right = words_10x[j]
                if bigram_counts.get(left):
                    if bigram_counts.get(left).get(right):          # should exist, but just in case...
                        count = bigram_counts.get(left).get(right)
                        class_counts_table[i][j] = count                  # set class bigram count
                        # class_to_word[i].append(left)
                        # class_to_word[j].append(right)

        # Arrays storing unigram class counts
        unigram_class_counts_lhs = n * [0]
        unigram_class_counts_rhs = n * [0]
        for i in range(n):
            cl = words_10x[i]           # class/word
            if unigram_lhs_counts.get(cl):
                count = unigram_lhs_counts[cl]
                unigram_class_counts_lhs[i] = count
            if unigram_rhs_counts.get(cl):
                count = unigram_rhs_counts[cl]
                unigram_class_counts_rhs[i] = count

        print(unigram_class_counts_rhs)

        # left = words_8k[0]
        # for right in words_8k[1:]:
        #     if left in words_10x and right in words_10x:
        #         i = words_10x.index(left)
        #         j = words_10x.index(right)
        #         count = bigram_counts[left][right]
        #         if count != 0:
        #             class_counts_table[i][j] = count
        #             non_zeroes[i].append(j)
        #             # word_to_class[left] = i
        #             # word_to_class[right] = j
        #             class_to_word[i].append(left)
        #             class_to_word[j].append(right)
        #         # i += 1
        #         # print(left,right,count, i,j)
        #
        #         # bigram_counts_8k[left][right] += 1
        #     left = right

        # This is the set of classes, starting as {0,1,2,...},
        # We will subtract classes after each merge
        classes = set(range(n))
        # print(classes)

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


    def mi(self):
        n = 8000
        T = self._T
        mi = 0
        # Get bigrams
        pairs = defaultdict(lambda: defaultdict(lambda: 0))
        left = self._tokens[0]
        for right in self._tokens[1:n]:
            pairs[left][right] += 1
            left = right

        for left, seconds in pairs.items():
            for right, count in seconds.items():
                c_l_r = self._bigram_counts[left][right]
                c_l = self._unigram_lhs_counts[left]
                c_r = self._unigram_rhs_counts[right]
                # if not c_l or not c_r:
                #     continue
                mi += (c_l_r / T) * np.log2(T * c_l_r / (c_l * c_r))
                # self.pmi_adj_pairs[left][right] = pmi
        # print('done with adj pairs')
        return mi

    def mi2(self):
        counts = self._class_counts_table
        n = len(counts[0])
        T = self._T
        # T = 8000
        # tokens = self._tokens[:8000]
        # tokens = ['<s>'] + tokens + ['</s>']

        mi = 0
        for i in range(n):
            for j in range(n):
                if counts[i][j] == 0:
                    continue
                left = self._words_10x[i]
                right = self._words_10x[j]
                # c_l_r = self._bigram_counts[left][right]
                # c_l = self._unigram_lhs_counts[left]
                # c_r = self._unigram_rhs_counts[right]
                #
                c_l_r = self._bigram_counts[left][right]
                c_l = self._unigram_counts[left]
                c_r = self._unigram_counts[right]
                # if not c_l or not c_r:
                #     continue
                mi += c_l_r / float(T) * np.log2(T * c_l_r / ((float(c_l) * c_r)))
                # self.pmi_adj_pairs[left][right] = pmi
        # print('done with adj pairs')
        return mi

    def mi3(self):
        counts = self._class_counts_table
        n = len(counts[0])

        T = self._T
        B = self._B
        U = self._U
        # T = len(sum(self.unigram_counts.values()))
        print('size of text', T)
        print('size of bigrams', B)
        print('size of unigrams', U)

        # T = 8000
        # tokens = self._tokens[:8000]
        # tokens = ['<s>'] + tokens + ['</s>']

        # print(self.bigram_counts)
        # print(self._unigram_counts)
        mi = 0

        for left in self.bigram_counts:
            for right in self.bigram_counts[left]:
                # print(left, right, self.bigram_counts[left][right])
                c_l_r = self._bigram_counts[left][right]
                # c_l = self._unigram_counts[left]
                # c_r = self._unigram_counts[right]
                c_l = self._unigram_lhs_counts[left]
                c_r = self._unigram_rhs_counts[right]
                mi += c_l_r / float(B) * np.log2(B * c_l_r / ((float(c_l) * c_r)))
            # self.pmi_adj_pairs[left][right] = pmi

        print('done with MI', mi, B)
        return mi

    def mi4(self):      # latest: get bigrams of 8k text
        counts = self._class_counts_table
        n = len(counts[0])

        T = self._T
        B = self._B
        U = self._U
        tokens = self._tokens[:8000]
        bg = list(zip(tokens, tokens[1:]))
        bg_uniq = set(bg)

        print(bg_uniq)
        mi = 0
        # unig = Counter(tokens)
        for (left, right) in bg_uniq:
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
        mi = self.mi4()     # initial mi
        classes = self._classes
        table_losses = [len(classes) * [0] for x in range(len(classes))]

        # Iterate until stop condition is reached
        while len(classes) > stop:

            # Get the first class A to merge
            for a in self._classes:
                s_k_a = self.sum_k(a)
                min_loss_a_b = math.inf
                min_loss_b = None
                for b in self._classes:

                    s_k_b = self.sum_k(b)

                    # Compute sums of columns and rows to be subtracted.
                    # These are all class bigrams containing a and those
                    # containing b. This is the subtraction subterm.
                    loss_a_b =  s_k_a + s_k_b - self.quality(a, b) - self.quality(b, a) + self.add(a, b)
                    if loss_a_b < min_loss_a_b:
                        min_loss_a_b = loss_a_b
                        min_loss_b = b

                    print(a,b,loss_a_b)
                    print('min loss for ', a, min_loss_b, min_loss_a_b)
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

        c_lhs_ab = self.unigram_class_counts_lhs[a] + self.unigram_class_counts_lhs[b]
        c_rhs_ab = self.unigram_class_counts_rhs[a] + self.unigram_class_counts_rhs[b]

        for c in self._classes:
            # count = self._class_counts_table[b][a]
            # c_l_r = self._bigram_counts[left][right]

            # This is a+b on the left-hand-side (lhs)
            c_ab_c = self._class_counts_table[a][c] + self._class_counts_table[b][c]
            # c_lhs_ab = self._unigram_lhs_counts[a] + self._unigram_lhs_counts[b]
            c_rhs_c = self.unigram_class_counts_rhs[c]
            s += c_ab_c / float(T) * np.log2(T * c_ab_c / ((float(c_lhs_ab) * c_rhs_c)))

            # This is a+b on the right-hand-side (lhs)
            c_c_ab = self._class_counts_table[c][a] + self._class_counts_table[c][b]
            # c_rhs_ab = self._unigram_rhs_counts[a] + self._unigram_rhs_counts[b]
            c_lhs_c = self.unigram_class_counts_lhs[c]
            s += c_c_ab / float(T) * np.log2(T * c_c_ab / ((float(c_lhs_c) * c_rhs_ab)))

        # This is a+b on both sides: (a+b)(a+b)
        c_ab_ab =  self._class_counts_table[a][a] + self._class_counts_table[a][b] + \
                   self._class_counts_table[b][a] + self._class_counts_table[b][b]
        s += c_ab_ab / float(T) * np.log2(T * c_ab_ab / ((float(c_lhs_ab) * c_rhs_ab)))

        return s

    # def subtract(self, a, b):
    #
    #     # Compute sums of columns and rows to be subtracted.
    #     # These are all class bigrams containing a and those
    #     # containing b. This is sk.
    #     sub = 0
    #     s_a = self.row_col_sum(a)
    #     s_b = self.row_col_sum(b)
    #     return s_a + s_b - self.quality(a, b) - self.quality(b, a)

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
        for b in self._classes:
            # count = self._class_counts_table[b][a]
            s += self.quality(b, a)
            s += self.quality(a, b)
        # subtract double counted intersection
        s - self.quality(a, a)
        return s

    def quality(self, a, b):
        T = self._B
        # c_l_r = self._bigram_counts[left][right]
        c_l_r = self._class_counts_table[a][b]
        # left = self._words_10x[a]
        # right = self._words_10x[b]
        # c_l = self._unigram_lhs_counts[left]
        # c_r = self.unigram_rhs_counts[right]
        c_l = self._unigram_class_counts_lhs[a]
        c_r = self._unigram_class_counts_rhs[b]


        return c_l_r / float(T) * np.log2(T * c_l_r / ((float(c_l) * c_r)))

    # def quality_merged_left(self, a, b, r):
    #     T = self._B
    #     # c_l_r = self._bigram_counts[left][right]
    #     c_l_r = self._class_counts_table[a][r] +  self._class_counts_table[b][r]
    #     c_l = self._unigram_lhs_counts[a] + self._unigram_lhs_counts[b]
    #     c_r = self.unigram_rhs_counts[r]
    #     return c_l_r / float(T) * np.log2(T * c_l_r / ((float(c_l) * c_r)))
    #
    # def quality_merged_right(self, a, b, l):
    #     T = self._B
    #     # c_l_r = self._bigram_counts[left][right]
    #     c_l_r = self._class_counts_table[l][a] +  self._class_counts_table[l][b]
    #     c_r = self._unigram_rhs_counts[a] + self._unigram_rhs_counts[b]
    #     c_l = self.unigram_lhs_counts[l]
    #     return c_l_r / float(T) * np.log2(T * c_l_r / ((float(c_l) * c_r)))
    #
    #
    # def quality_merged_both_sides(self, a, b):
    #     T = self._B
    #     # c_l_r = self._bigram_counts[left][right]
    #     c_l_r = self._class_counts_table[l][a] +  self._class_counts_table[l][b]
    #     c_r = self._unigram_rhs_counts[a] + self._unigram_rhs_counts[b]
    #     c_l = self.unigram_lrighths_counts[l]
    #     return c_l_r / float(T) * np.log2(T * c_l_r / ((float(c_l) * c_r)))
    #
    #
    # def quality_merged(self, a, b):
    #   # Sum the quality for all class pairs containing A
    #     for b in self._classes:
    #         # count = self._class_counts_table[b][a]
    #         s += self.quality(b, a)
    #         s += self.quality(a, b)
    #     # subtract double counted intersection
    #     s - self.quality(a,a)
    #     return
    #


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

    model = ClassBasedLM(tokens)

    print(model.words_10x)
    print(model.bigram_counts)
    print(model._unigram_counts)
    model.print_class_counts_table(8)
    print(model.words_10x)
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

    model.print_class_counts_table(5)
    # print('mi=', model.mi2())

    print('init mi=', model.mi4())

    model.merge(1)

    # for i in range(20):
    #     for j in range(20):
    #         print(model._class_counts[i][j], end=' ')
    #     print()
    #
    # print(model.words_10x[17])
    # print(model.words_10x[18])

        # print(model.counts_adj_pairs)
    # print(model.pmi_adj_pairs)

    # print(model._class_counts)

    # out = "adj.txt"
    # with open(out, 'w') as f:
    #     for w1, seconds in model.pmi_adj_pairs.items():
    #         for w2, count in seconds.items():
    #             f.write(str(count) + '\t' + w1 + '\t' + w2 + '\t' + '\n')
    # out = "dist.txt"
    # with open(out, 'w') as f:
    #     for w1, seconds in model.pmi_dist_pairs.items():
    #         for w2, count in seconds.items():
    #             f.write(str(count) + '\t' + w1 + '\t' + w2 + '\t' + '\n')
    #
    #
