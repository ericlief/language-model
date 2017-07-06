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
import itertools
import traceback

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

        # tokens = tokens[:8000]                # using the 8k tokens for the task?
        tokens = ['<s>'] + tokens + ['</s>']    # insert start and end symbol

        # Initialize bigram counts table
        B = 0
        bigram_counts = defaultdict(lambda: defaultdict(lambda: 0))
        left = tokens[0]
        for right in tokens[1:]:
            bigram_counts[left][right] += 1
            left = right
            B += 1
        self._bigram_counts = bigram_counts     # use the bigram count B = T

        # Initialize unigram counts dictionaries
        # Build from bigram counts,
        # Distinguishing lhs and unigram_class_counts_rhs
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

        self._U = U     # unigram count
        self._B = B     # bigram count

        # Initialize bigram counts table for first 8000 words
        # words_8k = tokens[:8000]
        words_8k = tokens       # to use as much data for tags (uncomment)
        word_count = Counter(words_8k)
        word_count_10x = [(w, c) for w, c in word_count.items() if c >= 10]              # [('man', 3), ('the', 22),  ...]
        word_count_10x = sorted(word_count_10x, key=lambda x: x[0], reverse=True)       # [('the', 22), ('man', 3), ...]
        words_10x = [w for w, c in word_count_10x]                                      # [('the', 'man', ...]
        word_to_class = {w: i for i, w in enumerate(words_10x)}                         # {'the': 0, 'man': 1, ...}
        class_to_word = {i: set([w]) for i, w in enumerate(words_10x)}                  # {0: {'the',...}, 1: {'man',...}, ...}

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
        # print('\nunigram class counts\n')
        # print(classes)
        # print(unigram_class_counts_lhs)
        # print(unigram_class_counts_rhs)
        # print('lhs\n', unigram_lhs_counts)
        # print('rhs\n', unigram_rhs_counts)

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

    def merge_greedy(self, stop=15):
        """
        Maximize mi of classes (a, b)
        in a greedy fashion by choosing all
        pairs and calculating the mi of the
        whole text.
        :param stop: stop condition
        """
        iter = 1
        max_mi_a_b = -math.inf              # maximize mi
        max_mi_pair = None                  # argmax  (a,b)

        while len(self._classes) >= stop:

            print('iter=', iter)

            # recompute all bigrams and unigrams
            # to check for any bugs
            tokens = self._tokens[:8000]
            tokens = ['<s>'] + tokens + ['</s>']
            bg = list(zip(tokens, tokens[1:]))
            unigram_lhs = [l for l, r in bg]
            unigram_rhs = [r for l, r in bg]
            bigram_counts = Counter(bg)
            bg_uniq_8k = set(bg)
            B = len(bg)
            unigram_lhs_counts = Counter(unigram_lhs)
            unigram_rhs_counts = Counter(unigram_rhs)
            words_10x = [w for w, c in unigram_lhs_counts.items() if c >= 10]  # [('the', 'man', ...]

            # Get all combos (a, b). NB: these are the
            # actual words not indices. If either A or B or
            # both occur in the bigrams in the 8k text,
            # calculate the mi with the potential merge.
            combos = list(itertools.combinations(words_10x, 2))
            for a, b in combos:
                # print(a,b)
                mi = 0
                for (left, right) in bg_uniq_8k:
                    # print(left, right)
                    if a in (left, right) and b in (left, right):
                        # print('both found')
                        c_l_r = bigram_counts[(a, b)] + bigram_counts[(b, a)] + \
                                bigram_counts[(a, a)] + bigram_counts[(b, b)]
                        c_l = unigram_lhs_counts[a] + unigram_lhs_counts[b]
                        c_r = unigram_rhs_counts[a] + unigram_rhs_counts[b]
                    elif left == a or left == b:
                        # print('on left')
                        c_l_r = bigram_counts[(a, right)] + bigram_counts[(b, right)]
                        c_l = unigram_lhs_counts[a] + unigram_lhs_counts[b]
                        c_r = unigram_rhs_counts[right]
                    elif right == a or right == b:
                        # print('on right')
                        c_l_r = bigram_counts[(left, a)] + bigram_counts[(left, b)]
                        c_l = unigram_lhs_counts[left]
                        c_r = unigram_rhs_counts[a] + unigram_rhs_counts[b]
                    else:
                        # print('none')
                        c_l_r = bigram_counts[(left, right)]
                        c_l = unigram_lhs_counts[left]
                        c_r = unigram_rhs_counts[right]

                    mi += c_l_r / float(B) * np.log2(B * c_l_r / ((float(c_l) * c_r)))


                if mi > max_mi_a_b:         # maximize
                    max_mi_a_b = mi
                    max_mi_pair = (a, b)

                    # print(a,b, max_mi_a_b)

            print('merging', max_mi_pair)
            print('max mi ', max_mi_a_b)
            print('removing ', max_mi_pair[1])
            print(self._classes)
            # self._classes.remove(max_mi_pair[1])
            iter += 1

            print("STOP")
            print('classes left', self._classes)

            break

    def mi(self):
        """
        Calculate mutual information (MI) of the whole
        text.
        :return: mi
        """
        # counts = self._class_counts_table
        # n = len(counts[0])

        B = self._B                         # bigram count
        tokens = self._tokens[:8000]        # the 8k tokens
        bg = list(zip(tokens, tokens[1:]))  # the bigrams therein
        bg_uniq_8k = set(bg)                # unique bigrams (set)
        # self._bg_uniq_8k = bg_uniq_8k
        # print(bg_uniq_8k)

        mi = 0
        # Iterate throught the bigram set of the 8k text and
        # lookup and sum the bigram and unigram counts/probabilities
        # previously calculated
        for (left, right) in bg_uniq_8k:
            # for right in self.bigram_counts[left]:
            # print(left, right, self.bigram_counts[left][right])
            c_l_r = self._bigram_counts[left][right]
            c_l = self._unigram_lhs_counts[left]
            c_r = self._unigram_rhs_counts[right]
            # c_l = unig[left]
            # c_r = unig[right]
            mi += c_l_r / float(B) * np.log2(B * c_l_r / ((float(c_l) * c_r)))
            # self.pmi_adj_pairs[left][right] = pmi

        print('done with MI', mi, B)
        return mi

    def merge(self, stop=15):
        """
        Merge those two classes (a, b) which maximizes
        the mi of the text. This efficient algorithm subtracts
        rather than adds, thus minimizing the loss in mi
        resulting from the merge.
        :param stop: stop condition = # classes remaining
        """

        # mi = self.mi()                                # initial mi
        n = k = len(self._classes)                      # initial number of classes
        table_losses = [[None for j in range(k)] for i in range(k)]  # store losses for each pair (a, b)
        s_k = [None for i in range(k)]               # previous sums, @k-1
        history = [None for i in range(k+1)]         # history of merges indexed at k: [(1,3),(2,4),...]
        q_k = [[0 for j in range(k)] for i in range(k)]   # quality history

        # Iterate until stop condition is reached
        while k >= stop:

            # print('k=', k)
            # print('classes remaining')
            # print(self._classes)

            min_loss_a_b = math.inf     # start at infinity, optimize
            min_loss_pair = None        # argmin  (a,b)

            # Select all pairs of classes (i, j)
            # from the set. NB: these are the indices
            # not the words.
            classes = list(self._classes)
            combos = list(itertools.combinations(classes, 2))

            for i, j in combos:  # Get the first class A to merge
                # print('a=', i, 'sk_a=', s_k_a)
                # print('b=', b)
                # if self._class_counts_table[i][j] == 0:  # skip zero count bigrams
                #     # print('bg=0, skipping')
                #     continue

                # First iteration. We need to initialize everything, including sums
                if k == n:
                    s_k[i] = self.sum_k(i)       # the sk subtraction term for A
                    s_k[j] = self.sum_k(j)       # the sk subtraction term for A

                    # Compute sums of columns and rows to be subtracted.
                    # These are all class bigrams containing a and those
                    # containing b. This is the subtraction subterm - the addition
                    # subterm, taking into account the intersection quality terms.

                    # Store quality for k
                    q_k[i][j] = self.quality(i, j)
                    q_k[j][i] = self.quality(j, i)
                    loss_i_j = s_k[i] + s_k[j] - q_k[i][j] - q_k[j][i] - self.add(i, j)
                    # table_losses[i][j] = loss_i_j

                    # Update table of losses (upper triangle) for
                    # pair (i,j)
                    # [[...[(k+1, loss), (k, loss)],... ]]
                    if i < j:
                        # table_losses[i][j].append((iter, loss_i_j))
                        table_losses[i][j] = loss_i_j
                    else:
                        # table_losses[j][i].append((iter, loss_i_j))
                        table_losses[j][i] = loss_i_j

                else:

                    # If a history of the merged (a,b) class exists, we can minimize the loss
                    # from the tabulated (stored) values

                    # Previous loss L_k(i,j), get from table of losses
                    loss_i_j = table_losses[i][j] if j > i else table_losses[j][i]

                if loss_i_j < 0:
                    print('NEGATIVE loss for ', i, j, loss_i_j)

                if loss_i_j < min_loss_a_b:     # update any minimum
                    min_loss_a_b = loss_i_j
                    min_loss_pair = (i, j)

            i = min_loss_pair[0]
            j = min_loss_pair[1]
            a = self._words_10x[i]        # this is the parent word of the class merged to
            b = self._words_10x[j]        # this is the parent word of the class merged to
            # a = self._class_to_word[i]      # this is the parent word of the class merged to
            # b = self._class_to_word[j]      # this is the parent word of the class merged to
            print('merging', a, b, i, j)
            print('min loss for ', a, b, min_loss_a_b)
            print('removing ', b, j)
            # print(self._classes)

            # Update data structs
            self._classes.remove(j)                                 # remove B from the class list
            self._class_to_word[i].update(self._class_to_word[j])   # add all words in class B to class A
            self._word_to_class[b] = i                              # change class for word j
            history[k] = (i, j)                                # update history H_k

            #print("Table before")
            #self.print_class_counts_table(len(self._classes))

            # Update bigram class counts table
            bg_counts_old = [x[:] for x in self._class_counts_table]
            #print('table before', bg_counts_old)
            for l in range(n):
                # print(k, self._class_counts_table[k][i], bg_counts_old[k][j])
                self._class_counts_table[l][i] += bg_counts_old[l][j]  # copy column j to i
                self._class_counts_table[i][l] += bg_counts_old[j][l]  # copy row j to i

            #print('new table', self._class_counts_table)

            # Update unigram class counts
            # ug_counts_lhs_old = dict(self._unigram_class_counts_lhs)
            ug_counts_lhs_old = self._unigram_class_counts_lhs[:]
            ug_counts_rhs_old = self._unigram_class_counts_rhs[:]

            # ug_counts_rhs_old = dict(self._unigram_class_counts_rhs)
            self._unigram_class_counts_lhs[i] += self._unigram_class_counts_lhs[j]
            self._unigram_class_counts_rhs[i] += self._unigram_class_counts_rhs[j]

            # Update q_k (quality) list for next (k-1) iteration
            q_k_old = [x[:] for x in q_k]
            for l in range(n):
                #print('update q-k for l=', l, 'i=', i)
                # print(i, l,self._unigram_class_counts_rhs)
                q_k[i][l] = self.quality_class(i, l, self._class_counts_table, self._unigram_class_counts_lhs, self._unigram_class_counts_rhs)
                q_k[l][i] = self.quality_class(l, i, self._class_counts_table, self._unigram_class_counts_lhs, self._unigram_class_counts_rhs)
                # print(q_k[i][l], q_k[l][i])

            # Update s_k list for k-1 iteration (Trick #4)
            s_k_old = s_k[:]
            for l in range(n):
                s_k[l] = s_k_old[l] - q_k_old[l][i] - q_k_old[i][l] - q_k_old[l][j] - q_k_old[j][l] + q_k[l][i] + q_k[i][l]

            # Update table of losses, all cells affected by the merge of i, j
            for l in range(n):
                for m in range(l + 1, n):

                    # Get quality for potential merges using tables (Trick #4)
                    # print('computing q-old for', l,m)
                    q_k_lm_i_old = self.quality_class((l, m), i, bg_counts_old, ug_counts_lhs_old, ug_counts_rhs_old)
                    q_k_i_lm_old = self.quality_class(i, (l, m), bg_counts_old, ug_counts_lhs_old, ug_counts_rhs_old)
                    q_k_lm_j_old = self.quality_class((l, m), j, bg_counts_old, ug_counts_lhs_old, ug_counts_rhs_old)
                    q_k_j_lm_old = self.quality_class(j, (l, m), bg_counts_old, ug_counts_lhs_old, ug_counts_rhs_old)
                    # print('computing q-new for', l,m)
                    q_k_lm_i = self.quality_class((l, m), i, self._class_counts_table, self._unigram_class_counts_lhs, self._unigram_class_counts_rhs)
                    q_k_i_lm = self.quality_class(i, (l, m), self._class_counts_table, self._unigram_class_counts_lhs, self._unigram_class_counts_rhs)

                    # Current loss = prev_loss - (the merged sums i,j) + (new sums for i,j)
                    loss = table_losses[l][m] - s_k_old[i] + s_k[i] - s_k_old[j] + s_k[j] + \
                                         q_k_lm_i_old + q_k_i_lm_old + q_k_lm_j_old + q_k_j_lm_old - \
                                         q_k_lm_i - q_k_i_lm

                    # print('loss', loss)
                    assert math.isfinite(loss)
                    assert loss == loss
                    table_losses[l][m] = loss


            #print("Table after")
            #self.print_class_counts_table(len(self._classes))

            # Write results
            # with open('out.txt', 'a') as f:
                # f.write('\n' + str(self.words_10x) + '\n')
                # f.write('classes left at k=' + str(k) + '\n')
                # f.write(str(history) + '\n\n')
                # f.write(str(i) + ' just merged with ' + str(j) + '\n')
                # f.write('members of class ' + str(i) + '\n')
                # f.write(str(self.class_to_word[i]) + '\n')
                # f.write('members of class ' + str(j) + '\n')
                # f.write(str(self.class_to_word[j]))
                #f.write(str(s_k) + '\n\n')
                #f.write(str(q_k) + '\n\n')
                #f.write(str(table_losses) + '\n\n')

            # Decrement class counter
            k -= 1

            print('\nclasses left', k)
            #self.print_class_counts_table(len(self._class_counts_table))


            # print(table_losses)

        # self.print_class_counts_table(len(self._class_counts_table))

        # Write final results
        with open('out.txt', 'w') as f:
            f.write('\nHistory of merges\n')
            f.write('k\ti\tj\n')

            for k in range(len(history)-1, -1, -1):
                if history[k] is not None:
                    i, j = history[k]
                    a = self.words_10x[i]
                    b = self.words_10x[j]
                    f.write(str(k) + '\t' + str(a) + '\t' + str(b) + '\n')

            for i in self._classes:
                members = self._class_to_word[i]
                parent_word = self.words_10x[i]
                f.write('Class parent word: ' + parent_word + '\n')
                f.write('Members of class\n')
                for j in members:
                    f.write(j + ' ')
                f.write('\n')

    def add(self, i, j):
        """
        Sum the quality for all class pairs for future (potential)
        merge of (a + b)

        :param i: index of class a
        :param j: index of class b
        :return: s
        """
        s = 0           # starting sum
        T = self._B     # bigram count

        # Left and right counts for a, b
        c_lhs_ab = self.unigram_class_counts_lhs[i] + self.unigram_class_counts_lhs[j]
        c_rhs_ab = self.unigram_class_counts_rhs[i] + self.unigram_class_counts_rhs[j]

        # print('a=', a, 'b=', b, 'c_lhs/rhs_ab', c_lhs_ab, c_rhs_ab)

        # Select a term from the text.
        # NB: I have tried using the set of terms for the sum
        # because using the whole text did not seem to work

        for t in set(self._tokens):      # ITERATE THROUGH ALL TOKENS??
        # for t in set(self._tokens[:8000]):      # ITERATE THROUGH 8000??

            c_ab_t = 0      # count (a + b, t)
            c_t_ab = 0      # count (t, a + b)

            # Sum all words in class A
            for a in self._class_to_word[i]:

                # This is a+b on the left-hand-side (lhs)
                c_ab_t += self._bigram_counts[a][t]

                # This is a+b on the right-hand-side (rhs)
                c_t_ab += self._bigram_counts[t][a]

            # Sum all words in class B
            for b in self._class_to_word[j]:
                c_ab_t += self._bigram_counts[b][t]
                c_t_ab += self._bigram_counts[t][b]

            # Sum the marginal for the term t
            c_rhs_t = self.unigram_rhs_counts[t]
            c_lhs_t = self.unigram_lhs_counts[t]

            # print('c=', c, 'c_ab_c', c_ab_c, 'c_rhc_c', c_rhs_c)

            # Update LHS quality
            if c_ab_t != 0:
                s += c_ab_t / float(T) * np.log2(T * c_ab_t / ((float(c_lhs_ab) * c_rhs_t)))

            # Update RHS quality
            if c_t_ab != 0:
                s += c_t_ab / float(T) * np.log2(T * c_t_ab / ((float(c_lhs_t) * c_rhs_ab)))

        # This is a+b on both sides: (a+b)(a+b)
        c_ab_ab = self._class_counts_table[i][i] + self._class_counts_table[i][j] + \
                  self._class_counts_table[j][i] + self._class_counts_table[j][j]

        # print('c_ab_ab', c_ab_ab)

        if c_ab_ab != 0:
            s += c_ab_ab / float(T) * np.log2(T * c_ab_ab / ((float(c_lhs_ab) * c_rhs_ab)))

        # print('add done', s)

        return s

    def sum_k(self, i):
        """
        Compute sum of quality (mi contribution) sk
        of all pairs (l, r) containing
        Class A or B. We do so by iterating through the text
        These are all class bigrams containing A or B.

        :param a:i index of the class
        :return: sum
        """

        # Sum the quality for all class pairs containing A
        s = 0
        tokens = set(self._tokens)
        # tokens = set(self._tokens[:8000])

        c_a_t = 0       # count (a, t)
        c_t_a = 0       # count (t, a)

        # Sum all words in class A
        for a in self._class_to_word[i]:

            for t in tokens:
                # c_a_t += self._unigram_lhs_counts[a]
                # c_t_a += self._unigram_rhs_counts[a]
                s += self.quality(t, a)
                s += self.quality(a, t)

        # subtract double counted intersection
        s -= self.quality(a, a)

        # print('sum ', s)

        return s

    def quality(self, a, b):
        """
        Calcuate q (mi of the pair), using the
        stored unigram and bigram counts for the
        text.
        :param a: class A (word)
        :param b: class B (word)
        :return: q
        """
        T = self._B
        q = 0
        c_l_r = self._bigram_counts[a][b]
        c_l = self._unigram_lhs_counts[a]
        c_r = self._unigram_rhs_counts[b]

        if c_l_r != 0:
            q = c_l_r / float(T) * np.log2(T * c_l_r / ((float(c_l) * c_r)))

        return q

    def quality_class(self, a, b, bg_counts, ug_counts_lhs, ug_counts_rhs):
        """
        Calculate q for classes. This is used for classes
        in the tables for subsequent iterations.
        :param a: class A (word) or tuple (i,j)
        :param b: class B (word) or tuple (i,j
        :return: q
        """
        T = self._B
        q = 0

        # This is a+b on both sides: (a+b)(a+b)
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            if len(a) == 2 and len(b) == 2:
                i = a[0]
                j = a[1]
                c_l_r = bg_counts[i][i] + bg_counts[i][j] + \
                          bg_counts[j][i] + bg_counts[j][j]
                c_l = ug_counts_lhs[i] + ug_counts_lhs[j]
                c_r = ug_counts_rhs[i] + ug_counts_rhs[j]
            else:
                raise TypeError

        # (i,j), b
        elif isinstance(a, (tuple, list)):
            if len(a) == 2:
                i = a[0]
                j = a[1]
                c_l_r = bg_counts[i][b] + bg_counts[j][b]
                c_l = ug_counts_lhs[i] + ug_counts_lhs[j]
                c_r = ug_counts_rhs[b]
            else:
                raise TypeError

        # a, (i,j)
        elif isinstance(b, (tuple, list)):
            if len(b) == 2:
                i = b[0]
                j = b[1]
                c_l_r = bg_counts[a][i] + bg_counts[a][j]
                c_l = ug_counts_lhs[a]
                c_r = ug_counts_rhs[i] + ug_counts_rhs[j]
            else:
                raise TypeError

        else:
            c_l_r = bg_counts[a][b]
            c_l = ug_counts_lhs[a]
            c_r = ug_counts_rhs[b]

        # print(c_l_r, '/', c_l, '*', c_r)

        if c_l_r != 0:
            # print(a,b)
            # print(self.words_10x)
            # print(bg_counts)
            # print(ug_counts_lhs)
            # print(ug_counts_rhs)
            # print(c_l_r, c_l, c_r)
            q = c_l_r / float(T) * np.log2(T * c_l_r / ((float(c_l) * c_r)))

        try:
            assert math.isfinite(q)
            assert q == q
        except AssertionError:
            print(a, b)
            print(c_l_r, c_l, c_r)
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)  # Fixed format
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]
            print('An error occurred on line {} in statement {}'.format(line, text))
            exit(1)

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
        print(' ', end='')
        for i in range(n):
            print(i, end=' ')
        print('\n')
        for i in range(n):
            print(i, ' ', end='')
            for j in range(n):
                print(self._class_counts_table[i][j], end='  ')
            print()


if __name__ == "__main__":
    # Test
    stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    #stream = io.TextIOWrapper(sys.stdin.buffer, encoding='iso-8859-2')

    tokens = []
    tags = []
    for line in stream:
        if line != '\n':
            # print(line)
            line = line.strip()
            word, tag = line.split('/')
            tokens.append(word)
            tags.append(tag)

    model_words = ClassBasedLM(tokens)
    model_words.merge()
    # model_tags = ClassBasedLM(tags)
    # model_tags.merge()
    # print(model_tags._bigram_counts['NNP'])
    # print(model_tags._bigram_counts['FW'])
    # print(model_tags._bigram_counts['WP'])
    # print(model_tags._bigram_counts['EX'])
    # print(model_tags._bigram_counts['VBN'])
