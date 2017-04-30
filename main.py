#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project: NLP
Created on 23.12.16
@author: Eric Lief
"""

from smoothedLM import *
from entropy import *
import sys
import io
import pickle

# This main method is used to facilitate the language modeling
# and data collection and analysis processes. The makefile can
# be used to automate the process.


# Save model
def save(self, model, fileOut="model.txt"):
    with open(fileOut, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":

    # Read textual data from stdin via buffer, with proper encoding
    stream = io.TextIOWrapper(sys.stdin.buffer, encoding='iso-8859-2')

    # Tokenize text from stdin
    tokens = []
    for line in stream:
        if line != '\n':
            tokens.append(line.strip())

    if len(sys.argv) < 3:

        print("""
        *******************************************
        Usage:
        python3 main.py [--option] [output file]\n

        options:
        --entropy:          train model
        --stats             text statistics
        --randomchars:      messup chars
        --randomwords:      messup words
        --smooth:           smoothed language model
        *******************************************
        """)

        sys.exit()

    elif sys.argv[1] == "--entropy":

        # Compute conditional entropy of texts and write results
        print("getting entropy")
        model = Entropy(tokens)
        print("Conditional entropy of text: ", model.h2())
        try:
            with open(sys.argv[2], 'w') as f:
                    f.write("Entropy of original text: {}\n".format(model.h2()))
                    f.write("Perplexity of original text: {}\n".format(model.g2()))
        except FileNotFoundError:
            sys.stderr.write("File not found")

    elif sys.argv[1] == "--stats":

        # Compute and write basic statistics of both texts
        with open(sys.argv[2], 'w') as f:
            model = Entropy(tokens)
            wc = model.wordCount()
            chars = model.charCount()
            f.write("words: {}\n".format(wc))
            f.write("characters: {}\n".format(chars))
            f.write("chars/word: {}\n".format(chars/wc))
            f.write("vocab size: {}\n".format(model._model.V))
            f.write("total bigrams: {}\n".format(len(model._model.bigrams)))
            f.write("most frequent words:\n")
            words = model.mostFrequentWords()
            f.write("total words" + str(len(words)) + '\n')
            for w, c in words[:20]:
                f.write("{}\t{}\n".format(w, c))
            chars = model.mostFrequentChars()
            f.write("total chars" + str(len(chars)))
            f.write("Character frequency\n")
            for k, v in chars:
                f.write(k + '\t' + str(v) + '\n')
            cnt = 0
            for w in model.hapax():
                print(w)
                cnt += 1
            f.write("Number of words occurring only once (hapax legomena): {}\n".format(cnt))

    elif sys.argv[1] == "--randomchars":

        # Messup characters with probability x and write results
        with open(sys.argv[2], 'w') as f:
            model = Entropy(list(tokens))
            f.write("Entropy of original text: {}\n".format(model.h2()))
            # f.write("Number of words occurring only once (hapax legomena) before randomization: {}\n".format(
            #     len(model.hapax())))
            f.write("Number of bigrams occurring only once (hapax legomena) before randomization: {}\n".format(
                len(model.uniqueBigrams())))
            probs = [.05, .01, .001, .0001, .00001]
            cnt = 0
            for p in probs:
                f.write("Randomizing characters with a likelihood of {}\n".format(p))
                results = []
                hapaxCount = []     # used to count words occurring only once
                for i in range(10):
                    newTokens = model.messupChars(p)
                    newModel = Entropy(newTokens)
                    h = newModel.h2()
                    results.append(h)
                    f.write("Rep {}: {}\n".format(i, h))
                    hapaxCount.append(len(newModel.uniqueBigrams()))
                    #hapaxCount.append(len(newModel.hapax()))

                # Write statistics
                f.write("min: {}\n".format(min(results)))
                f.write("max: {}\n".format(max(results)))
                f.write("mean: {}\n".format(sum(results) / float(len(results))))
                f.write("Average of words occurring only once (hapax legomena): {}\n".format((sum(hapaxCount) / float(len(hapaxCount)))))

    elif sys.argv[1] == "--randomwords":

        # Messup words with probability x and write results
        with open(sys.argv[2], 'w') as f:
            model = Entropy(list(tokens))
            f.write("Entropy of original text: {}\n".format(model.h2()))
            f.write("Number of bigrams occurring only once (hapax legomena) before randomization: {}\n".format(
                len(model.uniqueBigrams())))

            probs = [.05, .01, .001, .0001, .00001]
            for p in probs:
                f.write("Randomizing words with a likelihood of {}\n".format(p))
                results = []
                hapaxCount = []  # used to count words occurring only once
                bgCount = []    # used to count bigrams occurring only once
                for i in range(10):
                    newTokens = model.messupWords(p)
                    newModel = Entropy(newTokens)
                    h = newModel.h2()
                    results.append(h)
                    f.write("Rep {}: {}\n".format(i, h))

                    # count mean bigrams: total and unique
                    hapaxCount.append(len(newModel.uniqueBigrams()))
                    bgCount.append(len(newModel._model.bigrams))
                f.write("min: {}\n".format(min(results)))
                f.write("max: {}\n".format(max(results)))
                f.write("mean: {}\n".format(sum(results) / float(len(results))))
                f.write("Average of bigrams occurring only once (hapax legomena): {}\n".format(
                    (sum(hapaxCount) / float(len(hapaxCount)))))
                f.write("Average of total bigrams: {}\n".format(
                    (sum(bgCount) / float(len(bgCount)))))

    elif sys.argv[1] == "--smooth":
        with open(sys.argv[2], 'w') as f:

            # Construct data sets
            test = list(tokens[-20000:])
            tokens = list(tokens[:-20000])
            heldout = list(tokens[-40000:])
            train = list(tokens[:-40000])

            model = SmoothedLM(list(train), list(heldout), list(test))                # make model and compute parameters for heldout data

            smoothedL = model.EM([.25, .25, .25, .25], .0001)       # new smoothed lambdas

            # Save model
            save(smoothedL, "smoothedLM.bin")


            #Write results (lambdas and initial cross-entropy)
            f.write("smoothed lambdas: \n")
            f.write("L0 {}, L1 {} L2 {} L3 {}\n".format(smoothedL[0], smoothedL[1], smoothedL[2], smoothedL[3]))
            H = model.crossEntropy(smoothedL[:])
            f.write("Original cross entropy = {}\n".format(H))


            # Boost 100x% to L3
            for x in [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99]:
                newL = model.boostL3(x, smoothedL[:])
                H = model.crossEntropy(newL[:])
                f.write("original lambdas: {}\n".format(smoothedL))
                f.write("Boosting {}%\n".format(x))
                f.write("new lambdas: {}\tcross entropy = {}\n".format(newL, H))

            # add x% to L3
            for x in [.9, .8, .7, .6, .5, .4, .3, .2, .1, 0]:
                newL = model.discountL3(x, smoothedL[:])
                H = model.crossEntropy(newL[:])
                f.write("original lambdas: {}\n".format(smoothedL))
                f.write("Boosting {}%\n".format(x))
                f.write("new lambdas: {}\tcross entropy = {}\n".format(newL, H))

            f.write("Coverage graph: " + str(model.coverageGraph()))
