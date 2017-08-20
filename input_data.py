# -*- coding: utf-8 -*-
"""
Author: Sijie Shen

Date: Aug 10, 2017
"""

DATA_PATH = "c0810.txt"
DICT_PATH = "dictionary0810.txt"

DATA_SIZE = 415612  # Total length of the data set
VOCAB_SIZE = 28310  # Total length of the vocabulary(UNKNOWN included)

UNKNOWN_INDEX = 25656  # The index of UNKNOWN in vocabulary


def get_data(path=DATA_PATH):
    """
    Get all the words in the data set
    :param path: path of the data set file
    :return: list of all the input data in order
    """
    words = []
    f = open(path)
    lines = f.readlines()
    for line in lines:
        tokens = line.split("\t")
        words.append(tokens[1].strip().replace("\n", ""))
        words.append(tokens[2].strip().replace("\n", ""))
        words.append(tokens[3].strip().replace("\n", ""))

    return words


def get_vocab(path=DICT_PATH):
    """
    Generate the vocabulary and index-word dictionary
    :param path: path of the data set file
    :return: 2 dictionaries that convert word to index and index to word
    """
    print('Generating the vocabulary...')
    words = get_data()
    print('Total length of words:', len(words))

    words = []

    f = open(path)
    lines = f.readlines()
    for line in lines:
        words.append(line.strip().replace("\n", ""))

    word2index = {words[i]: i for i in range(len(words))}
    index2word = {v: k for k, v in word2index.items()}
    print('Total length of the vocabulary:', len(words))

    return word2index, index2word


if __name__ == "__main__":
    word2index, index2word = get_vocab()
    for i in range(2000):
        x = input()
        print(word2index[x])
