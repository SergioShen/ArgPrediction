# -*- coding: utf-8 -*-
"""
Author: Sijie Shen

Date: Aug 10, 2017
"""

import numpy
import random
import input_data
import cmd

DATA_SIZE = input_data.DATA_SIZE
VOCAB_SIZE = input_data.VOCAB_SIZE
cv_iters = 10  # k fold cross validation
UNKNOWN_INDEX = input_data.UNKNOWN_INDEX

full_test_steps = 83
full_test_batch_size = 500


def get_data():
    """
    Read data and convert word string to int value
    :return: word2index dictionary, index2word dictionary, data set
    """
    words = input_data.get_data()

    word2index, index2word = input_data.get_vocab()

    variables = [word2index[words[i]] for i in range(0, DATA_SIZE * 3, 3)]
    functions = [word2index[words[i]] for i in range(1, DATA_SIZE * 3, 3)]
    arguments = [word2index[words[i]] for i in range(2, DATA_SIZE * 3, 3)]
    var_fun = [[variables[i], functions[i]] for i in range(len(variables))]

    print("Data set length:", len(variables))

    return word2index, index2word, [var_fun, arguments]


def minibatch(data, batch_size=32):
    """
    Get a random mini batch from all the data
    :param data: data set returned for get_data()
    :param batch_size: batch size
    :return: a mini batch
    """
    sample = random.sample(range(len(data[0])), batch_size)
    batch = [[], []]
    for s in sample:
        # Generate this batch, one-hot
        var_tmp = numpy.zeros((VOCAB_SIZE,))
        var_tmp[data[0][s][0]] = 1
        fun_tmp = numpy.zeros((VOCAB_SIZE,))
        fun_tmp[data[0][s][1]] = 1
        arg_tmp = numpy.zeros((VOCAB_SIZE,))
        arg_tmp[data[1][s]] = 1
        batch[0].append(numpy.asarray([var_tmp, fun_tmp]))
        batch[1].append(numpy.asarray(arg_tmp))

    batch[0] = numpy.asarray(batch[0])
    batch[1] = numpy.asarray(batch[1])

    return batch


def minibatch_cv(data, batch_size, cv_index, training):
    """
    Get a random mini batch from training data or test data
    :param data: data set returned for get_data()
    :param batch_size: batch size
    :param cv_index: index of fold
    :param training: is this batch for training or not?
    :return: a mini batch
    """

    # get the position of training data set
    total_date_length = len(data[0])
    train_data_length = total_date_length / cv_iters
    train_data_begin = int(train_data_length * cv_index)
    train_data_end = int(train_data_begin + train_data_length)

    if training is True:
        sample = random.sample(range(train_data_begin, train_data_end), batch_size)
    else:
        train_range = list(range(train_data_begin)) + list(range(train_data_end, total_date_length))
        sample = random.sample(train_range, batch_size)

    batch = [[], [], []]
    for s in sample:
        # Generate this batch, one-hot
        var_tmp = numpy.zeros((VOCAB_SIZE,))
        var_tmp[data[0][s][0]] = 1
        fun_tmp = numpy.zeros((VOCAB_SIZE,))
        fun_tmp[data[0][s][1]] = 1
        arg_tmp = numpy.zeros((VOCAB_SIZE,))
        arg_tmp[data[1][s]] = 1
        batch[0].append(numpy.asarray([var_tmp, fun_tmp]))
        batch[1].append(numpy.asarray(arg_tmp))
        batch[2].append(numpy.asarray([UNKNOWN_INDEX]))

    batch[0] = numpy.asarray(batch[0])
    batch[1] = numpy.asarray(batch[1])
    batch[2] = numpy.asarray(batch[2])

    return batch


def minibatch_combine(data, batch_size, cv_index, step_index):
    """
    Get a minibatch for combine model test
    :param data: data set returned for get_data()
    :param batch_size: batch size
    :param cv_index: index of fold
    :param step_index: step index of current fold
    :return: a mini batch
    """
    # get the position of training data set
    total_date_length = len(data[0])
    test_data_length = total_date_length / cv_iters
    test_data_begin = int(test_data_length * cv_index)
    test_data_end = int(test_data_begin + test_data_length)

    if step_index == 0:
        print("Use data from", test_data_begin, "to", test_data_end)

    sample = range(test_data_begin + step_index * full_test_batch_size,
                   test_data_begin + (step_index + 1) * full_test_batch_size)

    batch = [[], [], []]
    for s in sample:
        # Generate this batch, one-hot
        var_tmp = numpy.zeros((VOCAB_SIZE,))
        var_tmp[data[0][s][0]] = 1
        fun_tmp = numpy.zeros((VOCAB_SIZE,))
        fun_tmp[data[0][s][1]] = 1
        arg_tmp = numpy.zeros((VOCAB_SIZE,))
        arg_tmp[data[1][s]] = 1
        batch[0].append(numpy.asarray([var_tmp, fun_tmp]))
        batch[1].append(numpy.asarray(arg_tmp))
        batch[2].append(numpy.asarray([UNKNOWN_INDEX]))

    batch[0] = numpy.asarray(batch[0])
    batch[1] = numpy.asarray(batch[1])
    batch[2] = numpy.asarray(batch[2])

    return batch


def get_ngram_pred(batch, index2word, cv_index):
    """
    Get N-Gram prediction of input batch
    :param batch: current test batch
    :param index2word: dictionary of index-word
    :param cv_index: index of fold
    :return: a numpy array of N-Gram model result
    """
    words = numpy.reshape(batch, [-1, VOCAB_SIZE])
    words = numpy.argmax(words, 1)  # get the index of input words
    results = []

    print('Fetching 3Gram pred...', end='')

    for i in range(0, len(words), 2):
        input_a = index2word[words[i]]
        input_b = index2word[words[i + 1]]
        result = cmd.get_prob(input_a, input_b, cv_index)
        results.append(numpy.asarray(result))
    print(' %d' % len(results))
    results = numpy.asarray(results)

    return results


if __name__ == '__main__':
    word2index, index2word, data_test = get_data()

    print(len(data_test[0]))

    for cvindex in range(cv_iters):

        total_date_length = len(data_test[0])
        train_data_length = total_date_length / cv_iters
        train_data_begin = int(train_data_length * cvindex)
        train_data_end = int(train_data_begin + train_data_length)

        f = open("./combine/test_set_{}.txt".format(cvindex), "w")

        for i in range(train_data_begin, train_data_end):
            f.write(index2word[data_test[0][i][0]])
            f.write(" ")
            f.write(index2word[data_test[0][i][1]])
            f.write(" ")
            f.write(index2word[data_test[1][i]])
            f.write("\n")

    for cvindex in range(cv_iters):

        total_date_length = len(data_test[0])
        train_data_length = total_date_length / cv_iters
        train_data_begin = int(train_data_length * cvindex)
        train_data_end = int(train_data_begin + train_data_length)

        f = open("./combine/train_set_{}.txt".format(cvindex), "w")

        for i in list(range(0, train_data_begin)) + list(range(train_data_end, total_date_length)):
            f.write(index2word[data_test[0][i][0]])
            f.write(" ")
            f.write(index2word[data_test[0][i][1]])
            f.write(" ")
            f.write(index2word[data_test[1][i]])
            f.write("\n")
