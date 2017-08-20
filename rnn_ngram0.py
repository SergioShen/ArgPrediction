# -*- coding: utf-8 -*-
"""
Author: Sijie Shen

Date: Aug 10, 2017
"""

import tensorflow as tf
import gzip, pickle
import reader

tf.set_random_seed(1911)

plot_path = './cv/rnn-plot-x{}'
model_save_path = "./cv/rnn-model-x{}.ckpt"
combine_test_path = './combine/test_result{}'

unknown_index = reader.UNKNOWN_INDEX

l_r = 1e-3
training_iters = 5001
cv_iters = reader.cv_iters
train_size = 100
test_size = 500

train_step = 10
test_step = 100

# combine test
full_test_steps = reader.full_test_steps
full_test_batch_size = reader.full_test_batch_size

n_inputs = reader.VOCAB_SIZE
n_steps = 2
n_hidden = 300
n_classes = reader.VOCAB_SIZE
n_em = 128

# Tensors for the RNN model
W_em = tf.Variable(tf.random_normal([n_inputs, n_em]), name='W_em')
b_em = tf.Variable(tf.constant(0.1, shape=[n_em, ]), name='b_em')

W = {
    'in': tf.Variable(tf.random_normal([n_em, n_hidden]), name='W_in'),
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]), name='W_out')
}
b = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden, ]), name='b_in'),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]), name='b_out')
}

saver = tf.train.Saver({
    'W_em': W_em,
    'b_em': b_em,
    'W_in': W['in'],
    'W_out': W['out'],
    'b_in': b['in'],
    'b_out': b['out'],
})


def compute_precision(prediction, y, unknown_array):
    """
    Compute precision of input batch
    :param prediction: prediction of RNN model
    :param y: correct reference answer
    :param unknown_array: an array used for mask
    :return: precision of the input batch
    """
    prediction = tf.argmax(prediction, 1)
    correct_pred = tf.cast(tf.equal(prediction, tf.argmax(y, 1)), tf.int32)
    unknown_array = tf.cast(tf.reshape(unknown_array, [-1]), dtype=tf.int32)
    mask = tf.cast(tf.not_equal(tf.cast(prediction, dtype=tf.int32), unknown_array), dtype=tf.int32)
    sum_ = tf.reduce_sum(tf.cast(tf.multiply(correct_pred, mask), dtype=tf.float32))
    num_ = tf.cast(tf.reduce_sum(mask), tf.float32)

    return tf.truediv(sum_, num_), sum_, num_


def compute_recall(prediction, y, unknown_array):
    """
    Compute precision of input batch
    :param prediction: prediction of RNN model
    :param y: correct reference answer
    :param unknown_array: an array used for mask
    :return: recall of the input batch
    """
    prediction = tf.argmax(prediction, 1)
    correct_pred = tf.cast(tf.equal(prediction, tf.argmax(y, 1)), tf.int32)
    unknown_array = tf.cast(tf.reshape(unknown_array, [-1]), dtype=tf.int32)
    mask = tf.cast(tf.not_equal(tf.cast(prediction, dtype=tf.int32), unknown_array), dtype=tf.int32)
    recall = tf.reduce_mean(tf.cast(tf.multiply(correct_pred, mask), dtype=tf.float32))

    return recall


def RNN(X, W, b):
    """
    The RNN model
    :param X: input data
    :param W: weights dictionary
    :param b: biases dictionary
    :return: out put of the model(without softmax layer)
    """
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.nn.relu(tf.matmul(X, W_em) + b_em)
    X_in = tf.matmul(X_in, W['in']) + b['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden])

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden,
                                             forget_bias=1.,
                                             state_is_tuple=True)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell,
                                             X_in,
                                             dtype=tf.float32)
    results = tf.matmul(final_state[1], W['out']) + b['out']

    return results


def combine(rnn_out, ngram_out):
    return tf.add(rnn_out, ngram_out) * 0.5


x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])
unknown_placeholder = tf.placeholder(tf.int32, [None, 1])
ngram_pred = tf.placeholder(tf.float32, [None, n_classes])

rnn_pred = tf.nn.softmax(RNN(x, W, b))

combine_pred = combine(rnn_pred, ngram_pred)

correct_pred = tf.equal(tf.argmax(combine_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

precision, sum_, num_ = compute_precision(combine_pred, y, unknown_placeholder)
recall = compute_recall(combine_pred, y, unknown_placeholder)

print("Model built")

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

init = tf.global_variables_initializer()
sess = tf.InteractiveSession(config=tf_config)

print("Model initialized")

word2index, index2word, data_set = reader.get_data()

print('Data loaded')

for k in range(cv_iters):

    test_result = [[], [], [], [], [], []]  # iter, accuracy, precision, recall, correct_sum, non-unknown_sum

    # initialize current model
    tf.global_variables_initializer().run()

    # load saved RNN model
    saver.restore(sess, model_save_path.format(k))
    print("Model", model_save_path.format(k), "restored")

    # begin testing
    for i in range(full_test_steps):
        # get batch
        batch_x, batch_y, unknown_array = reader.minibatch_combine(data_set, batch_size=full_test_batch_size, cv_index=k,
                                                                   step_index=i)
        batch_x = batch_x.reshape([-1, n_steps, n_inputs])
        ngram_pred_ = reader.get_ngram_pred(batch_x, index2word, k)
        test_result[0].append(i)

        test_accuracy = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, unknown_placeholder: unknown_array,
                                                      ngram_pred: ngram_pred_})
        test_result[1].append(test_accuracy)
        print("Batch %d accuracy: %.2f" % (i, test_accuracy))

        test_precision = sess.run(precision,
                                  feed_dict={x: batch_x, y: batch_y, unknown_placeholder: unknown_array,
                                             ngram_pred: ngram_pred_})
        test_result[2].append(test_precision)
        print("Batch %d precision: %.2f" % (i, test_precision))

        test_recall = sess.run(recall, feed_dict={x: batch_x, y: batch_y, unknown_placeholder: unknown_array,
                                                  ngram_pred: ngram_pred_})
        test_result[3].append(test_recall)
        print("Batch %d recall: %.2f" % (i, test_recall))

        test_correct_num = sess.run(sum_, feed_dict={x: batch_x, y: batch_y, unknown_placeholder: unknown_array,
                                                     ngram_pred: ngram_pred_})
        test_result[4].append(test_correct_num)
        print("Batch %d correct predicted number: %.0f" % (i, test_correct_num))

        test_non_unknown_num = sess.run(num_, feed_dict={x: batch_x, y: batch_y, unknown_placeholder: unknown_array,
                                                         ngram_pred: ngram_pred_})
        test_result[5].append(test_non_unknown_num)
        print("Batch %d non-unknown number: %.0f" % (i, test_non_unknown_num))

    # save current test data
    f = gzip.open(combine_test_path.format(k), 'wb')
    pickle.dump((test_result,), f)
    f.close()
    print("Test result saved at", combine_test_path.format(k))
