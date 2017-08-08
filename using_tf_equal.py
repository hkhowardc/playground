import tensorflow as tf
import numpy as np


def test_tf_equal_for_rnn_output():

    rnn_length = 3
    ph_x = tf.placeholder(dtype=tf.int32, shape=(None, rnn_length))
    ph_y = tf.placeholder(dtype=tf.int32, shape=(None, rnn_length))

    original_shape = tf.shape(input=ph_x)

    x_flatten = tf.reshape(tensor=ph_x, shape=(-1, ))
    y_flatten = tf.reshape(tensor=ph_y, shape=(-1, ))

    corrects = tf.cast(tf.equal(x_flatten, y_flatten), tf.float32)

    rnn_corrects = tf.reshape(tensor=corrects, shape=original_shape)
    sent_corrects = tf.floor(x=tf.reduce_mean(input_tensor=rnn_corrects, axis=1))

    with tf.Session() as sess:
        test_data_x = np.array([
            [11, 12, 13],
            [14, 15, 16],
            [17, 18, 19],

            [21, 22, 23],
            [24, 25, 26],
            [27, 28, 29],

            [31, 32, 33],
            [34, 35, 36],
            [37, 38, 39],

            [41, 42, 43],
            [44, 45, 46],
            [47, 48, 49],

            [51, 52, 53],
            [54, 55, 56],
            [57, 58, 59],
        ], dtype=np.int32)

        test_data_y = np.array([
            [0, 0, 0],
            [14, 15, 16],
            [0, 0, 19],

            [21, 0, 0],
            [24, 25, 26],
            [0, 0, 29],

            [31, 32, 33],
            [0, 35, 0],
            [37, 38, 39],

            [0, 42, 43],
            [44, 45, 46],
            [47, 0, 49],

            [51, 52, 0],
            [54, 55, 56],
            [57, 58, 59],
        ], dtype=np.int32)

        expected_rnn_results = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 1],

            [1, 0, 0],
            [1, 1, 1],
            [0, 0, 1],

            [1, 1, 1],
            [0, 1, 0],
            [1, 1, 1],

            [0, 1, 1],
            [1, 1, 1],
            [1, 0, 1],

            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
        ], dtype=np.int32)

        expected_sent_results = np.array([
            0,
            1,
            0,

            0,
            1,
            0,

            1,
            0,
            1,

            0,
            1,
            0,

            0,
            1,
            1
        ], dtype=np.int32)

        sess.run(tf.global_variables_initializer())

        rnn_results, sent_results = sess.run([rnn_corrects, sent_corrects], feed_dict={ph_x: test_data_x,
                                                                                       ph_y: test_data_y})

        assert np.array_equiv(expected_rnn_results, rnn_results)
        assert np.array_equiv(expected_sent_results, sent_results)

        print('rnn_results: {}'.format(rnn_results))
        print('sent_results: {}'.format(sent_results))

if __name__ == '__main__':
    test_tf_equal_for_rnn_output()
