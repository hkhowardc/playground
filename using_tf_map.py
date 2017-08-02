import tensorflow as tf
import numpy as np


def try_tf_map_fn():
    with tf.Session() as sess:
        depth = 5

        ph_flags = tf.placeholder(shape=(10, ), dtype=tf.bool, name="ph_flags")
        ph_words = tf.placeholder(shape=(10, depth), dtype=tf.int32, name="ph_words")

        def switch_ops(flag_and_word):
            def do_true():
                return tf.add(flag_and_word[1], 9000)

            def do_false():
                return tf.add(flag_and_word[1], 8000000)

            return tf.cond(flag_and_word[0], true_fn=do_true, false_fn=do_false)

        t_results = tf.map_fn(switch_ops, (ph_flags, ph_words), dtype=tf.int32)

        print('Using tf.map_fn() to apply if-then-else logic...')

        test_data_flags = np.array([True, True, True, True, True, False, False, False, False, False], dtype=np.bool)

        test_data_words = np.array([[i] * depth for i in range(10, 20)], dtype=np.int32)

        feed_dict = {
            ph_flags: test_data_flags,
            ph_words: test_data_words,
        }

        results = sess.run(t_results, feed_dict=feed_dict)

        print('tf.map_fn() result: \n{}'.format(results))


def try_tf_map_fn_3d():
    with tf.Session() as sess:
        depth = 5

        ph_flags = tf.placeholder(shape=(None, 10, ), dtype=tf.bool, name="ph_flags")
        ph_words = tf.placeholder(shape=(None, 10, depth), dtype=tf.int32, name="ph_words")

        count_do_batch = tf.Variable(initial_value=0, name="count_do_batch")
        count_do_true = tf.Variable(initial_value=0, name="count_do_true")
        count_do_false = tf.Variable(initial_value=0, name="count_do_false")

        def convert_batch(batch_flag_and_words):

            def switch_ops(flag_and_words):
                def do_true():
                    increment = tf.assign_add(count_do_true, 1)

                    with tf.control_dependencies([increment]):
                        return tf.add(flag_and_words[1], 9000)

                def do_false():
                    increment = tf.assign_add(count_do_false, 1)

                    with tf.control_dependencies([increment]):
                        return tf.add(flag_and_words[1], 8000000)

                return tf.cond(flag_and_words[0], true_fn=do_true, false_fn=do_false)

            increment_batch = tf.assign_add(count_do_batch, 1)

            with tf.control_dependencies([increment_batch]):
                return tf.map_fn(switch_ops, batch_flag_and_words, dtype=tf.int32)

        t_results = tf.map_fn(convert_batch, (ph_flags, ph_words), dtype=tf.int32)

        print('Using Nested tf.map_fn() to apply if-then-else logic...')

        test_data_flags = np.array([
            [True, False, True, False, True, False, True, False, True, False],
            [False, False, False, True, True, True, False, False, True, True],
            [False, False, False, False, False, True, True, True, True, True],
            [True, True, True, True, True, False, False, False, False, False],
        ])

        test_data_words = np.array([
            [[i] * depth for i in range(10, 20)],
            [[i] * depth for i in range(30, 40)],
            [[i] * depth for i in range(50, 60)],
            [[i] * depth for i in range(70, 80)],
        ], dtype=np.int32)

        feed_dict = {
            ph_flags: test_data_flags,
            ph_words: test_data_words,
        }

        sess.run(tf.global_variables_initializer())

        results, do_batch_count, do_true_count, do_false_count = \
            sess.run([t_results, count_do_batch, count_do_true, count_do_false], feed_dict=feed_dict)

        print('Nested tf.map_fn() result: \n{}'.format(results))
        print('do_batch() called: {}'.format(do_batch_count))
        print('do_true() called: {}'.format(do_true_count))
        print('do_false() called: {}'.format(do_false_count))

if __name__ == '__main__':
    try_tf_map_fn()
    try_tf_map_fn_3d()
