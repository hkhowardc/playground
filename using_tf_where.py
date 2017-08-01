import tensorflow as tf
import numpy as np


def test_tf_where():
    with tf.Session() as sess:
        ph_flags = tf.placeholder(shape=(10, ), dtype=tf.bool)
        ph_words = tf.placeholder(shape=(10, 100), dtype=tf.int32)

        # def switch_ops(flag_and_word):
        #     def opt_one():
        #         return tf.add(flag_and_word[1], 100)
        #
        #     def opt_two():
        #         return tf.multiply(flag_and_word[1], 100)
        #
        #     t_result = tf.where(flag_and_word[0], x=opt_one(), y=opt_two())
        #     return t_result
        #
        # t_results = tf.map_fn(switch_ops, (ph_flags, ph_words))

        def opt_one(val):
            return tf.add(val, 100)

        def opt_two(val):
            return tf.multiply(val, 100000)

        # flags_list = tf.unstack(flags)
        # word_list = tf.unstack(words)

        t_results = tf.where(condition=ph_flags, x=opt_one(ph_words), y=opt_two(ph_words))

        # test_data_flags = np.array([
        #     [True, True, True, True, True, False, False, False, False, False],
        #     [True, False, True, False, True, False, True, False, True, False],
        #     [True, True, True, False, False, False, True, True, False, False],
        # ], dtype=np.bool)
        test_data_flags = np.array([True, False, True, False, True, False, True, False, True, False], dtype=np.bool)

        # test_data_words = np.array([
        #     [[i] * 100 for i in range(1, 11)],
        #     [[i * 10] * 100 for i in range(1, 11)],
        #     [[i * 100] * 100 for i in range(1, 11)],
        # ])
        test_data_words = np.array([[i] * 100 for i in range(1, 11)])

        results = sess.run(t_results, feed_dict={ph_flags: test_data_flags, ph_words: test_data_words})

        print(results)

if __name__ == '__main__':
    test_tf_where()
