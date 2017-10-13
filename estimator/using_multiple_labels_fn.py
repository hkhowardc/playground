import tensorflow as tf
import numpy as np


tf.logging.set_verbosity(tf.logging.DEBUG)


def model_fn(features, labels, mode, params, config):
    print(features)

    a = features['a']

    # Negative numbers log always
    a = tf.Print(a, data=[a], message="Feature a", first_n=-1)

    # *** there is no machine learning here, it is just to try using tf.Estimator API
    a_reduce = tf.reduce_mean(a)
    tf.summary.scalar('a', a_reduce)

    c = labels['c']
    d = labels['d']

    # Negative numbers log always
    c = tf.Print(c, data=[c], message="Label c", first_n=-1)
    c = tf.Print(c, data=[d], message="Label d", first_n=-1)

    c_reduce = tf.reduce_mean(c)
    d_reduce = tf.reduce_mean(d)
    tf.summary.scalar('c', c_reduce)
    tf.summary.scalar('d', d_reduce)

    predictions = dict()
    predictions['c'] = c
    predictions['d'] = d

    # dummy loss to minimize
    dummy_var = tf.Variable(0.0, dtype=tf.float32)
    tf.summary.scalar('dummy_var', dummy_var)

    # dummy loss function
    total_loss = (a_reduce - c_reduce - d_reduce) + dummy_var

    global_step = tf.train.get_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=1)
    train_op = optimizer.minimize(total_loss, global_step=global_step)
    tf.summary.scalar('training_total_loss', total_loss)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=total_loss,
        train_op=train_op,
    )


def _create_input_fn():

    def gen_data():
        """It is assumed that every dict yielded from the dictionary represents a single sample.

        :return:
        """
        epoch_size = 10000
        for index in range(epoch_size):
            yield {
                'a': np.full(shape=(1, ), fill_value=(1 if index % 5000 == 0 else 10000), dtype=np.float32),
                'c': np.full(shape=(1, ), fill_value=(4 if index % 5000 == 0 else -3), dtype=np.float32),
                'd': np.full(shape=(1,), fill_value=(9 if index % 5000 == 0 else -123), dtype=np.float32),
            }

    from tensorflow.contrib.learn.python.learn.learn_io import generator_io
    input_fn = generator_io.generator_input_fn(
        gen_data,
        target_key=['c', 'd'],
        batch_size=1,
        shuffle=False,
        num_epochs=1000)

    return input_fn


if __name__ == '__main__':
    print('This is tf_playground.estimator.using_input_fn')

    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir="./using_generator_input_fn/" + ts, params=dict())
    estimator.train(input_fn=_create_input_fn())
