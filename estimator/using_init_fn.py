import tensorflow as tf
import numpy as np


tf.logging.set_verbosity(tf.logging.DEBUG)


def get_init_int():
    # return 87
    return 73843

def model_fn(features, labels, mode, params, config):
    ph_init_int = tf.placeholder(shape=(), dtype=tf.int32, name="ph_init_int")

    with tf.variable_scope("wemb"):
        init_int = tf.get_variable("init_int", initializer=ph_init_int, trainable=False)

    tf.summary.scalar("init_int", init_int)

    print(features)

    a = features['a']
    b = features['b']

    # Negative numbers log always
    a = tf.Print(a, data=[a], message="Feature a", first_n=-1)
    b = tf.Print(b, data=[b], message="Feature b", first_n=-1)

    # *** there is no machine learning here, it is just to try using tf.Estimator API
    a = tf.reduce_mean(a, axis=1)
    b = tf.reduce_mean(b, axis=1)

    predictions = dict()
    predictions['a'] = a
    predictions['b'] = b

    tf.summary.scalar('a', tf.reduce_sum(a))
    tf.summary.scalar('b', tf.reduce_sum(b))

    # dummy loss to minimize
    dummy_var = tf.Variable(0.0, dtype=tf.float32)

    tf.summary.scalar('dummy_var', dummy_var)

    total_loss = tf.reduce_sum(a - b) + dummy_var
    global_step = tf.train.get_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=1)
    train_op = optimizer.minimize(total_loss, global_step=global_step)
    tf.summary.scalar('training_total_loss', total_loss)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=total_loss,
        train_op=train_op,
        scaffold=tf.train.Scaffold(init_feed_dict={ph_init_int: get_init_int()})
    )


def _create_input_fn():
    a = np.array([[i, 0, 0, 0] for i in range(1, 101)], dtype=np.float32)

    b = np.array([[i, 0, 0, 0] for i in range(101, 201)], dtype=np.float32)

    c = np.array([i for i in range(201, 301)], dtype=np.float32)

    data = {
        'a': a,
        'b': b,
    }

    labels = c

    # able to duplicate data according to num_epochs
    numpy_input_fn = tf.estimator.inputs.numpy_input_fn
    return numpy_input_fn(x=data,
                          y=labels,
                          batch_size=10,
                          num_epochs=1000,
                          shuffle=True)


if __name__ == '__main__':
    print('This is tf_playground.estimator.using_input_fn')

    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir="./using_init_fn", params=dict())
    estimator.train(input_fn=_create_input_fn())
