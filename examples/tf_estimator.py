import argparse
import sys
import os
import shutil
from datetime import datetime

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def create_input_fn(images, labels, mini_batch_size, num_epochs, shuffle=True):

    # def input_fn():
    #     return tf.train.shuffle_batch([tf.constant(mnist_datasets.images), tf.constant(mnist_datasets.labels)],
    #                                   batch_size=batch_size, capacity=batch_size * 100, min_after_dequeue=batch_size * 99, enqueue_many=True)
    # return input_fn

    import numpy as np
    numpy_input_fn = tf.estimator.inputs.numpy_input_fn
    return numpy_input_fn(x={"x": images},
                          y=np.array(labels, dtype=np.int32),
                          batch_size=mini_batch_size,
                          num_epochs=num_epochs,
                          shuffle=shuffle)


def model_fn(features, labels, mode, params, config):

    with tf.device("/gpu:0"):

        # Feature: Placeholders to take in features
        # x = tf.placeholder(tf.float32, [None, 784])
        x = features["x"]

        # Label
        # label_y = tf.placeholder(tf.float32, [None, 10])
        label_y = labels

        # global step
        # global_step = tf.Variable(0,
        #                           trainable=False,
        #                           # collections=[tf.GraphKeys.GLOBAL_STEP],
        #                           name="global_step",
        #                           dtype=tf.int32)
        global_step = tf.train.get_global_step()

        with tf.variable_scope("algo"):
            # Variables
            W = tf.Variable(tf.zeros([784, 10]))
            b = tf.Variable(tf.zeros([10]))

            # Operations
            y = tf.matmul(x, W) + b
        tf.summary.histogram("W", W)
        tf.summary.histogram("b", b)

        # Predictions
        with tf.variable_scope("predictions"):
            predictions = tf.argmax(y, 1)
            corrects = tf.equal(predictions, tf.argmax(label_y, 1))
            accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

        # loss
        with tf.variable_scope("losses"):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=label_y, logits=y))
            loss = cross_entropy
        tf.summary.scalar("xentropy_loss", cross_entropy)
        tf.summary.scalar("total_loss", loss)

        # optimizer
        with tf.variable_scope("optimizer"):
            train_op = tf.train.GradientDescentOptimizer(0.5).minimize(loss, global_step=global_step)

        # Collect summaries
        # summary_op = tf.summary.merge_all()

        eval_metric_ops = dict()
        eval_metric_ops["accuracy"] = tf.metrics.accuracy(labels=tf.argmax(label_y, 1),
                                                          predictions=predictions)

    # return x, label_y, predictions, accuracy, loss, summary_op, train_op, global_step
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
    )


def train():
    tf.logging.set_verbosity(tf.logging.DEBUG)

    if os.path.exists(FLAGS.model_dir):
        shutil.move(FLAGS.model_dir, FLAGS.model_dir + "." + datetime.now().strftime('%Y%m%d_%H%M'))

    model_dir = FLAGS.model_dir
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, validation_size=5000)

    print(mnist.train)

    batch_size = 100
    num_epoch = 5
    steps_per_epoch = mnist.train.images.shape[0] / batch_size
    eval_per_steps = 100
    summary_per_steps = 25

    # model save/summary frequency
    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(
        model_dir=model_dir,
        save_summary_steps=summary_per_steps,
        save_checkpoints_steps=eval_per_steps,
        log_step_count_steps=eval_per_steps,
        keep_checkpoint_max=5,
    )

    model_params = dict()

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        params=model_params,
        config=run_config,
    )

    # Eval data is an in-memory constant here.
    def test_data_input_fn():
        return {"x": tf.constant(mnist.test.images)}, tf.constant(mnist.test.labels)

    def experiment_fn(run_config, params):

        # Define the experiment
        experiment = tf.contrib.learn.Experiment(
            estimator=estimator,  # Estimator
            train_input_fn=create_input_fn(mnist.train.images, mnist.train.labels, mini_batch_size=batch_size, num_epochs=num_epoch),  # First-class function
            eval_input_fn=create_input_fn(mnist.validation.images, mnist.validation.labels, mini_batch_size=batch_size, num_epochs=1),  # First-class function
            # eval_input_fn=eval_data_input_fn,
            train_steps=steps_per_epoch * num_epoch,  # Minibatch steps
            min_eval_frequency=eval_per_steps / 5,  # Eval frequency
            train_monitors=[],  # Hooks for training
            eval_hooks=[],  # Hooks for evaluation
            eval_steps=None,  # Use evaluation feeder until its empty
        )
        return experiment

    # Set the run_config and the directory to save the model and stats
    from tensorflow.contrib.learn import learn_runner
    learn_runner.run(
        experiment_fn=experiment_fn,  # First-class function
        schedule="train_and_evaluate",  # What to run
        run_config=run_config,
    )


def main(_):
    if FLAGS.mode == 'train':
        train()
    else:
        # test()
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/opt/data/dataset/mnist/',
                      help='Directory for storing input data')
    parser.add_argument('--model_dir', type=str, default="/opt/data/models/examples/tf_estimator",
                      help='Directory for storing model')
    parser.add_argument('--mode', type=str, default='train',
                      help='train / predict')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
