import argparse
import sys
import os
import shutil
from datetime import datetime

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def build_model():

    with tf.device("/gpu:0"):

        # Feature: Placeholders to take in features
        x = tf.placeholder(tf.float32, [None, 784])

        # Label
        label_y = tf.placeholder(tf.float32, [None, 10])

        # global step
        global_step = tf.Variable(0,
                                  trainable=False,
                                  # collections=[tf.GraphKeys.GLOBAL_STEP],
                                  name="global_step",
                                  dtype=tf.int32)

        with tf.variable_scope("algo"):
            # Variables
            W = tf.Variable(tf.zeros([784, 10]), name="W")
            b = tf.Variable(tf.zeros([10]), name="b")

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
        summary_op = tf.summary.merge_all()

    return x, label_y, predictions, accuracy, loss, summary_op, train_op, global_step


def main(_):
    if FLAGS.mode == 'train':
        train()
    else:
        test()


def test():

    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        # Build Graph
        ph_x, label_y, predictions, accuracy, loss, summary_op, train_op, global_step = build_model()

        # Initialize
        sess.run(tf.global_variables_initializer())

        # Determine the model weights from checkpoint
        model_dir = FLAGS.model_dir
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir=model_dir)
        mdl_ckpt_fpath = checkpoint.model_checkpoint_path

        # Model Saver
        saver = tf.train.Saver(max_to_keep=10)
        saver.restore(sess=sess, save_path=mdl_ckpt_fpath)

        # Import data
        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, validation_size=5000)

        # last global step
        final_step = tf.train.global_step(sess, global_step)

        # Training Completed, Final Eval with Validation Set
        eval_pred, eval_acc, eval_loss, eval_summary = sess.run(
            [predictions, accuracy, loss, summary_op],
            feed_dict={ph_x: mnist.validation.images, label_y: mnist.validation.labels})
        print('[Eval][%s] accuracy: %s, loss: %s' % (
            final_step, eval_acc, eval_loss))


def train():

    if os.path.exists(FLAGS.model_dir):
        shutil.move(FLAGS.model_dir, FLAGS.model_dir + "." + datetime.now().strftime('%Y%m%d_%H%M'))

    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        # Build Graph
        ph_x, label_y, predictions, accuracy, loss, summary_op, train_op, global_step = build_model()

        # Prepare model directory
        model_dir = FLAGS.model_dir
        os.makedirs(model_dir, exist_ok=True)

        # Initialize
        sess.run(tf.global_variables_initializer())

        # Model Saver
        saver = tf.train.Saver(max_to_keep=10)

        # Train Summary Writer
        summary_dir = os.path.join(model_dir, 'summaries', 'train')
        os.makedirs(summary_dir, exist_ok=True)
        summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

        # Eval Summary Writer
        eval_summary_dir = os.path.join(summary_dir, 'summaries', 'eval')
        os.makedirs(eval_summary_dir, exist_ok=True)
        eval_summary_writer = tf.summary.FileWriter(eval_summary_dir, sess.graph)

        # Import data
        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, validation_size=5000)

        # Save model before first training
        model_file_path = os.path.join(model_dir, "model.ckpt")
        init_step = tf.train.global_step(sess, global_step)
        saver.save(sess=sess, save_path=model_file_path, global_step=init_step)

        # Train
        step_per_epoch = 55000 / 100
        num_epoch = 5
        for b_idx in range(int(step_per_epoch * num_epoch)):
            # mini batch size 100
            batch_xs, batch_ys = mnist.train.next_batch(100)

            cur_pred, cur_acc, cur_loss, cur_summary, _, cur_step = sess.run(
                [predictions, accuracy, loss, summary_op, train_op, global_step],
                feed_dict={ph_x: batch_xs, label_y: batch_ys})

            if b_idx % 25 == 0:
                # Log train result regularly
                summary_writer.add_summary(summary=cur_summary, global_step=cur_step)
                print('[Train][%s] accuracy: %s, loss: %s' % (cur_step, cur_acc, cur_loss))

            if b_idx % 100 == 0:
                # Regular Saving
                saver.save(sess=sess, save_path=model_file_path, global_step=cur_step)

                # Eval with Validation Set regularly
                eval_pred, eval_acc, eval_loss, eval_summary = sess.run(
                    [predictions, accuracy, loss, summary_op],
                    feed_dict={ph_x: mnist.validation.images, label_y: mnist.validation.labels})
                eval_summary_writer.add_summary(summary=eval_summary, global_step=cur_step)
                print('[Eval][%s] accuracy: %s, loss: %s' % (cur_step, eval_acc, eval_loss))

        final_step = tf.train.global_step(sess, global_step)

        # Training Completed, Saved
        saver.save(sess=sess, save_path=model_file_path, global_step=final_step)

        # Training Completed, Final Eval with Validation Set
        eval_pred, eval_acc, eval_loss, eval_summary = sess.run(
            [predictions, accuracy, loss, summary_op],
            feed_dict={ph_x: mnist.validation.images, label_y: mnist.validation.labels})
        eval_summary_writer.add_summary(summary=eval_summary, global_step=final_step)
        print('[Eval][%s] accuracy: %s, loss: %s' % (
            final_step, eval_acc, eval_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/opt/data/dataset/mnist/',
                        help='Directory for storing input data')
    parser.add_argument('--model_dir', type=str, default="/opt/data/models/examples/tf_core",
                        help='Directory for storing model')
    parser.add_argument('--mode', type=str, default='train',
                        help='train / predict')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
