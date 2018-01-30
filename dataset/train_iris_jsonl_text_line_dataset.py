import json
import logging
import math
import sys
from functools import partial

import numpy as np
import tensorflow as tf

from iris import create_iris_train_test_jsonl_files, species_name_to_int

logger = logging.getLogger(__name__)

FEATURES_SEPAL_LENGTH = "Sepal Length"
FEATURES_SEPAL_WIDTH = "Sepal Width"
FEATURES_PETAL_LENGTH = "Petal Length"
FEATURES_PETAL_WIDTH = "Petal Width"

LABELS_SPECIES = "Species"


def iris_jsonl_input_fn(jsonl_file_path, mini_batch_size, num_epochs,
                        shuffle=True, shuffle_buffer=1024):

    def _parse_jsonline(jsonline_from_str_tensor):
        # text_from_str_tensor is an binary string, decoding needed here
        jsonl_unicode = str(jsonline_from_str_tensor, encoding="utf-8")

        sample_dict = json.loads(jsonl_unicode, encoding="utf-8")

        jsonline_as_items = [None] * 5
        jsonline_as_items[0] = np.float32(sample_dict[FEATURES_SEPAL_LENGTH])
        jsonline_as_items[1] = np.float32(sample_dict[FEATURES_SEPAL_WIDTH])
        jsonline_as_items[2] = np.float32(sample_dict[FEATURES_PETAL_LENGTH])
        jsonline_as_items[3] = np.float32(sample_dict[FEATURES_PETAL_WIDTH])
        jsonline_as_items[4] = np.int32(species_name_to_int(sample_dict[LABELS_SPECIES]))

        return tuple(jsonline_as_items)

    def _map_features_and_labels(text_line_tensor):

        # Have to use tf.py_func() inside a `map_func` (which is inside an `input_fn`)
        # to convert one jsonlines (which comes from tf.data.TextLineDataset) into list of values.
        # `tf.py_func()` does not support `dict` for now
        json_values = tf.py_func(_parse_jsonline,
                                 inp=[text_line_tensor],
                                 Tout=(tf.float32, tf.float32, tf.float32, tf.float32, tf.int32))

        # a trick here as `tf.py_func()` is unable to determine the shapes of its return values
        json_values[0].set_shape(tf.TensorShape([]))
        json_values[1].set_shape(tf.TensorShape([]))
        json_values[2].set_shape(tf.TensorShape([]))
        json_values[3].set_shape(tf.TensorShape([]))
        json_values[4].set_shape(tf.TensorShape([]))

        features = {
            FEATURES_SEPAL_LENGTH: json_values[0],
            FEATURES_SEPAL_WIDTH: json_values[1],
            FEATURES_PETAL_LENGTH: json_values[2],
            FEATURES_PETAL_WIDTH: json_values[3],
        }

        labels = {
            LABELS_SPECIES: json_values[4],
        }

        return features, labels

    ds = tf.data.TextLineDataset(jsonl_file_path)
    ds = ds.map(map_func=_map_features_and_labels)

    ds = ds.repeat(num_epochs)
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer)
    ds = ds.batch(mini_batch_size)

    iterator = ds.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels


def model_fn(features, labels, mode, params, config):
    num_classes = 3

    with tf.device("/gpu:0"):

        # Feature: Placeholders to take in features
        x_sepal_length = features[FEATURES_SEPAL_LENGTH]
        x_sepal_width = features[FEATURES_SEPAL_WIDTH]
        x_petal_length = features[FEATURES_PETAL_LENGTH]
        x_petal_width = features[FEATURES_PETAL_WIDTH]

        x = tf.stack(values=[x_sepal_length, x_sepal_width, x_petal_length, x_petal_width],
                     axis=1,
                     name="x")
        print("x: %s" % x)

        # Label
        label_y = labels[LABELS_SPECIES]
        print("label_y: %s" % label_y)

        label_y_one_hot = tf.one_hot(label_y, depth=num_classes, name="y_one_hot")
        print("label_y_one_hot: %s" % label_y_one_hot)

        # global step
        global_step = tf.train.get_global_step()

        with tf.variable_scope("algo"):
            # Variables
            W = tf.Variable(tf.zeros([x.shape[1], num_classes]))
            b = tf.Variable(tf.zeros([num_classes]))

            # Operations
            y = tf.matmul(x, W) + b
        tf.summary.histogram("W", W)
        tf.summary.histogram("b", b)

        # Predictions
        with tf.variable_scope("predictions"):
            predictions = tf.argmax(y, 1)
            corrects = tf.equal(predictions, tf.argmax(label_y_one_hot, 1))
            accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

        # loss
        with tf.variable_scope("losses"):
            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_y, logits=y))
            loss = cross_entropy
        tf.summary.scalar("xentropy_loss", cross_entropy)
        tf.summary.scalar("total_loss", loss)

        # optimizer
        with tf.variable_scope("optimizer"):
            train_op = tf.train.GradientDescentOptimizer(0.5).minimize(loss, global_step=global_step)

        # Collect summaries
        # summary_op = tf.summary.merge_all()

        eval_metric_ops = dict()
        eval_metric_ops["accuracy"] = tf.metrics.accuracy(labels=label_y,
                                                          predictions=predictions)

    # return x, label_y, predictions, accuracy, loss, summary_op, train_op, global_step
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
    )


def train_and_test_simple_estimator(num_epochs, model_dir):
    mini_batch_size = 16

    train_path, test_path = create_iris_train_test_jsonl_files()

    train_input_fn = partial(iris_jsonl_input_fn,
                             jsonl_file_path=train_path,
                             mini_batch_size=mini_batch_size,
                             num_epochs=num_epochs,
                             shuffle=True,
                             shuffle_buffer=mini_batch_size * 100)

    # with tf.Session() as sess:
    #     while True:
    #         try:
    #             feature, labels = train_input_fn()
    #             result = sess.run([feature, labels])
    #             print("result: %s" % result)
    #         except tf.errors.OutOfRangeError:
    #             break

    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=dict())
    estimator.train(input_fn=train_input_fn)

    test_input_fn = partial(iris_jsonl_input_fn,
                            jsonl_file_path=test_path,
                            mini_batch_size=mini_batch_size,
                            num_epochs=1,
                            shuffle=False)

    # with tf.Session() as sess:
    #     while True:
    #         try:
    #             feature, labels = test_input_fn()
    #             result = sess.run([feature, labels])
    #             print("result: %s" % result)
    #         except tf.errors.OutOfRangeError:
    #             break

    test_result = estimator.evaluate(input_fn=test_input_fn)
    print("test_result: %s" % test_result)


def train_and_test_learn_runner(num_epochs, model_dir):
    mini_batch_size = 16

    train_path, test_path = create_iris_train_test_jsonl_files()

    train_input_fn = partial(iris_jsonl_input_fn,
                             jsonl_file_path=train_path,
                             mini_batch_size=mini_batch_size,
                             num_epochs=num_epochs,
                             shuffle=True,
                             shuffle_buffer=mini_batch_size * 100)

    test_input_fn = partial(iris_jsonl_input_fn,
                            jsonl_file_path=test_path,
                            mini_batch_size=mini_batch_size,
                            num_epochs=1,
                            shuffle=False)

    steps_per_epoch = math.ceil(120 / mini_batch_size)
    steps_per_val_epoch = math.ceil(30 / mini_batch_size)
    steps_per_test_epoch = math.ceil(30 / mini_batch_size)
    eval_per_steps = 10
    summary_per_steps = 5

    # model save/summary frequency
    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(
        model_dir=model_dir,
        save_summary_steps=summary_per_steps,
        save_checkpoints_steps=eval_per_steps,
        log_step_count_steps=eval_per_steps,
        keep_checkpoint_max=5,
    )
    logger.debug('run_config: %s', run_config)

    # hyperparameters
    model_params = dict()

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        params=model_params,
        config=run_config,
    )

    def experiment_fn(run_config, params):

        # Define the experiment
        experiment = tf.contrib.learn.Experiment(
            estimator=estimator,  # Estimator
            train_input_fn=train_input_fn,  # First-class function
            eval_input_fn=test_input_fn,  # First-class function
            train_steps=steps_per_epoch * num_epochs,  # Minibatch steps
            min_eval_frequency=eval_per_steps / 5,  # Eval frequency after a checkpoint is saved
            train_monitors=[],  # Hooks for training
            eval_hooks=[],  # Hooks for evaluation
            eval_steps=steps_per_val_epoch,  # Use evaluation feeder until its empty
        )
        return experiment

    # Set the run_config and the directory to save the model and stats
    from tensorflow.contrib.learn import learn_runner
    learn_runner.run(
        experiment_fn=experiment_fn,  # First-class function
        schedule="train_and_evaluate",  # What to run
        run_config=run_config,
    )

    test_result = estimator.evaluate(input_fn=test_input_fn,  # First-class function
                                     steps=steps_per_test_epoch,
                                     hooks=[])
    logger.info('test_result: %s', test_result)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=log_format)

    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    model_dir = "./models/iris_jsonl_text_line_dataset/" + ts

    # train_and_test_simple_estimator(num_epochs=50, model_dir=model_dir)
    train_and_test_learn_runner(num_epochs=50, model_dir=model_dir)
