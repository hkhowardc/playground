import logging
from functools import partial
import math
import numpy as np
import tensorflow as tf
import random
import sys

logger = logging.getLogger(__name__)


CURVE_TYPE_UNKNOWN = 0
CURVE_TYPE_SIN = 1
CURVE_TYPE_LINEAR = 2


def gen_sin_seq():
    seq_lens = [sl for sl in range(70, 100 + 1)]
    random.shuffle(seq_lens)

    x_offsets = [offset * 0.01 for offset in range(1, 20 + 1)]
    random.shuffle(x_offsets)

    y_offsets = [0.1, 0.2, 0.3, 0.4, 0.5]
    random.shuffle(y_offsets)

    for x_offset in x_offsets:
        for seq_len in seq_lens:
            for y_offset in y_offsets:
                sin_seq = [np.sin((2 * np.pi * (pos * 0.1)) + x_offset) + y_offset for pos in range(100)]

                sin_seq = sin_seq[:seq_len]
                assert 70 <= len(sin_seq) <= 100

                yield sin_seq


def gen_linear_seq():
    seq_lens = [sl for sl in range(70, 100 + 1)]
    random.shuffle(seq_lens)

    x_offsets = [offset * 0.01 for offset in range(1, 20 + 1)]
    random.shuffle(x_offsets)

    y_offsets = [0.1, 0.2, 0.3, 0.4, 0.5]
    random.shuffle(y_offsets)

    for x_offset in x_offsets:
        for seq_len in seq_lens:
            for slope in y_offsets:
                lin_seq = [(slope * (pos * 0.1)) + x_offset for pos in range(100)]

                lin_seq = lin_seq[:seq_len]
                assert 70 <= len(lin_seq) <= 100

                yield lin_seq


def gen_num_seq():
    for sin_seq, lin_seq in zip(gen_sin_seq(), gen_linear_seq()):
        yield CURVE_TYPE_SIN, sin_seq
        yield CURVE_TYPE_LINEAR, lin_seq


def input_fn(mini_batch_size, num_epochs, shuffle=True, shuffle_buffer=1024):

    ds = tf.data.Dataset.from_generator(
        generator=gen_num_seq,
        output_types=(tf.int32, tf.float32),
        output_shapes=(tf.TensorShape([]), tf.TensorShape([None])))
    ds = ds.repeat(num_epochs)
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer)

    curve_type, num_seq = ds.make_one_shot_iterator().get_next()
    print("curve_type: %s" % curve_type)
    print("num_seq: %s" % num_seq)

    # Attention Here!
    # 1) `batch_size` parameter to `tf.contrib.training.bucket_by_sequence_length()`
    #   must be a variable if you want an unspecified batch size (i.e. `None`)
    # 2) must use `tf.assign()` to assign the mini batch size, otherwise Tensorflow
    #   will use the previously-saved batch size
    t_batch_size = tf.get_variable(name="mini_batch_size",
                                   dtype=tf.int32,
                                   initializer=0,
                                   trainable=False)
    t_batch_size = tf.assign(ref=t_batch_size, value=mini_batch_size)

    # `dynamic_pad` must be True if the input `tensors` is variable-length tensors
    bucketed = tf.contrib.training.bucket_by_sequence_length(
        input_length=tf.shape(num_seq)[-1],
        tensors=[curve_type, num_seq],
        batch_size=t_batch_size,
        # buckets: 70-80, 80-90, 90-100
        bucket_boundaries=[81, 91],
        dynamic_pad=True)

    bucket_seq_len, [bucket_curve_type, bucket_num_seq] = bucketed

    features = {
        "x": bucket_num_seq,
        "x_seq_len": bucket_seq_len,
    }

    labels = {
        "y": bucket_curve_type
    }

    print("features: %s" % features)
    print("labels: %s" % labels)

    return features, labels


def model_fn(features, labels, mode, params, config):

    # Features
    x = features["x"]
    x_seq_len = features["x_seq_len"]
    print("x: %s" % x)
    print("x_seq_len: %s" % x_seq_len)

    # Labels
    label_y = labels["y"]
    print("label_y: %s" % label_y)

    # Global Step
    global_step = tf.train.get_global_step()

    log_msg = "[%s] Global Step|Mini Batch Size|Bucket Seq Len" % mode
    mini_batch_size = tf.shape(x, name="mini_batch_size")[0]
    bucket_seq_len = tf.shape(x, name="bucket_seq_len")[-1]
    x = tf.Print(x,
                 data=[global_step, mini_batch_size, bucket_seq_len],
                 message=log_msg,
                 first_n=100, summarize=1)

    with tf.variable_scope("algo"):
        x_rnn_inputs = tf.expand_dims(x, axis=-1)
        print("x_rnn_inputs: %s" % x_rnn_inputs)

        gru_cell = tf.nn.rnn_cell.GRUCell(num_units=128)
        _, state = tf.nn.dynamic_rnn(cell=gru_cell,
                                     inputs=x_rnn_inputs, sequence_length=x_seq_len,
                                     dtype=tf.float32)
        print("gru_cell: %s, state: %s" % (gru_cell, state))

        logits = tf.layers.dense(inputs=state,
                                 units=3, activation=tf.nn.selu, use_bias=True)
        print("logits: %s" % logits)

    # Predictions
    with tf.variable_scope("predictions"):
        y = tf.nn.softmax(logits)
        predictions = tf.cast(tf.argmax(y, 1), dtype=tf.int32)
        corrects = tf.equal(predictions, label_y)
        accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

    loss = None
    eval_metric_ops = None
    train_op = None
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):

        # Losses
        with tf.variable_scope("losses"):
            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_y, logits=y))
            loss = cross_entropy
        tf.summary.scalar("xentropy_loss", cross_entropy)
        tf.summary.scalar("total_loss", loss)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # Optimizer
            with tf.variable_scope("optimizer"):
                train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(
                    loss=loss, global_step=global_step)

        if mode == tf.estimator.ModeKeys.EVAL:
            # Evaluation Metrics
            eval_metric_ops = dict()
            eval_metric_ops["accuracy"] = tf.metrics.accuracy(labels=label_y,
                                                              predictions=predictions)
            eval_metric_ops["xentropy_loss"] = tf.metrics.mean(values=cross_entropy)
            eval_metric_ops["total_loss"] = tf.metrics.mean(values=loss)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
    )


def train_and_test_learn_runner(num_epochs, model_dir):
    mini_batch_size = 32
    samples_per_epoch = len([l for l in gen_num_seq()])

    steps_per_epoch = math.ceil(samples_per_epoch / mini_batch_size)
    steps_per_val_epoch = 5
    steps_per_test_epoch = 1
    eval_per_steps = steps_per_epoch // 10
    summary_per_steps = steps_per_epoch // 20

    train_input_fn = partial(input_fn,
                             mini_batch_size=mini_batch_size,
                             num_epochs=num_epochs,
                             shuffle=True,
                             shuffle_buffer=mini_batch_size * 100)

    eval_input_fn = partial(input_fn,
                            mini_batch_size=mini_batch_size * 10,
                            num_epochs=1,
                            shuffle=True,
                            shuffle_buffer=mini_batch_size * 100)

    test_input_fn = partial(input_fn,
                            mini_batch_size=mini_batch_size,
                            num_epochs=1,
                            shuffle=True,
                            shuffle_buffer=mini_batch_size * 100)

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

    # Hyper-parameters
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
            eval_input_fn=eval_input_fn,  # First-class function
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
    model_dir = "./models/iris_bucketing/" + ts

    train_and_test_learn_runner(num_epochs=1, model_dir=model_dir)
