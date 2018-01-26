import tensorflow as tf
from functools import partial


def try_bucket_by_sequence_length_with_estimator():
    max_seq_len = 10

    def gen_var_len_seq():
        for i in range(1, 1001):
            seq_len = (i % max_seq_len)
            seq_len = seq_len if seq_len > 0 else 10
            seq = [i] * seq_len
            yield seq

    def input_fn(mini_batch_size, num_epochs, shuffle=True, shuffle_buffer=1024):

        ds = tf.data.Dataset.from_generator(
            generator=gen_var_len_seq,
            output_types=tf.int32,
            output_shapes=tf.TensorShape([None]))
        ds = ds.repeat(num_epochs)
        if shuffle:
            ds = ds.shuffle(buffer_size=shuffle_buffer)

        seq_tensor = ds.make_one_shot_iterator().get_next()

        # `dynamic_pad` must be True if the input `tensors` is variable-length tensors
        bucketed = tf.contrib.training.bucket_by_sequence_length(
            input_length=tf.shape(seq_tensor)[-1],
            tensors=[seq_tensor],
            batch_size=mini_batch_size,
            bucket_boundaries=[4, 7],
            dynamic_pad=True)

        bucket_seq_len, bucket_outputs = bucketed

        features = {
            "x": bucket_outputs[0],
            "x_seq_len": bucket_seq_len,
        }

        labels = dict()

        print("features: %s" % features)
        print("labels: %s" % labels)

        return features, labels

    def model_fn(features, labels, mode, params, config):

        x = features["x"]
        x_seq_len = features["x_seq_len"]

        # Predictions
        dummy_predictions = {
            "batch_size": tf.shape(x)[0],
            "bucket_size": tf.shape(x)[-1],
            "avg_seq_len": tf.reduce_mean(x_seq_len),
        }

        x = tf.Print(x, data=[x], message="x", summarize=1000)
        x = tf.Print(x, data=[x_seq_len], message="x_seq_len", summarize=1000)

        # Dummy Layer
        dummy_var = tf.get_variable(name="dummy_var", shape=())

        # Loss
        dummy_loss = tf.reduce_mean(tf.cast(x, dtype=tf.float32) * dummy_var)
        dummy_loss += tf.random_normal(shape=(), dtype=tf.float32)

        # Global Step
        global_step = tf.train.get_global_step()

        # Optimizer
        train_op = tf.train.GradientDescentOptimizer(0.5).minimize(
            loss=dummy_loss, global_step=global_step)

        # Evaluation Metrics
        eval_metric_ops = dict()
        eval_metric_ops["accuracy"] = tf.metrics.accuracy(
            labels=tf.ones(shape=(5, ), dtype=tf.int64),
            predictions=tf.ones(shape=(5, ), dtype=tf.int64))

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=dummy_predictions,
            loss=dummy_loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
        )

    train_input_fn = partial(input_fn, mini_batch_size=32, num_epochs=1000, shuffle=False)
    test_input_fn = partial(input_fn, mini_batch_size=32, num_epochs=1, shuffle=False)

    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    model_dir = "./models/iris_csv_text_line_dataset/" + ts

    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=dict())
    estimator.train(input_fn=train_input_fn)

    test_result = estimator.evaluate(input_fn=test_input_fn)
    print("test_result: %s" % test_result)


def try_bucket_by_sequence_length():
    max_seq_len = 10

    def gen_var_len_seq():
        for i in range(1, 1001):
            seq_len = (i % max_seq_len)
            seq_len = seq_len if seq_len > 0 else 10
            seq = [i] * seq_len
            # seq += [0] * (10 - seq_len)
            # yield (seq_len, seq)
            yield seq

    ds = tf.data.Dataset.from_generator(
        generator=gen_var_len_seq,
        # output_types=(tf.int32, tf.int32),
        # output_shapes=(tf.TensorShape([]), tf.TensorShape([None])))
        output_types=tf.int32,
        output_shapes=tf.TensorShape([None]))

    # seq_len, seq = ds.make_one_shot_iterator().get_next()
    seq = ds.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        # for i in range(500):
        #     print(sess.run([seq_len, seq]))

        # `dynamic_pad` must be True if the input `tensors` is variable-length tensors
        bucketed = tf.contrib.training.bucket_by_sequence_length(
            input_length=tf.shape(seq)[-1],
            tensors={
                "int_seq": seq,
                "int_seq_2": tf.stack([seq, seq]),
            },
            batch_size=32,
            bucket_boundaries=[4, 7],
            dynamic_pad=True)

        bucket_seq_len, bucket_outputs = bucketed
        print(bucket_seq_len, bucket_outputs)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        for i in range(10):
            r_bucket_seq_len, r_bucket_outputs = sess.run([bucket_seq_len, bucket_outputs])
            print(r_bucket_seq_len)
            print(r_bucket_outputs)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    # try_bucket_by_sequence_length()
    try_bucket_by_sequence_length_with_estimator()
