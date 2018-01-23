import math

import tensorflow as tf


def try_dataset_from_generator_repeated():
    print("")
    print("Try tf.data.Dataset.from_generator()")

    sample_count = 100
    repeat_times = 3

    def _gen_samples():
        for i in range(sample_count):
            yield {
                "+ve": i,
                "-ve": i * -1,
            }

    ds = tf.data.Dataset.from_generator(
        generator=_gen_samples,
        output_types={
            "+ve": tf.int64,
            "-ve": tf.int64
        },
        output_shapes={
            "+ve": tf.TensorShape([]),
            "-ve": tf.TensorShape([])
        })
    ds = ds.repeat(repeat_times)
    next = ds.make_one_shot_iterator().get_next()

    yield_count = 0
    with tf.Session() as sess:
        while True:
            try:
                print(sess.run(next))
                yield_count += 1
            except tf.errors.OutOfRangeError:
                break
    assert yield_count == repeat_times * sample_count
    print("yield_count: %d" % yield_count)


def try_dataset_from_generator_repeated_with_shard():
    print("")
    print("Try tf.data.Dataset.from_generator() with sharding")

    sample_count = 26
    repeat_times = 32

    def _gen_samples():
        for i in range(sample_count):
            yield {
                "+ve": i,
                "-ve": i * -1,
            }

    ds = tf.data.Dataset.from_generator(
        generator=_gen_samples,
        output_types={
            "+ve": tf.int64,
            "-ve": tf.int64
        },
        output_shapes={
            "+ve": tf.TensorShape([]),
            "-ve": tf.TensorShape([])
        })
    ds = ds.repeat(repeat_times)
    ds = ds.shard(3, 1)
    next = ds.make_one_shot_iterator().get_next()

    yield_count = 0
    with tf.Session() as sess:
        while True:
            try:
                print(sess.run(next))
                yield_count += 1
            except tf.errors.OutOfRangeError:
                break
    assert yield_count == ((repeat_times * sample_count) // 3)
    print("yield_count: %d" % yield_count)


def try_dataset_from_generator_repeated_in_batches():
    """Prove that it is possible to repeat a generator function using `Dataset.repeat()`

    :return:
    """
    print("")
    print("Try tf.data.Dataset.from_generator() with batch()")

    sample_count = 100
    repeat_times = 3
    batch_size = 32
    shuffle_data = True
    queue_size = batch_size * 100

    def _gen_samples():
        for i in range(sample_count):
            yield {
                "+ve": ([i] * 2),
                "-ve": ([i * -1] * 3),
            }

    ds = tf.data.Dataset.from_generator(
        generator=_gen_samples,
        output_types={
            "+ve": tf.int64,
            "-ve": tf.int64
        },
        output_shapes={
            "+ve": tf.TensorShape([2]),
            "-ve": tf.TensorShape([3])
        })
    ds = ds.repeat(repeat_times)
    if shuffle_data:
        ds = ds.shuffle(buffer_size=queue_size)
    ds = ds.batch(batch_size)
    next = ds.make_one_shot_iterator().get_next()

    yield_count = 0
    with tf.Session() as sess:
        while True:
            try:
                result = sess.run(next)
                print("result: %s" % result)
                assert result["+ve"].shape[0] == batch_size or result["+ve"].shape[0] == (repeat_times * sample_count % batch_size)
                assert result["+ve"].shape[1] == 2
                assert result["-ve"].shape[0] == batch_size or result["+ve"].shape[0] == (repeat_times * sample_count % batch_size)
                assert result["-ve"].shape[1] == 3
                yield_count += 1
            except tf.errors.OutOfRangeError:
                break
            print("yield_count: %d" % yield_count)

    assert yield_count == math.ceil(repeat_times * sample_count / batch_size)
    print("Final yield_count: %d" % yield_count)


def try_dataset_from_multiple_file_generators():
    """Prove that it is possible to repeat a generator function which reads from file using `Dataset.repeat()`

    :return:
    """
    print("")
    print("Try tf.data.Dataset.from_generator() with batch() for multiple files and concate multiple datasets")

    def _gen_samples(file_path):
        with open(file_path, mode="r") as f:
            print("Read from %s..." % file_path)
            for l in f:
                i = int(l.strip())
                yield {
                    "+ve": ([i] * 2),
                    "-ve": ([i * -1] * 3),
                }

    def make_gen_fn(file_path):
        # workarounds
        return lambda: _gen_samples(file_path=file_path)

    repeat_count = 3
    batch_size = repeat_count * 10

    file_setups = [
        (0, 22),
        (22, 47),
        (47, 87),
        (87, 100),
    ]
    file_gen_funcs = []

    import tempfile
    for idx, (start, end) in enumerate(file_setups):
        _, data_path = tempfile.mkstemp("%d.csv" % idx, text=True)
        file_gen_funcs.append(make_gen_fn(file_path=data_path))

        with open(data_path, mode="w") as write_f:
            for num in range(start, end):
                write_f.write("%d\n" % num)

        with open(data_path, mode="r") as f:
            print("%s:\n%s" % (data_path, f.read(-1)))

    for fn_idx, gen_func in enumerate(file_gen_funcs):
        for l in gen_func():
            print("[Testing] Reading from file %d: %s" % (fn_idx, l))

    datasets = []
    for file_gen_f in file_gen_funcs:
        ds = tf.data.Dataset.from_generator(
            generator=file_gen_f,
            output_types={
                "+ve": tf.int64,
                "-ve": tf.int64
            },
            output_shapes={
                "+ve": tf.TensorShape([2]),
                "-ve": tf.TensorShape([3])
            })
        datasets.append(ds)

    ds = datasets[0]
    for next_ds in datasets[1:]:
        ds = ds.concatenate(next_ds)
    ds = ds.repeat(repeat_count)
    ds = ds.batch(batch_size)
    next = ds.make_one_shot_iterator().get_next()

    yield_count = 0
    with tf.Session() as sess:
        while True:
            try:
                result = sess.run(next)
                print("result: %s" % result)
                assert result["+ve"].shape[0] == batch_size
                assert result["+ve"].shape[1] == 2
                assert result["-ve"].shape[0] == batch_size
                assert result["-ve"].shape[1] == 3
                yield_count += 1
            except tf.errors.OutOfRangeError:
                break
            print("yield_count: %d" % yield_count)

    assert yield_count == repeat_count * 100 / batch_size
    print("Final yield_count: %d" % yield_count)


def try_dataset_from_file_generator_repeated_in_batches():
    """Prove that it is possible to repeat a generator function which reads from file using `Dataset.repeat()`

    :return:
    """
    print("")
    print("Try tf.data.Dataset.from_generator() with batch(), reading from the same file multiple times")

    sample_count = 100
    repeat_times = 5
    batch_size = 16
    shuffle_data = True
    queue_size = batch_size * 100

    import tempfile
    _, data_path = tempfile.mkstemp("1.csv", text=True)

    with open(data_path, mode="w") as write_f:
        for num in range(sample_count):
            write_f.write("%d\n" % num)

    def _gen_samples():
        with open(data_path, mode="r") as f:
            print("Read from %s..." % data_path)
            for l in f:
                i = int(l.strip())
                yield {
                    "+ve": ([i] * 2),
                    "-ve": ([i * -1] * 3),
                }

    # for m in _gen_samples():
    #     print(m)

    ds = tf.data.Dataset.from_generator(
        generator=_gen_samples,
        output_types={
            "+ve": tf.int64,
            "-ve": tf.int64
        },
        output_shapes={
            "+ve": tf.TensorShape([2]),
            "-ve": tf.TensorShape([3])
        })
    ds = ds.repeat(repeat_times)
    if shuffle_data:
        ds = ds.shuffle(buffer_size=queue_size)
    ds = ds.batch(batch_size)
    next = ds.make_one_shot_iterator().get_next()

    yield_count = 0
    with tf.Session() as sess:
        while True:
            try:
                result = sess.run(next)
                print("result: %s" % result)
                assert result["+ve"].shape[0] == batch_size or result["+ve"].shape[0] == (repeat_times * sample_count % batch_size)
                assert result["+ve"].shape[1] == 2
                assert result["-ve"].shape[0] == batch_size or result["+ve"].shape[0] == (repeat_times * sample_count % batch_size)
                assert result["-ve"].shape[1] == 3
                yield_count += 1
            except tf.errors.OutOfRangeError:
                break
            print("yield_count: %d" % yield_count)

    assert yield_count == math.ceil(repeat_times * sample_count / batch_size)
    print("Final yield_count: %d" % yield_count)


if __name__ == '__main__':
    # try_dataset_from_generator_repeated()
    # try_dataset_from_generator_repeated_with_shard()
    # try_dataset_from_generator_repeated_in_batches()
    try_dataset_from_multiple_file_generators()
    # try_dataset_from_file_generator_repeated_in_batches()
