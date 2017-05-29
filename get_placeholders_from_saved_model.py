import tensorflow as tf


def save():
    # Prepare to feed input, i.e. feed_dict and placeholders
    ph_1 = tf.placeholder("float", shape=(None, 10, 256), name="ph_1")
    ph_2 = tf.placeholder("float", shape=(None, 10, 256), name="ph_2")
    b1 = tf.Variable(2.0, name="bias")

    # Define a test operation that we will restore
    add = tf.add(ph_1, ph_2)
    result = tf.multiply(add, b1, name="op_to_restore")

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Create a saver object which will save all the variables
    saver = tf.train.Saver()

    # Run the operation by feeding input
    # sess.run(w4, feed_dict)
    # Prints 24 which is sum of (w1+ph_2)*b1

    # Now, save the graph
    saver.save(sess, 'my_test_model', global_step=1000)
    print('model saved')


def load():
    sess = tf.Session()
    # First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('my_test_model-1000.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    sess.run(tf.global_variables_initializer())

    # Now, let's access and create placeholders variables and
    # create feed-dict to feed new data

    graph = tf.get_default_graph()
    input_1 = graph.get_tensor_by_name("ph_1:0")
    input_2 = graph.get_tensor_by_name("ph_2:0")
    output = graph.get_tensor_by_name("op_to_restore:0")

    print('loading model...')
    print('ph_1: %s' % input_1)
    print('ph_2: %s' % input_2)
    print('op_to_restore: %s' % output)

save()
load()
