import tensorflow as tf

def print_node(node):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        node_val = sess.run(node)
        print(node_val)

def print_shape(tensor):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        shape_val = sess.run(tf.shape(tensor))
        print(shape_val)