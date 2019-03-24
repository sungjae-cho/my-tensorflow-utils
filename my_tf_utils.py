import tensorflow as tf
        
def get_value(node):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        node_val = sess.run(node)
    return node_val
        
def get_values(nodes):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        nodes_val = sess.run(nodes)
    return nodes_val
        
def get_shape(node):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        nodes_val = sess.run(nodes)
    return nodes_val
