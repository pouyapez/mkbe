import tensorflow as tf


def mrr(higher_values):
    pos_index = higher_values + 1
    return tf.reduce_mean(1.0/ pos_index)

def hits_n(higher_values, n):
    hits_times = tf.cast(higher_values <= (n - 1), tf.float32)
    return tf.reduce_mean(hits_times)