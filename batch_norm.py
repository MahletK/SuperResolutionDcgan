import tensorflow as tf

from numpy.random import permutation

from discriminator import discriminator
from generator import generator
from costs_and_vars import costs_and_vars
from BatchGenerator import BatchGenerator
#from batch_norm import batch_norm
from conv2d import conv2d

def batch_norm(x, epsilon=1e-5, momentum = 0.999, scale=False, is_training=True, \
        name=None, reuse=False):
    with tf.variable_scope(name):
        if reuse:
            scope.reuse_variables()

        return tf.contrib.layers.batch_norm(x, decay=momentum, scale=scale, epsilon=epsilon, \
            updates_collections=None, is_training=is_training, scope=name)

def lrelu(x, leak=0.01):
    '''Leaky relu linear activation function with 'leak'.'''
    return tf.maximum(x, leak*x)

def linear(x, output_size, stddev=0.02, biases_start=0.0, name=None, reuse=False):
    '''Fully connected layer.'''
    with tf.variable_scope(name):
        if reuse:
            scope.reuse_variables()
    
        weights = tf.get_variable(name='weights', \
                shape=[x.get_shape()[1], output_size], dtype=tf.float32, \
                initializer=tf.random_normal_initializer(stddev=stddev))

        biases  = tf.get_variable(name='biases', shape=[output_size], dtype=tf.float32, \
                initializer=tf.constant_initializer(biases_start))

    return tf.nn.xw_plus_b(x, weights, biases)