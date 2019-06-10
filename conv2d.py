import tensorflow as tf

from numpy.random import permutation

from discriminator import discriminator
from generator import generator
from costs_and_vars import costs_and_vars
from BatchGenerator import BatchGenerator
from batch_norm import batch_norm
3from conv2d import conv2d

def conv2d(x, output_dim, kernel=3, stride=2, stddev=0.02, padding='SAME', name=None, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        weights = tf.get_variable(name='weights', \
                shape=[kernel, kernel, x.get_shape()[-1], output_dim], dtype=tf.float32, \
                initializer=tf.truncated_normal_initializer(stddev=stddev))
        
        conv    = tf.nn.conv2d(x, filter=weights, strides=[1, stride, stride, 1], padding=padding)

        biases  = tf.get_variable(name='biases', shape=[output_dim], \
                dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        out     = tf.nn.bias_add(conv, biases)
        
    return out

