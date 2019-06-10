import tensorflow as tf

from numpy.random import permutation

# from discriminator import discriminator
# from generator import generator
# #from costs_and_vars import costs_and_vars
# from BatchGenerator import BatchGenerator
# from batch_norm import batch_norm
# from conv2d import conv2d

def costs_and_vars(real, generated, real_disc, gener_disc, is_training=True):
    '''Return generative and discriminator networks\' costs,
    and variables to optimize them if is_training=True.'''
    d_real_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_disc, \
            labels=tf.ones_like(real_disc)))
    d_gen_cost  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gener_disc, \
            labels=tf.zeros_like(gener_disc)))
     
    g_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gener_disc, \
            labels=tf.ones_like(gener_disc))) * 0.1 + \
            tf.reduce_mean(tf.abs(tf.subtract(generated, real)))

    d_cost = d_real_cost + d_gen_cost
    
    if is_training:
        t_vars = tf.trainable_variables()
        
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
    
        return g_cost, d_cost, g_vars, d_vars

    else:
        return g_cost, d_cost


