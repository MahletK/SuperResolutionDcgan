import tensorflow as tf

from numpy.random import permutation

# from discriminator import discriminator
# #from generator import generator
# from costs_and_vars import costs_and_vars
# from BatchGenerator import BatchGenerator
# from batch_norm import batch_norm
# from conv2d import conv2d

def generator(x, is_training=True, reuse=False):
    '''Map input images from 64 x 64 x 3 to 128 x 128 x 3.'''
    with tf.variable_scope('generator') as scope:
        if reuse:            
            scope.reuse_variables()
   
        conv1 = conv2d(x, output_dim=32, stride=1, name='g_conv1')
        conv1 = batch_norm(conv1, is_training=is_training, name='g_conv1_bn')
        conv1 = lrelu(conv1)
        #64 x 64 x 32
        
        conv2 = conv2d(conv1, output_dim=128, stride=1, name='g_conv2')
        conv2 = batch_norm(conv2, is_training=is_training, name='g_conv2_bn')
        conv2 = lrelu(conv2)
        #64 x 64 x 128

        conv3 = conv2d(conv2, output_dim=128, stride=1, name='g_conv3')
        conv3 = batch_norm(conv3, is_training=is_training, name='g_conv3_bn')
        conv3 = lrelu(conv3)
        #64 x 64 x 128

        upsampled = tf.image.resize_images(conv3, size=[128, 128])

        conv4 = conv2d(upsampled, output_dim=128, stride=1, name='g_conv4')
        conv4 = batch_norm(conv4, is_training=is_training, name='g_conv4_bn')
        conv4 = lrelu(conv4)
        #128 x 128 x 128

        conv5 = conv2d(conv4, output_dim=64, stride=1, name='g_conv5')
        conv5 = batch_norm(conv5, is_training=is_training, name='g_conv5_bn')
        conv5 = lrelu(conv5)
        #128 x 128 x 64

        conv6 = conv2d(conv5, output_dim=3, stride=1, name='g_conv6')
        conv6 = tf.nn.sigmoid(conv6)
        #128 x 128 x 3

    return conv6

