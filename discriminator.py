import tensorflow as tf

from numpy.random import permutation

# #from discriminator import discriminator
# # from generator import generator
# # from costs_and_vars import costs_and_vars
# # from BatchGenerator import BatchGenerator
# from batch_norm import batch_norm
# from conv2d import conv2d



def discriminator(images, is_training=True, reuse=False):
    '''Discriminate 128 x 128 x 3 images fake or real within the range [fake, real] = [0, 1].'''

    with tf.variable_scope('discriminator') as scope:
        
        if reuse:
            scope.reuse_variables()
 
        conv1 = conv2d(images, output_dim=64, kernel=7, stride=1, name='d_conv1')
        conv1 = batch_norm(conv1, is_training=is_training, name='d_conv1_bn')
        conv1 = lrelu(conv1)
        #128 x 128 x 64
        
        conv2 = conv2d(conv1, output_dim=64, kernel=7, stride=2, name='d_conv2')
        conv2 = batch_norm(conv2, is_training=is_training, name='d_conv2_bn')
        conv2 = lrelu(conv2)
        #64 x 64 x 64
            
        conv3 = conv2d(conv2, output_dim=32, kernel=3, stride=2, name='d_conv3')
        conv3 = batch_norm(conv3, is_training=is_training, name='d_conv3_bn')
        conv3 = lrelu(conv3)
        #32 x 32 x 32

        conv4 = conv2d(conv3, output_dim=1, kernel=3, stride=2, name='d_conv4')
        conv4 = batch_norm(conv4, is_training=is_training, name='d_conv4_bn')
        conv4 = lrelu(conv4)
        #16 x 16 x 1

        fc = tf.reshape(conv4, [-1, 16 * 16 * 1])
        fc = linear(fc, output_size=1, name='d_fc')
    
    return fc
