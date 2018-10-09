import tensorflow as tf
import numpy as np
import op


def resnetx(cube, num_classes, channels, training, name='3d_resnetx', group=32, reuse=False):
    block_index = 0
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        l1 = op.conv3d(cube, channels, filter_size=7, stride=1, name='conv3d_first', activation=tf.nn.relu, normalization=True, padding='SAME')           
        print(l1.get_shape().as_list())
        
        # 3*3*# max pooling layer is located
        l2 = op.max_pool3d(l1, filter_size=3, stride=2, padding='SAME')
        print(l2.get_shape().as_list())
        
        l3 = op.block(l2, channels, group=group, name='block_%d'%block_index, training=training)
        block_index += 1
        print(l3.get_shape().as_list())

        l4 = op.block(l3, channels, group=group, name='block_%d'%block_index, training=training, down_sample=False)
        block_index += 1
        print(l4.get_shape().as_list())

        l5 = op.block(l4, channels, group=group, name='block_%d'%block_index, training=training, down_sample=False)
        print(l5.get_shape().as_list())
        
        l6 = op.global_avg_pool(l5)
        print(l6.get_shape().as_list())

        l7 = op.fc(l6, num_classes, activation=None, name='pre-softmax', bias=True) 
        print(l7.get_shape().as_list())


    return l7


if __name__ == "__main__":
    x = tf.get_variable('test', [1,61,61,61,1])
    resnetx(x, num_classes=9, channels=64, training=True) 
