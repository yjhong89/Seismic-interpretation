import numpy as np
import tensorflow as tf


def _batch_norm(x, training=True, name='batch_norm', epsilon=1e-5):
    return tf.layers.batch_normalization(x, center=True, scale=True, epsilon=epsilon, training=training)


def conv3d(x, out_channel, filter_size, stride, name='conv3d', activation=tf.nn.relu, normalization=True, padding='SAME', training=True, bias=True):
    # Data type: NHWDC
    if isinstance(x, np.ndarray):
        in_channel = x.shape[-1]
    else:
        _, _, _, _, in_channel = x.get_shape().as_list()

    with tf.variable_scope(name):
        weight = tf.get_variable('weight', shape=[filter_size, filter_size, filter_size, in_channel, out_channel], initializer=tf.truncated_normal_initializer(stddev=0.02))
        output = tf.nn.conv3d(x, weight, strides=[1, stride, stride, stride, 1], padding=padding, data_format='NDHWC')

        if bias:
            bias = tf.get_variable('bias', shape=[out_channel], initializer=tf.constant_initializer(0,1))
            output = output + bias

        if normalization:
            output = _batch_norm(output, training=training)
        if activation is not None:
            output = activation(output)

    return output
            

def max_pool3d(x, filter_size, stride, padding='SAME'):
    return tf.nn.max_pool3d(x, ksize=[1, filter_size, filter_size, filter_size, 1], strides=[1, stride, stride, stride, 1], padding=padding)

def global_avg_pool(x):
    return tf.reduce_mean(x, [1,2,3])

def fc(x, hidden, activation=tf.nn.relu, name='fc', normalization=True, training=True, bias=True):
    _, in_channels = x.get_shape().as_list()
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', shape=[in_channels, hidden], initializer=tf.truncated_normal_initializer(stddev=0.02))
        output = tf.matmul(x, weight)

        if bias:
            bias = tf.get_variable('bias', shape=[hidden], initializer=tf.constant_initializer(0.1))
            output = output + bias

        if normalization:
            output = _batch_norm(output, training=training)

        if activation is not None:
            output = activation(output)

    return output


def group_conv3d(x, channels, group, filter_size, stride,  name):
    group_index = 0
    in_channels = x.get_shape().as_list()[-1]

    assert(np.mod(in_channels, group) == 0 and np.mod(channels, group) == 0)
    num_group = int(in_channels/group)
    with tf.variable_scope(name):
        conv_index = 0
        conv_list = list()
        for i in range(num_group):
            conv = conv3d(x[:,:,:,:,i*group:(i+1)*group], int(channels/group), filter_size=filter_size, stride=stride, name='conv3d_%d'%conv_index, activation=None, normalization=False, bias=False, padding='SAME')
            conv_list.append(conv)
            conv_index += 1
        group_conv = tf.concat(conv_list, -1)

    return group_conv

            

def block(x, channels, group, name, training, down_sample=True):
    layer_index = 0
    in_channels = x.get_shape().as_list()[-1]

    with tf.variable_scope(name):
        if down_sample: 
            residual = conv3d(x, channels*2, filter_size=1, stride=1, name='conv3d_down', activation=None, padding='SAME', normalization=True, training=training, bias=False)
            residual = max_pool3d(residual, filter_size=3, stride=2, padding='SAME')
            stride = 2
        else:
            residual = x
            stride = 1

        # conv, 1^3, channels | BN | Relu
        layer1 = conv3d(x, channels, filter_size=1, stride=1, name='conv3d_%d'%layer_index, activation=tf.nn.relu, normalization=True, padding='SAME', training=training, bias=False)
        layer_index += 1

        # conv, 3^3, channels, group=32 | BN | Relu
        layer2 = group_conv3d(layer1, channels, group, filter_size=3, stride=stride, name='group_conv3d_%d'%layer_index)
        layer2 = tf.nn.relu(_batch_norm(layer2, training=training))
        layer_index += 1
        
        # conv, 1^3, channels*2 | BN
        # Down sampling is performed here
        layer3 = conv3d(layer2, channels*2, filter_size=1, stride=1, name='conv3d_%d'%layer_index, activation=None, padding='SAME', normalization=True, training=training, bias=False)


        # Residual connection 
        out = layer3 + residual
        out = tf.nn.relu(out)

    return out


