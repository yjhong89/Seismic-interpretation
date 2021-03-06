import tensorflow as tf
import numpy as np
import model.op as op


class Resnetx(object):
    def __init__(self, num_classes, channels, training, name='resnetx', group=32, reuse=False):
        self.num_classes = num_classes
        self.channels = channels
        self.training = training
        self.group = group
        self.reuse = reuse
        self.name = name

    def __call__(self, cube):
        block_index = 0

        with tf.variable_scope(self.name):
            if self.reuse:
                tf.get_variable_scope().reuse_variables()

            l1 = op.conv3d(cube, self.channels, filter_size=7, stride=1, name='conv3d_first', activation=tf.nn.relu, normalization=True, padding='SAME')           
            print(l1.get_shape().as_list())
        
            # 3*3*# max pooling layer is located
            l2 = op.max_pool3d(l1, filter_size=3, stride=2, padding='SAME')
            print(l2.get_shape().as_list())
            
            l3 = op.block(l2, self.channels, group=self.group, name='block_%d'%block_index, training=self.training)
            block_index += 1
            print(l3.get_shape().as_list())

            l4 = op.block(l3, self.channels, group=self.group, name='block_%d'%block_index, training=self.training, down_sample=False)
            block_index += 1
            print(l4.get_shape().as_list())

#            l6 = op.global_avg_pool(l5)
#            print(l6.get_shape().as_list())
            l4 = tf.reshape(l4, [-1, np.prod(l4.get_shape().as_list()[1:])])
            print(l4.get_shape().as_list())

            l5 = op.fc(l4, self.channels, name='fc', normalization=True, training=self.training, bias=True)  
            print(l5.get_shape().as_list())

            self.logits = op.fc(l5, self.num_classes, activation=None, name='pre-softmax', normalization=False, training=self.training, bias=True) 


    def _get_trainable_vars(self, scope):
        tf.logging.info('Getting trainable variables')
        is_trainable = lambda x: x in tf.trainable_variables()

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        var_list = list(filter(is_trainable, var_list))

        for i in var_list:
            print(i.op.name)

        return var_list

    def train_op(self, scope, loss, learning_rate, global_steps=None, clip_norm=5.0):
        var_list = self._get_trainable_vars(scope)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        
        # For batch normalization moving average update
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
        
            grad, var = zip(*optimizer.compute_gradients(loss, var_list=var_list))
            clipped_gradients = [gradient if gradient is None else tf.clip_by_value(gradient, -clip_norm, clip_norm) for gradient in grad]

            if global_steps is not None:
                optim = optimizer.apply_gradients(zip(clipped_gradients, var), global_step=global_steps)
            else:
                optim = optimizer.apply_gradients(zip(clipped_gradients, var))

        return optim


    def create_objective(self, labels):
        self.classification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=self.logits))
        
        return self.classification_loss

    def summary(self):
        if self.training:
            loss_summary = tf.summary.scalar('Train_loss', self.classification_loss)
            acc_summary = tf.summary.scalar('Train_accuracy', self.accuracy)
        else:
            loss_summary = tf.summary.scalar('Valid_loss', self.classification_loss)
            acc_summary = tf.summary.scalar('Valid_accuracy', self.accuracy)

        return tf.summary.merge([loss_summary, acc_summary])

    def metric(self, labels):
        correct_prediction = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(labels, axis=1)) 
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        return self.accuracy


if __name__ == "__main__":
    x = tf.get_variable('test', [1,61,61,61,1])
    resnetx(x, num_classes=9, channels=64, training=True) 
