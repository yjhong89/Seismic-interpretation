import tensorflow as tf
import numpy as np
import os
import dataset
import shutil
import importlib
import time
from model.resnetx import resnetx


class Seismic(object):
    def __init__(self, args, sess):
        self.args = args
        self.sess = sess

        base_dir = os.getcwd()
        self.log_dir = os.path.join(base_dir, self.args.log_dir)
        self.save_dir = os.path.join(base_dir, self.args.save_dir)
        self.data_dir = os.path.join(base_dir, self.args.data_dir)

        if args.delete and os.path.exists(os.path.join(self.save_dir, self.model_dir)):
            shutil.rmtree(os.path.join(self.save_dir, self.model_dir))

        os.makedirs(self.save_dir, exist_ok=True)
     
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Prepare data
        if not os.path.exists(self.data_dir):
            raise Exception('%s is not data directory' % self.data_dir) 
    
        self.cube_size = self.args.cube_incr*2 + 1
        self.segy_array, address_label, section, num_classes = dataset.get_segy_pts(self.data_dir, self.args.cube_incr, self.args.log_dir)
        

        self.train_seismic_model = resnetx(num_classes, self.args.channels, training=True, name=self.args.model, group=self.args.group, reuse=False)
        self.valid_seismic_model = resnetx(num_classes, self.args.channels, training=False, name=self.args.model, group=self.args.group, reuse=True)

        # Getting address, label
        self.train_addr_label_batch = dataset.get_batches(address_label, section, self.cube_size, num_classes, self.args.batch_size, self.sess, 'train', self.args.valid_split) 
        self.valid_addr_label_batch = dataset.get_batches(address_label, section, self.cube_size, num_classes, self.args.batch_size, self.sess, 'valid', self.args.valid_split)

        self.global_steps = tf.train.get_or_create_global_step()

        with tf.name_scope('Seismic_model'):
            self.cube = tf.placeholder(tf.float32, [None, self.cube_size, self.cube_size, self.cube_size, 1]) 
            self.labels = tf.placeholder(tf.int32, [None, num_classes])        
    
            self.train_seismic_model(self.cube)
            self.valid_seismic_model(self.cube)

            with tf.name_scope('create_objective'):
                self.train_loss = self.train_seismic_model.create_objective(self.labels) 
                self.valid_loss = self.valid_seismic_model.create_objective(self.labels)

            with tf.name_scope('accuracy'):
                self.train_acc = self.train_seismic_model.metric(self.labels)
                self.valid_acc = self.valid_seismic_model.metric(self.labels)

            with tf.name_scope('summaries'):
                self.train_summary = self.train_seismic_model.summary() 
                self.valid_summary = self.valid_seismic_model.summary()

            with tf.name_scope('optimizer'):
                self.train_op = self.train_seismic_model.train_op(self.args.model, self.train_loss, self.args.learning_rate, global_steps=self.global_steps, clip_norm=self.args.clip_norm)
        
        self.saver = tf.train.Saver(max_to_keep=5)

    def train(self):
        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        # Initialize all variables
        self.sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        try:
            best_valid_loss = 100000
            best_valid_accuracy = -1000

            for epoch in range(self.args.num_epochs):
                start_time = time.time()

                train_batch = self.sess.run(self.train_addr_label_batch)
                train_segy_batch = dataset.make_segy_batch(self.segy_array, train_batch[0], self.args.cube_incr)

                _, loss_, acc_, train_summary_, steps = self.sess.run([self.train_op, self.train_loss, self.train_acc, self.train_summary, self.global_steps], feed_dict={self.cube:train_segy_batch, self.labels:train_batch[1]})

    
                writer.add_summary(train_summary_, steps)
    
                tf.logging.info('Train loss: %3.3f, accuracy: %3.3f, at epoch %d' % (loss_, acc_, epoch+1))
                    
                if np.mod(epoch+1, self.args.valid_interval) == 0:
    
                    valid_batch = self.sess.run(self.valid_addr_label_batch)
                    valid_segy_batch = dataset.make_segy_batch(self.segy_array, valid_batch[0], self.args.cube_incr)
                    
                    valid_loss_, valid_acc_, valid_summary_ = self.sess.run([self.valid_loss, self.valid_acc, self.valid_summary], feed_dict={self.cube:valid_segy_batch, self.labels:valid_batch[1]})
    
                    tf.logging.info('Valid loss: %3.3f, accruacy: %3.3f' % (valid_loss_, valid_acc_))
    
                    writer.add_summary(valid_summary_, steps)
                    writer.flush()
    
                    if best_valid_accuracy < valid_acc_:
                        best_valid_accuracy = valid_acc_
                        best_valid_loss = valid_loss_
                        self.save(self.save_dir, steps)
    

        except KeyboardInterrupt:
            print('End training')

        
        finally:
            tf.logging.info('\nTraining finished')


    def save(self, ckpt_dir, save_step):
        checkpoint_dir = os.path.join(os.getcwd(), ckpt_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)    

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_dir), global_step=save_step)    
        tf.logging.info('Checkpoint saved\n')

    @property
    def model_dir(self):
        return '{}_cube_size_{}_channels'.format(self.args.cube_incr*2+1, self.args.channels) 
        

