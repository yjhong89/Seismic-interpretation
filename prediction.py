import tensorflow as tf
import numpy as np
import dataset
import importlib


class prediction(object):
    def __init__(self, args, sess):
        self.args = args
        self.sess = sess

        self.save_dir = os.path.expanduser(self.args.save_dir)
        self.data_dir = os.path.expanduser(self.args.data_dir) 
        
        self.cube_size = self.args.cube_incr*2 + 1
        self.segy_array, address_label, section, num_classes = dataset.get_segy_pts(self.data_dir, self.args.cube_cinr, save=False)

        try:
            # Getting model
            tf.logging.info('Getting %s model' % self.args.model)
            model_path = importlib.import_module(self.args.model_dir + '.' + self.args.model)
            seismic_model = getattr(model_path, self.args.model)

        except:
            raise ImportError

        with tf.name_Scope('batch'):
            self.pred_addr_label_batch = dataset.get_batches(address_label, section, self.cube_size, num_classes, self.args.pred_batch_size, self.sess, 'valid', self.args.valid_split)

        with tf.name_scope('Prediction'):
            self.pred_seismic_model = seismic_model(num_classes, self.args.channels, training=False, name=self.args.model, group=selfargs.group, reuse=False)

            self.cube = tf.placeholder(tf.float32, [None, self.cuibe_size, self.cube_size, self.cube_size, 1])
            self.labels = tf.placeholder(tf.int32, [None, num_classes])

            self.pred_seismic_model(self.cube)

            with tf.name_scope('accuracy'):
                self.pred_acc = self.pred_seismic_model.metric(self.labels)
        
        self.saver = tf.train.Saver()


    def pred(self):
        self.sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        if self.load(self.save_dir):
            tf.logging.info('%s checkpoint loaded' % os.path.join(self.save_dir, self.model_dir))
        else:
            raise Exception('Not loaded')

        try:
    

        except KeyboardInterrupt:
            pass


        finally:
            tf.logging.info('\nPrediction finished')



    def load(self, ckpt_dir):
        checkpoint_dir = os.path.join(ckpt_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt_name)

            tf.logging.info('Restoring checkpoint')

        else:
            raise Exception('Checkpoint not loaded')

    @property
    def model_dir(self):
        return '{}_{}cubesize_{}channels'.format(self.args.model, self.args.cube_incr*2+1, self.args.channels)

