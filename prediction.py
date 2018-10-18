import tensorflow as tf
import numpy as np
import dataset
import utils
import importlib
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec


class Prediction(object):
    def __init__(self, args, sess):
        self.args = args
        self.sess = sess

        self.save_dir = os.path.expanduser(self.args.save_dir)
        self.data_dir = os.path.expanduser(self.args.data_dir) 
        
        self.cube_size = self.args.cube_incr*2 + 1
        self.segy_array, address_label, section, self.num_classes = dataset.get_segy_pts(self.data_dir, self.args.cube_incr, save=False)

        try:
            # Getting model
            tf.logging.info('Getting %s model' % self.args.model)
            model_path = importlib.import_module(self.args.model_dir + '.' + self.args.model)
            seismic_model = getattr(model_path, self.args.model)

        except:
            raise ImportError

        # Define section
        self.pred_section = list()
        # inline start
        self.pred_section.append((self.args.section_list[0] - section[2]) // section[3])
        # inline end
        self.pred_section.append((self.args.section_list[1] - section[2]) // section[3])
        # xline start
        self.pred_section.append((self.args.section_list[2] - section[6]) // section[7])
        # xline end
        self.pred_section.append( (self.args.section_list[3] - section[6]) // section[7])
        # time start
        self.pred_section.append((self.args.section_list[4] - section[10]) // section[11])
        # time end
        self.pred_section.append((self.args.section_list[5] - section[10]) // section[11])

        if self.args.xline_ref < section[6] or self.args.xline_ref > section[5]:
            raise ValueError('Not valid xline %d')
        else: 
            self.xline_ref = (self.args.xline_ref - section[6]) // section[7] 

        with tf.name_scope('Prediction'):
            self.pred_seismic_model = seismic_model(self.num_classes, self.args.channels, training=False, name=self.args.model, group=self.args.group, reuse=False)

            self.cube = tf.placeholder(tf.float32, [None, self.cube_size, self.cube_size, self.cube_size, 1])

            self.pred_seismic_model(self.cube)

        self.saver = tf.train.Saver()


    def pred(self):
        self.sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        if self.load(self.save_dir):
            tf.logging.info('%s loaded' % os.path.join(self.save_dir, self.model_dir))
        else:
            raise Exception('Not loaded')

        try:
            # Make numpy array to hold prediction result, we only show classes, not prediction
            inline_length = self.pred_section[1] - self.pred_section[0] + 1
            xline_length = self.pred_section[3] - self.pred_section[2] + 1
            time_depth = self.pred_section[5] - self.pred_section[4] + 1
            if np.mod(time_depth,self.args.pred_batch) != 0:
                raise Exception('Prediction batch size must be dividable by time depth %d' % time_depth)
            else:
                tf.logging.info('Inline length: %d, xline length: %d, time depth: %d, prediction batch size: %d' % (inline_length, xline_length, time_depth, self.args.pred_batch))
            total_length = inline_length * xline_length*time_depth
            print('Total spot to estimate: %d' % total_length)

            prediction = np.empty([total_length, 1], dtype=np.float32)

            # Make data cube to feed to the network
            data = np.empty([self.args.pred_batch, self.cube_size, self.cube_size, self.cube_size, 1], dtype=np.float32)

            # index for counting each batch to put in prediction
            index = 0
            # Iterate through inline/xline/time
            for inline in range(self.pred_section[0], self.pred_section[1]+1):
                for xline in range(self.pred_section[2], self.pred_section[3]+1):
                    for t_line in range(self.pred_section[4], self.pred_section[5]+1):
                        for i in range(self.args.pred_batch):
                            data[i,:,:,:,:] = utils.cube_parse(self.segy_array, cube_incr=self.args.cube_incr, inline_num=inline, xline_num=xline, depth=t_line+i)                 
                        classes_ = self.sess.run(tf.argmax(tf.nn.softmax(self.pred_seismic_model.logits), axis=-1), feed_dict={self.cube:data})                
                        # Expand dimension to [batch size, 1]
                        prediction[index*self.args.pred_batch, :] = np.expand_dims(classes_, axis=1)
                        index += 1 
                        tf.logging.info('Index %d => Inline: %d, xline: %d, time depth: %d, class %d' % (index, inline, xline, t_line, np.squeeze(classes_)))   

            # Check loops
            if not index*self.args.pred_batch == total_length:
                raise Exception

            print('\tReshape prediction')
            prediction = prediction.reshape([inline_length, xline_length, time_depth, -1])

            if self.args.visualization:
                plt.figure(figsize=(20,20))
                # 1 row, 2 cols
                gs = gridspec.GridSpec(1, 2, width_ratios=[3,1])

                # Plot seismic image at xline ref
                plt.subplot(gs[0])                                             
                plt.title('%d x-line' % self.xline_ref)
                # x-axis: inline, y-aixs: t 
                # cmap: color distribution, extent: [horizontal min/max, vertical min/max] 
                plt.imshow(self.segy_array[self.pred_section[0]:self.pred_section[1]+1,self.xline_ref,self.pred_section[4]:self.pred_section[5]+1].T, interpolation="nearest", cmap="gray", extent=[self.pred_section[0],self.pred_section[1],-self.pred_section[5],-self.pred_section[4]])
                # Add color bar
                plt.colorbar()
                plt.show()

                # Plot classification
                plt.subplot(gs[1])
                plt.title('Classification')
                plt.imshow(prediction[:,self.xline_ref-self.pred_section[2], :,0].T, interpolation="nearest", cmap="gist_rainbow", clim=(0.0, self.num_classes-1), extent=[self.pred_section[0],self.pred_section[1],-self.pred_section[5],-self.pred_section[4]])
                plt.colorbar()
                plt.show()

 
        except:
            raise ValueError


        finally:
            tf.logging.info('\nPrediction finished')



    def load(self, ckpt_dir):
        checkpoint_dir = os.path.join(ckpt_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            tf.logging.info('Checkpoint path: %s' % ckpt_name)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

            return True

        else:
            raise Exception('Checkpoint not loaded')

    @property
    def model_dir(self):
        return '{}_{}cubesize_{}channels'.format(self.args.model, self.args.cube_incr*2+1, self.args.channels)

