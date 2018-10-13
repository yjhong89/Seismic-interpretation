import tensorflow as tf
import argparse
import shutil
import os
import data_process
import dataset
import random
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--delete', action='store_true')
    parser.add_argument('-l', '--log', action='store_true')
    parser.add_argument('-c', '--cube_incr', type=int, default=30, help='the number of increments included in each direction from the example to make a mini-cube')

    # Parsing argument
    args = parser.parse_args()
    if args.log:
        tf.logging.set_verbosity('INFO')

    # TFRecord directory
    tfrecord_dir = os.path.join(os.getcwd(), 'tfrecord')
    # Data directory include segy and label
    data_dir = os.path.join(os.getcwd(), 'data')

    if not os.path.exists(data_dir):
        raise IOError('Data does not exist')

    if args.delete:
        tf.logging.warn('Delete existing tfrecord files')
        shutil.rmtree(tfrecord_dir)

    if not os.path.exists(tfrecord_dir):
        os.makedirs(tfrecord_dir)

   
    # Get data files, and get segy object and address/label
    try:
        data_files = os.listdir(data_dir)
        pts_files = list()
        for f in data_files:
            full_filename = os.path.join(data_dir, f)
            ext = os.path.splitext(full_filename)[-1]
            if ext == '.pts':
                pts_files.append(full_filename)
            elif ext == '.segy':
                segy_files = full_filename
            else:
                raise ValueError('%s type is not a data format' % ext)

        pts_files.sort()

        segy_obj = data_process.segy_decomp(segy_files, plot_data=False)
        # Define the buffer zone around the edge of the cube that defines the legal/illegal adresses
        inl_min = segy_obj.inl_start + segy_obj.inl_step*args.cube_incr
        inl_max = segy_obj.inl_end - segy_obj.inl_step*args.cube_incr
        xl_min = segy_obj.xl_start + segy_obj.xl_step*args.cube_incr
        xl_max = segy_obj.xl_end - segy_obj.xl_step*args.cube_incr
        t_min = segy_obj.t_start + segy_obj.t_step*args.cube_incr
        t_max = segy_obj.t_end - segy_obj.t_step*args.cube_incr

        # Print the buffer zone edges
        print('Defining the buffer zone:')
        print('(inl_min,','inl_max,','xl_min,','xl_max,','t_min,','t_max)')
        print('(',inl_min,',',inl_max,',',xl_min,',',xl_max,',',t_min,',',t_max,')')
        section = [inl_min, inl_max, segy_obj.inl_start, segy_obj.inl_step, xl_min, xl_max, segy_obj.xl_start, segy_obj.xl_step, t_min, t_max, segy_obj.t_start, segy_obj.t_step]

        adr_label, num_classes = data_process.make_label(pts_files, save_dir=tfrecord_dir, save=True) 
        # Shuffle
            # np.take(arr, indices, axis=3) is equivalent to arr[:,:,:,indices,...]. If provided parameter 'out', the result will be placed in this array
        adr_label = np.take(adr_label, np.random.permutation(len(adr_label)), axis=0, out=adr_label)

    except:
        raise OSError

    # Make tfrecord
    try:
        #for i in ['train', 'valid']:
        #    data_process.create_tfrecord(segy_obj.data, adr_label, section, args.cube_incr, tfrecord_dir, num_classes, data_type=i, valid_split=0.15)

        with tf.Session() as sess:
            dataset.get_batches(segy_obj.data, adr_label, section, args.cube_incr, num_classes, sess)
    except:
        raise Exception




if __name__ == "__main__":
    main()
