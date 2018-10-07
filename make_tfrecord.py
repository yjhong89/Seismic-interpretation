import tensorflow as tf
import argparse
import shutil
import os
import data_process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--delete', action='store_true')
    parser.add_argument('-l', '--log', action='store_true')
    parser.add_argument('-cube_incr', type=int, default=30. help='the number of increments included in each direction from the example to make a mini-cube')

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
        shutil.rmtree(tfrecord)

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
        adr_label = data_process.make_labels(pts_files, save_dir=data_dir, save=True) 

    except:
        raise IOError

    # Make tfrecord
    try:
         create_tfrecord(segy_obj, adr_label, args.cube_incr, tfrecord_dir, valid_split=0.15)
    except:
        raise Exception




if __name__ == "__main__":
    main()
