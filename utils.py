import numpy as np
import segyio
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys


### ---- Functions for Input data(SEG-Y) formatting and reading ----
# Make a function that decompresses a segy-cube and creates a numpy array, and
# a dictionary with the specifications, like in-line range and time step length, etc.
def segy_decomp(segy_file, plot_data = False, read_direc='xline', inp_res = np.float64):
    # segy_file: filename of the segy-cube to be imported
    # plot_data: boolean that determines if a random xline should be plotted to test the reading
    # read_direc: which way the SEGY-cube should be read; 'xline', or 'inline'
    # inp_res: input resolution, the formatting of the seismic cube (could be changed to 8-bit data)

    # Make an empty object to hold the output data
    print('Starting SEG-Y decompressor')
    output = segyio.spec()

    # open the segyfile and start decomposing it
    with segyio.open(segy_file, "r" ) as segyfile:
        # Memory map file for faster reading (especially if file is big...)
        segyfile.mmap()

        # Store some initial object attributes
        output.inl_start = segyfile.ilines[0]
        output.inl_end = segyfile.ilines[-1]
        output.inl_step = segyfile.ilines[1] - segyfile.ilines[0]
        print('Inline: start ',segyfile.ilines[0], 'end ', segyfile.ilines[-1], 'step ', segyfile.ilines[1]- segyfile.ilines[0])

        output.xl_start = segyfile.xlines[0]
        output.xl_end = segyfile.xlines[-1]
        output.xl_step = segyfile.xlines[1] - segyfile.xlines[0]
        print('xline: start ',segyfile.xlines[0], 'end ', segyfile.xlines[-1], 'step ', segyfile.xlines[1]- segyfile.xlines[0])

        output.t_start = int(segyfile.samples[0])
        output.t_end = int(segyfile.samples[-1])
        output.t_step = int(segyfile.samples[1] - segyfile.samples[0])
        print('timeline: start ',segyfile.samples[0], 'end ', segyfile.samples[-1], 'step ', segyfile.samples[1]- segyfile.samples[0])


        # Pre-allocate a numpy array that holds the SEGY-cube
        output.data = np.empty((len(segyfile.ilines), len(segyfile.xline), (output.t_end - output.t_start)//output.t_step+1), dtype = np.float32)
        print('Segy shape ', output.data.shape)

        # Read the entire cube line by line in the desired direction
        if read_direc == 'inline':
            for il_index in range(len(segyfile.xline)):
                output.data[il_index,:,:] = segyfile.iline[segyfile.ilines[il_index]]

        elif read_direc == 'xline':
            for xl_index in range(len(segyfile.iline)):
                output.data[:,xl_index,:] = segyfile.xline[segyfile.xlines[xl_index]]

        elif read_direc == 'full':
            ## NOTE: 'full' for some reason invokes float32 data
            output.data = segyio.tools.cube(segy_file)
        else:
            print('Define reading direction(read_direc) using either ''inline'', ''xline'', or ''full''')


        # Convert the numpy array to span between -127 and 127 and convert to the desired format
        factor = 127/np.amax(np.absolute(output.data))
        if inp_res == np.float32:
            output.data = (output.data*factor)
        else:
            output.data = (output.data*factor).astype(dtype = inp_res)

        # If sepcified, plot a given x-line to test the read data
        if plot_data:
            # Take a given xline
            data = output.data[:,100,:]
            # Plot the read x-line, [x: ilines, y: time]
            # cmap = color map for scalar values
            plt.imshow(data.T, interpolation="nearest", cmap="gray")
            plt.colorbar()
            plt.show()


    # Return the output object
    print('Finished using the SEG-Y decompressor')
    return output


### ---- Function for making labels of each class from .pts files
def make_label(pts_files, save_dir='./', save=False, savename='addr_label'):
    if not isinstance(pts_files, list):
        raise Error

    num_classes = len(pts_files)
    print('%i classes' % num_classes)

    # Preallocate spaces for adr_list => [address, class_label]
    adr_label = np.empty([0,4], dtype=np.int32)

    # Balance
    len_array = np.zeros(len(pts_files), dtype=np.float32)
    for i in range(len(pts_files)):
        len_array[i] = len(np.loadtxt(pts_files[i], skiprows=0, usecols=range(3)))

    # Cnvert this array to a multiplier that determines how many times a given class set needs to be
    len_array /= max(len_array)
    multiplier = 1//len_array

    # Itterate through pts_files and store the class each file respectively
    for i in range(len(pts_files)):
        # Load pts file except last value, address of [ilines, xlines, time]
        # Shape of [length of lines, 3]
        pts = np.loadtxt(pts_files[i], skiprows=0, usecols = range(3), dtype=np.int32)

        print('%s pts_file assigns to label %d' % (pts_files[i], i))
        # Attach label next to address
        pts_label = np.append(pts, i*np.ones((len(pts), 1)), axis=1)
        adr_label = np.append(adr_label, pts_label, axis=0)

        # Make up label shortage
        for _ in range(int(multiplier[i]) - 1):
            pts_label = np.append(pts, i*np.ones((len(pts), 1)), axis=1)
            adr_label = np.append(adr_label, pts_label, axis=0)


    if save:
        np.savetxt(os.path.join(save_dir, savename + '.npz'), adr_label, fmt= '%i')

    return adr_label, num_classes
    

def cube_parse(segy_array, cube_incr, inline_num, xline_num, depth):
    cube_size = cube_incr*2 + 1
    num_channels = 1

    # Output array
    example = np.empty([1, cube_size, cube_size, cube_size, num_channels], dtype=np.float32)

    mini_cube = segy_array[inline_num-cube_incr:inline_num+cube_incr+1, xline_num-cube_incr:xline_num+cube_incr+1, depth-cube_incr:depth+cube_incr+1]    

    # NDHWC
    cube_data = np.moveaxis(mini_cube, -1, 0)
    cube_data = np.expand_dims(cube_data, axis=-1)

    example[0,:,:,:,:] = cube_data

    return example
        

def create_tfrecord(segy_array, addr_label, section, cube_incr, tfrecord_dir, num_classes, data_type='train', valid_split=0.2):
    cube_size = 2*cube_incr + 1

    # Divide into train/valid data
        # Inline, xline, time, label
    if data_type == 'train':
        address_label = addr_label[int(valid_split*len(addr_label)):,:]
    elif data_type == 'valid':
        address_label = addr_label[:int(valid_split*len(addr_label)),:]
    else:
        raise Exception('%s type is no proper' % data_type)

    tf.logging.info('%i examples in %s dataset' % (len(address_label), data_type))

    tfrecord_path = os.path.join(tfrecord_dir, data_type + '.tfrecord')

    with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
        num_addr_label = len(address_label)
        for i in range(num_addr_label):
            if ((address_label[i][0] >= section[0] and address_label[i][0] < section[1]) and \
                (address_label[i][1] >= section[4] and address_label[i][1] < section[5]) and \
                (address_label[i][2] >= section[8] and address_label[i][2] < section[9])):

                index = [int((address_label[i][0]-section[2])/section[3]), int((address_label[i][1]-section[6])/section[7]), int((address_label[i][2]-section[10])/section[11])]
                
                # 3D mini cube which has size of 2*cube_incr+1 for each dimension
                mini_cube = segy_array[index[0]-cube_incr:index[0]+cube_incr+1, index[1]-cube_incr:index[1]+cube_incr+1, index[2]-cube_incr:index[2]+cube_incr+1]

                # Get label of 3d mini-cube and make one hot encoded
                label = address_label[i][-1]
                label = _get_one_hot(label, num_classes)

                print('Address', address_label[i][:3], 'Label', label)
                example = _make_example(mini_cube, label) 

                writer.write(example.SerializeToString())                

            else:
                continue

        tf.logging.info('%s file has been completed' % tfrecord_path)

        sys.stdout.write('\n')
        sys.stdout.flush()


def _get_one_hot(integer_label, num_classes):
    if not isinstance(integer_label, int):
        label = int(integer_label)

    one_hot = np.zeros((num_classes,))
    one_hot[label] = 1

    return one_hot
                    
def _make_example(mini_cube, label):
    # Get raw data bytes of numpy array
    return tf.train.Example(features=tf.train.Features(feature={
                'cube_array': _bytes_features(tf.compat.as_bytes(mini_cube.tostring())),
                'label': _bytes_features(tf.compat.as_bytes(label.tostring()))
                }))

def _bytes_features(value):
    if not isinstance(value, (tuple, list, np.ndarray)):
        value = [value]

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))







if __name__ == "__main__":
    segy_file = 'data/F3_entire.segy'
    output = segy_decomp(segy_file, plot_data=True) 
    print(len(output.data))
