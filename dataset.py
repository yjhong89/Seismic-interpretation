import tensorflow as tf
import numpy as np
import utils
import os


def get_batches(addr_label, section, cube_size, num_classes, batch_size, sess, data_type='train', valid_split=0.2, shuffle_buffer=200000):
    # Divide into train/valid data
        # Inline, xline, time, label
    if data_type == 'train':
        address_label = addr_label[int(valid_split*len(addr_label)):,:]
    elif data_type == 'valid':
        address_label = addr_label[:int(valid_split*len(addr_label)),:]
    else:
        raise Exception('%s type is no proper' % data_type)

    address_list = list()
    label_list = list()

    num_addr_label = len(address_label)
    tf.logging.info('%i examples in %s dataset' % (len(address_label), data_type))

    for i in range(num_addr_label):
        if ((address_label[i][0] >= section[0] and address_label[i][0] < section[1]) and \
            (address_label[i][1] >= section[4] and address_label[i][1] < section[5]) and \
            (address_label[i][2] >= section[8] and address_label[i][2] < section[9])):

            index = [int((address_label[i][0]-section[2])/section[3]), int((address_label[i][1]-section[6])/section[7]), int((address_label[i][2]-section[10])/section[11])]
                
            # Get label of 3d mini-cube and make one hot encoded
            label = address_label[i][-1]
            label = utils._get_one_hot(label, num_classes)

            #print('Address', address_label[i][:3], 'Label', label)
            address_list.append(index)
            label_list.append(label)

    # Due to large size of data, define Dataset in terms of placeholder
    address_placeholder = tf.placeholder(tf.float32, [None, 3])
    label_placeholder = tf.placeholder(tf.float32, [None, num_classes])

    dataset = tf.data.Dataset.from_tensor_slices((address_placeholder, label_placeholder)).shuffle(buffer_size=shuffle_buffer).repeat().batch(batch_size)

    iterator = dataset.make_initializable_iterator()

    sess.run(iterator.initializer, feed_dict={address_placeholder:address_list, label_placeholder:label_list})
    next_element = iterator.get_next()
    
    return next_element    


def get_segy_pts(data_dir, cube_incr, save_dir=None, save=False, pred=False):
    files = os.listdir(data_dir)    
    pts_files = list()
    for f in files:
        full_filename = os.path.join(data_dir, f)
        ext = os.path.splitext(full_filename)[-1]
        if ext == '.pts':
            pts_files.append(full_filename)
        elif ext == '.segy':
            segy_files = full_filename
        else:
            raise ValueError('%s type is not a data format' % ext)

    pts_files.sort()
    print('Point files', pts_files)

    segy_obj = utils.segy_decomp(segy_files, plot_data=False, inp_res = np.float32)
    # Define the buffer zone around the edge of the cube that defines the legal/illegal adresses
    inl_min = segy_obj.inl_start + segy_obj.inl_step*cube_incr
    inl_max = segy_obj.inl_end - segy_obj.inl_step*cube_incr
    xl_min = segy_obj.xl_start + segy_obj.xl_step*cube_incr
    xl_max = segy_obj.xl_end - segy_obj.xl_step*cube_incr
    t_min = segy_obj.t_start + segy_obj.t_step*cube_incr
    t_max = segy_obj.t_end - segy_obj.t_step*cube_incr

    # Print the buffer zone edges
    print('Defining the buffer zone:')
    print('(inl_min,','inl_max,','xl_min,','xl_max,','t_min,','t_max)')
    print('(',inl_min,',',inl_max,',',xl_min,',',xl_max,',',t_min,',',t_max,')')
    section = [inl_min, inl_max, segy_obj.inl_start, segy_obj.inl_step, xl_min, xl_max, segy_obj.xl_start, segy_obj.xl_step, t_min, t_max, segy_obj.t_start, segy_obj.t_step]

    adr_label, num_classes = utils.make_label(pts_files, save_dir=save_dir, save=save) 
    # Shuffle
        # np.take(arr, indices, axis=3) is equivalent to arr[:,:,:,indices,...]. If provided parameter 'out', the result will be placed in this array
    adr_label = np.take(adr_label, np.random.permutation(len(adr_label)), axis=0, out=adr_label)

    if pred:
        label_dict = utils.label_dict(pts_files)
        return segy_obj.data, adr_label, section, num_classes, label_dict
    else:
        return segy_obj.data, adr_label, section, num_classes


def make_segy_batch(segy_array, address, cube_incr, num_channels=1):
    cube_size = cube_incr*2 + 1
    batch_size = address.shape[0]
    
    segy_batch = np.empty((batch_size, cube_size, cube_size, cube_size, num_channels))

    # Make it integer for indexing
    address = address.astype(int)

    for i in range(batch_size):
        # 3D mini cube which has size of 2*cube_incr+1 for each dimension
        mini_cube = segy_array[address[i][0]-cube_incr:address[i][0]+cube_incr+1, address[i][1]-cube_incr:address[i][1]+cube_incr+1, address[i][2]-cube_incr:address[i][2]+cube_incr+1]

        # Change its axis from (inline, xline, time) -> (time, inline, xline) for 'NDHWC'
            # np.moveaxis(arr, source_axis, dest_axis)
        cube_data = np.moveaxis(mini_cube, -1, 0)
        
        # Create channel dimension
        # TODO: More segy volumes, more channel
        cube_data = np.expand_dims(cube_data, -1)

        segy_batch[i,:,:,:,:] = cube_data 
        
    
    return segy_batch


def augmentation(segy_array, augmentation_prob):
    assert(0 < augmentation_prob < 1) 

    batch_size = segy_array.shape[0]
    
    for i in range(batch_size):
        if np.random.uniform(0,1) > augmentation_prob:
            segy_array[i,:,:,:,:] = segy_array[i, ::-1, :,:,:]
        
        if np.random.uniform(0,1) > augmentation_prob:
            segy_array[i,:,:,:,:] = segy_array[i, :, ::-1,:,:]

        if np.random.uniform(0,1) > augmentation_prob:
            segy_array[i,:,:,:,:] = segy_array[i, :,:, ::-1,:]

        # Hold the crossline
        if np.random.uniform(0,1) > augmentation_prob:
            segy_array[i,:,:,:,:] = np.transpose(segy_array[i,:,:,:,:], (1,0,2,3))

    return segy_array    



if __name__ == "__main__":
    a = get_segy_pts('data', 32, save=False)
