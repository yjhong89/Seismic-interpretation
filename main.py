import tensorflow as tf
import argparse
import os
from train import Seismic
from prediction import Prediction

os.environ['CUDA_VISIBLE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--save_dir', type=str, default='ckpt')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model_dir', type=str, default='model')

    # Model parameters
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--model', type=str, default='Resnetx')
    parser.add_argument('--group', type=int, default=32)
    parser.add_argument('--cube_incr', type=int, default=27)

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--valid_split', type=float, default=0.15)
    parser.add_argument('--num_epochs', type=int, default=100000)
    parser.add_argument('--valid_interval', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clip_norm', type=float, default=5.0)
    parser.add_argument('--augmentation', type=str2bool, default='t')
    parser.add_argument('--augmentation_prob', type=float, default=0.4)
    
    # Functions
    parser.add_argument('-d', '--delete', action='store_true')
    parser.add_argument('-l', '--tensor_log', action='store_true')
    parser.add_argument('-t', '--training', action='store_true')

    # Predictions
    parser.add_argument('--pred_batch', type=int, default=64)
    parser.add_argument('--xline_ref', type=int, default=700)
    parser.add_argument('--pred_save', type=str2bool, default='t')
    parser.add_argument('-v', '--visualize', type=str2bool, default='t')
    parser.add_argument('-s', '--section_list', type=int, dest='section_list', default=[150, 700, 350, 1200, 150, 1700], action='append') 

    args = parser.parse_args()
    print(args.section_list)
    if args.tensor_log:
        tf.logging.set_verbosity('INFO')


    run_config = tf.ConfigProto()
    run_config.log_device_placement = False
    run_config.gpu_options.allow_growth = True
    run_config.allow_soft_placement = True

    
    try:
        with tf.Session(config=run_config) as sess:
            if getattr(args, 'training'):  
                seismic = Seismic(args, sess)
                seismic.train()
            else:
                pred = Prediction(args, sess)
                pred.pred()


    except:
        raise OSError

def str2bool(v):
    if v.lower() in ('t', 1, 'true'):
        return True
    else:
        return False



if __name__ == "__main__":
    main()
