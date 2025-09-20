import argparse

class args:
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train',
                        help='train or test [default: train]')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use [default: GPU 0]')
    parser.add_argument('--name', default='release',
                        help="experiment name, prepended to log_dir")
    parser.add_argument('--log_dir', default='./model',
                        help='Log dir [default: log]')
    parser.add_argument('--result_dir', default ="./model/test/result", help='result directory')
    parser.add_argument('--ckpt', help='model to restore from')
    parser.add_argument('--num_point', type=int,  default='256',
                        help='Input Point Number [default: 256]')
    parser.add_argument('--num_shape_point', type=int, default='256',
                        help="Number of points per shape")
    parser.add_argument('--up_ratio', type=int, default=4,
                        help='Upsampling Ratio [default: 2]')
    parser.add_argument('--max_epochs', type=int, default=200,
                        help='Epoch to run [default: 200] 100 for stage 1, and another 100 for stage 2')
    parser.add_argument('--stage_1_max_epochs', type=int, default=1,
                        help='Epoch to run [default: 200] 100 for stage 1, and another 100 for stage 2')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch Size during training')
    parser.add_argument('--num_samples', type=int, default=6, 
                        help='Number of samples per Gaussian [default: 6] options: 4,6,20')
    parser.add_argument('--distribution', type=str, default='gaussian', 
                        help= 'Distribution type [default: gaussian] options: gaussian, uniform')
    parser.add_argument('--training_stage', type=int, default=1,
                        help='Training stage [default: 1] for stage 1, only train the Gaussian module; for stage 2, train the whole network. If you want to train the whole network from scratch, set training_stage as 2')
    
    


    #### PU1K
     # dataset
    parser.add_argument('--dataset', default='pugan', type=str, help='pu1k or pugan')
    parser.add_argument('--h5_file_path', default="/data/datasets/pugan_x20.h5", type=str, help='the path of train dataset') # /data/dataset.h5 is pugan our way
    parser.add_argument('--num_points', default=256, type=int, help='the points number of each input patch')
    parser.add_argument('--skip_rate', default=1, type=int, help='used for dataset')
    parser.add_argument('--use_random_input', default=False, type=bool, help='whether use random sampling for input generation')
    parser.add_argument('--jitter_sigma', type=float, default=0.01, help="jitter augmentation")
    parser.add_argument('--jitter_max', type=float, default=0.03, help="jitter augmentation")
    parser.add_argument('--data_augmentation', default=True, type=bool, help='whether use data augmentation')
    parser.add_argument('--test_data', default = '/data/PU-GAN/test_pointcloud/input_256_4X/input_256/', help='test data path')
    parser.add_argument('--gt_path', default = '/data/PU-GAN/test_pointcloud/input_256_4X/gt_1024/', help='test gt data path')
    parser.add_argument('--pugan_path', default="/data/datasets/pugan_x20.h5", type=str, help='the path of train dataset') # /data/dataset.h5 is pugan our way
    parser.add_argument('--pu1k_path', default="/data/datasets/pu1k_x20.h5", type=str, help='the path of train dataset') # /data/dataset.h5 is pugan our way
    parser.add_argument('--r', type=int, default=4, help='upsampling rate')



    parser.add_argument('--model_dir', default='models', help='model dir')
    parser.add_argument('--decay_iter', type=int, default=50000)
    parser.add_argument('--lr_init', type=float, default=0.001)
    parser.add_argument('--restore_epoch', type=int, default=11)
    parser.add_argument('--save_interval', type=int, default=10,
                        help='save_step during training')
    parser.add_argument('--print_step', type=int, default=100,
                        help='print_step during training')
    parser.add_argument('--patch_num_ratio', type=float, default=3)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='gaussian')
    parser.add_argument('--exp_number', type=int, default=1)

    # testing
    parser.add_argument('--save_dir', default='./results', type=str, help='Save directory for results')
    parser.add_argument('--up_rate', default=4, type=int, help='upsampling rate')
    parser.add_argument('--test_input_path', default='./data/PU1K/test/input_2048/', type=str, help='Input point clouds directory') # add the path of the input point clouds test set
    parser.add_argument('--test_gt_path', default='./data/PU1K/test/gt_8192/', type=str, help='Ground truth point clouds directory') # add the path of the gt point clouds test set
    

    args = parser.parse_args()
