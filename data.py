import torch
import torch.utils.data as data
import numpy as np
import h5py


class PUDataset(data.Dataset):
    def __init__(self, args):
        super(PUDataset, self).__init__()

        self.args = args
        # input and gt: (b, n, 3) radius: (b, 1)
        self.input_data, self.gt_data, self.radius_data, self.val, self.x6 = load_h5_data(args)

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, index):
        # (n, 3)
        input = self.input_data[index]
        gt = self.gt_data[index]
        radius = self.radius_data[index]
        val = self.val[index]
        x6 = self.x6[index]
        if self.args.use_random_input:
            sample_idx = nonuniform_sampling(input.shape[0], sample_num=self.args.num_points)
            input = input[sample_idx, :]
        # data augmentation
        if self.args.data_augmentation:
            if self.args.dataset == 'pugan':
                input = jitter_perturbation_point_cloud(input, sigma=self.args.jitter_sigma, clip=self.args.jitter_max) 
            
            input, gt, val, x6= rotate_point_cloud_and_gt(input, gt, val, x6)
            input, gt, val, x6 , scale = random_scale_point_cloud_and_gt(input, gt, val, x6,scale_low=0.8, scale_high=1.2)
            radius = radius * scale
        # ndarray -> tensor
        input = torch.from_numpy(input)
        gt = torch.from_numpy(gt)
        radius = torch.from_numpy(radius)
        val = torch.from_numpy(val)
        x6 = torch.from_numpy(x6)

        return input, gt, radius, val, x6





# load and normalize data
def load_h5_data(args):
    num_points = args.num_points
    num_4X_points = int(args.num_points * 4)
    num_out_points = int(args.num_points * 20)
    num_6X_points = int(args.num_points * 6)
    skip_rate = args.skip_rate
    use_random_input = args.use_random_input
    h5_file_path = args.h5_file_path

    if use_random_input:
        with h5py.File(h5_file_path, 'r') as f:
            # (b, n, 3)
            input = f['poisson_%d' % num_4X_points][:]
            # (b, n, 3)
            x20 = f['poisson_%d' % num_out_points][:]
            val = f['poisson_%d' % num_4X_points][:]
            x6 = f['poisson_%d' % num_6X_points][:]
    else:
        with h5py.File(h5_file_path, 'r') as f:
            input = f['poisson_%d' % num_points][:]
            x20 = f['poisson_%d' % num_out_points][:]
            val = f['poisson_%d' % num_4X_points][:]
            x6 = f['poisson_%d' % num_6X_points][:]

    # (b, n, c)
    assert input.shape[0] == x20.shape[0]

    # (b, 1)
    data_radius = np.ones(shape=(input.shape[0], 1))
    # the center point of input
    input_centroid = np.mean(input, axis=1, keepdims=True)
    input = input - input_centroid
    # (b, 1)
    input_furthest_distance = np.amax(np.sqrt(np.sum(input ** 2, axis=-1)), axis=1, keepdims=True)
    # normalize to a unit sphere
    input = input / np.expand_dims(input_furthest_distance, axis=-1)
    x20 = x20 - input_centroid
    x20 = x20 / np.expand_dims(input_furthest_distance, axis=-1)

    val = val - input_centroid
    val = val / np.expand_dims(input_furthest_distance, axis=-1)

    x6 = x6 - input_centroid
    x6 = x6 / np.expand_dims(input_furthest_distance, axis=-1)

    input = input[::skip_rate]
    x20 = x20[::skip_rate]
    data_radius = data_radius[::skip_rate]
    val = val[::skip_rate]
    x6 = x6[::skip_rate]
    print(f"input shape: {input.shape} x20 shape: {x20.shape} val shape: {val.shape} x6 shape: {x6.shape}")

    return input, x20, data_radius, val, x6


# nonuniform sample point cloud to get input data
def nonuniform_sampling(num, sample_num):
    sample = set()
    loc = np.random.rand() * 0.8 + 0.1
    while len(sample) < sample_num:
        a = int(np.random.normal(loc=loc, scale=0.3) * num)
        if a < 0 or a >= num:
            continue
        sample.add(a)
    return list(sample)


# data augmentation
def jitter_perturbation_point_cloud(input, sigma=0.005, clip=0.02):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, jittered batch of point clouds
    """
    N, C = input.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    jittered_data += input
    return jittered_data


def rotate_point_cloud_and_gt(input, gt=None, val = None, x6=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, rotated batch of point clouds
    """
    angles = np.random.uniform(size=(3)) * 2 * np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))
    input = np.dot(input, rotation_matrix)
    if gt is not None:
        gt = np.dot(gt, rotation_matrix)
    if val is not None:
        val = np.dot(val, rotation_matrix)
    if x6 is not None:
        x6 = np.dot(x6, rotation_matrix)
    return input, gt, val, x6


def random_scale_point_cloud_and_gt(input, gt=None, val=None, x6 = None,scale_low=0.5, scale_high=2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            Nx3 array, original batch of point clouds
        Return:
            Nx3 array, scaled batch of point clouds
    """
    scale = np.random.uniform(scale_low, scale_high)
    input = np.multiply(input, scale)
    if gt is not None:
        gt = np.multiply(gt, scale)
    if val is not None:
        val = np.multiply(val, scale)
    if x6 is not None:
        x6 = np.multiply(x6, scale)
    return input, gt, val, x6 , scale