
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import sys
sys.path.append('.')
sys.path.append('./utils')
sys.path.append('./models')
import argparse
from data import PUDataset
import torch.optim as optim
from glob import glob
import open3d as o3d
from einops import repeat
from utils.model_utils import *
import time
from datetime import datetime
import numpy as np 
import h5py

"""
    Generate dataset for point cloud upsampling 
    if we are generating pu1k dataset we need to generate pugan beforehand to add it to the meshes
    

    """


def _normalize_point_cloud(pc):
    """Normalize point cloud to unit sphere."""
    centroid = torch.mean(pc, dim=1, keepdim=True)
    pc = pc - centroid
    furthest_distance = torch.max(torch.sqrt(torch.sum(pc**2, dim=-1, keepdim=True)), dim=1, keepdim=True)[0]
    pc = pc / furthest_distance
    return pc

def chamfer_sqrt(p1, p2):
    """Compute Chamfer distance with square root normalization."""
    d1, d2, _, _ = chamfer_dist(_normalize_point_cloud(p1), _normalize_point_cloud(p2))
    d1 = torch.mean(d1)
    d2 = torch.mean(d2)
    return (d1 + d2)

def upsampling(args, model, input_pcd):
    """Upsample point cloud using the model."""
    pcd_pts_num = input_pcd.shape[-1]
    patch_pts_num = args.num_points
    sample_num = int(pcd_pts_num / patch_pts_num * args.patch_rate)
    seed = FPS(input_pcd, sample_num)
    patches = extract_knn_patch(patch_pts_num, input_pcd, seed)
    patches, centroid, furthest_distance = normalize_point_cloud(patches)
    coarse_pts,_,_ = model.forward(patches)
    print(f" coarse_pts shape: {coarse_pts.shape}")
    coarse_pts = centroid + coarse_pts * furthest_distance
    coarse_pts = rearrange(coarse_pts, 'b c n -> c (b n)').contiguous()
    coarse_pts = FPS(coarse_pts.unsqueeze(0), input_pcd.shape[-1] * args.up_rate)
    return coarse_pts






def process_datset(args):
    # extract patches from the dataset and save them as h5 file
    
    ###### extract patches of N points from the point clouds ######\

    # load the directory of the dataset 
    if args.dataset == 'pugan':
        args.input_dir = args.pugan_input_dir
    elif args.dataset == 'pu1k':
        args.input_dir = args.pu1k_input_dir

    test_input_path = glob(os.path.join(args.input_dir, '*.xyz'))
    input = []
    gt = []
    input_4x = []
    input_6x = []
    print(test_input_path)
  
    for i, path in enumerate(test_input_path):
        pcd = o3d.io.read_point_cloud(path)
        pcd_name = path.split('/')[-1]
        
        # Load ground truth
        # gt = torch.Tensor(np.asarray(o3d.io.read_point_cloud(os.path.join(args.gt_dir, pcd_name)).points)).unsqueeze(0).cuda()
        
        # Prepare input point cloud
        input_pcd = np.array(pcd.points)
        input_pcd = torch.from_numpy(input_pcd).float().cuda()
        input_pcd = rearrange(input_pcd, 'n c -> c n').contiguous()
        input_pcd = input_pcd.unsqueeze(0)
        input_pcd, centroid, furthest_distance = normalize_point_cloud(input_pcd)


        pcd_pts_num = input_pcd.shape[-1]
        patch_pts_num = args.num_points 
        # sample num 200 if pugan and 50 if pu1k
        sample_num = 200 if args.dataset == 'pugan' else 50
        seed = FPS(input_pcd, sample_num)
        gt_patches = extract_knn_patch(patch_pts_num* args.up_rate, input_pcd, seed)
        input_patches = FPS(gt_patches, patch_pts_num)
        x4 = FPS(gt_patches, patch_pts_num *4)
        x6 = FPS(gt_patches, patch_pts_num *6)
        print(f" gt_patches shape: {gt_patches.shape} input_patches shape: {input_patches.shape} x4 shape: {x4.shape}")


        input.append(input_patches)
        gt.append(gt_patches)
        input_4x.append(x4)
        input_6x.append(x6)
    
    input = torch.cat(input,dim=0).transpose(1,2)
    gt = torch.cat(gt,dim=0).transpose(1,2)
    input_4x = torch.cat(input_4x, dim=0).transpose(1,2)
    input_6x = torch.cat(input_6x, dim=0).transpose(1,2)
    input = input.cpu().numpy()
    gt = gt.cpu().numpy()
    input_4x = input_4x.cpu().numpy()
    input_6x = input_6x.cpu().numpy()
    
    if args.dataset == 'pu1k':

        h5_file_path = args.pugan_h5
        # open the h5 file 
        with h5py.File(h5_file_path, 'r') as f:
            input_pugan = f['poisson_%d' % input.shape[1]][:]
            # (b, n, 3)
            gt_pugan = f['poisson_%d' % gt.shape[1]][:]
            val_pugan = f['poisson_%d' % input_4x.shape[1]][:]
            x6_pugan = f['poisson_%d' % input_6x.shape[1]][:]

        print(f" shapes of the dataset: input shape: {input.shape} gt shape: {gt.shape}, input_4x shape {input_4x.shape} , input_6x shape {input_6x.shape}")
        print(f" shapes of the dataset: input shape: {input_pugan.shape} gt shape: {gt_pugan.shape}, input_4x shape {val_pugan.shape}, input_6x shape {x6_pugan.shape}")

        # concatenate both datasets on the first axis
        input = np.concatenate((input, input_pugan), axis=0)
        gt = np.concatenate((gt, gt_pugan), axis=0)
        input_4x = np.concatenate((input_4x, val_pugan), axis=0)
        input_6x = np.concatenate((input_6x, x6_pugan), axis=0)


    print(f"input shape: {input.shape} gt shape: {gt.shape}, input_4x_shape {input_4x.shape} , input_6x shape {input_6x.shape}")
    # save the patches as h5 file
    h5_file_path = args.dataset + '_x' + str(args.up_rate) + '.h5'
    h5_file_path = args.save_dir + '/' + h5_file_path
    # check if save_dir exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    h5_file = h5py.File(h5_file_path, 'w')
    h5_file.create_dataset(f"poisson_{input.shape[1]}", data=input)
    h5_file.create_dataset(f"poisson_{gt.shape[1]}", data=gt)
    h5_file.create_dataset(f"poisson_{input_4x.shape[1]}", data=input_4x)
    h5_file.create_dataset(f"poisson_{input_6x.shape[1]}", data=input_6x)
    h5_file.close()

    print(f" h5 file created at {h5_file_path}")
    #print(f" h5 keys {h5_file.keys()}")
    print("dataset saved successfully")
    


def main():
    parser = argparse.ArgumentParser(description='PUEVA VNN Model Testing')
    parser.add_argument('--dataset', default='pu1k', type=str, help='Dataset: pu1k or pugan')
    parser.add_argument('--r', default=4, type=int, help='Upsampling rate')
    parser.add_argument('--flexible', action='store_true', help='Use flexible upsampling')
    parser.add_argument('--pugan_input_dir', default='data/PU-GAN/train_pointcloud/input_2048_20X/gt_40960', type=str, help='Input point clouds directory') # add the path of the input point clouds
    parser.add_argument('--pu1k_input_dir', default='data/PU1K/train_pointcloud/input_2048_20X/gt_40960', type=str, help='Input point clouds directory') # add the path of the input point clouds 
    parser.add_argument('--save_dir', default='data/datasets', type=str, help='Save directory for results') # change to your path for saving the resulting dataset
    parser.add_argument('--num_points', default=256, type=int, help="Number of points in each patch")
    parser.add_argument('--patch_rate', default=3, type=int, help='used for patch generation')
    parser.add_argument('--up_rate', default=20, type=int, help='upsampling rate')
    parser.add_argument('--pugan_h5', default='data/PU-GAN/train/pugan_x20.h5', type=str, help='the path of pugan dataset') # add pugan dataset path here if using pu1k. if using pugan dataset only, ignore this argument

    args = parser.parse_args()
    
    process_datset(args)


if __name__ == "__main__":
    main()
