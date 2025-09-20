
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import sys
import argparse

from data import PUDataset
import torch.optim as optim
from glob import glob
import open3d as o3d
from einops import repeat
from utils.model_utils import *
from time import time
from datetime import datetime
import numpy as np 
from model.PUCRN import CRNet as pu_crn
from configs import args as cnfg
from train import PointCloudModel
from model.models import PU_Gaussian, Just_gaussian
from model.GaussianDistribution import GaussianDistribution
import matplotlib.pyplot as plt

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
    #model.training_stage = 3
    # coarse_pts,_, _= model.forward_test(patches)
    # coarse_pts, _ = model.forward(patches)
    coarse_pts, _, _ = model.forward_test(patches)
    coarse_pts = centroid + coarse_pts * furthest_distance
    coarse_pts = rearrange(coarse_pts, 'b c n -> c (b n)').contiguous()
    coarse_pts = FPS(coarse_pts.unsqueeze(0), input_pcd.shape[-1] * args.up_rate)
    return coarse_pts

def test(model, args):
    """Test the model on input point clouds."""
    with torch.no_grad():
        model.eval()
        test_input_path = glob(os.path.join(args.input_dir, '*.xyz'))
        total_cd = 0
        counter = 0
        txt_result = []
        total_pcd_time = 0.0

        for i, path in enumerate(test_input_path):
            # Load point cloud
            start = time()

            pcd = o3d.io.read_point_cloud(path)
            pcd_name = path.split('/')[-1]
            
            # Load ground truth
            gt = torch.Tensor(np.asarray(o3d.io.read_point_cloud(os.path.join(args.gt_dir, pcd_name)).points)).unsqueeze(0).cuda()
            
            # Prepare input point cloud
            input_pcd = np.array(pcd.points)
            input_pcd = torch.from_numpy(input_pcd).float().cuda()
            input_pcd = rearrange(input_pcd, 'n c -> c n').contiguous()
            input_pcd = input_pcd.unsqueeze(0)

            # Normalize and upsample
            input_pcd, centroid, furthest_distance = normalize_point_cloud(input_pcd)
            pcd_upsampled = upsampling(args, model, input_pcd)
            pcd_upsampled = centroid + pcd_upsampled * furthest_distance

            # Optional second upsampling for 16x
            if args.r == 16:
                pcd_upsampled, centroid, furthest_distance = normalize_point_cloud(pcd_upsampled)
                pcd_upsampled = upsampling(args, model, pcd_upsampled)
                pcd_upsampled = centroid + pcd_upsampled * furthest_distance

            # Save upsampled point cloud
            saved_pcd = rearrange(pcd_upsampled.squeeze(0), 'c n -> n c').contiguous()
            saved_pcd = saved_pcd.detach().cpu().numpy()
            save_folder = os.path.join(args.save_dir, 'xyz')
            os.makedirs(save_folder, exist_ok=True)
            np.savetxt(os.path.join(save_folder, pcd_name), saved_pcd, fmt='%.6f')
            end = time()
            total_pcd_time += end - start
            
            # Compute Chamfer distance
            cd = chamfer_sqrt(pcd_upsampled.permute(0,2,1).contiguous(), gt).cpu().item()  
            txt_result.append(f'{pcd_name}: {cd * 1e3}')    
            total_cd += cd
            counter += 1.0
            print(f" {i+1}/{len(test_input_path)}: {pcd_name} - CD: {cd * 1e3:.4f} mm")

        # Write results
        txt_result.append(f'overall: {total_cd/counter*1e3}')
        with open(os.path.join(args.save_dir, 'cd.txt'), "w") as f:
            for result in txt_result:
                f.writelines(result + '\n')
                
        print(f"average time per point cloud: {total_pcd_time/len(test_input_path)}")
    return total_cd/counter*1e3 , total_pcd_time/len(test_input_path)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='PU-Gaussian Test')
    parser.add_argument('--dataset', default='pugan', type=str, help='Dataset: pu1k or pugan')
    parser.add_argument('--r', default=4, type=int, help='Upsampling rate')
    parser.add_argument('--flexible', action='store_true', help='Use flexible upsampling')
    parser.add_argument('--test_input_path', default='/data/PU-GAN/test_pointcloud/input_2048_4X/input_2048', type=str, help='Input point clouds directory')  # uncomment for PU-GAN 
    parser.add_argument('--test_gt_path', default='/data/PU-GAN/test_pointcloud/input_2048_4X/gt_8192', type=str, help='Ground truth point clouds directory')
    # parser.add_argument('--test_input_path', default='/data/PU1K/test/input_2048/input_2048', type=str,help='Input point clouds directory') # add the path of the input point clouds test set
    # parser.add_argument('--test_gt_path', default='/data/PU1K/test/input_2048/gt_8192', type=str, help='Ground truth point clouds directory') # add the path of the gt point clouds test set
    parser.add_argument('--save_dir', default='./results', type=str, help='Save directory for results')
    parser.add_argument('--ckpt', default= 'pretrained_model/pu_gaussian_pugan_Best.pth', type=str, help='Model checkpoint path') # change to pugan model or pu1k model accordingly
    parser.add_argument('--num_points', default=256, type=int, help="Number of points in each patch")
    parser.add_argument('--patch_rate', default=3, type=int, help='used for patch generation')
    parser.add_argument('--up_rate', default=4, type=int, help='upsampling rate')
    parser.add_argument('--num_samples', default=6, type=int, help='Number of samples per Gaussian')
    parser.add_argument('--distribution', default='gaussian', type=str, help='Distribution type: gaussian or uniform')
    parser.add_argument('--training_stage', default=1, type=int, help='Training stage: 1 for Gaussian, 2 for full model')

    args = parser.parse_args()
    args.input_dir = args.test_input_path
    args.gt_dir = args.test_gt_path
     


    # Load model
    model = PointCloudModel(PU_Gaussian(args),phase='test',config=cnfg)

    model = model.net.cuda()
    model.load_state_dict(torch.load(args.ckpt))
    overall_cd = test(model, args)
    print(f"Overall Chamfer Distance: {overall_cd}")

 
if __name__ == '__main__':
    main()