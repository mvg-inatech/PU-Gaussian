import os
import sys
import argparse
import numpy as np
import torch
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from einops import rearrange
import gc

# Add project root to path
sys.path.append('.')

from model.models import PU_Gaussian
from utils.model_utils import FPS, normalize_point_cloud, extract_knn_patch
from configs import args as cnfg
from train import PointCloudModel
from tqdm import tqdm


def farthest_point_sampling_cpu(points, n_samples):
    """
    Farthest point sampling on CPU using numpy.
    
    Args:
        points: numpy array of shape (N, 3)
        n_samples: number of samples to select
    
    Returns:
        indices: numpy array of selected point indices
    """
    N = points.shape[0]
    if n_samples >= N:
        return np.arange(N)
    
    selected_indices = []
    distances = np.full(N, np.inf)
    
    # Start with random point
    current_idx = np.random.randint(0, N)
    selected_indices.append(current_idx)
    
    for i in range(1, n_samples):
        # Update distances to the last selected point
        last_point = points[current_idx]
        dist_to_last = np.sum((points - last_point) ** 2, axis=1)
        distances = np.minimum(distances, dist_to_last)
        
        # Select point with maximum distance
        current_idx = np.argmax(distances)
        selected_indices.append(current_idx)
        
    return np.array(selected_indices)


def extract_patches_knn_cpu(points, seed_indices, patch_size=10000):
    """
    Extract patches around seed points using k-NN search on CPU.
    
    Args:
        points: numpy array of shape (N, 3) - full point cloud
        seed_indices: numpy array of seed point indices
        patch_size: number of points per patch
    
    Returns:
        patches: list of numpy arrays, each of shape (patch_size, 3)
    """
    # Use sklearn's NearestNeighbors for efficient k-NN search
    nbrs = NearestNeighbors(n_neighbors=min(patch_size, len(points)), algorithm='auto').fit(points)
    
    patches = []
    for seed_idx in seed_indices:
        seed_point = points[seed_idx:seed_idx+1]  # Shape (1, 3)
        
        # Find k nearest neighbors
        distances, indices = nbrs.kneighbors(seed_point)
        patch_indices = indices[0]  # Remove batch dimension
        
        # Extract patch points
        patch = points[patch_indices]
        patches.append(patch)
        
    return patches

def upsampling(args, model, input_pcd):
    """Upsample point cloud using the model."""
    pcd_pts_num = input_pcd.shape[-1]
    patch_pts_num = 256
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
    coarse_pts = FPS(coarse_pts.unsqueeze(0), input_pcd.shape[-1] * args.r)
    return coarse_pts


def upsample_patch(model, patch_points, args):
    """
    Upsample a single patch using the model.
    
    Args:
        model: trained PU-Gaussian model
        patch_points: numpy array of shape (patch_size, 3)
        args: configuration arguments
    
    Returns:
        upsampled_points: numpy array of upsampled points
    """
    # Convert to torch tensor and move to GPU
    patch_tensor = torch.from_numpy(patch_points).float().cuda()
    patch_tensor = rearrange(patch_tensor, 'n c -> c n').contiguous()
    patch_tensor = patch_tensor.unsqueeze(0)  # Add batch dimension
    
    # Normalize patch
    upsampled = upsampling(args, model, patch_tensor)
        
    # Convert back to numpy
    upsampled_np = rearrange(upsampled.squeeze(0), 'c n -> n c').contiguous()
    upsampled_np = upsampled_np.detach().cpu().numpy()
    
    # Clear GPU memory
    del patch_tensor, upsampled
    torch.cuda.empty_cache()
    gc.collect()
    
    return upsampled_np


def infer_large_pointcloud(input_path, output_path, model_path, args):
    """
    Main inference function for large point clouds.
    
    Args:
        input_path: path to input point cloud (.ply or .pcd)
        output_path: path to save upsampled point cloud
        model_path: path to trained model checkpoint
        args: configuration arguments
    """
    print(f"Loading point cloud from: {input_path}")
    
    # Load point cloud on CPU
    pcd = o3d.io.read_point_cloud(input_path)
    points = np.asarray(pcd.points)
    N = len(points)
    
    print(f"Loaded point cloud with {N} points")
    
    # Load model
    print("Loading model...")
    model = PointCloudModel(PU_Gaussian(args), phase='test', config=cnfg).net.cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Calculate number of patches needed
    patch_size = args.patch_size

    n_patches =  int(N / patch_size * args.patch_rate)  # use 3 for best results
    print(f"Processing {n_patches} patches of size {patch_size}")
    # check if point coverage is 100%

    
    # Select seed points using farthest point sampling
    print("Selecting seed points...")
    if N > patch_size:
        seed_indices = farthest_point_sampling_cpu(points, n_patches)
    
        # Extract patches around seed points
        print("Extracting patches...")
        patches = extract_patches_knn_cpu(points, seed_indices, patch_size)
    else:
        patches = [points]
        print("Point cloud smaller than patch size, processing as single patch.")

    # Initialize output point cloud
    upsampled_pcd = o3d.geometry.PointCloud()
    all_upsampled_points = []
    
    # Process each patch
    print("Processing patches...")
    for i, patch in enumerate(tqdm(patches, desc="Processing patches", unit="patch")):
        # Upsample the patch
        upsampled_patch = upsample_patch(model, patch, args)
        all_upsampled_points.append(upsampled_patch)
    
    # Aggregate all upsampled points
    print("Aggregating results...")
    all_points = np.vstack(all_upsampled_points)
    # down sample to original size * r
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(all_points)
    all_points = o3d.geometry.uniform_down_sample(pc, int(len(all_points)/(N*args.r))).points
    
    
    # Remove duplicates if there's overlap (simple approach: remove points too close together)
    if args.remove_duplicates:
        print("Removing duplicate points...")
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(all_points)
        pcd_temp = pcd_temp.voxel_down_sample(voxel_size=args.duplicate_threshold)
        all_points = np.asarray(pcd_temp.points)
    
    # Create final point cloud
    upsampled_pcd.points = o3d.utility.Vector3dVector(all_points)
    
    # Save result
    print(f"Saving upsampled point cloud to: {output_path}")
    o3d.io.write_point_cloud(output_path, upsampled_pcd)
    
    print(f"Inference complete! Original: {N} points, Upsampled: {len(all_points)} points")
    print(f"Upsampling ratio: {len(all_points) / N:.2f}x")


def main():
    parser = argparse.ArgumentParser(description='Large Point Cloud Upsampling Inference')
    
    # Input/Output paths
    parser.add_argument('--inference_input_path', required=True, type=str, help='Input point cloud file') # add
    parser.add_argument('--inference_output_path', default='upsampled.ply', type=str, help='Output point cloud file') # add
    parser.add_argument('--ckpt',required=True, help='model to restore from')

    
    # Patch processing parameters
    parser.add_argument('--patch_size', default=10000, type=int,
                       help='Number of points per patch')
    parser.add_argument('--patch_rate', default=3, type=int, help='Overlap rate') # use 3 for best results

    parser.add_argument('--remove_duplicates', action='store_true',
                       help='Remove duplicate points in overlapping regions')
    parser.add_argument('--duplicate_threshold', default=0.001, type=float,
                       help='Distance threshold for duplicate removal')
    
    # Model parameters (should match training config)
    parser.add_argument('--num_samples', default=6, type=int,
                       help='Number of samples per Gaussian')
    parser.add_argument('--distribution', default='gaussian', type=str,
                       help='Distribution type: gaussian or uniform')
    parser.add_argument('--training_stage', default=2, type=int,
                       help='Training stage for inference')
    parser.add_argument('--r', default=4, type=int,
                       help='Upsampling ratio')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.inference_input_path):
        raise FileNotFoundError(f"Input file not found: {args.inference_input_path}")
    
    # Validate model file
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Model file not found: {args.ckpt}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.inference_output_path)), exist_ok=True)
    
    # Run inference
    infer_large_pointcloud(args.inference_input_path, args.inference_output_path, args.ckpt, args)


if __name__ == '__main__':
    main()
