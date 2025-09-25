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



def partition_point_cloud(points, patch_size):
    """
    Partition a 3D point cloud into axis-aligned grid cells.

    Args:
        points (np.ndarray): Input point cloud of shape (N, 3).
        patch_size (int): Target number of points per patch.

    Returns:
        list[np.ndarray]: List of patches, each containing points as a NumPy array.
    """
    from sklearn.cluster import KMeans

    # Step 1: Compute global bounding box and density
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    aabb_volume = np.prod(max_xyz - min_xyz)
    density = len(points) / aabb_volume

    # Step 2: Calculate cell size based on target patch size
    cell_volume = patch_size / density
    cell_size = cell_volume ** (1 / 3)

    # Step 3: Assign points to grid cells
    grid_indices = np.floor((points - min_xyz) / cell_size).astype(int)
    grid = {}
    for idx, grid_idx in enumerate(map(tuple, grid_indices)):
        if grid_idx not in grid:
            grid[grid_idx] = []
        grid[grid_idx].append(idx)
    
    for cell_indices in list(grid.values()):
         if len(cell_indices) < patch_size // 1.5:
            # Merge small cells with the nearest neighbor
            cell_center = np.mean(points[cell_indices], axis=0)
            nearest_cell = None
            min_distance = float('inf')
            
            # Track parsed cells
            parsed_cells = set()
            
            for key, indices in grid.items():
                if key in parsed_cells or key == tuple(grid_indices[cell_indices[0]]):
                    continue
                other_cell_center = np.mean(points[indices], axis=0)
                distance = np.linalg.norm(cell_center - other_cell_center)
                if distance < min_distance:
                    min_distance = distance
                    nearest_cell = key
            
            if nearest_cell is not None:
                grid[nearest_cell].extend(cell_indices)
                parsed_cells.add(nearest_cell)  # Mark the nearest cell as parsed
                del grid[tuple(grid_indices[cell_indices[0]])]  # Remove the small cell
    

    # Step 4: Process cells to ensure patch consistency
    patches = []
    for cell_indices in list(grid.values()):
        if len(cell_indices) > 1.5 * patch_size:
            # Split large cells using KMeans
            num_subclusters = max(1, len(cell_indices) // patch_size + 1)
            kmeans = KMeans(n_clusters=num_subclusters, random_state=0).fit(points[cell_indices])
            sub_labels = kmeans.labels_
            for sub_id in range(num_subclusters):
                sub_cluster_indices = np.array(cell_indices)[sub_labels == sub_id]
                patches.append(points[sub_cluster_indices])
        else:
            # Add medium-sized cells directly as patches
            patches.append(points[cell_indices])

    # Step 5: Ensure all points are included exactly once
    total_points = sum(len(patch) for patch in patches)
    print(f"Total points in patches: {total_points}, Original points: {len(points)}")
    assert total_points == len(points), "Mismatch in total points after partitioning. len"

    return patches

def upsampling(args, model, input_pcd):
    """Upsample point cloud using the model."""
    pcd_pts_num = input_pcd.shape[-1]
    patch_pts_num = 256
    sample_num = int(pcd_pts_num / patch_pts_num * args.patch_rate)
    seed = FPS(input_pcd, sample_num)
    patches = extract_knn_patch(patch_pts_num, input_pcd, seed)
    patches, centroid, furthest_distance = normalize_point_cloud(patches)
    
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

    if N > patch_size:
        print("partitioning point cloud into patches...")
        patches = partition_point_cloud(points, patch_size)
        print(f" patch size {len(patches)}")
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
    
    
    # Remove duplicates if there's overlap (simple approach: remove points too close together)
    if args.remove_duplicates:
        print("Removing duplicate points...")
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(all_points)
        pcd_temp = pcd_temp.voxel_down_sample(voxel_size=args.duplicate_threshold)
        all_points = np.asarray(pcd_temp.points)
    
    # Create final point cloud
    upsampled_pcd.points = o3d.utility.Vector3dVector(all_points)
    # take the color from the original point cloud nearest neighbor
    if len(pcd.colors) > 0 and args.return_color:
        print("Transferring colors from original point cloud...")
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(np.asarray(pcd.points))
        distances, indices = nbrs.kneighbors(all_points)
        colors = np.asarray(pcd.colors)[indices.flatten()]
        upsampled_pcd.colors = o3d.utility.Vector3dVector(colors)
        # if you want to add semantic labels as well, you can do similar process here for labels 
        # labels = np.asarray(pcd.labels)[indices.flatten()] # assuming pcd has labels attribute.
        # colors added 
        print("Colors transferred.")
    
    if args.add_original:
        upsampled_pcd += pcd
        upsampled_pcd = upsampled_pcd.voxel_down_sample(voxel_size= args.duplicate_threshold / 2)
        print("Original points added to upsampled point cloud.")
    
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
    parser.add_argument('--return_color', action='store_true',
                       help='Return colors from original point cloud if available')
    
    # Model parameters (should match training config)
    parser.add_argument('--num_samples', default=6, type=int,
                       help='Number of samples per Gaussian')
    parser.add_argument('--distribution', default='gaussian', type=str,
                       help='Distribution type: gaussian or uniform')
    parser.add_argument('--training_stage', default=2, type=int,
                       help='Training stage for inference')
    parser.add_argument('--r', default=4, type=int,
                       help='Upsampling ratio')
    parser.add_argument('--add_original', action='store_true',
                       help='Add original points to upsampled output')

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
