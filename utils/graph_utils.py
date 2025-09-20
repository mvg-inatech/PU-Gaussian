import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os 
os.sys.path.append('/workspaces/khater-pointcloud-upsampling/vn-pueva/network')

from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation , grouping_operation



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def square_distance(src, dst, flipped = True):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, C, N]
        dst: target points, [B, C, M]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    if flipped == False:
        src = src.permute(0, 2, 1)
        dst = dst.permute(0, 2, 1)
    B, N,_ = src.shape
    _, M, _ = dst.shape
    

    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
    dist += torch.sum(src ** 2, -1).view(B, N, 1) # targeting channels which is 2nd dimension.
    dist += torch.sum(dst ** 2, -1).view(B, 1, M) # 
    return dist


def query_knn_point(k, xyz, new_xyz, flipped = True, ignore_self = False):
    dist = square_distance(new_xyz, xyz, flipped = flipped)
    
    
    if ignore_self:
        _, group_idx = dist.topk(k + 1, largest=False)
        group_idx = group_idx[:,:,1:]
    else:
        _, group_idx = dist.topk(k , largest=False)
    return group_idx.int()

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx



def get_graph_feature(x,y, k=20, idx=None, return_idx = False):
    batch_size = x.size(0)
    num_points = x.size(2)
    if idx is None:
        idx = query_knn_point(k, x, y, flipped=False)   # (batch_size, num_points, k)
        idx_return = idx
        #idx = torch.rand(batch_size,num_points,k ) 
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()


    x = x.transpose(2, 1).contiguous()   # (batch_size, num_dims, num_points)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points) num_dim = 24
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()  
    if return_idx:
        return feature, idx_return
    
    return feature



def get_neighbors_feature(xyz, feature_space , k=20 ,r =4, idx=None, vnn = False):
    """
    function gets the features of the k nearest neighbors and the r nearest neighbors, the point cloud of the k nearest neighbors and the index of the k nearest neighbors, and the index of the r nearest neighbors
    xyz : point cloud of size b x 3 x n
    feature_space : feature space of size b x c x n
    k : number of nearest neighbors
    r : number of r nearest neighbors
    idx : index of the k nearest neighbors
    feature_space relies on the feature extraction network.
    in normal case with growth_channel, input_channel[0] of size 24 and input_channel[1] of size 48, n_dense =3, n_blocks = 3
    the dimension of the feature space is b x 480 x n
    """

    batch_size = xyz.size(0)
    num_points = xyz.size(2)

    if vnn:
        batch_size = xyz.size(0)
        num_points = xyz.size(3)

    xyz = xyz.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points) num_dim = 24
    feature_space = feature_space.view(batch_size, -1, num_points)
    if idx is None:
        if vnn: 
            xyz = xyz.squeeze(1)
            xyz = xyz.permute(0,2,1)
        idx = query_knn_point(k, xyz, xyz)   # (batch_size, num_points, k)
        # pick r random neighbors from the k nearest neighbors
        # r chould be in dimesion of b,n,r


    idx, idx_r = process_idx_r_and_k(idx, k, r)
    #print(f" xyz : {xyz.shape}, feature_space : {feature_space.shape}, idx : {idx.shape}, idx_r : {idx_r.shape}, k : {k}, r : {r}")
    #print(idx_r)

    if vnn:
        feature_k, xyz_k, idx_k = get_features_eva_vnn(xyz.permute(0,2,1), feature_space, idx, k)
        feature_r, _ , idx_r = get_features_eva_vnn(xyz.permute(0,2,1), feature_space, idx_r, r)
    else:
        feature_k, xyz_k, idx_k = get_features_eva(xyz, feature_space, idx, k)
        feature_r, _ , idx_r = get_features_eva(xyz, feature_space, idx_r, r)


    return feature_k, feature_r, xyz_k, idx_k, idx_r


def process_idx_r_and_k(idx, k, r, device=None):
    """ returns 2 1d index tensors of size b x n x r and b x n x k
    the idx_r tensor is randomly selected from the idx tensor
    idx tensor is only reshaped to 1d tensor
    """
    idx = idx.to(device)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, num_points = idx.size(0), idx.size(1)
   
    idx_r = torch.rand( (batch_size, num_points, k), device=device)
    _, idx_r = idx_r.topk(r, dim=-1) # get the top r indices to choose from 
    idx_r = idx.gather(-1, idx_r) # actual indices chosen from idx

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_base = idx_base.to(idx.device) 

    idx = idx + idx_base  # add the base index to the idx tensor 
    idx_r = idx_r + idx_base

    idx = idx.view(-1)
    idx_r = idx_r.view(-1)

    return idx, idx_r


def get_features_eva(xyz, feature_space, idx, k):
    """
    this function returns the features of the k nearest neighbors, the point cloud of the k nearest neighbors and the index of the k nearest neighbors as descriped in the EVA paper
    the concatation is done in the following order,
    original feature_space_ repeated k times , feature_space_k, featurespace_k - original_feature_space, distance between the features, point cloud of the k nearest neighbors - point cloud
    xyz : point cloud of size b x 3 x n
    feature_space : feature space of size b x c x n
    idx : index is a 1d tensor of size b x n x k
    k : number of nearest neighbors , or the number of r nearest neighbors
    """
    batch_size = xyz.size(0)
    num_points = xyz.size(1)
    _, _, num_dims = xyz.size()
    _, num_dims_f, _ = feature_space.size()

    #xyz = xyz.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points) num_dim = 24
    xyz_k = xyz.view(batch_size*num_points, -1)[idx, :]

    feature_space = feature_space.transpose(2,1).contiguous()

    feature = feature_space.view(batch_size*num_points, -1)[idx, :] #get the value of the neighbors.

    feature = feature.view(batch_size, num_points, k, num_dims_f)
    xyz_k = xyz_k.view(batch_size, num_points, k, num_dims)

    xyz = xyz.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature_space = feature_space.view(batch_size, num_points, 1, num_dims_f).repeat(1, 1, k, 1)

    d_feature = torch.unsqueeze(torch.sum(torch.pow(feature - feature_space, 2), dim = -1), -1)
    feature = torch.cat((feature_space, feature, feature - feature_space, d_feature, xyz_k - xyz), dim=-1).permute(0, 3, 1, 2).contiguous()

    xyz_k = xyz_k.permute(0,3,1,2).contiguous() # from bnkc to bcnk
    return feature , xyz_k, idx



def get_features_eva_vnn(xyz, feature_space, idx, k):
    """
    this function returns the features of the k nearest neighbors, the point cloud of the k nearest neighbors and the index of the k nearest neighbors as descriped in the EVA paper
    the concatation is done in the following order,
    original feature_space_ repeated k times , feature_space_k, featurespace_k - original_feature_space, distance between the features, point cloud of the k nearest neighbors - point cloud
    xyz : point cloud of size b x 3 x n
    feature_space : feature space of size b x c x n
    idx : index is a 1d tensor of size b x n x k
    k : number of nearest neighbors , or the number of r nearest neighbors
    """
    batch_size = xyz.size(0)
    num_points = xyz.size(2)
    _, num_dims, _ = xyz.size()
    num_dims = num_dims // 3
    _, num_dims_f, _ = feature_space.size()
    num_dims_f = num_dims_f // 3


    xyz = xyz.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points) num_dim = 24
    feature_space = feature_space.transpose(2,1).contiguous()

    xyz_k = xyz.view(batch_size*num_points, -1)[idx, :]
    feature = feature_space.view(batch_size*num_points, -1)[idx, :]

    feature = feature.view(batch_size, num_points, k, num_dims_f, 3)
    xyz_k = xyz_k.view(batch_size, num_points, k, num_dims, 3)
 

    
    xyz = xyz.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1,1)
    feature_space = feature_space.view(batch_size, num_points, 1, num_dims_f,3).repeat(1, 1, k, 1,1)
    d_feature = torch.unsqueeze(torch.sum(torch.pow(feature - feature_space, 2), dim = -2), -2)
    #d_feature = d_feature.repeat(1,1,1,2,1)
    # tile d_feature to two at the channel dimension for the whole thing to be devisable by 3
    # ^ the above comment is not necessary as the feature space which ( 3c + 3 + 1) is gonna be ( 3c//3 + 3//3 + 1//3) = c + 1 + 1  so the number of n_channel doesn't have to be devisable by 3. it's always going to be c+2
    # the number of n_channel is always going to be c + 2 

    #d_feature = d_feature.view(batch_size, num_points, k, 1, 3).repeat(1,1,1,num_dims,1)
    feature = torch.cat((feature_space, feature, feature - feature_space, d_feature, xyz_k - xyz), dim=-2).permute(0, 3,4, 1, 2).contiguous()
    
    xyz_k = xyz_k.permute(0,3,4,1,2).contiguous()

    return feature , xyz_k, idx


def midpoint_interpolate2(sparse_pts, k):
    """
    this function gets the sparse points and the index of the k nearest neighbors and returns the interpolated points
    k : number of nearest neighbors
    sparse_pts : sparse points of size b x 3 x n
    idx : index of the k nearest neighbors
    """
    batch_size = sparse_pts.size(0)
    num_points = sparse_pts.size(2)
    _, num_dims, _ = sparse_pts.size()
    up_points = num_points * k
    k = 2 * k 

    device = torch.device('cuda')
    #sparse_pts = sparse_pts.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points) num_dim = 24
    sparse_pts = sparse_pts.transpose(2,1).contiguous() # b x n x 3
    idx = query_knn_point(k, sparse_pts, sparse_pts) # (batch_size, num_points, k)


    #move the idx to the device
    idx = idx.to(device)
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)


   
    knn_pts = sparse_pts.view(batch_size*num_points, -1)[idx, :]


    knn_pts = knn_pts.view(batch_size, num_points, k, num_dims)
    knn_pts = knn_pts.permute(0,3,1,2).contiguous()



    sparse_pts = sparse_pts.view(batch_size, num_points , 1, num_dims).repeat(1, 1, k, 1)
    sparse_pts = sparse_pts.permute(0,3,1,2).contiguous()

    midpoints = (sparse_pts + knn_pts) / 2 # b ,c,n,k 
    #midpoints = midpoints.permute(0,3,1,2).contiguous()
    midpoints = midpoints.view(batch_size, num_dims, num_points*k).contiguous()
    midpoints = midpoints.permute(0,2,1).contiguous()   # b x n*k x c 
    #print(f"midpoints : {midpoints}")
    midpoints_idx = furthest_point_sample(midpoints,up_points) # b x up_points >>> b x n*k/2
    midpoints = midpoints.permute(0,2,1).contiguous() # b x c x n*k 
    midpoints = gather_operation(midpoints, midpoints_idx) # from b x c x n*k to b x c x up_points
    return midpoints



""" sparse_pts = torch.tensor([[[0,2,4,0],[1,2,3,0],[0,0,0,0]],[[1,2,3,0],[1,2,3,0],[1,2,3,4]]], dtype=torch.float32).to(device)

idx = knn(sparse_pts,2)
idx = idx.to(device)
idx_base = torch.arange(0, 2, device=device).view(-1, 1, 1)*4
idx = idx + idx_base
idx = idx.view(-1)

features, x, idx = get_features_eva(sparse_pts, sparse_pts,idx, 2)
x = x.view(2,3, -1)
""" 
""" 
print(f"features : {features.shape}")
print(f"x : {x}")
print(f"x : {x.shape}")
print(f"sparse_pts : {sparse_pts}") 
print(f"idx : {idx}") """ 