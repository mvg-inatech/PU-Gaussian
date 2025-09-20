import sys 
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.PUCRN import Transformer, MLP_CONV, SubNetwork
from utils.mv_utils import PCViews
from utils.model_utils import FPS, chamfer_dist
from model.GaussianNetwork import Just_gaussian, Regressor, Transformer_extractor
from model.GaussianDistribution import GaussianDistribution

def _normalize_point_cloud(pc):
    """Normalize point cloud to unit sphere."""
    centroid = torch.mean(pc, dim=1, keepdim=True)
    pc = pc - centroid
    furthest_distance = torch.max(torch.sqrt(torch.sum(pc**2, dim=-1, keepdim=True)), dim=1, keepdim=True)[0]
    pc = pc / furthest_distance
    return pc

def chamfer_distance(p1, p2):
    """Compute Chamfer distance with square root normalization."""
    d1, d2, _, _ = chamfer_dist(_normalize_point_cloud(p1), _normalize_point_cloud(p2))
    d1 = torch.mean(d1)
    d2 = torch.mean(d2)
    return (d1 + d2)

class PU_Gaussian(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Gaussian = Just_gaussian(config=config)
        self.refinement = SubNetwork(up_ratio=1, features=True)
        self.training_stage = config.training_stage
        self.r = config.r
        print(f"training_stage: {self.training_stage}, upsampling rate: {self.r}")
    
    def forward(self, x, gt= None, x20 = None, view= None, return_features = True):
        # x is the point cloud, gt is the ground truth
        # x is [batch_size, 3, num_points]
        # gt is [batch_size, 3, num_points]
        # view is [batch_size, 3, num_points]
        # return_features is a boolean to indicate if we want to return the features
        _,_,N = x.shape
        
        coarse, features, gaussians = self.Gaussian(x, gt=x20, view=view, return_features = return_features)
        self.loss_scale = self.Gaussian.loss_scale

        if self.training and self.training_stage == 1:
            rgb = self.Gaussian.get_img(coarse)
            gt = x20
            return coarse, rgb, gt
        refine = self.refinement(coarse, features=features)
        rgb = self.Gaussian.get_img(refine)
        if self.training:
            self.loss = chamfer_distance(coarse.permute(0, 2, 1).contiguous(), gt.permute(0, 2, 1).contiguous()) * 1000
            return refine, rgb, gt

        return FPS(refine, N*self.r), rgb, gt
    
    def forward_test(self, x, gt= None, view= None):
        # x is the point cloud, gt is the ground truth
        # x is [batch_size, 3, num_points]
        # gt is [batch_size, 3, num_points]
        # view is [batch_size, 3, num_points]
        # return_features is a boolean to indicate if we want to return the features
        _,_,N = x.shape
        
        coarse, features, gaussians = self.Gaussian(x, gt=gt, view=view, return_features = True)
        refine = self.refinement(coarse, features=features)
        refine = self.refinement(refine, features=features)

        rgb = self.Gaussian.get_img(refine)
        self.loss_scale = self.Gaussian.loss_scale
   

        return FPS(refine,N*self.r), rgb, gt

    def freeze_gaussian(self):
        for param in self.Gaussian.parameters():
            param.requires_grad = False
    

# x = torch.randn(1, 3, 256).cuda()
# config = type('', (), {})()
# config.num_samples = 6
# config.training_stage = 1
# config.distribution = 'gaussian'

# model = PU_Gaussian(config).cuda()
# coarse, rgb, gt = model(x, gt = x)
# print(f" coarse shape {coarse.shape}")