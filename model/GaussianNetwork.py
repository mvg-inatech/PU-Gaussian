import torch
import torch.nn as nn
import torch.nn.functional as F
from model.PUCRN import Transformer, MLP_CONV
from model.GaussianDistribution import GaussianDistribution
from utils.mv_utils import PCViews
from utils.model_utils import FPS




class Just_gaussian(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extractor = Regressor(64, 32, 3).to('cuda')
        self.gaussian = GaussianDistribution()
        self.Pc_views = PCViews()
        self._get_img = self.Pc_views.get_img
        self.num_samples = config.num_samples
        self.training_stage = config.training_stage
        self.distribution = config.distribution
        self.r = config.r
    
    def forward(self, x, gt = None, view = None, return_gaussians = False, return_features = False):
        mean, scale, rot, confidence, features = self.feature_extractor(x, return_features=True)
        mean = mean + x
        coarse = self.gaussian(mean, scale, rot, num_samples=self.num_samples, distribution = self.distribution)
        
        rgb = self.get_img(coarse)
        self.loss_scale = 0
        if gt is not None and self.training:
            self.loss_scale = self.gaussian.kernel_evalutation(mean, scale, rot, gt)

        # self.loss_scale = scale_reg_loss(scale)
        if return_gaussians:
            return coarse,rgb,gt, [mean, scale, rot]
        if return_features:
            return coarse, features, [mean, scale, rot]
        if self.training:
            return coarse, rgb, gt
        else:
            coarse = FPS(coarse, x.shape[-1]*self.r)
            return coarse,rgb,gt
    def get_img(self, pc):
        pc = pc.permute(0, 2, 1).contiguous()
        img = self._get_img(pc)
        return img


class Transformer_extractor(nn.Module):
    """
    Point-wise feature extractor.

    Input:
        points: input points, (B, 3, N_input)
    Output:
        point_feat: ouput feature, (B, dim_feat, N_input)
    """
    def __init__(self, dim_feat, hidden_dim, in_channel=3, feature_dim = 10):
        super(Transformer_extractor, self).__init__()
        self.mlp_1 = MLP_CONV(in_channel=in_channel, layer_dims=[64, dim_feat])
        self.mlp_proj = MLP_CONV(in_channel=feature_dim, layer_dims=[dim_feat, dim_feat])
        self.mlp_2 = MLP_CONV(in_channel=dim_feat * 2, layer_dims=[dim_feat*2, dim_feat])
        self.point_transformer = Transformer(dim_feat, dim=hidden_dim)

    def forward(self, points, features = None):
        if features is None:
            feature_1 = self.mlp_1(points)
        else:
            feature_1 = self.mlp_proj(features)
        global_feature = torch.max(feature_1, 2, keepdim=True)[0]
        feature_2 = torch.cat([feature_1, global_feature.repeat((1, 1, feature_1.size(2)))], 1)
        feature_3 = self.mlp_2(feature_2)
        point_feat = self.point_transformer(feature_3, points)
        return point_feat

class Regressor(nn.Module):
    def __init__(self, dim_feat, hidden_dim, in_channel=3):
        super().__init__()
        self.extractor = Transformer_extractor(dim_feat,hidden_dim,in_channel).to('cuda')
        zdim = dim_feat
        self.scale_head = nn.Sequential(
            nn.Conv1d(zdim, zdim // 2, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(zdim // 2, 3, 1, 1),
            nn.Softplus(beta=100))
        self.rot_head = nn.Sequential(
            nn.Conv1d(zdim, zdim // 2, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(zdim // 2, 4, 1, 1))

        self.mean_head = nn.Sequential(
            nn.Conv1d(zdim, zdim // 2, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(zdim // 2, 3, 1, 1))
        
        self.confidence_head = nn.Sequential(
            nn.Conv1d(zdim, zdim // 2, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(zdim // 2, 1, 1, 1),
            nn.Sigmoid())
    
    def forward(self, x, features = None, return_features = False):
        x = self.extractor(x, features)
        scale = self.scale_head(x)
        rot = self.rot_head(x)
        mean = self.mean_head(x)
        confidence = self.confidence_head(x)
        if return_features:
            return mean, scale, rot, confidence, x
        
        return mean, scale, rot, confidence


def scale_reg_loss( scale):
    scale = torch.clamp(scale, min=1e-8)
    scale = torch.log(scale)
    scale = torch.clamp(scale, min=-5, max=5)
    scale = torch.mean(scale**2)
    # return scale
    return scale

