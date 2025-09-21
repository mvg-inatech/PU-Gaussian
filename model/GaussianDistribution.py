import torch 
import torch.nn as nn
import sys 
from utils.model_utils import get_knn_pts, index_points

class GaussianDistribution(nn.Module):
    def __init__(self):
        super().__init__()


    def build_rotation(self, r):
        r = r.transpose(1,2)
        norm = torch.sqrt(r[:,:,0]*r[:,: ,0] + r[:,:,1]*r[:,:,1] + r[:,:,2]*r[:,:,2] + r[:,:,3]*r[:,:,3])

        q = r / norm[:,:, None]

        R = torch.zeros((q.size(0), q.shape[1],3, 3), device='cuda')

        r = q[:,:, 0]
        x = q[:,:, 1]
        y = q[:,:, 2]
        z = q[:,:, 3]

        R[:,:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, :, 0, 1] = 2 * (x*y - r*z)
        R[:,:, 0, 2] = 2 * (x*z + r*y)
        R[:,:, 1, 0] = 2 * (x*y + r*z)
        R[:,:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:,:, 1, 2] = 2 * (y*z - r*x)
        R[:,:, 2, 0] = 2 * (x*z - r*y)
        R[:,:, 2, 1] = 2 * (y*z + r*x)
        R[:,:, 2, 2] = 1 - 2 * (x*x + y*y)
        return R

    def build_scaling_rotation(self,s, r, inverse =False):
        L = torch.zeros((s.shape[0],s.shape[-1], 3, 3), dtype=torch.float, device="cuda")
        #L = torch.diag
        R = self.build_rotation(r)
        if inverse:
            s = 1 / (s + 1e-8)
        L[:,:,0,0] = s[:,0]
        L[:,:,1,1] = s[:,1]
        L[:,:,2,2] = s[:,2]
        L = R @ L
        return L


    def strip_lowerdiag(self, L):
        uncertainty = torch.zeros((L.shape[0], L.shape[1], 6), dtype=torch.float, device="cuda")

        uncertainty[:,:, 0] = L[:, :, 0, 0]
        uncertainty[:,:, 1] = L[:, :, 0, 1]
        uncertainty[:,:, 2] = L[:, :, 0, 2]
        uncertainty[:,:, 3] = L[:, :, 1, 1]
        uncertainty[:,:, 4] = L[:, :, 1, 2]
        uncertainty[:,:, 5] = L[:, :, 2, 2]
        return uncertainty

    def strip_symmetric(self, sym):
        return self.strip_lowerdiag(sym)

    def sample(self, num_samples, mean, scale, rot, epsilon = None,distribution = 'gaussian'):
        """
        Sample from the Gaussian distribution using the reparameterization trick.
        
        Args:
        num_samples (int): Number of samples to generate
        
        Returns:
        torch.Tensor: Samples with shape (batch_size, 3, num_samples * num_points)        
        """

        self.mean = mean
        self.scale = scale
        self.rot = rot
        L= self.build_scaling_rotation(self.scale, self.rot)
        actual_cov = L @ L.transpose(2, 3)

        # check if the covariance matrix is symmetric
        # assert torch.allclose(actual_cov, actual_cov.transpose(2, 3), atol=1e-5), "Covariance matrix is not symmetric"

        self.covariance = actual_cov
        batch_size, _, num_points = self.mean.shape
        # Generate standard normal samples)
        
        L = self.covariance.view(-1, 3,3).contiguous()
        if epsilon is None:
            if distribution == 'gaussian':
                epsilon = torch.randn(batch_size*num_points, num_samples, 3 ).to(self.mean.device)   #  mean = 0, std = 1.6 want to increase the std to 1.6
                epsilon = redraw_high_values(epsilon, threshold=2.0)

                
            elif distribution == 'uniform':
                epsilon = torch.rand(batch_size*num_points, num_samples, 3).to(self.mean.device) * 2 - 1
                epsilon = epsilon / torch.norm(epsilon, dim=-1, keepdim=True) * 0.5

        variance = epsilon @ L

        variance = variance.clamp(min=-1, max=1)
        

        variance = variance.view(batch_size,num_points, -1, 3).contiguous()
        self.mean = self.mean.permute(0, 2, 1).contiguous()
        samples = variance + self.mean.unsqueeze(2)
        samples = samples.view(batch_size, num_samples*num_points, 3).contiguous()
        samples = samples.permute(0, 2, 1).contiguous()
        
        return samples

    def forward(self, mean, scale, rot, num_samples=4, epsilon = None,distribution = 'gaussian'):


        return self.sample(num_samples=num_samples, mean=mean, scale=scale, rot=rot, epsilon=epsilon,distribution=distribution)
    
    def color_gaussian(self, mean, scale, rot, num_samples=6, epsilon = None):
        """
        Sample from the Gaussian distribution using the reparameterization trick.
        
        Args:
        num_samples (int): Number of samples to generate
        
        Returns:
        torch.Tensor: Samples with shape (batch_size, 3, num_samples * num_points)        
        """

        self.mean = mean
        self.scale = scale
        self.rot = rot
        L= self.build_scaling_rotation(self.scale, self.rot)
        actual_cov = L @ L.transpose(2, 3)

        # check if the covariance matrix is symmetric
        # assert torch.allclose(actual_cov, actual_cov.transpose(2, 3), atol=1e-5), "Covariance matrix is not symmetric"

        self.covariance = actual_cov
        batch_size, _, num_points = self.mean.shape
        # Generate standard normal samples)
        
        L = self.covariance.view(-1, 3,3).contiguous()
        if epsilon is None:
            epsilon = torch.rand(batch_size*num_points, num_samples, 3 ).to(self.mean.device)
        
        variance = epsilon @ L
        variance = variance.view(batch_size,num_points, -1, 3).contiguous()
        self.mean = self.mean.permute(0, 2, 1).contiguous()
        samples = variance + self.mean.unsqueeze(2)
        # color each sample from the same mean with the same color
        color = torch.rand(batch_size, num_samples, 3).to(self.mean.device)
        color = color.view(batch_size, num_samples, 3).contiguous()
        color = color.unsqueeze(2).repeat(1, 1, num_points, 1)
        color = color.view(batch_size, num_samples*num_points, 3).contiguous()
        samples = samples.view(batch_size, num_samples*num_points, 3).contiguous()
        color = color.permute(0, 2, 1).contiguous()
        samples = samples.permute(0, 2, 1).contiguous()
        print(f"color shape {color.shape}, samples shape {samples.shape}")
        return torch.cat([samples, color], dim=1)

    def kernel_evalutation(self, mean , scale ,rot , gt):
        """
        Evaluate the Gaussian distribution at the given points.
        
        Args:
        points (torch.Tensor): Points to evaluate the distribution at, shape (batch_size, 3, num_points)
        
        Returns:
        torch.Tensor: Evaluated distribution values, shape (batch_size, num_points)
        """
        # Compute the covariance matrix
        diff, scale, rot = map_gt2means(mean, gt, k=1, scale=scale, rotation=rot)
        
        L = self.build_scaling_rotation(scale, rot)
        actual_cov = L @ L.transpose(2, 3)

        # Compute the difference between the points and the mean
  
        diff = diff.permute(0, 2, 1).contiguous()
        # Compute the exponent
        exponent = diff.unsqueeze(2) @ actual_cov @ diff.unsqueeze(-1)

        # result = torch.logsumexp(exponent, dim=1)
        kernal_loss = torch.mean(exponent)

        return kernal_loss
    
    def loss_gaussian(self, mean , scale ,rot , gt):
        """
        Evaluate the Gaussian distribution at the given points.
        
        Args:
        points (torch.Tensor): Points to evaluate the distribution at, shape (batch_size, 3, num_points)
        
        Returns:
        torch.Tensor: Evaluated distribution values, shape (batch_size, num_points)
        """
        # Compute the covariance matrix
        diff, scale, rot = map_gt2means(mean, gt, k=1, scale=scale, rotation=rot)
        
        L = self.build_scaling_rotation(scale, rot, inverse=True)
        inverse_cov = L @ L.transpose(2, 3)
        L = self.build_scaling_rotation(scale, rot) 
        actual_cov = L @ L.transpose(2, 3)
        # Compute the difference between the points and the mean
  
        diff = diff.permute(0, 2, 1).contiguous()
        # Compute the exponent
        exponent = diff.unsqueeze(2) @ inverse_cov @ diff.unsqueeze(-1) 

        # result = torch.logsumexp(exponent, dim=1)
        kernal_loss = torch.mean(exponent)

        return kernal_loss
    
    def Gau_downsample(self, mean,scale, rot, upsampled, num_samples= 1024, epsilon = None):
        # to be done sample exactly 4 points from each mean.  ### NOT IMPLEMENTED YET

        diff, scale, rot = map_gt2means(mean, upsampled, k=1, scale=scale, rotation=rot)
        
        L = self.build_scaling_rotation(scale, rot)
        actual_cov = L @ L.transpose(2, 3)
        diff = diff.permute(0, 2, 1).contiguous()

        exponent = diff.unsqueeze(2) @ actual_cov @ diff.unsqueeze(-1)
        result, indices = torch.sort(exponent, dim=1, descending=True)
        # get the indices of the highest 4 points
        indices = indices[:, :num_samples, :,:].squeeze(-1)
        # indices = indices.permute(0, 2, 1).contiguous().long()

        result = index_points(upsampled, indices).squeeze(-1)
        # problems might be indices and how I index points near the mean. 
        # 
        return result


# get the nearest neighbors of each point in the point cloud and calculate the distance
def map_gt2means(mean,gt, k=1, idx=None, scale = None, rotation = None):
    # correct shapes
    selected_mean, idx = get_knn_pts(k= k, pts = mean, center_pts=gt, return_idx=True)
    # most repeated element ins
    diff = gt.unsqueeze(-1) - selected_mean
    diff = diff.contiguous()
    if k == 1:
        diff = diff.squeeze(-1)
        scale = index_points(scale, idx).squeeze(-1)
        rotation = index_points(rotation, idx).squeeze(-1)
    else: 
        scale = index_points(scale, idx).view(diff.shape[0], diff.shape[1], -1).contiguous()
        rotation = index_points(rotation, idx).view(diff.shape[0], 4, -1).contiguous()
        diff = diff.view(diff.shape[0], diff.shape[1], -1).contiguous()

    return diff, scale, rotation

def redraw_high_values(epsilon, threshold=1.0):
    """
    Redraw values in a tensor that are greater than the threshold.
    New values are sampled from a standard normal distribution.
    
    Args:
        epsilon: Input torch tensor with shape (batch_size*num_points, num_samples, 3)
        threshold: Values above this threshold will be redrawn (default: 1.0)
    
    Returns:
        Tensor with the same shape but with values > threshold redrawn
    """
    # Create masks for values that need to be redrawn (both > threshold and < -threshold)
    mask = torch.abs(epsilon) > threshold 
    
    # Count how many values need to be redrawn
    num_to_redraw = torch.sum(mask).item()
    
    # Only proceed if there are values to redraw
    if num_to_redraw > 0:
        # Sample new values from standard normal with the same device as input
        new_values = torch.randn(num_to_redraw, device=epsilon.device)
        # Replace the high values with new samples
        epsilon.masked_scatter_(mask, new_values)
        # print(f" I'm redrawing {num_to_redraw} values")
        # Recursively call the function to ensure all values are within the threshold
        epsilon = redraw_high_values(epsilon, threshold=threshold)
    
    return epsilon


def redraw_low_values(epsilon, threshold = 0.1):
    """
    Redraw values in a tensor that are less than the threshold.
    New values are sampled from a standard normal distribution.
    
    Args:
        epsilon: Input torch tensor with shape (batch_size*num_points, num_samples, 3)
        threshold: Values below this threshold will be redrawn (default: 0.1)
    
    Returns:
        Tensor with the same shape but with values < threshold redrawn
    """
    # Create masks for values that need to be redrawn (both > threshold and < -threshold)
    mask = torch.abs(epsilon) < threshold 
    
    # Count how many values need to be redrawn
    num_to_redraw = torch.sum(mask).item()
    
    # Only proceed if there are values to redraw
    if num_to_redraw > 0:
        # Sample new values from standard normal with the same device as input
        new_values = torch.randn(num_to_redraw, device=epsilon.device)
        # Replace the high values with new samples
        epsilon.masked_scatter_(mask, new_values)
        # print(f" I'm redrawing {num_to_redraw} values")
        # Recursively call the function to ensure all values are within the threshold
        epsilon = redraw_low_values(epsilon, threshold=threshold)
    
    return epsilon

