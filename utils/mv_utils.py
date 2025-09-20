import torch.nn as nn
import numpy as np
import torch
import open3d as o3d

RESOLUTION = 128
TRANS = -1.4

def euler2mat(angle):
    """Convert euler angles to rotation matrix."""
    if len(angle.size()) == 1:
        x, y, z = angle[0], angle[1], angle[2]
        _dim = 0
        _view = [3, 3]
    elif len(angle.size()) == 2:
        b, _ = angle.size()
        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]
        _dim = 1
        _view = [b, 3, 3]
    else:
        assert False

    cosz = torch.cos(z)
    sinz = torch.sin(z)
    zero = z.detach() * 0
    one = zero.detach() + 1
    zmat = torch.stack([cosz, -sinz, zero,
                        sinz, cosz, zero,
                        zero, zero, one], dim=_dim).reshape(_view)

    cosy = torch.cos(y)
    siny = torch.sin(y)
    ymat = torch.stack([cosy, zero, siny,
                        zero, one, zero,
                        -siny, zero, cosy], dim=_dim).reshape(_view)

    cosx = torch.cos(x)
    sinx = torch.sin(x)
    xmat = torch.stack([one, zero, zero,
                        zero, cosx, -sinx,
                        zero, sinx, cosx], dim=_dim).reshape(_view)

    rot_mat = xmat @ ymat @ zmat
    return rot_mat

def distribute(depth, _x, _y, size_x, size_y, image_height, image_width):
    """Distributes the depth associated with each point to the discrete coordinates."""
    assert size_x % 2 == 0 or size_x == 1
    assert size_y % 2 == 0 or size_y == 1
    batch, _ = depth.size()
    epsilon = torch.tensor([1e-12], requires_grad=False, device=depth.device)
    _i = torch.linspace(-size_x / 2, (size_x / 2) - 1, size_x, requires_grad=False, device=depth.device)
    _j = torch.linspace(-size_y / 2, (size_y / 2) - 1, size_y, requires_grad=False, device=depth.device)

    extended_x = _x.unsqueeze(2).repeat([1, 1, size_x]) + _i
    extended_y = _y.unsqueeze(2).repeat([1, 1, size_y]) + _j

    extended_x = extended_x.unsqueeze(3).repeat([1, 1, 1, size_y])
    extended_y = extended_y.unsqueeze(2).repeat([1, 1, size_x, 1])

    extended_x.ceil_()
    extended_y.ceil_()

    value = depth.unsqueeze(2).unsqueeze(3).repeat([1, 1, size_x, size_y])

    masked_points = ((extended_x >= 0)
                     * (extended_x <= image_height - 1)
                     * (extended_y >= 0)
                     * (extended_y <= image_width - 1)
                     * (value >= 0))

    true_extended_x = extended_x
    true_extended_y = extended_y

    extended_x = (extended_x % image_height)
    extended_y = (extended_y % image_width)

    distance = torch.abs((extended_x - _x.unsqueeze(2).unsqueeze(3))
                         * (extended_y - _y.unsqueeze(2).unsqueeze(3)))
    weight = (masked_points.float()
          * (1 / (value + epsilon)))
    weighted_value = value * weight

    weight = weight.view([batch, -1])
    weighted_value = weighted_value.view([batch, -1])

    coordinates = (extended_x.view([batch, -1]) * image_width) + extended_y.view([batch, -1])
    coord_max = image_height * image_width
    true_coordinates = (true_extended_x.view([batch, -1]) * image_width) + true_extended_y.view([batch, -1])
    true_coordinates[~masked_points.view([batch, -1])] = coord_max
    weight_scattered = torch.zeros([batch, image_width * image_height], device=depth.device).scatter_add(1, coordinates.long(), weight)

    masked_zero_weight_scattered = (weight_scattered == 0.0)
    weight_scattered += masked_zero_weight_scattered.float()

    weighed_value_scattered = torch.zeros([batch, image_width * image_height], device=depth.device).scatter_add(1, coordinates.long(), weighted_value)

    return weighed_value_scattered, weight_scattered

def points2depth(points, image_height, image_width, size_x=4, size_y=4):
    """Convert points to depth image."""
    epsilon = torch.tensor([1e-12], requires_grad=False, device=points.device)
    coord_x = (points[:, :, 0] / (points[:, :, 2] + epsilon)) * (image_width / image_height)
    coord_y = (points[:, :, 1] / (points[:, :, 2] + epsilon))

    batch, total_points, _ = points.size()
    depth = points[:, :, 2]
    _x = ((coord_x + 1) * image_height) / 2
    _y = ((coord_y + 1) * image_width) / 2

    weighed_value_scattered, weight_scattered = distribute(
        depth=depth,
        _x=_x,
        _y=_y,
        size_x=size_x,
        size_y=size_y,
        image_height=image_height,
        image_width=image_width)

    depth_recovered = (weighed_value_scattered / weight_scattered).view([batch, image_height, image_width])
    return depth_recovered

def batched_index_select(inp, dim, index):
    """Batched index select."""
    views = [inp.shape[0]] + [1 if i != dim else -1 for i in range(1, len(inp.shape))]
    expanse = list(inp.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(inp, dim, index)

def point_fea_img_fea(point_fea, point_coo, h, w):
    """Convert point features to image features."""
    assert len(point_fea.shape) == 3
    assert len(point_coo.shape) == 2
    assert point_fea.shape[0:2] == point_coo.shape

    coo_max = ((h - 1) * w) + (w - 1)
    mask_point_coo = (point_coo >= 0) * (point_coo <= coo_max)
    point_coo *= mask_point_coo.float()
    point_fea *= mask_point_coo.float().unsqueeze(-1)

    bs, _, fs = point_fea.shape
    point_coo = point_coo.unsqueeze(2).repeat([1, 1, fs])
    img_fea = torch.zeros([bs, h * w, fs], device=point_fea.device).scatter_add(1, point_coo.long(), point_fea)

    return img_fea

def distribute_img_fea_points(img_fea, point_coord):
    """Distribute image features to points."""
    B, C, H, W = list(img_fea.size())
    img_fea = img_fea.permute(0, 2, 3, 1).view([B, H*W, C])

    coord_max = ((H - 1) * W) + (W - 1)
    mask_point_coord = (point_coord >= 0) * (point_coord <= coord_max)
    mask_point_coord = mask_point_coord.float()
    point_coord = mask_point_coord * point_coord
    point_fea = batched_index_select(inp=img_fea, dim=1, index=point_coord.long())
    point_fea = mask_point_coord.unsqueeze(-1) * point_fea
    return point_fea

class PCViews:
    """For creating images from PC based on the view information."""

    def __init__(self):
        _views = np.asarray([
            [[0 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
            [[1 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
            [[2 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
            [[3 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
            [[0, -np.pi / 2, np.pi / 2], [0, 0, TRANS]],
            [[0, np.pi / 2, np.pi / 2], [0, 0, TRANS]]])
        self.num_views = 6
        angle = torch.tensor(_views[:, 0, :]).float().cuda()
        self.rot_mat = euler2mat(angle).transpose(1, 2)
        self.translation = torch.tensor(_views[:, 1, :]).float().cuda()
        self.translation = self.translation.unsqueeze(1)

    def get_img(self, points):
        """Get image based on the prespecified specifications."""
        b, _, _ = points.shape
        v = self.translation.shape[0]

        _points = self.point_transform(
            points=torch.repeat_interleave(points, v, dim=0),
            rot_mat=self.rot_mat.repeat(b, 1, 1),
            translation=self.translation.repeat(b, 1, 1))

        img = points2depth(
            points=_points,
            image_height=RESOLUTION,
            image_width=RESOLUTION,
            size_x=1,
            size_y=1,
        )
        return img

    @staticmethod
    def point_transform(points, rot_mat, translation):
        """Transform points using rotation matrix and translation."""
        rot_mat = rot_mat.to(points.device)
        translation = translation.to(points.device)
        points = torch.matmul(points, rot_mat)
        points = points - translation
        return points

def test():
    pc_views = PCViews()
     
    path = "/workspaces/khater-pointcloud-upsampling/epoch_99_count6_pu_eva_gt.pcd"
    pc2 = load_point_cloud(path)
    if pc2.shape[1] == 3:
        pc2 = pc2.transpose(1, 2).contiguous().to('cuda')

    points = pc2
    img = pc_views.get_img(points)
    # save the images
    import matplotlib.pyplot as plt
    for i in range(6):
        plt.imshow(img[i].cpu().detach().numpy())
        plt.savefig(f'img_{i}.png')

def load_point_cloud(path):
    pcd = o3d.io.read_point_cloud(path)
    points = torch.tensor(pcd.points).float().unsqueeze(0).to('cuda')
    return points
def test_backpropagation():
    """Test the backpropagation capabilities of the PCViews class and related functions."""
    import torch
    import numpy as np
    
    # Create a small batch of points that requires gradients
    batch_size = 2
    num_points = 100
    points = torch.randn(batch_size, num_points, 3, requires_grad=True, device='cuda')
    
    # Initialize PCViews
    pc_views = PCViews()
    
    # Define a loss function that depends on the output of get_img
    def compute_loss(images):
        # Simple loss: mean of all pixel values
        return images.mean()
    
    # Forward pass
    with torch.autograd.set_detect_anomaly(True):  # Enable anomaly detection
        images = pc_views.get_img(points)
        loss = compute_loss(images)
        
        # Check if loss is valid and not NaN
        if torch.isnan(loss):
            print("Error: Loss is NaN")
            return False
        
        # Backward pass
        loss.backward()
        
        # Check if gradients are computed and not NaN
        if points.grad is None:
            print("Error: No gradients were computed for points")
            return False
        
        if torch.isnan(points.grad).any():
            print("Error: NaN gradients detected")
            return False
        
        grad_magnitude = points.grad.abs().mean().item()
        print(f"Gradient magnitude: {grad_magnitude}")
        
        # Test if gradients are meaningful (non-zero)
        if grad_magnitude < 1e-10:
            print("Warning: Gradient magnitude is very small")
        
    # Test specific components
    test_component_backprop()
    
    return True

def test_component_backprop():
    """Test backpropagation through individual components."""
    import torch
    
    # Test euler2mat
    angle = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=True, device='cuda')
    rot_mat = euler2mat(angle)
    rot_loss = rot_mat.sum()
    rot_loss.backward()
    print(f"euler2mat gradient magnitude: {angle.grad.abs().mean().item()}")
    
    # Test points2depth
    points = torch.randn(2, 100, 3, requires_grad=True, device='cuda')
    depth = points2depth(points, RESOLUTION, RESOLUTION)
    depth_loss = depth.sum()
    depth_loss.backward()
    print(f"points2depth gradient magnitude: {points.grad.abs().mean().item()}")
    
    # Test distribute
    depth = torch.randn(2, 100, requires_grad=True, device='cuda')
    _x = torch.randint(0, RESOLUTION, (2, 100), dtype=torch.float, device='cuda')
    _y = torch.randint(0, RESOLUTION, (2, 100), dtype=torch.float, device='cuda')
    _x.requires_grad = True
    _y.requires_grad = True
    
    weighted_value, weight = distribute(
        depth=depth,
        _x=_x,
        _y=_y,
        size_x=2,
        size_y=2,
        image_height=RESOLUTION,
        image_width=RESOLUTION
    )
    
    dist_loss = (weighted_value / weight).sum()
    dist_loss.backward()
    print(f"distribute gradient magnitude (depth): {depth.grad.abs().mean().item()}")

if __name__ == "__main__":
    test_backpropagation()

