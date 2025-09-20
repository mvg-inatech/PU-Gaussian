import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.model_utils import chamfer_dist
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import os
from datetime import datetime
sys.path.append('.')
from data import PUDataset
from configs import args as confg
from utils.mv_utils import PCViews
from model.models import PU_Gaussian, Just_gaussian 
# Initialize PC Views
pc_views = PCViews()

def chamfer_distance(p1, p2):
    """Compute Chamfer distance with square root normalization."""
    d1, d2, _, _ = chamfer_dist(_normalize_point_cloud(p1), _normalize_point_cloud(p2))
    d1 = torch.mean(d1)
    d2 = torch.mean(d2)
    return (d1 + d2)

def _normalize_point_cloud(pc):
    """Normalize point cloud to unit sphere."""
    centroid = torch.mean(pc, dim=1, keepdim=True)
    pc = pc - centroid
    furthest_distance = torch.max(torch.sqrt(torch.sum(pc**2, dim=-1, keepdim=True)), dim=1, keepdim=True)[0]
    pc = pc / furthest_distance
    return pc


def get_timestamp():
    """Get current timestamp in format suitable for filenames."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



class PointCloudModel:
    """Model class for point cloud upsampling."""
    
    def __init__(self, network, phase='train', config=None):
        """
        Initialize the model with network and training parameters.
        
        Args:
            network: Neural network model
            phase: 'train' or 'test'
            config: Configuration object with hyperparameters
        """
        self.net = network
        self.phase = phase
        self.loss_history = []
        self.step = 0

        if config is None:
            raise ValueError("Config must be provided")

        if self.phase == 'train':
            self.error_log = defaultdict(int)
            self.lr = config.lr_init
            self.optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=config.lr_init,
                betas=(0.9, 0.999)
            )
            self.lr_scheduler = ExponentialLR(self.optimizer, gamma=0.7)
            self.decay_step = config.decay_iter
            
            # Initialize loss functions
            self.chamfer_loss_fn = chamfer_distance
            self.mse_loss = nn.MSELoss()
            self.training_stage = config.training_stage

    def set_input(self, input_pc, radius=None, x6=None, x20=None):
        """
        Set input and ground truth point clouds.
        
        Args:
            input_pc: Input point cloud tensor [B,3,N]
            radius: Radius for repulsion loss
            label_pc: Ground truth point cloud tensor [B,3,N']
        """
        self.input = input_pc.detach().transpose(1, 2).float()
        self.radius = radius
        
        if x20 is not None:
            self.x20 = x20.detach().transpose(1, 2).float().cuda()
        if x6 is not None: # wrong way to put if statement but it works for now
            self.x6 = x6.detach().transpose(1, 2).float().cuda()
        if self.training_stage == 2 and x6 is not None:
            self.gt = self.x6
        elif self.training_stage == 1 and x20 is not None:
            self.gt = self.x20
        else:
            self.gt = None
        
    def forward(self):
        """Forward pass through the network."""
        # Use fixed views for consistency
        self.view = [0, 6, 12, 18]

        self.predicted, self.img, self.gt = self.net(self.input, gt=self.gt,x20 =self.x20, view=self.view)

    def get_lr(self):
        """Get current learning rate."""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
    
    def get_gradient_norm(self):
        """Calculate gradient norm for all parameters."""
        total_norm = 0
        for name, param in self.net.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                if param_norm < 1e-5 and 'bias' not in name:
                    print(f'Layer: {name}, Gradient Norm: {param_norm:.4f}, requires_grad: {param.requires_grad}')
                total_norm += param_norm ** 2
        
        return total_norm ** 0.5
    
    @torch.no_grad()
    def validate(self):
        """Validate model on validation set."""
        self.net.eval()
        torch.cuda.empty_cache()
        self.forward()
        
        # Ensure correct tensor shapes
        if self.predicted.shape[1] == 3:
            self.predicted = self.predicted.permute(0, 2, 1).contiguous()
        if self.gt.shape[1] == 3:
            self.gt = self.gt.permute(0, 2, 1).contiguous()

        # Compute validation loss
        val_loss = self.chamfer_loss_fn(self.predicted, self.gt) * 1000
        
        
        return val_loss

    def optimize(self, epoch=None):
        """
        Run optimization step.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary containing loss values and other metrics
        """
        self.optimizer.zero_grad()
        self.net.train()
        self.forward()
        
        # Reset loss history every 100 steps
        if self.step % 100 == 0:
            self.loss_history = []
        
        # Ensure correct tensor shapes
        if self.predicted.shape[1] == 3:
            self.predicted = self.predicted.permute(0, 2, 1).contiguous()
        if self.gt.shape[1] == 3:
            self.gt = self.gt.permute(0, 2, 1).contiguous()
        
        if hasattr(self.net, 'loss_scale'):
            reg =  self.net.loss_scale 
        else: 
            reg = 0.0
        # Compute losses
        if hasattr(self.net, 'loss'):
            # Compute Chamfer loss and image loss
            gt_render = pc_views.get_img(self.gt)
            chamfer_loss = self.chamfer_loss_fn(self.predicted, self.gt) * 1000 
            img_loss = self.mse_loss(self.img, gt_render)

            loss = self.net.loss + reg * 0.01 + chamfer_loss + img_loss
        else:
            chamfer_loss = self.chamfer_loss_fn(self.predicted, self.gt) * 1000 + reg * 0.01
        
            # Compute image loss
            gt_render = pc_views.get_img(self.gt)
            img_loss = self.mse_loss(self.img, gt_render)
            loss = chamfer_loss + img_loss
        # Backpropagation
        loss.backward()
        grad_norm = self.get_gradient_norm()
        self.loss_history.append(loss.item())
        
        # Update parameters
        self.optimizer.step()
        
        # Learning rate scheduling
        if self.step % self.decay_step == 0 and self.step > 0:
            self.lr_scheduler.step()
            
        # Save visualization periodically
        if epoch is not None and epoch % 10 == 0:
            pass  # Uncomment to save images
            # save_img(self.img[0], f"img_{epoch}_{self.step}")
            # save_img(gt_render[0], f"gt_{epoch}_{self.step}")
            
        # Update step counter
        self.step += 1
        return { 
            'total_loss': loss.item(),
            'chamfer_loss': chamfer_loss.item(),
            'img_loss': img_loss.item(),
            'loss_scale': reg.item(),
            'lr': self.get_lr(),
            'grad_norm': grad_norm,
            'avg_loss': sum(self.loss_history) / len(self.loss_history) if self.loss_history else 0,
            'predicted': self.predicted,
            'gt': self.gt
        }


def train():
    """Main training function."""
    
    # Use configurations from args
    config = confg.args
    
    # Initialize model
    network_name = "PU_Gaussian"  # Change this to use different models #best model pu_gau2 with kernel loss
    model_map = {
        "Just_gaussian": Just_gaussian,
        "PU_Gaussian": PU_Gaussian
        }
    
    network = model_map[network_name](config)
    
    # Create a descriptive log directory name
    timestamp = get_timestamp()
    log_desc = f"{network_name}_{config.dataset}_{config.num_samples}x_{config.up_ratio}x_{config.distribution}_{timestamp}"
    log_dir = f"logs/{timestamp}_{log_desc}"
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create model
    model = PointCloudModel(network, 'train', config)
    model.net.to(config.device)
    
    # Print model summary
    num_params = sum(p.numel() for p in model.net.parameters()) / 1000
    print(f"Model: {network_name}")
    print(f"Number of parameters: {num_params:.2f}K")
        

    
    if config.dataset == "pugan":
        # Set the path for the dataset
        config.h5_file_path = config.pugan_path

    elif config.dataset == "pu1k":
        # Set the path for the dataset
        config.h5_file_path = config.pu1k_path

        
    print(f"Using dataset: {config.h5_file_path}")
    # Create dataset and dataloaders
    train_dataset = PUDataset(config)
    train_dataset, valid_dataset = train_test_split(train_dataset, test_size=0.1)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)
    
    print(f"Training on {len(train_dataset)} samples, validating on {len(valid_dataset)} samples")
    print(f"Logging to {log_dir}")
    # load checkpoint if provided
    # Training loop
    for epoch in range(config.max_epochs):
        # Training
        model.net.train()
        train_losses = []
        print(f"Starting epoch {epoch+1}/{config.max_epochs}, config.training_stage: {config.training_stage}, config.stage_1_max_epochs: {config.stage_1_max_epochs}")
        
        if epoch + 1 > config.stage_1_max_epochs and config.training_stage == 1:
            model.net.training_stage = 2
            model.training_stage = 2
            config.training_stage = 2
            print("Switching to training stage 2")

        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.max_epochs}")):
            input_pc, x20, radius, val, x6 = data
            input_pc = input_pc.to(config.device)
            x20 = x20.to(config.device)
            
            model.set_input(input_pc, 0, x6=x6, x20=x20)
            metrics = model.optimize(epoch=epoch)
            
            # Log metrics
            writer.add_scalar('Loss/Total', metrics['total_loss'], model.step)
            writer.add_scalar('Loss/Chamfer', metrics['chamfer_loss'], model.step)
            writer.add_scalar('Loss/Image', metrics['img_loss'], model.step)
            writer.add_scalar('Training/LR', metrics['lr'], model.step)
            writer.add_scalar('Training/GradNorm', metrics['grad_norm'], model.step)
            writer.add_scalar('Loss/Average', metrics['avg_loss'], model.step)
            writer.add_scalar('Loss/Scale', metrics['loss_scale'], model.step)
            
            train_losses.append(metrics['total_loss'])
        
        # # Validation
        model.net.eval()
        val_losses = []
        
        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                input_pc, _ , radius, gt_pc, x6= data # val is used for validation because it is always 4x upsampled unlike gt_pc.
                input_pc = input_pc.to(config.device)
                gt_pc = gt_pc.to(config.device)
                
                model.set_input(input_pc, 0, gt_pc, x6)
                val_loss = model.validate()
                val_losses.append(val_loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        # Log epoch metrics
        writer.add_scalar('Epoch/TrainLoss', avg_train_loss, epoch)
        writer.add_scalar('Epoch/ValidationLoss', avg_val_loss, epoch)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        # print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}")        
        # Save model periodically
        if epoch % config.save_interval == 0 or epoch == config.max_epochs - 1:
            model_path = os.path.join(config.model_dir, f"{network_name}_epoch_{epoch}_{config.dataset}_{config.decay_iter}_decay_steps_{config.distribution}_{timestamp}_refined.pth")
            torch.save(model.net.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    
    writer.close()
    print("Training completed!")

if __name__ == '__main__':
    train()